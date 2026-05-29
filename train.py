"""
train.py — AE overlap + all losses + full extraction pipeline
Saves: vae.pt, gmm.pkl, pca.pkl, scaler.pkl, meta.pkl,
       checkpoint.pt, latents_raw.npy, band_stats.npy,
       mel_chunks.npy, raw_chunks.npy

Usage:
    python train.py --data_dir ./audio --output_dir ./model
    python train.py --data_dir ./audio --output_dir ./model --epochs 50 --resume
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SR            = 22050
N_FFT         = 2048
HOP           = 512
WIN           = N_FFT
N_MELS        = 128
SEQ_FRAMES    = 64   # overridden by --seq_frames
LATENT_DIM    = 64
PCA_DIM       = 32
GMM_COMP      = 12
BATCH_SIZE    = 16
EPOCHS        = 120
LR            = 3e-4
OVERLAP       = SEQ_FRAMES // 2
CONSISTENCY_W = 1.0
SPECTRAL_W    = 1.0
STD_W         = 2.0
VAR_W         = 0.5
MSSTFT_W      = 1.0
PRE_EMPHASIS  = 0.92

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ─── PREPROCESSING ────────────────────────────────────────────────────────────
def apply_pre_emphasis(y, coef=PRE_EMPHASIS):
    return np.append(y[0], y[1:] - coef * y[:-1]).astype(np.float32)


def normalize_mel_per_band(mel_db):
    mean = mel_db.mean(axis=1, keepdims=True)
    std  = mel_db.std(axis=1,  keepdims=True) + 1e-8
    return (mel_db - mean) / std


def build_isophonic_weights(n_mels, sr):
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr//2)
    weights   = np.ones(n_mels, dtype=np.float32)
    for i, f in enumerate(mel_freqs):
        if   f <  200: weights[i] = 0.4
        elif f <  500: weights[i] = 0.7
        elif f < 1000: weights[i] = 1.0
        elif f < 4000: weights[i] = 1.8
        elif f < 8000: weights[i] = 1.3
        else:          weights[i] = 0.6
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ─── DATASET ──────────────────────────────────────────────────────────────────
class OverlapPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]


def load_audio(data_dir):
    exts = (".mp3", ".wav", ".flac", ".ogg")
    files = [
        os.path.join(data_dir, f)
        for f in sorted(os.listdir(data_dir))
        if f.lower().endswith(exts) and not f.startswith("._")
    ]
    if not files:
        raise ValueError(f"No audio files in {data_dir}")
    print(f"Found {len(files)} files")

    pairs           = []
    singles         = []
    all_mel_db      = []
    mel_chunks      = []
    raw_chunks      = []
    mel_frames_list = []
    audio_chunks_list = []
    chunk_samples     = SEQ_FRAMES * HOP
    window     = np.hanning(WIN).astype(np.float32)

    for path in files:
        print(f"  {os.path.basename(path)}")
        try:
            y, _ = librosa.load(path, sr=SR, mono=True)
        except Exception as e:
            print(f"    skip: {e}")
            continue

        y_pre  = apply_pre_emphasis(y)

        # mel for AE training
        mel    = librosa.feature.melspectrogram(
            y=y_pre, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
        )
        mel_db = librosa.power_to_db(mel + 1e-8, ref=np.max)
        all_mel_db.append(mel_db)
        mel_db_norm = normalize_mel_per_band(mel_db)

        # raw FFT magnitude for vocoder
        n_frames = (len(y) - WIN) // HOP
        if n_frames < SEQ_FRAMES:
            continue
        fft_idx  = np.arange(n_frames)[:, None] * HOP + np.arange(WIN)
        fmag     = np.abs(np.fft.rfft(y[fft_idx] * window, axis=1)).astype(np.float32)

        T = min(mel_db_norm.shape[1], len(fmag))

        for start in range(0, T - SEQ_FRAMES - OVERLAP, OVERLAP):
            c0 = mel_db_norm[:, start:start + SEQ_FRAMES]
            c1 = mel_db_norm[:, start + OVERLAP:start + OVERLAP + SEQ_FRAMES]
            pairs.append((
                torch.tensor(c0, dtype=torch.float32),
                torch.tensor(c1, dtype=torch.float32),
            ))
            singles.append(torch.tensor(c0, dtype=torch.float32))

            # mel chunk for cosine similarity in generate
            mel_db_chunk = mel_db[:, start:start + SEQ_FRAMES]
            mel_chunks.append(mel_db_chunk.flatten())

            # raw FFT chunk for vocoder
            raw = fmag[start:start + SEQ_FRAMES]
            if len(raw) == SEQ_FRAMES:
                raw_chunks.append(raw)
                # per-frame mel aligned with raw frames
                for fi in range(SEQ_FRAMES):
                    mel_frames_list.append(mel_db[:, start + fi])
                # audio samples aligned with raw chunk
                s_start = start * HOP
                s_end   = s_start + chunk_samples
                if s_end <= len(y):
                    audio_chunks_list.append(y[s_start:s_end])
                else:
                    pad = s_end - len(y)
                    audio_chunks_list.append(
                        np.concatenate([y[s_start:], np.zeros(pad, dtype=np.float32)])
                    )

    all_mel_cat = np.concatenate(all_mel_db, axis=1)
    band_mean   = all_mel_cat.mean(axis=1)
    band_std    = all_mel_cat.std(axis=1) + 1e-8

    mel_chunks = np.array(mel_chunks, dtype=np.float32)
    raw_chunks = np.array(raw_chunks, dtype=np.float32)

    # align lengths
    n = min(len(pairs), len(mel_chunks), len(raw_chunks))
    pairs      = pairs[:n]
    singles    = singles[:n]
    mel_chunks = mel_chunks[:n]
    raw_chunks = raw_chunks[:n]

    mel_frames   = np.array(mel_frames_list,   dtype=np.float32)
    audio_chunks = np.array(audio_chunks_list, dtype=np.float32)
    n = min(n, len(audio_chunks))
    audio_chunks = audio_chunks[:n]
    mel_frames   = mel_frames[:n * SEQ_FRAMES]
    print(f"Total pairs: {n}  raw_chunks: {raw_chunks.shape}  audio_chunks: {audio_chunks.shape}  mel_frames: {mel_frames.shape}")
    return pairs, singles, band_mean, band_std, mel_chunks, raw_chunks, mel_frames, audio_chunks


# ─── AE ───────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, n_mels, seq_frames, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),    nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),    nn.LeakyReLU(0.2),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, n_mels, seq_frames)).view(1,-1).shape[1]
        self.flat = flat
        self.fc = nn.Sequential(
            nn.Linear(flat, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim),
        )
    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


class Decoder(nn.Module):
    def __init__(self, flat, latent_dim, n_mels, seq_frames):
        super().__init__()
        self.seq_frames  = seq_frames
        self._flat_shape = (512, flat // 512)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, flat),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(256, n_mels, kernel_size=4, stride=2, padding=1),
        )
    def forward(self, z):
        h = self.fc(z).view(z.size(0), *self._flat_shape)
        return self.deconv(h)[:, :, :self.seq_frames]


class AE(nn.Module):
    def __init__(self, n_mels=N_MELS, seq_frames=SEQ_FRAMES, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(n_mels, seq_frames, latent_dim)
        self.decoder = Decoder(self.encoder.flat, latent_dim, n_mels, seq_frames)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ─── LOSSES ───────────────────────────────────────────────────────────────────
def weighted_spectral_loss(recon, target, band_weights):
    diff = (recon - target) ** 2
    diff = diff.mean(dim=(0, 2))
    return (diff * band_weights.to(diff.device)).mean()


def mel_std_loss(recon, target):
    return nn.functional.mse_loss(recon.std(dim=2), target.std(dim=2))


def multiscale_stft_loss(recon, target):
    loss = 0.0
    for scale in [1, 2, 4]:
        if recon.shape[2] >= scale:
            loss = loss + nn.functional.mse_loss(
                recon[:, :, ::scale], target[:, :, ::scale]
            )
    return loss / 3.0


def total_loss(r0, x0, r1, x1, z0, z1, band_weights):
    recon   = nn.functional.mse_loss(r0, x0) + nn.functional.mse_loss(r1, x1)
    spec    = weighted_spectral_loss(r0, x0, band_weights) + \
              weighted_spectral_loss(r1, x1, band_weights)
    std     = mel_std_loss(r0, x0) + mel_std_loss(r1, x1)
    ms      = multiscale_stft_loss(r0, x0) + multiscale_stft_loss(r1, x1)
    cons    = nn.functional.mse_loss(z0, z1)
    z_all   = torch.cat([z0, z1], dim=0)
    var_pen = torch.mean((1.0 - z_all.std(dim=0)) ** 2)
    return (recon + SPECTRAL_W*spec + STD_W*std + MSSTFT_W*ms +
            CONSISTENCY_W*cons + VAR_W*var_pen,
            recon, spec, std)


# ─── TRAINING ─────────────────────────────────────────────────────────────────
def train(data_dir, output_dir, epochs, latent_dim, resume):
    os.makedirs(output_dir, exist_ok=True)

    pairs, singles, band_mean, band_std, mel_chunks, raw_chunks, mel_frames, audio_chunks = load_audio(data_dir)
    loader = DataLoader(
        OverlapPairDataset(pairs), batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True
    )

    model = AE(N_MELS, SEQ_FRAMES, latent_dim).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start_epoch = 1
    ckpt_path   = os.path.join(output_dir, "checkpoint.pt")
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    band_weights = build_isophonic_weights(N_MELS, SR)

    params = sum(p.numel() for p in model.parameters())
    print(f"\nDevice: {DEVICE}  |  Params: {params:,}")
    print(f"Latent: {latent_dim}  PCA→{PCA_DIM}  GMM: {GMM_COMP}")
    print(f"Losses: recon + spec×{SPECTRAL_W} + std×{STD_W} + ms×{MSSTFT_W} + var×{VAR_W} + cons×{CONSISTENCY_W}")
    print(f"Training epochs {start_epoch}→{epochs}...\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        tot = tot_r = tot_s = tot_std = 0
        for x0, x1 in loader:
            x0, x1 = x0.to(DEVICE), x1.to(DEVICE)
            r0, z0 = model(x0)
            r1, z1 = model(x1)
            loss, rl, sl, stdl = total_loss(r0, x0, r1, x1, z0, z1, band_weights)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); tot_r += rl.item()
            tot_s += sl.item(); tot_std += stdl.item()
        sched.step()
        if epoch % 10 == 0 or epoch == start_epoch:
            n = len(loader)
            print(f"  epoch {epoch:4d}/{epochs}  "
                  f"loss={tot/n:.4f}  recon={tot_r/n:.4f}  "
                  f"spec={tot_s/n:.4f}  std={tot_std/n:.4f}")
        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "opt": opt.state_dict()}, ckpt_path)

    # ─── LATENT EXTRACTION ────────────────────────────────────────────────────
    print("\nExtracting latent vectors...")
    model.eval()

    class SingleDS(Dataset):
        def __init__(self, s): self.s = s
        def __len__(self): return len(self.s)
        def __getitem__(self, i): return self.s[i]

    all_z = []
    with torch.no_grad():
        for batch in DataLoader(SingleDS(singles), batch_size=128):
            all_z.append(model.encoder(batch.to(DEVICE)).cpu().numpy())
    Z = np.concatenate(all_z, axis=0)
    print(f"  Latent matrix: {Z.shape}  std: {Z.std(axis=0).mean():.3f}")

    # ─── PCA + GMM ────────────────────────────────────────────────────────────
    print(f"Fitting PCA {latent_dim} → {PCA_DIM}...")
    scaler   = StandardScaler()
    Z_scaled = scaler.fit_transform(Z)
    pca      = PCA(n_components=PCA_DIM, random_state=42)
    Z_pca    = pca.fit_transform(Z_scaled)
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    print(f"Fitting GMM ({GMM_COMP} components)...")
    gmm = GaussianMixture(n_components=GMM_COMP, covariance_type="full",
                          max_iter=300, random_state=42)
    gmm.fit(Z_pca)
    print(f"  BIC: {gmm.bic(Z_pca):.2f}")

    # ─── SAVE ALL ─────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(output_dir, "vae.pt"))
    for name, obj in [("gmm.pkl", gmm), ("pca.pkl", pca), ("scaler.pkl", scaler)]:
        with open(os.path.join(output_dir, name), "wb") as f:
            pickle.dump(obj, f)

    np.save(os.path.join(output_dir, "latents_raw.npy"), Z.astype(np.float32))
    np.save(os.path.join(output_dir, "band_stats.npy"),
            np.stack([band_mean, band_std], axis=0))
    np.save(os.path.join(output_dir, "mel_chunks.npy"),  mel_chunks)
    np.save(os.path.join(output_dir, "raw_chunks.npy"),  raw_chunks)
    np.save(os.path.join(output_dir, "mel_frames.npy"),  mel_frames)
    np.save(os.path.join(output_dir, "audio_chunks.npy"), audio_chunks)

    meta = dict(
        n_mels=N_MELS, seq_frames=SEQ_FRAMES, latent_dim=latent_dim,
        pca_dim=PCA_DIM, gmm_comp=GMM_COMP, sr=SR, n_fft=N_FFT, hop=HOP,
        pre_emphasis=PRE_EMPHASIS, model_type="ae_overlap_full",
    )
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"\nSaved → {output_dir}/")
    print("  vae.pt | gmm.pkl | pca.pkl | scaler.pkl | meta.pkl | checkpoint.pt")
    print("  latents_raw.npy | band_stats.npy | mel_chunks.npy | raw_chunks.npy | mel_frames.npy | audio_chunks.npy")


# ─── ENTRY ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--output_dir", default="./model")
    p.add_argument("--epochs",     type=int,  default=EPOCHS)
    p.add_argument("--latent_dim", type=int,  default=LATENT_DIM)
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--seq_frames", type=int, default=SEQ_FRAMES,
                   help="Frames per chunk (default 64, try 128 for noise)")
    p.add_argument("--hop",        type=int, default=HOP,
                   help="STFT hop size (default 512, try 256 for more resolution)")
    args = p.parse_args()
    # override globals before load_audio reads them
    import sys
    mod = sys.modules[__name__]
    mod.SEQ_FRAMES = args.seq_frames
    mod.HOP        = args.hop
    mod.OVERLAP    = args.seq_frames // 2
    mod.WIN        = mod.N_FFT
    train(args.data_dir, args.output_dir, args.epochs, args.latent_dim, args.resume)
