"""
generate.py — frame-wise mel retrieval + direct audio concatenation
No ISTFT, no OLA, no beating.
Usage:
    python generate.py --duration 60 --output out.wav
    python generate.py --duration 60 --variation 2.0 --top_k 24 --output harsh.wav
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import pickle
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.special import softmax
import warnings
warnings.filterwarnings("ignore")

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ─── AE ───────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, n_mels, seq_frames, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 4, stride=2, padding=1),    nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 4, stride=2, padding=1),    nn.LeakyReLU(0.2),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, n_mels, seq_frames)).view(1,-1).shape[1]
        self.flat = flat
        self.fc = nn.Sequential(
            nn.Linear(flat,256), nn.LeakyReLU(0.2), nn.Linear(256,latent_dim)
        )
    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0),-1))

class Decoder(nn.Module):
    def __init__(self, flat, latent_dim, n_mels, seq_frames):
        super().__init__()
        self.seq_frames  = seq_frames
        self._flat_shape = (512, flat//512)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim,256), nn.LeakyReLU(0.2), nn.Linear(256,flat)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(512,512,4,stride=2,padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(512,256,4,stride=2,padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(256,n_mels,4,stride=2,padding=1),
        )
    def forward(self, z):
        h = self.fc(z).view(z.size(0), *self._flat_shape)
        return self.deconv(h)[:, :, :self.seq_frames]

class AE(nn.Module):
    def __init__(self, n_mels, seq_frames, latent_dim):
        super().__init__()
        self.encoder = Encoder(n_mels, seq_frames, latent_dim)
        self.decoder = Decoder(self.encoder.flat, latent_dim, n_mels, seq_frames)
    def forward(self, x):
        z = self.encoder(x); return self.decoder(z), z


# ─── FRAME DB ─────────────────────────────────────────────────────────────────
def power_compress(x, exp=0.7):
    return np.abs(x) ** exp

def rms_normalize(x):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True)) + 1e-8
    out = x / rms
    return np.where(np.isfinite(out), out, 0.0)

def build_frame_db(mel_frames):
    return rms_normalize(power_compress(mel_frames))


# ─── CHUNK-LEVEL RETRIEVAL ────────────────────────────────────────────────────
def retrieve_chunk(query_mel_chunk, chunk_db, audio_chunks,
                   top_k=8, temperature=0.2, ema_penalty=None, blacklist=None):
    """
    query_mel_chunk: [n_mels, seq_frames] — mel chunk from decoder
    chunk_db:        [N, n_mels*seq_frames] normalized — chunk-level db
    audio_chunks:    [N, chunk_samples] — raw audio
    blacklist:       set of indices to exclude (hard no-repeat)
    Returns blended audio chunk [chunk_samples] and best index.
    """
    q    = rms_normalize(power_compress(query_mel_chunk.reshape(-1)))
    sims = chunk_db @ q

    if ema_penalty is not None:
        ema_n = rms_normalize(ema_penalty)
        sims  = sims - 0.5 * np.clip(chunk_db @ ema_n, 0, 1)

    if blacklist:
        for b in blacklist:
            if b < len(sims):
                sims[b] = -np.inf

    top_idx  = np.argpartition(sims, -top_k)[-top_k:]
    top_idx  = top_idx[np.argsort(sims[top_idx])]
    weights  = softmax(sims[top_idx] / (temperature + 1e-8))

    chunk_samples = audio_chunks.shape[1]
    blended = np.zeros(chunk_samples, dtype=np.float64)
    for w, idx in zip(weights, top_idx):
        blended += w * audio_chunks[idx].astype(np.float64)

    return blended.astype(np.float32), int(top_idx[-1])


# ─── CROSSFADE ────────────────────────────────────────────────────────────────
def morph_chunks(chunks, overlap_ratio=0.75):
    """
    Continuous morphing — bell-shaped window on each chunk, heavy overlap.
    No cuts, no boundaries. Every chunk fuses into the next.
    overlap_ratio: fraction of chunk that overlaps (0.5-0.95)
    """
    if len(chunks) == 1:
        return chunks[0]

    n       = len(chunks[0])
    overlap = int(n * overlap_ratio)
    step    = max(1, n - overlap)
    total   = step * (len(chunks) - 1) + n
    out     = np.zeros(total, dtype=np.float64)
    weight  = np.zeros(total, dtype=np.float64)

    # raised cosine bell — smooth entry and exit
    t_bell = (1 - np.cos(np.linspace(0, 2*np.pi, n))) / 2

    for i, chunk in enumerate(chunks):
        s = i * step
        out[s:s+n]    += chunk.astype(np.float64) * t_bell
        weight[s:s+n] += t_bell

    out = np.where(weight > 1e-8, out / weight, 0.0)
    return out.astype(np.float32)


def crossfade_chunks(chunks, fade_samples=2048):
    return morph_chunks(chunks, overlap_ratio=0.75)


# ─── WALK ─────────────────────────────────────────────────────────────────────
def fbm1d(n, H, sigma):
    k   = np.arange(n)
    cov = 0.5*(np.abs(k-1)**(2*H)+np.abs(k+1)**(2*H)-2*k**(2*H))
    cov[0] = 1.0
    p = np.sqrt(np.abs(np.fft.rfft(cov, n=2*n)))
    x = np.fft.irfft(p * np.fft.rfft(np.random.randn(2*n)))[:n]
    x -= x.mean()
    return (x/(x.std()+1e-8)*sigma).astype(np.float32)

def multiscale_fbm(n, dim):
    b1 = fbm1d(n, H=0.95, sigma=0.05)
    b2 = fbm1d(n, H=0.70, sigma=0.02)
    b3 = fbm1d(n, H=0.40, sigma=0.005)
    proj = np.random.randn(3, dim).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
    return b1[:,None]*proj[0] + b2[:,None]*proj[1] + b3[:,None]*proj[2]

def cubic_bezier(p0, p1, p2, p3, n):
    t = np.linspace(0,1,n)[:,None]
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3

def walk(latents, n, n_ctrl=16, step=0.3, smooth=3):
    dim  = latents.shape[1]
    lstd = latents.std(axis=0).mean()
    N    = len(latents)
    idx  = [np.random.randint(N)]
    for _ in range(n_ctrl+1):
        md = np.full(N, np.inf)
        for c in idx:
            d  = np.linalg.norm(latents - latents[c], axis=1)
            md = np.minimum(md, d)
        md[idx] = 0
        thr = np.percentile(md[md>0], 95)
        idx.append(int(np.random.choice(np.where(md>=thr)[0])))
    ctrl = [latents[i].copy() for i in idx]
    ns   = n_ctrl-1; spp = max(1,n//ns); rem = n-spp*ns
    segs = []
    for i in range(ns):
        p0,p3 = ctrl[i],ctrl[i+1]
        p1 = p0+0.4*(ctrl[i+1]-ctrl[max(0,i-1)])
        p2 = p3-0.4*(ctrl[min(n_ctrl,i+2)]-ctrl[i])
        nn = spp+(rem if i==ns-1 else 0)
        segs.append(cubic_bezier(p0,p1,p2,p3,nn))
    spline = np.concatenate(segs)[:n]
    noise  = multiscale_fbm(n,dim)*lstd*step
    traj   = spline+noise
    vel    = uniform_filter1d(np.diff(traj,axis=0), size=max(2,smooth), axis=0)
    t2     = [traj[0]]
    for v in vel: t2.append(t2[-1]+v)
    return np.array(t2)


# ─── GENERATE ─────────────────────────────────────────────────────────────────
def generate(model_dir, duration, output, step_size, smooth, n_ctrl, top_k,
             variation, temperature, rms_percentile, fade_ms):

    with open(os.path.join(model_dir,"meta.pkl"),"rb") as f:
        meta = pickle.load(f)
    n_mels,seq_frames,ldim = meta["n_mels"],meta["seq_frames"],meta["latent_dim"]
    sr,n_fft,hop = meta["sr"],meta["n_fft"],meta["hop"]

    step          = step_size * variation
    chunk_samples = seq_frames * hop
    chunk_sec     = chunk_samples / sr
    overlap_ratio = max(0.02, min(0.95, 1.0 - fade_ms/1000.0))
    step_sec      = chunk_sec * (1.0 - overlap_ratio)
    n_chunks      = max(2, int(np.ceil(duration / step_sec)) + 1)

    print(f"{duration}s → {n_chunks} chunks × {chunk_sec:.3f}s  step={step:.3f} top_k={top_k}")

    # load model
    model = AE(n_mels, seq_frames, ldim).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(model_dir,"vae.pt"), map_location=DEVICE))
    model.eval()

    # check required files
    for fname in ["audio_chunks.npy","mel_frames.npy","latents_raw.npy"]:
        if not os.path.exists(os.path.join(model_dir,fname)):
            print(f"ERROR: {fname} not found.")
            if fname == "audio_chunks.npy":
                print("Run: python extract_audio_chunks.py --data_dir ./audio --model_dir ./model")
            elif fname == "mel_frames.npy":
                print("Run: python extract_mel_frames.py --model_dir ./model")
            sys.exit(1)

    print("Loading data...")
    ac_all  = np.load(os.path.join(model_dir,"audio_chunks.npy"))  # [N, chunk_samples]
    mf_all  = np.load(os.path.join(model_dir,"mel_frames.npy"))    # [N*seq_frames, n_mels]
    lr      = np.load(os.path.join(model_dir,"latents_raw.npy"))

    N_chunks_total = len(ac_all)

    # validate mel_frames alignment
    if len(mf_all) != N_chunks_total * seq_frames:
        raise ValueError(
            f"mel_frames.npy {len(mf_all)} != {N_chunks_total}×{seq_frames}. "
            "Re-run extract_mel_frames.py"
        )

    # silence filter — same mask on both audio_chunks and mel_frames
    chunk_rms  = np.sqrt(np.mean(ac_all**2, axis=1))
    rms_thresh = np.percentile(chunk_rms, rms_percentile)
    mask       = chunk_rms >= rms_thresh

    ac_kept = ac_all[mask]
    mf_kept = mf_all.reshape(N_chunks_total, seq_frames, n_mels)[mask]  # [N_kept, seq_frames, n_mels]

    # RMS normalize audio chunks
    ac_rms  = np.sqrt(np.mean(ac_kept**2, axis=1, keepdims=True)) + 1e-8
    ac_kept = ac_kept / ac_rms * 0.15

    # build chunk-level mel db: full temporal sequence flattened
    # preserves texture, micrody namics and transients — not just average color
    chunk_db = build_frame_db(
        mf_kept.reshape(len(mf_kept), -1)             # [N_kept, seq_frames*n_mels]
    )

    print(f"  {len(ac_kept)} chunks  latent std={lr.std(axis=0).mean():.3f}")

    # walk
    print("Walking...")
    n_ctrl = min(n_ctrl, max(2, n_chunks // 2))  # clamp to available chunks
    traj = walk(lr, n_chunks, n_ctrl=n_ctrl, step=step, smooth=smooth)

    # decode + retrieve
    print("Decoding + retrieval...")
    chunk_db_dim = len(mf_kept[0].reshape(-1))
    ema_penalty  = np.zeros(chunk_db_dim, dtype=np.float32)
    blacklist    = []
    blacklist_sz = max(6, top_k * 2)
    audio_chunks_out = []

    for i, z in enumerate(traj):
        z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mel = model.decoder(z_t).squeeze(0).cpu().numpy()

        blended, best_idx = retrieve_chunk(
            mel, chunk_db, ac_kept,
            top_k=top_k, temperature=temperature,
            ema_penalty=ema_penalty if i > 0 else None,
            blacklist=set(blacklist)
        )
        ema_penalty = 0.80*ema_penalty + 0.20*mf_kept[best_idx].reshape(-1)
        blacklist.append(best_idx)
        if len(blacklist) > blacklist_sz:
            blacklist.pop(0)
        audio_chunks_out.append(blended)
        if (i+1)%20==0 or i==0: print(f"  {i+1}/{n_chunks}")

    # stitch with crossfade
    print("Stitching...")
    print(f"  overlap_ratio={overlap_ratio:.2f}")
    audio = morph_chunks(audio_chunks_out, overlap_ratio=overlap_ratio)
    audio = audio[:int(duration*sr)]

    peak = np.abs(audio).max()
    if peak > 1e-8: audio = audio/peak*0.95
    sf.write(output, audio, sr)
    print(f"→ {output}  ({duration}s @ {sr}Hz)")


# ─── ENTRY ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",        default="./model")
    p.add_argument("--duration",         type=float, required=True)
    p.add_argument("--output",           default="output.wav")
    p.add_argument("--step_size",        type=float, default=0.3)
    p.add_argument("--smoothing_window", type=int,   default=3)
    p.add_argument("--n_control",        type=int,   default=16)
    p.add_argument("--top_k",            type=int,   default=8,
                   help="Retrieval candidates (default 8)")
    p.add_argument("--variation",        type=float, default=1.0,
                   help="<1=drone  >1=harsh")
    p.add_argument("--temperature",      type=float, default=0.2,
                   help="Retrieval diffusion (default 0.2)")
    p.add_argument("--rms_percentile",   type=float, default=0.0,
                   help="Filter silent chunks (default 20)")
    p.add_argument("--fade_ms",          type=float, default=50.0,
                   help="Crossfade duration ms between chunks (default 50)")
    args = p.parse_args()
    if not os.path.isdir(args.model_dir):
        print(f"not found: {args.model_dir}"); sys.exit(1)
    generate(args.model_dir, args.duration, args.output,
             args.step_size, args.smoothing_window,
             args.n_control, args.top_k, args.variation,
             args.temperature, args.rms_percentile, args.fade_ms)
