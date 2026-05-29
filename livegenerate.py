"""
livegenerate.py — continuous audio streaming, no GUI
Parameters set at launch, exit with Ctrl+C or press Q+Enter

Usage:
    python livegenerate.py --duration_hint 60
    python livegenerate.py --variation 2.0 --fade_ms 100 --top_k 12
"""

import os, sys, argparse, threading, queue, time
import numpy as np
import torch
import torch.nn as nn
import pickle
import sounddevice as sd
from scipy.ndimage import uniform_filter1d
from scipy.special import softmax
import warnings
warnings.filterwarnings("ignore")


# ─── PARAM UPDATE THREAD ──────────────────────────────────────────────────────
UPDATABLE = {"step_size", "smoothing_window", "n_control", "top_k",
             "variation", "temperature", "rms_percentile", "fade_ms"}

def param_input_thread(params, stop_event):
    """Reads stdin for param=value updates. Type param=value and press Enter."""
    while not stop_event.is_set():
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        line = line.strip()
        if not line:
            continue
        if "=" not in line:
            print(f"  format: param=value  (available: {', '.join(sorted(UPDATABLE))})")
            continue
        key, _, val = line.partition("=")
        key = key.strip(); val = val.strip()
        if key not in UPDATABLE:
            print(f"  unknown param '{key}'  (available: {', '.join(sorted(UPDATABLE))})")
            continue
        try:
            # int params
            if key in ("smoothing_window", "n_control", "top_k"):
                params[key] = int(val)
            else:
                params[key] = float(val)
            print(f"  → {key} = {params[key]}")
        except ValueError:
            print(f"  invalid value: {val}")

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


# ─── RETRIEVAL ────────────────────────────────────────────────────────────────
def power_compress(x, exp=0.7): return np.abs(x)**exp
def rms_normalize(x):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True)) + 1e-8
    out = x / rms
    return np.where(np.isfinite(out), out, 0.0)
def build_frame_db(m): return rms_normalize(power_compress(m))

def retrieve_chunk(query_mel, chunk_db, audio_chunks,
                   top_k=8, temperature=0.2, ema_penalty=None, blacklist=None):
    q    = rms_normalize(power_compress(query_mel.reshape(-1)))
    sims = chunk_db @ q
    if ema_penalty is not None:
        ema_n = rms_normalize(ema_penalty)
        sims  = sims - 0.5 * np.clip(chunk_db @ ema_n, 0, 1)
    if blacklist:
        for b in blacklist:
            if b < len(sims): sims[b] = -np.inf
    top_idx  = np.argpartition(sims, -top_k)[-top_k:]
    top_idx  = top_idx[np.argsort(sims[top_idx])]
    weights  = softmax(sims[top_idx] / (temperature + 1e-8))
    blended  = np.zeros(audio_chunks.shape[1], dtype=np.float64)
    for w, idx in zip(weights, top_idx):
        blended += w * audio_chunks[idx].astype(np.float64)
    return blended.astype(np.float32), int(top_idx[-1])


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

def walk_segment(latents, n, n_ctrl=16, step=0.3, smooth=3):
    dim  = latents.shape[1]
    lstd = latents.std(axis=0).mean()
    N    = len(latents)
    n_ctrl = min(n_ctrl, max(2, n//2))
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
    noise  = multiscale_fbm(n, dim) * lstd * step
    traj   = spline + noise
    vel    = uniform_filter1d(np.diff(traj, axis=0), size=max(2,smooth), axis=0)
    t2     = [traj[0]]
    for v in vel: t2.append(t2[-1]+v)
    return np.array(t2)


# ─── GENERATOR THREAD ─────────────────────────────────────────────────────────
class GeneratorThread(threading.Thread):
    def __init__(self, model, chunk_db, audio_chunks, mf_kept,
                 latents, meta, params, audio_queue, stop_event):
        super().__init__(daemon=True)
        self.model       = model
        self.chunk_db    = chunk_db
        self.audio_chunks = audio_chunks
        self.mf_kept     = mf_kept
        self.latents     = latents
        self.meta        = meta
        self.params      = params
        self.audio_queue = audio_queue
        self.stop_event  = stop_event

        # walk state
        self.chunk_db_dim = chunk_db.shape[1]
        self.ema_penalty  = np.zeros(self.chunk_db_dim, dtype=np.float32)
        self.blacklist    = []

    def run(self):
        p         = self.params
        n_ctrl    = p['n_control']
        step      = p['step_size'] * p['variation']
        smooth    = p['smoothing_window']
        top_k     = p['top_k']
        temp      = p['temperature']
        fade_ms   = p['fade_ms']
        sr        = self.meta['sr']
        seq_frames = self.meta['seq_frames']
        chunk_samples = seq_frames * self.meta['hop']

        overlap_ratio = max(0.02, min(0.95, 1.0 - fade_ms/1000.0))
        step_samples  = int(chunk_samples * (1.0 - overlap_ratio))
        blacklist_sz  = max(6, top_k * 2)

        # bell window for morph
        bell = (1 - np.cos(np.linspace(0, 2*np.pi, chunk_samples))) / 2

        # rolling morph buffer
        buf_size  = chunk_samples * 4
        buf       = np.zeros(buf_size, dtype=np.float64)
        buf_w     = np.zeros(buf_size, dtype=np.float64)
        write_pos = 0

        # generate walk in segments of 32 chunks
        seg_size = 32
        while not self.stop_event.is_set():
            traj = walk_segment(self.latents, seg_size,
                                n_ctrl=n_ctrl, step=step, smooth=smooth)
            for z in traj:
                if self.stop_event.is_set(): break
                # re-read params every chunk for live updates
                p             = self.params
                top_k         = int(p['top_k'])
                temp          = p['temperature']
                fade_ms       = p['fade_ms']
                step          = p['step_size'] * p['variation']
                overlap_ratio = max(0.02, min(0.95, 1.0 - fade_ms/1000.0))
                step_samples  = max(1, int(chunk_samples * (1.0 - overlap_ratio)))
                blacklist_sz  = max(6, top_k * 2)
                z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    mel = self.model.decoder(z_t).squeeze(0).cpu().numpy()

                chunk, best_idx = retrieve_chunk(
                    mel, self.chunk_db, self.audio_chunks,
                    top_k=top_k, temperature=temp,
                    ema_penalty=self.ema_penalty,
                    blacklist=set(self.blacklist)
                )
                self.ema_penalty = 0.80*self.ema_penalty + 0.20*self.mf_kept[best_idx].reshape(-1)
                self.blacklist.append(best_idx)
                if len(self.blacklist) > blacklist_sz: self.blacklist.pop(0)

                # write chunk into rolling buffer with bell window
                end = write_pos + chunk_samples
                if end > buf_size:
                    write_pos = 0; end = chunk_samples
                buf[write_pos:end]   += chunk.astype(np.float64) * bell
                buf_w[write_pos:end] += bell

                # emit step_samples of normalized audio
                out = np.zeros(step_samples, dtype=np.float32)
                for i in range(step_samples):
                    pos = write_pos + i
                    if pos < buf_size and buf_w[pos] > 1e-8:
                        out[i] = buf[pos] / buf_w[pos]

                # normalize
                peak = np.abs(out).max()
                if peak > 1e-8: out = out / peak * 0.9

                # block if queue full
                while self.audio_queue.qsize() >= 8 and not self.stop_event.is_set():
                    time.sleep(0.01)
                self.audio_queue.put(out)

                write_pos += step_samples


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",        default="./model")
    p.add_argument("--step_size",        type=float, default=0.3)
    p.add_argument("--smoothing_window", type=int,   default=3)
    p.add_argument("--n_control",        type=int,   default=16)
    p.add_argument("--top_k",            type=int,   default=8)
    p.add_argument("--variation",        type=float, default=1.0)
    p.add_argument("--temperature",      type=float, default=0.2)
    p.add_argument("--rms_percentile",   type=float, default=20.0)
    p.add_argument("--fade_ms",          type=float, default=200.0)
    args = p.parse_args()

    params = vars(args)
    model_dir = args.model_dir

    print("Loading model...")
    with open(os.path.join(model_dir,"meta.pkl"),"rb") as f:
        meta = pickle.load(f)
    n_mels,seq_frames,ldim = meta["n_mels"],meta["seq_frames"],meta["latent_dim"]
    sr = meta["sr"]

    model = AE(n_mels, seq_frames, ldim).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(model_dir,"vae.pt"), map_location=DEVICE))
    model.eval()

    print("Loading corpus...")
    ac_all = np.load(os.path.join(model_dir,"audio_chunks.npy"))
    mf_all = np.load(os.path.join(model_dir,"mel_frames.npy"))
    lr     = np.load(os.path.join(model_dir,"latents_raw.npy"))

    chunk_rms  = np.sqrt(np.mean(ac_all**2, axis=1))
    mask       = chunk_rms >= np.percentile(chunk_rms, args.rms_percentile)
    ac_kept    = ac_all[mask]
    mf_kept    = mf_all.reshape(len(ac_all), seq_frames, n_mels)[mask]
    ac_rms     = np.sqrt(np.mean(ac_kept**2, axis=1, keepdims=True)) + 1e-8
    ac_kept    = ac_kept / ac_rms * 0.15
    chunk_db   = build_frame_db(mf_kept.reshape(len(mf_kept), -1))
    print(f"  {len(ac_kept)} chunks  latent std={lr.std(axis=0).mean():.3f}")

    audio_queue = queue.Queue(maxsize=16)
    stop_event  = threading.Event()

    gen_thread = GeneratorThread(
        model, chunk_db, ac_kept, mf_kept,
        lr, meta, params, audio_queue, stop_event
    )

    # pre-buffer before starting audio
    print("Buffering...")
    gen_thread.start()
    while audio_queue.qsize() < 4:
        time.sleep(0.05)

    buf = np.zeros(0, dtype=np.float32)

    def callback(outdata, frames, time_info, status):
        nonlocal buf
        while len(buf) < frames:
            try:
                chunk = audio_queue.get_nowait()
                buf   = np.concatenate([buf, chunk])
            except queue.Empty:
                buf = np.concatenate([buf, np.zeros(frames, dtype=np.float32)])
                break
        outdata[:,0] = buf[:frames]
        buf = buf[frames:]

    # start param input thread
    input_thread = threading.Thread(
        target=param_input_thread, args=(params, stop_event), daemon=True
    )
    input_thread.start()

    print(f"\nStreaming — press Ctrl+C to stop")
    print(f"Update params live: type param=value and press Enter")
    print(f"Available: {', '.join(sorted(UPDATABLE))}")
    print(f"\n  variation={args.variation}  temperature={args.temperature}  "
          f"fade_ms={args.fade_ms}  top_k={args.top_k}\n")

    try:
        with sd.OutputStream(samplerate=sr, channels=1, dtype='float32',
                             blocksize=1024, callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
