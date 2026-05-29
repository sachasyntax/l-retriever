"""
extract_mel_frames.py — deriva mel_frames.npy da raw_chunks.npy
Garantisce allineamento perfetto senza rileggere il dataset.
Usage: python extract_mel_frames.py --model_dir ./model
"""

import os
import argparse
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

def extract(model_dir):
    with open(os.path.join(model_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    SR     = meta["sr"]
    N_FFT  = meta["n_fft"]
    N_MELS = meta["n_mels"]

    rc = np.load(os.path.join(model_dir, "raw_chunks.npy"))
    # rc: [N_chunks, seq_frames, N_FFT//2+1]
    N_chunks, seq_frames, n_bins = rc.shape
    print(f"raw_chunks: {rc.shape}")

    # build mel filterbank — same as librosa
    import librosa
    mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
    # mel_basis: [N_MELS, N_FFT//2+1]

    # project each frame: raw_frame @ mel_basis.T → [N_MELS]
    # reshape rc to [N_total_frames, n_bins]
    rc_flat = rc.reshape(-1, n_bins)  # [N_chunks*seq_frames, n_bins]
    print(f"Projecting {len(rc_flat)} frames to mel space...")

    # mel = mel_basis @ frame.T → do in batch
    mel_frames = (mel_basis @ rc_flat.T).T  # [N_frames, N_MELS]
    mel_frames = np.log1p(mel_frames).astype(np.float32)

    out_path = os.path.join(model_dir, "mel_frames.npy")
    np.save(out_path, mel_frames)
    print(f"Saved {out_path}  shape: {mel_frames.shape}")
    print(f"  {len(mel_frames)} frames = {N_chunks} chunks × {seq_frames} — perfectly aligned")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="./model")
    args = p.parse_args()
    extract(args.model_dir)
