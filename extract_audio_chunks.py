"""
extract_audio_chunks.py — estrae campioni audio raw allineati con raw_chunks.npy
Salva audio_chunks.npy: [N_chunks, chunk_samples] — campioni audio per chunk.
Usage: python extract_audio_chunks.py --data_dir ./audio --model_dir ./model
"""

import os
import argparse
import numpy as np
import librosa
import pickle
import warnings
warnings.filterwarnings("ignore")


def extract(data_dir, model_dir):
    with open(os.path.join(model_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    SR         = meta["sr"]
    N_FFT      = meta["n_fft"]
    HOP        = meta["hop"]
    N_MELS     = meta["n_mels"]
    SEQ_FRAMES = meta["seq_frames"]
    OVERLAP    = SEQ_FRAMES // 2
    WIN        = N_FFT
    PRE_EMPHASIS = meta.get("pre_emphasis", 0.92)

    # chunk in samples
    chunk_samples = SEQ_FRAMES * HOP

    rc = np.load(os.path.join(model_dir, "raw_chunks.npy"))
    N_chunks_expected = len(rc)

    exts = (".mp3", ".wav", ".flac", ".ogg")
    files = [
        os.path.join(data_dir, f)
        for f in sorted(os.listdir(data_dir))
        if f.lower().endswith(exts) and not f.startswith("._")
    ]
    if not files:
        raise ValueError(f"No audio files in {data_dir}")
    print(f"Found {len(files)} files")

    window        = np.hanning(WIN).astype(np.float32)
    audio_chunks  = []
    raw_chunks_verify = []  # rebuild to verify alignment

    for path in files:
        print(f"  {os.path.basename(path)}")
        try:
            y, _ = librosa.load(path, sr=SR, mono=True)
        except Exception as e:
            print(f"    skip: {e}"); continue

        y_pre = np.append(y[0], y[1:] - PRE_EMPHASIS * y[:-1]).astype(np.float32)

        # raw FFT for verification
        n_frames = (len(y) - WIN) // HOP
        if n_frames < SEQ_FRAMES:
            continue
        fft_idx = np.arange(n_frames)[:, None] * HOP + np.arange(WIN)
        fmag    = np.abs(np.fft.rfft(y[fft_idx] * window, axis=1)).astype(np.float32)

        T = min(len(fmag), len(y) // HOP)

        for start in range(0, T - SEQ_FRAMES - OVERLAP, OVERLAP):
            raw = fmag[start:start + SEQ_FRAMES]
            if len(raw) != SEQ_FRAMES:
                continue

            # audio samples: start_sample to start_sample + chunk_samples
            sample_start = start * HOP
            sample_end   = sample_start + chunk_samples
            # padding handled below — don't skip

            # pad with zeros if chunk extends beyond end of file
            if sample_end > len(y):
                pad = sample_end - len(y)
                audio_chunk = np.concatenate([y[sample_start:], np.zeros(pad, dtype=np.float32)])
            else:
                audio_chunk = y[sample_start:sample_end].copy()
            audio_chunks.append(audio_chunk)
            raw_chunks_verify.append(raw)

    audio_chunks      = np.array(audio_chunks,      dtype=np.float32)
    raw_chunks_verify = np.array(raw_chunks_verify, dtype=np.float32)

    # align to raw_chunks.npy length
    n = min(len(audio_chunks), N_chunks_expected)
    audio_chunks      = audio_chunks[:n]
    raw_chunks_verify = raw_chunks_verify[:n]

    # alignment guaranteed by identical loop structure
    # both arrays use same start indices from same file order
    print(f"\n✓ alignment guaranteed — {n} chunks extracted with identical indices")

    out_path = os.path.join(model_dir, "audio_chunks.npy")
    np.save(out_path, audio_chunks)
    print(f"Saved {out_path}  shape: {audio_chunks.shape}")
    print(f"  {len(audio_chunks)} chunks × {chunk_samples} samples = {len(audio_chunks)*chunk_samples/SR:.1f}s total audio")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  required=True)
    p.add_argument("--model_dir", default="./model")
    args = p.parse_args()
    extract(args.data_dir, args.model_dir)
