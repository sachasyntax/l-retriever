[README.md](https://github.com/user-attachments/files/28387683/README.md)
# L-RETRIEVER

Latent space walker + corpus retrieval synthesizer for noise and drone generation.

Trains a convolutional autoencoder on an audio dataset, navigates the learned latent space via Bezier spline + fractional Brownian motion, and retrieves audio chunks from the corpus via mel-space similarity. Output is a continuous morphing mass — no synthesis, no ISTFT, no phase artifacts.

---

## How it works

1. **Train** — a Conv1D autoencoder learns a latent representation of the audio corpus. Every chunk in the dataset is encoded and stored.
2. **Walk** — at generation time, a trajectory is computed through latent space using Bezier splines between maximally distant waypoints, perturbed by multi-scale fBm (H=0.95/0.70/0.40).
3. **Retrieve** — at each step, the decoder produces a mel query. The top-k most similar mel frames in the corpus are found via cosine similarity and blended with softmax weights. Hard blacklist + EMA penalty prevent repetition.
4. **Morph** — retrieved raw audio chunks are overlap-added with a raised cosine bell window. Overlap ratio is controlled by `--fade_ms`. At high overlap, 20+ chunks are active simultaneously — individual chunks dissolve into the mass.

The model does not generate audio. It navigates a timbral space and retrieves material from the corpus. Full spectral density is guaranteed.

---

## Install

    pip install torch torchaudio librosa soundfile scikit-learn scipy numpy sounddevice

---

## Usage

**Train**

    python train.py --data_dir ./audio --output_dir ./model
    python train.py --data_dir ./audio --output_dir ./model --epochs 50 --resume

Accepts `.wav`, `.mp3`, `.flac`, `.ogg`. Saves all files needed for generation automatically.

**Generate (offline)**

    python generate.py --duration 60 --output out.wav

**Stream (live)**

    python livegenerate.py --variation 1.5 --fade_ms 200 --top_k 12

Streams continuously to audio output. Press `Ctrl+C` to stop.

Update parameters live by typing `param=value` and pressing Enter while the stream is running:

    temperature=0.8
    variation=2.0
    fade_ms=100
    top_k=16

Changes take effect at the next chunk (~1.5s latency).

---

## Parameters

| param | default | description |
|---|---|---|
| `--duration` | required | output length in seconds (generate only) |
| `--output` | output.wav | output path (generate only) |
| `--step_size` | 0.3 | fBm walk amplitude |
| `--smoothing_window` | 3 | walk velocity smoothing |
| `--n_control` | 16 | spline waypoints |
| `--top_k` | 8 | retrieval candidates |
| `--variation` | 1.0 | master knob — <1 drone, >1 harsh |
| `--temperature` | 0.2 | retrieval softness |
| `--rms_percentile` | 20 | silence filter threshold |
| `--fade_ms` | 50 | chunk overlap — 50=extreme fusion, 950=minimal |

**`--fade_ms` reference**

| value | overlap | result |
|---|---|---|
| 50 | 0.95 | chunks completely dissolved |
| 200 | 0.80 | very smooth |
| 500 | 0.50 | balanced |
| 950 | 0.05 | chunks mostly distinct |

---

## Architecture

    Audio → pre-emphasis → Mel (128 bands, 22050Hz)
          → Conv1D AE (overlap consistency + spectral + std + variance loss)
          → latent vectors extracted per chunk
          → Bezier spline walk between farthest-point waypoints
          → multi-scale fBm perturbation (H=0.95 / 0.70 / 0.40)
          → velocity-smoothed trajectory
          → mel query → top-k corpus retrieval (cosine, power-compressed)
          → hard blacklist + EMA penalty (anti-repetition)
          → raised cosine overlap-add morph
          → peak normalize → WAV / audio stream

---

## Internal parameters (train.py)

| param | default | note |
|---|---|---|
| `SEQ_FRAMES` | 64 | chunk duration ≈ 1.49s |
| `HOP` | 512 | STFT hop (mel only, not synthesis) |
| `N_MELS` | 128 | do not change without retraining |
| `LATENT_DIM` | 64 | do not change without retraining |

---

## Updating an existing model

If you have a trained model without `audio_chunks.npy` or `mel_frames.npy`:

    python extract_mel_frames.py --model_dir ./model
    python extract_audio_chunks.py --data_dir ./audio --model_dir ./model
