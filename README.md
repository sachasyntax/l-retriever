═══════════════════════════════════════════════════════════
NEURAL LATENT AUDIO GENERATOR
Overlap AE + Bezier/fBm Walk + Continuous Audio Morphing
═══════════════════════════════════════════════════════════

DEPENDENCIES
────────────
pip install torch torchaudio librosa soundfile scikit-learn scipy numpy sounddevice

FILES
─────
train.py                  — full training + extraction pipeline
generate.py               — offline generation, exports WAV
livegenerate.py           — realtime streaming with tkinter parameter control
extract_audio_chunks.py   — extracts audio_chunks.npy from existing model (no retraining)
extract_mel_frames.py     — extracts mel_frames.npy from existing model (no retraining)
./model/                  — created by train.py

MODEL FILES
───────────
vae.pt            — trained autoencoder weights
meta.pkl          — architecture parameters
checkpoint.pt     — resumable training checkpoint
latents_raw.npy   — raw latent vectors per chunk (used for walk)
mel_frames.npy    — per-frame mel in log space, aligned with audio_chunks
audio_chunks.npy  — raw audio samples per chunk (used for synthesis)
band_stats.npy    — per-band mel statistics
mel_chunks.npy    — flattened mel per chunk
raw_chunks.npy    — raw FFT magnitude per chunk
gmm.pkl           — gaussian mixture model on latent space
pca.pkl           — PCA reduction
scaler.pkl        — latent scaler

TRAINING
────────
python train.py --data_dir /path/to/audio --output_dir ./model

options:
  --epochs      120   default, 50 sufficient for testing
  --latent_dim  64    default
  --resume            resume from checkpoint.pt

input: .wav .mp3 .flac .ogg
auto-converted to mono 22050Hz

After training, all files needed by generate are saved automatically.
No separate extraction scripts needed for a fresh training run.

estimated time on Apple M2 with ~1h of audio:
  50 epochs   →  20–30 min
  120 epochs  →  50–70 min

UPDATING EXISTING MODEL (no retraining)
────────────────────────────────────────
If you have an existing model without audio_chunks.npy or mel_frames.npy:

  python extract_mel_frames.py --model_dir ./model
  python extract_audio_chunks.py --data_dir ./audio --model_dir ./model

GENERATION
──────────
python generate.py --duration 60 --output out.wav

parameters:
  --duration          output length in seconds (required)
  --output            WAV path (default output.wav)
  --model_dir         model directory (default ./model)
  --step_size         fBm walk amplitude (default 0.3)
  --smoothing_window  walk velocity smoothing (default 3)
  --n_control         spline waypoints (default 16)
  --top_k             retrieval candidates (default 8)
  --variation         master variation knob (default 1.0)
  --temperature       retrieval softness (default 0.2)
  --rms_percentile    filter silent chunks below this percentile (default 20)
  --fade_ms           controls chunk overlap and morphing (default 50)

PARAMETER LOGIC
───────────────
step_size         small = slow latent evolution
                  large = wide timbral jumps

smoothing_window  high = inertial, gradual trajectory
                  low  = more reactive changes

n_control         more waypoints = more diverse regions visited
                  fewer = longer coherent arcs
                  auto-clamped to duration — never crashes

top_k             low  = sharp retrieval, closest match
                  high = diffuse blend, hybrid timbres

variation         master knob — scales step_size
                  < 1 = drone mode
                  > 1 = harsh/chaotic mode

temperature       low  = retrieval dominated by best match
                  high = all top-k candidates equally weighted
                  0.8–1.0 recommended for maximum chunk variety

rms_percentile    removes silent chunks from dataset pool
                  0 = no filter, all chunks available
                  raise if output still has silent gaps

fade_ms           controls overlap between chunks in the morph
                  50   → overlap 0.95 — extreme fusion, chunks unrecognizable
                  200  → overlap 0.80 — very smooth, good default for drone
                  500  → overlap 0.50 — balanced
                  900  → overlap 0.10 — minimal overlap, chunks more distinct
                  higher fade_ms = more chunks needed per second of output

ANTI-REPETITION
───────────────
Two mechanisms prevent loop artifacts:
  hard blacklist    — last top_k*2 chunks are excluded from selection
  EMA penalty       — recently used chunks are softly penalized over time
Both are always active. Do not disable unless intentional loop is desired.

EXAMPLES BY TYPE
────────────────
DRONE / SUSTAINED TEXTURE
  python generate.py --duration 300 --output drone.wav \
    --variation 0.2 --temperature 0.05 --smoothing_window 8 \
    --fade_ms 1000 --top_k 4 --n_control 8

HARSH NOISE
  python generate.py --duration 60 --output harsh.wav \
    --variation 2.0 --temperature 0.2 --top_k 16 \
    --fade_ms 30 --n_control 24

EVOLVING TEXTURE / AMBIENT
  python generate.py --duration 180 --output ambient.wav \
    --variation 0.8 --temperature 0.15 --n_control 20 \
    --fade_ms 200

MAXIMUM FUSION
  python generate.py --duration 60 --output fusion.wav \
    --fade_ms 50 --temperature 0.8 --top_k 16 --rms_percentile 0

LONG FORM
  python generate.py --duration 600 --output long.wav \
    --variation 1.0 --temperature 0.15 --n_control 48 --fade_ms 150

REALTIME
────────
python livegenerate.py --model_dir ./model

sliders control: step_size, smoothing_window, H, fbm_sigma, top_k
parameters update live at next chunk

ARCHITECTURE
────────────
Audio → pre-emphasis → Mel Spectrogram (128 mel, 22050Hz)
      → Conv1D AE (overlap consistency + std + isophonic + variance loss)
      → PCA (64 → 32) + GMM (12 components)
      → Bezier spline walk between farthest-point waypoints
      → multi-scale fBm perturbation (H=0.95/0.70/0.40)
      → velocity-smoothed latent trajectory
      → chunk-level soft retrieval (top-k weighted blend, full mel sequence)
      → hard blacklist + EMA penalty (anti-repetition)
      → continuous bell-window morph (raised cosine overlap-add)
      → peak normalize → WAV

No ISTFT. No OLA. No phase artifacts.
Synthesis uses raw audio samples from dataset — full spectral density guaranteed.
Chunks are never heard in isolation — always fused into a continuous morphing mass.

INTERNAL PARAMETERS (hardcoded in train.py)
────────────────────────────────────────────
SEQ_FRAMES = 64    chunk duration ≈ 1.49s
                   larger = longer coherent textures, slower training
                   smaller = faster transitions, more fragmented

HOP = 512          STFT hop size (mel extraction only, not synthesis)
N_MELS = 128       mel bands — do not change without retraining
LATENT_DIM = 64    latent space dimension — do not change without retraining

═══════════════════════════════════════════════════════════
