# ADR-004: OpGAN Inference, Evaluation, and Model Comparison

**Date:** 2026-03-14

**Status:** accepted

---

## Context

After training the OpGAN for 100 epochs on 146,200 noisy/clean pairs, we needed infrastructure to:

1. Run inference on arbitrary-length audio (the model expects fixed 32,000-sample inputs)
2. Evaluate restoration quality with standard metrics (SDR, PESQ, STOI)
3. Compare OpGAN against UVR (the production baseline) on the same dataset
4. Test on out-of-distribution audio (real historical recordings vs. synthetic noise)

The model operates at 16 kHz mono -- a constraint inherited from the training data pipeline (ADR-002). Any input at a different sample rate or channel count must be converted before inference and evaluation.

## Decision

### Inference Pipeline (`dataset/infer.py`, `dataset/infer_uvr.py`)

**Overlap-add chunking** for arbitrary-length audio:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Frame length | 32,000 samples (2s) | Matches generator input size |
| Overlap | 1,600 samples (100ms) | 5% overlap with linear crossfade eliminates boundary artifacts |
| Window | Linear fade-in/fade-out | Simple, effective; first chunk skips fade-in, last skips fade-out |

**Checkpoint loading** filters out Self-ONN internal cache keys (`_operator_weights_cache`, `_cache_valid`) that are present in the saved state dict but cause errors on load. Uses `strict=False` since the filtered keys are runtime caches, not learned parameters.

**Auto-resampling** via `librosa.resample` (Kaiser-windowed sinc filter) for non-16kHz inputs. Chosen over `scipy.signal.resample` (FFT-based) due to better anti-aliasing and fewer ringing artifacts at non-integer resampling ratios.

**UVR inference** (`infer_uvr.py`) uses `audio-separator`'s `Separator` class directly (not the backend's `DenoiserService`) to keep dataset scripts independent. UVR internally upsamples to 44.1 kHz stereo, so outputs are resampled back to 16 kHz mono via librosa for apples-to-apples evaluation.

**Atomic writes** via temp file + rename prevent corrupt output files if the process is interrupted.

**Resume support**: both scripts skip files that already exist in the output directory, enabling interrupted batch runs to continue without reprocessing.

### Evaluation Pipeline (`dataset/evaluate.py`)

**Metrics chosen:**

| Metric | Range | Why |
|--------|-------|-----|
| SDR (Signal-to-Distortion Ratio) | -inf to +inf dB | Standard signal-level quality measure; computed directly (not via mir_eval) to avoid O(n!) permutation solver overhead for the single-source case |
| PESQ (Perceptual Evaluation of Speech Quality) | -0.5 to 4.5 | Best available perceptual quality metric; designed for speech but widely used for music when no music-specific alternative exists |
| STOI (Short-Time Objective Intelligibility) | 0.0 to 1.0 | Complements PESQ with a time-domain intelligibility perspective |

**File matching**: restored files are matched to clean references by stem name. Noisy variants (`frame__v00.wav`) are mapped back to their base clean frame (`frame.wav`).

**Failure handling**: files that fail evaluation (silent frames, PESQ "No utterances detected" on quiet passages) are skipped and counted. The OpGAN benchmark skipped 15,168 of 146,200 files (10.4%) -- all quiet classical music frames where PESQ requires minimum signal energy. This 89.6% coverage is acceptable and documented.

### Comparison Methodology

Both models are evaluated on the same 146,200-file dataset using identical metrics and matching logic. The clean references are the original unmodified frames from the dataset pipeline. This ensures a fair head-to-head comparison, with the caveat that the OpGAN was trained on this data (in-distribution) while UVR was not.

## Results (2026-03-11)

### OpGAN v1 Benchmark (in-distribution)

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| SDR (dB) | 23.74 | 4.66 | 23.85 |
| PESQ | 4.04 | 0.59 | 4.28 |
| STOI | 0.960 | 0.069 | 0.989 |

- 131,027 files evaluated, 15,168 skipped (quiet frames)
- Inference: 146,195/146,195 files, zero failures, ~267x realtime on Tesla T4
- Training: 100 epochs, ~51 hours on Tesla T4, ~$48 GCP Compute Engine cost

### Out-of-Distribution Test (William Primrose recording)

The model was tested on a real historical recording (44.1 kHz MP3, ~3:30 duration). Results: the model did not generalize well to real analog noise, producing audible artifacts rather than clean output. This is expected -- the model was trained on synthetic noise only and has never seen real tape/vinyl degradation patterns. Future work should include real noise samples in the training data.

### UVR Baseline

Pending -- `infer_uvr.py` is written and tested locally, awaiting GCP VM availability to run on the full dataset.

## Consequences

**Positive:**
- Overlap-add chunking handles any audio length without quality loss at boundaries
- Resume support makes batch processing robust to interruptions (critical for 146k-file runs)
- Direct SDR formula avoids mir_eval's expensive permutation solver
- Atomic writes prevent data corruption on crash
- Same evaluation pipeline works for both OpGAN and UVR, ensuring fair comparison

**Negative:**
- 16 kHz mono limitation means the model cannot restore high-frequency content above 8 kHz -- production use requires either training at higher sample rates or a super-resolution post-processing step (see AudioSR in references.md)
- In-distribution evaluation inflates metrics -- real-world performance will be lower
- PESQ is a speech metric used as a proxy for music quality; no standard music-specific perceptual metric exists

**Neutral:**
- librosa is now a runtime dependency for inference (was previously only needed for preprocessing)
- Checkpoint loading requires filtering Self-ONN cache keys -- a quirk of the library, not a design flaw

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| mir_eval.bss_eval_sources for SDR | O(n!) permutation solver is unnecessary for single-source denoising; direct formula gives identical results in O(n) |
| scipy.signal.resample for resampling | FFT-based approach causes ringing artifacts at non-integer ratios (e.g., 44100 -> 16000); librosa's Kaiser-windowed sinc filter produces cleaner output |
| Evaluate at native sample rate | Would require resampling clean references up; better to standardize everything at 16 kHz to match the model's operating frequency |
| Skip PESQ for music evaluation | Despite being speech-focused, PESQ is the most widely reported perceptual metric in the audio restoration literature and enables comparison with published results |
