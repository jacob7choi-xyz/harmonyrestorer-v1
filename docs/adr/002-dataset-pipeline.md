# ADR-002: Dataset Pipeline for OpGAN Training

**Date:** 2026-03-07

**Status:** accepted

---

## Context

HarmonyRestorer's production denoiser uses UVR (via `audio-separator`), a pretrained model. The project's long-term goal is to train a custom OpGAN (1D Operational GAN) that specializes in reducing hiss from analog classical music recordings -- a domain where general-purpose denoisers underperform because analog noise is not purely additive white noise.

Training the OpGAN requires paired data: clean audio and synthetically degraded versions of the same audio. No existing public dataset targets analog music noise specifically (DEMAND and VoiceBank-DEMAND focus on speech; DNS Challenge focuses on modern noise environments).

## Decision

Build a modular dataset pipeline in `dataset/` that handles four stages:

1. **Acquisition** (`acquire.py`) -- Download CC/public domain classical recordings from Internet Archive (Musopen collection) using the `internetarchive` Python library. Supports `--formats` (e.g. `.flac`), `--exclude` (skip by keyword), and `--manifest` for custom sources.

2. **Preprocessing** (`preprocess.py`) -- Resample to 16kHz mono, peak-normalize, slice into 2-second frames (32,000 samples matching the generator's `input_length`). Discard near-silent frames automatically.

3. **Noise synthesis** (`noise.py`) -- Generate realistic analog degradation:
   - Tape hiss (pink/brown noise with frequency shaping, not white noise)
   - Vinyl crackle (Poisson-distributed impulse noise with decay)
   - Mains hum (60/50 Hz fundamental + harmonics)
   - High-frequency rolloff (Butterworth low-pass, simulating old equipment)
   - Tape saturation (tanh soft clipping, simulating magnetic hysteresis)

4. **Pair generation** (`generate_pairs.py`) -- For each clean frame, produce N noisy variants with randomized degradation combinations and SNR levels. Save degradation metadata as JSON for reproducibility.

5. **PyTorch integration** (`torch_dataset.py`) -- `AnalogAudioDataset` class returns `(noisy, clean)` tensor pairs shaped `[1, 32000]`, ready for the OpGAN generator.

### Key Design Choices

| Choice | Rationale |
|--------|-----------|
| 16kHz mono, 2s frames | Matches OpGAN generator's expected input `[B, 1, 32000]` |
| Synthetic noise over real noise | Controllable, reproducible, unlimited quantity, legally clean |
| Pink noise over white noise | Real tape hiss has -3 dB/octave rolloff, not flat spectrum |
| SNR range 5-30 dB | Covers heavy degradation (5 dB) to barely perceptible (30 dB) |
| Randomized multi-effect combinations | Forces the model to generalize, not overfit to one noise type |
| Metadata JSON per pair | Enables analysis of which noise types the model handles well/poorly |
| Internet Archive as primary source | No API key needed, direct downloads, public domain, reliable |
| `internetarchive` Python lib | Official IA client, handles retries/auth/metadata natively |
| `--exclude` keyword filter | Filters before download to avoid wasting bandwidth |
| scipy for filters | Standard DSP library, already a transitive dependency via audio-separator |

## Results (2026-03-07)

First full run of the pipeline:
- **Source**: MusopenCollectionAsFlac -- 145 FLAC files, 7.2 GB, 14 composers (Bach, Beethoven, Borodin, Brahms, Dvorak, Grieg, Haydn, Mendelssohn, Mozart, Rimsky-Korsakov, Schubert, Smetana, Suk, Tchaikovsky)
- **Clean frames**: 29,240 (2s each at 16kHz mono)
- **Training pairs**: 146,200 (5 noisy variants per frame)
- **Total disk**: 33 GB (gitignored under `data/`)

## Consequences

**Positive:**
- Training can begin once ~50 tracks are downloaded (minutes, not days)
- Noise synthesis is modular -- new degradation types can be added without changing the pipeline
- All training data is legally distributable (CC/public domain sources + synthetic noise)
- Metadata enables ablation studies (e.g., "does the model struggle with crackle?")

**Negative:**
- Synthetic noise may not perfectly match real analog recordings -- the model's real-world performance depends on how well our synthesis approximates actual degradation
- Resampling to 16kHz discards high-frequency content above 8kHz -- acceptable for the current model but limits future work at higher sample rates
- ~~Linear interpolation for resampling~~ Resolved: switched to `librosa.resample` (polyphase filtering) for higher quality resampling

**Limitations:**
- No real analog noise samples yet -- could improve by recording actual tape/vinyl noise floors and using those instead of synthetic approximations
- No data augmentation beyond noise (no pitch shifting, time stretching, etc.)
- Single validation track (Primrose/Tchaikovsky) kept local, not in the dataset pipeline
