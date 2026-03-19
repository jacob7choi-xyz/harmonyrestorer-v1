# Benchmarks

## OpGAN v1 (2026-03-11)

Training: 100 epochs, ~51 hours on Tesla T4, ~$100 total GCP cost (all training, inference, and eval for both models). Model operates at 16kHz mono. Input at other sample rates is resampled automatically; output is always 16kHz.

| Metric   | Mean  | Std  | Median |
|----------|-------|------|--------|
| SDR (dB) | 23.74 | 4.66 | 23.85  |
| PESQ     | 4.04  | 0.59 | 4.28   |
| STOI     | 0.960 | 0.069| 0.989  |

- Evaluated 131,027 / 146,200 files (89.6% coverage)
- 15,168 skipped: quiet/silent frames where PESQ detected no utterances (natural pauses, soft endings). SDR and STOI computed fine for these -- only PESQ requires minimum signal energy. Skipping these makes results slightly conservative.
- Dataset: Musopen classical, 146,200 noisy/clean pairs, 5 noise variants per frame
- Caveat: evaluated on same distribution as training (in-distribution)
- Full per-file results: `opgan_metrics.json` (not committed, 33 MB)
- Checkpoint: `checkpoints/final.pt` (not committed, 17 MB)

## UVR Baseline (2026-03-18)

Model: UVR-DeNoise.pth via audio-separator. UVR internally upsamples to 44.1kHz stereo; output resampled back to 16kHz mono (librosa Kaiser-windowed sinc filter) for fair comparison. Inference: ~65 hours on Tesla T4 CPU (no ONNX GPU acceleration), ~$100 total GCP cost shared with OpGAN (training + all inference + eval).

| Metric   | Mean  | Std  | Median |
|----------|-------|------|--------|
| SDR (dB) | 11.86 | 6.52 | 11.42  |
| PESQ     | 3.72  | 0.93 | 4.14   |
| STOI     | 0.953 | 0.079| 0.985  |

- Evaluated 131,013 / 146,195 files (89.6% coverage)
- 15,182 skipped: same quiet frames as OpGAN (PESQ no utterances detected)
- Full per-file results: `uvr_metrics.json` (not committed)

## Head-to-Head Comparison

| Metric   | OpGAN  | UVR   | Delta        |
|----------|--------|-------|--------------|
| SDR (dB) | **23.74** | 11.86 | +11.88 dB |
| PESQ     | **4.04**  | 3.72  | +0.32     |
| STOI     | **0.960** | 0.953 | +0.007    |

OpGAN outperforms UVR across all three metrics. The SDR gap (~12 dB) is substantial -- roughly 4x better noise reduction. PESQ and STOI differences are smaller but consistent.

**Caveat**: Both models evaluated in-distribution (same dataset as OpGAN training). UVR was not trained on this data, so it is a genuinely out-of-distribution test for UVR but not for OpGAN. Out-of-distribution evaluation for OpGAN is a planned next step.
