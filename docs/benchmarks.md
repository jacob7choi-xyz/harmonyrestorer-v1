# Benchmarks

## OpGAN v1 (2026-03-11)

Training: 100 epochs, ~51 hours on Tesla T4, ~$41 total GCP cost (training + inference + eval). Model operates at 16kHz mono. Input at other sample rates is resampled automatically; output is always 16kHz.

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

## UVR Baseline

TODO: benchmark UVR on the same dataset for head-to-head comparison.
