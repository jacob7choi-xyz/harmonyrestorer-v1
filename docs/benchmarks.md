# Benchmarks

> **Note:** The figures below are **withdrawn as generalization evidence** as of 2026-07-24. They are retained as a record of what was measured and how. An internal audit found that the evaluation population was not disjoint from training at the level of source material, and that coverage accounting was not comparable between the two systems. Details in [Withdrawal notice](#withdrawal-notice-2026-07-24).

## OpGAN v1 (2026-03-11)

Training: 100 epochs, ~51 hours on Tesla T4, ~$100 total GCP cost (all training, inference, and eval for both models). Model operates at 16kHz mono. Input at other sample rates is resampled automatically; output is always 16kHz.

| Metric   | Mean  | Std  | Median |
|----------|-------|------|--------|
| SDR (dB) | 23.74 | 4.66 | 23.85  |
| PESQ     | 4.04  | 0.59 | 4.28   |
| STOI     | 0.960 | 0.069| 0.989  |

- Evaluated 131,027 / 146,200 files (89.6% coverage)
- 15,168 skipped: quiet/silent frames where PESQ detected no utterances (natural pauses, soft endings). SDR and STOI computed fine for these; only PESQ requires minimum signal energy. Skipping these makes results slightly conservative.
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

OpGAN outperforms UVR across all three metrics. The SDR gap (~12 dB) is substantial, roughly 4x better noise reduction. PESQ and STOI differences are smaller but consistent.

**Caveat**: Both models evaluated in-distribution (same dataset as OpGAN training). UVR was not trained on this data, so it is a genuinely out-of-distribution test for UVR but not for OpGAN. Out-of-distribution evaluation for OpGAN is a planned next step.

## Withdrawal notice (2026-07-24)

The figures above are withdrawn as evidence of generalization. They are kept here because the record of a flawed measurement is more useful than its deletion.

**Source material was shared between training and evaluation.** The corpus holds 5 noise variants per clean 2-second frame (29,240 frames, 146,200 pairs), and `train.py` splits 90/10 with `random_split` over *pairs*, not over frames or recordings. The probability that a given frame's five variants all land in training is 0.9^5, so roughly 41% of clean frames are expected to have variants on both sides of the split. Beyond that, frames are consecutive slices of 145 recordings, so even a frame-disjoint split would share performer, instrument, room, and mastering across the boundary. The only defensible split unit is the recording, and every one of the 145 recordings contributed to training. No track-disjoint held-out data exists in this corpus.

**The evaluation population was the full corpus.** Evaluated plus skipped equals the dataset total for both systems (131,027 + 15,173 = 146,200; 131,013 + 15,182 = 146,195), so evaluation was not restricted to the validation split.

**Coverage accounting was not comparable between systems.** `evaluate_directory` increments `skipped` on a missing clean reference *or* on any exception raised while scoring a file, so each system's denominator was shaped by its own failures rather than by an eligibility rule fixed before either ran. Separately, the SDR mean filters non-finite values while the reported `count` does not, so the published N is not the N behind the SDR figure. The two runs also report different totals (146,200 vs 146,195) for reasons not established.

**What the figures do and do not support.** They characterize reconstruction quality over the corpus the model was fit on, which is a real but narrow result: at most, generalization to unseen noise realizations of largely seen content. They do not support a claim about unseen recordings. The head-to-head margin is the most affected claim of all, because UVR is a pretrained baseline that never saw any of this data while OpGAN trained on roughly 90% of it, so the comparison carries a structural advantage for OpGAN that no caveat about distribution conveys.

**Replacement.** A benchmark protocol is being written before any new data is acquired, fixing the split unit as the recording, assigning partitions before frame extraction so every derived artifact inherits its partition, determining evaluation eligibility before either system runs, enforcing an identical scored population fail-closed, and aggregating per recording rather than per frame (adjacent frames of one performance are not independent observations, so frame-count N overstates confidence). A separate frozen test partition will be reserved from model-selection decisions, and demo material will be curated from a distinct pool so that choosing appealing examples cannot touch the test set.
