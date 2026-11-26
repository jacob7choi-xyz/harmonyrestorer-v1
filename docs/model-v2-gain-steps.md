# Model v2: which parts of the degradation does v1's SI-SNR gain come from?

This document fixes, before the measurement runs, how it is measured and how its result is read. The code is `benchmark/measurements/gain_steps.py`.

> **Note:** No v1 output on any of the conditions below had been computed when this document was written on 2026-11-25. The benchmark's systems stage scored test recordings only, and Phase 0 did not use these conditions. The merge of this document is the timestamp for the readings below.

## Why

Two measurements of v1's SI-SNR improvement differ (6.03 and 5.33 dB against 1.90 dB):

- In the 2026-11-23 benchmark, the mean SI-SNR of v1's output was 6.03 dB (piano) and 5.33 dB (chamber) above that of the degraded input (the README's results tables).
- In Phase 0 (`docs/model-v2-phase0.md`), v1's mean SI-SNR improvement on its own synthetic noise was 1.90 dB over the six demo recordings.

The two differ in four ways:

1. The benchmark's degradation always applies a low-pass filter (cutoff 4 to 8 kHz). Phase 0 moved it to the Nyquist rate.
2. The benchmark's degradation applies saturation on some frames. Phase 0 switched it off.
3. The benchmark's noise is at the level training drew it. Phase 0 rescaled each noise to 5, 10, 15 and 20 dB SNR. `benchmark/degradation.py` notes that training's hiss is scaled as though its unit peak were unit power, which made it quieter than its recorded SNR.
4. The low-pass delays the degraded input by about one sample, and SI-SNR credits undoing that delay.

The README already reports part of difference 4. Under the shift-tolerant SI-SNR, which scores each output at the best of a -1, 0 and +1 sample shift, v1's output is 3.30 dB (piano) and 2.25 dB (chamber) above the degraded input. This measurement separates differences 1, 2 and 3 on the same frames. It answers two questions the README does not:

- Does the noise level explain Phase 0's low synthetic figure? If so, Phase 0's comparison of real and synthetic noise could be redone at training level.
- Does undoing the low-pass carry part of v1's gain beyond the one-sample delay? If so, the README may need a sentence about it.

Model v2's training work stays stopped whatever this shows, as Phase 0's rule states.

## Data

The clean audio and the frames are Phase 0's: the benchmark's `demo` partition (6 CC0 solo piano recordings of the 2026-11-23 walk, `data/benchmark_runs/20261123T214738Z`), up to 40 eligible frames per recording at evenly spaced positions (`real_noise_gap.frame_indices`). No test recording is scored, and no new data is downloaded.

## Conditions

Each frame is degraded four ways. All four use the frame's benchmark seed, gain and parameter draws, so the noise is drawn once: crackle and hum share their samples across conditions, and hiss shares its waveform, with its level set per condition (see Limits).

| Condition | Low-pass | Saturation | Noise level |
|---|---|---|---|
| `benchmark` | As drawn | As drawn | As drawn |
| `no_lowpass` | Moved to the Nyquist rate | As drawn | As drawn |
| `added_noise_only` | Moved to the Nyquist rate | Off | As drawn |
| `snr_grid` | Moved to the Nyquist rate | Off | 5, 10, 15 and 20 dB SNR |

- `benchmark` is the benchmark's degraded frame exactly; the code refuses to run if it is not identical to `degrade_with_reference`'s.
- `snr_grid` takes the noise of `added_noise_only` (the degraded frame minus the clean frame at training level) and scales it so that the clean reference's power over the noise's power is each SNR, as Phase 0 did. Its values are means over the frames and the four SNRs.

## What is measured

v1 (the shipped checkpoint, through the benchmark's in-process engine) restores each input. The input and the output are scored against the clean reference with the benchmark's `score_output`. Per recording and condition, the measurement reports the mean of:

- the input's SI-SNR and the output's SI-SNR;
- the SI-SNR improvement: the output's SI-SNR minus the input's;
- the shift-tolerant SI-SNR improvement: the same with v13's shift-tolerant SI-SNR (maximum shift 1 sample);
- the log-spectral distance of v1's output.

## The steps

Each step removes one part, in this order, and compares the two adjacent conditions:

| Step | Difference |
|---|---|
| `lowpass` | `benchmark` minus `no_lowpass` |
| `saturation` | `no_lowpass` minus `added_noise_only` |
| `noise_level` | `added_noise_only` minus `snr_grid` |

For each step, and for both the SI-SNR improvement and the shift-tolerant SI-SNR improvement, the per-recording difference in the mean improvement is computed. The step's statistic is the mean of the six differences, with the 95% bootstrap interval that Phase 0 used (recordings resampled with replacement, with the protocol's seed and iteration count).

A step is called a contributor on a metric when both hold:

1. the mean difference is at least 1 dB;
2. its 95% interval lies above zero.

> **Note:** The 1 dB margin is a judgement made before the run, not a derived threshold. The steps interact, so removing the parts in another order could give other shares; the order above is fixed here and not varied.

## How the result is read

These readings are fixed here. Each one leads only to a proposal, which needs its own plan and approval.

- **`noise_level` is a contributor on SI-SNR improvement:** the SNR grid accounts for part of Phase 0's low synthetic figure. The proposal is a new comparison of real and synthetic noise at training level, under its own written rule. It would be a new measurement prompted by this result, not a second attempt at Phase 0's gate, whose result stands.
- **`lowpass` is a contributor on SI-SNR improvement but not on the shift-tolerant improvement:** the low-pass's share is the delay that the README's shift-tolerant rows already disclose. No README change is proposed.
- **`lowpass` is a contributor on both:** undoing the low-pass carries part of v1's gain beyond the delay. The proposal is a README sentence saying so, in its own pull request.
- **No step is a contributor:** the next step is decided from what the measurement shows.

Every step's figures are reported, whichever way they fall.

## Limits

- Six recordings, all solo piano, two of them by one performer.
- The `benchmark` condition here is on demo recordings, not the test recordings, so its figures are not expected to equal the README's.
- Hiss is scaled from the signal's power after the low-pass and saturation, so its level can differ slightly between `benchmark`, `no_lowpass` and `added_noise_only`. The input SI-SNR column shows how much.
- The `snr_grid` noise is drawn from the benchmark's seeds, not Phase 0's, so Phase 0's 1.90 dB is not expected to be reproduced exactly.
- The degradation clips to [-1, 1], and the noise of `added_noise_only` includes any clipping. How many samples clip was not measured.
