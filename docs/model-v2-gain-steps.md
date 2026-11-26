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

## Result

The measurement ran on 2026-11-26 on main `8d14461` (the merge of this document) and wrote `data/gain_steps_1/gain_steps.json` (not tracked; SHA-256 `c0431079b9b03d83df61b60c82382877a8887fe35b297e8f270c4eb37d29a51f`). All six recordings were measured on 40 frames each, and v1 was called 1,680 times without a failure.

### The steps

| Step | SI-SNR improvement: mean difference (95% interval), contributor | Shift-tolerant SI-SNR improvement: mean difference (95% interval), contributor |
|---|---|---|
| `lowpass` | 10.89 dB (9.99 to 11.90), yes | 6.61 dB (5.89 to 7.34), yes |
| `saturation` | 0.999 dB (0.67 to 1.38), no | 1.02 dB (0.70 to 1.40), yes |
| `noise_level` | -7.42 dB (-8.34 to -6.44), no | -6.90 dB (-7.78 to -5.95), no |

> **Note:** The `saturation` step's SI-SNR mean is 0.9985 dB, just under the 1 dB margin, so it is not a contributor on that metric; on the shift-tolerant metric it is 1.022 dB and is one. No reading above depends on the `saturation` step.

> **Note:** The Limits section, written before the run, says the input SI-SNR column shows how much the hiss level differs between conditions. It does not: the input SI-SNR also changes because the low-pass and saturation change the signal against the full-band reference. The hiss level difference was computed afterwards on the 240 measured frames, from the same signal powers the degradation uses and with no call to v1. Between `benchmark` and `no_lowpass` the hiss level differs by 0.008 dB or less on every frame. Between `no_lowpass` and `added_noise_only`, hiss is 0.80 dB louder on average in `no_lowpass` (0 to 6.83 dB per frame), because saturation raises the signal's power.

### Means over the six recordings

| Condition | Input SI-SNR (dB) | Output SI-SNR (dB) | SI-SNR improvement (dB) | Shift-tolerant improvement (dB) | Output log-spectral distance |
|---|---|---|---|---|---|
| `benchmark` | 14.66 | 20.70 | 6.04 | 3.17 | 14.00 |
| `no_lowpass` | 21.60 | 16.75 | -4.85 | -3.44 | 13.98 |
| `added_noise_only` | 22.65 | 16.80 | -5.85 | -4.46 | 13.72 |
| `snr_grid` | 12.54 | 14.10 | 1.56 | 2.43 | 16.84 |

### Per recording

| Recording | SI-SNR improvement (dB): `benchmark`, `no_lowpass`, `added_noise_only`, `snr_grid` | Shift-tolerant improvement (dB), same order |
|---|---|---|
| Ballade no. 3, Op. 47 | 6.79, -3.40, -4.16, 1.35 | 3.43, -2.05, -2.84, 2.17 |
| Mazurka, Op. 56 no. 2 | 5.10, -5.49, -6.98, 1.86 | 1.66, -3.99, -5.52, 2.71 |
| Mazurka, Op. 50 no. 3 | 6.77, -6.23, -6.80, 1.64 | 2.90, -4.73, -5.30, 2.53 |
| Scherzo no. 2, Op. 31 | 6.22, -5.36, -6.05, 0.99 | 3.46, -4.02, -4.70, 1.89 |
| Waltz, Op. 64 no. 3 | 6.23, -4.56, -6.33, 2.11 | 4.58, -3.16, -4.93, 2.95 |
| Waltz in A minor, B. 150 | 5.11, -4.07, -4.80, 1.43 | 2.98, -2.69, -3.48, 2.36 |

### Readings

As fixed above:

- **`lowpass` is a contributor on both metrics.** In the rule's words, undoing the low-pass carries part of v1's gain beyond the one-sample delay. The measured result is larger than "part": on these frames, v1's mean SI-SNR improvement is positive only when the input was low-passed (6.04 dB for `benchmark`) and negative without the low-pass (-4.85 dB for `no_lowpass`), so the step (10.89 dB) is larger than the whole improvement on `benchmark`. The proposal is a README sentence saying so, in its own pull request.
- **`noise_level` is not a contributor.** Its mean is negative: v1's SI-SNR improvement was larger at Phase 0's SNR grid (1.56 dB) than at training level (-5.85 dB). The SNR grid does not account for Phase 0's low synthetic figure, and no new real-versus-synthetic comparison is proposed from this result.

### Observations outside the rule

These were seen in the results and were not among the readings fixed before the run. They are reported, not acted on.

- When the input had no low-pass at training level, v1's mean SI-SNR improvement was negative on every recording: -3.40 to -6.23 dB for `no_lowpass` and -4.16 to -6.98 dB for `added_noise_only`. The mean shift-tolerant improvement is also negative on every recording for both conditions, so a shift of one sample does not account for it. These are means over each recording's 40 frames; single frames were not examined.
- v1's output log-spectral distance is about the same for `benchmark` (14.00) and `no_lowpass` (13.98), while its output SI-SNR differs by 3.95 dB. One possible reading is that, on input without the low-pass, v1's output differs from the clean reference mostly in phase or timing rather than in its magnitude spectrum. This measurement does not test that, and whether the difference is audible was not checked.
- Training-level noise is quieter than Phase 0's grid: the mean input SI-SNR is 22.65 dB for `added_noise_only` against 12.54 dB for `snr_grid`.
- The `benchmark` condition's mean SI-SNR improvement on these demo recordings, 6.04 dB, is close to the README's 6.03 dB on the piano test recordings. The two are on different recordings, as Limits states.

## Limits

- Six recordings, all solo piano, two of them by one performer.
- The `benchmark` condition here is on demo recordings, not the test recordings, so its figures are not expected to equal the README's.
- Hiss is scaled from the signal's power after the low-pass and saturation, so its level can differ slightly between `benchmark`, `no_lowpass` and `added_noise_only`. The input SI-SNR column shows how much.
- The `snr_grid` noise is drawn from the benchmark's seeds, not Phase 0's, so Phase 0's 1.90 dB is not expected to be reproduced exactly.
- The degradation clips to [-1, 1], and the noise of `added_noise_only` includes any clipping. How many samples clip was not measured.
