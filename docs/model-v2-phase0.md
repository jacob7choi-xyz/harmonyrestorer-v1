# Model v2, Phase 0: does a measurement show v1's gap on real 78 noise?

This document fixes, before the measurement runs, how Phase 0 is measured and what result lets model v2's training work start. The code is `benchmark/measurements/real_noise_gap.py`, and the disc split it reads is `benchmark/manifests/aalto_disc_split.json`.

> **Note:** No v1 output on any real-noise mixture had been computed when this document was written on 2026-11-25. The merge of this document is the timestamp for the rule below.

## Why

v1 was trained on synthetic noise only (`dataset/noise.py`; the copy that made the training pairs is `benchmark/degradation.py`). In a dry run of the listening study, the author heard no clear difference between conditions on real 78 rpm transfers. That impression comes from one listener who built v1, in a session that was not scored, so it is not evidence. Phase 0 asks a narrower question first: on clean recordings with noise added, is v1 clearly worse on real 78 surface noise than on its own kind of synthetic noise? If not, this measurement cannot see the problem, and model v2's training work does not start on its basis.

## Data

| Part | Source | Use |
|---|---|---|
| Clean audio | The benchmark's `demo` partition: 6 CC0 solo piano recordings of the 2026-11-23 walk (`data/benchmark_runs/20261123T214738Z`), which protocol v14 says may be auditioned and curated freely | Clean reference |
| `great78` noise | The Aalto gramophone noise set (Moliner and Valimaki, ICASSP 2022), noise-only segments cut from Great 78 Project transfers; zip SHA-256 `b97a78a4da7ed7e05a7b10db31cbcab37ffe927d14e1c1c6948f55da0b720a2c`, `info.csv` SHA-256 `9ba1848b790f2ec12cfcbbc09d6ca387c5049ab643bfed7d92e6de208b086c20` | Only the 174 evaluation discs of the frozen split |
| `freesound` noise | Three CC0 78 rpm surface-noise recordings from Freesound (IDs 341761, 244676 and 481811; 370.59 s in all), each licence read on its own page on 2026-11-25 | Evaluation only, never training |

> **Note:** The Great 78 Project offers its collection "for use in research, teaching, and private study only". The Aalto set's noise is used here for evaluation, and later for a research-only training arm whose model is never shipped.

### The disc split

The Aalto set's own splits were made by recording side, and 21 of its 48 test discs have their other side in its training split. Phase 0 therefore uses its own split by disc. A Great 78 identifier carries the disc as a seven-digit `gbia` number and the side as a trailing letter; for example, the study item `gbia0520150b` and the item `gbia0520150a` are the two sides of one Victrola disc from 1915, matrix 6332B and 6332A.

- Every disc in the listening study's universe listing (all 297 items, kept or excluded) is removed first. One Aalto recording is on such a disc.
- The remaining 1,161 discs are permuted with seed 20261125: 15% (174 discs) are evaluation noise, 5% (58) are validation noise, and the rest (929) are training noise for the research-only arm.
- Three segment folders whose names match no `info.csv` row, and three `info.csv` rows with no folder, are left out.
- The manifest's SHA-256 is pinned in `real_noise_split.py`, and `load_split` refuses any other manifest. Evaluation discs can never become training noise.

## Conditions

Each demo recording is cut into the benchmark's 2-second frames at 16 kHz, and its eligible frames are found with the benchmark's eligibility rules. Up to 40 eligible frames per recording are taken at evenly spaced positions. Each frame is brought to training level as the systems stage does (`degrade_with_reference`), and three noises are added to it:

1. `synthetic`: v1's training noise from `benchmark.degradation`, with parameters drawn from the protocol's training ranges. The low-pass is set to the Nyquist rate and saturation is switched off, so this condition, like the other two, is added noise only. The noise is the degraded frame minus the frame.
2. `great78`: a random evaluation disc, and from it a random 2-second window of that disc's segments, joined in a random order with 50 ms linear crossfades where one segment is too short.
3. `freesound`: a random 2-second window of the three CC0 recordings, joined the same way.

Noise is read as float64, averaged to mono, resampled to 16 kHz with `librosa.resample` (its default method), and its mean is removed. Each noise is then scaled so that the clean frame's power over the noise's power is 5, 10, 15 and 20 dB, giving four mixtures per frame and condition. One noise excerpt is drawn per frame and condition from a generator seeded with 20261126, the recording's canonical digest, the frame index and the condition, so the four SNRs differ only in scale.

## What is measured

v1 (the shipped checkpoint, through the benchmark's in-process engine) restores each mixture. The input and the output are scored against the clean frame with the benchmark's `score_output`, which applies the benchmark's conditioning. Per recording and condition, the measurement reports the mean over frames and SNRs of:

- the SI-SNR improvement: SI-SNR of v1's output minus SI-SNR of the mixture, in dB;
- the log-spectral distance of v1's output.

SI-SNR is the benchmark's scale-invariant metric (`benchmark/metrics.py`).

## The gate

For each recording, the difference is the mean SI-SNR improvement on `synthetic` minus the mean SI-SNR improvement on `great78`. The gate passes when both hold:

1. the mean of the six differences is at least 3 dB;
2. the 95% bootstrap interval of that mean lies above zero. The bootstrap resamples the six recordings with replacement, with the benchmark protocol's seed and iteration count, and the interval runs from the 2.5th to the 97.5th percentile of the resampled means.

`freesound` is reported beside `great78` and does not decide the gate, because it comes from three recordings.

> **Note:** The 3 dB margin is a judgement made before the run, not a derived threshold. It asks for a difference larger than small variations between noise sets, and the result is reported with its interval whichever side of the margin it falls on.

If the gate passes, Phase 1 (freezing the training experiment) is planned. If it fails, training work does not start from this result, and the next step is decided from what the measurement shows.

## Limits

- Six recordings, all solo piano, two of them by one performer; the interval will be wide.
- The mixtures put old surface noise over modern recordings. They lack the recording character of acoustic-era transfers, so a pass is necessary for model v2 but not sufficient; the listening study remains the final judge.
- The `great78` noise and v1's future research-only training noise share one transfer chain (George Blood's). The `freesound` noise is the only real noise here from other chains.
- The `synthetic` condition clips the degraded frame to [-1, 1] as v1's process did. How many samples clip was not measured.
