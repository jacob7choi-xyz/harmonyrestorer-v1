# v1 on clean, full-band input

This document fixes, before anything runs, how v1's effect on clean, full-band audio is measured and checked by ear, and what result leads to a proposal for the live site. The code is `benchmark/measurements/clean_input.py` (the measurement) and `benchmark/measurements/clean_input_listen.py` (the listening check).

> **Note:** No v1 or UVR output on clean input from these recordings had been computed by this project's measurements, and no listening clip had been built, when this document was written on 2026-11-26. The merge of this document is the timestamp for the rule below.

## Why

In the gain-steps measurement (`docs/model-v2-gain-steps.md`), v1's mean SI-SNR improvement was 6.04 dB when its input had passed through the benchmark's low-pass and -4.85 dB when it had not, and it was negative without the low-pass on all six demo recordings. v1's training noise applied that low-pass to every pair (`benchmark/degradation.py`). The live site runs v1 on every upload, and an upload that is full-band at 16 kHz, such as a later tape or LP transfer or a modern recording, did not pass through that filter. Whether v1's change to such audio is audible was not checked. This check asks that question for the site, starting from the simplest case: clean audio with nothing added.

## Data

The clean audio is the benchmark's `demo` partition: 6 CC0 solo piano recordings of the 2026-11-23 walk (`data/benchmark_runs/20261123T214738Z`), at 16 kHz mono and at training level (the recording scaled so that its peak is the degradation's headroom, as the systems stage does). Nothing is added and no low-pass is applied. No test recording is used, and nothing is downloaded.

## Part A: measurement

Part A describes the change and decides nothing on its own.

- The frames are Phase 0's: up to 40 eligible 2-second frames per recording at evenly spaced positions (`real_noise_gap.frame_indices`).
- v1 (the shipped checkpoint) and UVR-DeNoise each restore every frame through the benchmark's in-process engines. UVR-DeNoise is the backend's other engine (`DENOISER_ENGINE=uvr`); here it runs as the benchmark ran it, one 2-second 16 kHz mono frame at a time (see Limits).
- Each output is scored against the frame it was given, with the benchmark's `score_output`: SI-SNR, the shift-tolerant SI-SNR (maximum shift 1 sample), and the log-spectral distance.
- Per system, the mean over each recording's frames is reported, then the mean over recordings with the recording bootstrap's 95% interval, and the same for v1's value minus UVR's on each recording.

## Part B: blind listening check

### Clips

- From each recording, two 8-second excerpts (four consecutive frames) are cut. A candidate is a frame whose next four frames are all eligible, and the excerpts start at the candidates one third and two thirds of the way through the list. The positions are fixed by this rule, not chosen by listening.
- v1 is given each excerpt with one extra frame (2 seconds) of the recording on each side where the recording has one, so that the excerpt's own edges are not the input's edges, as they are not inside a whole upload. The middle 8 seconds of v1's output are kept, and the same 8 seconds of the input.
- Each excerpt is presented in two conditions: `original`, the excerpt untouched, and `opgan`, v1's output on it. Both are at 16 kHz mono, the rate at which the site returns restored audio, trimmed to a common length.
- The two conditions of an excerpt are loudness matched with the listening protocol's settings, read through `load_listening_protocol` (study 2's protocol v2 when this was written): ITU-R BS.1770 integrated loudness, a target of -23 LKFS, a sample-peak ceiling of -1 dBFS, and a lowering step of 0.1 LU. They are written as 16-bit PCM WAV.

### Trials

- 12 scored trials, one per excerpt, each with one condition on side A and the other on side B.
- 2 catch trials, each playing one condition of one excerpt on both sides; the two catch excerpts differ.
- The sides, the catch picks and the trial order are drawn from a seed taken from the operating system. The seed is written only to the sealed key, in a directory apart from the clips, which is not opened until the answers are recorded.
- The files are named `trial_01_A.wav` to `trial_14_B.wav`, so the names carry the trial and the side only.

### Listening

The listener is the author. For each trial, the listener plays both sides as often as wanted, on headphones at one volume setting throughout, and answers A, B or N (no preference). The answers are sent in chat and written to a file before the key is opened. Part A runs before the clips are built, and its figures are not shown to the listener until the answers are recorded.

## The rule

Over the 12 scored trials, the answers are mapped through the key to the condition chosen.

- **If `original` is preferred in at least 10 of the 12 scored trials,** v1's change to clean full-band audio is taken as audible and unwanted for the site. Options for the live site are then proposed, each with its own plan and approval: switching the site's default engine to UVR, a note on the site, or keeping v1 until model v2. Switching to UVR is considered only if, in Part A, the 95% interval of v1's SI-SNR minus UVR's lies below zero and the 95% interval of v1's log-spectral distance minus UVR's lies above zero, and even then only after a separate check of UVR run as the site would run it.
- **Otherwise,** no change to the site is proposed from this check, and model v2's work goes ahead as planned.

The catch answers are reported beside the result and do not change it.

> **Note:** The threshold is a judgement made before the check. For a listener choosing A or B at random with no N answers, at least 10 of 12 choices fall on one named condition with probability 79/4096, about 1.9%; answers of N lower it further. That figure describes chance agreement only; it is not a test of listeners in general.

## Limits

- One listener, who built v1 and knows what the check is looking for. The sides are blind; the purpose is not.
- Six recordings, all solo piano, and two excerpts from each.
- `original` is the excerpt at 16 kHz, not a recording at its source rate. The check compares v1's output with what the site would return if v1 changed nothing.
- Clean input is the simplest case. Real uploads also carry noise, and real 78 rpm transfers are band-limited; study 2's unscored dry run on real 78 transfers is the only listening on those so far.
- Part A's metrics are signal-level; they do not say what is audible.
- Part A's UVR runs one 2-second frame at a time at 16 kHz mono, which the README notes is likely outside the setting UVR was designed for; the site would run it on whole uploads. The Dockerfile does not copy a UVR model file into the image (read on 2026-11-26), so whether the live function can run UVR was not checked.
