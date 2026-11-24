# Benchmark post-hoc analyses

This document fixes the analyses that will be reported beside the benchmark's primary results
and are not part of the frozen protocol. It was written and merged after the protocol froze
(v14, `benchmark/protocols/v14.json`) and after the systems stage of run `20261123T214738Z`
wrote its score files, but before any score in those files was read or summarised. The
commit that adds this file is the timestamp for that claim.

> **Note:** Everything here is secondary to the protocol. The primary results, their
> performer-cluster bootstrap intervals and the publication gates are exactly as v14
> defines them, and nothing below changes them. Every analysis listed here is reported for
> every stratum the report publishes as a comparison, whatever it shows. A stratum the
> report withholds or publishes as a reliability finding gets no post-hoc numbers either.

## Why these analyses exist

A methodology review, made before any score was read, raised three points that the
protocol's outputs do not answer on their own:

1. Each stratum has few performer clusters in its test partition (11 for solo piano and 14
   for chamber, counted from the walk manifest). A percentile bootstrap over that few
   clusters may give intervals narrower than their nominal 95%.
2. The degradation includes a low-pass filter with its cutoff drawn per frame between 4 and
   8 kHz (`degradation.ranges.rolloff_hz`). Log-spectral distance over all bins mixes
   reconstruction of the band every frame keeps with the band some frames lose.
3. A capped or silent output would be scored without being visible in a mean.

## Rules

- **Labelling:** every number from this document is labelled "post-hoc, added after the
  protocol froze, before any score was read".
- **No significance language:** no p-value, no hypothesis test, no "significant", and no
  overall winner, as the protocol's reporting rules require of the primary results.
- **Sign convention:** each paired difference is oriented so that a positive value means
  OpGAN scored better on that metric. For reconstruction SNR and SI-SNR that is OpGAN minus
  UVR; for log-spectral distance, where lower is better, it is UVR minus OpGAN.
- **Inputs:** the same per-frame scores, test admissions, clusters and conditioned output
  pairs the report uses. Nothing is re-run through either system.

## Analysis 1: cluster-level paired summary

For each stratum and each primary metric (reconstruction SNR, SI-SNR, log-spectral
distance):

1. **Per recording:** the paired difference of the two systems' recording means, each
   recording mean being the arithmetic mean over that recording's eligible frames, as the
   protocol defines it.
2. **Per performer cluster:** the unweighted arithmetic mean of the paired differences of
   that cluster's test recordings. This gives one number per cluster, k numbers in all.
3. **Reported:**
   - k, and the mean of the k cluster values;
   - a 95% t-interval, mean plus or minus t(0.975, k - 1) times s / sqrt(k), where s is the
     sample standard deviation of the k values with k - 1 in the denominator;
   - the number of clusters with a positive, negative, and exactly zero value.

## Analysis 2: log-spectral distance below 4 kHz

For each stratum, each system and the degraded input baseline: log-spectral distance
computed exactly as v14 defines it (`metrics.log_spectral_distance`: n_fft 1024, hop 256,
periodic Hann window, centred framing with zero padding, `20*log10(|X| + 1e-08)`, RMS over
bins, mean over STFT frames), except that the bins are restricted to indices 0 to 256
inclusive. At 16 kHz with n_fft 1024 the bin spacing is 15.625 Hz, so bin 256 is 4000 Hz,
the lowest cutoff the degradation draws. It is computed on the same conditioned pair as the
primary metric. Reported as the primary metrics are: per-stratum mean and median over test
recordings for each system and the baseline, the paired difference, and Analysis 1's
cluster-level summary for that difference.

## Analysis 3: capped and silent outputs

For each stratum and each system, counted over eligible test frames:

- the number of frames at the 60 dB cap, separately for reconstruction SNR and SI-SNR;
- the number of frames whose conditioned output is exactly zero in every sample.

The same counts are reported for the degraded input baseline.

## Not done here

Results broken down by degradation component, a held-out noise family, UVR run at its
native 44.1 kHz rate, further metrics (multi-resolution STFT distance, Frechet Audio
Distance), an oracle ceiling, and a listening test are left to a separate study, to be
specified and externally timestamped before any of its runs.

## Note added 2026-11-24

This document was merged in PR #33 at 00:35:18 UTC on 2026-11-24 (merge commit c40aaa6). The report's `report.json` was last modified at 00:35:51 UTC, about 33 seconds later. The merge therefore precedes the report's files, which orders this plan before the report ran. It cannot show that no score was viewed before the merge; that rests on the author's account.

The sentence at line 25, "the band every frame keeps", is inexact. The degradation's lowest cutoff is 4 kHz, and its 4th-order Butterworth low-pass attenuates the top of the band in frames whose cutoff is near 4 kHz, by about 3 dB at 4 kHz for a 4 kHz cutoff (measured with `scipy.signal.butter(4, 4000, fs=16000)` and `freqz`). The band is therefore not kept unchanged in every frame.

The label quoted under Rules, "post-hoc, added after the protocol froze, before any score was read", was changed in the code on 2026-11-24 to "post-hoc, fixed in docs/benchmark-posthoc-analyses.md before the report ran", because the first form cannot be shown from the repository.
