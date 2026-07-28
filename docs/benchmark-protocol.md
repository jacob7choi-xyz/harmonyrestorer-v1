# Benchmark protocol v1

Frozen before candidate identities are selected, downloaded for evaluation,
auditioned, or scored. The source collection has already been inspected in general
terms; what has not happened is any decision about which recordings enter the
benchmark.

Its purpose is to remove the freedom to make reasonable-sounding choices after
seeing results. Where a decision could be influenced by an outcome, it is fixed
here instead.

The previous benchmark is withdrawn: its figures were computed over a population
including training material, because the split was drawn over augmented pairs
rather than over source recordings. See [benchmarks.md](benchmarks.md). The record
is kept rather than deleted.

## What this benchmark answers

Restoration quality on **recordings the model has never heard**, under the frozen
degradation protocol below.

| Stratum | Question | Status |
|---|---|---|
| A | Unseen recordings, controlled synthetic degradation | this protocol |
| B | Recordings from other collections | not run |
| C | Real archival damage, no clean reference | not run |

Results may be stated as performance on unseen recordings from this source
collection under this degradation protocol. They may not be stated as performance
on historical or archival recordings.

Membership in one collection does not imply homogeneous performers, venues,
microphones, eras, or mastering. No such claim is made.

## Identity and disjointness

Training corpus frozen at `benchmark/manifests/training_corpus.json`: 145
recordings. The manifest carries full SHA-256 values; the corpus digest is the
SHA-256 of the newline-joined per-file digests in sorted filename order.

**Automatic exclusion**, no discretion:

1. Source item and file identity within the collection
2. Raw file SHA-256
3. Decoded PCM SHA-256, which catches re-encodes of the same master

**Flag for adjudication**, never automatic exclusion: aligned MFCC sequence
similarity at or above **0.50**.

Method, frozen: librosa 0.11.0 `feature.mfcc`, `n_mfcc=20`, `hop_length=512`,
mono at 16 kHz, per-coefficient z-normalisation; alignment search over +/-5 s at a
stride of 2 frames; similarity is the best mean per-frame cosine over the
overlapping region, requiring at least 100 overlapping frames.

The threshold is calibrated, not chosen. Using training recordings only, so that
benchmark candidates can never become the detector's tuning set:

| Set | n | Result |
|---|---|---|
| Positives: same performance, trimmed 3 s / gain -6 dB / resampled through 8 kHz | 15 | min 0.8471 |
| Negatives: different recordings | 28 | max 0.0995 |
| Negatives, same composer | 6 | max 0.0808 |
| Negatives, same session: consecutive movements, one performer, one room, one mic | 21 | max 0.1033 |

0.50 sits nearly five times above the highest observed negative and well below the
lowest positive. The same-session set is the hardest false-positive case available:
identical performer, venue, microphone, and mastering, differing only in the music.
It separates cleanly, which indicates the detector responds to performance identity
rather than to recording conditions or timbre.

**Known calibration gap.** The most adversarial case, the same composition played
by a different performer, could not be tested: the training corpus contains one
performance per work. The same-session negatives do not substitute for it: they test
recording-condition similarity with different note sequences, whereas the untested
case shares the note sequence and differs in performance. Those are different
failure axes. The mechanism argues it should still separate, since the metric
requires frame-level alignment and differing tempo and rubato destroy
correspondence, but that is reasoning rather than measurement. It is a further
reason the detector only flags for adjudication instead of excluding
automatically.

Chroma similarity was considered and rejected. Chroma encodes pitch-class content,
so two performances of the same composition are near-identical by construction; it
is the standard feature for cover-song detection, which is the opposite of the
question here. Using it would have excluded legitimate alternate performances and
silently distorted the test distribution.

Adjudication answers exactly one question: same underlying performance, yes or no.
It is performed without reference to any model output, and every decision is logged
with its reason in the run manifest.

Recorded per candidate: source identifier, filename, composer, work, performer
where available, byte size, both hashes, maximum aligned-MFCC similarity against
the training corpus, and any adjudication.

## Selection

Candidates are not downloaded, inspected, and then filtered. Filtering after
inspection is how a benchmark becomes curated. The admission algorithm has no
discretionary branch.

1. Enumerate the complete candidate universe from collection metadata
2. Order it deterministically by source identifier, then shuffle with seed `20260726`
3. Walk that order strictly. For each candidate: fetch, apply the frozen technical
   and disjointness predicates, admit or reject
4. Rejected candidates are skipped and the walk continues. There is no manual
   substitution and no reordering

Acquisition failures do not determine membership. A transient failure (network,
timeout, 5xx) retries the same candidate three times with exponential backoff and
does not advance the walk. Only a permanent source condition (404, 410, or a file
that cannot be decoded) excludes a candidate, and it is logged with its cause.

The candidate universe is enumerated once and frozen to
`benchmark/manifests/candidate_universe.json` with its own digest, recording the
collection identifier, query parameters, enumeration timestamp, and every candidate
identity. All assignment works from that frozen manifest, never a live API, because
the source collection can gain or lose files and would otherwise silently change
the test set.

A full selection audit trail is recorded for every candidate examined: position in
the walk, source identifier, admitted or rejected, the predicate responsible for a
rejection, and the assigned partition. Anyone can then ask why a given recording is
absent without trusting the script's summary.

Partitions are assigned by position in the admitted sequence, before any audio is
auditioned:

| Partition | N | Use |
|---|---|---|
| test | 30 | Frozen. Never used for any system decision. |
| demo | 6 | Public samples. May be auditioned and curated freely. |
| test_reserve | 6 | Replacement test material only, consumed in order. |
| development_reserve | 6 | Future training and validation material. |

Reserve is split because one pool cannot serve both roles: a track used for
development is contaminated and can never replace a pristine test track. Splitting
them now removes the future question of which unused track to promote, which would
reintroduce selection freedom at the worst moment.

N is fixed here and is not adjusted because intervals are wide or results are
unfavourable. Thirty recordings is a real sample; it will produce wide intervals
and outlier recordings will matter. Both are acceptable and neither justifies
expanding N afterwards.

## Governance

Contamination is **test information reaching a system decision**, through any
channel. Listening to test outputs, reading per-track scores, or looking at
aggregates and then changing the system all burn the partition equally. Burned
means a replacement is drawn from reserve.

Reserve tracks are clean only until used. Once any reserve identity informs a
training or validation decision it becomes development data permanently and can
never serve as test material.

If a future model is changed in response to today's test results, today's numbers
remain publishable as history, but they are no longer a pristine test of that
model. The new model needs a fresh partition.

Test identities may never migrate into a training set. The split manifests are the
authority.

## Eligibility

Determined **before either system runs**, from source audio and ground truth only,
never from model output. The withdrawn benchmark decided eligibility at scoring
time, so each system's denominator was shaped by its own failures.

**Benchmark eligibility and metric applicability are separate concerns.** A metric
must never shrink the population the other metrics use, or the benchmark silently
becomes "performance on the material that metric happens to accept."

**Benchmark eligibility** is domain-neutral. A recording is eligible if it decodes
cleanly, is at least 60 s long, has full-file RMS above -50 dBFS, and passes
disjointness. A frame is eligible if its clean reference has RMS above -45 dBFS.

That floor exists to reject degenerate frames where a reconstruction ratio is
numerically meaningless: digital silence, zero padding, decode artifacts. It is not
intended to remove quiet music. A pianissimo passage is legitimate classical
content and difficulty is not a reason for exclusion. Across 4,000 sampled clean
frames from the *training* corpus the distribution bottoms out near -39 dBFS at the
0.5th percentile, so the floor excluded nothing there; its effect on the frozen
candidate population was unknown when it was chosen, which is the point of choosing
it beforehand.

**Metric applicability** is then determined per metric, from clean reference
material only, never from model output. The ~15,000 frames dropped by the withdrawn
benchmark were not quiet by RMS at all; PESQ's internal voice-activity check
refused to score them. So PESQ is reported over the subset where
`PESQ(clean, clean)` computes successfully, fixed before either system runs and
identical for both. Because PESQ's acceptance of musical content is unlikely to be
random with respect to texture or dynamics, that subset is reported as its own
population and never used to define the benchmark.

The primary metrics apply to every eligible frame.

Populations tracked and reported separately: eligible, per-metric applicable,
submitted, scored.

## Degradation

Frames are cut only after partition assignment, so every derived artifact inherits
its recording's partition and siblings cannot cross the boundary.

Degradation is procedurally synthesized, not sampled: `tape_hiss`,
`vinyl_crackle`, and `mains_hum` generate from a seeded `np.random.Generator` with
no asset library. There is therefore no noise-source pool to be disjoint from, and
a benchmark seed distinct from the training seed produces genuinely fresh
realizations of the same generative process rather than replays.

Frozen for the run: degradation implementation version, types, SNR distribution,
framing and overlap, resampling behaviour, channel handling, sample rate, and
seeds. Every generated sample carries `source_track_id`, `partition`,
`frame_index`, `degradation_id`, and `seed`.

## Metric conditioning

Frozen before results exist, because "alignment fix" is otherwise indistinguishable
from shifting outputs until the metric improves.

- Metrics computed at 16 kHz mono. Resampling uses librosa 0.11.0 `resample` with
  its default `res_type`. Metric libraries frozen: `pesq`, `pystoi` 0.4.1, numpy
  2.3.5, scipy 1.18.0, soundfile 0.14.0. Exact versions are recorded per run
- Clean reference and system output are compared at equal length. Where they differ
  by at most **160 samples (10 ms at 16 kHz)**, the longer is front-aligned and
  truncated to the shorter, and the discarded duration is recorded per item
- A difference above 160 samples makes that item invalid for that system rather
  than truncating silently, and feeds the coverage rule below as a system failure

The 160-sample bound has provenance rather than being a round number. OpGAN is
exactly length-preserving: its chunking reconstructs the input sample count, and a
182 s file was verified to return 2,913,083 samples against 2,913,083 expected. UVR
is not: it resamples 16 kHz to 44.1 kHz and back, and polyphase resampling does not
round-trip, previously observed as a 128-sample deficit on a 32,000-sample frame, which is the
same 2-second 16 kHz unit this benchmark scores.
160 covers that observed mechanism with margin and is far too small for a system to
escape being scored on a missing tail.
- **No cross-correlation alignment search is performed.** Outputs are compared at
  the sample offset they arrive with. A latency-compensation policy may be added in
  a future protocol version if a system is shown to introduce fixed latency, but it
  will be declared before that version is run, not fitted afterwards

## Metrics

**Benchmark v1 measures reference-signal fidelity, not perceived audio quality.**
It reports how closely a restoration matches a known clean reference. It does not
report which restoration a listener would prefer. That distinction is stated here
because the two are routinely conflated and only one of them is being measured.

The withdrawn benchmark reported SDR, PESQ, and STOI. Two of those are speech
metrics, inherited because denoising research lives largely in the
speech-enhancement literature. This benchmark restores classical instrumental
music.

- **PESQ** is ITU-T P.862, perceptual evaluation of *speech* quality, built for
  narrowband telephone networks and speech codecs, withdrawn by the ITU in 2024 in
  favour of the P.863 family.
- **STOI** is Short-Time Objective *Intelligibility*, predicting how intelligible
  processed speech is. There are no words in a Bach partita whose intelligibility
  could be recovered.

Neither is primary here. Fixing leakage and denominators does not rescue a metric
whose construct does not match the domain.

### Primary: signal fidelity

| Metric | Measures | Better | Blind to |
|---|---|---|---|
| Reconstruction SNR | sample-wise fidelity | higher | nothing; penalises gain, timing, distortion |
| SI-SNR | fidelity with global gain projected out | higher | level mismatch |
| Log-spectral distance | spectral magnitude fidelity | lower | phase and fine time structure |

These are three views of one waveform comparison, not three independent
experiments. They are strongly correlated by construction and are reported together
because each is blind to something the others catch: SI-SNR forgives a system that
outputs `0.3 x clean`, which reconstruction SNR punishes correctly; LSD tolerates
phase error that both SNRs punish. Agreement between them is not corroboration.

**Reconstruction SNR** is the quantity the withdrawn benchmark called "SDR". It is
not BSS-eval SDR: no projection or filtering is permitted, so a one-sample shift
degrades it substantially, which is why the no-alignment-search rule is
load-bearing. Renamed because keeping an overloaded name and attaching a warning is
worse than naming the estimand accurately.

```
reconstruction_snr = 10 * log10( sum(clean^2) / sum((clean - restored)^2) )
    residual_floor = 1e-10        # below this, value caps
    cap_db         = 60.0
```

**SI-SNR**, frozen in full because "standard SI-SNR" is not one thing:

```
s      = clean    - mean(clean)          # zero-mean both
s_hat  = restored - mean(restored)
target = (dot(s_hat, s) / (dot(s, s) + eps)) * s
noise  = s_hat - target
si_snr = 10 * log10( (sum(target^2) + eps) / (sum(noise^2) + eps) )
    eps    = 1e-8
    cap_db = 60.0
```

**Log-spectral distance**, every STFT and floor parameter fixed because the
magnitude floor alone materially changes the result on quiet bins:

```
lsd = mean over frames of
        sqrt( mean over bins of ( 20*log10(|X|+floor) - 20*log10(|Y|+floor) )^2 )
    sample_rate     = 16000
    n_fft           = 1024
    win_length      = 1024
    hop_length      = 256
    window          = hann
    center          = True
    pad_mode        = constant
    magnitude_floor = 1e-8
    bins            = all, including DC and Nyquist
```

All three operate on the 2-second evaluation unit at 16 kHz mono. RMS, where used
for eligibility, is `sqrt(mean(x^2))` on float samples normalised to full scale
1.0, expressed as `20*log10(rms)` dBFS over the whole frame.

### Effect direction

The paired effect is defined per metric so that **a positive delta always favours
OpGAN**, which prevents a sign error from inverting a published conclusion:

| Metric | Paired effect |
|---|---|
| Reconstruction SNR | `OpGAN - UVR` |
| SI-SNR | `OpGAN - UVR` |
| Log-spectral distance | `UVR - OpGAN` |

### Legacy diagnostics

PESQ and STOI are computed and recorded in the machine-readable artifact for
comparability with the withdrawn figures. They are **not** included in the headline
README table, because a reader who sees `PESQ 4.1` will read it as an audio-quality
claim regardless of the disclaimer beside it. Preventing foreseeable
misinterpretation is part of reporting honestly, not just computing correctly.

### Not run

ViSQOLAudio is the scientifically stronger perceptual candidate, having been
evaluated against listener judgements on music rather than speech. It is deferred
from v1 for engineering cost: it is a native library requiring a source build, and
adopting it means pinning a commit and build environment and validating its
configuration for restoration noise rather than the codec impairments its published
evaluation emphasised. That is a real cost, not an impossibility, and a pinned
container would resolve it.

**Benchmark v1 therefore has no validated perceptual quality metric, and makes no
perceptual claim.** The strongest such evidence would be a blinded listening test
against clean reference, degraded input, and both systems. Out of scope here; the
natural direction for a later stratum.

## Scoring

Both systems receive an identical population by identifier. The run fails closed if
the scored identifier sets are not equal to each other and to the eligible set.
Comparing whichever items both systems happened to succeed on rewards failure and
is not permitted.

Two failure kinds are distinguished, because collapsing them lets a real
reliability defect hide behind reruns:

- **Harness or infrastructure failure** (crash, corruption, interrupted run,
  manifest mismatch): the execution is invalid. Fix and rerun. Does not burn the
  test partition
- **System failure** (a model deterministically cannot process an eligible item):
  this is a benchmark result, not an invalid run. Failed items are never dropped to
  manufacture a common-success population. Coverage below 100% is published as a
  reliability finding, and the paired headline quality comparison is **withheld**
  for that run, because a comparison over survivors rewards failure. Full coverage
  on both systems is a precondition for publishing the quality comparison

## Aggregation

Frames within a recording are not independent observations; reporting a frame count
as N overstates confidence by roughly the frames-per-recording factor.

1. Per-frame metrics, primary and legacy
2. Within each recording, the arithmetic mean of its eligible frames, unweighted.
   The published estimand is therefore the **mean frame-level metric for a
   recording**, which is not the same quantity as a metric computed over the whole
   recording's aggregate signal energy
3. Across recordings, the macro mean and median
4. Bootstrap over recordings, never frames: 10,000 resamples, percentile method,
   95% level, seed `20260726`. For the paired comparison, per-track differences
   are formed first and then track indices are resampled once per iteration, so
   the pairing is preserved; the two systems are never resampled independently

Because both systems see the same recordings the comparison is paired, so the
headline comparison is the per-recording difference `OpGAN − UVR` with its own
bootstrap interval, which is more informative than two independent intervals.

Reported per metric: per-track mean and median, 95% CI by recording, paired
difference CI, N recordings, N eligible frames, per-metric applicable frames, and
coverage. Legacy speech metrics are reported in a separate labelled section so they
cannot be read as headline results.

## Publication

The machine-readable artifact is authoritative; README summarises it and carries no
independently maintained numbers.

The artifact records: protocol version, run identifier, git commit, checkpoint
SHA-256, training corpus digest, split manifest digest, degradation config version
and seeds, evaluation commit, metric library versions, expected and eligible and
scored counts for tracks and frames, per-system failures, per-track metrics,
aggregates with intervals, bootstrap method and seed, timestamp, and runtime
environment.

## Amendment

This document is not edited silently. A protocol defect discovered before results
exist produces v2, with the reason recorded, before evaluation runs. A defect
discovered after results exist invalidates those results rather than retroactively
changing the rules that produced them.
