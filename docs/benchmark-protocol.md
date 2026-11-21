# Benchmark protocol v10

Frozen before candidate identities are selected, downloaded for evaluation,
auditioned, or scored. The source collection has already been inspected in general
terms; what has not happened is any decision about which recordings enter the
benchmark.

> **Note:** v10 amends v9, which amended v8, which amended v7, which amended v6, which
> amended v5, which amended v4, which amended v3, which amended v2, which amended v1, all
> before any evaluation ran. The
> operational values live in `benchmark/protocols/v10.json`, which is normative; this
> document explains them and
> its numbers are illustrative. Superseded versions are preserved unmodified, as
> [benchmark-protocol-v1.md](benchmark-protocol-v1.md),
> [benchmark-protocol-v2.md](benchmark-protocol-v2.md),
> [benchmark-protocol-v3.md](benchmark-protocol-v3.md),
> [benchmark-protocol-v4.md](benchmark-protocol-v4.md),
> [benchmark-protocol-v5.md](benchmark-protocol-v5.md),
> [benchmark-protocol-v6.md](benchmark-protocol-v6.md),
> [benchmark-protocol-v7.md](benchmark-protocol-v7.md),
> [benchmark-protocol-v8.md](benchmark-protocol-v8.md), and
> [benchmark-protocol-v9.md](benchmark-protocol-v9.md) with their configs, so an
> artifact recording an older `protocol_version` stays interpretable. See
> [Amendment](#amendment) for what changed and why.

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

1. Source item and file identity within the collection. A candidate's source
   identifier is `MusopenCollectionAsFlac/<path in the item>`. Training files are named
   with the item identifier, two underscores, and the path with each slash replaced by
   an underscore (`dataset/acquire.py`), so a candidate is compared in that form
2. Raw file SHA-256
3. Decoded PCM SHA-256, which catches lossless re-encodes of the same master. The hash
   covers the samples as the pinned decoder returns them, float64 and interleaved, as
   contiguous little-endian bytes with no header, before downmix and resampling

From v9, rules 2 and 3 also compare each candidate with every candidate admitted earlier
in the walk, so two copies of one recording cannot both be admitted. A match against
training is recorded under the training code.

**Flag for adjudication**, never automatic exclusion: aligned MFCC sequence
similarity at or above **0.30** (0.50 until v9), against the training corpus and then
against every candidate admitted earlier in the walk.

Method, frozen: librosa 0.11.0 `feature.mfcc`, `n_mfcc=20`, `hop_length=512`, computed
on the canonical waveform, mono at 16 kHz, per-coefficient z-normalisation with the
population standard deviation; alignment search over +/-5 s at a stride of 2 frames;
similarity is the best mean per-frame cosine over the overlapping region, requiring at
least 100 overlapping frames. From v9, every other argument the features depend on is
written in `v9.json` as a value, including the ones that were librosa's defaults when the
screen was calibrated (`n_fft` 2048, Hann window, centred frames with constant padding,
128 mel bands from 0 to 8000 Hz with Slaney normalisation and a float32 filter bank,
decibels with reference 1.0, floor 1e-10 and an 80 dB range, an orthonormal type 2 DCT,
and no lifter), so a library upgrade that changes a default cannot change the screen. The
offsets run over every multiple of the stride up to floor(5 * 16000 / 512) = 156 frames,
about 4.99 s. A coefficient with zero variance over a recording, a frame vector with zero
norm, or a comparison with too little overlap makes the screen's answer uncertain, which
excludes the candidate.

The threshold is calibrated, not chosen, on training recordings only, so that benchmark
candidates can never become the detector's tuning set. v9 recalibrated it on all 145
training recordings, because the sets v1 reported, and the code that scored them, are not
in the repository:

| Set | n | Result |
|---|---|---|
| Positives: each recording against itself trimmed 3 s / gain -6 dB / resampled through 8 kHz | 435 | min 0.7550 |
| Same-session takes: two files named as take 1 and take 2 of one movement | 1 | 0.4832 |
| Negatives: every other pair of different recordings | 10,439 | max 0.1435 |

The take pair is the third movement of Brahms's Symphony No. 2, and its files end
`take1` and `take2`. It was identified from the file names only and was not listened to.
Over all 10,440 pairs including it, the median score was 0.0239 and the 99.9th percentile
0.1112. v1's figures were a positives minimum of 0.8471 (n=15), a negatives maximum of
0.0995 (n=28), and a same-session maximum of 0.1033 (n=21, consecutive movements by one
performer); they are kept in the archived documents.

0.30 sits above every negative and below every positive and the one take pair observed,
so that pair would be flagged; how other takes would score was not measured. Under v8's
0.50, the take pair would have passed unflagged, and the margin above the highest negative
was a ratio of about 1.03 rather than the "nearly five times" v1 reported.

**Known calibration gap.** The same composition played by a different performer is
still untested: no pair of training recordings was identified as one work by two
performers. The mechanism argues it should separate, since the metric requires
frame-level alignment and differing tempo and rubato destroy correspondence, but that is
reasoning rather than measurement. v9 decides the adjudication answer for it in advance
(below), so a flag on such a pair does not need a decision made after candidates are seen.

Chroma similarity was considered and rejected. Chroma encodes pitch-class content,
so two performances of the same composition are near-identical by construction; it
is the standard feature for cover-song detection, which is the opposite of the
question here. Using it would have excluded legitimate alternate performances and
silently distorted the test distribution.

Adjudication answers exactly one question: same underlying performance, with the
answer yes, no, or uncertain. An uncertain answer excludes the candidate. It is
performed without reference to any model output, and every decision is logged with its
reason in the run manifest. v9 fixes the answers for the two cases the screen is most
likely to raise:

- **Another take of the same work from the same session** as a training recording, or as
  a candidate admitted earlier, is the same underlying performance, and the candidate is
  excluded. File names or Internet Archive metadata naming the same work and session decide
  it, and a flag that cannot be ruled out excludes the candidate.
- **The same composition played by a different performer** is a different performance,
  and the candidate is admitted. The benchmark measures unseen recordings, not unseen
  pieces.

From v10, a recorded adjudication applies to a flagged candidate only when it names the
candidate by its source identifier, raw digest, and decoded-PCM digest, names every
reference recording the screen flagged it against by the same three values (a training
recording by its manifest file name), and answers for each of those references. Without
such a record the walk halts, awaiting adjudication, and nothing is published; the answers
are recorded with their evidence and the walk is run again from the start. A record that
names any recording with a different identifier or digest does not apply, so an answer is
never reused for bytes it was not given for. The candidate is admitted only
when every flagged reference is answered as a different performance. Otherwise the
training corpus is considered before earlier admissions, and within the first of those
groups with another answer, a same-performance answer is recorded as that group's
`SCREEN_SAME_PERFORMANCE_*` code and an uncertain answer as `SCREEN_UNCERTAIN`, same
performance taking precedence. A screen answer that is itself uncertain, from degenerate
features or too little overlap, excludes the candidate as `SCREEN_UNCERTAIN` without
adjudication, because no answer could admit it.

Recorded per candidate: source identifier, byte size, both hashes, maximum
aligned-MFCC similarity against the training corpus and against earlier admissions, and
any adjudication with its evidence. Composer,
work, and performer are recorded where the collection metadata provides them; v6 does
not require them.

## Selection

Candidates are not downloaded, inspected, and then filtered. Filtering after
inspection is how a benchmark becomes curated. The admission algorithm has no
discretionary branch.

1. Enumerate the complete candidate universe from the metadata of the Internet Archive
   item `MusopenCollectionAsFlac`, the collection the training recordings came from. From
   v9, a listed file is a candidate when the suffix of the final component of its path,
   lowercased, is `.flac`, the test `dataset/acquire.py` applied when the training
   recordings were downloaded. The training recordings pass that test and stay in the
   universe; the first exclusion rule removes each of them before any fetch
2. Sort the source identifiers by Unicode code point, then walk them in the order
   `numpy.random.Generator(PCG64(20260726)).permutation(n)` gives, where `n` is the
   size of the frozen universe. Identifiers are compared exactly as the listing gives
   them, with no Unicode normalisation or case folding
3. Walk that order strictly. For each candidate, in this order: the first exclusion rule
   (before any fetch); fetch; the raw digest; the source-outcome stages; the decoded-PCM
   digest; the eligibility floors; the duplicate screen. The first check a candidate fails
   is the predicate responsible
4. Rejected candidates are skipped and the walk continues. There is no manual
   substitution and no reordering

Acquisition failures do not determine membership. A transient failure (network,
timeout, 5xx) retries the same candidate three times with exponential backoff and
does not advance the walk. Only a permanent source condition (404, 410, from v10 also
401, 403, or 451, or a file that cannot be decoded) excludes a candidate, and it is
logged with its cause. A 404 or
410 is recorded as `CANDIDATE_PERMANENTLY_UNAVAILABLE`; a file that cannot be decoded
carries its source-ineligible code.

From v10, a 401, 403, or 451 is also permanent and is recorded as
`CANDIDATE_PERMANENTLY_UNAVAILABLE` with its status. The run cannot fetch a file whose
access is refused, and treating the refusal as transient would make every run invalid for
as long as it lasts. A refusal can be lifted or differ between accounts, so the logged
status records why the file was excluded. Every other fetch failure is transient: a
network error, a timeout, a 5xx, any other HTTP status, and a transfer that ends short.
"Three times" means three retries after the first attempt, so at most four attempts are
made. If every attempt fails transiently, the run is invalid under
`ACQUISITION_RETRIES_EXHAUSTED` and nothing is published. Skipping the candidate instead
would let network conditions choose the test set. The wait between attempts is a harness
setting rather than a protocol value, because it cannot change which candidates are
admitted.

The candidate universe is enumerated once and frozen to
`benchmark/manifests/candidate_universe.json` with its own digest, recording the
collection identifier, query parameters, enumeration timestamp, and every candidate
identity. All assignment works from that frozen manifest, never a live API, because
the source collection can gain or lose files and would otherwise silently change
the test set.

A full selection audit trail is recorded for every candidate examined: position in
the walk, source identifier, admitted or rejected, the predicate responsible for a
rejection, and the assigned partition. Anyone can then ask why a given recording is
absent without trusting the script's summary. From v9 each disjointness rejection has its
own code: `TRAINING_SOURCE_IDENTITY`, `TRAINING_RAW_SHA256`, `ADMITTED_RAW_SHA256`,
`TRAINING_DECODED_PCM_SHA256`, `ADMITTED_DECODED_PCM_SHA256`,
`SCREEN_SAME_PERFORMANCE_TRAINING`, `SCREEN_SAME_PERFORMANCE_ADMITTED`, and
`SCREEN_UNCERTAIN`. A source-outcome or eligibility failure keeps its own code from those
sections.

Partitions are assigned by position in the admitted sequence, before any audio is
auditioned. Admissions are numbered from 1 in walk order: the first 30 are test, the next
6 demo, the next 6 test_reserve, and the next 6 development_reserve, and the walk stops at
the 48th admission. If the universe runs out first, the run is invalid and nothing is
published:

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

## Canonicalisation

Every downstream artifact is identified by the SHA-256 of its samples, so the
transformation from a source file to a canonical waveform has to be pinned end to
end. v3 named the output shape, 16 kHz mono, and left the steps producing it
implementation-defined. That is not a stylistic gap: it makes each content hash
undefined, and a hash that two faithful implementations disagree about identifies
nothing.

The measurements below are on a training FLAC and are why each item is pinned
rather than left to a reasonable default.

| Free choice | Effect on the canonical bytes |
|---|---|
| `res_type` in `soxr_hq`, `soxr_vhq`, `kaiser_best`, `kaiser_fast` | four different digests |
| downmix by channel mean or by taking the left channel | two different digests |
| downmix before resampling or after | differ, 5.2e-10 maximum absolute difference |

The order is fixed: decode, validate the source, downmix, resample, normalise
signed zero, validate the canonical result. Both orderings of the middle two are
natural readings of "16 kHz mono", so the protocol names one.

- **Decode** with soundfile 0.14.0 (libsndfile 1.2.2) to float64, `always_2d=True`,
  from a buffer the harness has already read and hashed. Always two-dimensional so
  the downmix step never branches on source shape. float64 is pinned for future
  corpora rather than for this one: the corpus is PCM_24 and a signed 24-bit integer
  is exactly representable in float32, so the two decode identically here, but a
  32-bit or float source would not
- **Downmix** in float64, stated per channel count rather than as a general mean: one
  channel passes through unchanged, two channels become `(left + right) / 2`, and any
  other count makes the source unsupported. At two channels there is no reduction
  order to choose, and `np.mean`, sum-then-divide, `np.add.reduce`, and the explicit
  form were verified byte-identical on three training recordings. Accepting arbitrary
  channel counts would instead require a layout policy this benchmark has not
  analysed, so refusing is the honest answer. All 145 training recordings are stereo
- **Resample** with librosa 0.11.0 (soxr 1.1.0) to 16 kHz, `res_type="soxr_hq"`,
  `fix=True`, `scale=False`, `axis=-1`. Every argument is named even though
  `soxr_hq` is librosa's default today, because relying on a default is exactly how
  "librosa's default `res_type`" entered the v3 prose. The output length is asserted
  against `(source_samples * 16000 + source_sample_rate - 1) // source_sample_rate`,
  in integer arithmetic so the check cannot round differently from the thing it
  checks. Verified against 200 random source lengths with no mismatch
- **Forbidden**, as an exact list matched in order rather than a set that contains
  something: integer quantisation, dither, clipping, peak normalisation, loudness
  normalisation, gain adjustment. None belong in a fidelity measurement and all of
  them change the digest. The list is load-bearing rather than declarative, as the
  amplitude rules below show
- **Signed zero** is normalised to positive. Positive and negative zero are
  numerically equal and have different bytes, so identical audio would otherwise
  produce different digests. The pipeline emits only positive zeros on the fixture
  examined, which makes this latent rather than live, and it is cheaper to remove the
  hazard than to argue it cannot fire
- **Non-finite samples** are rejected rather than repaired
- **Hash bytes** are contiguous little-endian IEEE-754 float64 samples with no
  container or header. Byte order and dtype are pinned because both change the digest
  for identical audio. Array contiguity is deliberately not pinned: numpy serialises
  in logical C order regardless of stride, so a view and a copy of equal values
  produce equal bytes

### Source domain

The transformation is defined for a bounded class of sources, frozen now rather than
after the candidate universe is acquired. Meeting an unexpected source later and
widening the contract in response would be a rule changed once its population is
visible, which is the shape this protocol exists to avoid.

Admissible sources are FLAC paired with `PCM_16`, `PCM_24`, or `PCM_32`, one or two
channels, decoding to finite samples within `[-1.0, 1.0]` inclusive at a positive
rate. Container and subtype are validated as pairs rather than as two independent
sets, so a permitted container and a permitted subtype cannot be recombined into a
pairing nobody approved.

Two separate reasons produce that set and they should not be conflated. **Determinism:**
integer PCM converts exactly into float64 under the frozen full-scale convention and
its decoded amplitude is bounded by construction, neither of which holds for float or
lossy subtypes. **Reference fidelity:** bit depths below 16 are excluded because their
quantisation noise would become part of the supposed clean reference. `PCM_8`
satisfies the first reason and is refused under the second, which is why exact
representability alone is not the rule. Admissibility is not a claim that the three
subtypes are interchangeable in quality.

Container and subtype come from the decoder's own report, never from the filename. A
file named `.wav` whose contents are FLAC reports `format=FLAC subtype=PCM_24`, so
extension-based dispatch would decode it under the wrong contract or refuse it for
the wrong reason.

The artifact is read into memory once under a size bound enforced *before* the read,
and that same buffer is hashed, inspected, and decoded. Re-reading the path between
hashing and decoding would leave a window in which the recorded digest identifies
bytes other than the ones decoded, and the recorded digest is the only thing tying a
result to an input.

The ordering matters as much as the bound. A limit checked after `read_bytes()` has
returned does not bound the allocation it names, because the buffer is already
resident. The size comes from `fstat` on the open handle, and the read is capped at
`max_source_bytes + 1` so that stale or misleading size metadata cannot defeat the
check: a buffer longer than the bound is refused rather than kept. The bounded read is
the real control; `fstat` is a cheap early rejection and a provenance datum.

Only regular files are accepted. `fstat` reports nothing meaningful about a FIFO, a
device, or a virtual filesystem object, so for a local corpus of ordinary files the
honest rule is to refuse anything else rather than reason about each exotic case.

Frames, channels, and sample rate arrive from the container and are checked against the
frozen bounds before they are multiplied into a projection and before that projection
decides an allocation. Python integers do not overflow, so the hazard is not arithmetic
but trust: absurd metadata must be refused rather than believed and sized.

Sources below the canonical rate are refused, and the floor is required to equal that
rate rather than merely to be positive. Upsampling a lower-rate source would
manufacture a clean reference with no content in the band being measured, and the
equality is also what makes the memory bound below derivable, since it is what keeps
the canonical waveform no longer than the downmix.

### Resource bounds

The decoded array, not the file, is what a source can make the process allocate. The
largest training source is 166.5 MB and decodes to 807.3 MB of float64, an expansion
of 4.8x, and one recording held as bytes, decoded array, downmix, and canonical
waveform at once reaches about 1.49 GB. Bounding only the buffer would admit a file
that then allocates several times its own size.

Both bounds are therefore stated, and the decoded bound is checked before decoding:
inspection reports frames and channels, so `frames * channels * 8` is known and can
be refused before a sample is allocated.

Headroom over the observed maxima, stated exactly rather than as a round multiple:

| Bound | Value | Observed maximum | Factor |
|---|---|---|---|
| `max_source_bytes` | 512 MiB | 166,468,316 B | 3.23x |
| `max_decoded_bytes` | 1.5 GiB | 807,269,680 B | 2.00x |
| `max_source_duration_s` | 3600 | 1051 s | 3.42x |

Bounding the pieces individually still does not bound the process. The downmix is
never larger than the decoded array, since `frames * 8` is at most
`frames * channels * 8` for one or more channels, and the canonical waveform is never
larger than the downmix, since the source-rate floor equals the target rate. The worst
set alive at once is therefore `max_source_bytes + 3 * max_decoded_bytes`, which is
5.00 GiB. The declared execution budget is 6 GiB and the loader refuses any
configuration whose own bounds could exceed it.

> **Note:** an earlier draft of v4 used `max_source_bytes + 2 * max_decoded_bytes` and
> silently assumed two channels. At one channel, which this protocol admits, the
> downmix equals the decoded array instead of halving it, so the same limits reached
> 5.00 GiB against what was then a 4 GiB budget. The corpus is entirely stereo, so the
> defect was in what the protocol admits rather than in what it currently holds, which
> is exactly the kind of gap that only appears when the bound is checked against the
> declared domain rather than against the material on hand.

For a stereo 44.1 kHz source the decoded bound binds at about 38 minutes, well below
the 3600 s duration cap, so the duration limit is a statement about the domain rather
than the operative resource limit at this format.

The per-source projection is `source_bytes + frames * channels * 8 + frames * 8 +
canonical_samples * 8`, evaluated from inspected metadata before any sample is
decoded, so a source that cannot fit is refused rather than discovered halfway
through.

The budget is per process, so it declares nothing about the machine on its own. One
canonicalisation at a time is part of the same assumption and the value is enforced at
load; raising it multiplies the requirement and is a protocol change.

> **Note:** validating the number one is not the same as running one at a time.
> Enforcing the concurrency in the runner, measuring peak resident memory under that
> exact execution path, and recovering an operating-system or container out-of-memory
> kill from the worker's exit status are obligations on the implementation. A kill of
> that kind terminates the process before any structured record can be written, so
> `except MemoryError` cannot be the only path to `ENVIRONMENT_CAPACITY_FAILURE`. v8
> defines the worker model and the exit mapping and records the peak measurement (see
> [Runner](#runner)); no runner exists yet, so none of it is enforced today.

A source that sits inside every frozen bound but that this machine cannot allocate for
is a capacity failure, not an ineligible source. It aborts under
`ENVIRONMENT_CAPACITY_FAILURE`, listed apart from `ENVIRONMENT_MISMATCH` because only
one of the two is fixed by installing something. A `MemoryError`, an out-of-memory
kill, and an undersized container are all facts about where the run happened, and
recording any of them against the recording would publish our own capacity as a
property of someone else's audio.

**The budget is a declared assumption, not a measured fact.** The 1024 MiB it leaves
above the projection is an allowance for the interpreter, NumPy, and resampler
temporaries that has not been measured. Canonicalisation is obliged to measure peak
resident memory on the largest admitted fixture and compare it against the projection.
Until then the budget is a bound the configuration is checked against, not a proof
that the projection is exact.

That check is a relationship between frozen bounds, not a per-source verdict, and the
distinction is deliberate. A source that any conforming implementation could process
must not become ineligible because this particular machine was small: capacity is ours
and is not a property of the source. An implementation that cannot fit a source inside
the budget raises a harness failure and aborts, which is visible, rather than quietly
shrinking the population.

### Amplitude

The source is bounded and the canonical waveform is not, and the asymmetry is a
decision rather than an omission.

The source bound follows from the permitted subtypes rather than being imposed on
them: integer PCM decodes into `[-1.0, 1.0]` by construction, reaching exactly -1.0
at the negative rail, which is why the bound is inclusive.

The canonical waveform carries no amplitude bound. Band-limited interpolation
overshoots: resampled to 16 kHz, a 0.999 square wave reaches 1.194538 and a
full-scale 5 kHz sine reaches 1.020793. Bounding the canonical waveform would reject
loud legitimate material, and the only ways to force a result back inside `[-1, 1]`
are clipping, peak normalisation, and gain, all three forbidden above. Canonical
validation therefore requires float64, one dimension, non-empty, and finite, and
nothing further.

Overflow of the RMS energy sum is a representation limit rather than a fact about
audio. It is unreachable here by a factor of about 1e154, and it is a fail-closed
guard in the implementation that raises a harness error, never a rule stated in the
protocol and never a recording's eligibility verdict.

### Environment

Python package versions are enforced when the protocol loads, because the lockfile
makes them deterministic. Native library versions, libsndfile and soxr, are recorded
as provenance rather than enforced, because they vary by platform.

The same check refuses to load when `LIBROSA_CACHE_DIR` is set. With that variable set
and librosa's cache level at 20 or more, a second process was observed returning a cached
resample result without calling soxr, so canonical samples would come from a pickle file
on disk rather than the pinned resampler.

Semantic validation and runtime conformance are separate responsibilities and are
separate functions. "Is this a valid protocol" and "may this machine authoritatively
execute it" are different questions, so the environment check is a public function of
its own. The loader calls it with no way to switch it off; no other code calls it
today, and no separate CI conformance job exists. `load_protocol()` is the authoritative
execution loader, not a neutral historical parser, and a parameter that could
weaken it would be the same hole as a loader that accepts a path.

**Reproducibility is claimed narrowly.** Pinning package versions does not guarantee
bit-identical native DSP output across architectures and builds. The claim is that
canonical artifacts generated by the pinned reference implementation are
authoritative and their digests identify those artifacts; other environments verify
against the frozen artifacts rather than being assumed to regenerate every bit. A
golden fixture in the test suite makes cross-platform agreement an observation rather
than an assumption.

### Golden fixtures

Two fixtures are run through inspection, decode, and canonicalisation before any
candidate is opened, and a run that cannot reproduce their frozen canonical digests
aborts with `GOLDEN_FIXTURE_FAILURE`. Each is generated in code from seeded integers
(PCG64, full signed range, the first frame at the negative rail and the second at
zero) and encoded to FLAC in memory. The digest covers the decoded canonical samples,
not the FLAC bytes, because the encoded bytes carry the libFLAC version and can differ
between builds while decoded integer PCM cannot.

| Fixture | Source | Digest asserted on | Covers |
|---|---|---|---|
| `analytic` | 16 kHz mono PCM_16, 64,777 frames | every platform | inspection, the metadata and waveform checks, decode, the one-channel downmix, signed-zero normalisation, canonical validation, the digest |
| `resampler` | 44.1 kHz stereo PCM_24, 132,423 frames | the reference platform | the same steps with the two-channel downmix, and the resampler |

At the target rate the resampler returns its input, so the analytic fixture's
canonical samples are its integers divided by full scale, and its digest can be
recomputed without the decoder. Signed-zero normalisation runs on it but has nothing to
change, because integer PCM decodes no negative zero. The resampler's output is not
bit-identical across architectures, so its digest is asserted only on the reference
platform, `darwin_arm64`, where canonical artifacts are generated.

The benchmark runs on the reference platform. A run on any other platform aborts with
`ENVIRONMENT_MISMATCH` before any fixture is tried. "Digest asserted on" names where
the test suite checks a fixture, so the analytic fixture is checked on Linux CI as
well; it does not permit a run elsewhere.

## Framing

Evaluation frames are 32,000 samples, two seconds at 16 kHz, cut from the canonical
waveform with a hop of 32,000 samples, so frames do not overlap. The first frame
starts at sample 0.

The duration lives in the protocol as its own field, and `frame_samples ==
frame_duration_s * sample_rate` is enforced, so a config claiming two seconds can
never carry ten seconds of samples. Prose cannot enforce that, because code cannot
read prose, and a loader constant would move the magnitude out of the protocol that
is supposed to own it.

A trailing remainder shorter than one frame is not a candidate, and its sample count
is recorded. Dropping it is the obvious choice; counting it is the part worth
stating, because a remainder that is neither scored nor counted hides how the
candidate set was formed.

A frame is identified by the SHA-256 of its recording's canonical waveform together
with its zero-based index, frame `i` covering samples `[i * 32000, (i + 1) * 32000)`.

## Eligibility

Determined **before either system runs**, from source audio and ground truth only,
never from model output. Every eligibility figure is computed from the canonical
waveform, not from the source file, so a recording's verdict does not depend on which
of two decode paths reached it. The withdrawn benchmark decided eligibility at scoring
time, so each system's denominator was shaped by its own failures.

**Benchmark eligibility and metric applicability are separate concerns.** A metric
must never shrink the population the other metrics use, or the benchmark silently
becomes "performance on the material that metric happens to accept."

**Benchmark eligibility** is domain-neutral. A recording is eligible if it decodes
cleanly, is at least 60 s long, has full-file RMS above -50 dBFS, and passes
disjointness. A frame is eligible if its clean reference has RMS above -45 dBFS.
Duration is compared in integer samples, and both RMS floors are strict
greater-than. "Above" reads that way in English, but the config states the
comparator rather than leaving code to infer strictness from prose. Energy is
`numpy.mean(x * x)` over the float64 samples with the pinned NumPy, so the order of
the sum is fixed rather than left to whichever of `np.mean`, `np.sum`, or `np.dot` an
implementation picks, which can differ in the last bits.

**Eligibility codes.** A recording can fail with `RECORDING_BELOW_MIN_DURATION`,
`RECORDING_BELOW_RMS_FLOOR`, or `RECORDING_WITHOUT_ELIGIBLE_FRAMES`, and a frame with
`FRAME_BELOW_RMS_FLOOR` or `FRAME_NOT_SCOREABLE_BY_PRIMARY_METRIC`, each list in
precedence order. The first failure in that order is the primary reason, and any others
are recorded as supplemental. Both recording floors are always evaluated, and both frame
checks are always evaluated, so neither verdict in a pair hides the other.
`RECORDING_WITHOUT_ELIGIBLE_FRAMES` is evaluated only for a recording that cleared both
recording floors, once its frames are cut after partition assignment, so it is never
supplemental to a floor code. Such a recording keeps its partition position,
contributes no frames, and is left out of the across-recording count.

Every candidate frame is recorded with its RMS and its verdict, not only the
eligible ones. The population considered before filtering is then provable from the
manifest rather than asserted, which is the same denominator discipline the
withdrawn benchmark failed at scoring time.

That floor exists to reject degenerate frames where a reconstruction ratio is
numerically meaningless: digital silence, zero padding, decode artifacts. It is not
intended to remove quiet music. A pianissimo passage is legitimate classical
content and difficulty is not a reason for exclusion. Across 4,000 sampled clean
frames from the *training* corpus the distribution bottoms out near -39 dBFS at the
0.5th percentile, so the floor excluded nothing there; its effect on the frozen
candidate population was unknown when it was chosen, which is the point of choosing
it beforehand.

**Metric applicability** is decided from the clean reference only, before either
system runs, never from model output. Each primary metric's reference-side validity is
checked on every clean frame, and a frame that any primary metric cannot score, such
as a constant reference for SI-SNR, is ineligible for every metric with
`FRAME_NOT_SCOREABLE_BY_PRIMARY_METRIC`. The primary metrics therefore share one
population: they apply to every eligible frame, and no metric shrinks the population
the others use. The ~15,000 frames dropped by the withdrawn benchmark were not quiet by
RMS at all; PESQ's internal voice-activity check refused to score them, which is the
failure this rule prevents.

Populations tracked and reported separately: eligible, submitted, scored.

## Source outcomes

What record exists when the transformation cannot complete, frozen here because an
implementation needs a deterministic answer before it meets its first unreadable
file, and because these verdicts decide population membership.

**A decoder rejection is evidence about our own local artifact.** It proves that the
bytes on this machine could not be inspected or decoded. It does not prove the
artifact is malformed, truncated, or corrupt, since a rejection can also come from an
unusual but valid encoding feature or a decoder limitation. It certainly does not
prove the publisher's file is defective, because our own retrieval could have damaged
it. The reason codes are therefore neutral about cause: `SOURCE_INSPECTION_REJECTED`
and `SOURCE_DECODE_REJECTED`, never a code asserting corruption. The closed
attribution field is named for what the preconditions establish, a verified local
artifact, rather than for source ownership; a machine-readable field that claims more
than the note beside it is worse than prose, because tools quote the field.

The native decoder diagnostic is supplemental provenance, never the stable
classification, because its wording varies by platform and build. The published record
carries the reason code and a sanitised diagnostic while the raw string stays in the
run log, since native diagnostics can carry filesystem paths and local filenames and
nothing about a decode failure is worth leaking a local path into a public artifact.

**The collection publishes no checksum and the acquisition script records none.** The
local SHA-256 identifies the exact artifact inspected and decoded and establishes
nothing about equivalence to the publisher's original bytes. Publication language
must hold that line: a locally acquired candidate artifact could not be decoded, not
a collection file was corrupt. Re-acquiring and comparing digests is corroboration
rather than proof, since differing bytes demonstrate that retrieval is unstable while
identical bytes demonstrate only that it is repeatable.

A decode rejection may be charged to the source only after the preconditions hold, in
this order: protocol semantics, runtime conformance, a successfully decoded golden
fixture, the exact local artifact bytes and their digest, then inspection and decoding
of those same bytes, then classification. The fixture comes before any candidate for
a specific reason. A decoder that cannot decode anything would otherwise be charged
to whichever source it met first.

Two vocabularies, and they are structurally separate rather than merely distinguished
by convention:

| | Effect | Codes |
|---|---|---|
| Source-ineligible | recorded, published, run continues | the 13 codes below |
| Abort | run stops, publication blocked | `SOURCE_HASH_MISMATCH`, `LOCAL_ARTIFACT_UNSTABLE`, `LOCAL_ARTIFACT_NOT_REGULAR_FILE`, `GOLDEN_FIXTURE_FAILURE`, `ENVIRONMENT_MISMATCH`, `ENVIRONMENT_CAPACITY_FAILURE`, `UNEXPECTED_IO_FAILURE`, `INTERNAL_CANONICALISATION_ERROR` |

Keeping them in separate closed vocabularies is what makes a harness failure not
merely discouraged from becoming a source verdict but unable to be spelled as one.

The ineligible codes are reported by category and never as one undifferentiated count
of bad files. A recording refused for exceeding a predeclared size or duration limit
has nothing wrong with it; it simply sits outside a domain this benchmark declared
before looking. Publishing that beside a decoder rejection would invite the reader to
conclude the collection is defective.

| Category | Codes |
|---|---|
| Frozen domain exclusion | `SOURCE_EXCEEDS_MAX_BYTES`, `SOURCE_EXCEEDS_MAX_DURATION`, `SOURCE_EXCEEDS_MAX_DECODED_BYTES`, `SOURCE_BELOW_MIN_SAMPLE_RATE` |
| Unsupported representation | `UNSUPPORTED_CONTAINER`, `UNSUPPORTED_SUBTYPE`, `UNSUPPORTED_CHANNEL_COUNT` |
| Artifact or decoder outcome | `SOURCE_INSPECTION_REJECTED`, `SOURCE_DECODE_REJECTED`, `SOURCE_METADATA_INCONSISTENT` |
| Decoded waveform violation | `EMPTY_SOURCE`, `NONFINITE_SOURCE`, `INVALID_SOURCE_AMPLITUDE` |

Every code belongs to exactly one category and the categories cover the vocabulary
completely. Every frozen bound has a code to reject against, which is less obvious than
it sounds: `max_decoded_bytes` was enforced with no way to record it, and a highly
compressible 192 kHz source can pass both the byte cap and the duration cap while
decoding to 10.3 GiB against a 1.50 GiB limit.

### The evaluation stage machine

Everything below leans on the word "stage", so the stages are frozen rather than left to
prose. Without that, two conforming implementations could disagree on whether inspection
and metadata validation are one stage or two, and emit different records for the same
artifact while both claiming conformance.

```
local_file_check -> inspection -> metadata_validation
  -> decode -> decoded_metadata_validation -> waveform_validation
```

This axis is eligibility adjudication only, and identity acquisition is deliberately not
on it.

> **Note:** an earlier draft of v4 placed identity acquisition in this order and produced
> a contradiction. An oversized artifact terminates at `local_file_check`, so the derived
> unevaluated set named the identity stage, while the same record carried a streaming
> digest that only that stage could have produced. A stage cannot be both skipped and
> executed. Evidence collection is not eligibility adjudication, and putting them on one
> axis forced the collision.

Each stage declares what it checks, so that "evaluate every safe check in the stage"
means the same thing in two implementations:

| Stage | Checks |
|---|---|
| `local_file_check` | regular file, source bytes |
| `inspection` | decoder inspection |
| `metadata_validation` | container, subtype, channel count, minimum sample rate, duration, projected decoded bytes |
| `decode` | decoder decode |
| `decoded_metadata_validation` | metadata matches decoded array |
| `waveform_validation` | non-empty, finite, amplitude range |

A stage completes when every check assigned to it has executed, and processing stops
after the first stage that establishes a violation. Every ineligible reason names exactly
one stage, so the stage on a record is derived rather than chosen. Six checks share
`metadata_validation`, which is where supplemental reasons actually arise.

`unevaluated_evaluation_stages` is derived, not authored: the stages after the
terminating one, in stage order, as an ordered tuple. Ordered rather than a set, because
two implementations serialising the same stages differently would produce different
manifest bytes and therefore different digests.

This axis is also distinct from the precondition order, which sequences run-level gating
such as the golden fixture and does not describe one candidate's journey.

### Identity as evidence, not adjudication

Which method establishes a digest follows from the source size alone:

| Source size | Status | Method |
|---|---|---|
| at or below `max_source_bytes` | `complete_sha256` | `bounded_single_buffer_sha256` |
| above that, at or below `max_identity_stream_bytes` | `complete_sha256` | `bounded_streaming_sha256` |
| above `max_identity_stream_bytes` | `unavailable_above_identity_stream_bound` | `not_computed_above_bound` |

The first two both report `complete_sha256` and reach it by materially different evidence
paths, so the method is recorded explicitly rather than inferred from a status that cannot
distinguish them.

The names describe how the digest was established and nothing else.

> **Note:** an earlier draft named the first method for hashing, inspecting, and decoding
> one buffer. That is the canonicalisation byte-path contract, not an identity method, and
> as a record it overstated what happened: a source rejected at inspection or on its
> subtype is never decoded, so the record would have asserted execution that did not occur.
> What ran is already carried by the terminating evaluation stage and the unevaluated
> suffix.

Status, method, digest presence, and source size are four views of one fact, and any pair
of them can be made to disagree. `complete_sha256` beside a method of not-computed, an
absent digest beside a complete status, or a streaming method for a source small enough to
buffer are each individually plausible as a typo and each would publish an identity claim
that never happened, so the record constraints tie all four together.

**A candidate can violate several rules at once.** Exactly one primary reason is
recorded and counted, taken as the first applicable entry in a frozen precedence.
Without that order, two conforming implementations publish different category totals for
the same artifact.

The precedence follows what can be known at each stage rather than any judgement about
severity. The size is known from `fstat` before anything is read, inspection either
succeeds or does not, container and subtype and channels and rate and duration and
decoded footprint all come from the same inspected metadata, decoding either succeeds
or does not, and the array checks require a decoded array. Within the metadata stage the
order lists what the file is before what it contains, which is a frozen convention and
not a claim that one violation matters more.

Evaluation order and primary-reason precedence are not the same thing. Processing stops
at the first stage that terminates it, but within the stage it reaches, every check that
can be evaluated safely is evaluated before the primary reason is chosen. Otherwise an
implementation that noticed an unsupported subtype would stop and never record an
already-knowable duration violation.

**Supplemental reasons are those established, not those applicable.** Every additional
reason established before processing terminated is recorded beside the primary one, and
none of them enters a count. Counting them would make the categories overlap and stop
them summing to the narrowing they describe.

> **Note:** an earlier draft of v4 promised every *applicable* reason, which is
> structurally impossible. An oversized source is rejected from `fstat` and is never
> inspected or decoded, so whether its subtype, channel count, rate, or duration would
> also have failed is unknowable without doing exactly the work the exclusion exists to
> avoid.

Every stage never reached is therefore named in the record. Without that, the absence of
`UNSUPPORTED_SUBTYPE` reads as evidence that the subtype was supported, when inspection
may never have happened. Not evaluated and passed are different facts.

### Artifact identity on failure

`SOURCE_EXCEEDS_MAX_BYTES` is decided from `fstat`, before the artifact is buffered, so
the in-memory path cannot produce a whole-artifact digest for it. Hashing the bounded
`max + 1` prefix and calling it `local_sha256` would be a provenance lie, since that
field identifies the whole artifact.

Such a source is streamed through SHA-256 in bounded chunks and never decoded or
retained, up to `max_identity_stream_bytes`. That second bound is on work rather than
memory, and it exists because streaming bounds memory and bounds nothing else: at
200 MB/s a 100 GB file costs about 8 minutes of I/O and a 2 TB sparse file about 3
hours, all to populate one audit field for a candidate the benchmark has already
excluded. The bound is on bytes processed, not on elapsed time: storage speed, scanning
layers, and network-mounted volumes all vary, so a wall-clock guarantee is not something
this bound can make. A runner may add a timeout, and a timeout is a harness abort rather
than a source verdict. The bound is also derived from `max_source_bytes` rather than left
free: it must equal exactly sixteen times it. An accepted range would leave a band of
configurations that are semantically permitted and have never been analysed, so the value
is stated for readability and enforced as a relationship. A validly excluded source must not be able to buy unbounded work from the
harness as its parting act. Above the bound no digest is claimed and `identity_status`
records why. The reason code is unchanged either way: how large a file is does not
change why it was excluded, only what could be established about it.

Two identity paths therefore exist and are named separately, and the declared basis
names both. Decode identity is the bounded single buffer that is hashed, inspected, and
decoded. Exclusion identity is the streaming hash with no decode. An earlier draft
declared the basis as the decoded buffer alone, which described one of the two paths
while claiming to state the basis for both.

The single-buffer path is stable by construction. The streaming path reads over an
interval, so `st_dev`, `st_ino`, `st_size`, `st_mtime_ns`, and `st_ctime_ns` are
compared before and after on the same open descriptor, and hashing goes through that
descriptor rather than reopening the path, which is what stops a path replacement from
silently changing which inode was read.

> **Note:** this detects instability; it does not prove stability. Metadata equality
> cannot exclude a content mutation made and reverted during hashing. What the protocol
> can say is that no instability was detected by these checks. A mismatch in any of the
> five values aborts under `LOCAL_ARTIFACT_UNSTABLE`, because a file mutating on our own
> disk is a fact about this environment and not about the audio. A failure to obtain the
> second stat is `UNEXPECTED_IO_FAILURE` instead, since not knowing is not the same as
> having detected a change. In both cases the digest computed during that read is
> discarded.

### The failure record

Required core plus stage-specific evidence, rather than one flat tuple every failure
must somehow fill. A pre-inspection resource exclusion has no decoder diagnostic and no
audio metadata; a decoded-waveform rejection has both. A universal field list forces
placeholders that pretend evidence exists, and a reader cannot then tell an absent value
from an inapplicable one.

| | Fields |
|---|---|
| Always | `logical_source_id`, `terminating_evaluation_stage`, `primary_reason_code`, `identity_status`, `identity_method`, `protocol_sha256`, `unevaluated_evaluation_stages` |
| When `identity_status` is `complete_sha256` | `local_sha256` |
| When inspection succeeded | `observed_metadata` |
| When the decoder rejected the artifact | `decoder_diagnostic` |
| When further violations were established | `supplemental_reason_codes` |

Stating the condition beside each conditional field makes an invalid combination
unrepresentable rather than merely discouraged. Each condition is a closed token rather
than a sentence, for the same reason every other rule here is one.

This schema covers source-ineligible outcomes only. An abort can happen before any source
is in hand, so an environment mismatch or a golden-fixture failure has no logical source
id, no stage in this machine, and no reason from this vocabulary. The structure of abort
records belongs to the runner and is deliberately not frozen here.

A record is further constrained: the primary reason comes from the ineligible vocabulary,
the supplemental codes are a subset of it that excludes the primary, they are unique, they
all share the primary reason's stage, they are ordered by the reason precedence,
`unevaluated_evaluation_stages` is ordered by the stage order, identity status and method
and digest presence and source size agree with one another,
and any digest computed during a read later found unstable is discarded. Ordering is part
of the contract because two implementations holding the same set and serialising it
differently produce different manifest bytes, and therefore different digests, which would
break the content addressing the whole benchmark rests on. The stage-membership constraint
is what gives "established before termination" a meaning a machine can check: an earlier
stage cannot hold an unrecorded violation, because it would already have terminated
processing, and a later stage was never reached, so every valid supplemental comes from the
terminating stage itself. The last constraint is a safety rule rather than a formatting
one: a digest present in a record reads as an established
identity, so one computed during an unstable read must never survive as supplemental
evidence.

**Ownership of the decoded bound.** A projected decoded size above the frozen protocol
maximum is a domain exclusion and is recorded as `SOURCE_EXCEEDS_MAX_DECODED_BYTES`. A
projection within that maximum that this runtime cannot allocate for is
`ENVIRONMENT_CAPACITY_FAILURE` and aborts. The distinction stops an implementation
substituting its own machine's memory for the protocol's limit and publishing the
difference as a property of the audio.

## Degradation

Frames are cut only after partition assignment, so every derived artifact inherits
its recording's partition and siblings cannot cross the boundary.

Degradation is procedurally synthesized, not sampled, so there is no noise-source
pool to be disjoint from. The process is the one that generated the training pairs:
`benchmark/degradation.py`, a port of `dataset/noise.py` at commit `d8993b2`. The
current `dataset/noise.py` is not that process (see [v5 to v6](#v5-to-v6)). The evidence
that the port is training's process:

- On a seeded sample of 400 unclipped training pairs with crackle and hum off, the
  added hiss peaked at `sqrt(signal_power / snr_linear)` to float32 rounding (median
  relative error 8.3e-8, maximum 1.7e-6), which is `d8993b2`'s unit-peak scaling
  (`benchmark/measurements/hiss_peak.py`).
- The training hiss's median spectral slope matched the port's: pink -1.000 against
  -0.999, brown -1.934 against -1.936 (`benchmark/measurements/hiss_colour.py`).
- Given the same random stream, every ported primitive and the composite were
  bit-identical to `d8993b2` in 500 of 500 comparisons, and the parameter draws matched
  for 200 of 200 seeds (`benchmark/measurements/d8993b2_equivalence.py`).

Frozen in `v7.json` under `degradation`:

- **Process and ranges.** Every frame always gets tape hiss (SNR uniform over 5 to 30 dB
  as recorded, pink or brown) and a low-pass rolloff (4 to 8 kHz). Vinyl crackle, mains
  hum, and tape saturation are each on with training's probability (0.5, 0.3, 0.4), with
  training's ranges. The recorded SNR is nominal: training's hiss scaling made the added
  hiss quieter than it.
- **Seed and draws.** The run seed is 20261122. Each frame draws from its own generator,
  seeded from the run seed, the eight 32-bit words of its recording's canonical digest,
  and its frame index, first the parameters and then the hiss, crackle, and hum samples.
  Training recorded no seed, so this seed cannot be shown to differ from training's; what
  it fixes is that the benchmark's draws are reproducible.
- **One degraded copy per frame.** More copies would multiply each frame's weight in the
  scored population.
- **Training level.** Training normalised each recording to a peak of 0.95 before
  cutting frames, and crackle and hum are added at fixed amplitudes. Each frame is
  therefore multiplied by `0.95 / peak`, where `peak` is the largest absolute sample of
  its complete canonical waveform, before it is degraded. Systems receive the degraded
  frame at that level, and the metrics compare their output with the canonical frame
  times the same gain. Eligibility, the frame identifier, and the loudness floors stay
  on the unscaled canonical waveform.
- **Golden digest.** Frame 0 of the analytic golden fixture, degraded with seed 2 (the
  smallest seed whose draws turn on saturation, crackle, and hum), has a frozen SHA-256
  over its float32 samples. It is asserted on the reference platform only, as the
  resampler fixture is. A run must recompute it after the golden fixtures and before any
  candidate, and a mismatch aborts with `GOLDEN_FIXTURE_FAILURE`.

Each degraded frame is to be recorded with its recording's logical source identifier,
its partition, its frame index, and its seed. The record's schema belongs to the runner
and is not frozen here.

## System execution

Frozen in `v7.json` under `system_execution`. Each degraded frame is written as a 16 kHz
mono float WAV and passed to each system's shipped backend engine with one `denoise`
call; the output WAV is read as float64. OpGAN is the checkpoint whose SHA-256
`backend/app/config.py` pins. UVR is `UVR-DeNoise.pth` through audio-separator. Before
any candidate, the runner must compare the digest of each file an engine loads with the
frozen digests: the OpGAN checkpoint, the UVR model, and the two UVR parameter files.

From v9 both engines run in the run's own process, each built once after the run
preconditions pass. An exception raised inside a `denoise` call is that system's failure
for that frame. A native crash ends the process, so no publication artifact is written
and the run is invalid; it is charged to neither system. Before each call the runner
writes, atomically, a marker naming the recording, the frame, and the system, so the frame
a crash happened on can be identified afterwards.

Before v7 was frozen, both engines were run on 23 degraded float WAV frames built from
synthetic tones and training clean frames. Every call succeeded, both returned 16 kHz
mono 16-bit WAV, OpGAN returned exactly as many samples as it was given, and UVR
returned 130 fewer on every frame.

> **Note:** this describes the shipped engines on training-format input, not the
> product's full upload path. Float input bypasses the product's ffmpeg decode to 16-bit
> PCM, and the product decodes UVR uploads at 44.1 kHz where here both engines receive
> 16 kHz frames.

> **Note:** audio-separator 0.44.2 ships no integrity hash for `UVR-DeNoise.pth`, and it
> downloads the model's parameter files from the `main` branch of the
> `TRvlvr/application_data` GitHub repository when it loads the model. The three UVR
> digests were recorded on first download, on 2026-11-20, and are pinned from then on;
> nothing upstream lets them be checked against an earlier source.

The versions the systems and the degradation run on are pinned and checked at load:
scipy 1.17.1, audio-separator 0.44.2, and torch 2.14.1. `uv.lock` resolves torch to
2.14.1 on macOS and to the CPU build 2.14.1+cpu elsewhere, and the check accepts exactly
those two builds; runs execute only on the reference platform, where the build is 2.14.1.

## Runner

Frozen in `v8.json` under `runner`. These are duties on the code that runs sources, which
the loader cannot carry out; no runner exists yet.

- **One child process per source, one at a time.** A worker that dies cannot take a
  record with it, so the parent decides what its death means.
- **A model file that does not match its frozen digest** aborts with
  `ENVIRONMENT_MISMATCH`.
- **A worker that catches `MemoryError` and reports it, or that is killed by `SIGKILL`,**
  aborts with `ENVIRONMENT_CAPACITY_FAILURE`. The `SIGKILL` rule assumes the system kills
  an out-of-memory worker with `SIGKILL`: a real out-of-memory kill was not triggered
  before v8 was frozen, and only a `SIGKILL` a process sent itself was observed. A manual
  `kill -9` is recorded the same way. Either way the run aborts and no source receives a
  verdict, so only the abort's label depends on it.
- **Any other death of the worker,** such as a segmentation fault or an exit without a
  record, aborts with `INTERNAL_CANONICALISATION_ERROR` naming the source. It is never
  recorded as a verdict on that source: a crash the runner cannot attribute is not
  charged to the recording.
- **Each worker's peak memory footprint is recorded per source.**

On the reference platform, before v8 was frozen, a worker's failures reached its parent
as follows: `SIGKILL` as return code -9, a segmentation fault as -11, and an uncaught
`MemoryError` as status 1, the same as any other uncaught exception. That last is why the
worker has to catch `MemoryError` and report it.

**The capacity fixture.** The admitted source that maximises the projection is mono, so
the downmix is a full copy of the decoded array, decoded at `max_decoded_bytes`, and long
enough for the 3600 s cap to leave its rate at 55,926 Hz: 201,326,592 frames of 24-bit
FLAC carrying two tones, generated in code. The metric is the lifetime peak memory
footprint a fresh process reports through `proc_pid_rusage` after acquisition,
inspection, decode, and canonicalisation; it agreed with `/usr/bin/time -l` to within
about 16 KB in a check at 1 GB and 3 GB. Over five runs on `darwin_arm64` the footprint
was between 4,881,993,664 and 4,883,025,856 bytes. The largest is 75.8 percent of the
6 GiB budget and about 963 MiB above that source's projection, within the 1024 MiB
allowance v5 set for the interpreter and temporaries. It is an observation, not an upper
bound. The peak resident set size was not used: it does not count memory the system has
compressed, and over the same five runs it varied between 3,627,614,208 and
4,944,855,040 bytes. The fixture's FLAC is 191,367,147 bytes. A near-incompressible mono
source of 510,537,464 bytes, close to the source byte bound, decoded to 1.36 GB; it was
measured once and by resident set only, at 4,517,314,560 bytes. A source at both bounds
at once was not measured.

> **Note:** the budget is measured, not imposed. On the reference platform `setrlimit`
> refused `RLIMIT_AS` at 1, 6, 8, and 64 GiB ("current limit exceeds maximum limit"), so
> a worker's memory cannot be capped at the budget through it. The runner therefore
> records each worker's peak footprint rather than capping it, and the measurement holds
> for this platform only.

## Metric conditioning

Frozen before results exist, because "alignment fix" is otherwise indistinguishable
from shifting outputs until the metric improves.

- Metrics computed at 16 kHz mono. The reference is a frame of the waveform produced by
  [Canonicalisation](#canonicalisation), which pins every step of getting there,
  multiplied by the training-level gain described under [Degradation](#degradation).
  Metric libraries frozen: numpy 2.3.5 and soundfile 0.14.0. v6 dropped the legacy
  PESQ and STOI diagnostics, so `pesq` and `pystoi` are not metric libraries here.
  scipy is pinned for the degradation rather than for the metrics. Exact versions are
  recorded per run
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
same 2-second 16 kHz unit this benchmark scores. Before v7 was frozen, the shipped UVR
engine returned 130 samples fewer on each of 23 degraded 32,000-sample frames, which
leaves a margin of 30 samples under this bound.
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

SI-SNR needs a nonzero reference direction, and centring a constant reference leaves
none. A reference whose samples are all equal is therefore invalid for this metric, while
clearing the loudness floor comfortably. `eps` stabilises otherwise-valid arithmetic and
must not make such an item scoreable: without the rule it returns a plausible number,
measured at 0.0 dB against a constant estimate and -109.8 dB against a noise estimate.

The condition is constancy, not zero energy after centring. Those look equivalent and are
not. Whether a constant array centres to exactly zero depends on whether its value is
representable in binary: 0.5 and 0.25 are, while 0.1, 0.3, 0.7, and 1/3 leave residues
near 1e-30 that a zero-energy rule accepts and then scores at 0.0 dB. Constancy is exact,
so no tolerance is defined and none is needed; a near-constant threshold would be an
unstated protocol parameter.

**Log-spectral distance.** Every convention that moves the number is stated, because
several of them are otherwise inherited from whichever library is installed. v1 fixed
the parameter values but named the window only as "hann" and never named an FFT
normalisation; both change the result. The config expresses each as a closed
vocabulary rather than an equation string, so code branches on a named operation
instead of parsing prose.

```
lsd = mean over frames of
        sqrt( mean over bins of ( 20*log10(|X|+offset) - 20*log10(|Y|+offset) )^2 )

    n_fft         = 1024
    win_length    = 1024
    hop_length    = 256
    bins          = all, including DC and Nyquist

    window        kind hann, symmetry periodic
                  w[n] = 0.5 - 0.5*cos(2*pi*n/win_length),  0 <= n < win_length
    fft           kind rfft, norm backward (unnormalised forward transform)
    framing       center true, pad_mode constant, pad_value 0.0
                  pad_left_samples 512, pad_right_samples 512
                  frame starts at multiples of hop from zero
                  no partial trailing frame
    log_magnitude operation amplitude_db_additive_offset, offset 1e-8
                  that operation is defined as 20*log10(|X| + offset)
    reduction     rms across bins, then arithmetic mean across frames
```

The dB factor is not a configurable setting. The 20 in `20*log10` is what separates
amplitude spectra from power spectra, so it belongs to the named operation rather than
to the list of numbers a config may choose; 10 would halve every distance while looking
entirely plausible. The offset is a magnitude and does live in the config.

The window is defined by that equation rather than by a library function name. The
symmetric convention divides by `win_length - 1`, answers to the same name, and is a
different window: `numpy.hanning` returns the symmetric one and
`scipy.signal.get_window` returns the periodic one.

Normalisation matters here for a reason specific to this metric. In a pure log ratio
a uniform magnitude scaling would cancel between the two spectra. The offset is
additive, so scaling moves magnitudes relative to it and does not cancel.

The offset is an additive term, not a clamp. `max(|X|, offset)` is a different
metric. The reduction order is also load-bearing: RMS across bins followed by a mean
across frames is not the same statistic as the reverse.

Worked consequence, for the 2-second evaluation unit at 16 kHz. 32000 samples padded
by 512 on each side gives 33024; with a 1024-sample window, a hop of 256, and no
partial trailing frame, that is 126 frames of 513 bins. A production transform
yielding any other shape for a normal unit is wrong.

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

### Reporting contract

The three primary metrics are co-primary estimands, reported together and
interpreted individually. There is no composite score, no majority vote, and no
omnibus "overall winner", because collapsing three correlated views into a single
verdict recreates the freedom this protocol exists to remove. Requiring all three to
agree before anything may be said is the opposite error: SI-SNR is built to ignore
gain, so letting it veto a claim about gain fidelity would penalise the design that
makes the three worth reporting.

Two rules follow, and both are enforced by the loader rather than left to editorial
judgement:

- **Three or none.** Wherever any primary effect appears, all three appear, in the
  order above. A generator that emits one emits all of them. This is a
  presentation-integrity rule against selective emphasis. It is not a
  multiple-comparison correction and is not described as one.
- **No significance language.** Each interval is a marginal 95% bootstrap interval
  for its named metric. Results are stated as `reconstruction SNR difference +X dB
  [95% CI a, b]`, never as "significantly better" and never as "better overall".
  Allowing a directional claim whenever any one of three intervals excludes zero
  would inflate the family-wise false-positive rate above the nominal per-metric
  level. Because the metrics are correlated by construction, the size of that
  inflation is not the independence figure and is not quantified here. Reporting
  effects with intervals and drawing no verdict sidesteps the question rather than
  answering it badly.

### Legacy diagnostics

Dropped in v6. v5 listed PESQ and STOI as legacy diagnostics, kept for comparability
with the withdrawn figures and excluded from the headline, and defined PESQ's
applicability subset as the frames where `PESQ(clean, clean)` succeeds. No version
defined the parameters that computing either metric, or that subset, needs: the PESQ
mode and the STOI settings.
The withdrawn figures remain in [benchmarks.md](benchmarks.md) as history.

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

Both systems receive an identical population, checked by identifier at two levels.
Recording identifiers must be equal across systems and equal to the frozen expected
set. Within each recording, frame identifiers must also be equal across systems and
equal to that recording's frozen eligible set: equal counts are not sufficient,
because two systems can score the same number of different frames and produce
per-recording means that are not comparable.

Eligibility is computed once from the clean reference and frozen, never recomputed
from a system's own output. The expected population is always read from the
manifest, never inferred from whichever results exist, because deriving it from
results makes coverage 100% by construction and turns the coverage gate into
decoration. Comparing whichever items both systems happened to succeed on rewards
failure and is not permitted.

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

**Coverage** is computed per system: the frames with a valid output divided by the
eligible frames. The headline requires 1.0 for both systems. A failure inside one
system's denoise call for a frame, whether an exception or an output that conditioning
rejects, is a system failure for that frame: the frame stays in the population and
lowers that system's coverage. A failure outside the call, such as preparing the input,
reading the output, or the environment, is a harness failure and aborts the run, so the
harness's faults are never charged to a system.

From v9, the reference an output is scored against is the canonical frame times the
frame's gain, computed in float64, then truncated to the common length by conditioning.
Eligibility checks the frame as cut, so a frame can pass eligibility and still give a
reference that a primary metric refuses once truncated: a frame that is constant apart
from its last 130 samples, scored against an output 130 samples shorter, is one. That is a
harness failure and aborts the run; it is never charged to the system whose output was
being scored.

## Aggregation

Frames within a recording are not independent observations; reporting a frame count
as N overstates confidence by roughly the frames-per-recording factor.

1. Per-frame primary metrics
2. Within each recording, the arithmetic mean of its eligible frames, unweighted.
   The published estimand is therefore the **mean frame-level metric for a
   recording**, which is not the same quantity as a metric computed over the whole
   recording's aggregate signal energy
3. Across recordings, the unweighted macro mean and the median. With an even number
   of recordings, the median is the mean of the middle two
4. Bootstrap over recordings, never frames: 10,000 resamples, percentile method,
   95% level, seed `20260726`. For the paired comparison, per-recording differences
   are formed first and then recording indices are resampled once per iteration, so
   the pairing is preserved; the two systems are never resampled independently

The draw is fixed: a single `numpy.random.Generator(PCG64(20260726))` draws one index
array per iteration, `integers(0, n, size=n)`, with replacement, in iteration order,
and the same array is applied to every estimand. The interval bounds are
`numpy.percentile` of the iteration means with `method="linear"` at 2.5 and 97.5.
The index arrays pick positions, so the order of the recordings changes the intervals;
from v9 the recordings are in the test partition's admission order.

Bootstrap reproducibility depends on the pinned runtime, not on the seed alone. A
seed selects a stream only within a fixed generator implementation, so the protocol
pins the BitGenerator (`PCG64`), the NumPy version, and the sampling procedure and
draw order. A runtime whose NumPy version differs from the pinned one fails to load
the protocol rather than running with a silently weakened guarantee, because a
skipped integrity check leaves the suite green while the guarantee is gone.

Because both systems see the same recordings the comparison is paired, so each
primary metric is reported as a per-recording difference with its own bootstrap
interval, which is more informative than two independent intervals. "Headline"
means that set of three differences, never a single summary of them.

Reported per metric: per-recording mean and median, 95% CI by recording, paired
difference CI, N recordings, N eligible frames, and coverage.

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
exist produces a new version, with the reason recorded, before evaluation runs. A
defect discovered after results exist invalidates those results rather than
retroactively changing the rules that produced them. Superseded versions stay in the
tree rather than being replaced, and each version names its predecessor by digest;
the loader recomputes those digests, so the chain is checked rather than asserted.

### v9 to v10

v9 said a transient fetch failure is retried three times and does not advance the walk,
but not what happens when the retries run out, nor which failures count as transient
beyond a network error, a timeout, or a 5xx. It said a flagged candidate is adjudicated by
a person, but not how a recorded decision is matched to the candidate, what the walk does
while a decision is pending, or which code a candidate flagged against several references
is recorded under. v10 defines them (see
[Identity and disjointness](#identity-and-disjointness) and [Selection](#selection)):
401, 403, and 451 join 404 and 410 as permanent, every other fetch failure is transient,
exhausted retries make the run invalid, and a recorded adjudication applies only to the
identifiers and digests it names, with the walk halting until one exists. No candidate
universe had been enumerated by the benchmark, and no candidate audio had been acquired or
inspected, when v10 was frozen. The Internet Archive item's file listing was read by
`dataset/acquire.py` when the training recordings were downloaded.

### v8 to v9

v8 left open which listed files are candidates, how admissions map to partitions and when
the walk stops, the duplicate screen's full parameters and its handling of degenerate
features, how a flagged take or another performer's recording is adjudicated, duplicates
among admitted candidates, the bootstrap's recording order, the precision of the scored
reference and a reference that truncation makes unscoreable, where the engines run, and
the order and codes of the selection checks. v9 defines them (see
[Identity and disjointness](#identity-and-disjointness), [Selection](#selection),
[System execution](#system-execution), [Scoring](#scoring), and
[Aggregation](#aggregation)).

Before v9 was frozen, the duplicate screen was recalibrated on the 145 training
recordings, on the reference platform, twice with identical output. The highest score
between two different recordings, 0.4832, came from a pair whose file names mark them as
two takes of one movement, so v9 counts a same-session take as the same performance and
moves the flag threshold from 0.50 to 0.30, between the next highest pair (0.1435) and
that take pair. No candidate universe had been enumerated by the benchmark, and no
candidate audio had been acquired or inspected. The Internet Archive item's file listing
was read by `dataset/acquire.py` when the training recordings were downloaded.

### v7 to v8

v4 to v7 placed four duties on the runner without defining them: which abort a model
digest mismatch records, how a worker's exit maps to an abort, which fixture the peak
memory measurement uses, and what that measurement showed. v8 defines all four (see
[Runner](#runner)) and replaces the resource bound note's "not yet measured" with the
measurement. The measurements used synthetic sources on the reference platform only. No
candidate universe had been enumerated, and no candidate audio had been acquired or
inspected.

### v6 to v7

v6 deferred degradation until the process that generated the training pairs was ported
and checked, and left open how systems receive input and which files they are. Before
v7 was frozen, the port was checked against the training pairs and against `d8993b2`
(see [Degradation](#degradation)), and both engines were run on 23 degraded frames built
from synthetic tones and training clean frames (see [System execution](#system-execution)).
No candidate universe had been enumerated, and no candidate audio had been acquired or
inspected.

**Frozen.** The degradation process, ranges, run seed, per-frame seed derivation, draw
order, one copy per frame, training level, and a golden digest; system execution through
the shipped engines with float WAV input and float64 output; the OpGAN checkpoint digest
and the three UVR file digests; and pins for scipy, torch, and audio-separator, checked
at load with the existing package check, where torch's pin names the two builds
`uv.lock` resolves.

> **Note:** v7 sets two run-time obligations on a runner that does not exist yet:
> recompute the degradation golden after the golden fixtures and before any candidate,
> and compare each model file an engine loads with its frozen digest. Until a runner does
> both, the pinned digests are recorded rather than enforced at run time.

**Corrections to the v6 document.** v6 said no benchmark module outside the tests
imports scipy. `benchmark/degradation.py` does, for the rolloff filter, so scipy is now
pinned. v6 said metrics are computed on waveforms produced by canonicalisation; under v7
the reference is that waveform times the training-level gain.

**Disclosures.** The run seed cannot be shown to differ from training's, which recorded
none. The UVR digests are trust on first use. The engines are measured on training-format
input, which bypasses the product's decode.

### v5 to v6

Found by an audit of v5 against the steps not yet built (the runner, selection,
scoring, aggregation, and publication), before any candidate universe was enumerated.
No candidate audio, canonical waveform, eligibility outcome, degraded input, system
output, metric score, aggregate, or comparative conclusion had been acquired,
generated, or inspected. The measurements behind v6 used training data only: the
metadata of the 146,200 training pairs and 25 of their audio pairs.

**Rules that decide which recordings are tested or how scores are formed.** v5 left
these undefined or stated them inconsistently. v6 defines:

- the collection, the source identifier, and the ordering the selection seed applies
  to. A seed selects a walk only together with all three;
- the decoded representation the third exclusion rule hashes;
- `CANDIDATE_PERMANENTLY_UNAVAILABLE` for a 404 or 410, which v5 excluded without a
  code;
- the recording and frame eligibility codes and their precedence;
- one scored population for all primary metrics, decided from the clean reference
  before either system runs. v5 said the primary metrics apply to every eligible frame
  and also that a constant reference is invalid for SI-SNR while clearing the loudness
  floor, and no rule said which held;
- coverage, and the boundary between a system failure and a harness failure;
- the bootstrap draw, the interval rule, and the median convention for an even count.
  The v5 document said the sampling procedure and draw order were pinned, but
  `v5.json` did not pin them.

**Dropped.** The legacy PESQ and STOI diagnostics. No version defined the parameters
that computing them, or PESQ's applicability subset, needs.

**Corrections to the v5 document.** The abort table listed six of the eight abort
codes. The failure record table used `stage` and `unevaluated_stages`, and left out
`identity_method`, where `v5.json` names `terminating_evaluation_stage`,
`unevaluated_evaluation_stages`, and `identity_method`. "Track" is now "recording" in
the aggregation keys (`within_recording`, `across_recordings`) and in the text that
describes them. The same-composer calibration row, which `v5.json` does not carry, is
marked as unchecked. The third exclusion rule catches lossless re-encodes, not
re-encodes in general.

**Deferred, not required by v6.** Composer, work, and performer per candidate; a
byte-exact manifest serialisation, so manifest digests are taken over the bytes of the
written file; and a coded check for a burned partition, which stays a written rule.

**Disclosures about the training data,** recorded so the benchmark is not described
more favourably than the evidence allows:

- The training pairs were written on 2026-03-07, between commits `d8993b2` and
  `2174dce`, and most likely came from `dataset/noise.py` at `d8993b2`. That version
  scaled tape hiss as though it had unit power after normalising it to unit peak. On 25
  pairs with only hiss and rolloff enabled, the realised hiss SNR was 5.41 to 13.06 dB
  above the recorded `hiss_snr_db`, median 10.25 dB. The current `dataset/noise.py`
  normalises the hiss power, so it is not the training process. The training metadata
  records no seed.
- Training dropped frames whose linear RMS was below 0.01 after normalising each
  recording to a peak of 0.95. The benchmark floor is -45 dBFS on the unnormalised
  canonical waveform, so the two frame populations differ. v6 does not change the
  benchmark floor.
- Degradation is not frozen in v6 (see [Degradation](#degradation)).

### v4 to v5

Discovered while implementing canonicalisation, before any candidate universe was
frozen and before any candidate canonical waveform, candidate frame eligibility
outcome, degraded input, system output, metric score, aggregate, or comparative
conclusion had been generated or inspected. The measurements behind v5 used synthetic
signals only, together with the training-corpus measurements v4 already cites.

**The golden fixture had no definition.** v4 required a golden fixture to be decoded
before any candidate but defined no fixture, no expected digest, and no field to hold
one, so the precondition could be neither executed nor checked. v5 freezes the two
fixtures described under [Golden fixtures](#golden-fixtures). Run on both platforms
available, with soxr 1.1.0, the same libsoxr build string (`0.1.3-14-ga66f3ee`), and
libsndfile 1.2.2 on each, the analytic fixture produced the same digest and the
resampler fixture did not:

- analytic, both platforms: `a092fd3b42e4c2142ce999ad11e7c6b8e05ff673408922d924c5367aec1d23b1` <!-- pragma: allowlist secret -->
- resampler, `darwin_arm64`: `a82cd013caa67872a1eccc17c8b20a55afa785044c6952731398e739da7b3df5` <!-- pragma: allowlist secret -->
- resampler, `linux_x86_64` (CI runner): `85cef603a2e5c2f319dd6255a09dac084e466b3e6d0f006240b341bddb2d5b36` <!-- pragma: allowlist secret -->

Under the existing `reference_artifact_authoritative` model, v5 names `darwin_arm64`
as the reference platform and requires runs to execute there, which is a new rule: a
run elsewhere aborts with `ENVIRONMENT_MISMATCH`.

**Two rules eligibility needs were left open.** v4 required frame identifiers to agree
across systems without defining them, and named `mean(x*x)` in prose without fixing
the reduction. v5 defines the frame identifier and the RMS reduction (see
[Framing](#framing) and [Eligibility](#eligibility)). No eligibility code existed, so
nothing computed under v4 changes.

**Clarifications, stating as rules what the v4 implementation already does where v4
was silent.** None changes a computation.

- A FLAC header whose total sample count is zero, meaning unknown, is reported by
  libsndfile 1.2.2 as 9,223,372,036,854,775,807 frames and excluded at metadata
  validation as exceeding the duration and decoded-size bounds.
- A subtype under an unadmitted container is not evaluated; the record carries
  `UNSUPPORTED_CONTAINER` alone.
- NaN fails finiteness only, having no position to compare against the amplitude
  range; an infinity fails both finiteness and amplitude.
- A FLAC header that understates its sample count decodes that many samples and the
  rest is dropped without an error; the record cannot show it.
- `EMPTY_SOURCE` stays a check although no FLAC buffer tried reached it under
  libsndfile 1.2.2: a zero-frame write produced zero bytes, and a total of zero means
  unknown.
- A canonical waveform that fails canonical validation aborts the run with
  `INTERNAL_CANONICALISATION_ERROR`; by then the source checks have passed, so no
  ineligibility stage is left to name.
- FLAC/PCM_32 stays admitted. libsndfile 1.2.2 cannot write it, so no fixture covers
  it, and reading it is untested.
- The environment check refuses to load when `LIBROSA_CACHE_DIR` is set (see
  [Environment](#environment)).

**Corrections to the v4 document.** The source-ineligible vocabulary has 13 codes, not
twelve. The metric environment runs scipy 1.17.1, which `uv.lock` resolves for
Python 3.11; v4 listed 1.18.0, which requires Python 3.12 or later. v4 said
canonicalisation and a CI conformance job call the environment check directly;
neither does, and no such job exists.

> **Note:** the v4 document was edited after it was frozen. Commit 048c6f2 added the
> sentence on refusing a configured librosa cache to the live document without an
> amendment. The archived [benchmark-protocol-v4.md](benchmark-protocol-v4.md) is the
> document as frozen at 630fe8c, without that sentence, and v5 adopts it here.

### v3 to v4

Discovered while implementing canonicalisation, before any system output,
candidate-selection outcome, or test-partition result had been generated or
inspected. v3 required content-addressed artifacts and specified the transformation
that produces their content only as "16 kHz mono", with resampling deferred to
"librosa's default `res_type`".

Every artifact in this benchmark is identified by a digest of its samples, so an
underspecified transformation is not a documentation gap. It makes the identity
undefined. Measured on a training FLAC: four `res_type` choices give four digests;
mean-of-channels and left-channel downmix give two; downmixing before resampling
differs from doing it after by 5.2e-10. Each of those is a defensible reading of v3,
and each yields a different answer to "is this the same audio".

A default is the worst of the choices available. It is invisible in review, it can
change between library releases without any edit to this protocol, and citing it as
the rule makes the library's future behaviour normative rather than this document's.
v4 names every argument, including the ones that happen to be defaults today.

Two smaller commitments come with it. Signed zero is normalised, because positive and
negative zero are numerically equal with different bytes; the current pipeline emits
only positive zeros on the fixture examined, so this closes a latent hazard rather
than a live one. Reproducibility is claimed only for artifacts produced by the pinned
reference implementation, since pinning package versions does not make native DSP
bit-identical across architectures; a golden fixture makes cross-platform agreement
something observed instead of assumed.

Framing and the eligibility comparators are pinned in the same amendment for the same
reason: frame size, hop, first-frame offset, and the treatment of a trailing
remainder all determine which frames exist, and "RMS above the floor" left strictness
to be inferred from English. A trailing remainder is dropped and counted, so the
manifest shows how the candidate set was formed rather than only what survived.

Two smaller commitments travel with the transformation. The forbidden-operation list
is matched exactly rather than checked for being non-empty, because a non-empty check
accepts a list naming one irrelevant operation while permitting every operation that
matters. The frame contract is stated as a relationship rather than as a constant in
the loader: the duration is a field of the protocol and `frame_samples ==
frame_duration_s * sample_rate` is enforced, so the magnitude stays where the digest
pins it while the claim becomes machine-checkable.

Source outcomes are part of the same amendment rather than a separate one, because an
implementation of the transformation needs a deterministic answer to what record
exists when it cannot complete, and those verdicts decide population membership. The
attribution rule is the substance: a decoder rejection is evidence about our own
locally acquired artifact, and charging it to the collection would blame an external
party for bytes our own retrieval may have damaged. Since no upstream checksum exists
for this collection, the protocol records that limitation rather than letting
publication imply an identity the run cannot establish.

Nothing already frozen changed. v4 adds `canonicalisation`, `framing`, and
`source_outcomes`, and adds keys to `eligibility` that state how existing thresholds
are applied without moving any of them.

### v2 to v3

Discovered while implementing the metrics, before any system output, candidate-selection
outcome, or test-partition result had been generated or inspected. v2 did not state
whether SI-SNR is defined when the centred reference has no energy.

A constant reference clears the raw energy check and has nothing left after centring,
leaving no direction to project onto. The configured epsilon then makes every subsequent
term finite, so the metric returned a score for an undefined quantity: 0.0 dB against a
constant estimate, -109.8 dB against a noise estimate. Both look like measurements.

The rule states constancy rather than zero energy after centring, and the distinction is
not cosmetic. A first draft of v3 said "exactly zero energy in float64, no tolerance",
which sounds stricter and is less faithful: four of six constant references tested,
including 0.1 and 1/3, do not centre to exactly zero because their value is not
binary-representable, and that draft scored all four at 0.0 dB. When the computation
producing zero is itself rounded, demanding exactness weakens the rule.

That decides whether an item receives a score, and therefore decides the valid
population, so it belongs in the protocol rather than in one implementation. Two teams
implementing v2 from the frozen config could otherwise disagree on the denominator, which
is the ambiguity class that produced v2 itself. v3 states the rule normatively and
changes no formula, constant, gate, or reporting behaviour.

### v1 to v2

Discovered while planning the metric implementation, before any system output,
candidate-selection outcome, or test-partition result had been generated or
inspected. Three distinct problems, only two of which were genuine gaps:

**Unspecified conventions.** v1 named its analysis window "hann" without fixing the
periodic or symmetric convention, and never stated an FFT normalisation. Each is a
real degree of freedom that changes the metric, so an independent implementation
working from v1 could not reproduce a v1 number. v2 defines the window by equation
and pins `norm = backward`.

**Machine-readable underspecification.** v1's config named the additive log term
`magnitude_floor` and did not encode the operation. The name implies clamping while
the v1 document had already fixed the additive form, so code reading the config alone
had to guess. v2 renames the field to `log_magnitude.offset` and states the operation
as a named kind, `amplitude_db_additive_offset`, which also removes v1's separate
`log_scale` number: a factor implied by the kind of measurement should not be a
setting.

**No change of numerical policy.** v2 preserves v1's additive `20*log10(|X| + 1e-8)`
exactly. Switching to a clamp was tempting and would have been wrong: an amendment
removes ambiguity, and altering an already-specified computation under cover of a
clarification is the behaviour this process exists to prevent. No threshold, seed,
partition size, or metric definition changed.

Pre-evaluation characterisation, for context on why these were treated as defects
rather than pedantry. On the fixtures examined, one clean 2-second frame from the
training corpus plus synthetic additive noise, window symmetry moved LSD by roughly
0.0015 dB and FFT normalisation by roughly 0.088 dB, with the `forward` convention
pushing a spectral bin below the offset. Those figures describe those fixtures. They
are not a bound on benchmark impact, and the noise fixture is a far worse restoration
than either system should produce.
