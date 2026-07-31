# Benchmark protocol v4

Frozen before candidate identities are selected, downloaded for evaluation,
auditioned, or scored. The source collection has already been inspected in general
terms; what has not happened is any decision about which recordings enter the
benchmark.

> **Note:** v4 amends v3, which amended v2, which amended v1, all before any
> evaluation ran. The operational values live in `benchmark/protocols/v4.json`, which
> is normative; this document explains them and its numbers are illustrative.
> Superseded versions are preserved unmodified, as
> [benchmark-protocol-v1.md](benchmark-protocol-v1.md),
> [benchmark-protocol-v2.md](benchmark-protocol-v2.md), and
> [benchmark-protocol-v3.md](benchmark-protocol-v3.md) with their configs, so an
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
> `except MemoryError` cannot be the only path to `ENVIRONMENT_CAPACITY_FAILURE`. None
> of this is enforced today and the protocol does not claim it is.

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

Semantic validation and runtime conformance are separate responsibilities and are
separate functions. "Is this a valid v4 protocol" and "may this machine
authoritatively execute it" are different questions, and canonicalisation and the CI
conformance job call the environment check directly under its own name. The loader
calls it too, with no way to switch it off: `load_protocol()` is the authoritative
execution loader, not a neutral historical parser, and a parameter that could
weaken it would be the same hole as a loader that accepts a path.

**Reproducibility is claimed narrowly.** Pinning package versions does not guarantee
bit-identical native DSP output across architectures and builds. The claim is that
canonical artifacts generated by the pinned reference implementation are
authoritative and their digests identify those artifacts; other environments verify
against the frozen artifacts rather than being assumed to regenerate every bit. A
golden fixture in the test suite makes cross-platform agreement an observation rather
than an assumption.

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
comparator rather than leaving code to infer strictness from prose.

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
| Source-ineligible | recorded, published, run continues | the twelve codes below |
| Abort | run stops, publication blocked | `SOURCE_HASH_MISMATCH`, `GOLDEN_FIXTURE_FAILURE`, `ENVIRONMENT_MISMATCH`, `ENVIRONMENT_CAPACITY_FAILURE`, `UNEXPECTED_IO_FAILURE`, `INTERNAL_CANONICALISATION_ERROR` |

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
| Always | `logical_source_id`, `stage`, `primary_reason_code`, `identity_status`, `protocol_sha256`, `unevaluated_stages` |
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

- Metrics computed at 16 kHz mono, on waveforms produced by
  [Canonicalisation](#canonicalisation), which pins every step of getting there.
  Metric libraries frozen: `pesq`, `pystoi` 0.4.1, numpy 2.3.5, scipy 1.18.0,
  soundfile 0.14.0. Exact versions are recorded per run
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

Both systems receive an identical population, checked by identifier at two levels.
Track identifiers must be equal across systems and equal to the frozen expected set.
Within each track, frame identifiers must also be equal across systems and equal to
that track's frozen eligible set: equal counts are not sufficient, because two
systems can score the same number of different frames and produce per-track means
that are not comparable.

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
exist produces a new version, with the reason recorded, before evaluation runs. A
defect discovered after results exist invalidates those results rather than
retroactively changing the rules that produced them. Superseded versions stay in the
tree rather than being replaced, and each version names its predecessor by digest;
the loader recomputes those digests, so the chain is checked rather than asserted.

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
