# Blind listening study protocol (historical recordings)

This document explains the listening study that compares three versions of each recording: the original transfer, the OpGAN restoration, and the UVR restoration. The frozen values are in `benchmark/listening/protocols/v1.json`. This page describes them and gives the reason for each choice where a reason exists.

> **Note:** The protocol is frozen when this change is merged. No audio was downloaded before the merge. The listing used to build the universe read archive.org metadata only.

## What the study asks

Listeners hear pairs of short clips of the same performance, each clip processed in one of three ways. They say which clip they prefer overall, or that they have no preference. The study reports how often each version was preferred in each pair type. It does not name an overall winner, and it does not report significance tests.

## The source

The recordings come from the Internet Archive collection `georgeblood`, the 78 rpm transfers made by George Blood. The search is `collection:georgeblood AND date:[1900-01-01 TO 1922-12-31] AND subject:(classical)`, sorted by identifier. The date range is applied to the archive's date field, and it selects items dated from 1900 to 1922.

The listing was read on 2026-11-24. The count of listed items and the reasons for each exclusion are in `benchmark/listening/manifests/universe_v1.json`.

## The universe rule

An item enters the universe when all of the following hold.

1. Its metadata has an instrumentation line. This is the description text that is not a Performer or Writer label and not a digitisation or matrix note.
2. It has exactly one main file, an original 24-bit FLAC at the top level of the item.
3. Its instrumentation line matches a solo or string-ensemble pattern: `violin solo`, `violoncello solo`, `cello solo`, `viola solo`, `piano solo`, `pianoforte solo`, `harp solo`, `string quartet`, `string trio` or `string quintet`, ignoring case.

An item is then dropped if its instrumentation line or its title matches a vocal term, such as tenor, soprano, sung, song, chorus or duet, or a phrase such as `in italian`. It is also dropped if its instrumentation line or title matches `accordion`.

The rule is written as regular expressions in the protocol file. The code compiles them and applies them. The file records each kept item and each exclusion with the reason and the text that matched.

> **Note:** The rule is narrow. Recordings described as `violin with piano` or `violoncello; pianoforte` do not match a solo pattern, so they are excluded. Some solo recordings are excluded because of words in their title or work name, not because of their performance. The title check, for example, excludes the items titled `On Wings of Song` that carry a violin solo line. The listing counts in the last section show how many items each exclusion removed. This version does not change the rule.

## Selection

Ten recordings are selected. The walk takes the universe in a fixed order: the identifiers are sorted, then permuted with numpy's PCG64 generator seeded with 20261124. Each listed performer name is split on semicolons, on the whole word `and`, and on `&`. A name such as `Fritz Kreisler and Efrem Zimbalist` therefore gives two performers, and `Flonzaley Quartet` stays one. An item is taken when none of its performer parts has already been taken. Parts are compared after casefolding and whitespace collapse, so accents and punctuation still distinguish names.

If a taken item later fails to download, fails its size, md5 or sha1 check, or fails the canonicalisation of protocol v14, its choice is removed and the walk continues after the last item it considered. It does not start again from the beginning. The reason is recorded. This choice matters: a removed item frees its performer names, and a walk that started again would take an item it had already passed over. The rule and its test are in `benchmark/tests/test_listening_selection.py`.

## Clips

Each selected recording gives one clip of 8 seconds. The start is a whole number of seconds drawn from a generator seeded with the study seed and the SHA-256 of the identifier. The range is inclusive, from 15 seconds to the recording's duration in whole seconds minus 15 minus 8. A recording shorter than 38 seconds has no valid clip and is replaced.

The clip is cut from the same position in every condition, after the whole side has been processed.

> **Note:** The windows in the selection report use the duration the archive lists. The frozen rule uses the canonical sample count after decoding. A window drawn from the listed duration is provisional until the decoded duration is known.

## Conditions

Each clip is presented in three conditions.

- `original` is the canonical 16 kHz mono waveform from the protocol v14 recipe.
- `opgan` is the OpGAN service run on the whole canonical side.
- `uvr` is the UVR service run on the whole side, downmixed to mono and resampled to 44.1 kHz. Its output is downmixed to mono if needed and resampled to 16 kHz.

All three are presented at 16 kHz mono. Listeners never hear the 44.1 kHz output, only its 16 kHz version.

## Loudness

Each condition is measured with the ITU-R BS.1770-4 integrated loudness method on the 8-second window, and gained to an integrated loudness of -23 LKFS. If any condition's sample peak after gain would exceed -1 dBFS, the target for all three conditions of that clip is lowered together, in steps of 0.1 LU, until no condition exceeds the ceiling. No step size is specified elsewhere, so this protocol sets 0.1 LU. Every applied gain is recorded.

The presented files are 16-bit PCM WAV. Each sample is scaled by 2 to the power 15 and rounded. A window whose rounded samples fall outside the 16-bit range is refused, so no sample is clipped.

> **Note:** The sample peak is the largest absolute sample value. It is not the BS.1770 true peak, which also estimates the peak between samples.

## Trials

Each listener completes 30 scored trials and 2 catch trials. Each clip appears in three pairs: original against OpGAN, original against UVR, and OpGAN against UVR. That is 10 clips times 3 pairs, which gives 30. Each trial asks "Which sounds better overall?" with the answers A, B and No preference. An optional comment can be added to each trial.

A catch trial plays one condition on both sides. The answer a listener gives to an identical pair is the signal the catch rule uses, which is why a listener who answers A or B on both catch trials is excluded. The trial order and the side of each condition are drawn from a generator seeded with the study seed and the listener's index.

## Listeners and exclusions

Listeners are adults with self-reported normal hearing who listen on headphones. The author of the OpGAN model is not part of the scored panel. Listeners have pseudonymous identifiers, and no name or email address is stored with their answers.

A listener is excluded if they answer A or B on both catch trials, or if they leave any trial unanswered. Excluded listeners are counted and reported with the reason. Their answers are not analysed.

The consent text is one sentence, shown before the first trial.

## Stages

The pilot is the first complete listeners, up to 10. The full study runs if at least 15 listeners consent to take part in total, counting pilot listeners. The decision depends only on that count, never on any answer. Pilot answers are pooled with the full study only if nothing in the protocol or the stimuli changed between the stages.

> **Note:** The stage rule counts listeners who consent to take part, which this protocol treats as signing up. Counting consenting listeners rather than completed ones was decided on 2026-11-24, before the merge.

## Analysis

The analysis is descriptive. For each pair type it reports the counts and shares of judgments preferring the first condition, the second condition, and no preference. It reports the counts per listener and per clip. When at least 10 listeners are included, it reports a 95% percentile interval from a bootstrap that resamples listeners (10000 iterations, seed 20261124), labelled as descriptive.

The analysis reports no p-values, uses no significance language, and names no overall winner.

## Publication

The results are published. The Great 78 audio is not published.

## Overlap with the training recordings

The raw FLAC digests of the candidates are compared with the 145 training recordings listed in `benchmark/manifests/training_corpus.json`. Their decoded PCM digests are compared with `benchmark/manifests/training_decoded_pcm.json`.

The MFCC similarity screen used in protocol v14 is skipped. The reasoning is this. If the training recordings are performances from after 1922, a recording from before 1923 cannot be a copy of one of them, so an MFCC match would not show copying. This is reasoning and not a measurement, and the protocol does not check the training dates. The screen has also not been calibrated on noisy audio.

## What is not in this change

- No audio is downloaded or processed.
- The loudness measurement, the condition generation and the trial scheduling are specified in the protocol but not implemented.
- The scored panel does not exist yet, so no result exists.

## Listing counts

The listing read on 2026-11-24 returned 297 items. Each identifier appeared once, and the search total was 297. The rule kept 85 items. The exclusions were 94 for not being a solo or string-ensemble recording, 79 for a vocal term, 36 for having no instrumentation line, 2 for an accordion term, and 1 for not having exactly one main FLAC. The file `benchmark/listening/manifests/universe_v1.json` records each decision.

Measured on the metadata saved at 06:20 UTC on 2026-11-24, which is not stored in this repository. Its 297 decisions match the pinned manifest item for item:

- Without the accordion term, 86 items would be kept. The accordion term removes one item that would otherwise be kept, `Il Trovatore` (`ACCORDION-PIANO SOLO`). The other accordion item, `Stradella Overture` (`Accordion solo`), has no keep pattern, so the solo rule would exclude it in any case.
- Five items have a solo or string keep pattern on the instrumentation line and are excluded only because a vocal word appears in the title: `Chorus of Dervishes`, `Evening Song`, two items titled `On Wings of Song`, and `Spinning Song (Song Without Words, No. 34)`.
- Twenty-three items have the word solo in the instrumentation line and are not kept. Some carry a vocal or accordion term. The others, such as `Cornet Solo` or the misspelt `Violincello Solo`, have no keep pattern.

> **Note:** The split rule is a choice made for this protocol. Names are split on `;`, on `and` and on `&`, as the protocol's `performer_split_patterns` list states. Two items that share a performer after the split cannot both be taken. The universe manifest keeps each listed name as the archive gives it, so the split is applied in the selection, not stored in the manifest.
>
> **Note:** Two kept items have a creator field that is a list. The metadata reader joins such a list with ` | `, so their performer is recorded as `Brahms | Joachim` and `Chopin | Sarasate` instead of two names. The split rule does not split on `|`, so these two names stay joined. Adding `|` to the split would not change the ten selected items, as measured, but it would change the rule the protocol freezes, so the protocol leaves it as is. The two items were kept with their names joined, a decision made on 2026-11-24 before the merge.
