# Model v2, Phase 1: training noise

This document fixes, before any noise file is downloaded, which noise model v2 may train on, how candidate recordings are screened, how they are recorded in a manifest, and how they are split between training and validation. Pair generation, the validation split of the clean audio, compute, the evaluation rule and training each get their own document later.

> **Note:** When this document was written on 2026-11-26, no candidate noise file had been downloaded or listened to; the only Freesound audio this project had downloaded was the three evaluation clips below. For an earlier survey, 70 Freesound sound pages were read on 2026-11-24, and 38 of them are among the candidates found below; no other candidate's page had been read. The merge of this document is the timestamp for the rules below.

## Why

Starting model v2's training is the author's decision, taken on 2026-11-26. It does not follow from a rule: Phase 0's gate failed (`docs/model-v2-phase0.md`), and the gain-steps rule (`docs/model-v2-gain-steps.md`) did not call for training. What those measurements did show, on the six demo recordings:

- v1's mean SI-SNR improvement was lower on real 78 surface noise than on v1's synthetic noise added without the low-pass or saturation, on all six recordings (Phase 0).
- v1's mean SI-SNR improvement was 6.04 dB with the benchmark's low-pass and -4.85 dB without it (gain steps).

v1 was trained on synthetic noise only (`dataset/noise.py`). This phase adds real recorded surface noise to what v2 can train on. Whether that changes v2's results is not known until v2 is trained and evaluated under its own rule.

## Sources

| Source | What it is | Use |
|---|---|---|
| `synthetic` | v1's noise synthesis, `dataset/noise.py`, unchanged | Training and validation, as the control ingredient |
| `freesound_vinyl` | CC0 recordings of record surface noise from Freesound that pass the screening below | Training and validation, split by uploader |

### Deferred: 78-like synthesis

A synthesis shaped like 78 rpm shellac noise is not built in this phase. The only 78 rpm noise this project holds is either reserved for evaluation (the three Freesound clips below) or excluded from training (the Aalto set), so no source for its settings exists that is not an evaluation set. Real shellac noise for training could later come from the author's own recordings of 78 rpm run-in and run-out grooves, under its own document.

## Exclusions

1. **The Aalto gramophone set is not used to train model v2.** None of it is used, including the 929 discs that `benchmark/manifests/aalto_disc_split.json` labels as training noise. Every Aalto segment is cut from a Great 78 Project transfer, and the Great 78 Project offers its collection "for use in research, teaching, and private study only". `docs/model-v2-phase0.md` leaves room for a research-only arm whose model is never shipped; model v2 may be shipped, so it is not that arm.
2. **The three CC0 78 rpm clips of Phase 0 are evaluation only:** Freesound IDs 244676, 341761 and 481811. The manifest test refuses their IDs and SHA-256 digests.
3. **The other uploads of those three clips' uploaders are not used** (`hifijohn`, `nervousneal`, `notsawry`). This is a judgement: the same uploader may have used the same turntable, transfer chain or room, and the exclusion keeps that chain out of training as whole discs are kept out of the Aalto split.
4. **Nothing from the listening study is used:** none of the 10 selected recordings, nothing in its universe listing, and no other side of a disc in it. A candidate described as taken from the Internet Archive or the Great 78 Project is rejected (screening, criterion 5).
5. **No noise parameter is chosen by measuring or listening to evaluation noise.** That covers Phase 0's `great78` and `freesound` conditions and the clips above. Under the benchmark protocol's contamination rule (`docs/benchmark-protocol.md`), doing so would burn them as evaluation material.

## Screening

### Stage A: the description page

Each candidate's own Freesound page is read, and the candidate passes only if all of these hold:

1. The licence shown is CC0 (`creativecommons.org/publicdomain/zero/1.0`).
2. The duration is at least 10 seconds, and the sample rate is at least 16 kHz.
3. The title, tags or description say that it is a recording of a physical record (vinyl, shellac or acetate) being played.
4. Nothing on the page says that music, speech or another sound is audible, or that the noise was made or layered in software (for example a plugin, a synthesizer or "simulated").
5. Nothing on the page says that the audio was taken from another source, such as the Internet Archive, the Great 78 Project, a film or a commercial sample library.
6. The uploader is not excluded above, and the candidate is not one of the excluded IDs.

A page that says too little to decide criterion 3, 4 or 5 is rejected. Each decision is recorded with the date and a short summary in the screener's own words of what the page says; the uploader's description itself is not copied.

### Stage B: listening

After download, the author listens to every candidate that passed stage A, in full. A candidate is rejected if music, speech or any sound other than record surface noise and the playback chain's own hum or rumble is audible anywhere. Only candidate clips are listened to, never evaluation audio. This stage is one listener's judgement, and the manifest says so.

## Download

The search that found the candidates read Freesound's search result pages on 2026-11-26 (11 queries, CC0 and at least 10 seconds, up to 12 pages each). It found 339 distinct sounds, 713.2 minutes in all, from 244 uploaders, before any screening. That is an upper bound: the search was not exhaustive, and no candidate had been screened.

The original files need a logged-in Freesound account. The author downloads them through their own account, from a list of the stage A passes with each file's ID and size and the total size, after the total is stated. Files go into a new directory (`data/noise_freesound_cc0_train`) and are treated as untrusted: they are read only by code that does not run from that directory, with Python's isolated mode.

## Manifest

Each accepted clip has one manifest entry with: the Freesound ID, the page URL, the uploader, the title, the licence URL, the date the licence was checked, the stage A summary, the stage B decision, the file name, the byte count, the SHA-256 of the file, the format, the sample rate, the channel count and the duration. The manifest's SHA-256 is pinned in code, as the Aalto split's is, and the loader refuses any other manifest.

## Split

Clips are split by uploader, so that validation noise never comes from an uploader whose clips are in training. The uploaders are put in an order by a permutation seeded with 20261127 and gone through in that order. An uploader is taken into validation if adding their clips keeps validation at or below 20% of the accepted duration, and is skipped otherwise; the pass stops once validation holds at least 10%. Every uploader not taken into validation is training noise. If the pass ends with validation below 10%, the split is not used, and this document is amended before any training. The split is written into the manifest.

How much each source and each uploader contributes to a training pair is decided in the pair-generation document, after the accepted duration is known.

## Limits

- The real training noise is vinyl and other records that pass screening, not 78 rpm shellac. Its spectrum and clicks may differ from those of the 78 transfers that study 2 uses.
- Stage A depends on what uploaders wrote. Stage B is one listener.
- The candidate count is from search result pages only. The accepted duration is not known until screening ends, and it may be much smaller.
- Nothing here checks that training on this noise changes v2's results on real 78 transfers.
