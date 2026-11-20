# To Do

## Research Reading

- [ ] Re-read Ince et al. 2022 cover-to-cover: check if our training hyperparameters and loss functions match their recommendations
- [ ] Read Fonseca 2020 noise generation section: check if we're missing degradation types (wow/flutter, speed drift)
- [ ] Skim Moliner & Valimaki 2022: closest competitor, benchmark to beat, two-stage refinement idea
- [ ] Skim HiFi-GAN loss design: multi-scale discriminator + feature matching loss if our audio quality needs sharpening
- [ ] Skim Noise2Noise: technique for training on real historical recordings without clean references

## After Benchmark (see docs/benchmarks.md, whose earlier figures are withdrawn)

- [ ] Compare OpGAN and UVR on unseen recordings under the replacement protocol (docs/benchmark-protocol.md). An earlier head-to-head over the generated pair corpus (dc28a8b, 2026-03-19) was withdrawn on 2026-07-24; see the withdrawal notice in docs/benchmarks.md.
- [ ] Listen to restored samples side-by-side (OpGAN vs UVR vs noisy original). ADR 004 records listening notes on one real recording ("Out-of-Distribution Test").
- [x] OpGAN made the default denoiser (d0da515, 2026-03-19), after the head-to-head figures in dc28a8b. Those figures were withdrawn on 2026-07-24, and the replacement benchmark is in progress.
- [ ] If the replacement benchmark shows OpGAN does not beat UVR on unseen recordings: review the hyperparameters from Ince et al. 2022, consider longer training or loss function changes
- [ ] Test on out-of-distribution audio (live recordings, different genres, real-world noise)

## Benchmark Protocol

- [ ] Reconcile the scipy version in `docs/benchmark-protocol.md`, which lists scipy 1.18.0 among the frozen metric libraries. The project runs on Python 3.11, where `uv.lock` resolves scipy 1.17.1, because scipy 1.18.0 requires Python 3.12 or later. `benchmark/protocols/v4.json` does not pin scipy and `load_protocol` does not check it. Settle this through the amendment process before the replacement benchmark runs, together with the next amendment if one comes first.
- [ ] Settle these open questions in the next protocol amendment (v5). Each was found while implementing v4 canonicalisation, and the code cited takes the reading described until the amendment decides:
  - Golden fixture digests: v4 has no field for the expected digests of the golden fixtures, so they need an amendment before the fixtures are committed.
  - Unknown-length FLAC: a header whose total sample count is zero (unknown) was reported by libsndfile 1.2.2 as 9,223,372,036,854,775,807 frames, so the source is excluded as too long when it may be short (`benchmark/source_inspection.py`).
  - Subtype under an unadmitted container: v4 does not say whether the subtype is also established as unsupported; the code records the container reason only (`benchmark/source_inspection.py`).
  - FLAC/PCM_32: v4 admits it, but libsndfile 1.2.2 cannot write it, and reading it is untested.
  - NaN and infinity against the amplitude range: v4 is silent; the code counts NaN as non-finite only and an infinity as both non-finite and out of range (`benchmark/source_decode.py`).
  - A FLAC header that understates its total samples decodes that many and drops the rest without an error, and the record cannot show it (`benchmark/tests/test_source_decode.py`).
  - `EMPTY_SOURCE` looks unreachable for FLAC under libsndfile 1.2.2: a zero-frame write produced zero bytes, and a total of zero means unknown.
  - A failed canonical validation: v4 does not name its outcome; the code returns `INTERNAL_CANONICALISATION_ERROR`, an abort (`benchmark/canonicalisation.py`).
  - `docs/benchmark-protocol.md` says "the twelve codes below" for the source-ineligible vocabulary, which has 13 codes.

## Future Enhancements

- [ ] Try two-stage approach (coarse OpGAN denoise -> fine refinement) inspired by Moliner
- [ ] Borrow HiFi-GAN discriminator architecture if adversarial loss isn't producing sharp audio
- [ ] Explore Noise2Noise training on real historical recordings (no clean references needed)
- [x] Frontend tests
- [x] Duration validation for MP3/M4A/AAC on the backend (c6a2d3b, 2026-06-26)
