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

- [x] Reconcile the scipy version in `docs/benchmark-protocol.md`: protocol v5 corrects it to 1.17.1, which `uv.lock` resolves for Python 3.11. v5 does not enforce it at load, since the benchmark code does not import scipy.
- [x] Settle the open questions found while implementing v4 canonicalisation: protocol v5 (`benchmark/protocols/v5.json`) freezes the golden fixtures, the frame identifier, and the RMS reduction, states the v4 implementation's readings as rules, and corrects the v4 document. The "v4 to v5" section of `docs/benchmark-protocol.md` lists each item.
- [x] Amend the protocol to name eligibility outcomes, before any candidate recording is acquired: protocol v6 (`benchmark/protocols/v6.json`) names the recording and frame eligibility codes and their precedence. `benchmark/eligibility.py` still returns measured facts; mapping them to codes belongs to the code that records outcomes, which does not exist yet.
- [ ] Freeze the benchmark degradation in a protocol amendment, before any candidate recording is acquired. v6 defers it: the process that generated the training pairs (most likely `dataset/noise.py` at `d8993b2`, whose hiss was quieter than recorded) has to be ported, seeded, and checked against the training pairs first. The same amendment decides how systems receive input and return output: the shipped engines or the bare models, the input format, the level matching, and one degraded copy per frame. `benchmark/degradation.py` (the port, not yet called by a run) imports `scipy.signal`, so the amendment also has to pin scipy, which `pyproject.toml` admits as `>=1.10.0` and `uv.lock` resolves to 1.17.1, and correct the v6 document's statement that no benchmark module outside the tests imports scipy. Its parameter ranges are the constant `D8993B2_RANGES` in that module until the amendment moves them into the protocol file.

## Future Enhancements

- [ ] Try two-stage approach (coarse OpGAN denoise -> fine refinement) inspired by Moliner
- [ ] Borrow HiFi-GAN discriminator architecture if adversarial loss isn't producing sharp audio
- [ ] Explore Noise2Noise training on real historical recordings (no clean references needed)
- [x] Frontend tests
- [x] Duration validation for MP3/M4A/AAC on the backend (c6a2d3b, 2026-06-26)
