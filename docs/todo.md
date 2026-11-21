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
- [x] Freeze the benchmark degradation in a protocol amendment, before any candidate recording is acquired: protocol v7 freezes the port of `dataset/noise.py` at `d8993b2`, its ranges and seed, training level, one copy per frame, system execution through the shipped engines with float WAV input, the model digests, and pins for scipy, torch, and audio-separator, and corrects the v6 document's scipy statement.
- [ ] Build the benchmark runner that protocols v7 and v8 require. Done: the run preconditions in `benchmark/run_preconditions.py`, which run the golden fixtures, recompute the degradation golden (`GOLDEN_FIXTURE_FAILURE` on a mismatch), and compare the OpGAN checkpoint, the UVR model, and its two parameter files with their frozen digests (`ENVIRONMENT_MISMATCH` on a mismatch); and the canonicalisation stage in `benchmark/runner.py` and `benchmark/canonical_worker.py`, which checks those preconditions first and then runs each source in its own child process, one at a time, mapping a reported `MemoryError` or a `SIGKILL` to `ENVIRONMENT_CAPACITY_FAILURE` and any other worker death to `INTERNAL_CANONICALISATION_ERROR` naming the source, recording each worker's peak memory footprint (read by the parent, on macOS only), and recording a source above the byte bound as `SOURCE_EXCEEDS_MAX_BYTES`. Nothing calls `run_canonicalisation` yet, because candidate selection does not exist, so until something does the digests are recorded, not enforced. Still to build: the systems stage, which must build the UVR separator only after the preconditions have passed, from the same model directory they checked (audio-separator reads `AUDIO_SEPARATOR_MODEL_DIR` when it is set), so that a missing file aborts the run rather than being downloaded by audio-separator mid-run (it re-downloads the UVR parameter files from GitHub `main` whenever its model directory is cleared), with the files provisioned and pinned in a separate step; reading a canonical file back for the later stages, checking its digest again; and writing the run's records to disk. Open: whether to add a worker timeout, which the protocol permits without defining an abort code for it.

## Future Enhancements

- [ ] Try two-stage approach (coarse OpGAN denoise -> fine refinement) inspired by Moliner
- [ ] Borrow HiFi-GAN discriminator architecture if adversarial loss isn't producing sharp audio
- [ ] Explore Noise2Noise training on real historical recordings (no clean references needed)
- [x] Frontend tests
- [x] Duration validation for MP3/M4A/AAC on the backend (c6a2d3b, 2026-06-26)
