# ADR-005: Denoiser Contract

**Date:** 2026-10-17

**Status:** accepted

---

## Context

The backend runs one of two denoising engines, chosen with `DENOISER_ENGINE`: OpGAN, the default, and UVR-DeNoise through audio-separator 0.44.2. Before this change, the code shared by both engines was sized and documented for 16 kHz mono 16-bit output. The upload reservation was the input plus two such files at the maximum accepted duration, and the `download_audio` and `transcode_output` docstrings stated that format. `JobManager` typed its denoiser as a union of the two service classes and chose one with an if/else on the engine string.

UVR did not meet those assumptions. The evidence, gathered on 2026-10-16 and 2026-10-17:

- In real runs on macOS arm64, and in one linux/amd64 image run with FLOAT input, UVR stems were 44.1 kHz stereo. PCM_24 and PCM_32 inputs gave PCM_24 and PCM_32 stems, and FLOAT and PCM_16 inputs gave PCM_16.
- The VR separator writes and returns both stems when `output_single_stem` is unset (`vr_separator.py:214-247`). In a direct run of the service, it returned the denoised stem and left the other in its output directory; in the backend that is `processed_dir`, which job cleanup does not remove.
- Read from pydub 0.25.1's code, each stem export writes an int16 staging file and an ffmpeg output file to the system temp directory, then copies the output to the stem path (`audio_segment.py:875-981`). The temp-directory sizes measured in the run below fit that.
- On one 65.65 s PCM_32 input on macOS arm64, 81,056,990 bytes were present at once across the output and temp directories. The reservation for 600 s, not counting the input, was 38,402,048 bytes.

## Decision

1. **Contract.** `services/engines.py` declares a `Denoiser` Protocol with one method, `denoise(Path) -> Path`, the only method the job pipeline calls. An engine is expected to return a 16 kHz mono 16-bit WAV inside its output directory and to remove the other files it writes there. This is not checked at runtime.
2. **Engine selection.** `DenoiserEngine` is a StrEnum in `config.py`, validated by exact value. `build_denoiser` uses a `match` with `assert_never`. In scratch copies, mypy reported an enum member with no branch and a service whose `denoise` signature no longer matched the Protocol.
3. **Per-engine reservation.** The upload handler reserves the input plus `peak_output_bytes(engine, max_duration)`:

   | Engine | Formula, for d seconds | At 600 s |
   |--------|------------------------|----------|
   | OpGAN | 2 x (d x 16,000 x 2 + 1,024): the intermediate and final 16 kHz mono 16-bit WAVs | 38,402,048 (unchanged) |
   | UVR | 3 x (d x 352,800 + 1,024) + d x 176,400 + 1,024: two stems and the ffmpeg output at up to 44.1 kHz stereo 32-bit, plus the int16 staging file | 740,884,096 |

   > **Note:** Updated 2026-10-18. The UVR row was derived for audio-separator's pydub writer. The service now uses its soundfile writer (see Update below), which peaked at a sampled 46,318,168 bytes on the same 65.65 s PCM_32 input; the figure is kept as a conservative bound.

   The UVR figure counts temp-directory files that the artifact budget scan does not see, because if the temp directory is on the same memory-backed filesystem, they use the same memory (see Negative below). The reservation uses the maximum accepted duration, as before, rather than the validated duration of the upload.
4. **One served format.** The UVR service reads the denoised stem as mono at 16 kHz, writes it atomically, and removes the raw stems, so both engines serve the same format. The read and the atomic write live in `services/wav_io.py`, shared with OpGAN.

### Non-goals

- Loading models or engine code at runtime from packages or paths.
- Choosing the engine per request. The engine stays a deployment setting.
- Accepting a model name or path outside an allowlist. The `UVR_MODEL_NAME` allowlist in `config.py` is unchanged.
- Changing `benchmark/`. Protocol v4 is frozen and names the two systems (`opgan_minus_uvr` in `benchmark/protocol.py` and `v4.json`), and `docs/benchmark-protocol.md:167-169` says a model changed in response to the test results needs a fresh partition.

## Consequences

### Positive
- The UVR reservation covers the peak measured on one 65.65 s input, and a test ties it to that measurement.
- Both engines serve 16 kHz mono 16-bit WAV, which is what the download and transcode paths describe. For UVR this was checked in one macOS run with the real library.
- UVR no longer leaves its second stem in `processed_dir`.
- A new engine needs an enum member, a `build_denoiser` branch, and a `peak_output_bytes` branch. In a scratch copy with an extra enum member, mypy flagged both functions.

### Negative
- A UVR upload reserves its own size plus 846,966,032 bytes at the 600 s cap: the 740,884,096-byte peak above and, since ADR 006, the decoded WAV at its 106,081,936-byte size cap. With the default 1.5 GiB budget, it is refused with 503 when persisted and reserved bytes exceed 763,646,704 minus the upload's size.
- The UVR figure is extrapolated from the output formats and one 65.65 s measurement, not measured at 600 s. It assumes stems are no longer than the input, as they were in every run here.
- Counting temp-directory files assumes `/tmp` shares the memory-backed filesystem. `scripts/deploy_backend.sh` states that the Cloud Run filesystem is memory-backed; `/tmp` specifically has not been verified.

### Neutral
- `16_000` appears in three places: `engines.py` (OpGAN's bytes per second), OpGAN's `_TARGET_SR`, and UVR's `_OUTPUT_SR`. The first two describe OpGAN's native rate, and the third the served rate; they are kept apart on purpose.
- The invalid-engine error and the denoiser initialization log line changed wording.

## Update 2026-10-18

Follow-up changes closed two of the gaps first listed here (leftover stems and pydub's temp files), narrowed a third (platforms), and corrected how the leftover-stem gap was described.

- **Leftover stems.** The first version of this ADR said stems were left "from a `separate()` call that raises". In fact, when an exception from a stem write reaches `Separator.separate()`, it is logged and `separate()` returns without the stems already written (`separator.py:978-987`), and an `OSError` from a stem export is caught earlier and that stem's path is still returned (`common_separator.py:388-390`). The Separator now writes into `processed_dir/.uvr-staging`, which the service empties before and after each job under a lock. The artifact budget scan now counts files in subdirectories, without following symlinks, and skips a file removed while it is scanning.
- **Truncated stems.** A stem cut short by a failed write reads as valid audio with fewer frames. The service now compares the denoised stem's length with the input's decoded length and fails the job when they differ by more than 0.1 s or the stem is missing. In the runs measured on macOS arm64 (2 s to 600 s; WAV, FLAC, MP3, M4A, AAC and OGG inputs), stems were 0 to 9.43 ms shorter than the input, and in the linux/amd64 image (2 s and 65 s inputs) about 1 to 9 ms shorter than the decoded input. The input is measured as `librosa.load` reads it, because the duration audioread reports can be an estimate: an ADTS AAC file reported 66.5 s and decoded to 65.69 s in the image.
- **pydub staging files.** The Separator is built with `use_soundfile=True`, so stems are written with `sf.write`; in the soundfile-writer runs measured, the temp directory stayed empty. In an earlier comparison over 12 inputs, the converted 16 kHz output was within 2 LSB of the pydub writer's, with the same length.
- **Platforms.** The UVR path was rerun in the linux/amd64 image on Apple silicon, which is emulated, not native and not Cloud Run: direct runs on WAV, MP3, M4A and AAC inputs, and HTTP uploads of a WAV and an AAC file with `DENOISER_ENGINE=uvr`. Each produced 16 kHz mono PCM_16 output, and afterwards no stem or temp file remained.

## Open Gaps

> **Note:** `DENOISER_ENGINE=uvr` has not been validated against the deployment memory budget in `scripts/deploy_backend.sh` (runtime peak plus `MAX_ARTIFACT_BYTES` plus a 1 GiB reserve within 4 GiB). UVR's peak RSS was 2,043.9 MiB in one image run that processed a 2 s and a 65.65 s input, and its model file, about 127 MB, is downloaded to `/tmp/audio-separator-models/`. Neither has been checked against that budget.

- The UVR path has not run natively on linux/amd64 or on Cloud Run; the image runs above were emulated.
- The soundfile writer catches any error from `sf.write` and logs it instead of raising (`common_separator.py:448-451`). For the denoised stem, the length and existence checks are what turn such a failure into a failed job.
- The UVR reservation still uses the pydub-writer figure. Lowering it toward the soundfile writer's peak would need its own measurement.
- `output_single_stem` could avoid writing the unused stem. It is not adopted, pending a real run that shows the denoised output is unchanged.
- Input decoding, outside this ADR's contract: a VBR MP3 without a Xing header read as 19.16 s of a 30.09 s file through soundfile and librosa.load, so, from how both engines load audio, each would process only part of it (not run through either engine); with the default engine, an M4A upload failed in a `JobManager.process` run and AAC files failed to load, because OpGAN reads input with soundfile, which cannot open either format; and the upload route's 600 s cap uses the duration audioread reports for M4A and AAC.

> **Note:** Updated 2026-10-18. The input-decoding issues in the last item are addressed by ADR 006, which decodes every upload once before validation and inference.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| A registry dict of engine builders | A `match` with `assert_never` gives mypy a coverage check without a runtime table. |
| Asking audio-separator for 16 kHz output with its `sample_rate` option | The option is the write rate (`common_separator.py:343`, `:448`) and the load rate in the common mix preparation (`:253`), while the VR path loads at its own band rates (`vr_separator.py:273`). Its default writer is fixed to two channels, so the option alone would not give mono. |
| Keeping UVR at 44.1 kHz stereo and letting each engine declare its served format | The download and transcode paths and the persisted-size reasoning in `scripts/deploy_backend.sh` would need per-engine handling. One served format keeps them as they are. |
| Reserving from the validated duration of each upload | Tighter, but a change in behavior. The maximum duration is the existing, more conservative rule. |
| Loading engines as runtime plugins | Widens the surface for deserializing untrusted weights and running unreviewed code. |
