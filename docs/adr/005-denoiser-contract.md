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

   The UVR figure counts temp-directory files that the artifact budget scan does not see, because if the temp directory is on the same memory-backed filesystem, they use the same memory (see Negative below). The reservation uses the maximum accepted duration, as before, rather than the validated duration of the upload.
4. **One served format.** The UVR service reads the denoised stem as mono at 16 kHz, writes it atomically, and removes the raw stems, so both engines serve the same format. The read and the atomic write live in `services/wav_io.py`, shared with OpGAN.

### Non-goals

- Loading models or engine code at runtime from packages or paths.
- Choosing the engine per request. The engine stays a deployment setting.
- Accepting a model name or path outside an allowlist. The `UVR_MODEL_NAME` allowlist in `config.py` is unchanged.
- Changing `benchmark/`. Protocol v4 is frozen and names the two systems (`opgan_minus_uvr` in `benchmark/protocol.py` and `v4.json`), and `docs/benchmark-protocol.md:167-169` says a changed model needs a fresh partition.

## Consequences

### Positive
- The UVR reservation covers the peak measured on one 65.65 s input, and a test ties it to that measurement.
- Both engines serve 16 kHz mono 16-bit WAV, which is what the download and transcode paths describe. For UVR this was checked in one macOS run with the real library.
- UVR no longer leaves its second stem in `processed_dir`.
- A new engine needs an enum member, a `build_denoiser` branch, and a `peak_output_bytes` branch. In a scratch copy with an extra enum member, mypy flagged both functions.

### Negative
- A UVR upload reserves about 741 MB at the 600 s cap. With the default 1.5 GiB budget, it is refused with 503 when persisted and reserved bytes exceed 869,728,640 minus the upload's size.
- The UVR figure is extrapolated from the output formats and one 65.65 s measurement, not measured at 600 s. It assumes stems are no longer than the input, as they were in every run here.
- Counting temp-directory files assumes `/tmp` shares the memory-backed filesystem. `scripts/deploy_backend.sh` states that the Cloud Run filesystem is memory-backed; `/tmp` specifically has not been verified.

### Neutral
- `16_000` appears in three places: `engines.py` (OpGAN's bytes per second), OpGAN's `_TARGET_SR`, and UVR's `_OUTPUT_SR`. The first two describe OpGAN's native rate, and the third the served rate; they are kept apart on purpose.
- The invalid-engine error and the denoiser initialization log line changed wording.

## Open Gaps

> **Note:** `DENOISER_ENGINE=uvr` has not been validated against the deployment memory budget in `scripts/deploy_backend.sh` (runtime peak plus `MAX_ARTIFACT_BYTES` plus a 1 GiB reserve within 4 GiB). UVR's peak RSS was 2,043.9 MiB in one image run that processed a 2 s and a 65.65 s input, and its model file, about 127 MB, is downloaded to `/tmp/audio-separator-models/`. Neither has been checked against that budget.

- The UVR changes ran on macOS arm64 only, not in the linux/amd64 image and not through the HTTP path.
- Stems from a `separate()` call that raises are not known to the service and are not removed.
- If ffmpeg fails, pydub raises (`audio_segment.py:969-970`) before deleting its staging files (`audio_segment.py:980-981`), so, read from that code, they stay in the temp directory.
- `output_single_stem` could avoid writing the unused stem. It is not adopted, pending a real run that shows the denoised output is unchanged.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| A registry dict of engine builders | A `match` with `assert_never` gives mypy a coverage check without a runtime table. |
| Asking audio-separator for 16 kHz output with its `sample_rate` option | The option is the write rate (`common_separator.py:343`, `:448`) and the load rate in the common mix preparation (`:253`), while the VR path loads at its own band rates (`vr_separator.py:273`). Its default writer is fixed to two channels, so the option alone would not give mono. |
| Keeping UVR at 44.1 kHz stereo and letting each engine declare its served format | The download and transcode paths and the persisted-size reasoning in `scripts/deploy_backend.sh` would need per-engine handling. One served format keeps them as they are. |
| Reserving from the validated duration of each upload | Tighter, but a change in behavior. The maximum duration is the existing, more conservative rule. |
| Loading engines as runtime plugins | Widens the surface for deserializing untrusted weights and running unreviewed code. |
