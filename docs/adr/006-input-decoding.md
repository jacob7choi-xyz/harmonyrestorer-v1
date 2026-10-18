# ADR-006: Decode Every Upload Once Before Validation and Inference

**Date:** 2026-10-18

**Status:** accepted

---

## Context

The upload route validated duration from metadata and passed the upload as received to the engines. Duration came from `sf.info` for WAV, FLAC and OGG and from `librosa.get_duration` for MP3, M4A and AAC, which returns soundfile's figure or, for files soundfile cannot open, the duration audioread reports. OpGAN read the upload with `sf.read`; UVR's audio-separator read it with `librosa.load`.

Experiments on 2026-10-18, on macOS arm64 and in the linux/amd64 image (emulated on Apple silicon), showed where that went wrong:

| Finding | Engine | How it was shown |
|---------|--------|------------------|
| A VBR MP3 without a Xing header read as 19.16 s of a 30.09 s file through soundfile and librosa | both | measured; through either engine, inferred from how they load audio |
| A 20-minute M4A whose header claimed 10 s passed the 600 s check in the image | both | measured with the route's check |
| 20-minute MP3s that read as 124 s and 279 s passed on both platforms | both | measured with the route's check |
| A direct UVR run on the 20-minute M4A in the image was killed out of memory in an 8.3 GB VM | UVR | measured |
| A 1.9 MB FLAC of 600 s at 96 kHz with 8 channels passed, and loading it for OpGAN raised the process's peak RSS from 292 to 2,442 MiB | OpGAN | measured on macOS |
| An M4A upload failed in `JobManager.process`, and AAC files failed to load, because `sf.read` cannot open either | OpGAN | measured |
| audioread's missing-binary and timeout errors subclass `DecodeError`, which the route mapped to 400 | both | read from `audioread/ffdec.py:36-54` |
| audioread sets duration to 0 when ffmpeg prints no duration line, which would pass a `duration > max` check | both | read from `audioread/ffdec.py:277-278`; no file found that triggers it |

## Decision

1. **Decode once, then use only the decoded file.** `services/audio_decode.py` decodes each accepted upload with ffmpeg into a 16-bit PCM WAV. The duration checked against the cap is that WAV's frame count, and the engines read only that WAV; the upload as received is deleted.
2. **Do not let ffmpeg probe the upload.** The demuxer is pinned from the extension and magic bytes the route has already validated (`demuxer_for`), only the file protocol is allowed, and for MP4 the demuxer's `enable_drefs` and `use_absolute_path` options are passed as off, although both already default to off in the image's ffmpeg 7.1.5. Output is limited to the first audio stream, without metadata or chapters, at the maximum duration plus one second and a matching byte cap. Without `-f`, ffmpeg 9.0.2 followed an ffconcat script named `.mp3` to a relative path and decoded the referenced file, even with `-protocol_whitelist file`; with the pinned demuxer it failed.
3. **Decode to the rate each engine reads at, and keep mono or stereo.** `engines.input_sample_rate` gives 16 kHz for OpGAN and audio-separator's default 44.1 kHz for UVR. Mono and stereo keep their channel count, and sources with more channels are mixed to stereo, because ffmpeg's channel mixing can change level: its mono-to-stereo upmix lowered the level by 3 dB, and pre-decoding a 16 kHz mono input that way changed UVR's output by -10.7 dB relative to the signal. With channels kept, outputs compared with the previous path differed at rounding level: OpGAN at most 26 LSB (RMS 0.64 LSB, -57.8 dB) on a 44.1 kHz stereo MP3 and at most 2 LSB on a 16 kHz mono WAV, and UVR at most 5 LSB (RMS 0.35 LSB) on the MP3 and at most 9 LSB (RMS 0.72 LSB) on a 16 kHz mono WAV, all on macOS.
4. **Fail closed.** No upload reaches inference without a decoded length checked against the cap. An upload that is too long, cut off at the length or size cap, or not decodable is answered 400. A missing ffmpeg or a decode past `DECODE_TIMEOUT_SECONDS` (default 60) is answered 500 and logged at ERROR. A 600 s MP3 or M4A decoded in 0.4 to 0.9 s in the runs measured.
5. **Keep the decode inside the request's resource accounting.** The decode runs in the threadpool while the upload holds its admission slot, and the byte reservation adds the decoded WAV at its size cap: 76,931,584 bytes for OpGAN and 846,966,032 for UVR at 600 s, excluding the upload. If the request is cancelled during the decode, the route waits for ffmpeg to stop and removes its output before releasing the slot and the reservation; a test covers this with a stand-in decode.
6. **Record the ffmpeg build; do not pin it yet.** The app logs the first line of `ffmpeg -version` at startup, and logs an error if ffmpeg cannot run. The security-relevant options are passed explicitly rather than relied on as defaults.

## Consequences

### Positive
- After the change, under a 25 s cap, the 30 s VBR MP3 was rejected; under the default cap it completed with OpGAN with a 30.10 s output (macOS).
- The 20-minute M4A claiming 10 s was rejected with 400, including over HTTP in the image with UVR, where the container was not killed.
- The 96 kHz 8-channel FLAC completed with OpGAN, with the test process peaking at 800 MiB RSS (macOS).
- M4A and AAC uploads completed with OpGAN (macOS) and with UVR (over HTTP in the image).
- UVR's stem length check now compares against a PCM WAV, where `sf.info` counts the frames written.

### Negative
- Each upload costs one ffmpeg decode, and the reservation per upload rises by the decoded size cap.
- ffmpeg's parsers still see every upload's bytes. The controls above limit which demuxer and protocol run; they do not remove the parser itself as attack surface.
- ADTS has no field to signal AAC encoder delay and padding, so they are decoded as audio and a decoded ADTS file runs slightly longer than its source (2.112 s for a 2 s file encoded at 16 kHz).

### Neutral
- `wav_io.read_duration_seconds` is no longer expected to reach its audioread branch from the upload route, because `decode_to_wav` has already opened the decoded WAV with `sf.info`. The branch is kept.
- Tests that need ffmpeg run the real binary, which CI installs. CI installs ffmpeg with apt on `ubuntu-latest` (`.github/workflows/ci.yml:20`, `:45`), so CI tests a different build from the image's.

## Open Gaps

- ffmpeg is not version-pinned (`Dockerfile:12`), and the base image `python:3.11-slim` is referenced by tag rather than digest. Pinning is left for the deployment reproducibility work.
- An MP4 external data reference was crafted and did not open its target with ffmpeg 9.0.2. In the image's 7.1.5 both reference options default to off; their defaults in other builds, and the crafted file against other builds, have not been checked.
- The HLS playlist named `.m4a` was not followed with or without `-f` in ffmpeg 9.0.2; its test guards against a build that probes it differently, rather than showing the pin is needed.
- `services/transcode.py`, which converts the server's own WAV for downloads, hardcodes a 120 s timeout and does not pass `-nostdin`. Its input is not an upload, so these controls were not applied there.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| Count decoded samples in the duration check only, and keep giving engines the upload | The check would pass a 30 s VBR MP3 at 30.09 s, but soundfile and librosa.load, which the engines read with, read 19.16 s of it, and UVR's stem check would compare against the same 19.16 s, so the truncation would likely stay silent. |
| Decode to 16 kHz mono for OpGAN and 44.1 kHz stereo for UVR | ffmpeg's channel mixing changes level, and the result depends on the output sample format; the mono-to-stereo upmix changed UVR's output by -10.7 dB on one input. |
| Decode at the source rate, capped at 48 kHz and two channels | Up to 115.2 MB per upload as 16-bit, or 230.4 MB as float, against 38.5 MB (OpGAN) and 106.1 MB (UVR) at the engines' rates. |
| Let ffmpeg probe the format | It followed an ffconcat script disguised as MP3 to a local file. |
| Pin the ffmpeg package version in the Dockerfile | An exact Debian pin may stop installing once a newer security update is published (not verified for this package), which would delay fixes; deferred to the deployment reproducibility work together with pinning the base image by digest. |
