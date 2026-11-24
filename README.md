# HarmonyRestorer v1

A web app built to restore analog-era classical recordings (tape hiss, vinyl crackle, hum, high-frequency loss). It runs a custom OpGAN at 16 kHz mono, chosen with historical transfers in mind. The model was trained and benchmarked on synthetic versions of these artifacts; it has not yet been tested on real analog transfers. You upload a recording, the backend runs it through the OpGAN, a 1D Operational GAN trained from scratch on classical recordings, and you get the restored audio back to compare and download.

- **Model**: a custom-trained OpGAN, based on [Ince et al. 2022](https://arxiv.org/abs/2212.14618). UVR-DeNoise, through [audio-separator](https://github.com/karaokenerds/python-audio-separator), can be selected instead with `DENOISER_ENGINE=uvr`.
- **Formats**: WAV, MP3, FLAC, OGG, M4A and AAC in; WAV, MP3, FLAC, OGG and M4A out. Restored audio is 16 kHz mono, the model's native rate.
- **Stack**: FastAPI and PyTorch on the backend, React and TypeScript on the frontend.
- **Tests**: backend, dataset, benchmark, script and frontend suites, run by the [quality gate](#quality-gate).

## Try it

The site is at [harmonyrestorer.online](https://harmonyrestorer.online). Its backend runs as a single AWS Lambda function (see [Deployment](#deployment)); on 2026-11-08 a deploy's smoke test sent it a one-second test tone, the job completed, and its result downloaded as WAV. Running that way, it has a few limits:

- Uploads up to 4 MB and 10 minutes. 4 MB holds about 24 seconds of CD-quality WAV or about 4.4 minutes of 128 kbps MP3; a 10-minute recording fits only at about 56 kbps or less.
- One restore at a time. While one runs, another upload fails (going by the code, the page shows "Upload failed (429)"), and a page waiting on a status check keeps retrying for up to 5 minutes before it reports a timeout.
- A request that needs a fresh environment waits for the backend to start. In four such starts on 2026-11-07, all of an image that had already run once, the first request took about 16 seconds twice, and startup took about 6 seconds twice. The first start of a newly deployed image took about 78 seconds, which is why each deploy warms the new version before visitors reach it. Starts after a quiet period have not been measured on their own.
- Results can be downloaded for 20 minutes. One can be lost sooner: AWS replaces the backend's environment every few hours, according to its documentation, and may do so after a quiet period; each deploy also starts a new one. A lost result returns 404.

Run locally, the defaults are 50 MB and 10 minutes, and jobs are kept for an hour.

## Quick Start

```bash
git clone https://github.com/jacob7choi-xyz/harmonyrestorer-v1.git
cd harmonyrestorer-v1

# Copy env files before running anything
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local

# Every dependency group and extra, the environment the quality gate checks
uv sync --all-groups --all-extras

# The trained OpGAN checkpoint, from its GitHub release, checked against its recorded digest
uv run python scripts/fetch_checkpoint.py

# Backend
cd backend && uv run uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm ci && npm run dev
```

In development the frontend sends `/api/v1` requests to its own dev server, which forwards them to `localhost:8000`. To use another backend, set `VITE_API_URL` to its base URL ending in `/api/v1`, for example `http://localhost:8000/api/v1`. The upload limit comes from `VITE_MAX_UPLOAD_MB` (50 when unset). The dev server reads both when it starts, and a production build when it is built.

## API

```bash
# Upload. Processing runs inside this request, so the response carries the job's final
# status, completed or failed.
curl -X POST http://localhost:8000/api/v1/denoise -F "file=@audio.wav"

# Check status
curl http://localhost:8000/api/v1/status/{job_id}

# Download the result (format: wav, mp3, flac, ogg, or m4a; default wav)
curl -O "http://localhost:8000/api/v1/download/{job_id}?format=mp3"
```

The status endpoint is there for clients that poll. The format choice changes only the container and encoding, and downloads are named after the uploaded file, reduced to safe characters, with a `_restored` suffix.

> **Note:** Interactive docs are disabled by default. Set `ENABLE_DOCS=true` in `.env` to enable `/api/docs` and `/api/redoc` for local development.

## Architecture

```
backend/app/
├── main.py              # App factory, middleware, lifespan
├── composition.py       # Composition root: builds the configured adapters (ADR 010)
├── config.py            # Settings (env vars, validated at startup)
├── schemas.py           # Pydantic models, JobStatusEnum
├── exceptions.py        # Error handlers (no stack traces to users)
├── middleware.py        # Per-client rate limits on uploads and downloads
├── client_identity.py   # Which client a request comes from, for the rate limits and job caps
├── routes/
│   ├── health.py        # GET /, GET /health
│   └── denoise.py       # Upload, status, download
├── models/
│   ├── op_gan.py        # OpGAN generator + discriminator
│   ├── self_onn.py      # Self-ONN layers
│   └── chunking.py      # Overlap-add chunking for arbitrary-length audio
└── services/
    ├── audio_decode.py    # Decodes every upload once with ffmpeg, with length and size caps
    ├── engines.py         # Denoiser contract, engine selection, per-engine peak output bytes
    ├── opgan_denoiser.py  # OpGAN inference service (default)
    ├── denoiser.py        # UVR wrapper, lazy model loading (fallback)
    ├── wav_io.py          # Mono read and atomic WAV write shared by both services
    ├── admission.py       # Fail-fast inference admission (one at a time)
    ├── rate_limit.py      # RateLimiter contract, in-memory adapter, builders
    ├── job_store.py       # JobStore contract (job records, caps), in-memory adapter, builder
    ├── file_store.py      # FileStore contract (result files), local-disk adapter, builder
    ├── transcode.py       # Download format conversion, own admission slot
    ├── artifacts.py       # Atomic byte budget for stored files
    └── jobs.py            # Runs the denoiser in a worker thread; download guard and cleanup

frontend/src/
├── main.tsx             # Vite entry point
├── App.tsx              # Wizard orchestrator (upload/processing/complete), page-wide drop
├── config.ts            # Upload limit from the build setting
├── fileValidation.ts    # Type, size and duration checks before a file is selected
├── types.ts             # Shared types
├── api/client.ts        # Backend API calls + polling
├── hooks/               # useAudioDecoder, useAudioPlayback, useCrossfadePlayback
└── components/          # TapeStrip, UploadArea, AudioPlayer, WaveformCanvas, ErrorBoundary

ops/kill_switch/         # Closes the public function URL when the cost alarm fires
scripts/                 # Quality gates, checkpoint download, image check, Lambda deploy
```

## Deployment

The hosted backend is one AWS Lambda function in us-east-1, running the same container image CI builds and tests. The decision, its costs and limits, and the checks run before the site used it are in [ADR 009](docs/adr/009-pay-per-use-hosting.md); the setup, step by step, is in the [runbook](docs/aws-stage-1-runbook.md).

- **One environment.** Reserved concurrency is 1, so job records and rate limits, which are held in memory, are held by one environment at a time. This is also why only one restore runs at a time.
- **Deploys from CI, after review.** After CI passes on `main`, a deploy job copies the tested image to Amazon ECR by digest, publishes a new version, warms it, moves the `live` alias to it, and runs a smoke test that moves the alias back if it fails. The job waits for a reviewer's approval and signs in to AWS with a short-lived OIDC token, so no AWS key is stored in the repository. The first such deploy, on 2026-11-08, made version 5 live. Moving back to a version whose image a cleanup rule has deleted (ECR keeps the 5 most recent) was not tested.
- **Cost limits.** If the function runs for more than 15 minutes in an hour, an alarm triggers a small function that closes the public URL until someone reopens it. This was tested by setting the alarm by hand, not by a real overrun. A monthly budget sends email as spending grows; it does not stop spending.
- **Frontend.** The site is on Vercel, and its build settings point it at the function URL and set its upload limit to 4 MB.

An earlier deployment on Google Cloud Run is deferred, not removed; `scripts/deploy_backend.sh` still describes it.

## Security

How to report a vulnerability, and notes on the supply chain, are in [SECURITY.md](SECURITY.md). The controls in the code, in short:

**Uploads**
- Size and duration limits (50 MB and 10 minutes by default; 4 MB on the hosted site). The duration checked is the decoded length, so a file whose header claims less is still caught.
- Magic-byte validation: the content must match the declared format.
- Each upload is decoded once with ffmpeg into a WAV the engines read, with the demuxer pinned from the validated format, only the file protocol allowed, and the output capped by length and size. The ffmpeg build is logged at startup.
- Files are stored under generated UUIDs, never client filenames, and job IDs are validated as UUIDs before any storage access.

**Load and abuse**
- Per-client rate limits: 10 uploads a minute and 60 downloads an hour by default.
- Per-client and global job caps (`MAX_JOBS_PER_IP`, `MAX_TOTAL_JOBS`), with expired jobs cleaned up.
- One inference at a time. A busy upload is refused at once with 503 and `Retry-After`, and the slot belongs to the work itself, so a cancelled request cannot free it early. Download conversions have their own one-at-a-time slot, and a file being downloaded is protected from cleanup while it is downloaded, for up to 5 minutes by default.
- A byte budget for stored files, reserved before the work starts, so concurrent jobs cannot together exceed it as long as each job's worst-case estimate holds.
- On the hosted site, the client is identified from the source address in the request context the Lambda Web Adapter passes to the app. In one check on 2026-11-07 (ADR 009, precondition 1), a forged `X-Forwarded-For` or request-context header did not change it; the same check logged two download requests it could not explain. Whether hosted requests carry `X-Forwarded-For` at all was not checked.

**Responses and configuration**
- Errors are sanitized; no stack traces reach users.
- CORS allows only the origins in `CORS_ORIGINS` (`http://localhost:3000` by default), and only GET and POST. On AWS the function URL allows only the site's origin, and the app adds no CORS headers of its own.
- The OpenAPI schema and docs pages are off unless `ENABLE_DOCS=true`.
- Numeric limits, the denoiser engine, the UVR model, and the log settings are validated at startup, so a bad value stops the app instead of degrading it quietly. The upload and output directories, CORS origins, `ENABLE_DOCS` and the checkpoint path are read without that validation.
- Result files are written to a temporary file, synced, and renamed into place, so a crash does not leave a partly written result, though the temporary file can remain.

**Docker Compose setup**
- nginx sets CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy and HSTS (HSTS only takes effect when TLS ends upstream), limits `/health` to private addresses, and caps request bodies at 50 MB.
- Uvicorn trusts forwarded client addresses only from the pinned internal Docker subnet.

## Docker

```bash
# Required on first run: env files must exist before docker compose reads them
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local

# The backend image copies the checkpoint when it is built
uv run python scripts/fetch_checkpoint.py

docker compose up --build    # Start full stack
docker compose down          # Stop
```

After startup, the app is accessible at `http://localhost:3000`. The backend is not exposed directly to the host; all traffic routes through the frontend's nginx reverse proxy.

Both images use non-root users. The backend includes a health check and has resource limits in `docker-compose.yml`. `FORWARDED_ALLOW_IPS` controls which proxy sources Uvicorn trusts for forwarded client IP headers. Leave it at the safe default unless the deployment platform's trusted proxy source has been verified; never use `*` in production.

## Quality Gate

```bash
# Python (run from repo root). CI runs this same script.
scripts/gate_python.sh

# Frontend (run from repo root after npm ci in frontend/). CI runs this same script.
scripts/gate_frontend.sh
```

## Dataset Pipeline

The `dataset/` package builds training data for the OpGAN from public domain classical recordings.

```bash
# The Quick Start already installs the dataset extra; this is the same command
uv sync --all-groups --all-extras

# Download from Internet Archive (Musopen collection)
uv run python -m dataset.acquire --output data/raw/ --max-tracks 200 --formats .flac

# Preprocess to 16kHz mono, 2-second frames
uv run python -m dataset.preprocess data/raw/ --output data/clean_frames/

# Generate noisy/clean training pairs (5 variants per frame, seeded for reproducibility)
uv run python -m dataset.generate_pairs data/clean_frames/ --output data/pairs/ --seed 42

# Train OpGAN (CUDA GPU recommended; 100 epochs took about 51 hours on a T4)
uv run python -m dataset.train --pairs data/pairs/ --epochs 100 --batch-size 16
```

Current dataset: 145 tracks from 14 composers, 29,240 clean frames, 146,200 training pairs. Source audio is CC/public domain; noise is synthetically generated (tape hiss, vinyl crackle, mains hum, HF rolloff, tape saturation).

## OpGAN

From-scratch implementation of **1D Operational GANs** based on [Ince et al. 2022](https://arxiv.org/abs/2212.14618), living in `backend/app/models/`. Training script at `dataset/train.py` supports CUDA mixed precision, checkpoint resume, and gradient health monitoring.

### Benchmark Results

On solo piano and chamber recordings that are not in OpGAN's training set, degraded by OpGAN's own synthetic training noise and scored at 16 kHz mono, OpGAN's outputs were, on average, closer to the clean recording than UVR-DeNoise's on all three metrics in both strata. This is a narrow result:

- **It is a test on OpGAN's training-family noise.** The degradation adds hiss and may add crackle, hum and saturation, and it always applies a low-pass filter (cutoff 4 to 8 kHz). OpGAN was trained on this same mix. UVR-DeNoise is a denoising model and, as far as we know, was not trained to restore filtered-out high frequencies or undo saturation, so part of OpGAN's lead likely comes from that difference rather than from denoising alone.
- **UVR likely ran outside the setting it was designed for.** It received 16 kHz mono input, one 2-second frame at a time, with no surrounding context.
- **There was no formal listening test.** The metrics are signal-level, not perceptual.
- **Some references may be live recordings.** Room or audience noise that a system removes from one is scored as distortion.
- **Some material was not tested:** orchestral music, real degraded transfers (for example 78 rpm discs), and noise outside the training family.

#### What was measured

Two strata, each reported on its own and never pooled. **Solo piano:** Musopen's Chopin set on the Internet Archive (CC0). **Chamber:** MusicNet. The checks found none of it in OpGAN's training set: a seeded selection walk checked each candidate's file and decoded-audio digests against OpGAN's 145 training recordings and against every earlier admission, and screened it for a duplicate performance. UVR-DeNoise's training data was not checked, so the report's frozen claim scope, which says neither model was trained on these recordings, is unverified for UVR. Each stratum's test set is 30 recordings; piano has 3,909 scored 2-second frames from 11 performers, chamber 6,391 frames from 14 performer groups. Every frame was degraded once with a reproducible seed, run through both systems, and scored against the clean original; the degraded input was scored the same way as a baseline. All seven publication gates held in both strata, and both systems produced a valid output for every frame (coverage 1.000).

Means over the 30 test recordings. The interval is a 95% bootstrap interval that resamples performers, not frames or recordings. A positive paired value means OpGAN scored better.

**Solo piano**

| Metric | Degraded input | UVR | OpGAN | OpGAN advantage, 95% interval |
|---|---|---|---|---|
| Reconstruction SNR (dB, higher is better) | 13.05 | 13.75 | 19.78 | +6.03 [5.49, 6.49] |
| SI-SNR (dB, higher is better) | 15.08 | 15.95 | 21.11 | +5.16 [4.96, 5.30] |
| Log-spectral distance (dB, lower is better) | 17.94 | 15.00 | 13.39 | +1.61 [1.38, 1.78] |
| *Diagnostic:* shift-tolerant reconstruction SNR | 15.71 | 16.76 | 20.29 | +3.54 [2.95, 4.03] |
| *Diagnostic:* shift-tolerant SI-SNR | 18.41 | 19.77 | 21.71 | +1.94 [1.65, 2.19] |

**Chamber**

| Metric | Degraded input | UVR | OpGAN | OpGAN advantage, 95% interval |
|---|---|---|---|---|
| Reconstruction SNR (dB, higher is better) | 12.14 | 12.48 | 18.06 | +5.59 [4.47, 6.58] |
| SI-SNR (dB, higher is better) | 14.06 | 14.45 | 19.39 | +4.94 [4.54, 5.33] |
| Log-spectral distance (dB, lower is better) | 13.94 | 12.20 | 11.26 | +0.94 [0.35, 1.51] |
| *Diagnostic:* shift-tolerant reconstruction SNR | 15.17 | 15.66 | 18.64 | +2.99 [1.76, 4.02] |
| *Diagnostic:* shift-tolerant SI-SNR | 17.85 | 18.46 | 20.10 | +1.64 [1.28, 1.94] |

The low-pass filter delays the degraded input by about one sample, and reconstruction SNR and SI-SNR credit undoing that delay, which OpGAN was trained on and UVR, as far as we know, was not. The shift-tolerant rows score both systems at the best of a -1, 0 and +1 sample shift; with the delay set aside, OpGAN's SI-SNR lead falls from about 5 dB to about 2 dB in both strata. A single integer shift likely corrects this only in part, because the filter's delay varies with frequency. These rows are a diagnostic, not a headline. The same is true of the degraded input: against it, OpGAN raised reconstruction SNR by about 6.7 dB (piano) and 5.9 dB (chamber), UVR by about 0.7 and 0.3 dB.

> **Note:** 11 and 14 performers are few for a bootstrap interval, which can then be narrower than its nominal 95%. Treat the intervals as descriptive uncertainty, not as a significance test. The post-hoc analyses below give a second interval that weights each performer equally rather than each recording; it is wider than the bootstrap interval in 5 of the 6 primary comparisons.

#### Post-hoc analyses

These were fixed in [docs/benchmark-posthoc-analyses.md](docs/benchmark-posthoc-analyses.md), merged before any score was read, and are secondary to the results above.

- **One paired difference per performer, with a t-interval over performers:**

  | Stratum | Reconstruction SNR | SI-SNR | Log-spectral distance | Log-spectral distance below 4 kHz |
  |---|---|---|---|---|
  | Solo piano (11 performers) | +5.76 [5.31, 6.21], 11 of 11 | +5.03 [4.69, 5.36], 11 of 11 | +1.50 [1.20, 1.79], 11 of 11 | +0.36 [0.02, 0.71], 9 of 11 |
  | Chamber (14 performers) | +5.20 [4.10, 6.29], 14 of 14 | +4.85 [4.36, 5.34], 14 of 14 | +0.70 [-0.07, 1.47], 13 of 14 | +1.30 [0.49, 2.12], 14 of 14 |

  Each cell gives the mean OpGAN advantage, its 95% t-interval, and how many performers favour OpGAN. Every performer favours OpGAN on reconstruction SNR and SI-SNR. For chamber log-spectral distance, the t-interval includes zero while the bootstrap interval above does not, so that result is less certain under the t-interval.
- **Log-spectral distance below 4 kHz** (the lowest cutoff the degradation draws, so every frame keeps this band), means for degraded input, UVR and OpGAN: piano 8.86, 8.55, 8.19; chamber 6.30, 7.23, 5.78. On piano, OpGAN's lead below 4 kHz (+0.36 dB) is much smaller than its full-band lead (+1.50 dB). This suggests, but does not measure, that most of its spectral lead lies above 4 kHz, where the filter's cutoff falls. On chamber, UVR's output is further from the clean recording below 4 kHz than the degraded input is; this was observed, not explained.
- No output from either system reached the metrics' 60 dB cap, and none was silent.

#### Informal blind listen

One listener, the author of OpGAN, compared eight 10-second clips (four per stratum) chosen by a seeded rule written down before the clips were made, with the two systems' outputs in random order and the key revealed after the answers were recorded. OpGAN was preferred in 5 clips, UVR in 1, and 2 were close to a tie (one leaning UVR). This is not a listening test: one listener, eight clips, no statistics.

#### Protocol and disclosures

The benchmark follows a protocol frozen in code (`benchmark/protocols/v14.json`) and checked by a loader that refuses unknown keys, a different version, and a broken amendment chain. It is a frozen protocol with disclosed amendments, not a pre-registration held by a third party. Three amendments followed contact with candidate material or system outputs, and none followed a score of the test recordings:

| Version | What prompted it | What changed |
|---|---|---|
| v12 | Listing the original candidate collection showed it held exactly the 145 training recordings | Two new strata from other sources (Chopin piano, MusicNet chamber) |
| v13 | An informal pilot on 4 tracks outside both training and test sources, which saw system outputs, found the filter's one-sample delay | Added the shift-tolerant diagnostic and the degraded-input baseline; primary metrics unchanged |
| v14 | A selection walk under v13 (no system output) admitted no chamber recording, because MusicNet ships 32-bit float WAV | Admitted the WAV float format; the piano admissions were identical under v14 |

The full record is in [docs/benchmark-protocol.md](docs/benchmark-protocol.md). One recording in the solo-piano test set is Chopin's Variations for Flute and Piano (B. 9); the frozen candidate list excluded the cello sonata but not this piece, and it was left in rather than changing the protocol after selection.

#### Reproduce

> **Note:** The run's preconditions pass only on the reference platform (darwin_arm64); elsewhere they abort with `ENVIRONMENT_MISMATCH` before any audio is fetched. The UVR model and its parameter files must already be in the model directory with their pinned digests, because the run refuses to download them, and the OpGAN checkpoint must be fetched first with `scripts/fetch_checkpoint.py`.

```bash
uv run python -m benchmark.run_walk --uvr-model-dir DIR
PYTHONPATH=backend uv run python -m benchmark.run_systems --walk-dir WALK_DIR --run-dir NEW_DIR --uvr-model-dir DIR
PYTHONPATH=backend uv run python -m benchmark.run_report --walk-dir WALK_DIR --systems-dir SYSTEMS_DIR --output-dir NEW_DIR
PYTHONPATH=backend uv run python -m benchmark.run_posthoc --walk-dir WALK_DIR --systems-dir SYSTEMS_DIR --output-dir NEW_DIR
```

Run digests (SHA-256): protocol `c381a0103c0ad8337c8abd3e5a3c0425757ddfa4d69325152345045983c3c2f2`, walk manifest `4c05b1c3db675260c8bd263c8e14fc69959cf8c7d6a9dee47be3e20143c976e6`, audit trail `5454914179ee4148232fde69b655c8ba94853be8f533ff68e6dccf3c0d6d9c2d`, systems results `8ee44a681e9019bfd31c9b7ecf57fe179e926a7711065dad9b8682c0b4b31169`.

#### Earlier figures

Earlier SDR, PESQ and STOI figures were withdrawn: they were measured on the training corpus, so training and evaluation shared source recordings. They are kept, with that context, in [docs/benchmarks.md](docs/benchmarks.md).

## Samples

Three 6-second passages of Chopin piano music from the benchmark's demo partition: recordings from the CC0 Musopen Chopin collection that the selection checks found outside OpGAN's training set, and that no score used. The passages were chosen by ear. Each was degraded with one fixed noise profile inside the training ranges (pink hiss at 10 dB SNR, crackle, 60 Hz hum, a 5 kHz low-pass, no saturation), chosen so the noise is clearly audible; the benchmark itself draws its noise at random within those ranges. Both systems restored the same noisy file through the backend's own denoiser services. `scripts/make_demo_samples.py` regenerates the files, and `samples/MANIFEST.json` records the settings and each file's SHA-256. Each link opens the WAV file.

| Piece | Performer | Noisy | OpGAN | UVR | Clean |
|---|---|---|---|---|---|
| Ballade no. 3, Op. 47 (from 7:30) | Frank Levy | [noisy](samples/opgan/ballade_3_noisy.wav) | [restored](samples/opgan/ballade_3_restored.wav) | [restored](samples/uvr/ballade_3_restored.wav) | [clean](samples/opgan/ballade_3_clean.wav) |
| Mazurka in C major, Op. 56 no. 2 (from 0:31) | Xuan He | [noisy](samples/opgan/mazurka_op56_2_noisy.wav) | [restored](samples/opgan/mazurka_op56_2_restored.wav) | [restored](samples/uvr/mazurka_op56_2_restored.wav) | [clean](samples/opgan/mazurka_op56_2_clean.wav) |
| Waltz in A flat major, Op. 64 no. 3 (from 0:53) | Diana Hughes | [noisy](samples/opgan/waltz_op64_3_noisy.wav) | [restored](samples/opgan/waltz_op64_3_restored.wav) | [restored](samples/uvr/waltz_op64_3_restored.wav) | [clean](samples/opgan/waltz_op64_3_clean.wav) |

## Roadmap

- [x] Working denoising API (UVR)
- [x] Security hardening (P0/P1 complete)
- [x] React frontend with wizard flow, real waveform, audio playback, before/after comparison
- [x] Frontend redesign: aurora tape strip, live A/B crossfade compare
- [x] Docker + CI
- [x] Build/acquire paired training dataset (146,200 pairs from 14 composers)
- [x] Train OpGAN (100 epochs on T4)
- [x] Swap in OpGAN as default denoiser
- [x] First backend deployment on Cloud Run (now deferred)
- [x] Backend on AWS Lambda (stage 1 of [ADR 009](docs/adr/009-pay-per-use-hosting.md)), with
      deploys from CI after review
- [x] Withdraw the first benchmark: its evaluation population was not disjoint from
      training at the level of source material. See
      [benchmarks.md](docs/benchmarks.md)
- [x] Freeze a replacement protocol before selecting any evaluation material, enforced
      by a validating loader rather than by prose. See
      [benchmark-protocol.md](docs/benchmark-protocol.md)
- [x] Run the replacement benchmark on recordings outside OpGAN's training set, and publish
      the result with performer-level intervals (see [Benchmark Results](#benchmark-results))
- [ ] A second study, specified and timestamped before it runs: real historical transfers
      with a listening test, noise outside the training family, and UVR at its native
      44.1 kHz
- [ ] Architecture ablation: a plain Conv1D GAN and a small U-Net trained on the same data,
      to test whether the Self-ONN layers account for the result
- [ ] A full-bandwidth model (44.1 or 48 kHz) for recordings with content above 8 kHz
- [ ] Stage 2 of ADR 009: storage for files and jobs that outlives one Lambda environment

## License

MIT
