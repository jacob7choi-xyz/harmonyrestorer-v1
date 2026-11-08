# HarmonyRestorer v1

A web app for denoising audio. You upload a recording, the backend runs it through a 1D Operational GAN (OpGAN) trained from scratch on classical recordings, and you get the restored audio back to compare and download.

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

**Withdrawn pending re-evaluation.** Trained 100 epochs on Tesla T4 (~51 hours, ~$100 total GCP). Model operates at 16 kHz mono.

Previously published SDR, PESQ, and STOI figures were computed across the full generated pair corpus rather than a source-track-disjoint evaluation set. Training and evaluation therefore shared source recordings, and in many cases the identical clean 2-second frame: the dataset holds five noise variants per frame, and the 90/10 split was drawn over pairs rather than over recordings, so roughly 41% of clean frames are expected to have sibling variants on both sides. Those figures characterize reconstruction over the corpus; they do not support a claim about generalization to unseen recordings, and the head-to-head margin against UVR is the most affected claim, since UVR never trained on any of this data and OpGAN trained on most of it.

The figures and their full context are retained in [docs/benchmarks.md](docs/benchmarks.md).

A replacement is in progress, and its protocol was frozen before any evaluation material was selected: recording-level partitions, per-recording bootstrap intervals rather than per-frame, three signal-fidelity metrics chosen for music instead of the speech metrics used before, and publication gates that withhold a comparison unless both systems covered an identical declared population. The protocol is [documented](docs/benchmark-protocol.md) and, more usefully, enforced by a validating loader that refuses to run a configuration contradicting it.

## Samples

Three before/after/restored triplets for each model. Click to listen in the browser.

**OpGAN** (custom-trained, in-distribution):

| Track | Noisy | Restored | Clean |
|-------|-------|----------|-------|
| Bach | [noisy](samples/opgan/noisy_bach.wav) | [restored](samples/opgan/restored_bach.wav) | [clean](samples/opgan/clean_bach.wav) |
| Beethoven | [noisy](samples/opgan/noisy_beethoven.wav) | [restored](samples/opgan/restored_beethoven.wav) | [clean](samples/opgan/clean_beethoven.wav) |
| Tchaikovsky | [noisy](samples/opgan/noisy_tchaikovsky.wav) | [restored](samples/opgan/restored_tchaikovsky.wav) | [clean](samples/opgan/clean_tchaikovsky.wav) |

**UVR** (pretrained baseline, for comparison):

| Track | Noisy | Restored | Clean |
|-------|-------|----------|-------|
| Bach | [noisy](samples/uvr/bach_noisy.wav) | [restored](samples/uvr/bach_restored.wav) | [clean](samples/uvr/bach_clean.wav) |
| Beethoven | [noisy](samples/uvr/beethoven_noisy.wav) | [restored](samples/uvr/beethoven_restored.wav) | [clean](samples/uvr/beethoven_clean.wav) |
| Mozart | [noisy](samples/uvr/mozart_noisy.wav) | [restored](samples/uvr/mozart_restored.wav) | [clean](samples/uvr/mozart_clean.wav) |

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
- [ ] Run the replacement benchmark on track-disjoint recordings and publish the result
      with per-recording intervals
- [ ] Out-of-distribution audio evaluation
- [ ] Stage 2 of ADR 009: storage for files and jobs that outlives one Lambda environment

## License

MIT
