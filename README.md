# HarmonyRestorer v1

AI-powered audio denoising. Upload noisy audio, get clean audio back.

- **Model**: Custom-trained OpGAN (1D Operational GAN), with UVR-DeNoise as fallback via [audio-separator](https://github.com/karaokenerds/python-audio-separator)
- **Formats**: WAV, MP3, FLAC, OGG, M4A, AAC (max 50 MB, 10 minutes)
- **Stack**: FastAPI + React/TypeScript, Docker-ready
- **Tests**: 347 passing (backend 108, dataset 155, frontend 84)

## Quick Start

```bash
git clone https://github.com/jacob7choi-xyz/harmonyrestorer-v1.git
cd harmonyrestorer-v1

uv sync --all-groups

# Backend
cd backend && uv run uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm ci && npm run dev
```

## API

```bash
# Upload and denoise
curl -X POST http://localhost:8000/api/v1/denoise -F "file=@audio.wav"

# Check status
curl http://localhost:8000/api/v1/status/{job_id}

# Download result
curl -O http://localhost:8000/api/v1/download/{job_id}
```

Interactive docs: http://localhost:8000/api/docs

## Architecture

```
backend/app/
├── main.py              # App factory, middleware, lifespan
├── config.py            # Settings (env vars)
├── schemas.py           # Pydantic models, JobStatusEnum
├── exceptions.py        # Error handlers (no stack traces to users)
├── middleware.py         # Per-IP rate limiting
├── routes/
│   ├── health.py        # GET /, GET /health
│   └── denoise.py       # Upload, status, download
├── models/
│   ├── op_gan.py        # OpGAN generator + discriminator
│   ├── self_onn.py      # Self-ONN layers
│   └── chunking.py      # Overlap-add chunking for arbitrary-length audio
├── services/
│   ├── opgan_denoiser.py  # OpGAN inference service (default)
│   ├── denoiser.py        # UVR wrapper, lazy model loading (fallback)
│   └── jobs.py            # JobManager, background processing

frontend/src/
├── App.tsx              # Wizard orchestrator (upload/processing/complete)
├── types.ts             # Shared types
├── api/client.ts        # Backend API calls + polling
├── hooks/               # useAudioDecoder, useAudioPlayback
└── components/          # TechnoBackground, UploadArea, WaveformCanvas, AudioPlayer, ComparisonView, ErrorBoundary
```

## Security

- File size limit (50 MB) and audio duration limit (10 min)
- Magic byte validation (content must match declared format)
- UUID-based file storage (client filenames never trusted)
- Rate limiting (10 req/min per IP on upload)
- CORS restricted to configured origins
- Error sanitization (no stack traces in responses)
- Nginx: HSTS, CSP, X-Frame-Options, health endpoint restricted to private IPs

## Docker

```bash
docker compose up --build    # Start full stack
docker compose down          # Stop
```

Both Dockerfiles use non-root users. The backend Dockerfile includes a health check; resource limits are configured in `docker-compose.yml`.

## Quality Gate

```bash
# Backend + dataset
cd backend && uv run ruff format --check . && uv run ruff check . && uv run mypy . && uv run pytest -v
cd .. && uv run pytest dataset/tests/ -v

# Frontend
cd frontend && npm run type-check && npm run lint && npm run build && npm run test
```

## Dataset Pipeline

The `dataset/` package builds training data for the OpGAN from public domain classical recordings.

```bash
uv sync --extra dataset

# Download from Internet Archive (Musopen collection)
uv run python -m dataset.acquire --output data/raw/ --max-tracks 200 --formats .flac

# Preprocess to 16kHz mono, 2-second frames
uv run python -m dataset.preprocess data/raw/ --output data/clean_frames/

# Generate noisy/clean training pairs (5 variants per frame, seeded for reproducibility)
uv run python -m dataset.generate_pairs data/clean_frames/ --output data/pairs/ --seed 42

# Train OpGAN (CUDA GPU recommended, ~5-10 min/epoch on T4)
uv run python -m dataset.train --pairs data/pairs/ --epochs 100 --batch-size 16
```

Current dataset: 145 tracks from 14 composers, 29,240 clean frames, 146,200 training pairs. Source audio is CC/public domain; noise is synthetically generated (tape hiss, vinyl crackle, mains hum, HF rolloff, tape saturation).

## OpGAN

From-scratch implementation of **1D Operational GANs** based on [Kiranyaz et al. 2022](https://arxiv.org/abs/2212.14618), living in `backend/app/models/`. Training script at `dataset/train.py` supports CUDA mixed precision, checkpoint resume, and gradient health monitoring.

### Benchmark Results

Trained 100 epochs on Tesla T4 (~51 hours, ~$100 total GCP). Model operates at 16kHz mono. Evaluated on 131,027 noisy/clean pairs:

| Metric   | Mean  | Median |
|----------|-------|--------|
| SDR (dB) | 23.74 | 23.85  |
| PESQ     | 4.04  | 4.28   |
| STOI     | 0.960 | 0.989  |

### UVR Baseline

UVR-DeNoise.pth evaluated on the same dataset (131,013 files evaluated, 15,182 skipped for quiet frames, resampled to 16kHz mono for fair comparison):

| Metric   | OpGAN  | UVR   | Delta       |
|----------|--------|-------|-------------|
| SDR (dB) | **23.74** | 11.86 | +11.88 dB |
| PESQ     | **4.04**  | 3.72  | +0.32     |
| STOI     | **0.960** | 0.953 | +0.007    |

OpGAN outperforms UVR across all metrics. See [docs/benchmarks.md](docs/benchmarks.md) for details.

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
- [x] Security hardening (3 phases)
- [x] React frontend with wizard flow, real waveform, audio playback, before/after comparison
- [x] Frontend redesign: animated canvas background, blue accent, social footer, 84 tests
- [x] Docker + CI/CD
- [x] Build/acquire paired training dataset (146,200 pairs from 14 composers)
- [x] Train OpGAN (100 epochs on T4)
- [x] Benchmark OpGAN (SDR 23.74 dB, PESQ 4.04, STOI 0.960)
- [x] Benchmark UVR on same dataset for head-to-head comparison
- [x] Swap in OpGAN as default denoiser (outperforms UVR by ~12 dB SDR)

## License

MIT
