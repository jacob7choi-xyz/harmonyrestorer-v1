# HarmonyRestorer v1

AI-powered audio denoising. Upload noisy audio, get clean audio back.

- **Model**: UVR-DeNoise from [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) via [audio-separator](https://github.com/karaokenerds/python-audio-separator)
- **Formats**: WAV, MP3, FLAC, OGG, M4A, AAC (max 50 MB, 10 minutes)
- **Stack**: FastAPI + React/TypeScript, Docker-ready
- **Tests**: 39 passing (0.2s)

## Quick Start

```bash
git clone https://github.com/jacob7choi-xyz/harmonyrestorer-v1.git
cd harmonyrestorer-v1

conda env create -f environment.yml
conda activate harmonyrestorer-v1

# Backend
cd backend && uvicorn app.main:app --reload --port 8000

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
├── services/
│   ├── denoiser.py      # UVR wrapper, lazy model loading
│   └── jobs.py          # JobManager, background processing
└── _archive/            # OpGAN research code (see below)

frontend/src/
├── App.tsx              # Main component
├── types.ts             # Shared types
├── api/client.ts        # Backend API calls + polling
└── components/          # UploadArea, ProgressCard, Waveform, ErrorBoundary
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

Backend and frontend have independent Dockerfiles with non-root users, health checks, and resource limits.

## Quality Gate

```bash
cd backend
black --check . && isort --check . && ruff check . && mypy . && pytest -v
```

## Dataset Pipeline

The `dataset/` package builds training data for the OpGAN from public domain classical recordings.

```bash
pip install -e ".[dataset]"

# Download from Internet Archive (Musopen collection)
python -m dataset.acquire --collection MusopenCollectionAsFlac --output data/raw --formats .flac

# Preprocess to 16kHz mono, 2-second frames
python -m dataset.preprocess data/raw/ --output data/clean_frames/

# Generate noisy/clean training pairs (5 variants per frame)
python -m dataset.generate_pairs data/clean_frames/ --output data/pairs/ --variants 5
```

Current dataset: 145 tracks from 14 composers, 29,240 clean frames, 146,200 training pairs. Source audio is CC/public domain; noise is synthetically generated (tape hiss, vinyl crackle, mains hum, HF rolloff, tape saturation).

## OpGAN (Archived)

The `_archive/` folder contains a from-scratch implementation of **1D Operational GANs** based on [Kiranyaz et al. 2022](https://arxiv.org/abs/2110.10149). Architecture is complete and trainable but not yet trained. The production API uses pretrained UVR models.

## Roadmap

- [x] Working denoising API (UVR)
- [x] Security hardening (3 phases)
- [x] React frontend with real API integration
- [x] Docker + CI/CD
- [x] Build/acquire paired training dataset (146,200 pairs from 14 composers)
- [ ] Train OpGAN
- [ ] Benchmark OpGAN vs UVR (SDR, PESQ, STOI)
- [ ] Swap in OpGAN if it outperforms UVR

## License

MIT
