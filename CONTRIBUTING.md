# Contributing to HarmonyRestorer

## Prerequisites

- Python 3.11+
- Node.js 18+
- conda (recommended) or venv
- Docker (optional, for full stack)

## Setup

### Backend

```bash
# Create environment
conda create -n harmonyrestorer-v1 python=3.11 -y
conda activate harmonyrestorer-v1

# Install dependencies
pip install -e ".[dev]"

# Run dev server
cd backend && uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm ci
npm run dev    # Dev server on :3000
```

### Docker (full stack)

```bash
docker compose up --build
```

## Quality Gate

All checks must pass before submitting a PR:

```bash
cd backend
black --check . && isort --check . && ruff check . && mypy . && pytest -v
```

Auto-fix formatting:

```bash
black . && isort . && ruff check --fix .
```

Frontend checks:

```bash
cd frontend
npm run lint && npm run type-check && npm run build
```

## Dataset Pipeline (for OpGAN training)

The dataset pipeline downloads public domain classical recordings, preprocesses them, and generates noisy/clean training pairs. You need ~40 GB of free disk space.

### 1. Install dataset dependencies

```bash
pip install -e ".[dataset]"
```

### 2. Download source audio

Downloads FLAC recordings from the Musopen collection on Internet Archive (145 tracks, ~7 GB):

```bash
python -m dataset.acquire --output data/raw/ --max-tracks 200 --formats .flac
```

Useful flags:
- `--formats .flac` -- Only download FLAC files (skip MP3/OGG duplicates)
- `--exclude goldberg bach` -- Skip files matching keywords (case-insensitive)
- `--max-tracks 50` -- Limit download count (useful for testing)

### 3. Preprocess to training frames

Resamples to 16 kHz mono and slices into 2-second frames, discarding silence:

```bash
python -m dataset.preprocess data/raw/ --output data/frames/
```

### 4. Generate noisy/clean pairs

Applies synthetic analog degradation (tape hiss, vinyl crackle, mains hum, HF rolloff, tape saturation) to create 5 noisy variants per clean frame:

```bash
python -m dataset.generate_pairs data/frames/ --output data/pairs/ --seed 42
```

- `--seed 42` makes the noise generation deterministic and reproducible
- `--variants 10` increases variants per frame (default: 5)
- `--snr-min 5.0 --snr-max 30.0` controls noise intensity range

### 5. Train the OpGAN

Requires a CUDA GPU for practical training times. A Tesla T4 (15 GB VRAM) works well:

```bash
python -m dataset.train --pairs data/pairs/ --epochs 100 --batch-size 16
```

Training features:
- CUDA mixed precision (float16) enabled automatically on CUDA GPUs
- Checkpoints saved to `checkpoints/` (best.pt, every 10 epochs, final.pt)
- Resume from checkpoint: `--resume checkpoints/best.pt`
- Training config saved to `checkpoints/train_config.json`

MPS (Apple Silicon) works but is ~6x slower than a T4. CPU is not practical.

## Project Structure

```
backend/app/           # FastAPI backend
  routes/              # API endpoints
  services/            # Business logic (denoiser, job manager)
  _archive/            # OpGAN model code (do not modify)
frontend/src/          # React/TypeScript frontend
dataset/               # Training data pipeline
  acquire.py           # Download from Internet Archive
  preprocess.py        # Resample, normalize, slice to frames
  noise.py             # Analog noise synthesis
  generate_pairs.py    # Create noisy/clean training pairs
  torch_dataset.py     # PyTorch Dataset for training
  train.py             # OpGAN training loop
data/                  # Generated data (gitignored)
  raw/                 # Downloaded source audio
  frames/              # Preprocessed 2s frames
  pairs/               # Noisy/clean training pairs
    clean/             # Clean reference frames
    noisy/             # Degraded variants
    metadata/          # Degradation params per variant (JSON)
checkpoints/           # Model checkpoints (gitignored)
```

## Code Standards

- **Python**: Black formatter, isort imports, ruff linter, mypy type checker
- **TypeScript**: ESLint, strict mode
- **Line length**: 100
- **Type hints**: Required on all public functions
- **Docstrings**: Google style, required on all public classes and functions
- **Logging**: Use `%s` formatting (not f-strings), include context (IDs, paths, counts)
- **No hardcoded values**: Use `config.py` for settings
- **No functions over 50 lines**

## Running Tests

```bash
# All tests
cd backend && pytest -v

# Single test
pytest tests/test_health.py::test_health_returns_ok -v

# With coverage
pytest --cov=app --cov-report=html -v
```

## Security Rules

These are enforced and tested:

- Max file size: 50 MB
- Max audio duration: 10 minutes
- Allowed extensions: .wav, .mp3, .flac, .ogg, .m4a, .aac
- Filenames stored as UUIDs (never trust client input)
- Path traversal prevention on all file operations
- Rate limiting: 10 requests/minute per IP on upload
- No stack traces in error responses
- Credentials via environment variables only
