FROM python:3.11-slim

WORKDIR /app

# System dependencies for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

# PyTorch CPU — separate layer for caching (largest download)
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Project dependencies — install before copying code for layer caching.
# Create a stub package so `pip install .` resolves deps from pyproject.toml.
COPY pyproject.toml ./
RUN mkdir -p backend/app && \
    touch backend/__init__.py backend/app/__init__.py && \
    pip install --no-cache-dir . && \
    rm -rf backend

# Application code
COPY backend/app/ backend/app/

# Non-root user + writable directories
RUN useradd --create-home appuser && \
    mkdir -p backend/uploads backend/processed && \
    chown -R appuser:appuser backend
USER appuser

WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
