FROM python:3.11-slim

WORKDIR /app

# System dependencies for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# PyTorch CPU: separate layer for caching (largest download).
# Pinned to the exact versions resolved by the first successful Docker build.
# torch and torchaudio currently pin to different minor versions; revisit
# both together in a future dependency upgrade to a matched PyTorch release.
RUN pip install --no-cache-dir \
    torch==2.12.1+cpu \
    torchaudio==2.11.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Project dependencies: install before copying code for layer caching.
# Create a stub package so `pip install .` resolves deps from pyproject.toml.
COPY pyproject.toml ./
RUN mkdir -p backend/app && \
    touch backend/__init__.py backend/app/__init__.py && \
    pip install --no-cache-dir . && \
    rm -rf backend

# Application code
COPY backend/app/ backend/app/
COPY checkpoints/final.pt checkpoints/final.pt

# Non-root user + writable directories
RUN useradd --create-home --shell /sbin/nologin appuser && \
    mkdir -p backend/uploads backend/processed && \
    chown -R appuser:appuser backend
USER appuser

WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --proxy-headers --forwarded-allow-ips=\"${FORWARDED_ALLOW_IPS:-127.0.0.1}\""]
