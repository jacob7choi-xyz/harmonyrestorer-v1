# uv is pinned to the version CI and local development use, and by digest so the binary
# stays the one whose provenance attestation was checked.
FROM ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 AS uv

# ADR 009 (docs/adr/009-pay-per-use-hosting.md) plans to run the app on AWS Lambda
# through the Lambda Web Adapter.
# 1.1.0's release notes add AWS_LWA_READINESS_CHECK_TIMEOUT_SECONDS, set below. The digest
# is the multi-platform index. No GitHub provenance attestation was found for it, so the
# digest fixes which bytes are used but does not show who built them.
FROM public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0@sha256:17cfd08eff1dfea3f6a9a1e9c65fdac80aa4919b6085e746615530f43f57d2f1 AS lambda-adapter

# Stages: build installs the locked dependencies, compiling diffq; build-test adds the
# test group; base holds the system packages used at run time (see below); test and
# runtime start from base and copy in a finished .venv, so neither contains the compiler.
# test runs the backend tests against the image's own ffmpeg and libsndfile. runtime is
# last, so a build without --target produces the deployable image.
FROM python:3.11-slim AS build

WORKDIR /app

# build-essential compiles diffq, which is published without a wheel for this platform.
# build-test inherits it; test and runtime do not include it.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

# Dependencies are installed from uv.lock, the population the gate audits and tests,
# instead of being resolved at build time. uv installs each package from the URL the
# lock records and checks it against the recorded hash. uv must use the base image's
# Python, never download one. The cache stays out of the image, as with pip's
# --no-cache-dir before, and bytecode is compiled at install time, as pip did by
# default, because the non-root runtime user cannot write it later.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./

# Two passes. The first installs everything except diffq, with --no-build so that any
# other package needing a source build stops the image build instead of compiling. The
# second builds diffq without isolation, against the Cython and setuptools the first pass
# installed from the lock, because diffq's source distribution declares its build
# requirements without versions and an isolated build would fetch them unpinned and
# unhashed.
RUN uv sync --frozen --no-dev --no-install-project --no-install-package diffq --no-build && \
    uv sync --frozen --no-dev --no-install-project --no-build-isolation-package diffq

FROM build AS build-test

RUN uv sync --frozen --no-dev --no-install-project --group test

# The .venv's python links to /usr/local/bin/python3, so base names the same image as
# build. The application uses ffmpeg and libsndfile1; the HEALTHCHECK uses curl.
FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:${PATH}"

# CI runs the backend tests here; they passed with only the dependencies, pyproject.toml
# and backend/ present, so this stage needs no checkpoint (CI fetches one for runtime,
# docs/adr/008-checkpoint-release.md). pyproject.toml holds the pytest settings,
# including asyncio_mode. .dockerignore keeps backend/.env out.
FROM base AS test

COPY --from=build-test /app/.venv /app/.venv
COPY pyproject.toml ./
COPY backend/ backend/
CMD ["python", "-m", "pytest", "backend/tests", "-q"]

FROM base AS runtime

COPY --from=build /app/.venv /app/.venv

# Application code
COPY backend/app/ backend/app/
COPY checkpoints/final.pt checkpoints/final.pt

# The path is the one the adapter's README installs it to. Under Docker the CMD below
# still serves on its own (scripts/check_runtime_image.sh). Readiness uses /, which runs
# no checks (backend/app/routes/health.py); /health returns 503 when a check fails. The
# README says the adapter fails init when the readiness timeout expires. On Lambda on
# 2026-11-07 the app was not ready within 8 s, and the adapter failed init. With a longer
# timeout, the first start of the image was ready after about 78 s, inside the first
# request (reported duration 78.6 s), so the 120 s function timeout appears to cover a
# slow start and that request's own work. 100 s is a ceiling below 120 s, not a need.
COPY --from=lambda-adapter /lambda-adapter /opt/extensions/lambda-adapter
ENV AWS_LWA_PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/ \
    AWS_LWA_READINESS_CHECK_TIMEOUT_SECONDS=100 \
    AWS_LWA_INVOKE_MODE=response_stream

# Non-root user. Under /app it owns only the upload and processed directories, so the
# process cannot rewrite the application code, the venv or the checkpoint.
RUN useradd --create-home --shell /sbin/nologin appuser && \
    mkdir -p backend/uploads backend/processed && \
    chown appuser:appuser backend/uploads backend/processed
USER appuser

WORKDIR /app/backend

EXPOSE 8000

# Checks the port the CMD below serves on, which follows PORT when a platform sets it
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8000}/health" || exit 1

# Exactly one worker: inference admission control is process-local, so the
# instance-wide inference bound requires a single application process.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1 --proxy-headers --forwarded-allow-ips=\"${FORWARDED_ALLOW_IPS:-127.0.0.1}\""]

# Connects the image to this repository when it is pushed to the GitHub Container Registry
LABEL org.opencontainers.image.source="https://github.com/jacob7choi-xyz/harmonyrestorer-v1"
