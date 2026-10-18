# Security Policy

## Supported Versions

Only the current `main` branch is actively maintained. No prior releases are supported.

| Version | Supported |
|---------|-----------|
| main    | Yes       |

## Reporting a Vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Report vulnerabilities via one of the following:

- **GitHub private advisory**: use the "Report a vulnerability" button on the [Security tab](https://github.com/jacob7choi-xyz/harmonyrestorer-v1/security/advisories/new)
- **Email**: jacob77choi@gmail.com

Include a description of the issue, steps to reproduce, and potential impact. Expect an acknowledgment within 72 hours. A resolution timeline will be provided within 14 days of confirmation.

## Scope

In scope:

- Backend API (file upload, job processing, download endpoints)
- File validation logic (magic byte checks, size and duration limits) and upload decoding with ffmpeg
- Job management and file handling
- Dependency supply chain (pyproject.toml, uv.lock, package-lock.json)

Out of scope:

- The hosted frontend code for this project is in scope; platform-level issues with Vercel itself are out of scope and should be reported to Vercel directly
- The GCP training infrastructure (decommissioned after model training)
- Large-scale DoS attacks that exceed the documented rate limits and resource controls

## Supply Chain

### Python dependencies

Dependencies are managed with [uv](https://github.com/astral-sh/uv) and pinned via a committed `uv.lock`. The Python gate, `scripts/gate_python.sh`, runs identically in CI and locally. It checks that the lock is current and that the environment matches it, then audits the locked population with pip-audit.

The backend image installs from the same lock with `uv sync --frozen` rather than resolving versions at build time, so the application environment (`/app/.venv`) is built from the population the gate audits and tests. uv installs each package from the URL and hash the lock records. The base image's own system Python packages, such as its `pip`, sit outside that environment and outside the gate's audit. `diffq` has no wheel for the production platform, so it is compiled from its source distribution against the locked Cython and setuptools instead of build dependencies fetched during the build.

**PyTorch index**: `torch` is sourced from the official PyTorch CPU wheel index (`https://download.pytorch.org/whl/cpu`) via `[tool.uv.sources]` in `pyproject.toml`. The index is marked `explicit`, so only `torch` resolves from it, which keeps torch itself on its CPU build. `torchvision`, which `audio-separator` pulls in through `onnx2torch`, still resolves from PyPI as a CUDA-linked build. In images built from this Dockerfile on 2026-10-16, its compiled extension did not load (`libcudart.so.13` is absent), so `import torchvision` failed there. No tracked code imports torchvision, and the OpGAN path and the UVR-DeNoise path ran without it.

**UVR model files**: these are not Python packages and sit outside `uv.lock` and the image. When `DENOISER_ENGINE=uvr`, audio-separator downloads `UVR-DeNoise.pth` on the first UVR job from the `TRvlvr/model_repo` GitHub releases, or from the `nomadkaraoke/python-audio-separator` releases if the first request returns a status other than 200, and a model list and model parameter files from `raw.githubusercontent.com`, into `/tmp/audio-separator-models/` in the container, which is audio-separator's default because the app passes no `model_file_dir` and nothing in the repository sets `AUDIO_SEPARATOR_MODEL_DIR`. audio-separator writes whatever a successful response returns without checking it against a hash, and it reuses a file already present by path alone. It writes the response straight to the final path, so, read from its code, an interrupted transfer could leave a partial file there that a later run would reuse. Its VR loader calls `torch.load` without `weights_only`, which in the locked torch 2.14.1 resolves to `weights_only=True` unless `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` is set; nothing in the repository sets it. `UVR_MODEL_NAME` is limited to `UVR-DeNoise.pth` in `backend/app/config.py`, which keeps MDX and Demucs model names out; the Demucs loader calls `torch.load` with `weights_only=False`. That check limits the name only. audio-separator decides the architecture from the downloaded model list, and in the copy fetched on 2026-10-16 `UVR-DeNoise.pth` is listed as a VR model under a name without "VIP", which is what selects the `TRvlvr/model_repo` source over a VIP repository. `scripts/deploy_backend.sh` does not list `DENOISER_ENGINE` in `--set-env-vars`, and the default engine, OpGAN, does not download these files.

**System packages**: ffmpeg and libsndfile are installed from the Debian archive of the `python:3.11-slim` base image, which is referenced by tag, and neither is version-pinned (`Dockerfile:12`). Every upload is decoded by ffmpeg, so its parsers see untrusted bytes. The decode pins the demuxer from the validated format, allows only the file protocol, passes the MP4 demuxer's external-reference options as off, and caps the output by length and size (`docs/adr/006-input-decoding.md`). The ffmpeg version in use is logged at startup. Pinning is left for the deployment reproducibility work.

**Waived advisories**: none. The gate fails on any advisory pip-audit reports for the locked population.

### JavaScript dependencies

Frontend dependencies are audited in CI via `npm audit --audit-level=high`. The `package-lock.json` is committed and updated as part of any dependency change.

## Dependency Update Policy

- Python: packages are upgraded by name with `uv lock --upgrade-package <name>`; `uv.lock` committed after review. A blanket `uv lock --upgrade` is avoided because the benchmark protocol pins exact versions of numpy, soundfile, librosa, and soxr, and `load_protocol` refuses to run when they differ. In a test on 2026-10-08, a blanket upgrade moved librosa from 0.11.0 to 1.0.0 for Python 3.12 and later
- JavaScript: `npm audit fix` run when vulnerabilities are reported; `package-lock.json` committed after review
- CVE ignores are re-evaluated on each update cycle and removed as soon as a patched version is available and tested
