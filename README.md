# HarmonyRestorer v1

AI-powered audio denoising API.

## What it does

Upload noisy audio → Get clean audio back.

- **Model**: UVR-DeNoise-Lite
- **Speed**: ~2 seconds for 5 seconds of audio (M4 Mac)
- **Formats**: WAV, MP3, FLAC, OGG, M4A, AAC

## Quick Start
```bash
git clone https://github.com/jacob7choi-xyz/harmonyrestorer-v1.git
cd harmonyrestorer-v1

conda env create -f environment.yml
conda activate harmonyrestorer-v1

cd backend
uvicorn app.main:app --reload --port 8000
```

## API
```bash
# Denoise
curl -X POST "http://localhost:8000/api/v1/denoise" -F "file=@audio.wav"

# Check status
curl http://localhost:8000/api/v1/status/{job_id}

# Download
curl -O http://localhost:8000/api/v1/download/{job_id}
```

Docs: http://localhost:8000/api/docs

## Architecture
```
backend/app/
├── main.py              # FastAPI app
├── services/
│   └── denoiser.py      # UVR wrapper
└── _archive/            # OpGAN implementation
    ├── op_gan.py        # Generator/Discriminator
    ├── self_onn.py      # Self-ONN layers
    └── opgan_restorer.py # Inference pipeline
```

## OpGAN Implementation (Archived)

The `_archive/` folder contains a from-scratch implementation of **1D Operational GANs** based on [Kiranyaz et al. 2022](https://arxiv.org/abs/2110.10149).

### What's implemented

- **Self-ONN layers**: Learnable nonlinear operators (sin, cos, tanh, exp) with softmax-weighted combinations per neuron
- **U-Net Generator**: 10-layer encoder-decoder with skip connections
- **Inference pipeline**: Chunked processing with overlap-add, automatic resampling, stereo support

### Stability engineering

The original Self-ONN architecture is prone to gradient explosion. Fixes implemented:

- Gradient clipping at 1.0
- Reduced operator count (q=3 vs q=5 in paper)
- Conservative weight initialization
- GroupNorm in bottleneck layer
- Output clamping to [-1, 1]

These changes allow the model to train without NaN/Inf gradients.

### Current status

Architecture is **complete and trainable**, but **not yet trained**. Requires:

1. Paired noisy/clean audio dataset
2. Training script
3. GPU compute time

The production API uses pretrained UVR models while OpGAN training is in progress.

## Roadmap

- [x] Working denoising API (UVR)
- [ ] Build/acquire paired training dataset
- [ ] Train OpGAN
- [ ] Benchmark OpGAN vs UVR (SDR, PESQ, STOI)
- [ ] Swap in OpGAN if it outperforms UVR

## License

MIT
