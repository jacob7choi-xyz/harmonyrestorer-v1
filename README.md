# 🎵 HarmonyRestorer v1

**AI-powered audio restoration platform with cutting-edge research foundation**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![React](https://img.shields.io/badge/react-18.2+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)

## 🚧 Project Status

**Current Phase**: Full-stack foundation complete, AI implementation in progress

### ✅ **What's Working Now**
- **Professional FastAPI backend** with async processing and WebSocket support
- **Beautiful Apple-style React frontend** with glassmorphism design
- **Complete project architecture** ready for AI model integration
- **Production deployment setup** (Docker, Railway-ready)
- **Comprehensive API documentation** with Swagger/OpenAPI

### 🔬 **What's Coming Next**
- **1D Operational GANs implementation** based on cutting-edge research
- **Real audio processing pipeline** with professional-grade enhancement
- **Integration with latest 2025 AI models** (Demucs v4, SpeechT5, AudioSR)

## 🎯 **Vision & Research Foundation**

This project implements **"Blind Restoration of Real-World Audio by 1D Operational GANs"** - a breakthrough 2022 research paper achieving:
- **7.2 dB SDR improvement** on speech restoration
- **4.9 dB improvement** on music restoration  
- **First-ever blind restoration** (no prior assumptions about corruption types)

Combined with **latest 2025 AI models**:
- **Meta Demucs v4**: Advanced source separation
- **Microsoft SpeechT5**: Speech enhancement
- **AudioSR**: Diffusion-based super-resolution
- **NVIDIA Studio Voice NIM**: Sub-5ms latency processing

## ✨ **Planned Features**

### Core AI Capabilities
- 🤖 **1D Operational GANs** - Blind audio restoration with Self-ONNs
- 🎵 **Real-time Processing** - Live progress tracking via WebSocket
- 🎤 **Speech Enhancement** - Optimized voice clarity and intelligibility  
- 👤 **Voice Isolation** - Advanced source separation using Demucs v4
- 🏠 **Reverberation Removal** - AI-powered dereverberation
- 📈 **Audio Super-Resolution** - Upscaling with AudioSR diffusion models

### Technical Features  
- ⚡ **GPU Acceleration** - CUDA optimization for real-time processing
- 🌍 **Multiple Formats** - WAV, FLAC, MP3, AIFF support
- 🔄 **Batch Processing** - Handle multiple files simultaneously
- 📊 **Quality Metrics** - SDR, STOI, PESQ measurement and reporting
- 🌐 **RESTful API** - Professional API with comprehensive documentation
- 📱 **Progressive Web App** - Mobile-optimized interface

## 🎯 **Quick Demo**

```bash
# Health check (currently working)
curl -X GET "http://localhost:8000/health"

# Audio processing (coming soon)
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@your_audio.wav" \
  -F "settings={\"noise_reduction\":\"medium\",\"enhance_speech\":true}"
```

## 🏗️ **Architecture**

### Backend (FastAPI) - ✅ **Complete**
- **Framework**: FastAPI with async support and automatic OpenAPI docs
- **Real-time Updates**: WebSocket connections for live progress tracking  
- **Audio Pipeline**: Ready for PyTorch + TorchAudio integration
- **Database**: SQLModel for type-safe operations (PostgreSQL ready)
- **Background Tasks**: Async task processing with progress tracking
- **Security**: CORS, rate limiting, error handling

### Frontend (React + TypeScript) - ✅ **Complete**  
- **Framework**: React 18 with TypeScript for full type safety
- **Styling**: TailwindCSS with custom Apple-inspired glassmorphism
- **State Management**: React hooks with TypeScript interfaces
- **API Integration**: Axios client ready for backend connection
- **Audio Visualization**: Waveform components with Web Audio API
- **Build Tool**: Vite for lightning-fast development

### AI Models - 🔬 **In Development**
- **1D Operational GANs**: Self-ONNs implementation from research paper
- **Integration Layer**: Ready for Demucs v4, SpeechT5, AudioSR
- **Processing Pipeline**: Audio preprocessing and postprocessing
- **Performance Optimization**: GPU memory management and batching

## 🚀 **Quick Start**

### Prerequisites
- Python 3.11+
- Node.js 18+  
- Conda (recommended)
- Git

### Backend Setup
```bash
# Clone and setup
git clone https://github.com/jacob7choi-xyz/harmonyrestorer-v1.git
cd HarmonyRestorer-v1/backend

# Create environment
conda create -n harmonyrestorer-v1 python=3.11 -y
conda activate harmonyrestorer-v1

# Install dependencies
pip install fastapi "uvicorn[standard]" python-multipart websockets soundfile pydantic python-dotenv

# Run development server
python app/main.py
```

**Available at**: http://localhost:8000 (API docs at `/api/docs`)

### Frontend Setup
```bash
# Navigate and install
cd ../frontend
npm install

# Start development server  
npm run dev
```

**Available at**: http://localhost:3000

## 🔬 **Research Implementation Plan**

### Phase 1: Foundation Models (Current)
```python
# Integrate existing state-of-the-art models
- Meta Demucs v4 for source separation
- Microsoft SpeechT5 for speech enhancement  
- Basic noise reduction and audio processing
```

### Phase 2: 1D Op-GANs Implementation  
```python
# Implement research paper architecture
class SelfONN(nn.Module):
    # Self-Organized Operational Neural Networks
    # Equations 1-7 from research paper
    
class OpGANGenerator(nn.Module):
    # 10-layer architecture from Figure 4
    # Temporal + spectral loss functions
```

### Phase 3: Advanced Integration
```python
# Combine multiple state-of-the-art approaches
Audio → 1D Op-GANs → Demucs v4 → AudioSR → Enhanced Output
```

## 📊 **Target Performance Benchmarks**

### Quality Metrics (Research Goals)
- **SDR Improvement**: 7+ dB (speech), 5+ dB (music)
- **STOI Score**: 80%+ speech intelligibility  
- **Processing Speed**: 3-5x real-time on M2 MacBook

### Technical Performance  
- **API Latency**: <100ms for upload/status
- **Memory Efficiency**: Optimized GPU usage with cleanup
- **Throughput**: 100+ concurrent processing jobs

## 🛠️ **Development Roadmap**

### v0.1 (Current) - ✅ **Complete**
- [x] FastAPI backend with WebSocket support
- [x] React frontend with Apple-style design
- [x] Project structure and deployment setup
- [x] API documentation and type safety

### v0.2 (Next 2-4 weeks) - 🔬 **In Progress**
- [ ] Basic audio processing with existing models
- [ ] File upload and download functionality  
- [ ] Real progress tracking implementation
- [ ] Integration with Demucs or SpeechT5

### v1.0 (Target: 2-3 months)
- [ ] 1D Operational GANs implementation
- [ ] Multi-model processing pipeline
- [ ] Performance optimization and benchmarking
- [ ] Production deployment and monitoring

### v2.0 (Future)
- [ ] Custom model training interface
- [ ] Advanced preprocessing and postprocessing
- [ ] Enterprise features and API management
- [ ] Mobile app and offline capabilities

## 🤝 **Contributing**

This project implements cutting-edge research and welcomes contributions:

1. **AI/ML**: Help implement research papers and optimize models
2. **Backend**: Improve async processing and API performance  
3. **Frontend**: Enhance user experience and visualization
4. **Research**: Stay current with latest audio AI developments

## 📚 **Research References**

- **[1D Op-GANs Paper]**: "Blind Restoration of Real-World Audio by 1D Operational GANs" (2022)
- **[Meta Demucs v4]**: github.com/facebookresearch/demucs
- **[Microsoft SpeechT5]**: github.com/microsoft/SpeechT5  
- **[AudioSR]**: Diffusion-based audio super-resolution (2024)
- **[NVIDIA Studio Voice]**: Real-time speech enhancement

## 📞 **Connect**

- **GitHub**: Issues for bugs, discussions for questions
- **API Docs**: `/api/docs` when running locally
- **Research**: Following latest audio AI developments

---

**Building the future of audio restoration with cutting-edge AI research** 🎵

*Current focus: Implementing breakthrough 1D Operational GANs for blind audio restoration*
