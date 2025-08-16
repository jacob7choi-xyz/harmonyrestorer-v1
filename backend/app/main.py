# backend/app/main.py
"""
HarmonyRestorer v1 - Enterprise Audio Processing Platform
CTO-approved architecture with proper separation of concerns
"""

# ========================
# STANDARD LIBRARY IMPORTS
# ========================
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from contextlib import asynccontextmanager

# ========================
# THIRD-PARTY IMPORTS  
# ========================
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
import torch

# ========================
# LOCAL/CUSTOM IMPORTS
# ========================
from app.services.opgan_restorer import OpGANRestorer
from app.ml_models.op_gan import OpGANGenerator

# ========================
# CONFIGURATION & SETUP
# ========================

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
MODEL_DIR = BASE_DIR / "model_checkpoints"

# Device
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)

# Create directories
for directory in [UPLOAD_DIR, PROCESSED_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True)
    

# ========================
# DATA MODELS (Keep these - they're good)
# ========================

class ProcessingSettings(BaseModel):
    """Audio processing configuration with type-safe enums"""
    intensity: Literal["light", "medium", "strong"] = "medium"
    output_format: Literal["wav", "mp3", "flac", "ogg"] = "wav"
    quality: Literal["low", "medium", "high", "ultra"] = "high"


class JobStatus(BaseModel):
    """Processing job status"""
    job_id: str
    status: str
    progress: int
    message: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    download_url: Optional[str] = None
    processing_time: Optional[float] = None
    file_info: Optional[Dict[str, Any]] = None


class AudioInfo(BaseModel):
    """Audio file information"""
    duration: float
    sample_rate: int
    channels: int
    format: str
    file_size: int
    bitrate: Optional[int] = None
    peak_db: Optional[float] = None
    rms_db: Optional[float] = None


# ========================
# CORE SERVICES (Simplified)
# ========================

class AudioProcessor:
    """Simplified audio processor using existing services"""
    
    def __init__(self):
        self.device = DEVICE
        self.supported_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        self._restorer = None
    
    def get_restorer(self) -> OpGANRestorer:
        """Get or create OpGANRestorer (singleton)"""
        if self._restorer is None:
            generator = OpGANGenerator(input_length=32000, q=3)
            generator.eval()
            
            self._restorer = OpGANRestorer(
                generator=generator,
                model_sr=16000,
                frame_len=32000,
                overlap=0.5,
                fade_ms=20,
                batch_frames=8,
                headroom=0.9
            )
            logger.info("✅ OpGANRestorer initialized")
        return self._restorer
    
    async def process_audio_file(
        self, 
        input_path: Path, 
        output_path: Path, 
        settings: ProcessingSettings,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Process audio file using OpGANRestorer"""
        start_time = datetime.now()
        
        try:
            if progress_callback:
                await progress_callback(10, "🎵 Loading audio...")
            
            # Use your existing OpGANRestorer
            restorer = self.get_restorer()
            
            if progress_callback:
                await progress_callback(20, "🧠 Processing with Op-GANs...")
            
            # Load, process, save using existing code
            import torchaudio
            waveform, sample_rate = torchaudio.load(str(input_path))
            
            # Convert to numpy for OpGANRestorer
            audio_np = waveform.numpy()
            
            # Process with your breakthrough implementation
            enhanced_audio = restorer.restore_track(audio_np, sr=sample_rate)
            enhanced_tensor = torch.from_numpy(enhanced_audio)

            # Apply intensity blending based on settings
            intensity_map = {"light": 0.3, "medium": 0.5, "strong": 0.7, "extreme": 0.8}
            blend_factor = intensity_map.get(settings.noise_reduction, 0.2)

            # Blend enhanced with original
            if blend_factor < 1.0:
                enhanced_tensor = (1 - blend_factor) * waveform + blend_factor * enhanced_tensor

            # Convert back to numpy for saving
            enhanced_audio = enhanced_tensor.numpy()
            
            if progress_callback:
                await progress_callback(90, "💾 Saving enhanced audio...")
            
            # Save result
            enhanced_tensor = torch.from_numpy(enhanced_audio)
            torchaudio.save(str(output_path), enhanced_tensor, sample_rate)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if progress_callback:
                await progress_callback(100, "🎉 Complete!")
            
            return {
                "success": True,
                "processing_time": processing_time,
                "output_path": str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            if progress_callback:
                await progress_callback(-1, f"❌ Error: {str(e)}")
            raise


# Initialize processor
audio_processor = AudioProcessor()

# ========================
# JOB MANAGEMENT (Simplified)
# ========================

active_jobs: Dict[str, JobStatus] = {}


async def update_job_progress(job_id: str, progress: int, message: str):
    """Update job progress"""
    if job_id in active_jobs:
        job = active_jobs[job_id]
        job.progress = progress
        job.message = message
        
        if progress == 100:
            job.status = "completed"
            job.completed_at = datetime.now()
        elif progress == -1:
            job.status = "failed"
            job.completed_at = datetime.now()


# ========================
# LIFESPAN MANAGEMENT
# ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🎵 HarmonyRestorer v1 starting up...")
    logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
    logger.info(f"📁 Processed directory: {PROCESSED_DIR}")
    logger.info(f"🤖 Device: {DEVICE}")
    logger.info("🚀 Ready to restore audio!")
    
    yield
    
    # Shutdown
    logger.info("👋 HarmonyRestorer v1 shutting down...")

# ========================
# FASTAPI APP
# ========================

app = FastAPI(
    title="HarmonyRestorer v1",
    description="🎵 Enterprise AI audio restoration platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ========================
# API ENDPOINTS (Simplified)
# ========================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "platform": "HarmonyRestorer v1",
        "version": "1.0.0",
        "description": "🎵 Enterprise AI audio restoration",
        "status": "🚀 Ready for audio magic!",
        "docs": "/api/docs",
        "device": str(DEVICE),
        "supported_formats": list(audio_processor.supported_formats)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE)
    }


@app.post("/api/v1/process")
async def process_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings: str = Form(default='{"noise_reduction": "medium"}')
):
    """Upload and process audio file"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in audio_processor.supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format {file_ext}"
        )
    
    # Parse settings
    try:
        processing_settings = ProcessingSettings.parse_raw(settings)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid settings: {e}")
    
    # Generate job ID and paths
    job_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}{file_ext}"
    output_path = PROCESSED_DIR / f"{job_id}_enhanced.wav"
    
    # Save uploaded file
    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)
        logger.info(f"📁 Saved upload: {input_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Create job entry
    active_jobs[job_id] = JobStatus(
        job_id=job_id,
        status="queued",
        progress=0,
        message="🎵 Audio uploaded, queued for processing",
        created_at=datetime.now()
    )
    
    # Start background processing
    background_tasks.add_task(
        process_audio_background,
        job_id,
        input_path,
        output_path,
        processing_settings
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "🎵 Audio uploaded and queued for processing"
    }


async def process_audio_background(
    job_id: str,
    input_path: Path,
    output_path: Path,
    settings: ProcessingSettings
):
    """Background processing task"""
    try:
        active_jobs[job_id].status = "processing"
        await update_job_progress(job_id, 1, "🚀 Starting processing...")
        
        async def progress_callback(progress: int, message: str):
            await update_job_progress(job_id, progress, message)
        
        # Process using existing services
        result = await audio_processor.process_audio_file(
            input_path,
            output_path,
            settings,
            progress_callback
        )
        
        # Update job
        job = active_jobs[job_id]
        job.download_url = f"/api/v1/download/{job_id}"
        job.processing_time = result["processing_time"]
        
        logger.info(f"✅ Processing completed: {job_id}")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {job_id} - {e}")
        await update_job_progress(job_id, -1, f"❌ Processing failed: {str(e)}")
    finally:
        input_path.unlink(missing_ok=True)


@app.get("/api/v1/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return active_jobs[job_id]


@app.get("/api/v1/download/{job_id}")
async def download_processed_audio(job_id: str):
    """Download processed audio"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")
    
    output_files = list(PROCESSED_DIR.glob(f"{job_id}_enhanced.*"))
    if not output_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(output_files[0]),
        filename=f"enhanced_audio.wav",
        media_type="audio/wav"
    )

# ========================
# DEVELOPMENT SERVER
# ========================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )