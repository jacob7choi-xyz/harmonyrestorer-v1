# backend/app/main.py
"""
HarmonyRestorer v1 - Audio Denoising Platform
"""

# ========================
# STANDARD LIBRARY IMPORTS
# ========================
import uuid
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Literal
from datetime import datetime
from contextlib import asynccontextmanager

# ========================
# THIRD-PARTY IMPORTS  
# ========================
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# ========================
# LOCAL/CUSTOM IMPORTS
# ========================
from app.services.denoiser import DenoiserService

# ========================
# CONFIGURATION & SETUP
# ========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"

# Create directories
for directory in [UPLOAD_DIR, PROCESSED_DIR]:
    directory.mkdir(exist_ok=True)

# ========================
# DATA MODELS
# ========================

class ProcessingSettings(BaseModel):
    """Audio processing configuration"""
    output_format: Literal["wav", "mp3", "flac"] = "wav"


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


# ========================
# CORE SERVICE
# ========================

class AudioProcessor:
    """Audio processor using UVR denoising"""
    
    def __init__(self):
        self.supported_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        self._denoiser: Optional[DenoiserService] = None
    
    def get_denoiser(self) -> DenoiserService:
        """Get or create DenoiserService (singleton)"""
        if self._denoiser is None:
            self._denoiser = DenoiserService(output_dir=PROCESSED_DIR)
            logger.info("✅ UVR Denoiser initialized")
        return self._denoiser
    
    def process_audio_file(self, input_path: Path, job_id: str) -> Path:
        """Denoise audio file"""
        denoiser = self.get_denoiser()
        output_path = denoiser.denoise(input_path)
        
        # Rename to standard format
        final_path = PROCESSED_DIR / f"{job_id}_denoised.wav"
        output_path.rename(final_path)
        
        return final_path


# Initialize processor
audio_processor = AudioProcessor()

# ========================
# JOB MANAGEMENT
# ========================

active_jobs: Dict[str, JobStatus] = {}


# ========================
# LIFESPAN MANAGEMENT
# ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🎵 HarmonyRestorer v1 starting up...")
    logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
    logger.info(f"📁 Processed directory: {PROCESSED_DIR}")
    logger.info("🚀 Ready to denoise audio!")
    yield
    logger.info("👋 HarmonyRestorer v1 shutting down...")


# ========================
# FASTAPI APP
# ========================

app = FastAPI(
    title="HarmonyRestorer v1",
    description="🎵 AI-powered audio denoising",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ========================
# API ENDPOINTS
# ========================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "platform": "HarmonyRestorer v1",
        "version": "1.0.0",
        "description": "AI-powered audio denoising using UVR",
        "status": "ready",
        "docs": "/api/docs",
        "supported_formats": list(audio_processor.supported_formats)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/denoise")
async def denoise_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and denoise audio file"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in audio_processor.supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file_ext}"
        )
    
    # Generate job ID and save file
    job_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}{file_ext}"
    
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
        message="Audio uploaded, queued for processing",
        created_at=datetime.now()
    )
    
    # Start background processing
    background_tasks.add_task(process_audio_background, job_id, input_path)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Audio uploaded and queued for denoising"
    }


def process_audio_background(job_id: str, input_path: Path):
    """Background processing task"""
    start_time = datetime.now()
    
    try:
        job = active_jobs[job_id]
        job.status = "processing"
        job.progress = 10
        job.message = "Denoising audio..."
        
        # Process
        output_path = audio_processor.process_audio_file(input_path, job_id)
        
        # Update job
        processing_time = (datetime.now() - start_time).total_seconds()
        job.status = "completed"
        job.progress = 100
        job.message = "Denoising complete"
        job.completed_at = datetime.now()
        job.download_url = f"/api/v1/download/{job_id}"
        job.processing_time = processing_time
        
        logger.info(f"✅ Completed: {job_id} in {processing_time:.1f}s")
        
    except Exception as e:
        logger.error(f"❌ Failed: {job_id} - {e}")
        job = active_jobs[job_id]
        job.status = "failed"
        job.progress = -1
        job.message = f"Processing failed: {str(e)}"
        job.completed_at = datetime.now()
    finally:
        input_path.unlink(missing_ok=True)


@app.get("/api/v1/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return active_jobs[job_id]


@app.get("/api/v1/download/{job_id}")
async def download_audio(job_id: str):
    """Download denoised audio"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")
    
    output_path = PROCESSED_DIR / f"{job_id}_denoised.wav"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(output_path),
        filename="denoised_audio.wav",
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
