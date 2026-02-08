"""Pydantic request/response models."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class JobStatusEnum(StrEnum):
    """Valid states for a processing job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(BaseModel):
    """Processing job status returned by status/download endpoints."""

    job_id: str
    status: JobStatusEnum
    progress: int = Field(ge=-1, le=100)
    message: str
    created_at: datetime
    completed_at: datetime | None = None
    download_url: str | None = None
    processing_time: float | None = None
