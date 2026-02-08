"""Pydantic request/response models."""

from datetime import datetime

from pydantic import BaseModel


class JobStatus(BaseModel):
    """Processing job status."""

    job_id: str
    status: str
    progress: int
    message: str
    created_at: datetime
    completed_at: datetime | None = None
    download_url: str | None = None
    processing_time: float | None = None
