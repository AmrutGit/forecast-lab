"""Retraining orchestration for the API's /retrain endpoints.

Retraining runs the full ml.pipeline.run_training pipeline in a background
thread (this is a portfolio-scale project — a real deployment would hand
this off to a task queue like Celery/RQ, but the job-status contract here is
written the same way so swapping the executor later wouldn't change the API).
Job status is tracked in an in-memory dict, keyed by job_id; this process
does not need to survive restarts for this project's purposes.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Literal

JobStatus = Literal["running", "completed", "failed"]


@dataclass
class RetrainJob:
    job_id: str
    status: JobStatus = "running"
    detail: str | None = None
    result: dict | None = None


class RetrainingService:
    def __init__(self):
        self._jobs: dict[str, RetrainJob] = {}
        self._lock = threading.Lock()

    def start_job(self) -> str:
        job_id = str(uuid.uuid4())
        job = RetrainJob(job_id=job_id, status="running")
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
        thread.start()
        return job_id

    def _run(self, job_id: str) -> None:
        from ml.pipeline.run_training import run_training_pipeline

        try:
            metadata, promotion = run_training_pipeline()
            with self._lock:
                self._jobs[job_id].status = "completed"
                self._jobs[job_id].result = {
                    "version": metadata.version,
                    "trained_at": metadata.trained_at,
                    "data_version": metadata.data_version,
                    "metrics": metadata.metrics,
                    "params": metadata.params,
                    "promoted": promotion.promoted,
                    "promotion_reason": promotion.reason,
                }
        except Exception as exc:  # noqa: BLE001 - surface any training failure via job status
            with self._lock:
                self._jobs[job_id].status = "failed"
                self._jobs[job_id].detail = str(exc)

    def get_job(self, job_id: str) -> RetrainJob | None:
        with self._lock:
            return self._jobs.get(job_id)


retraining_service = RetrainingService()
