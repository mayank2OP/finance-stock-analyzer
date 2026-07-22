import threading
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable


_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="analysis-job")
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_MAX_JOBS = 100
logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update(job_id: str, **values: Any) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(values)


def _run(job_id: str, ticker: str, worker: Callable[[str], dict[str, Any]]) -> None:
    _update(job_id, status="running", stage="agent_analysis", progress_percent=15, updated_at=_now())
    try:
        result = worker(ticker)
        _update(
            job_id,
            status="completed",
            stage="completed",
            progress_percent=100,
            result=result,
            updated_at=_now(),
        )
    except Exception as exc:
        logger.exception("Background analysis job failed", extra={"job_id": job_id, "ticker": ticker})
        _update(
            job_id,
            status="failed",
            stage="failed",
            error="Analysis could not be completed. Please try again.",
            updated_at=_now(),
        )


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if not key.startswith("_")}


def create_analysis_job(
    ticker: str, worker: Callable[[str], dict[str, Any]], owner_id: int | None = None
) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "ticker": ticker,
        "status": "queued",
        "stage": "queued",
        "progress_percent": 0,
        "created_at": _now(),
        "updated_at": _now(),
        "_owner_id": owner_id,
    }
    with _lock:
        if len(_jobs) >= _MAX_JOBS:
            oldest = min(_jobs, key=lambda key: _jobs[key]["created_at"])
            _jobs.pop(oldest, None)
        _jobs[job_id] = job
    _executor.submit(_run, job_id, ticker, worker)
    return deepcopy(_public_job(job))


def get_analysis_job(job_id: str, owner_id: int | None = None) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        if not job or (owner_id is not None and job.get("_owner_id") != owner_id):
            return None
        return deepcopy(_public_job(job))
