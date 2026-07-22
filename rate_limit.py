import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, status


_events: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def enforce_rate_limit(key: str, limit: int, window_seconds: int) -> None:
    """Small single-instance limiter suitable for the zero-cost demo deployment."""
    now = time.monotonic()
    cutoff = now - window_seconds
    with _lock:
        bucket = _events[key]
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(window_seconds - (now - bucket[0])))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Request limit reached. Please try again later.",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)


def clear_rate_limits() -> None:
    """Used only by isolated automated tests."""
    with _lock:
        _events.clear()
