from __future__ import annotations

import time


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.events: dict[int, list[float]] = {}

    def allow(self, user_id: int) -> bool:
        now = time.time()
        bucket = self.events.setdefault(user_id, [])
        bucket[:] = [ts for ts in bucket if now - ts <= self.window_seconds]
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(now)
        return True
