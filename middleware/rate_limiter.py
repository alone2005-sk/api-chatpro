"""
Rate Limiter Middleware for DAMN BOT AI System
"""

import time
import asyncio
from typing import Dict, Any
from collections import defaultdict, deque
from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
        self.lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        client_id = self._get_client_id(request)

        async with self.lock:
            current_time = time.time()
            q = self.requests[client_id]

            # Remove old timestamps
            while q and q[0] < current_time - self.window_seconds:
                q.popleft()

            if len(q) >= self.max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Try again later."
                )

            q.append(current_time)

        return await call_next(request)

    def _get_client_id(self, request: Request) -> str:
        """Identify client by user ID header or IP"""
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return f"user:{user_id}"
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
