"""Hard caps on live mandate-demo agent runs, enforced server-side.

Every live run makes real OpenAI API calls (billed to whoever's
OPENAI_API_KEY the server process has). Since this prototype may be run
by judges outside the author's control, both a per-browser-session cap
and a global per-minute cap are enforced here, in the backend, before
any live run is allowed to start -- not just left to good behavior in
the UI. In-memory and per-process by design: this is a local single-
instance demo, not a distributed service, so a simple lock-protected
counter is the right amount of machinery, not a placeholder for
something bigger.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

MAX_LIVE_RUNS_PER_SESSION = int(os.environ.get("MANDATE_DEMO_MAX_RUNS_PER_SESSION", "5"))
MAX_LIVE_RUNS_PER_MINUTE = int(os.environ.get("MANDATE_DEMO_MAX_RUNS_PER_MINUTE", "3"))


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    reason: str | None
    session_runs_used: int
    session_runs_remaining: int
    global_runs_in_last_minute: int


class LiveRunRateLimiter:
    def __init__(self, max_per_session: int = MAX_LIVE_RUNS_PER_SESSION, max_per_minute: int = MAX_LIVE_RUNS_PER_MINUTE) -> None:
        self._max_per_session = max_per_session
        self._max_per_minute = max_per_minute
        self._lock = threading.Lock()
        self._session_counts: dict[str, int] = defaultdict(int)
        self._global_run_timestamps: deque[float] = deque()

    @property
    def max_per_session(self) -> int:
        return self._max_per_session

    @property
    def max_per_minute(self) -> int:
        return self._max_per_minute

    def _prune_global_window(self, now: float) -> None:
        cutoff = now - 60.0
        while self._global_run_timestamps and self._global_run_timestamps[0] < cutoff:
            self._global_run_timestamps.popleft()

    def check_and_reserve(self, session_id: str) -> RateLimitDecision:
        """Atomically checks both caps and, if allowed, reserves a slot
        (counts it immediately) so concurrent requests can't race past
        the limit between check and use."""

        with self._lock:
            now = time.monotonic()
            self._prune_global_window(now)

            session_used = self._session_counts[session_id]
            global_used = len(self._global_run_timestamps)

            if session_used >= self._max_per_session:
                return RateLimitDecision(
                    allowed=False,
                    reason=f"This session has used all {self._max_per_session} live runs it's allowed.",
                    session_runs_used=session_used,
                    session_runs_remaining=0,
                    global_runs_in_last_minute=global_used,
                )

            if global_used >= self._max_per_minute:
                return RateLimitDecision(
                    allowed=False,
                    reason=f"Global limit of {self._max_per_minute} live runs per minute reached across all users -- try again shortly.",
                    session_runs_used=session_used,
                    session_runs_remaining=self._max_per_session - session_used,
                    global_runs_in_last_minute=global_used,
                )

            self._session_counts[session_id] = session_used + 1
            self._global_run_timestamps.append(now)

            return RateLimitDecision(
                allowed=True,
                reason=None,
                session_runs_used=session_used + 1,
                session_runs_remaining=self._max_per_session - (session_used + 1),
                global_runs_in_last_minute=global_used + 1,
            )

    def status(self, session_id: str) -> RateLimitDecision:
        with self._lock:
            now = time.monotonic()
            self._prune_global_window(now)
            session_used = self._session_counts[session_id]
            global_used = len(self._global_run_timestamps)
            return RateLimitDecision(
                allowed=session_used < self._max_per_session and global_used < self._max_per_minute,
                reason=None,
                session_runs_used=session_used,
                session_runs_remaining=max(self._max_per_session - session_used, 0),
                global_runs_in_last_minute=global_used,
            )


live_run_limiter = LiveRunRateLimiter()
