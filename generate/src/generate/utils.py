"""Small sampling helpers shared across vector generators."""

from __future__ import annotations

import datetime as dt

import numpy as np

from .config import BASE_DATE, HISTORY_WINDOW_DAYS

_BASE = dt.datetime.fromisoformat(BASE_DATE)


def random_timestamp(rng: np.random.Generator, window_days: int = HISTORY_WINDOW_DAYS) -> str:
    offset_seconds = int(rng.integers(0, window_days * 86_400))
    ts = _BASE - dt.timedelta(seconds=offset_seconds)
    return ts.isoformat(timespec="seconds")


def timestamp_at_hour(rng: np.random.Generator, hour: int, window_days: int = HISTORY_WINDOW_DAYS) -> str:
    day_offset = int(rng.integers(0, window_days))
    minute = int(rng.integers(0, 60))
    ts = _BASE - dt.timedelta(days=day_offset)
    ts = ts.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return ts.isoformat(timespec="seconds")


def lognormal_amount(rng: np.random.Generator, mean: float = 4.0, sigma: float = 0.8) -> float:
    return float(np.round(rng.lognormal(mean=mean, sigma=sigma), 2))
