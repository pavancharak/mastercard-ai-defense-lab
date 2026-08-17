"""Launches the mock merchant API on a real local HTTP socket, bound only
to 127.0.0.1 (never 0.0.0.0), on an OS-assigned free port. Everything the
agent does goes over real HTTP to this address -- so "nothing left the
local mock API" can be verified directly (see runner.py's call log),
not just asserted.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time

import requests
import uvicorn

from .mock_merchant import app


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LocalMerchantServer:
    def __init__(self) -> None:
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self, timeout_seconds: float = 10.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=1)
                if r.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.1)
        raise RuntimeError("local mock merchant API did not become healthy in time")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
