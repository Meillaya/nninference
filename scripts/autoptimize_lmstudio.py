#!/usr/bin/env python3
# ─── How to run ───
# uv run python scripts/autoptimize_lmstudio.py
"""LM Studio availability probing for the optimization harness."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from typing import TypedDict
from urllib.parse import urlparse


class LMStudioStatusJson(TypedDict):
    status: str
    reason: str
    cli_path: str | None
    server_url: str | None
    version: str | None
    loaded_models: str | None
    server_reachable: bool
    tokens_per_second: float | None
    ttft_seconds: float | None
    model_load_time_seconds: float | None


@dataclass(frozen=True, slots=True)
class LMStudioProbe:
    status: str
    reason: str
    cli_path: str | None
    server_url: str | None
    version: str | None
    loaded_models: str | None
    server_reachable: bool

    def to_json(self) -> LMStudioStatusJson:
        return {
            "status": self.status,
            "reason": self.reason,
            "cli_path": self.cli_path,
            "server_url": self.server_url,
            "version": self.version,
            "loaded_models": self.loaded_models,
            "server_reachable": self.server_reachable,
            "tokens_per_second": None,
            "ttft_seconds": None,
            "model_load_time_seconds": None,
        }


def _server_reachable(url: str, timeout_seconds: float = 0.2) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname
    if host is None:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _run_text(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = proc.stdout.strip() or proc.stderr.strip()
    return text or None


def detect_lmstudio() -> LMStudioProbe:
    """Return deterministic LM Studio comparator status without failing the run."""
    cli_path = shutil.which("lms")
    server_url = os.environ.get("LM_STUDIO_URL")
    if cli_path is None:
        return LMStudioProbe(
            status="skipped_with_reason",
            reason="lms CLI not found on PATH; LM Studio comparison skipped",
            cli_path=None,
            server_url=server_url,
            version=None,
            loaded_models=None,
            server_reachable=False,
        )
    version = _run_text([cli_path, "--version"])
    loaded_models = _run_text([cli_path, "ps"])
    reachable = _server_reachable(server_url) if server_url else False
    if server_url and not reachable:
        return LMStudioProbe(
            status="skipped_with_reason",
            reason=f"LM Studio server unreachable at {server_url}; comparison skipped",
            cli_path=cli_path,
            server_url=server_url,
            version=version,
            loaded_models=loaded_models,
            server_reachable=False,
        )
    return LMStudioProbe(
        status="available_unmeasured",
        reason="lms CLI metadata detected; live benchmark adapter is not executed during dry-run",
        cli_path=cli_path,
        server_url=server_url,
        version=version,
        loaded_models=loaded_models,
        server_reachable=reachable,
    )


if __name__ == "__main__":
    import json

    print(json.dumps(detect_lmstudio().to_json(), indent=2))
