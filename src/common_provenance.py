"""Provenance stamping for result files.

A number in a paper must be traceable to the code and config that produced it.
Every script that writes to results/ calls ``stamp`` so the git SHA, the dirty
flag, the config snapshot and the timestamp land next to the output.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_state(root: Path) -> dict:
    def run(*a):
        try:
            return subprocess.run(a, cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "sha": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def stamp(out_dir: Path, script: str, root: Path, **extra) -> Path:
    """Write provenance.json beside a result. Returns the path written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "script": script,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(root),
        **extra,
    }
    path = out_dir / "provenance.json"
    existing = json.loads(path.read_text()) if path.exists() else []
    if isinstance(existing, dict):
        existing = [existing]
    existing.append(rec)
    path.write_text(json.dumps(existing, indent=2))
    return path
