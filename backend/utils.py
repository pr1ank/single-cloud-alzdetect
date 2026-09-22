"""Paths, atomic persistence, and bounded public HTTP downloads."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def public_json(url: str):
    # Prevent implicit netrc authentication and proxy credentials.
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(url, timeout=(10, 30))
        response.raise_for_status()
        return response.json()


def download(url: str, target: Path, max_bytes=350_000_000, deadline=240):
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    start, size = time.monotonic(), 0
    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.get(url, stream=True, timeout=(10, 30)) as response:
                response.raise_for_status()
                if int(response.headers.get("Content-Length", 0)) > max_bytes:
                    raise ValueError("Public download exceeds the automatic size budget")
                with part.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        size += len(chunk)
                        if size > max_bytes or time.monotonic() - start > deadline:
                            raise TimeoutError("Public download exceeded size/time budget")
                        output.write(chunk)
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
    return target
