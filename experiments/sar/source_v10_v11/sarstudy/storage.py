"""Atomic durable records; refuse accidental cross-protocol reuse."""
import hashlib
import json
import os
from pathlib import Path

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def file_hash(path):
    result = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def require_identity(path, identity):
    path = Path(path)
    if path.exists() and read_json(path) != identity:
        raise ValueError(f"Run/split configuration changed: {path}. Choose a new experiment ID.")
    if not path.exists():
        atomic_json(path, identity)
