"""
app/model_loader.py — Sync model artifacts from S3 at runtime.

On a deployed server the freshest model is produced nightly by CI and uploaded
to S3 under the prefix "models/latest/". This module downloads those objects
into the local model_artifacts/ and preprocessed/ directories so the serving
app can load them.

Contract (consumed by app/main.py):
    - BUCKET            module-level, read from MODEL_S3_BUCKET (None in local dev)
    - sync_from_s3()    download every object under models/latest/ into the
                        appropriate local dir; no-op if BUCKET is unset
    - check_and_reload(reload_callback)
                        re-sync only if the manifest ETag changed, then invoke
                        reload_callback() so the app can reload the new model
    - status()          dict describing the loader's current state

If MODEL_S3_BUCKET is unset (local development) every function is a safe no-op,
letting the app fall back to whatever artifacts are already on disk.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Optional

# Prefix under which CI uploads the current model bundle.
S3_PREFIX = "models/latest/"
# Object whose ETag we watch to detect a fresh upload.
MANIFEST_KEY = S3_PREFIX + "manifest.json"

# Local destination directories.
MODEL_DIR = os.environ.get("MODEL_DIR", "model_artifacts")
PREPROCESSED_DIR = os.environ.get("PREPROCESSED_DIR", "preprocessed")

# Bucket comes from the environment; unset means local dev -> no-op.
BUCKET: Optional[str] = os.environ.get("MODEL_S3_BUCKET") or None

# Filenames that belong in model_artifacts/; everything else goes to preprocessed/.
_MODEL_ARTIFACT_FILES = {
    "model.pt",
    "metadata.json",
    "metrics.json",
    "comparison.json",
    "comparison.txt",
    "index_to_label.json",
    "label_to_index.json",
    "train_config.json",
}

_lock = threading.Lock()
_last_manifest_etag: Optional[str] = None
_last_synced_keys: int = 0


def _dest_dir_for(filename: str) -> str:
    """Route a basename to model_artifacts/ or preprocessed/."""
    if filename in _MODEL_ARTIFACT_FILES:
        return MODEL_DIR
    return PREPROCESSED_DIR


def _client():
    import boto3  # imported lazily so local dev without boto3 still works

    return boto3.client("s3")


def _get_manifest_etag(client) -> Optional[str]:
    """Return the current ETag of the manifest object, or None if absent."""
    try:
        head = client.head_object(Bucket=BUCKET, Key=MANIFEST_KEY)
        return head.get("ETag")
    except Exception:
        return None


def sync_from_s3() -> bool:
    """
    Download every object under S3_PREFIX into the correct local directory.

    Returns True if a sync ran, False if it was a no-op (no bucket configured).
    """
    global _last_manifest_etag, _last_synced_keys

    if not BUCKET:
        return False

    with _lock:
        client = _client()
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(PREPROCESSED_DIR, exist_ok=True)

        paginator = client.get_paginator("list_objects_v2")
        count = 0
        for page in paginator.paginate(Bucket=BUCKET, Prefix=S3_PREFIX):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Skip "directory" placeholder keys.
                if key.endswith("/"):
                    continue
                filename = os.path.basename(key)
                if not filename:
                    continue
                dest = os.path.join(_dest_dir_for(filename), filename)
                client.download_file(BUCKET, key, dest)
                count += 1

        _last_synced_keys = count
        _last_manifest_etag = _get_manifest_etag(client)
        return True


def check_and_reload(reload_callback: Callable[[], None]) -> bool:
    """
    Re-sync from S3 only if the manifest ETag changed since the last sync,
    then invoke reload_callback() to let the app pick up the new artifacts.

    Returns True if a reload happened, False otherwise.
    """
    global _last_manifest_etag

    if not BUCKET:
        return False

    client = _client()
    current = _get_manifest_etag(client)
    if current is not None and current == _last_manifest_etag:
        return False

    sync_from_s3()
    reload_callback()
    return True


def status() -> dict:
    """Return a small dict describing loader state (safe in all environments)."""
    return {
        "bucket": BUCKET,
        "enabled": bool(BUCKET),
        "prefix": S3_PREFIX,
        "last_manifest_etag": _last_manifest_etag,
        "last_synced_keys": _last_synced_keys,
        "model_dir": MODEL_DIR,
        "preprocessed_dir": PREPROCESSED_DIR,
    }
