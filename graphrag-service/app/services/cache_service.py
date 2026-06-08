from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


_CACHE_LOCK = threading.Lock()


def load_json_cache(path: str) -> dict[str, Any]:
    """Load a small JSON cache file, returning an empty cache when unavailable."""

    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def save_json_cache(path: str, cache: dict[str, Any]) -> None:
    """Persist a JSON cache atomically enough for local demo workloads."""

    cache_path = Path(path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with _CACHE_LOCK:
            with temp_path.open("w", encoding="utf-8") as file:
                json.dump(cache, file, ensure_ascii=False)
            os.replace(temp_path, cache_path)
    except Exception:
        return


def get_cache_value(path: str, keys: list[str]) -> Any | None:
    """Return the first matching cached value from exact-to-loose keys."""

    cache = load_json_cache(path)
    for key in keys:
        if key in cache:
            return cache[key]
    return None


def set_cache_values(path: str, values: dict[str, Any]) -> None:
    """Merge cache values into an existing JSON cache."""

    cache = load_json_cache(path)
    cache.update(values)
    save_json_cache(path, cache)
