"""An LRU cache for explainer generations, keyed by the exact prompt.

Why it is exact: the explainer decodes greedily (temperature 0) and resets the KV cache before
each prompt, so a given prompt always yields the same text. The key is the whole prompt plus
``max_new_tokens`` and the stop condition, so two calls share an entry only when they would
decode the same tokens. A hit returns the stored text, so the output text is unchanged; only the
LLM call is skipped. Each returned timing carries ``cached``, and a hit has no ``ttft_ms`` or
tok/s keys because nothing was decoded.

Latency, energy and soak measurements of the LLM must run with the cache off
(``--explainer-cache 0``), otherwise the hits make the per-call times look far faster than the
model is.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any


def _stop_key(stop_when: Any) -> str | None:
    if stop_when is None:
        return None
    module = getattr(stop_when, "__module__", None)
    qualname = getattr(stop_when, "__qualname__", None)
    if module is None or qualname is None:
        return repr(stop_when)
    return f"{module}.{qualname}"


class CachedGenerator:
    """Wraps an object with ``generate(prompt, max_new_tokens=..., stop_when=...)``."""

    def __init__(self, inner: Any, capacity: int = 256, clock=time.perf_counter) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"capacity must be an int >= 1, got {capacity!r}")
        self.inner = inner
        self.capacity = capacity
        self.clock = clock
        self._entries: collections.OrderedDict[tuple, tuple[str, dict[str, Any]]] = (
            collections.OrderedDict()
        )
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def generate(self, prompt: str, max_new_tokens: int = 64, stop_when=None):
        key = (prompt, int(max_new_tokens), _stop_key(stop_when))
        t0 = self.clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                self.hits += 1
        if entry is not None:
            text, stored = entry
            wall_ms = round((self.clock() - t0) * 1000.0, 3)
            return text, {
                "wall_ms": wall_ms,
                "new_tokens": stored["new_tokens"],
                "prompt_tokens": stored["prompt_tokens"],
                "stop_hit": stored["stop_hit"],
                "cached": True,
            }

        # The lock is not held while the model runs. If inner raises, nothing is stored.
        text, timing = self.inner.generate(
            prompt, max_new_tokens=max_new_tokens, stop_when=stop_when
        )
        stored = {
            "new_tokens": timing.get("new_tokens"),
            "prompt_tokens": timing.get("prompt_tokens"),
            "stop_hit": timing.get("stop_hit"),
        }
        with self._lock:
            self._entries[key] = (text, stored)
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
            self.misses += 1
        out = dict(timing)
        out["cached"] = False
        return text, out

    def stats(self) -> dict[str, int]:
        with self._lock:
            size = len(self._entries)
            hits, misses = self.hits, self.misses
        return {"capacity": self.capacity, "size": size, "hits": hits, "misses": misses}

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes not found normally; forwards to the wrapped runner.
        if name in ("inner", "_entries", "_lock"):
            raise AttributeError(name)
        return getattr(self.inner, name)
