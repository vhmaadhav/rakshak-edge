"""ONNX Runtime session options for offline use.

``offline_session_options`` turns off ONNX Runtime's telemetry events (once per process, when the
installed onnxruntime provides ``disable_telemetry_events``) and returns a fresh ``SessionOptions``
for the caller to tune. onnxruntime is imported inside the function, so importing this module
never loads it.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_TELEMETRY_OFF = False


def _disable_telemetry_once(ort) -> None:
    global _TELEMETRY_OFF
    with _LOCK:
        if _TELEMETRY_OFF:
            return
        disable = getattr(ort, "disable_telemetry_events", None)
        if callable(disable):
            try:
                disable()
            except Exception:  # best effort: a failing telemetry switch must not block inference
                pass
        _TELEMETRY_OFF = True


def offline_session_options(ort=None):
    """Return a new ``onnxruntime.SessionOptions`` with telemetry events disabled.

    Pass ``ort`` when the caller already imported onnxruntime (keeps its own ImportError
    handling); otherwise it is imported here.
    """
    if ort is None:
        import onnxruntime as ort
    _disable_telemetry_once(ort)
    return ort.SessionOptions()
