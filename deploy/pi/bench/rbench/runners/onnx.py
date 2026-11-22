"""ONNX Runtime runners on CPUExecutionProvider, one item per call (batch size 1).

Featurisation is the repo's own code:
- SMS (dual char-CNN): rakshak.models.features.featurize (raw and canonical bytes, canonical
  rule flags), exactly as sms_student.py exported the model.
- URL (char-CNN + side features): rakshak.models.encoding.encode_bytes (max_len 200) and
  rakshak.url_features.url_features, standardised with the job's url_side_scaler.json through
  rakshak.models.encoding.apply_standardizer.
- NIDS (MLP): the caller supplies the 19 live features as float32 rows.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from rbench.errors import MissingDependency

URL_MAX_LEN = 200


def _ort():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise MissingDependency(
            "onnxruntime is not installed (pip install onnxruntime in ~/rakshak-bench/.venv)"
        ) from exc
    return ort


def make_session(path: Path, threads: int):
    ort = _ort()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(threads)
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    z = math.exp(logit)
    return z / (1.0 + z)


class _Base:
    kind = ""
    input_names: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.session: Any = None
        self.threads = 1

    def load(self, path: Path, threads: int, extra: dict[str, Path]) -> None:
        self.threads = int(threads)
        self.session = make_session(path, threads)
        names = tuple(i.name for i in self.session.get_inputs())
        if names != self.input_names:
            raise MissingDependency(
                f"model inputs {names} do not match runner {self.kind} "
                f"(expected {self.input_names})"
            )
        self.load_extra(extra)

    def load_extra(self, extra: dict[str, Path]) -> None:  # noqa: B027 - optional hook
        pass

    def run(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self.session.run(None, feed)


class SmsRunner(_Base):
    kind = "onnx_sms"
    input_names = ("raw", "canon", "side")

    def __init__(self) -> None:
        super().__init__()
        try:
            from rakshak.models.features import featurize
        except ImportError as exc:
            raise MissingDependency(
                f"featurisation needs regex, numpy and tldextract: {exc}"
            ) from exc
        self._featurize = featurize

    def make_feed(self, item: dict[str, Any]) -> dict[str, np.ndarray]:
        return self.make_feed_batch([item])

    def make_feed_batch(
        self,
        items: list[dict[str, Any]],
        *,
        canons: list[Any] | None = None,
        canon_rules: list[Any] | None = None,
    ) -> dict[str, np.ndarray]:
        """Featurise N messages into one feed (one row per message, same as N single feeds).

        ``canons`` and ``canon_rules`` are optional precomputed canonicalisations and canonical rule
        evidence (see rakshak.models.features.featurize). The feed reads only the canonical rule
        flags, so the raw-view rules are not computed here (raw_side=False).
        """
        feats = self._featurize(
            [item["text"] for item in items],
            buckets=None,
            canons=canons,
            canon_rules=canon_rules,
            raw_side=False,
        )
        return {
            "raw": feats.raw_bytes.astype(np.int64),
            "canon": feats.canon_bytes.astype(np.int64),
            "side": feats.side_canon.astype(np.float32),
        }

    def score(self, outputs: list[np.ndarray]) -> tuple[float, float]:
        logit = float(np.asarray(outputs[0]).reshape(-1)[0])
        return _sigmoid(logit), logit

    def score_batch(self, outputs: list[np.ndarray]) -> list[float]:
        """Scam probability per row. Uses the scalar _sigmoid per logit, so each probability is
        bit-identical to what ``score`` gives for the same logit."""
        logits = np.asarray(outputs[0]).reshape(-1)
        return [_sigmoid(float(x)) for x in logits]


class UrlRunner(_Base):
    kind = "onnx_url"
    input_names = ("raw", "side")

    def __init__(self) -> None:
        super().__init__()
        try:
            from rakshak.models.encoding import apply_standardizer, encode_bytes
            from rakshak.url_features import url_features
        except ImportError as exc:
            raise MissingDependency(f"URL featurisation needs regex and tldextract: {exc}") from exc
        self._encode = encode_bytes
        self._url_features = url_features
        self._standardise = apply_standardizer
        self.columns: list[str] = []
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def load_extra(self, extra: dict[str, Path]) -> None:
        import json

        scaler_path = extra.get("side_scaler")
        if scaler_path is None:
            raise MissingDependency("url model needs its side_scaler sidecar")
        doc = json.loads(Path(scaler_path).read_text(encoding="utf-8"))
        self.columns = list(doc["columns"])
        self.mean = np.asarray(doc["mean"], dtype=np.float32)
        self.std = np.asarray(doc["std"], dtype=np.float32)

    def side_vector(self, url: str) -> np.ndarray:
        feats = self._url_features(url)
        raw = np.asarray([[float(feats.get(c, 0.0)) for c in self.columns]], dtype=np.float32)
        return self._standardise(raw, self.mean, self.std)

    def make_feed(self, item: dict[str, Any]) -> dict[str, np.ndarray]:
        url = item["text"]
        return {
            "raw": self._encode(url, URL_MAX_LEN)[None].astype(np.int64),
            "side": self.side_vector(url),
        }

    def score(self, outputs: list[np.ndarray]) -> tuple[float, float]:
        logit = float(np.asarray(outputs[0]).reshape(-1)[0])
        return _sigmoid(logit), logit


class NidsRunner(_Base):
    kind = "onnx_nids"
    input_names = ("features",)

    def make_feed(self, item: dict[str, Any]) -> dict[str, np.ndarray]:
        return {"features": np.asarray(item["features"], dtype=np.float32)[None]}

    def score(self, outputs: list[np.ndarray]) -> tuple[float, float]:
        probs = np.asarray(outputs[0]).reshape(-1)
        # family 0 is benign (the schema's family order); the attack score is 1 - P(benign)
        return float(1.0 - probs[0]), float("nan")
