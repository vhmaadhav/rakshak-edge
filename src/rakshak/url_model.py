"""The URL model evidence item, shared by the edge service and ``eval_groundlm.py``.

The URL student (registry ``url_student``, ONNX) scores each URL of a message. When the best URL
score of a message is at or above the registered threshold, one evidence item of type
``phishing_url_model`` is added. Its text is ``"<registered domain> score=<score:.3f>"``: the
path is never stored, so nothing downstream (explainer facts, prompts, logs) can show the full
URL. The threshold comes from the registry; with no threshold the URL path is off and no item is
ever added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rakshak.canonicalize import Canon
from rakshak.rules import Evidence
from rakshak.url_features import registered_domain

URL_MODEL_TYPE = "phishing_url_model"
# Fixed rank weight. The score is not the weight: a weight of 0.5 keeps this item below the
# rule weights that set MEDIUM (0.6) and HIGH (0.85), so its risk effect is only the state rule
# in ``rakshak.evidence._risk`` (UNCERTAIN at least, HIGH with the SMS student over threshold).
URL_MODEL_WEIGHT = 0.5
_SCORE_TAG = " score="


class UrlScorer(Protocol):
    def score_urls(self, urls: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class UrlHit:
    """The best-scoring URL of a message. ``url`` is the canonical URL string."""

    url: str
    score: float


def best_url_hit(urls: list[str], scorer: UrlScorer | None) -> UrlHit | None:
    """The URL with the highest score (first one on ties), or None when there is no scorer or no
    URL. The scorer is never called without URLs."""
    if scorer is None or not urls:
        return None
    scores = [float(s) for s in scorer.score_urls(list(urls))]
    best = max(range(len(urls)), key=lambda i: (scores[i], -i))
    return UrlHit(urls[best], scores[best])


def url_model_text(domain: str, score: float) -> str:
    return f"{domain}{_SCORE_TAG}{score:.3f}"


def url_model_fields(text: str) -> tuple[str, float | None]:
    """(registered domain, score) back out of an evidence text made by ``url_model_text``."""
    domain, sep, rest = text.partition(_SCORE_TAG)
    if not sep:
        return text, None
    try:
        return domain, float(rest)
    except ValueError:
        return domain, None


def add_url_model_evidence(
    evidence: list[Evidence], canon: Canon, hit: UrlHit | None, threshold: float | None
) -> list[Evidence]:
    """``evidence`` plus one ``phishing_url_model`` item when ``hit.score >= threshold``.

    With no threshold (URL path disabled) or no hit, the list is returned unchanged (as a copy).
    The new id is one above the largest existing id, so ids stay unique after a filter.
    """
    out = list(evidence)
    if hit is None or threshold is None or hit.score < threshold:
        return out
    start = hit.url.lower()
    idx = canon.normalized.lower().find(start)
    span = (idx, idx + len(start)) if idx >= 0 else (0, 0)
    number = max((int(e.id[1:]) for e in out), default=0) + 1
    text = url_model_text(registered_domain(hit.url), hit.score)
    out.append(
        Evidence(
            id=f"E{number}",
            type=URL_MODEL_TYPE,
            span=span,
            text=text,
            weight=URL_MODEL_WEIGHT,
        )
    )
    return out
