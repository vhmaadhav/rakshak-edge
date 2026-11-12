"""Runners: how each model kind is loaded, fed one item at a time, and scored.

``make_runner(kind)`` returns an object with ``load(path, threads)``, ``make_feed(item)``
(the featurisation, timed as part of the latency) and ``score(outputs)`` (probability and
logit). The ONNX runners reuse src/rakshak featurisation; nothing is re-implemented here.
"""

from __future__ import annotations

from rbench.errors import MissingDependency


def make_runner(kind: str):
    if kind in ("onnx_sms", "onnx_url", "onnx_nids"):
        from rbench.runners import onnx as onnx_runners

        return {
            "onnx_sms": onnx_runners.SmsRunner,
            "onnx_url": onnx_runners.UrlRunner,
            "onnx_nids": onnx_runners.NidsRunner,
        }[kind]()
    if kind == "pte_llm":
        from rbench.runners import pte_llm

        return pte_llm.PteLlmRunner()
    if kind == "pte_llm_direct":
        from rbench.runners import pte_direct

        return pte_direct.PteDirectRunner()
    raise MissingDependency(f"no runner for kind {kind!r}")


__all__ = ["make_runner", "MissingDependency"]
