"""Late fusion fitted strictly on named calibration rows, with portable inference."""

from __future__ import annotations

from dataclasses import fields

import lightgbm as lgb
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from rakshak.canonicalize import Flags, canonicalize
from rakshak.fusion.calibrate import labelled, scores_array
from rakshak.rules import EVIDENCE_TYPES, run_rules

FEATURE_NAMES = ("message_prob", "url_prob", "has_url", *EVIDENCE_TYPES,
                 *(f.name for f in fields(Flags)))
TRANSACTIONAL_NAMES = ("do_not_share", "otp_protective_no_url", "transaction_keywords")
MODELS = ("logistic", "lightgbm", "mean", "max")


def feature_matrix(texts, message_prob, url_prob, *, has_url, transactional=False) -> np.ndarray:
    """Explicit URL presence; absent URL score is NaN, encoded as 0 PLUS has_url=0.

    A present URL without a score, or an absent URL with a score, is an error.
    Mean/max use only available experts. Rule and obfuscation features are binary.
    """
    message = scores_array(message_prob, probability=True)
    url = np.asarray(url_prob, float)
    present = np.asarray(has_url)
    if (len(texts) != len(message) or url.shape != message.shape
            or present.shape != message.shape or not np.isin(present, [False, True]).all()):
        raise ValueError("unaligned features or invalid has_url")
    present = present.astype(bool)
    scores_array(url[present], probability=True)
    if not np.isnan(url[~present]).all():
        raise ValueError("absent URLs require missing (NaN) scores")
    x = np.zeros((len(message), len(FEATURE_NAMES)))
    x[:, :3] = np.column_stack((message, np.where(present, url, 0), present))
    rule_index = {name: 3 + i for i, name in enumerate(EVIDENCE_TYPES)}
    for i, text in enumerate(texts):
        canon = canonicalize(text)
        if bool(canon.urls) != present[i]:
            raise ValueError("has_url disagrees with canonical URL extraction")
        for evidence in run_rules(canon):
            x[i, rule_index[evidence.type]] = 1
        x[i, 3 + len(EVIDENCE_TYPES):] = [bool(getattr(canon.flags, f.name))
                                        for f in fields(Flags)]
    if transactional:
        from rakshak.fusion.transactional_profile import categories

        flags = [categories(text) for text in texts]
        extra = np.asarray([["do_not_share" in row, "otp_protective_no_url" in row,
                             bool(row & {"bank", "upi", "delivery", "telecom_utility"})]
                            for row in flags], dtype=float).reshape(len(texts), 3)
        x = np.column_stack((x, extra))
    return x


class FusionModel:
    """Logistic/monotone GBM stacker, or available-expert mean/max baseline.

    fit selects calibration IDs BEFORE inspecting features/labels. Split membership
    is mandatory and all IDs must be unique. Expert training provenance remains
    the caller's responsibility: saved scores must be out of sample for cal/test.
    """

    def __init__(self, name: str = "logistic", *, transactional=False):
        if name not in MODELS:
            raise ValueError(f"unknown fusion model {name}")
        self.name = name
        self.feature_names = FEATURE_NAMES + (TRANSACTIONAL_NAMES if transactional else ())
        self.state: dict | None = None

    def fit(self, x, y, *, ids, cal_ids, train_ids, test_ids) -> FusionModel:
        ids = list(ids)
        cal, train, test = set(cal_ids), set(train_ids), set(test_ids)
        if len(set(ids)) != len(ids) or not cal or not cal <= set(ids):
            raise ValueError("unique IDs and nonempty available cal IDs required")
        if cal & (train | test) or train & test:
            raise ValueError("train/cal/test IDs overlap")
        select = np.array([i for i, value in enumerate(ids) if value in cal])
        x = self._matrix(np.asarray(x)[select])
        _, labels = labelled(x[:, 0], np.asarray(y)[select], probability=True)
        self.fit_ids = [str(ids[i]) for i in select]
        self.state = {}
        if self.name == "logistic":
            model = LogisticRegression(max_iter=1000).fit(x, labels)
            self.state = {"weights": model.coef_[0].tolist(),
                          "intercept": float(model.intercept_[0])}
        elif self.name == "lightgbm":
            model = lgb.LGBMClassifier(
                n_estimators=40, num_leaves=7, max_depth=3, learning_rate=0.05,
                min_child_samples=10, n_jobs=1, verbosity=-1, random_state=0,
                monotone_constraints=[1, 1] + [0] * (x.shape[1] - 2),
            ).fit(x, labels)
            self.state = model.booster_.dump_model()
        return self

    def _matrix(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError("invalid fusion feature matrix")
        if ((x < 0) | (x > 1)).any() or not np.isin(x[:, 2:], [0, 1]).all():
            raise ValueError("probabilities and binary indicators required")
        if np.any((x[:, 2] == 0) & (x[:, 1] != 0)):
            raise ValueError("absent URL must use gated zero encoding")
        return x

    def transform(self, x) -> np.ndarray:
        if self.state is None:
            raise ValueError("fusion model is not fitted")
        x = self._matrix(x)
        if self.name == "mean":
            return (x[:, 0] + x[:, 1] * x[:, 2]) / (1 + x[:, 2])
        if self.name == "max":
            return np.maximum(x[:, 0], x[:, 1] * x[:, 2])
        if self.name == "logistic":
            return expit(x @ np.asarray(self.state["weights"]) + self.state["intercept"])

        def leaf(node: dict, row: np.ndarray) -> float:
            if "leaf_value" in node:
                return node["leaf_value"]
            if node["decision_type"] != "<=":
                raise ValueError("unsupported tree split")
            go_left = row[node["split_feature"]] <= node["threshold"]
            child = "left_child" if go_left else "right_child"
            return leaf(node[child], row)

        return expit(np.array([sum(leaf(t["tree_structure"], row)
                                   for t in self.state["tree_info"]) for row in x]))

    def to_dict(self) -> dict:
        if self.state is None:
            raise ValueError("fusion model is not fitted")
        return {"name": self.name, "features": list(self.feature_names),
                "state": self.state, "fit_ids": self.fit_ids}

    @classmethod
    def from_dict(cls, state: dict) -> FusionModel:
        features = tuple(state["features"])
        if features not in (FEATURE_NAMES, FEATURE_NAMES + TRANSACTIONAL_NAMES):
            raise ValueError("incompatible feature schema")
        obj = cls(state["name"], transactional=len(features) > len(FEATURE_NAMES))
        obj.state, obj.fit_ids = state["state"], state["fit_ids"]
        return obj
