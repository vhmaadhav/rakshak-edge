"""Source-label confound checks.

Many sources are single-class or nearly so (IMC25 is all smishing), so label predictions can
be driven by source identity. Two cheap probes quantify this:

1. source_identity_accuracy: a char TF-IDF + logistic-regression classifier that predicts
   *which source* a message came from (80/20 stratified, per-source sample capped to keep the
   run under a minute). High accuracy means source style is easy to read off the text.
2. label_from_source_accuracy: predict each message's label as the majority label of its
   source. Accuracy near the base rate of the pooled labels means source alone is not a
   shortcut; accuracy far above it means it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from rakshak.datasets.common import SEED


def source_identity_accuracy(
    kept: pd.DataFrame, per_source_cap: int = 2000, seed: int = SEED
) -> dict:
    parts = []
    for _src, grp in kept.groupby("source"):
        parts.append(grp.sample(n=min(per_source_cap, len(grp)), random_state=seed))
    sample = pd.concat(parts).reset_index(drop=True)
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=100_000, sublinear_tf=True
    )
    X = vec.fit_transform(sample["text_canon"].astype(str))
    y = sample["source"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = float(accuracy_score(y_te, pred))
    majority = float(pd.Series(y_te).value_counts(normalize=True).max())
    per_source = {}
    for src in sorted(set(y_te)):
        m = y_te == src
        per_source[src] = float(accuracy_score(y_te[m], pred[m]))
    confusions: dict[str, int] = {}
    for a, b in zip(y_te, pred, strict=True):
        if a != b:
            key = f"{a} -> {b}"
            confusions[key] = confusions.get(key, 0) + 1
    top = dict(sorted(confusions.items(), key=lambda kv: -kv[1])[:8])
    return {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "per_source_cap": per_source_cap,
        "accuracy": acc,
        "majority_baseline": majority,
        "per_source_recall": per_source,
        "top_confusions": top,
    }


def label_from_source_accuracy(kept: pd.DataFrame) -> dict:
    """Accuracy of predicting every message's binary label from its source's majority label."""
    maj = kept.groupby("source")["label_binary"].agg(lambda s: int(s.mean() >= 0.5))
    pred = kept["source"].map(maj).to_numpy()
    acc = float(np.mean(pred == kept["label_binary"].to_numpy()))
    base = float(max(kept["label_binary"].mean(), 1 - kept["label_binary"].mean()))
    single_class = {
        src: int(grp["label_binary"].iloc[0])
        for src, grp in kept.groupby("source")
        if grp["label_binary"].nunique() == 1
    }
    return {
        "accuracy": acc,
        "pooled_majority_baseline": base,
        "source_majority_label": {k: int(v) for k, v in maj.items()},
        "single_class_sources": single_class,
    }
