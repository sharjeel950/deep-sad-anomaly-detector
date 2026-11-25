"""Reference methods Deep SAD is compared against.

All baselines receive exactly the same training information as Deep SAD:

* Unsupervised baselines (Isolation Forest, One-Class SVM, Deep SVDD) are fit
  on every training sample that is *not* a labeled anomaly - they cannot use
  anomaly labels, but we still give them the benefit of dropping the known ones.
* The supervised Random Forest sees the labeled anomalies as class 1 and
  everything else as class 0 (the usual "unlabeled = normal" PU shortcut).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM

from .data import SemiSplit
from .trainer import DeepSAD

NAMES = {
    "deepsad": "Deep SAD (semi-supervised)",
    "deepsvdd": "Deep SVDD (unsupervised)",
    "iforest": "Isolation Forest",
    "ocsvm": "One-Class SVM",
    "rf": "Random Forest (supervised)",
}


def run_baselines(split: SemiSplit, deep_model: DeepSAD, contamination: float = 0.05,
                  seed: int = 42, include: Optional[list[str]] = None) -> dict[str, dict]:
    include = include or ["deepsvdd", "iforest", "ocsvm", "rf"]
    X, y_semi = split.X_train, split.y_semi
    clean = y_semi != -1
    out: dict[str, dict] = {}

    if "deepsvdd" in include:
        t0 = time.time()
        m = deep_model.spawn(seed=seed)
        m.fit(X[clean], np.zeros(clean.sum()))
        out[NAMES["deepsvdd"]] = {"scores": m.score(split.X_test), "seconds": time.time() - t0}

    if "iforest" in include:
        t0 = time.time()
        iso = IsolationForest(n_estimators=200, contamination=min(max(contamination, 1e-3), 0.5),
                              random_state=seed, n_jobs=-1).fit(X[clean])
        out[NAMES["iforest"]] = {"scores": -iso.score_samples(split.X_test), "seconds": time.time() - t0}

    if "ocsvm" in include:
        t0 = time.time()
        Xc = X[clean]
        if len(Xc) > 4000:
            Xc = Xc[np.random.default_rng(seed).choice(len(Xc), 4000, replace=False)]
        oc = OneClassSVM(kernel="rbf", gamma="scale", nu=float(np.clip(contamination, 0.01, 0.5))).fit(Xc)
        out[NAMES["ocsvm"]] = {"scores": -oc.decision_function(split.X_test), "seconds": time.time() - t0}

    if "rf" in include and (y_semi == -1).sum() >= 1:
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                    min_samples_leaf=2, n_jobs=-1, random_state=seed)
        rf.fit(X, (y_semi == -1).astype(int))
        out[NAMES["rf"]] = {"scores": rf.predict_proba(split.X_test)[:, 1], "seconds": time.time() - t0}

    return out
