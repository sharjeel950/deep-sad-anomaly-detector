"""Datasets and the semi-supervised train/test protocol.

Label convention
----------------
* ``y``        : ground truth, 0 = normal, 1 = anomaly (used for evaluation only)
* ``atype``    : 0 = normal, 1..K = anomaly *type* (lets us simulate "known" vs
                 "unknown" anomaly types, the central scenario of Deep SAD)
* ``y_semi``   : what the *learner* sees -> 0 unlabeled, +1 labeled normal,
                 -1 labeled anomaly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.datasets import load_breast_cancer, load_digits, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    atype: np.ndarray
    feature_names: list[str]
    description: str = ""
    type_names: dict[int, str] = field(default_factory=dict)

    @property
    def n_anomaly_types(self) -> int:
        return int(len(np.unique(self.atype[self.atype > 0])))

    @property
    def summary(self) -> dict:
        return {
            "samples": len(self.y),
            "features": self.X.shape[1],
            "anomalies": int(self.y.sum()),
            "anomaly_rate": float(self.y.mean()),
            "anomaly_types": self.n_anomaly_types,
        }


@dataclass
class SemiSplit:
    X_train: np.ndarray            # standardised
    y_semi: np.ndarray             # 0 / +1 / -1
    y_train_true: np.ndarray       # hidden ground truth (diagnostics only)
    X_test: np.ndarray
    y_test: np.ndarray
    atype_test: np.ndarray
    scaler: StandardScaler
    feature_names: list[str]
    counts: dict

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


# --------------------------------------------------------------------------- #
# Synthetic generators
# --------------------------------------------------------------------------- #
def _shuffle(rng, *arrays):
    perm = rng.permutation(len(arrays[0]))
    return [a[perm] for a in arrays]


def _uniform_far_from(rng, X_ref: np.ndarray, n: int, low, high, min_dist: float) -> np.ndarray:
    """Uniform samples in a box that keep >= min_dist from every reference point."""
    if n <= 0:
        return np.empty((0, X_ref.shape[1]))
    tree = cKDTree(X_ref)
    kept: list[np.ndarray] = []
    got = 0
    for _ in range(50):
        cand = rng.uniform(low, high, size=(max(n * 20, 200), X_ref.shape[1]))
        d, _ = tree.query(cand)
        cand = cand[d > min_dist]
        kept.append(cand)
        got += len(cand)
        if got >= n:
            break
    return np.vstack(kept)[:n]


def make_two_moons(n_samples=3000, anomaly_ratio=0.08, noise=0.08, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    n_anom = int(round(n_samples * anomaly_ratio))
    n_norm = n_samples - n_anom
    Xn, _ = make_moons(n_samples=n_norm, noise=noise, random_state=seed)
    k = np.array_split(np.arange(n_anom), 3)
    a1 = rng.normal([0.5, 1.45], 0.14, (len(k[0]), 2))
    a2 = rng.normal([0.5, -0.95], 0.14, (len(k[1]), 2))
    a3 = _uniform_far_from(rng, Xn, len(k[2]), [-1.8, -1.5], [2.8, 2.0], 0.30)
    X = np.vstack([Xn, a1, a2, a3])
    atype = np.concatenate([np.zeros(n_norm), np.full(len(a1), 1), np.full(len(a2), 2), np.full(len(a3), 3)]).astype(int)
    y = (atype > 0).astype(int)
    X, y, atype = _shuffle(rng, X, y, atype)
    return Dataset("Two Moons (2-D)", X, y, atype, ["x1", "x2"],
                   "Two interleaved half-circles are normal. Anomalies: two compact clusters + scattered uniform noise.",
                   {1: "cluster A", 2: "cluster B", 3: "scattered"})


def make_gaussian_clusters(n_samples=3000, anomaly_ratio=0.08, noise=0.65, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    n_anom = int(round(n_samples * anomaly_ratio))
    n_norm = n_samples - n_anom
    means = np.array([[-2.5, 0.0], [2.5, 0.0], [0.0, 3.5]])
    comp = rng.integers(0, 3, n_norm)
    Xn = means[comp] + rng.normal(0, noise, (n_norm, 2))
    k = np.array_split(np.arange(n_anom), 3)
    a1 = rng.normal([0.0, -3.0], 0.35, (len(k[0]), 2))
    a2 = rng.normal([5.5, 3.5], 0.35, (len(k[1]), 2))
    a3 = _uniform_far_from(rng, Xn, len(k[2]), [-6, -5], [7, 7], 0.9)
    X = np.vstack([Xn, a1, a2, a3])
    atype = np.concatenate([np.zeros(n_norm), np.full(len(a1), 1), np.full(len(a2), 2), np.full(len(a3), 3)]).astype(int)
    y = (atype > 0).astype(int)
    X, y, atype = _shuffle(rng, X, y, atype)
    return Dataset("Gaussian Clusters (2-D)", X, y, atype, ["x1", "x2"],
                   "Three Gaussian blobs are normal. Anomalies: two small remote clusters + scattered uniform noise.",
                   {1: "cluster A", 2: "cluster B", 3: "scattered"})


def make_high_dim(n_samples=3000, anomaly_ratio=0.08, noise=0.5, n_features=20, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    d = int(n_features)
    n_anom = int(round(n_samples * anomaly_ratio))
    n_norm = n_samples - n_anom
    means = rng.normal(0, 2.0, size=(3, d))
    A = rng.normal(size=(d, max(2, d // 2))) * 0.5

    def normal_like(n, noise_scale=1.0):
        comp = rng.integers(0, 3, n)
        z = rng.normal(size=(n, A.shape[1]))
        return means[comp] + z @ A.T + rng.normal(0, noise * noise_scale, (n, d))

    Xn = normal_like(n_norm)
    k = np.array_split(np.arange(n_anom), 3)
    # 1) mean-shifted cluster
    shift = rng.normal(size=d)
    shift = shift / np.linalg.norm(shift) * 7.0
    a1 = normal_like(len(k[0])) + shift
    # 2) sparse feature spikes
    a2 = normal_like(len(k[1]))
    sd = Xn.std(axis=0)
    for i in range(len(a2)):
        cols = rng.choice(d, size=min(3, d), replace=False)
        a2[i, cols] += rng.choice([-1, 1], size=len(cols)) * 5.0 * sd[cols]
    # 3) variance-inflated samples
    a3 = normal_like(len(k[2]), noise_scale=1.0)
    a3 = a3.mean(axis=0, keepdims=True) + (a3 - a3.mean(axis=0, keepdims=True)) * 2.2
    X = np.vstack([Xn, a1, a2, a3])
    atype = np.concatenate([np.zeros(n_norm), np.full(len(a1), 1), np.full(len(a2), 2), np.full(len(a3), 3)]).astype(int)
    y = (atype > 0).astype(int)
    X, y, atype = _shuffle(rng, X, y, atype)
    return Dataset(f"High-Dim Tabular ({d}-D)", X, y, atype, [f"x{i + 1}" for i in range(d)],
                   "Correlated 3-mode Gaussian mixture is normal. Anomalies: mean shift, sparse feature spikes, inflated variance.",
                   {1: "mean shift", 2: "feature spikes", 3: "variance inflation"})


# --------------------------------------------------------------------------- #
# Real (bundled with scikit-learn -> works offline)
# --------------------------------------------------------------------------- #
def load_wdbc(anomaly_ratio=0.10, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    data = load_breast_cancer()
    malignant = data.target == 0
    benign_idx = np.where(~malignant)[0]
    mal_idx = np.where(malignant)[0]
    n_anom = int(round(anomaly_ratio / (1 - anomaly_ratio) * len(benign_idx)))
    n_anom = int(np.clip(n_anom, 10, len(mal_idx)))
    sel = rng.choice(mal_idx, n_anom, replace=False)
    idx = np.concatenate([benign_idx, sel])
    X = data.data[idx]
    y = malignant[idx].astype(int)
    atype = y.copy()
    X, y, atype = _shuffle(rng, X, y, atype)
    return Dataset("Breast Cancer – WDBC", X, y, atype, list(data.feature_names),
                   "Wisconsin Diagnostic Breast Cancer: benign tumours are normal, malignant are anomalies (subsampled).",
                   {1: "malignant"})


def load_digits_ovr(normal_digit=0, anomaly_ratio=0.10, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    data = load_digits()
    norm_idx = np.where(data.target == normal_digit)[0]
    other_idx = np.where(data.target != normal_digit)[0]
    n_anom = int(round(anomaly_ratio / (1 - anomaly_ratio) * len(norm_idx)))
    n_anom = int(np.clip(n_anom, 10, len(other_idx)))
    sel = rng.choice(other_idx, n_anom, replace=False)
    idx = np.concatenate([norm_idx, sel])
    X = data.data[idx].astype(float)
    digits = data.target[idx]
    y = (digits != normal_digit).astype(int)
    atype = np.where(y == 1, digits + 1, 0)           # 0 reserved for "normal"
    X, y, atype = _shuffle(rng, X, y, atype)
    return Dataset(f"Digits (normal = {normal_digit})", X, y, atype, [f"px{i}" for i in range(X.shape[1])],
                   f"8x8 handwritten digits: digit {normal_digit} is normal, all other digits are anomalies.",
                   {int(d) + 1: f"digit {d}" for d in range(10) if d != normal_digit})


# --------------------------------------------------------------------------- #
# User-supplied tabular data
# --------------------------------------------------------------------------- #
def dataset_from_dataframe(df: pd.DataFrame, label_col: str, anomaly_values: Sequence,
                           name: str = "Uploaded CSV", max_onehot: int = 20) -> Dataset:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")
    if not anomaly_values:
        raise ValueError("Select at least one label value that means 'anomaly'.")
    labels = df[label_col].astype(str)
    anomaly_values = [str(v) for v in anomaly_values]
    y = labels.isin(anomaly_values).astype(int).to_numpy()
    type_map = {v: i + 1 for i, v in enumerate(anomaly_values)}
    atype = np.where(y == 1, labels.map(type_map).fillna(0).astype(int), 0)

    feats = df.drop(columns=[label_col])
    numeric = feats.select_dtypes(include="number")
    cat = feats.select_dtypes(exclude="number")
    keep = [c for c in cat.columns if cat[c].nunique(dropna=True) <= max_onehot]
    parts = [numeric]
    if keep:
        parts.append(pd.get_dummies(cat[keep], dtype=float))
    X_df = pd.concat(parts, axis=1)
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)
    X_df = X_df.loc[:, X_df.std() > 0]
    if X_df.shape[1] == 0:
        raise ValueError("No usable numeric feature columns found.")
    if y.sum() < 4 or (1 - y).sum() < 4:
        raise ValueError("Need at least 4 normal and 4 anomalous rows.")
    return Dataset(name, X_df.to_numpy(float), y, atype, list(map(str, X_df.columns)),
                   f"Uploaded table: label column '{label_col}', anomaly = {anomaly_values}.",
                   {v: f"'{k}'" for k, v in type_map.items()})


def subsample(ds: Dataset, max_rows: int, seed: int = 0) -> Dataset:
    if len(ds.y) <= max_rows:
        return ds
    idx, _ = train_test_split(np.arange(len(ds.y)), train_size=max_rows, stratify=ds.y, random_state=seed)
    return Dataset(ds.name, ds.X[idx], ds.y[idx], ds.atype[idx], ds.feature_names, ds.description, ds.type_names)


# --------------------------------------------------------------------------- #
# Semi-supervised protocol
# --------------------------------------------------------------------------- #
def make_semi_supervised_split(
    ds: Dataset,
    test_size: float = 0.30,
    n_labeled_anomalies: int = 10,
    labeled_normal_frac: float = 0.0,
    pollution: float = 0.02,
    n_known_types: Optional[int] = None,
    seed: int = 42,
    label_seed: Optional[int] = None,
) -> SemiSplit:
    """Build the training signal Deep SAD sees and a fully labeled test set.

    * ``test`` keeps the natural anomaly ratio and is fully labeled.
    * ``train`` = unlabeled pool (normal + ``pollution`` anomalies, unlabeled)
                 + ``labeled_normal_frac`` of normals (+1)
                 + ``n_labeled_anomalies`` anomalies (-1) drawn only from the
                   first ``n_known_types`` anomaly types.
    * ``seed`` fixes the train/test partition, ``label_seed`` only changes
      *which* samples get labeled (used by the label-efficiency study).
    """
    label_seed = seed if label_seed is None else label_seed
    rng = np.random.default_rng(label_seed)
    idx = np.arange(len(ds.y))
    if ds.y.sum() < 4:
        raise ValueError("Dataset needs at least 4 anomalies.")
    tr, te = train_test_split(idx, test_size=test_size, stratify=ds.y, random_state=seed)

    norm_idx = tr[ds.y[tr] == 0]
    anom_idx = tr[ds.y[tr] == 1]

    types = np.unique(ds.atype[anom_idx])
    known = types[: n_known_types] if n_known_types else types
    cand = anom_idx[np.isin(ds.atype[anom_idx], known)]
    n_lab_a = int(min(n_labeled_anomalies, len(cand)))
    lab_a = rng.choice(cand, n_lab_a, replace=False) if n_lab_a else np.array([], dtype=int)
    rest_a = np.setdiff1d(anom_idx, lab_a)

    n_lab_n = int(round(labeled_normal_frac * len(norm_idx)))
    lab_n = rng.choice(norm_idx, n_lab_n, replace=False) if n_lab_n else np.array([], dtype=int)
    unl_n = np.setdiff1d(norm_idx, lab_n)

    n_poll = int(min(round(pollution * len(unl_n)), len(rest_a)))
    poll_a = rng.choice(rest_a, n_poll, replace=False) if n_poll else np.array([], dtype=int)

    train_idx = np.concatenate([unl_n, poll_a, lab_n, lab_a]).astype(int)
    y_semi = np.concatenate([
        np.zeros(len(unl_n) + len(poll_a)), np.ones(len(lab_n)), -np.ones(len(lab_a))
    ])
    perm = rng.permutation(len(train_idx))
    train_idx, y_semi = train_idx[perm], y_semi[perm]

    scaler = StandardScaler().fit(ds.X[train_idx])
    X_train = scaler.transform(ds.X[train_idx])
    X_test = scaler.transform(ds.X[te])
    y_test = ds.y[te]
    if y_test.sum() < 2:
        raise ValueError("Test set has fewer than 2 anomalies – increase samples or anomaly ratio.")

    counts = {
        "train_total": len(train_idx),
        "unlabeled": int((y_semi == 0).sum()),
        "unlabeled_anomalies": int(len(poll_a)),
        "labeled_normal": int((y_semi == 1).sum()),
        "labeled_anomaly": int((y_semi == -1).sum()),
        "test_total": len(te),
        "test_anomalies": int(y_test.sum()),
        "known_types": [int(t) for t in known],
    }
    return SemiSplit(X_train, y_semi, ds.y[train_idx], X_test, y_test, ds.atype[te],
                     scaler, ds.feature_names, counts)
