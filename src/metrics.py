"""Evaluation helpers (threshold-free and thresholded)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)


def threshold_from_quantile(scores_unlabeled: np.ndarray, contamination: float) -> float:
    """Operating point: flag the top ``contamination`` fraction of the (mostly
    normal) training scores as anomalous."""
    return float(np.quantile(scores_unlabeled, 1.0 - contamination))


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """[[TN, FP], [FN, TP]]"""
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def prf(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores > threshold).astype(int)
    cm = confusion(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / cm.sum(), "cm": cm}


def best_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    p, r, _ = precision_recall_curve(y_true, scores)
    f = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(f.max())


def threshold_free(y_true: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "ap": float(average_precision_score(y_true, scores)),
        "best_f1": best_f1(y_true, scores),
    }


def roc_pr_curves(y_true: np.ndarray, scores: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    m = threshold_free(y_true, scores)
    return {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec, **m}
