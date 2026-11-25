"""Plotly figure factory – keeps all visual styling in one place."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

C = {
    "normal": "#3D6FE0",
    "anomaly": "#E4572E",
    "unlabeled": "#9AA5B1",
    "lab_normal": "#1B998B",
    "lab_anomaly": "#F2A541",
    "accent": "#6C4AB6",
}
MODEL_COLORS = ["#6C4AB6", "#3D6FE0", "#1B998B", "#F2A541", "#E4572E", "#7A8794"]


def _base(fig: go.Figure, title: Optional[str] = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.01, font=dict(size=15)) if title else None,
        height=height,
        margin=dict(l=10, r=10, t=48 if title else 16, b=10),
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=12.5),
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def project_2d(X: np.ndarray, fit_on: Optional[np.ndarray] = None):
    """Return (X2, projector, axis_labels).  2-D data is returned untouched."""
    if X.shape[1] == 2:
        return X, None, ("feature 1", "feature 2")
    pca = PCA(n_components=2, random_state=0).fit(X if fit_on is None else fit_on)
    ev = pca.explained_variance_ratio_
    return pca.transform(X), pca, (f"PC1 ({ev[0]:.0%})", f"PC2 ({ev[1]:.0%})")


# ---------------------------------------------------------------- data views
def fig_data_overview(X2: np.ndarray, y_semi: np.ndarray, labels=("x", "y")) -> go.Figure:
    fig = go.Figure()
    groups = [
        (0, "Unlabeled", C["unlabeled"], 5, 0.45, "circle", 0),
        (1, "Labeled normal (+1)", C["lab_normal"], 8, 0.95, "circle", 1),
        (-1, "Labeled anomaly (−1)", C["lab_anomaly"], 12, 1.0, "diamond", 1.5),
    ]
    for val, name, color, size, op, sym, lw in groups:
        m = y_semi == val
        if not m.any():
            continue
        fig.add_trace(go.Scattergl(
            x=X2[m, 0], y=X2[m, 1], mode="markers", name=f"{name} · {int(m.sum())}",
            marker=dict(color=color, size=size, opacity=op, symbol=sym, line=dict(width=lw, color="white")),
            hovertemplate="%{x:.2f}, %{y:.2f}<extra>" + name + "</extra>"))
    fig.update_xaxes(title=labels[0], zeroline=False, showgrid=True, gridcolor="rgba(128,128,128,.15)")
    fig.update_yaxes(title=labels[1], zeroline=False, showgrid=True, gridcolor="rgba(128,128,128,.15)")
    return _base(fig, "Training data as the learner sees it", 440)


def fig_loss(history: dict) -> go.Figure:
    pre, tr = history.get("pretrain", []), history.get("train", [])
    n = 2 if pre else 1
    titles = (["Stage 1 · autoencoder pretraining"] if pre else []) + ["Stage 2 · Deep SAD objective"]
    fig = make_subplots(rows=1, cols=n, subplot_titles=titles)
    col = 1
    if pre:
        fig.add_trace(go.Scatter(x=list(range(1, len(pre) + 1)), y=pre, mode="lines",
                                 line=dict(color=C["normal"], width=2.5), name="reconstruction MSE",
                                 showlegend=False), row=1, col=col)
        fig.update_xaxes(title="epoch", row=1, col=col)
        fig.update_yaxes(title="loss", row=1, col=col)
        col += 1
    fig.add_trace(go.Scatter(x=list(range(1, len(tr) + 1)), y=tr, mode="lines",
                             line=dict(color=C["accent"], width=2.5), name="Deep SAD loss",
                             showlegend=False), row=1, col=col)
    fig.update_xaxes(title="epoch", row=1, col=col)
    fig.update_yaxes(title="loss", row=1, col=col)
    return _base(fig, None, 300)


# ------------------------------------------------------------ evaluation views
def fig_roc(curves: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dot", width=1))
    for i, (name, c) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=f"{name} (AUC {c['auroc']:.3f})",
                                 line=dict(color=MODEL_COLORS[i % 6], width=3 if i == 0 else 2)))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return _base(fig, "ROC curve", 400)


def fig_pr(curves: dict, prevalence: float) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, x1=1, y0=prevalence, y1=prevalence, line=dict(color="gray", dash="dot", width=1))
    for i, (name, c) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(x=c["recall"], y=c["precision"], mode="lines", name=f"{name} (AP {c['ap']:.3f})",
                                 line=dict(color=MODEL_COLORS[i % 6], width=3 if i == 0 else 2)))
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    return _base(fig, "Precision–recall curve", 400)


def fig_score_hist(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> go.Figure:
    fig = go.Figure()
    hi = float(np.quantile(scores, 0.995))
    s = np.clip(scores, None, hi)                       # keep a few outliers from squashing the plot
    bins = dict(start=float(s.min()), end=hi, size=(hi - float(s.min())) / 60 or 1)
    for val, name, color in [(0, "Normal", C["normal"]), (1, "Anomaly", C["anomaly"])]:
        fig.add_trace(go.Histogram(x=s[y_true == val], name=name, marker_color=color, opacity=0.65,
                                   histnorm="probability", xbins=bins))
    fig.add_vline(x=min(threshold, hi), line=dict(color="black", dash="dash", width=2),
                  annotation_text="threshold", annotation_position="top right")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Anomaly score  ‖φ(x) − c‖²")
    fig.update_yaxes(title="share of class")
    return _base(fig, "Score distribution on the test set", 380)


def fig_confusion(cm: np.ndarray) -> go.Figure:
    labels = ["Normal", "Anomaly"]
    text = [[f"TN<br><b>{cm[0, 0]}</b>", f"FP<br><b>{cm[0, 1]}</b>"],
            [f"FN<br><b>{cm[1, 0]}</b>", f"TP<br><b>{cm[1, 1]}</b>"]]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, text=text, texttemplate="%{text}",
                               colorscale=[[0, "#EEF2FB"], [1, C["accent"]]], showscale=False,
                               hovertemplate="actual %{y} → predicted %{x}: %{z}<extra></extra>"))
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return _base(fig, "Confusion matrix", 340)


def fig_latent(Z2: np.ndarray, y_true: np.ndarray, center2: np.ndarray, labels=("PC1", "PC2")) -> go.Figure:
    fig = go.Figure()
    for val, name, color, size, op in [(0, "Normal", C["normal"], 5, 0.5), (1, "Anomaly", C["anomaly"], 7, 0.85)]:
        m = y_true == val
        fig.add_trace(go.Scattergl(x=Z2[m, 0], y=Z2[m, 1], mode="markers", name=name,
                                   marker=dict(color=color, size=size, opacity=op)))
    fig.add_trace(go.Scatter(x=[center2[0]], y=[center2[1]], mode="markers", name="centre c",
                             marker=dict(symbol="star", size=18, color="black", line=dict(color="white", width=1.5))))
    fig.update_xaxes(title=labels[0], zeroline=False)
    fig.update_yaxes(title=labels[1], zeroline=False)
    return _base(fig, "Learned latent space (test set, PCA of φ(x))", 400)


def fig_boundary(xs, ys, dist_grid, X_orig: np.ndarray, y_true: np.ndarray, labeled_anom: Optional[np.ndarray] = None) -> go.Figure:
    fig = go.Figure(go.Contour(x=xs, y=ys, z=dist_grid, colorscale="YlOrRd", ncontours=22,
                               contours=dict(coloring="heatmap", showlines=False),
                               colorbar=dict(title="‖φ(x)−c‖", thickness=12), opacity=0.9))
    for val, name, color in [(0, "Normal (test)", C["normal"]), (1, "Anomaly (test)", "#111111")]:
        m = y_true == val
        fig.add_trace(go.Scattergl(x=X_orig[m, 0], y=X_orig[m, 1], mode="markers", name=name,
                                   marker=dict(color=color, size=4 if val == 0 else 6, opacity=0.65,
                                               line=dict(width=0.5, color="white"))))
    if labeled_anom is not None and len(labeled_anom):
        fig.add_trace(go.Scatter(x=labeled_anom[:, 0], y=labeled_anom[:, 1], mode="markers", name="Labeled anomalies (train)",
                                 marker=dict(color=C["lab_anomaly"], size=12, symbol="diamond",
                                             line=dict(width=1.5, color="black"))))
    fig.update_xaxes(title="feature 1")
    fig.update_yaxes(title="feature 2")
    return _base(fig, "Anomaly-score surface (2-D data)", 480)


def fig_label_efficiency(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, (model, g) in enumerate(df.groupby("model", sort=False)):
        g = g.sort_values("labels")
        fig.add_trace(go.Scatter(
            x=g["labels"], y=g["mean"], mode="lines+markers", name=model,
            error_y=dict(type="data", array=g["std"].fillna(0), visible=True, thickness=1.2, width=3),
            line=dict(color=MODEL_COLORS[i % 6], width=3 if i == 0 else 2,
                      dash="solid" if i == 0 else "dot"),
            marker=dict(size=8)))
    fig.update_xaxes(title="# labeled anomalies in training set")
    fig.update_yaxes(title="Test AUROC", range=[max(0.0, float(df["mean"].min()) - 0.08), 1.005])
    return _base(fig, "Label efficiency – what do a few labeled anomalies buy?", 440)
