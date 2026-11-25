"""Deep SAD – semi-supervised anomaly detection workbench.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from src import plots
from src.baselines import NAMES, run_baselines
from src.data import (dataset_from_dataframe, load_digits_ovr, load_wdbc, make_gaussian_clusters,
                      make_high_dim, make_semi_supervised_split, make_two_moons, subsample)
from src.metrics import prf, roc_pr_curves, threshold_free, threshold_from_quantile
from src.trainer import DeepSAD, DeepSADConfig

st.set_page_config(page_title="Deep SAD · Anomaly Detector", page_icon="🛰️", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1320px;}
.hero {padding: 1.15rem 1.5rem; border-radius: 16px; margin-bottom: 1.1rem; color: #fff;
       background: linear-gradient(115deg, #6C4AB6 0%, #3D6FE0 100%);}
.hero h1 {margin: 0; font-size: 1.75rem; color: #fff; letter-spacing: -0.01em;}
.hero p {margin: .3rem 0 0; opacity: .93; font-size: 1.0rem;}
div[data-testid="stMetric"] {background: rgba(108,74,182,.07); border: 1px solid rgba(108,74,182,.20);
       padding: .65rem .9rem; border-radius: 12px;}
div[data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .8;}
.small-note {font-size: .86rem; opacity: .75;}
</style>
<div class="hero">
  <h1>🛰️ Deep SAD — Semi-supervised Anomaly Detector</h1>
  <p>Deep semi-supervised anomaly detection (Ruff et al., ICLR 2020): learn what <i>normal</i> looks like from mostly-unlabeled
  data, then sharpen the boundary with just a handful of labeled anomalies.</p>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
SOURCES = [
    "Two Moons (2-D synthetic)",
    "Gaussian Clusters (2-D synthetic)",
    "High-Dimensional Tabular (synthetic)",
    "Breast Cancer – WDBC (real)",
    "Digits – one vs rest (real)",
    "Upload your own CSV",
]
HIDDEN = {"Small · 64-32": (64, 32), "Medium · 128-64": (128, 64), "Large · 256-128-64": (256, 128, 64)}


@st.cache_data(show_spinner=False)
def cached_dataset(source: str, n: int, ratio: float, noise: float, d: int, digit: int, seed: int):
    if source.startswith("Two Moons"):
        return make_two_moons(n, ratio, noise, seed)
    if source.startswith("Gaussian"):
        return make_gaussian_clusters(n, ratio, noise, seed)
    if source.startswith("High-Dim"):
        return make_high_dim(n, ratio, noise, d, seed)
    if source.startswith("Breast"):
        return load_wdbc(ratio, seed)
    return load_digits_ovr(digit, ratio, seed)


@st.cache_data(show_spinner=False)
def read_csv_bytes(blob: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(blob))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def build_sidebar():
    P: dict = {}
    ds = None
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        st.markdown("**1 · Dataset**")
        P["source"] = st.selectbox("Data source", SOURCES, label_visibility="collapsed")
        src = P["source"]
        P.update(n=3000, ratio=0.08, noise=0.1, d=20, digit=8, seed=42)
        P["seed"] = int(st.number_input("Random seed", 0, 10_000, 42, 1))

        if src.startswith("Upload"):
            up = st.file_uploader("CSV with a label column", type=["csv"])
            if up is not None:
                df = read_csv_bytes(up.getvalue())
                label_col = st.selectbox("Label column", list(df.columns), index=len(df.columns) - 1)
                vals = df[label_col].astype(str).value_counts()
                if len(vals) > 60:
                    st.error("That column has more than 60 distinct values – pick a class/label column.")
                else:
                    default = [vals.index[-1]] if len(vals) > 1 else []
                    anomalies = st.multiselect("Which label value(s) mean ANOMALY?", list(vals.index), default=default)
                    max_rows = st.number_input("Max rows (stratified subsample)", 1000, 100_000, 20_000, 1000)
                    try:
                        ds = subsample(dataset_from_dataframe(df, label_col, anomalies, up.name), int(max_rows), P["seed"])
                        P.update(csv=(up.name, up.size, label_col, tuple(anomalies), int(max_rows)))
                    except ValueError as e:
                        st.warning(str(e))
        else:
            if not src.startswith(("Breast", "Digits")):
                P["n"] = st.slider("Samples", 1000, 10_000, 3000, 500)
                P["noise"] = st.slider("Noise level", 0.02, 1.0, 0.08 if src.startswith("Two") else (0.65 if src.startswith("Gaussian") else 0.5), 0.01)
            if src.startswith("High-Dim"):
                P["d"] = st.slider("Features", 6, 60, 20, 2)
            if src.startswith("Digits"):
                P["digit"] = st.selectbox("Normal digit", list(range(10)), index=8)
            P["ratio"] = st.slider("Anomaly share of the dataset", 0.02, 0.25, 0.08 if not src.startswith("Breast") else 0.10, 0.01,
                                   format="%.2f")
            ds = cached_dataset(src, P["n"], P["ratio"], P["noise"], P["d"], P["digit"], P["seed"])

        st.markdown("**2 · Supervision signal**")
        S = dict(test_size=0.30, n_labeled_anomalies=10, labeled_normal_frac=0.0, pollution=0.02, n_known_types=None)
        S["n_labeled_anomalies"] = st.slider("Labeled anomalies", 0, 100, 10, 1,
                                              help="The few anomalies a human has flagged. 0 turns Deep SAD into Deep SVDD.")
        S["labeled_normal_frac"] = st.slider("Labeled normal share", 0.0, 0.30, 0.0, 0.01, format="%.2f",
                                              help="Fraction of normal training samples that are explicitly confirmed normal (+1).")
        S["pollution"] = st.slider("Unlabeled-pool contamination", 0.0, 0.15, 0.02, 0.005, format="%.3f",
                                    help="Hidden anomalies mixed into the unlabeled data, as a share of the unlabeled normals.")
        if ds is not None and ds.n_anomaly_types > 1:
            S["n_known_types"] = st.slider("Anomaly types covered by labels", 1, ds.n_anomaly_types, ds.n_anomaly_types,
                                            help="Labels only come from the first k anomaly types – the rest are 'unknown unknowns'.")
        S["test_size"] = st.slider("Test split", 0.15, 0.5, 0.30, 0.05)

        st.markdown("**3 · Model**")
        arch = st.selectbox("Encoder", list(HIDDEN), index=1)
        rep = st.select_slider("Latent dimension", [4, 8, 16, 32, 64], value=16)
        eta = st.select_slider("η  (weight of labeled data)", [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0], value=1.0)
        c1, c2 = st.columns(2)
        pre_e = c1.number_input("Pretrain epochs", 0, 200, 30, 5)
        epochs = c2.number_input("Train epochs", 5, 300, 50, 5)
        c3, c4 = st.columns(2)
        lr = c3.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], value=1e-3, format_func=lambda v: f"{v:g}")
        bs = c4.select_slider("Batch size", [32, 64, 128, 256], value=128)
        wd = st.select_slider("Weight decay (λ)", [0.0, 1e-6, 1e-5, 1e-4], value=1e-6, format_func=lambda v: f"{v:g}")
        contamination = st.slider("Expected anomaly rate", 0.01, 0.30, 0.05, 0.01, format="%.2f",
                                  help="Sets the default decision threshold (quantile of training scores).")
        cfg = DeepSADConfig(hidden=HIDDEN[arch], rep_dim=int(rep), eta=float(eta), lr=float(lr), weight_decay=float(wd),
                            pretrain_epochs=int(pre_e), epochs=int(epochs), batch_size=int(bs), seed=P["seed"])
    return P, S, cfg, contamination, ds


P, S, CFG, CONTAM, DS = build_sidebar()

if DS is None:
    st.info("⬅️ Upload a CSV and choose its label column / anomaly values in the sidebar to continue.")
    st.stop()

try:
    SPLIT = make_semi_supervised_split(DS, seed=P["seed"], **S)
except ValueError as err:
    st.error(f"Cannot build the semi-supervised split: {err}")
    st.stop()

DATA_FP = repr((P, S))
tab_over, tab_data, tab_train, tab_eval, tab_label, tab_infer = st.tabs(
    ["📘 Overview", "🗂️ Data", "🧠 Train", "📊 Evaluate", "🔬 Label efficiency", "🔍 Inference"])

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_over:
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### The idea")
        st.markdown(
            """
Anomalies are rare, varied and expensive to label – but a few are usually known. **Deep SAD** uses both kinds of information:

* **Unlabeled data** (assumed mostly normal) is mapped by a neural network φ into a *small hypersphere* around a fixed centre **c** — the classic Deep SVDD one-class idea.
* **Labeled anomalies** are mapped *away* from **c** by minimising the *inverse* distance, which carves out regions that must stay empty.
* **Labeled normals** (optional) are pulled in, exactly like unlabeled data but weighted by **η**.

The anomaly score of a new point is simply its squared distance to the centre.
"""
        )
        st.latex(r"""
\min_{\mathcal W}\;\frac{1}{n+m}\sum_{i=1}^{n}\lVert\phi(x_i;\mathcal W)-c\rVert^{2}
+\frac{\eta}{n+m}\sum_{j=1}^{m}\Big(\lVert\phi(\tilde x_j;\mathcal W)-c\rVert^{2}\Big)^{\tilde y_j}
+\frac{\lambda}{2}\sum_{\ell}\lVert W^{\ell}\rVert_F^{2}
""")
        st.caption("ỹ = +1 for labeled normal, −1 for labeled anomaly.  Score:  s(x) = ‖φ(x;W) − c‖².")
    with right:
        st.markdown("#### Pipeline implemented here")
        st.graphviz_chart(
            """
digraph G {
  rankdir=TB; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fillcolor="#EEF2FB", color="#6C4AB6", fontcolor="#222222"];
  edge [color="#6C4AB6"];
  data  [label="Training data\\n(unlabeled + few labels)"];
  ae    [label="① Autoencoder pretraining\\n(excludes labeled anomalies)"];
  c     [label="② Centre  c = mean φ(x)\\n|c_k| ≥ 0.1"];
  sad   [label="③ Deep SAD fine-tuning\\nbias-free encoder"];
  score [label="④ Score  ‖φ(x) − c‖²"];
  thr   [label="⑤ Threshold → normal / anomaly", fillcolor="#6C4AB6", fontcolor="white"];
  data -> ae -> c -> sad -> score -> thr;
}
""")
    st.markdown("---")
    a, b, c_ = st.columns(3)
    a.markdown("**Why bias-free layers?**  \nWithout biases the network can output the constant *c* for every input – "
               "a trivial zero-loss solution (*hypersphere collapse*).")
    b.markdown("**Why pretrain?**  \nAn autoencoder gives φ a sensible starting point and a meaningful centre; "
               "training from random weights is unstable.")
    c_.markdown("**Why η?**  \nη trades off trust in the labels against the unlabeled bulk. "
                "Larger η = labeled anomalies repel harder.")

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
with tab_data:
    cnt = SPLIT.counts
    st.markdown(f"**{DS.name}** — {DS.description}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Training samples", f"{cnt['train_total']:,}")
    m2.metric("Unlabeled", f"{cnt['unlabeled']:,}", help=f"of which {cnt['unlabeled_anomalies']} are hidden anomalies")
    m3.metric("Labeled normal", f"{cnt['labeled_normal']:,}")
    m4.metric("Labeled anomalies", f"{cnt['labeled_anomaly']:,}")
    m5.metric("Test set", f"{cnt['test_total']:,}", f"{cnt['test_anomalies']} anomalies", delta_color="off")

    X2, _, labels2 = plots.project_2d(SPLIT.X_train)
    cl, cr = st.columns([1.6, 1])
    with cl:
        st.plotly_chart(plots.fig_data_overview(X2, SPLIT.y_semi, labels2), width="stretch")
    with cr:
        st.markdown("##### What the model is allowed to know")
        st.markdown(
            f"""
- **{cnt['unlabeled']:,}** points carry *no* label – {cnt['unlabeled_anomalies']} of them are secretly anomalies (contamination).
- **{cnt['labeled_anomaly']}** anomalies are flagged (−1){' from ' + str(len(cnt['known_types'])) + ' of ' + str(DS.n_anomaly_types) + ' anomaly types' if DS.n_anomaly_types > 1 else ''}.
- **{cnt['labeled_normal']}** points are confirmed normal (+1).
- The test set is fully labeled and **never** touched during training or scaling.
""")
        if DS.type_names:
            st.markdown("##### Anomaly types")
            known = set(cnt["known_types"])
            st.dataframe(pd.DataFrame({"type": [DS.type_names[k] for k in sorted(DS.type_names)],
                                       "seen in labels": ["✅" if k in known else "—" for k in sorted(DS.type_names)]}),
                         hide_index=True, width="stretch")
    with st.expander("Preview raw data"):
        prev = pd.DataFrame(DS.X[:300], columns=DS.feature_names)
        prev.insert(0, "is_anomaly", DS.y[:300])
        st.dataframe(prev, width="stretch", height=280)

# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
LABELS = {"pretrain": "Stage 1 · autoencoder pretraining", "train": "Stage 2 · Deep SAD training"}


def train_everything(with_baselines: bool):
    prog = st.progress(0.0, text="Preparing…")
    slot = st.empty()
    hist = {"pretrain": [], "train": []}
    total = max(1, (CFG.pretrain_epochs) + CFG.epochs)
    state = {"done": 0}

    def cb(stage, epoch, n_epochs, loss):
        hist[stage].append(loss)
        state["done"] += 1
        prog.progress(min(state["done"] / total, 1.0), text=f"{LABELS[stage]} · epoch {epoch}/{n_epochs} · loss {loss:.4f}")
        if state["done"] % 4 == 0 or epoch == n_epochs:
            slot.plotly_chart(plots.fig_loss(hist), key=f"live_{state['done']}", width="stretch")

    t0 = time.time()
    model = DeepSAD(SPLIT.n_features, CFG)
    if CFG.pretrain_epochs > 0:
        model.pretrain(SPLIT.X_train, SPLIT.y_semi, cb)
    model.init_center(SPLIT.X_train, SPLIT.y_semi)
    model.fit(SPLIT.X_train, SPLIT.y_semi, cb)

    scores_test = model.score(SPLIT.X_test)
    scores_train = model.score(SPLIT.X_train)
    thr = threshold_from_quantile(scores_train[SPLIT.y_semi == 0], CONTAM)
    base = {}
    if with_baselines:
        prog.progress(1.0, text="Training baselines…")
        base = run_baselines(SPLIT, model, CONTAM, P["seed"])
    prog.empty()
    st.session_state["run"] = dict(id=int(time.time() * 1000), model=model, cfg=CFG, split=SPLIT, fp=DATA_FP,
                                   scores_test=scores_test, scores_train=scores_train, thr=thr, baselines=base,
                                   seconds=time.time() - t0, contamination=CONTAM)


with tab_train:
    n_par = DeepSAD(SPLIT.n_features, CFG).n_parameters
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Encoder", " → ".join(map(str, (SPLIT.n_features, *CFG.hidden, CFG.rep_dim))))
    k2.metric("Trainable parameters", f"{n_par:,}")
    k3.metric("η", f"{CFG.eta:g}")
    k4.metric("Epochs (pre / train)", f"{CFG.pretrain_epochs} / {CFG.epochs}")
    cA, cB = st.columns([1, 3])
    with_base = cA.checkbox("Also train baselines", value=True,
                            help="Deep SVDD, Isolation Forest, One-Class SVM and a supervised Random Forest.")
    go_btn = cA.button("🚀 Train Deep SAD", type="primary", width="stretch")
    cB.markdown("<div class='small-note'>Training runs on CPU in seconds for the bundled datasets. "
                "Change anything in the sidebar and press the button again to compare.</div>", unsafe_allow_html=True)
    if go_btn:
        with st.spinner("Training…"):
            train_everything(with_base)
        st.success("Done – head to the **Evaluate** tab.")
    run = st.session_state.get("run")
    if run:
        if run["fp"] != DATA_FP:
            st.warning("The data / supervision settings changed since the last training – results below are from the previous run.")
        st.markdown(f"##### Last run · {run['seconds']:.1f}s")
        st.plotly_chart(plots.fig_loss(run["model"].history), width="stretch")

# --------------------------------------------------------------------------- #
# Evaluate
# --------------------------------------------------------------------------- #
with tab_eval:
    run = st.session_state.get("run")
    if not run:
        st.info("Train a model first (🧠 Train tab).")
    else:
        sp, model = run["split"], run["model"]
        y = sp.y_test
        all_scores = {NAMES["deepsad"]: run["scores_test"], **{k: v["scores"] for k, v in run["baselines"].items()}}
        curves = {n: roc_pr_curves(y, s) for n, s in all_scores.items()}

        st.markdown("#### Model comparison (test set)")
        table = pd.DataFrame([{"Model": n, "AUROC": c["auroc"], "Average precision": c["ap"], "Best F1": c["best_f1"]}
                              for n, c in curves.items()])
        cc = {k: st.column_config.ProgressColumn(k, min_value=0.0, max_value=1.0, format="%.3f")
              for k in ["AUROC", "Average precision", "Best F1"]}
        st.dataframe(table, hide_index=True, column_config=cc, width="stretch")
        if len(curves) > 1:
            ours = curves[NAMES["deepsad"]]["auroc"]
            best_name, best_other = max(((n, c["auroc"]) for n, c in curves.items() if n != NAMES["deepsad"]), key=lambda t: t[1])
            diff = ours - best_other
            st.caption(f"Deep SAD AUROC is {abs(diff):.3f} {'above' if diff >= 0 else 'below'} the best alternative ({best_name}). "
                       "Average precision is the more telling metric when anomalies are rare.")
        r1, r2 = st.columns(2)
        r1.plotly_chart(plots.fig_roc(curves), width="stretch")
        r2.plotly_chart(plots.fig_pr(curves, float(y.mean())), width="stretch")

        st.markdown("#### Operating point")
        s_te = run["scores_test"]
        lo, hi = float(np.min(s_te)), float(np.quantile(s_te, 0.999))
        thr = st.slider("Decision threshold on the anomaly score", lo, hi, float(np.clip(run["thr"], lo, hi)),
                        step=(hi - lo) / 400, format="%.4f", key=f"thr_{run['id']}",
                        help="Default = (1 − expected anomaly rate) quantile of the training scores.")
        m = prf(y, s_te, thr)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Precision", f"{m['precision']:.3f}")
        q2.metric("Recall", f"{m['recall']:.3f}")
        q3.metric("F1", f"{m['f1']:.3f}")
        q4.metric("Accuracy", f"{m['accuracy']:.3f}")
        h1, h2 = st.columns([1.7, 1])
        h1.plotly_chart(plots.fig_score_hist(s_te, y, thr), width="stretch")
        h2.plotly_chart(plots.fig_confusion(m["cm"]), width="stretch")

        st.markdown("#### Inside the model")
        i1, i2 = st.columns(2)
        Z = model.embed(sp.X_test)
        Z2, pca, ax = plots.project_2d(Z)
        c_np = model.center.numpy()
        c2d = pca.transform(c_np[None, :])[0] if pca is not None else c_np
        i1.plotly_chart(plots.fig_latent(Z2, y, c2d, ax), width="stretch")
        if sp.n_features == 2:
            allx = np.vstack([sp.X_train, sp.X_test])
            lo2, hi2 = allx.min(0) - 0.6, allx.max(0) + 0.6
            gx, gy = np.linspace(lo2[0], hi2[0], 150), np.linspace(lo2[1], hi2[1], 150)
            G = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
            d = np.sqrt(model.score(G)).reshape(150, 150)
            d = np.clip(d, None, np.quantile(d, 0.97))
            mu, sd = sp.scaler.mean_, sp.scaler.scale_
            i2.plotly_chart(plots.fig_boundary(gx * sd[0] + mu[0], gy * sd[1] + mu[1], d,
                                               sp.scaler.inverse_transform(sp.X_test), y,
                                               sp.scaler.inverse_transform(sp.X_train[sp.y_semi == -1])), width="stretch")
        else:
            dist_tr = np.sqrt(run["scores_train"])
            fig_d = plots.fig_score_hist(run["scores_train"], np.where(sp.y_semi == -1, 1, 0), thr)
            fig_d.update_layout(title="Training scores: unlabeled/normal vs labeled anomalies")
            i2.plotly_chart(fig_d, width="stretch")
            i2.caption(f"Mean distance to centre – labeled anomalies: {dist_tr[sp.y_semi == -1].mean() if (sp.y_semi == -1).any() else float('nan'):.2f}, "
                       f"unlabeled: {dist_tr[sp.y_semi == 0].mean():.2f}.")
        if run["fp"] != DATA_FP:
            st.warning("Showing the previous run – sidebar settings have changed since.")

# --------------------------------------------------------------------------- #
# Label efficiency
# --------------------------------------------------------------------------- #
with tab_label:
    st.markdown("#### How many labeled anomalies are enough?")
    st.markdown("Re-trains Deep SAD for increasing numbers of labeled anomalies (same test set; only *which* samples are labeled changes between repeats). "
                "**0 labels = Deep SVDD**, so the left edge of the curve is the purely unsupervised starting point.")
    e1, e2, e3 = st.columns([2, 1, 1])
    counts = e1.multiselect("Labeled-anomaly counts", [0, 1, 2, 5, 10, 20, 50, 100], default=[0, 1, 2, 5, 10, 20, 50])
    n_rep = e2.number_input("Repeats per setting", 1, 5, 2)
    ep_fast = e3.number_input("Epochs (fast mode)", 10, 100, 25, 5)
    if st.button("▶ Run experiment", disabled=not counts):
        bar = st.progress(0.0, text="Running…")
        rows, total, done = [], len(counts) * int(n_rep), 0
        fast = replace(CFG, epochs=int(ep_fast), pretrain_epochs=min(CFG.pretrain_epochs, 20))
        for k in sorted(counts):
            res = {"Deep SAD": [], "Isolation Forest": [], "Random Forest (supervised)": []}
            for r in range(int(n_rep)):
                kwargs = {**S, "n_labeled_anomalies": k}
                sp = make_semi_supervised_split(DS, seed=P["seed"], label_seed=P["seed"] + 1000 + r, **kwargs)
                m_ = DeepSAD(sp.n_features, replace(fast, seed=P["seed"] + r))
                if fast.pretrain_epochs:
                    m_.pretrain(sp.X_train, sp.y_semi)
                m_.fit(sp.X_train, sp.y_semi)
                res["Deep SAD"].append(threshold_free(sp.y_test, m_.score(sp.X_test))["auroc"])
                others = run_baselines(sp, m_, CONTAM, P["seed"] + r, include=["iforest", "rf"])
                res["Isolation Forest"].append(threshold_free(sp.y_test, others[NAMES["iforest"]]["scores"])["auroc"])
                if NAMES["rf"] in others:
                    res["Random Forest (supervised)"].append(threshold_free(sp.y_test, others[NAMES["rf"]]["scores"])["auroc"])
                done += 1
                bar.progress(done / total, text=f"{k} labeled anomalies · repeat {r + 1}/{int(n_rep)}")
            for name, vals in res.items():
                if vals:
                    rows.append({"labels": k, "model": name, "mean": float(np.mean(vals)),
                                 "std": float(np.std(vals)) if len(vals) > 1 else 0.0})
        bar.empty()
        st.session_state["label_eff"] = dict(df=pd.DataFrame(rows), fp=DATA_FP)
    le = st.session_state.get("label_eff")
    if le:
        if le["fp"] != DATA_FP:
            st.warning("These curves were computed for different data settings.")
        st.plotly_chart(plots.fig_label_efficiency(le["df"]), width="stretch")
        wide = le["df"].pivot(index="labels", columns="model", values="mean").round(3)
        st.dataframe(wide, width="stretch")

# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
with tab_infer:
    run = st.session_state.get("run")
    if not run:
        st.info("Train a model first (🧠 Train tab).")
    else:
        sp, model = run["split"], run["model"]
        thr = float(st.session_state.get(f"thr_{run['id']}", run["thr"]))
        st.markdown("#### Score new observations")
        st.caption(f"Threshold in use: **{thr:.4f}** (set it in the Evaluate tab). Features: {len(sp.feature_names)}.")
        mode = st.radio("Input", ["Edit / paste rows", "Upload CSV"], horizontal=True, label_visibility="collapsed")

        def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
            X = frame[sp.feature_names].astype(float)
            X = X.fillna(pd.Series(sp.scaler.mean_, index=sp.feature_names))
            s = model.score(sp.scaler.transform(X.to_numpy()))
            out = frame.copy()
            out.insert(0, "anomaly_score", s)
            out.insert(1, "verdict", np.where(s > thr, "🔴 anomaly", "🟢 normal"))
            out.insert(2, "score / threshold", s / thr)
            return out

        if mode == "Edit / paste rows":
            sig = (run["id"], "edit")
            if st.session_state.get("edit_sig") != sig or st.button("🎲 Load random test rows"):
                pick = np.random.default_rng().choice(len(sp.X_test), size=min(8, len(sp.X_test)), replace=False)
                st.session_state["edit_df"] = pd.DataFrame(sp.scaler.inverse_transform(sp.X_test[pick]), columns=sp.feature_names)
                st.session_state["edit_true"] = sp.y_test[pick]
                st.session_state["edit_sig"] = sig
            edited = st.data_editor(st.session_state["edit_df"], num_rows="dynamic", width="stretch", key="editor")
            if len(edited):
                res = score_frame(edited.reset_index(drop=True))
                st.dataframe(res, hide_index=True, width="stretch",
                             column_config={"anomaly_score": st.column_config.NumberColumn(format="%.4f"),
                                            "score / threshold": st.column_config.ProgressColumn(min_value=0.0, max_value=3.0, format="%.2fx")})
                st.markdown("<div class='small-note'>Rows start as random test samples – edit any value and the verdict updates live.</div>",
                            unsafe_allow_html=True)
        else:
            tmpl = pd.DataFrame(sp.scaler.inverse_transform(sp.X_test[:5]), columns=sp.feature_names)
            st.download_button("⬇️ Download CSV template", tmpl.to_csv(index=False).encode(), "template.csv", "text/csv")
            up = st.file_uploader("CSV containing the model's feature columns", type=["csv"], key="infer_up")
            if up is not None:
                frame = read_csv_bytes(up.getvalue())
                missing = [c for c in sp.feature_names if c not in frame.columns]
                if missing:
                    st.error(f"Missing {len(missing)} feature column(s): {', '.join(missing[:8])}{'…' if len(missing) > 8 else ''}")
                else:
                    res = score_frame(frame)
                    n_flag = int((res["verdict"] == "🔴 anomaly").sum())
                    st.success(f"Scored {len(res):,} rows – {n_flag:,} flagged ({n_flag / len(res):.1%}).")
                    st.dataframe(res.head(500), hide_index=True, width="stretch")
                    st.download_button("⬇️ Download scored CSV", res.to_csv(index=False).encode(), "scored.csv", "text/csv")

        st.markdown("#### Export the trained detector")
        blob = model.to_bytes(extra=dict(feature_names=sp.feature_names, scaler_mean=sp.scaler.mean_,
                                         scaler_scale=sp.scaler.scale_, threshold=thr))
        st.download_button("⬇️ Download model checkpoint (.pt)", blob, "deep_sad_model.pt", "application/octet-stream")
        with st.expander("How to load it again"):
            st.code(
                "from src.trainer import DeepSAD\n"
                "model, meta = DeepSAD.from_bytes(open('deep_sad_model.pt','rb').read())\n"
                "X = (raw_features - meta['scaler_mean']) / meta['scaler_scale']\n"
                "is_anomaly = model.score(X) > meta['threshold']",
                language="python")
