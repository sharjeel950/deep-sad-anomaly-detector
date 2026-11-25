"""Reproducible benchmark: Deep SAD vs. baselines on all bundled datasets.

    python scripts/benchmark.py --seeds 3 --labels 10

Writes docs/BENCHMARK.md (mean ± std of test AUROC / average precision).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import NAMES, run_baselines  # noqa: E402
from src.data import (load_digits_ovr, load_wdbc, make_gaussian_clusters, make_high_dim,  # noqa: E402
                      make_semi_supervised_split, make_two_moons)
from src.metrics import threshold_free  # noqa: E402
from src.trainer import DeepSAD, DeepSADConfig  # noqa: E402

DATASETS = {
    "Two Moons (2-D)": lambda s: make_two_moons(3000, 0.08, 0.08, s),
    "Gaussian Clusters (2-D)": lambda s: make_gaussian_clusters(3000, 0.08, 0.65, s),
    "High-Dim Tabular (20-D)": lambda s: make_high_dim(3000, 0.08, 0.5, 20, s),
    "Breast Cancer WDBC": lambda s: load_wdbc(0.10, s),
    "Digits (normal = 8)": lambda s: load_digits_ovr(8, 0.10, s),
}
ORDER = [NAMES[k] for k in ("deepsad", "deepsvdd", "iforest", "ocsvm", "rf")]


def run(n_labels: int, known_types, seeds: int, cfg: DeepSADConfig):
    table = {}
    for dname, make in DATASETS.items():
        auc = {m: [] for m in ORDER}
        ap = {m: [] for m in ORDER}
        for s in range(seeds):
            ds = make(s)
            kt = known_types if ds.n_anomaly_types > 1 else None
            sp = make_semi_supervised_split(ds, n_labeled_anomalies=n_labels, pollution=0.02, n_known_types=kt, seed=s)
            model = DeepSAD(sp.n_features, DeepSADConfig(**{**cfg.__dict__, "seed": s}))
            model.pretrain(sp.X_train, sp.y_semi)
            model.fit(sp.X_train, sp.y_semi)
            res = {NAMES["deepsad"]: model.score(sp.X_test)}
            res.update({k: v["scores"] for k, v in run_baselines(sp, model, 0.05, s).items()})
            for name, sc in res.items():
                m = threshold_free(sp.y_test, sc)
                auc[name].append(m["auroc"])
                ap[name].append(m["ap"])
        table[dname] = (auc, ap)
        print(f"  done {dname}", flush=True)
    return table


def fmt(table, which: int, title: str) -> str:
    lines = [f"**{title}**", "", "| Dataset | " + " | ".join(ORDER) + " |", "|---|" + "---|" * len(ORDER)]
    for dname, pair in table.items():
        vals = pair[which]
        best = max(np.mean(v) for v in vals.values() if v)
        cells = []
        for m in ORDER:
            v = vals[m]
            if not v:
                cells.append("–")
                continue
            c = f"{np.mean(v):.3f} ± {np.std(v):.3f}"
            cells.append(f"**{c}**" if np.isclose(np.mean(v), best) else c)
        lines.append(f"| {dname} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", type=int, default=3)
    ap_.add_argument("--labels", type=int, default=10)
    args = ap_.parse_args()
    cfg = DeepSADConfig(hidden=(128, 64), rep_dim=16, pretrain_epochs=30, epochs=50)

    t0 = time.time()
    out = [
        "# Benchmark results",
        "",
        f"Test-set metrics, mean ± std over {args.seeds} seeds. {args.labels} labeled anomalies, 2% hidden contamination in the unlabeled pool, "
        "no labeled normals. Encoder 128-64-16, 30 pretrain + 50 train epochs. Best mean per row in bold.",
        "",
        "Regenerate with `python scripts/benchmark.py`.",
        "",
    ]
    for label, kt in [("A. Labels cover ALL anomaly types", None), ("B. Labels cover only ONE of three anomaly types (synthetic sets) – the realistic 'unknown unknowns' case", 1)]:
        print(label, flush=True)
        tb = run(args.labels, kt, args.seeds, cfg)
        out += [f"## {label}", "", fmt(tb, 0, "Test AUROC"), "", fmt(tb, 1, "Test average precision"), ""]
    out.append(f"_Runtime: {time.time() - t0:.0f}s on CPU._")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "BENCHMARK.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
