"""Unit tests for the Deep SAD pipeline.  Run:  pytest -q"""

import numpy as np
import pandas as pd
import pytest

from src.baselines import NAMES, run_baselines
from src.data import (dataset_from_dataframe, load_digits_ovr, load_wdbc, make_gaussian_clusters,
                      make_high_dim, make_semi_supervised_split, make_two_moons)
from src.metrics import prf, threshold_free, threshold_from_quantile
from src.trainer import DeepSAD, DeepSADConfig

FAST = DeepSADConfig(hidden=(64, 32), rep_dim=8, pretrain_epochs=15, epochs=25, seed=0)


@pytest.fixture(scope="module")
def moons_split():
    ds = make_two_moons(2000, 0.08, 0.08, seed=0)
    return make_semi_supervised_split(ds, n_labeled_anomalies=10, pollution=0.02, seed=0)


def test_generators_shapes_and_rates():
    for ds in (make_two_moons(1000, 0.1, 0.08, 1), make_gaussian_clusters(1000, 0.1, 0.6, 1),
               make_high_dim(1000, 0.1, 0.5, 12, 1)):
        assert len(ds.y) == len(ds.X) == len(ds.atype) == 1000
        assert abs(ds.y.mean() - 0.1) < 0.01
        assert set(np.unique(ds.atype[ds.y == 1])) == {1, 2, 3}
        assert (ds.atype[ds.y == 0] == 0).all()
    assert load_wdbc(0.1).X.shape[1] == 30
    assert load_digits_ovr(3, 0.1).X.shape[1] == 64


def test_split_protocol_counts_and_no_leakage():
    ds = make_two_moons(2000, 0.1, 0.08, seed=3)
    sp = make_semi_supervised_split(ds, n_labeled_anomalies=7, labeled_normal_frac=0.05, pollution=0.03, seed=3)
    assert sp.counts["labeled_anomaly"] == 7
    assert (sp.y_semi == -1).sum() == 7 and (sp.y_semi == 1).sum() == sp.counts["labeled_normal"] > 0
    # every labeled anomaly is a true anomaly, every labeled normal a true normal
    assert (sp.y_train_true[sp.y_semi == -1] == 1).all()
    assert (sp.y_train_true[sp.y_semi == 1] == 0).all()
    # hidden anomalies in the unlabeled pool == requested contamination
    unl = sp.y_semi == 0
    assert sp.y_train_true[unl].sum() == sp.counts["unlabeled_anomalies"] > 0
    # test set untouched by labeling choices and naturally imbalanced
    sp2 = make_semi_supervised_split(ds, n_labeled_anomalies=0, pollution=0.0, seed=3)
    assert np.allclose(np.sort(sp.y_test), np.sort(sp2.y_test)) and sp.X_test.shape == sp2.X_test.shape


def test_known_types_restriction():
    ds = make_two_moons(2000, 0.1, 0.08, seed=4)
    sp = make_semi_supervised_split(ds, n_labeled_anomalies=20, n_known_types=1, seed=4)
    assert sp.counts["known_types"] == [1]


def test_training_decreases_loss_and_detects(moons_split):
    m = DeepSAD(moons_split.n_features, FAST)
    m.pretrain(moons_split.X_train, moons_split.y_semi)
    m.fit(moons_split.X_train, moons_split.y_semi)
    assert m.history["pretrain"][-1] < m.history["pretrain"][0]
    assert m.history["train"][-1] < m.history["train"][0]
    assert threshold_free(moons_split.y_test, m.score(moons_split.X_test))["auroc"] > 0.9


def test_labeled_anomalies_are_pushed_away(moons_split):
    m = DeepSAD(moons_split.n_features, FAST)
    m.pretrain(moons_split.X_train, moons_split.y_semi)
    m.fit(moons_split.X_train, moons_split.y_semi)
    s = m.score(moons_split.X_train)
    lab = moons_split.y_semi == -1
    assert s[lab].mean() > 5 * s[moons_split.y_semi == 0].mean()


def test_center_is_bounded_away_from_zero(moons_split):
    m = DeepSAD(moons_split.n_features, FAST)
    m.pretrain(moons_split.X_train, moons_split.y_semi)
    c = m.init_center(moons_split.X_train, moons_split.y_semi)
    assert (c.abs() >= FAST.center_eps - 1e-9).all()


def test_network_is_bias_free():
    m = DeepSAD(5, FAST)
    assert all("bias" not in n for n, _ in m.encoder.named_parameters())
    assert all("bias" not in n for n, _ in m.ae.named_parameters())


def test_save_load_roundtrip(moons_split):
    m = DeepSAD(moons_split.n_features, FAST)
    m.pretrain(moons_split.X_train, moons_split.y_semi)
    m.fit(moons_split.X_train, moons_split.y_semi)
    m2, _ = DeepSAD.from_bytes(m.to_bytes({"a": 1}))
    assert np.allclose(m.score(moons_split.X_test), m2.score(moons_split.X_test), atol=1e-5)


def test_unsupervised_mode_works(moons_split):
    m = DeepSAD(moons_split.n_features, FAST)
    m.pretrain(moons_split.X_train)
    m.fit(moons_split.X_train, np.zeros(len(moons_split.X_train)))
    assert threshold_free(moons_split.y_test, m.score(moons_split.X_test))["auroc"] > 0.8


def test_baselines_run(moons_split):
    m = DeepSAD(moons_split.n_features, FAST)
    m.pretrain(moons_split.X_train, moons_split.y_semi)
    m.fit(moons_split.X_train, moons_split.y_semi)
    out = run_baselines(moons_split, m, 0.05, seed=0)
    assert set(out) == {NAMES[k] for k in ("deepsvdd", "iforest", "ocsvm", "rf")}
    for v in out.values():
        assert len(v["scores"]) == len(moons_split.y_test)


def test_metrics_and_threshold():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    assert threshold_free(y, s)["auroc"] == 1.0
    r = prf(y, s, 0.65)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["cm"].tolist() == [[4, 0], [0, 2]]
    assert threshold_from_quantile(np.arange(100), 0.05) == pytest.approx(94.05)


def test_csv_loader():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200),
                       "cat": rng.choice(["x", "y"], 200), "label": rng.choice(["ok", "fraud"], 200, p=[.9, .1])})
    ds = dataset_from_dataframe(df, "label", ["fraud"])
    assert ds.X.shape[1] == 4 and ds.y.sum() == (df["label"] == "fraud").sum()
    with pytest.raises(ValueError):
        dataset_from_dataframe(df, "label", [])
