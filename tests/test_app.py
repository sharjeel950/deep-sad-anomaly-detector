"""Headless smoke test of the Streamlit UI (no browser needed)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _fresh():
    return AppTest.from_file(APP, default_timeout=180).run()


def test_app_renders_without_error():
    at = _fresh()
    assert not at.exception, at.exception


def test_train_and_evaluate_flow():
    at = _fresh()
    # shrink training so the test is quick
    for ni in at.sidebar.number_input:
        if ni.label == "Pretrain epochs":
            ni.set_value(5)
        if ni.label == "Train epochs":
            ni.set_value(5)
    at.run()
    train_btn = next(b for b in at.button if "Train Deep SAD" in b.label)
    train_btn.click().run()
    assert not at.exception, at.exception
    assert "run" in at.session_state
    run = at.session_state["run"]
    assert len(run["scores_test"]) == len(run["split"].y_test)
    assert len(run["baselines"]) == 4


def test_every_data_source_builds():
    at = _fresh()
    for src in list(at.sidebar.selectbox[0].options):
        if src.startswith("Upload"):
            continue
        at.sidebar.selectbox[0].select(src).run()
        assert not at.exception, (src, at.exception)
