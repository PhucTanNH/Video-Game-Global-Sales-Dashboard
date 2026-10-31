"""Streamlit smoke tests using AppTest (run: python -m pytest -q tests)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from vgsales_core.evaluation import ExperimentResult  # noqa: E402

SECTIONS = ["1. Dataset Overview", "2. Exploratory Data Analysis", "3. Model Comparison", "4. Error Analysis", "5. Leakage Audit", "6. Sales Prediction"]


def _app(root: Path = ROOT) -> AppTest:
    at = AppTest.from_file(str(root / "app.py"), default_timeout=600)
    at.run()
    return at


def _assert_clean(at: AppTest):
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.parametrize("section", SECTIONS)
def test_each_section_renders(section):
    at = _app()
    at.sidebar.radio[0].set_value(section).run()
    _assert_clean(at)
    assert not at.error, [e.value for e in at.error]


def test_overview_confirms_saved_metrics():
    at = _app()
    assert any("match model_meta.json" in s.value for s in at.success)


def test_empty_filters_show_empty_state():
    at = _app()
    at.sidebar.multiselect(key="f_platform").set_value(["PS4"]).run()
    at.sidebar.slider(key="f_years").set_value((1980, 1985)).run()
    _assert_clean(at)
    assert any("No titles match" in i.value for i in at.info)
    at.sidebar.radio[0].set_value("2. Exploratory Data Analysis").run()
    _assert_clean(at)
    assert any("No titles match" in i.value for i in at.info)


def test_quick_experiment_and_downstream_views():
    at = _app()
    at.sidebar.radio[0].set_value("3. Model Comparison").run()
    at.multiselect(key="exp_models").set_value(["Ridge"]).run()
    at.slider(key="exp_folds").set_value(2).run()
    at.button(key="exp_run").click().run()
    _assert_clean(at)
    exps = at.session_state["experiments"]
    assert {r.model for r in exps.values()} == {"Ridge", "Median baseline", "Publisher-Genre median baseline"}
    assert all(r.status == "ok" for r in exps.values())
    ridge = next(r for r in exps.values() if r.model == "Ridge")
    assert ridge.holdout_rows == 2090 and len(ridge.fold_scores) == 2
    # Holdout predictions only cover years after the cut-off (never training rows).
    assert ridge.holdout_predictions["Year"].min() == 2013
    assert set(ridge.cv_predictions["valid_year"]) == {2011, 2012}
    for section in ["4. Error Analysis", "6. Sales Prediction"]:
        at.sidebar.radio[0].set_value(section).run()
        _assert_clean(at)
    opts = at.selectbox(key="pred_model").options
    session_opt = next(o for o in opts if o.startswith("Session: Ridge"))
    at.selectbox(key="pred_model").set_value(session_opt).run()
    at.button(key="pred_go").click().run()
    _assert_clean(at)
    assert any(m.label == "Predicted lifetime global sales" for m in at.metric)


def test_saved_model_prediction_and_invalid_year():
    at = _app()
    at.sidebar.radio[0].set_value("6. Sales Prediction").run()
    at.button(key="pred_go").click().run()
    _assert_clean(at)
    assert any(m.label == "Predicted lifetime global sales" for m in at.metric)
    at.selectbox(key="pred_platform").set_value("PS4").run()
    at.number_input(key="pred_year").set_value(2010).run()
    _assert_clean(at)
    assert any("impossible" in e.value for e in at.error)


def test_failed_run_is_reported_without_crash():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.session_state["experiments"] = {
        "deadbeef0000": ExperimentResult(model="kNN", feature_config="original", feature_label="Original-27 (saved model inputs)",
                                         params={}, tuning_status="-", protocol="p", cutoff=2012, seed=1, status="failed",
                                         error="ValueError: deliberate")
    }
    at.run()
    at.sidebar.radio[0].set_value("3. Model Comparison").run()
    _assert_clean(at)
    assert any("Some runs failed" in e.value for e in at.error)


def test_missing_model_files_are_handled(tmp_path):
    """Run in a fresh interpreter so the copied project (without models/) is imported, not this one."""
    import subprocess
    import textwrap

    for item in ["app.py", "ADY201m_enhanced_v2.xlsx"]:
        shutil.copy(ROOT / item, tmp_path / item)
    for pkg in ["dashboard", "vgsales_core"]:
        shutil.copytree(ROOT / pkg, tmp_path / pkg, ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "models").mkdir()
    runner = tmp_path.parent / "run_missing_models.py"
    runner.write_text(textwrap.dedent(f"""
        import sys
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file({str(tmp_path / "app.py")!r}, default_timeout=600)
        at.run()
        sections = {SECTIONS!r}
        for s in sections:
            at.sidebar.radio[0].set_value(s).run()
            assert not at.exception, (s, [e.message for e in at.exception])
        assert any("No model available" in e.value for e in at.error), [e.value for e in at.error]
        import vgsales_core
        assert {str(tmp_path)!r} in vgsales_core.__file__
        print("OK")
    """))
    out = subprocess.run([sys.executable, "-I", str(runner)], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0 and "OK" in out.stdout, out.stdout[-2000:] + out.stderr[-2000:]
