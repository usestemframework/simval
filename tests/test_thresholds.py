import json

import pytest

from simval.fixtures import make_run_dir
from simval.pipeline import diagnose
from simval.thresholds import load as load_thresholds


def test_threshold_override_takes_effect(tmp_path):
    run = make_run_dir(tmp_path / "good", good=True)
    default = diagnose(run, selection="protein")
    default_ed = [d for d in default["diagnostics"] if d["name"] == "energy_drift"][0]
    assert default_ed["passed"] is True

    tight = diagnose(run, selection="protein", thresholds={"energy_drift": 1e-6})
    tight_ed = [d for d in tight["diagnostics"] if d["name"] == "energy_drift"][0]
    assert tight_ed["threshold"] == 1e-6
    assert tight_ed["passed"] is False  # the good run's rel-range now exceeds 1e-6


def test_thresholds_json_in_run_dir(tmp_path):
    run = make_run_dir(tmp_path / "bad", good=False)
    (run / "thresholds.json").write_text(json.dumps({"energy_drift": 10.0}))
    m = diagnose(run, selection="protein")
    by_name = {d["name"]: d for d in m["diagnostics"]}
    assert by_name["energy_drift"]["threshold"] == 10.0


# --- ORA-004: externally supplied thresholds must be finite/positive/bounded ---



@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0, 1e9])
def test_bad_override_thresholds_rejected(bad):
    with pytest.raises(ValueError, match="threshold 'energy_drift'"):
        load_thresholds(None, {"energy_drift": bad})


def test_nan_thresholds_json_rejected(tmp_path):
    # JSON parses the NaN literal into a float; it must not reach any check.
    run = make_run_dir(tmp_path / "good", good=True)
    (run / "thresholds.json").write_text('{"energy_drift": NaN}')
    with pytest.raises(ValueError, match="must be finite"):
        diagnose(run, selection="protein")


def test_infinity_cli_override_rejected(tmp_path):
    run = make_run_dir(tmp_path / "good", good=True)
    with pytest.raises(ValueError, match="finite"):
        diagnose(run, selection="protein", thresholds={"rmsd_plateau": float("inf")})


def test_non_numeric_threshold_rejected():
    with pytest.raises(ValueError, match="must be a number"):
        load_thresholds(None, {"energy_drift": "loose"})


def test_thresholds_file_must_be_object(tmp_path):
    run = make_run_dir(tmp_path / "good", good=True)
    (run / "thresholds.json").write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        diagnose(run, selection="protein")


# --- THR-001: unknown threshold names are rejected, not silently stored ---


def test_unknown_threshold_name_rejected():
    # "energy_drif" used to be accepted, stored, and never used - the
    # intended energy_drift override silently did nothing.
    with pytest.raises(ValueError, match="unknown threshold 'energy_drif'"):
        load_thresholds(None, {"energy_drif": 0.001})
    with pytest.raises(ValueError, match="unknown threshold 'typo'"):
        load_thresholds(None, {"typo": 1.0})


def test_unknown_threshold_name_in_run_dir_file_rejected(tmp_path):
    run = make_run_dir(tmp_path / "good", good=True)
    (run / "thresholds.json").write_text('{"rmsd_plato": 0.5}')
    with pytest.raises(ValueError, match="unknown threshold 'rmsd_plato'"):
        diagnose(run, selection="protein")


def test_correctly_named_override_applies(tmp_path):
    run = make_run_dir(tmp_path / "good", good=True)
    m = diagnose(run, selection="protein", thresholds={"energy_drift": 1e-9})
    ed = [d for d in m["diagnostics"] if d["name"] == "energy_drift"][0]
    assert ed["threshold"] == 1e-9
    assert ed["passed"] is False


def test_registered_extension_threshold_accepted():
    import simval.thresholds as T

    T.register_threshold("custom_gate", 2.0, kwarg="min_ratio")
    try:
        resolved = load_thresholds(None, {"custom_gate": 3.0})
        assert resolved["custom_gate"] == 3.0
        assert T.kwargs_for("custom_gate", resolved) == {"min_ratio": 3.0}
        with pytest.raises(ValueError, match="already registered"):
            T.register_threshold("custom_gate", 1.0)
    finally:
        T.DEFAULT_THRESHOLDS.pop("custom_gate", None)
        T.CHECK_KWARG.pop("custom_gate", None)
