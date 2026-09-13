"""Engine-selection tests (audit ENG-001): every detector evaluates, exactly
one must match; ambiguity fails closed naming the candidates; the synthetic
signature cannot swallow concrete domains."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from simval.context import select_engine
from simval.fixtures import make_run_dir


def _wave_run(path: Path, *, cfl: float = 0.5) -> Path:
    path.mkdir(parents=True)
    (path / "wave.json").write_text(
        json.dumps(
            {
                "c": 1.0,
                "dx": 1.0,
                "dt": cfl,
                "nx": 64,
                "n_steps": 40,
                "source": {"pos": 16, "freq": 0.25, "n_on": 10},
            }
        )
    )
    return path


def test_synthetic_fixture_detected(tmp_path):
    run = make_run_dir(tmp_path / "s", good=True)
    assert select_engine(run).name == "synthetic"


def test_stray_npy_no_longer_makes_a_domain_synthetic(tmp_path):
    # A CFL-unstable wave run plus a stray energy.npy used to diagnose as
    # synthetic (first match), skipping the CFL check and provenance. The
    # precise signature must let the wave engine own the run-dir.
    run = _wave_run(tmp_path / "w", cfl=1.4)
    (run / "energy.npy").write_bytes(b"stray")
    assert select_engine(run).name == "wave-fdtd"

    from simval.pipeline import run_checks

    ctx = select_engine(run).load_context(run, selection="n/a")
    results = {r.name: r for r in run_checks(ctx)}
    assert results["cfl_stability"].passed is False


def test_two_domain_configs_in_one_dir_rejected(tmp_path):
    run = _wave_run(tmp_path / "w")
    (run / "fluid.json").write_text("{}")
    with pytest.raises(ValueError, match="ambiguous run-dir.*fluid-lbm.*wave-fdtd"):
        select_engine(run)


def test_synthetic_triple_plus_trajectory_is_ambiguous(tmp_path):
    run = make_run_dir(tmp_path / "s", good=True)
    (run / "traj.xtc").write_bytes(b"x")
    with pytest.raises(ValueError, match="ambiguous run-dir.*gromacs.*synthetic"):
        select_engine(run)


def test_bare_npy_dir_without_the_signature_is_unrecognized(tmp_path):
    (tmp_path / "lonely").mkdir()
    (tmp_path / "lonely" / "energy.npy").write_bytes(b"0")
    with pytest.raises(ValueError, match="no engine recognized"):
        select_engine(tmp_path / "lonely")
