import shutil

import numpy as np
import pytest

pytest.importorskip("MDAnalysis")
datafiles = pytest.importorskip("MDAnalysisTests.datafiles")

from simval.cli import main
from simval.manifest import load_manifest
from simval.pipeline import diagnose


def _write_stable_xvg(path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    e = np.full(n, -12345.0) + rng.normal(0.0, 2.0, n)
    t = np.arange(n) * 1.0
    lines = ['@ s0 legend "Total Energy"', '@    xaxis label "Time (ps)"']
    for ti, ei in zip(t, e):
        lines.append(f"{ti:.3f} {ei:.4f}")
    path.write_text("\n".join(lines) + "\n")


def _make_real_run(tmp_path):
    run = tmp_path / "real"
    run.mkdir()
    shutil.copy(datafiles.XTC, run / "traj.xtc")
    shutil.copy(datafiles.TPR, run / "topol.tpr")
    shutil.copy(datafiles.GRO, run / "conf.gro")
    _write_stable_xvg(run / "energy.xvg")
    return run


def test_real_gromacs_dir_diagnosed(tmp_path):
    run = _make_real_run(tmp_path)
    manifest = diagnose(run)
    assert manifest["params"]["engine"] == "gromacs"
    assert manifest["params"]["n_selected_atoms"] > 0
    names = {d["name"] for d in manifest["diagnostics"]}
    assert {"rmsd_plateau", "structural_equilibration", "energy_drift"} <= names
    loaded = load_manifest(run / "provenance.json")
    assert loaded["files"]


def test_adk_morph_correctly_fails_plateau(tmp_path):
    run = _make_real_run(tmp_path)
    manifest = diagnose(run, selection="protein and name CA")
    by_name = {d["name"]: d for d in manifest["diagnostics"]}
    assert by_name["rmsd_plateau"]["passed"] is False
    assert by_name["structural_equilibration"]["passed"] is False
    assert manifest["verdict"] == "fail"


def test_cli_main_real_run(tmp_path, capsys):
    run = _make_real_run(tmp_path)
    rc = main(["diagnose", str(run), "--selection", "protein and name CA"])
    out = capsys.readouterr().out
    assert "rmsd_plateau" in out
    assert "verdict: FAIL" in out
    assert rc == 1


# --- GROM-001: a normal conf.gro + topol.tpr + traj.xtc dir loads and
# hashes both applicable files ---


def test_gro_tpr_xtc_dir_loads_deterministically_and_hashes_both(tmp_path):
    run = _make_real_run(tmp_path)
    manifest = diagnose(run, selection="protein")
    # MAN-003: files are stored as canonical run-relative paths.
    hashed = set(manifest["files"])
    assert {"conf.gro", "topol.tpr", "traj.xtc"} <= hashed
    assert manifest["params"]["n_selected_atoms"] > 0


def test_tpr_only_dir_uses_tpr_as_trajectory_topology(tmp_path):
    run = tmp_path / "tpr_only"
    run.mkdir()
    shutil.copy(datafiles.XTC, run / "traj.xtc")
    shutil.copy(datafiles.TPR, run / "topol.tpr")
    manifest = diagnose(run, selection="protein")
    assert manifest["params"]["engine"] == "gromacs"
    assert manifest["params"]["n_selected_atoms"] > 0
