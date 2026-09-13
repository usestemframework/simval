import shutil

import pytest

from simval.oracle import compare_metrics, list_cases, validate

pytest.importorskip("MDAnalysis")
datafiles = pytest.importorskip("MDAnalysisTests.datafiles")


def test_cases_available():
    cases = list_cases()
    assert "adk_morph" in cases


def test_compare_metrics_match():
    cand = {"n_selected_atoms": 214, "mean_rmsd_nm": 1.50, "mean_rg_nm": 2.24}
    ref = {"n_selected_atoms": 214, "mean_rmsd_nm": 1.514, "mean_rg_nm": 2.236}
    result = compare_metrics(cand, ref)
    assert result["__passed__"] is True


def test_compare_metrics_drift():
    cand = {"n_selected_atoms": 214, "mean_rmsd_nm": 3.0, "mean_rg_nm": 2.24}
    ref = {"n_selected_atoms": 214, "mean_rmsd_nm": 1.514, "mean_rg_nm": 2.236}
    result = compare_metrics(cand, ref)
    assert result["__passed__"] is False
    assert result["mean_rmsd_nm"]["passed"] is False
    assert result["mean_rg_nm"]["passed"] is True


def test_compare_exact_atom_count():
    cand = {"n_selected_atoms": 215}
    ref = {"n_selected_atoms": 214}
    assert compare_metrics(cand, ref)["__passed__"] is False


def test_validate_unknown_case_raises():
    with pytest.raises(KeyError):
        validate(".", "nonexistent_case_xyz")


def test_validate_cross_domain_mismatch_fails(tmp_path):
    run = tmp_path / "adk"
    run.mkdir()
    shutil.copy(datafiles.XTC, run / "traj.xtc")
    shutil.copy(datafiles.GRO, run / "conf.gro")
    result = validate(run, "kepler_two_body")  # MD run vs N-body case
    assert result.passed is False
    assert result.detail["n_checked"] == 0


def test_validate_adk_self_match(tmp_path):
    run = tmp_path / "adk"
    run.mkdir()
    shutil.copy(datafiles.XTC, run / "traj.xtc")
    shutil.copy(datafiles.GRO, run / "conf.gro")
    result = validate(run, "adk_morph")
    assert result.passed is True, result.detail
    assert result.detail["n_failed"] == 0


# --- ORA-005: canonical scenario enumeration + force-field contract ---


def _adk_run(tmp_path):
    run = tmp_path / "adk"
    run.mkdir()
    shutil.copy(datafiles.XTC, run / "traj.xtc")
    shutil.copy(datafiles.GRO, run / "conf.gro")
    return run


def test_added_tpr_fails_identity_before_metrics(tmp_path):
    # Identity used to hash only the selected trajectory topology (conf.gro
    # won precedence), so a changed/added topol.tpr was invisible.
    run = _adk_run(tmp_path)
    (run / "topol.tpr").write_bytes(b"not the golden's tpr")
    result = validate(run, "adk_morph")
    assert result.passed is False
    assert result.detail["n_checked"] == 0  # fails before any metric
    identity = result.detail["identity"]
    assert "topol.tpr" in identity.get("__undeclared_inputs__", {}).get("problem", []) or (
        identity.get("topol.tpr", {}).get("problem") is not None
    )


def test_methods_json_force_field_contract_fails_before_metrics(tmp_path):
    # methods.json declaring a different force field than the run's own
    # topology derives must fail the identity/contract gate, not flow into
    # metrics (audit ORA-005).
    from simval._util import gromacs_force_field_problem

    run = _adk_run(tmp_path)
    (run / "topol.top").write_text('#include "amber99sb-ildn.ff/forcefield.itp"\n')
    (run / "methods.json").write_text('{"force_field": "charmm36", "water": "tip3p"}')
    assert "force-field contract violation" in gromacs_force_field_problem(run)

    result = validate(run, "adk_morph")
    assert result.passed is False
    assert result.detail["n_checked"] == 0
    assert "force-field contract violation" in result.detail["error"]


def test_matching_methods_json_passes_the_contract(tmp_path):
    from simval._util import gromacs_force_field_problem

    run = _adk_run(tmp_path)
    (run / "topol.top").write_text('#include "amber99sb-ildn.ff/forcefield.itp"\n')
    (run / "methods.json").write_text('{"force_field": "amber99sb-ildn", "water": "tip3p"}')
    assert gromacs_force_field_problem(run) is None


def test_gromacs_scenario_inputs_enumerates_every_present_role(tmp_path):
    from simval._util import gromacs_scenario_inputs

    run = _adk_run(tmp_path)
    for name in ("topol.tpr", "energy.xvg", "mdout.mdp", "topol.top"):
        (run / name).write_bytes(b"x")
    (run / "params.json").write_text("{}")
    (run / "methods.json").write_text('{"force_field": "amber99sb-ildn"}')
    assert [p.name for p in gromacs_scenario_inputs(run)] == [
        "conf.gro",
        "energy.xvg",
        "mdout.mdp",
        "methods.json",
        "params.json",
        "topol.top",
        "topol.tpr",
        "traj.xtc",
    ]

    from simval.context import GromacsEngine

    ctx = GromacsEngine().load_context(run, selection="protein and name CA")
    assert {p.name for p in ctx.consumed_inputs} == {
        p.name for p in gromacs_scenario_inputs(run)
    }
