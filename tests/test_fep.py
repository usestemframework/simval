import pytest

pytest.importorskip("alchemlyb", reason="needs alchemlyb")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simval.fep import (  # noqa: E402
    FepEngine,
    check_free_energy,
    check_hysteresis,
    check_overlap,
    synthetic_u_nk,
)
from simval.oracle import get_case, list_cases  # noqa: E402
from simval.oracle.validate import compare_metrics  # noqa: E402

EXAMPLE = (
    __import__("pathlib").Path(__file__).parent.parent / "examples" / "fep" / "synthetic"
)
LN2 = 0.6931471805599453


def _ladder_u_nk(centers, n_per, *, std=0.5, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for i, c in enumerate(centers):
        x = rng.normal(c, std, n_per[i])
        lam = float(i)
        idx = pd.MultiIndex.from_arrays(
            [np.arange(sum(n_per[:i]), sum(n_per[:i]) + n_per[i]),
             np.full(n_per[i], lam)],
            names=["time", "fep-lambda"])
        row = {float(j): 0.5 * ((x - cj) / std) ** 2 for j, cj in enumerate(centers)}
        parts.append(pd.DataFrame(row, index=idx))
    return pd.concat(parts)


def test_synthetic_recovers_known_free_energy():
    u = synthetic_u_nk()
    res = check_free_energy(u)
    assert res.name == "free_energy"
    assert res.passed is True
    assert abs(res.detail["deltaG_kT"] - LN2) < 0.02
    assert res.detail["deltaG_kJmol"] == pytest.approx(res.detail["deltaG_kT"] * 2.479)


def test_overlap_passes_for_well_sampled_fixture():
    res = check_overlap(synthetic_u_nk())
    assert res.name == "fep_overlap"
    assert res.passed is True
    assert res.value >= 0.05


def test_overlap_fails_for_undersampled_state():
    centers = [0.0, 1.0, 2.0, 3.0, 4.0]
    n_per = [4000, 4000, 15, 4000, 4000]
    res = check_overlap(_ladder_u_nk(centers, n_per))
    assert res.passed is False
    assert res.value < 0.05


def test_hysteresis_skips_without_reverse_leg():
    res = check_hysteresis(synthetic_u_nk())
    assert res.name == "fep_hysteresis"
    assert res.passed is True
    assert res.detail["skipped"] is True


def test_hysteresis_with_consistent_reverse_leg():
    u = synthetic_u_nk()
    res = check_hysteresis(u, u_nk_reverse=u)
    assert res.passed is True
    assert res.value < 0.5


def test_fep_engine_detects_manifest_and_dhdl_files():
    eng = FepEngine()
    assert eng.name == "fep"
    assert eng.detect(EXAMPLE) is True


def test_fep_engine_load_context_reads_csv_u_nk():
    eng = FepEngine()
    ctx = eng.load_context(EXAMPLE, selection="n/a")
    assert ctx.extra["u_nk"] is not None
    assert ctx.run_params["domain"] == "fep"
    assert ctx.run_params["kind"] == "u_nk"
    res = check_free_energy(ctx.extra["u_nk"])
    assert res.passed is True
    assert np.isfinite(res.value)


def test_oracle_fep_case_exists_and_self_matches():
    assert "fep_synthetic" in list_cases()
    assert "benzene_hydration_fep" not in list_cases()  # retired (FEP-001, see AUDIT.md)
    case = get_case("fep_synthetic")
    u = synthetic_u_nk()
    fe = check_free_energy(u)
    ov = check_overlap(u)
    candidate = {
        "deltaG_kT": fe.detail["deltaG_kT"],
        "uncertainty_kT": fe.detail["uncertainty_kT"],
        "overlap_min_eigenvalue": ov.detail["overlap_min_eigenvalue"],
    }
    compared = compare_metrics(candidate, case.reference_metrics, case.tolerances)
    assert compared.pop("__passed__") is True
    assert all(v["passed"] for v in compared.values())


def test_validate_fep_synthetic_through_the_real_path():
    # FEP-002: the oracle's own metric names must line up with the golden
    # so validate() works end to end (the old test renamed keys by hand).
    from simval.oracle import validate

    result = validate(EXAMPLE, "fep_synthetic")
    assert result.passed is True, result.detail
    assert set(result.detail["metrics"]) == {
        "deltaG_kT", "uncertainty_kT", "overlap_min_eigenvalue",
    }
    assert all(m["passed"] for m in result.detail["metrics"].values())


def test_validate_fep_synthetic_fails_on_deliberate_deltaG_mutation(tmp_path):
    # FEP-002 acceptance: a deliberate deltaG mutation of the fixture must
    # fail — the mutated inputs are also caught by the ORA-003 identity gate.
    import shutil

    from simval.oracle import validate

    run = tmp_path / "synthetic"
    shutil.copytree(EXAMPLE, run)
    assert validate(run, "fep_synthetic").passed is True

    import pandas as pd

    df = pd.read_csv(run / "dhdl.csv")
    df.iloc[:, 1] = df.iloc[:, 1] + 5.0  # shift every reduced potential
    df.to_csv(run / "dhdl.csv", index=False)
    result = validate(run, "fep_synthetic")
    assert result.passed is False


def test_oracle_fep_flags_bad_candidate():
    case = get_case("fep_synthetic")
    bad = dict(case.reference_metrics)
    bad["deltaG_kT"] = -5.0
    bad["overlap_min_eigenvalue"] = 0.001
    compared = compare_metrics(bad, case.reference_metrics, case.tolerances)
    assert compared["__passed__"] is False
    assert compared["deltaG_kT"]["passed"] is False
    assert compared["overlap_min_eigenvalue"]["passed"] is False


# --- FEP-001: overlap is a min-bound invariant, not distance-from-golden ---


def test_overlap_min_bound_rejects_zero_overlap():
    # A zero-overlap candidate passed the old abs-0.05 rule against the
    # benzene golden's stored 0.0; the min bound must fail it exactly like
    # check_overlap() does.
    case = get_case("fep_synthetic")
    candidate = dict(case.reference_metrics)
    candidate["overlap_min_eigenvalue"] = 0.0
    compared = compare_metrics(candidate, case.reference_metrics, case.tolerances)
    assert compared["overlap_min_eigenvalue"]["passed"] is False
    assert compared["__passed__"] is False
    # Consistency with the domain check: the same value is declared unreliable.
    from simval.fep import _OVERLAP_MIN_EIGENVALUE

    assert 0.0 < _OVERLAP_MIN_EIGENVALUE


def test_overlap_min_bound_boundary_accepts_threshold():
    case = get_case("fep_synthetic")
    candidate = dict(case.reference_metrics)
    candidate["overlap_min_eigenvalue"] = 0.05
    compared = compare_metrics(candidate, case.reference_metrics, case.tolerances)
    assert compared["overlap_min_eigenvalue"]["passed"] is True


def test_nonfinite_overlap_fails_closed():
    case = get_case("fep_synthetic")
    for bad in (float("nan"), float("inf")):
        candidate = dict(case.reference_metrics)
        candidate["overlap_min_eigenvalue"] = bad
        compared = compare_metrics(candidate, case.reference_metrics, case.tolerances)
        assert compared["overlap_min_eigenvalue"]["passed"] is False


# --- PROV-001: the fep engine registers its consumed inputs ---


def test_fep_engine_registers_manifest_and_data_files(tmp_path):
    import shutil

    run = tmp_path / "synthetic"
    shutil.copytree(EXAMPLE, run)
    ctx = FepEngine().load_context(run, selection="n/a")
    assert {p.name for p in ctx.consumed_inputs} == {"fep.json", "dhdl.csv"}


# --- FEP-003: declared-but-missing files fail the run contract ---


def test_declared_missing_reverse_leg_fails_not_skips(tmp_path):
    # reverse_files: ["missing.csv"] used to load nothing, leaving
    # u_nk_reverse absent so check_hysteresis returned passing-skipped and
    # the overall verdict was PASS. It must be a failing run-contract error.
    import json
    import shutil

    from simval.pipeline import run_checks

    run = tmp_path / "synthetic"
    shutil.copytree(EXAMPLE, run)
    manifest = json.loads((run / "fep.json").read_text())
    manifest["reverse_files"] = ["missing.csv"]
    (run / "fep.json").write_text(json.dumps(manifest))

    ctx = FepEngine().load_context(run, selection="n/a")
    results = {r.name: r for r in run_checks(ctx)}
    contract = results["fep_run_contract"]
    assert contract.passed is False
    assert "declared reverse file 'missing.csv' is missing" in contract.detail["error"]
    assert not all(r.passed for r in results.values())


def test_declared_missing_forward_file_fails_not_skips(tmp_path):
    import json
    import shutil

    from simval.pipeline import run_checks

    run = tmp_path / "synthetic"
    shutil.copytree(EXAMPLE, run)
    manifest = json.loads((run / "fep.json").read_text())
    manifest["files"] = manifest["files"] + ["missing.csv"]
    (run / "fep.json").write_text(json.dumps(manifest))

    ctx = FepEngine().load_context(run, selection="n/a")
    results = {r.name: r for r in run_checks(ctx)}
    assert results["fep_run_contract"].passed is False
    assert "declared forward file 'missing.csv' is missing" in results["fep_run_contract"].detail["error"]


def test_oracle_fep_metrics_fail_closed_on_missing_declared_file(tmp_path):
    import json
    import shutil

    from simval.oracle.validate import _fep_metrics

    run = tmp_path / "synthetic"
    shutil.copytree(EXAMPLE, run)
    manifest = json.loads((run / "fep.json").read_text())
    manifest["reverse_files"] = ["missing.csv"]
    (run / "fep.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="declared reverse file 'missing.csv' is missing"):
        _fep_metrics(run)


def test_absent_reverse_key_still_means_intentional_no_reverse(tmp_path):
    # Only DECLARED lists are mandatory: a manifest with no reverse_files
    # key keeps the documented no-reverse semantics (hysteresis skipped,
    # no contract error).
    import shutil

    from simval.pipeline import run_checks

    run = tmp_path / "synthetic"
    shutil.copytree(EXAMPLE, run)
    ctx = FepEngine().load_context(run, selection="n/a")
    results = {r.name: r for r in run_checks(ctx)}
    assert "fep_run_contract" not in results
    assert results["fep_hysteresis"].detail.get("skipped") is True
