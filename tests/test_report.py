"""Display-layer tests: report.py renderers are pure formatting and must
never touch verdict semantics (those live in oracle/compare tests)."""
from simval.oracle.validate import compare_metrics
from simval.report import (
    fmt_dur,
    fmt_num,
    render_check_row,
    render_identity,
    render_oracle_row,
    tol_str,
)


def test_fmt_num():
    assert fmt_num(None) == "         -"
    assert fmt_num(0.5, 4) == " 0.5"
    assert fmt_num(2000) == "      2000"
    assert fmt_num(1.96e-06, 11) == "   1.96e-06"
    assert fmt_num("weird", 6) == " weird"


def test_fmt_dur():
    assert fmt_dur(0.4) == "0.40s"
    assert fmt_dur(63.7) == "1m 04s"
    assert fmt_dur(3900) == "1h 05m"
    assert fmt_dur(None) == "-"


def test_tol_str():
    assert tol_str("exact", None) == "exact"
    assert tol_str("rel", 0.1) == "rel 0.1"
    assert tol_str("max", 1.0) == "max 1"
    assert tol_str("interval", [0.0, 1.0]) == "interval [0, 1]"


def test_render_check_row_pass_and_exceeded():
    ok = {"name": "cfl_stability", "passed": True, "value": 0.5, "threshold": 1.0, "detail": {}}
    line = render_check_row(ok)
    assert "[PASS]" in line and "value=" in line and "threshold=" in line
    assert "exceeded" not in line

    over = {"name": "energy_drift", "passed": False, "value": 0.04, "threshold": 0.01, "detail": {}}
    assert "exceeded 4.00x" in render_check_row(over)

    # failed but value within threshold (compound check): no ratio claim
    compound = {"name": "structural_equilibration", "passed": False, "value": 3.0, "threshold": 10.0, "detail": {}}
    assert "exceeded" not in render_check_row(compound)

    # failed by exception: the error replaces the numbers
    err = {"name": "hydrogen_bonds", "passed": False, "value": 0.0, "threshold": 0.0,
           "detail": {"status": "error", "error": "ValueError: boom"}}
    assert "error: ValueError: boom" in render_check_row(err)


def test_render_oracle_row_all_tolerance_kinds():
    ref = {
        "mean_rmsd_nm": 0.35,        # rel, drifted
        "n_frames": 500,             # exact, ok
        "norm_drift": 1e-12,         # abs, drifted
        "overlap_min_eigenvalue": 0.05,  # min, missing from candidate
    }
    cand = {"mean_rmsd_nm": 0.40, "n_frames": 500, "norm_drift": 5e-9}
    compared = compare_metrics(cand, ref)
    compared.pop("__passed__")
    lines = {k: render_oracle_row(k, v) for k, v in compared.items()}
    assert "[DRIFT]" in lines["mean_rmsd_nm"]
    assert "tol=rel 0.1" in lines["mean_rmsd_nm"]
    assert "drel 0.143 > tol 0.1" in lines["mean_rmsd_nm"]
    assert "[ ok ]" in lines["n_frames"] and "tol=exact" in lines["n_frames"]
    assert "|delta| 5e-09 > tol 1e-09" in lines["norm_drift"]
    assert "missing from candidate" in lines["overlap_min_eigenvalue"]


def test_render_identity():
    detail = {
        "wave.json": {"golden": "a" * 64, "candidate": "b" * 64, "problem": "content mismatch"},
        "topol.tpr": {"golden": "c" * 64, "candidate": None, "problem": "missing input"},
        "ok.txt": {"golden": "d" * 64, "candidate": "d" * 64},
        "__undeclared_inputs__": {"problem": ["extra.json"]},
    }
    lines = render_identity(detail)
    text = "\n".join(lines)
    assert "[FAIL] wave.json" in text and "(content mismatch)" in text
    assert "(missing input)" in text
    assert "undeclared inputs present: extra.json" in text
    assert "(1 input matched)" in text
