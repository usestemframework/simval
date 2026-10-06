from simval.cli import main
from simval.fixtures import make_run_dir
from simval.manifest import load_manifest
from simval.pipeline import diagnose


def test_good_run_dir_passes(tmp_path):
    run = make_run_dir(tmp_path / "good", good=True)
    manifest = diagnose(run)
    assert manifest["verdict"] == "pass"
    assert len(manifest["diagnostics"]) >= 4
    loaded = load_manifest(run / "provenance.json")
    assert loaded["verdict"] == "pass"
    assert loaded["files"]


def test_bad_run_dir_fails(tmp_path):
    run = make_run_dir(tmp_path / "bad", good=False)
    manifest = diagnose(run)
    assert manifest["verdict"] == "fail"
    failed = [d["name"] for d in manifest["diagnostics"] if not d["passed"]]
    assert len(failed) >= 1


def test_cli_main_exit_code_and_output(tmp_path, capsys):
    run = make_run_dir(tmp_path / "g2", good=True)
    rc = main(["diagnose", str(run)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "verdict: PASS" in out
    assert "checks passed" in out
    assert "[PASS]" in out
    assert "provenance ->" in out


def test_cli_main_failing_run_nonzero(tmp_path, capsys):
    run = make_run_dir(tmp_path / "b2", good=False)
    rc = main(["diagnose", str(run)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "verdict: FAIL" in out
    # failing upper-bound checks show the exceed ratio
    assert "exceeded" in out
    assert "provenance ->" in out
