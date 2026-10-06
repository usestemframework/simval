import shutil
from pathlib import Path

from simval.cli import main

WAVE = Path(__file__).parent.parent / "examples" / "wave" / "pulse"


def _wave_copy(parent: Path, name: str = "w") -> Path:
    run = parent / name
    run.mkdir()
    shutil.copy(WAVE / "wave.json", run / "wave.json")
    return run


def test_engines(capsys):
    assert main(["engines"]) == 0
    assert "wave-fdtd" in capsys.readouterr().out


def test_cases(capsys):
    assert main(["cases"]) == 0
    assert "kepler_two_body" in capsys.readouterr().out


def test_inspect(tmp_path, capsys):
    run = _wave_copy(tmp_path)
    assert main(["inspect", str(run), "--selection", "n/a"]) == 0
    assert "wave-fdtd" in capsys.readouterr().out


def test_diagnose_then_verify_manifest(tmp_path, capsys):
    run = _wave_copy(tmp_path)
    assert main(["diagnose", str(run), "--selection", "n/a", "--out", "prov.json"]) == 0
    assert main(["verify-manifest", str(run / "prov.json")]) == 0


def test_validate_wave(tmp_path):
    run = _wave_copy(tmp_path)
    assert main(["validate", str(run), "--case", "wave_pulse_stable", "--selection", "n/a"]) == 0


def test_sweep(tmp_path, capsys):
    root = tmp_path / "sw"
    root.mkdir()
    _wave_copy(root, "a")
    _wave_copy(root, "b")
    assert main(["sweep", str(root), "--selection", "n/a"]) == 0
    out = capsys.readouterr().out
    assert "2 runs" in out
    assert "run" in out and "energy_growth" in out
    assert set(out.splitlines()[2].strip()) == {"-"}  # rule under the table header


def test_compare_cli(tmp_path, capsys):
    root = tmp_path / "cmp"
    root.mkdir()
    a = _wave_copy(root, "a")
    b = _wave_copy(root, "b")
    assert main(["compare", str(a), str(b), "--selection", "n/a"]) == 0
    out = capsys.readouterr().out
    assert "metric" in out and "drel" in out
    assert "cfl" in out


def test_validate_identity_mismatch_shows_what_differs(tmp_path, capsys):
    import json

    run = _wave_copy(tmp_path, "tampered")
    cfg = json.loads((run / "wave.json").read_text())
    cfg["nx"] = 180
    (run / "wave.json").write_text(json.dumps(cfg))
    rc = main(["validate", str(run), "--case", "wave_pulse_stable", "--selection", "n/a"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "DRIFT" in out
    assert "identity mismatch" in out
    assert "wave.json" in out and "content mismatch" in out


def test_bad_run_dir_is_clean_not_traceback(capsys):
    rc = main(["diagnose", "/tmp/simval_definitely_missing_xyz"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "error" in out
    assert "Traceback" not in out
