"""Orchestrator tests: ontos grid runs, tamper detection, tabulate/outliers, CLI."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from simval.cli import main
from simval.orchestrate import (
    load_grid,
    outliers,
    row_ok,
    run_grid,
    tabulate,
    validate_run_names,
    verify_run,
)

_ONTOS_CANDIDATES = (
    Path.home() / "Documents" / "Code Projects" / "Sides" / "ontos",
    Path.home() / "Documents" / "Code Projects" / "ontos",
)


def _ensure_bin() -> Path | None:
    env = os.environ.get("ONTOS_BIN")
    if env and Path(env).exists():
        return Path(env)
    for repo in _ONTOS_CANDIDATES:
        bin_path = repo / "target" / "release" / "ontos"
        if bin_path.exists():
            return bin_path
        if shutil.which("cargo") is not None and (repo / "Cargo.toml").exists():
            subprocess.run(
                ["cargo", "build", "--release", "--quiet", "--manifest-path", str(repo / "Cargo.toml")],
                capture_output=True,
                text=True,
            )
            if bin_path.exists():
                return bin_path
    return None


BIN = _ensure_bin()
requires_bin = pytest.mark.skipif(BIN is None, reason="ontos binary unavailable and cargo build failed")


def _cell(workdir, i: int) -> Path:
    return Path(workdir) / f"cell-{i:04d}"


GRID = [
    {"name": "all_fine", "mode": "gravity", "ticks": 40, "seed": 42, "bodies": 8},
    {"name": "window", "mode": "gravity", "ticks": 50, "seed": 42, "bodies": 8,
     "events": [["demote-at", 20, 0, 0], ["promote-at", 40, 0, 0]]},
    {"name": "observer", "mode": "gravity", "ticks": 60, "seed": 5, "bodies": 8,
     "observer": 777},
    {"name": "collapse", "mode": "gravity", "ticks": 40, "seed": 42, "bodies": 8,
     "events": [["collapse-at", 6, 0, 0], ["expand-at", 36, 0, 0]]},
    {"name": "radial", "mode": "gravity", "ticks": 40, "seed": 42, "bodies": 8, "radial": True,
     "events": [["collapse-at", 6, 0, 0], ["expand-at", 36, 0, 0]]},
    {"name": "restitution", "mode": "gravity", "ticks": 40, "seed": 11, "bodies": 16,
     "contacts": True, "restitution": 0.5, "friction": 0.25},
    {"name": "walls", "mode": "gravity", "ticks": 40, "seed": 22, "bodies": 8,
     "walls": True},
]


@requires_bin
def test_grid_verifies_clean(tmp_path):
    results = run_grid(GRID, ontos_bin=BIN, workdir=tmp_path)
    assert [r["run"] for r in results] == [s["name"] for s in GRID]
    for i, (r, spec) in enumerate(zip(results, GRID)):
        assert "_error" not in r, r
        assert r["mismatch_count"] == 0
        assert r["checks_failed"] == 0
        assert r["ticks_verified"] == spec["ticks"]
        assert r["wall_s"] > 0
        assert (_cell(tmp_path, i) / "ontos.stream").exists()
        meta = json.loads((_cell(tmp_path, i) / "ontos.json").read_text())
        assert meta["seed"] == spec["seed"]
        assert meta["ticks"] == spec["ticks"]
    collapse = results[3]
    assert collapse["collapse_events"] == 1
    assert collapse["expand_events"] == 1
    radial = results[4]
    assert radial["radial_events"] == 1
    assert radial["radial_worst"] < 1e-9
    restitution = results[5]
    assert restitution["contact_events"] >= 1
    walls = results[6]
    assert walls["contact_events"] >= 0


@requires_bin
def test_tampered_stream_reports_mismatch(tmp_path):
    results = run_grid(GRID[:2], ontos_bin=BIN, workdir=tmp_path)
    assert all(r["mismatch_count"] == 0 for r in results)
    stream = _cell(tmp_path, 0) / "ontos.stream"
    data = bytearray(stream.read_bytes())
    data[-1] ^= 0x01
    stream.write_bytes(bytes(data))
    row = verify_run(_cell(tmp_path, 0))
    assert row["mismatch_count"] > 0
    assert row["checks_failed"] > 0


@requires_bin
def test_cli_smoke(tmp_path, capsys):
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps(GRID[:2]))
    out = tmp_path / "results.json"
    rc = main(["orchestrate", "--grid", str(grid), "--ontos-bin", str(BIN), "--out", str(out)])
    captured = capsys.readouterr().out
    assert rc == 0
    assert "orchestrate" in captured
    assert "all_fine" in captured and "window" in captured
    assert "outliers" in captured
    # per-run progress + final grid verdict summary
    assert "[1/2] all_fine: ok" in captured
    assert "grid: CLEAN | 2/2 runs clean" in captured
    assert "results ->" in captured
    saved = json.loads(out.read_text())
    assert len(saved["runs"]) == 2
    assert all(r["mismatch_count"] == 0 for r in saved["runs"])


def _gravity_row(name, **over):
    row = {
        "run": name, "ticks": 100, "mismatch_count": 0,
        "max_position_deviation": 1e-5, "momentum_drift": 1e-3, "energy_drift": 1e-4,
        "post_expansion_deviation": 0.0, "collapse_events": 0, "expand_events": 0,
        "collapse_energy_worst": 0.0, "checks_failed": 0, "wall_s": 1.0,
    }
    row.update(over)
    return row


def test_tabulate_renders_rows_and_missing_metrics():
    table = tabulate([
        _gravity_row("a"),
        _gravity_row("b", momentum_drift=None),
        {"run": "c", "_error": "boom"},
    ])
    lines = table.splitlines()
    # header: status column + short aliases, rule underneath
    assert "status" in lines[0] and "pos_dev" in lines[0] and "mismatch" in lines[0]
    assert set(lines[1].strip()) == {"-"}
    assert "a" in lines[2] and "1e-05" in lines[2] and " ok" in lines[2]
    assert "-" in lines[3].split()
    assert "ERROR: boom" in lines[4]
    assert table.endswith("\n")


def test_tabulate_flags_dirty_rows():
    table = tabulate([
        _gravity_row("clean"),
        _gravity_row("mismatched", mismatch_count=3),
        _gravity_row("checks", checks_failed=2),
    ])
    lines = table.splitlines()
    assert "ok" in lines[2].split()
    assert "FAIL" in lines[3].split()
    assert "FAIL" in lines[4].split()


def test_row_ok_matches_exit_criterion():
    # the exact predicate the CLI exit code has always used
    assert row_ok(_gravity_row("a"))
    assert not row_ok(_gravity_row("m", mismatch_count=1))
    assert not row_ok(_gravity_row("c", checks_failed=1))
    assert not row_ok({"run": "e", "_error": "boom"})
    assert not row_ok({"run": "x"})  # missing mismatch_count fails closed


def test_run_grid_reports_progress_per_row(tmp_path):
    fake = _write_fake_ontos(tmp_path / "fake-ontos")
    specs = [
        {"name": "ok_a", "mode": "gravity", "ticks": 10, "seed": 42, "bodies": 8},
        {"name": "bad", "mode": "nonsense"},
        {"name": "ok_b", "mode": "gravity", "ticks": 10, "seed": 7, "bodies": 8},
    ]
    seen = []
    results = run_grid(
        specs, ontos_bin=fake, workdir=tmp_path / "grid",
        progress=lambda row, i, total: seen.append((i, total, row["run"])),
    )
    assert seen == [(0, 3, "ok_a"), (1, 3, "bad"), (2, 3, "ok_b")]
    assert "_error" in results[1]


def test_outliers_flags_mad_deviation():
    rows = [
        _gravity_row("a"),
        _gravity_row("b"),
        _gravity_row("c"),
        _gravity_row("d", momentum_drift=1e-1),
    ]
    flags = outliers(rows, k=3.0)
    assert [(f["run"], f["metric"]) for f in flags] == [("d", "momentum_drift")]
    assert flags[0]["median"] == 1e-3

    assert outliers(rows[:3], k=3.0) == []

    spread = [_gravity_row("a"), _gravity_row("b", momentum_drift=1.2e-3),
              _gravity_row("c", momentum_drift=0.8e-3), _gravity_row("d", momentum_drift=1.1e-3),
              _gravity_row("e", momentum_drift=1e-1)]
    flagged = outliers(spread, k=3.0)
    assert [(f["run"], f["metric"]) for f in flagged] == [("e", "momentum_drift")]
    assert flagged[0]["mad"] > 0


def test_outliers_ignores_error_rows():
    rows = [_gravity_row("a"), _gravity_row("b"), {"run": "c", "_error": "boom"}]
    assert outliers(rows) == []


def test_outliers_needs_population():
    assert outliers([_gravity_row("a")]) == []


# --- ONT-001: the orchestrator persists every CLI-affecting field ---


def test_ontos_json_persists_full_contract():
    from simval.orchestrate import _ontos_json, normalize_spec

    spec = normalize_spec(
        {
            "name": "full",
            "mode": "gravity",
            "ticks": 40,
            "seed": 11,
            "bodies": 16,
            "events": [["collapse-at", 6, 1, 1], ["expand-at", 30, 1, 1]],
            "observer": 777,
            "contacts": True,
            "radial": True,
            "restitution": 0.5,
            "friction": 0.25,
        }
    )
    meta = _ontos_json(spec)
    assert meta["mode"] == "gravity"
    assert meta["seed"] == 11 and meta["ticks"] == 40 and meta["bodies"] == 16
    assert meta["events"] == [[6, 1, 1, 2], [30, 1, 1, 1]]
    assert meta["observer"] == 777
    assert meta["contacts"] is True and meta["radial"] is True
    assert meta["shells"] is False and meta["walls"] is False
    assert meta["multipole"] is True
    assert meta["restitution"] == 0.5 and meta["friction"] == 0.25
    # round-trips through the gravity contract parser
    from simval.ontos_gravity import GravityContract

    contract = GravityContract.from_metadata(meta)
    assert contract.body_count == 16 and contract.ticks == 40
    assert contract.events == ((6, 3, 2), (30, 3, 1))


def test_ontos_json_life_events_persisted():
    from simval.orchestrate import _ontos_json, normalize_spec

    spec = normalize_spec({"name": "l", "mode": "life", "events": [["demote", 1, 0]]})
    meta = _ontos_json(spec)
    assert meta["mode"] == "life"
    assert meta["events"] == [["demote", 1, 0]]


# --- ORCH-001/002/003: safe names, per-cell failure isolation, empty grids ---


def _dummy_bin(tmp_path) -> Path:
    bin_path = tmp_path / "ontos-dummy"
    bin_path.write_text("#!/bin/sh\nexit 0\n")
    bin_path.chmod(0o755)
    return bin_path


@pytest.mark.parametrize(
    "name",
    ["../escape", "/tmp/absolute", "a/b", "..", "."],
)
def test_unsafe_run_names_rejected_before_execution(tmp_path, name):
    sentinel = tmp_path / "outside.txt"
    sentinel.write_text("precious")
    grid = [{"name": name, "mode": "gravity", "ticks": 5}]
    with pytest.raises(ValueError, match="unsafe grid run name"):
        run_grid(grid, ontos_bin=_dummy_bin(tmp_path), workdir=tmp_path / "work")
    assert sentinel.read_text() == "precious"
    assert not (tmp_path / "work").exists() or not any((tmp_path / "work").iterdir())


def test_duplicate_run_names_rejected_before_execution(tmp_path):
    grid = [
        {"name": "same", "mode": "gravity", "ticks": 5},
        {"name": "same", "mode": "gravity", "ticks": 6},
    ]
    with pytest.raises(ValueError, match="duplicate grid run name"):
        run_grid(grid, ontos_bin=_dummy_bin(tmp_path), workdir=tmp_path / "work")


def test_validate_run_names_accepts_plain_names():
    validate_run_names(["all_fine", "run-1", "window_42"])
    with pytest.raises(ValueError):
        validate_run_names(["ok", "../escape"])
    with pytest.raises(ValueError):
        validate_run_names(["dup", "dup"])


@requires_bin
def test_malformed_cell_isolated_and_grid_continues(tmp_path):
    grid = [
        {"name": "good1", "mode": "gravity", "ticks": 10, "seed": 42, "bodies": 4},
        {"name": "bad", "mode": "nonsense"},
        {"name": "good2", "mode": "gravity", "ticks": 10, "seed": 43, "bodies": 4},
    ]
    results = run_grid(grid, ontos_bin=BIN, workdir=tmp_path)
    assert len(results) == 3
    assert "_error" not in results[0] and results[0]["mismatch_count"] == 0
    assert "_error" in results[1] and "cell 1" in results[1]["_error"]
    assert "_error" not in results[2] and results[2]["mismatch_count"] == 0
    assert (_cell(tmp_path, 2) / "ontos.stream").exists()


def test_load_grid_rejects_empty_list(tmp_path):
    grid = tmp_path / "empty.json"
    grid.write_text("[]")
    with pytest.raises(ValueError, match="no run specs"):
        load_grid(grid)


def test_cli_empty_grid_exits_nonzero(tmp_path, capsys):
    grid = tmp_path / "empty.json"
    grid.write_text("[]")
    rc = main(["orchestrate", "--grid", str(grid), "--ontos-bin", str(_dummy_bin(tmp_path))])
    assert rc == 1
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "no run specs" in out


def test_cli_dirty_grid_reports_failure_summary(tmp_path, capsys):
    fake = _write_fake_ontos(tmp_path / "fake-ontos")
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps([
        {"name": "good", "mode": "gravity", "ticks": 10, "seed": 42, "bodies": 8},
        {"name": "badmode", "mode": "nonsense"},
    ]))
    rc = main(["orchestrate", "--grid", str(grid), "--ontos-bin", str(fake)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "[2/2] badmode: ERROR" in out
    assert "grid: FAILED | 1/2 runs clean | 1 error" in out


def test_cli_success_requires_rows(tmp_path, capsys):
    # A grid that loads but produces zero rows can never report success.
    from simval.cli import main as cli_main

    assert cli_main(["orchestrate", "--grid", "does-not-exist.json"]) == 1


# --- ORCH-004: a hung producer is killed, isolated, and never retried ---


def _write_fake_ontos(path: Path) -> Path:
    src = str(Path(__file__).parent.parent / "src")
    script = f'''#!/usr/bin/env python3
import struct
import sys
import time

sys.path.insert(0, r"{src}")

from pathlib import Path

from simval.ontos_gravity import GravityWorld

LEVELS = {{"--demote-at": 0, "--promote-at": 1, "--collapse-at": 2, "--expand-at": 1}}


def emit_gravity(seed, bodies, ticks, schedule, out):
    world = GravityWorld(seed, bodies)
    buf = bytearray(b"ONTO") + struct.pack("<IIII", 2, 128, 128, bodies)
    events = {{}}
    for t, region, level in schedule:
        world.schedule(t, region, level)
        events.setdefault(t, []).append((region, level))
    for _ in range(ticks):
        entering = world.tick + 1
        for region, level in events.get(entering, []):
            buf += b"\\x04" + struct.pack("<IIB", region % 2, region // 2, level)
        world.step()
        buf += b"\\x01" + struct.pack("<Q", world.tick)
        buf += b"\\x02" + struct.pack("<Q", bodies)
        fine, coarse_n, mass, px, py, energy = world.totals()
        buf += b"\\x07" + struct.pack("<QQQdddd", world.tick, fine, coarse_n, mass, px, py, energy)
        for rx, ry in ((0, 0), (1, 0), (0, 1), (1, 1)):
            level, pop, rhash = world.region_hash(ry * 2 + rx)
            buf += b"\\x05" + struct.pack("<QIIBQQ", world.tick, rx, ry, level, pop, rhash)
        for i in range(bodies):
            b, region, level = world.emitted_state(i)
            buf += b"\\x06" + struct.pack(
                "<QIBBddddd", world.tick, i, region, level,
                b["x"], b["y"], b["vx"], b["vy"], b["mass"])
    Path(out).write_bytes(bytes(buf))


def main(argv):
    seed, ticks, bodies, out, schedule = 42, 20, 8, None, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--ticks":
            ticks = int(argv[i + 1]); i += 2
        elif a == "--seed":
            seed = int(argv[i + 1]); i += 2
        elif a == "--bodies":
            bodies = int(argv[i + 1]); i += 2
        elif a == "--out":
            out = argv[i + 1]; i += 2
        elif a in LEVELS:
            t, rx, ry = int(argv[i + 1]), int(argv[i + 2]), int(argv[i + 3])
            schedule.append((t, ry * 2 + rx, LEVELS[a])); i += 4
        else:
            i += 2 if argv[i].startswith("--") else 1
    if seed == 999:
        time.sleep(60)
    emit_gravity(seed, bodies, ticks, schedule, out)


main(sys.argv[1:])
'''
    path.write_text(script)
    path.chmod(0o755)
    return path


def test_hung_cell_times_out_isolated_and_grid_continues(tmp_path):
    fake = _write_fake_ontos(tmp_path / "fake-ontos")
    specs = [
        {"name": "ok_a", "mode": "gravity", "ticks": 20, "seed": 42, "bodies": 8},
        {"name": "hung", "mode": "gravity", "ticks": 20, "seed": 999, "bodies": 8},
        {"name": "ok_b", "mode": "gravity", "ticks": 20, "seed": 7, "bodies": 8},
    ]
    results = run_grid(
        specs, ontos_bin=fake, workdir=tmp_path / "grid", cell_timeout_s=5.0
    )
    assert len(results) == 3
    assert "_error" not in results[0]
    assert results[0]["mismatch_count"] == 0
    assert "did not finish within 5.0s" in results[1]["_error"]
    assert "no retry" in results[1]["_error"]
    assert "_error" not in results[2], results[2]
    assert results[2]["mismatch_count"] == 0
    assert results[2]["ticks_verified"] == 20


def test_run_grid_rejects_bad_timeout(tmp_path):
    fake = _write_fake_ontos(tmp_path / "fake-ontos")
    with pytest.raises(ValueError, match="cell_timeout_s"):
        run_grid([], ontos_bin=fake, workdir=tmp_path, cell_timeout_s=0.0)
