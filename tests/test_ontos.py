"""Ontos adapter tests: independent reference vs recorded streams."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from simval.context import select_engine
from simval.ontos import (
    FNV_OFFSET_BASIS,
    ReferenceWorld,
    check_reference_match,
    check_tick_monotonicity,
    fnv1a64,
    parse_stream,
    verify_stream,
)
from simval.ontos_eng import OntosEngine
from simval.pipeline import run_checks

EXAMPLES = Path(__file__).parent.parent / "examples" / "ontos"
R_PENTOMINO = EXAMPLES / "r_pentomino"
ALL_FINE = EXAMPLES / "all_fine"


def test_fnv_known_values():
    assert fnv1a64(b"") == FNV_OFFSET_BASIS
    assert fnv1a64(b"") == 0xCBF29CE484222325
    assert fnv1a64(b"a") == 0xAF63DC4C8601EC8C


@pytest.mark.parametrize(
    "run_dir,seed",
    [(R_PENTOMINO, 42), (ALL_FINE, 7)],
)
def test_reference_matches_stream(run_dir, seed):
    summary = verify_stream(run_dir / "ontos.stream", seed)
    assert summary["mismatch_count"] == 0
    assert summary["tick_monotonic"]
    assert summary["ticks_verified"] == 64
    assert summary["records_compared"] == 320


def test_corrupted_hash_fails(tmp_path):
    data = bytearray((R_PENTOMINO / "ontos.stream").read_bytes())
    header, records = parse_stream(R_PENTOMINO / "ontos.stream")
    first_state = next(i for i, r in enumerate(records) if r[0] == "state")
    offset = 16
    for record in records:
        size = {"tick": 9, "snapshot": 9, "flip": 17, "level": 10, "state": 34}[record[0]]
        if record[0] == "state" and records.index(record) == first_state:
            data[offset + size - 1] ^= 0xFF
            break
        offset += size
    corrupt = tmp_path / "ontos.stream"
    corrupt.write_bytes(bytes(data))
    summary = verify_stream(corrupt, 42)
    assert summary["mismatch_count"] > 0
    result = check_reference_match(summary)
    assert not result.passed


def test_parser_rejects_bad_magic(tmp_path):
    bad = tmp_path / "ontos.stream"
    bad.write_bytes(b"NOPE" + b"\x00" * 12)
    with pytest.raises(ValueError, match="bad magic"):
        parse_stream(bad)


def test_parser_rejects_truncation(tmp_path):
    data = (R_PENTOMINO / "ontos.stream").read_bytes()
    truncated = tmp_path / "ontos.stream"
    truncated.write_bytes(data[:-4])
    with pytest.raises((ValueError, struct.error)):
        parse_stream(truncated)


def test_engine_detect_and_diagnose(tmp_path):
    import shutil

    run = tmp_path / "ontos_run"
    shutil.copytree(R_PENTOMINO, run)
    engine = select_engine(run)
    assert isinstance(engine, OntosEngine)
    assert engine.name == "ontos"
    ctx = engine.load_context(run, selection="default")
    results = run_checks(ctx)
    names = {r.name for r in results}
    assert "ontos_reference_match" in names
    assert "ontos_tick_monotonicity" in names
    assert all(r.passed for r in results if r.name.startswith("ontos_"))


def test_wrong_seed_fails_on_promote_fixture(tmp_path):
    import shutil

    run = tmp_path / "ontos_run"
    shutil.copytree(EXAMPLES / "promote_roundtrip", run)
    summary_seeded = verify_stream(run / "ontos.stream", 42)
    assert summary_seeded["mismatch_count"] == 0
    summary_wrong = verify_stream(run / "ontos.stream", 43)
    assert summary_wrong["mismatch_count"] > 0
    assert not check_reference_match(summary_wrong).passed


def _emit_stream(path, world, schedule, ticks):
    """Serialize a reference run to stream bytes per spec section 9."""
    out = bytearray(b"ONTO")
    out += struct.pack("<III", 1, 128, 128)
    for rx, ry, level in schedule:
        world.set_level(rx, ry, "fine" if level else "coarse")
        out += b"\x04" + struct.pack("<IIB", rx, ry, level)
    for _ in range(ticks):
        world.step()
        out += b"\x01" + struct.pack("<Q", world.tick)
        out += b"\x02" + struct.pack("<Q", world.population())
        for ry in (0, 1):
            for rx in (0, 1):
                region = world.regions[world.region_index(rx, ry)]
                out += b"\x05" + struct.pack(
                    "<QIIBQQ",
                    world.tick,
                    rx,
                    ry,
                    1 if region.level == "fine" else 0,
                    region.population(),
                    region.hash(),
                )
    Path(path).write_bytes(bytes(out))


def test_randomized_schedules_self_consistent(tmp_path):
    import random

    rng = random.Random(20260906)
    for trial in range(12):
        seed = rng.randrange(2**64)
        ops = rng.randrange(0, 5)
        schedule = [
            (rng.randrange(2), rng.randrange(2), rng.randrange(2)) for _ in range(ops)
        ]
        ticks = rng.randrange(1, 40)
        world = ReferenceWorld(seed=seed)
        world.seed_r_pentomino()
        stream = tmp_path / f"case_{trial}.stream"
        _emit_stream(stream, world, schedule, ticks)
        summary = verify_stream(stream, seed)
        assert summary["mismatch_count"] == 0, (trial, schedule, summary["mismatches"])
        assert summary["tick_monotonic"] is True
        assert summary["ticks_verified"] == ticks


def test_module_cli_verifies_and_rejects(tmp_path, capsys):
    from simval.ontos import _main

    meta = str(R_PENTOMINO / "ontos.json")
    assert _main([str(R_PENTOMINO / "ontos.stream"), "42", "--metadata", meta]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "mismatches=0" in out
    data = bytearray((R_PENTOMINO / "ontos.stream").read_bytes())
    data[-1] ^= 0xFF
    corrupt = tmp_path / "corrupt.stream"
    corrupt.write_bytes(bytes(data))
    assert _main([str(corrupt), "42", "--metadata", meta]) == 1
    assert _main([str(tmp_path / "missing.stream"), "42", "--metadata", meta]) == 2


# --- ONT-007: the standalone CLI requires the expected-run contract ---


def test_module_cli_rejects_bare_stream_without_contract(tmp_path, capsys):
    from simval.ontos import _main

    rc = _main([str(R_PENTOMINO / "ontos.stream"), "42"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no expected-run contract" in err


def test_module_cli_no_contract_flag_warns_loudly(tmp_path, capsys):
    from simval.ontos import _main

    rc = _main([str(R_PENTOMINO / "ontos.stream"), "42", "--no-contract"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err and "NOT validated" in captured.err
    assert "[NO CONTRACT]" in captured.out


def test_module_cli_explicit_flags_build_life_contract(tmp_path, capsys):
    from simval.ontos import _main

    rc = _main(
        [str(R_PENTOMINO / "ontos.stream"), "42", "--mode", "life", "--ticks", "64",
         "--demote", "1", "0", "--demote", "0", "1"]
    )
    assert rc == 0
    assert "[NO CONTRACT]" not in capsys.readouterr().out


def test_module_cli_contract_flags_detect_wrong_schedule(tmp_path, capsys):
    from simval.ontos import _main

    rc = _main(
        [str(R_PENTOMINO / "ontos.stream"), "42", "--mode", "life", "--ticks", "64",
         "--promote", "0", "0"]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "CONTRACT" in out


def test_module_cli_metadata_tamper_fails(tmp_path, capsys):
    from simval.ontos import _main

    meta = json.loads((R_PENTOMINO / "ontos.json").read_text())
    meta["ticks"] = 50  # claims a shorter horizon than the stream carries
    meta_path = tmp_path / "ontos.json"
    meta_path.write_text(json.dumps(meta))
    rc = _main([str(R_PENTOMINO / "ontos.stream"), "42", "--metadata", str(meta_path)])
    assert rc == 1
    assert "ticks=50" in capsys.readouterr().out


def test_module_cli_rejects_metadata_plus_flags(tmp_path, capsys):
    from simval.ontos import _main

    rc = _main(
        [str(R_PENTOMINO / "ontos.stream"), "42", "--metadata",
         str(R_PENTOMINO / "ontos.json"), "--ticks", "64"]
    )
    assert rc == 2
    assert "mutually exclusive" in capsys.readouterr().err


# --- ONT-002/004/005: strict grammar, frame-tick equality, canonical encodings ---

_V1_SIZES = {"tick": 9, "snapshot": 9, "flip": 17, "level": 10, "state": 34}


def _v1_stream_bytes(tmp_path, name, schedule=(), ticks=4):
    world = ReferenceWorld(seed=42)
    world.seed_r_pentomino()
    path = tmp_path / name
    _emit_stream(path, world, list(schedule), ticks)
    return path.read_bytes()


def test_v1_header_only_rejected(tmp_path):
    p = tmp_path / "h.stream"
    p.write_bytes(b"ONTO" + struct.pack("<III", 1, 128, 128))
    with pytest.raises(ValueError, match="no tick frames"):
        parse_stream(p)


def test_v1_tickheader_only_rejected(tmp_path):
    data = _v1_stream_bytes(tmp_path, "base.stream")
    p = tmp_path / "t.stream"
    p.write_bytes(data[:16] + b"\x01" + struct.pack("<Q", 1))
    with pytest.raises(ValueError, match="incomplete tick frame"):
        parse_stream(p)


def _v1_first_offset(data: bytes, kind: str) -> int:
    path_like = data
    off = 16
    while off < len(path_like):
        tag = path_like[off]
        name = {1: "tick", 2: "snapshot", 3: "flip", 4: "level", 5: "state"}[tag]
        if name == kind:
            return off
        off += _V1_SIZES[name]
    raise AssertionError(f"no {kind} record found")


def test_v1_drop_one_snapshot_rejected(tmp_path):
    data = _v1_stream_bytes(tmp_path, "base.stream")
    off = _v1_first_offset(data, "snapshot")
    p = tmp_path / "m.stream"
    p.write_bytes(data[:off] + data[off + 9 :])
    with pytest.raises(ValueError):
        parse_stream(p)


def test_v1_drop_one_state_rejected(tmp_path):
    data = _v1_stream_bytes(tmp_path, "base.stream")
    off = _v1_first_offset(data, "state")
    p = tmp_path / "m.stream"
    p.write_bytes(data[:off] + data[off + 34 :])
    with pytest.raises(ValueError):
        parse_stream(p)


def test_v1_duplicate_snapshot_rejected(tmp_path):
    data = _v1_stream_bytes(tmp_path, "base.stream")
    off = _v1_first_offset(data, "snapshot")
    p = tmp_path / "m.stream"
    p.write_bytes(data[: off + 9] + data[off : off + 9] + data[off + 9 :])
    with pytest.raises(ValueError):
        parse_stream(p)


def test_v1_duplicate_tick_frame_rejected(tmp_path):
    data = _v1_stream_bytes(tmp_path, "base.stream")
    off = _v1_first_offset(data, "tick")
    p = tmp_path / "m.stream"
    p.write_bytes(data + data[off:])
    with pytest.raises(ValueError, match="non-consecutive"):
        parse_stream(p)


def test_v1_append_after_last_tick_rejected(tmp_path):
    data = _v1_stream_bytes(tmp_path, "base.stream")
    stray = b"\x02" + struct.pack("<Q", 0)
    p = tmp_path / "m.stream"
    p.write_bytes(data + stray)
    with pytest.raises(ValueError, match="outside an open tick frame"):
        parse_stream(p)


def test_v1_dangling_level_at_eof_rejected(tmp_path):
    data = _v1_stream_bytes(tmp_path, "base.stream")
    p = tmp_path / "m.stream"
    p.write_bytes(data + b"\x04" + struct.pack("<IIB", 0, 0, 0))
    with pytest.raises(ValueError, match="dangling"):
        parse_stream(p)


def test_v1_state_tick_mismatch_rejected(tmp_path):
    # ONT-004: a RegionState tick off by one from the frame must fail.
    data = bytearray(_v1_stream_bytes(tmp_path, "base.stream"))
    off = _v1_first_offset(bytes(data), "state")
    data[off + 1] += 1
    p = tmp_path / "m.stream"
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="does not match the open frame tick"):
        parse_stream(p)


def test_v1_level_two_aliases_coarse_rejected(tmp_path):
    # ONT-005: level 2 in a version-1 stream must not be read as coarse 0.
    data = bytearray(_v1_stream_bytes(tmp_path, "sched.stream", schedule=[(1, 0, 0)]))
    data[16 + 1 + 4 + 4] = 2  # level byte of the first RegionLevel record
    p = tmp_path / "m.stream"
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="version-1 region level 2"):
        parse_stream(p)


def test_v1_region_coord_rewrite_rejected(tmp_path):
    # ONT-005: (0,1) rewritten as (2,0) must not alias region 2.
    data = bytearray(_v1_stream_bytes(tmp_path, "sched.stream", schedule=[(0, 1, 0)]))
    struct.pack_into("<II", data, 16 + 1, 2, 0)
    p = tmp_path / "m.stream"
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="noncanonical region coordinates"):
        parse_stream(p)


def test_reference_match_fails_on_zero_compared_records():
    from simval.ontos import check_reference_match

    summary = {"mismatch_count": 0, "records_compared": 0, "ticks_verified": 0, "mismatches": []}
    assert not check_reference_match(summary).passed
    summary = {"mismatch_count": 0, "records_compared": 5, "ticks_verified": 1, "mismatches": []}
    assert check_reference_match(summary).passed


# --- ONT-001: life-mode run contract from ontos.json metadata ---


def _life_run(tmp_path, meta):
    import shutil

    run = tmp_path / "ontos_run"
    if run.exists():
        shutil.rmtree(run)
    shutil.copytree(R_PENTOMINO, run)
    merged = json.loads((run / "ontos.json").read_text())
    merged.update(meta)
    if "events" in meta:
        merged["events"] = meta["events"]
    (run / "ontos.json").write_text(json.dumps(merged))
    return run


def test_life_contract_verifies(tmp_path):
    run = _life_run(
        tmp_path,
        {"mode": "life", "ticks": 64, "events": [["demote", 1, 0], ["demote", 0, 1]]},
    )
    ctx = OntosEngine().load_context(run, selection="default")
    results = run_checks(ctx)
    names = [r.name for r in results]
    assert "ontos_run_contract" not in names or all(
        r.passed for r in results if r.name == "ontos_run_contract"
    )


def test_life_contract_wrong_ticks_fails(tmp_path):
    run = _life_run(tmp_path, {"mode": "life", "ticks": 32, "events": []})
    ctx = OntosEngine().load_context(run, selection="default")
    results = run_checks(ctx)
    contract = next((r for r in results if r.name == "ontos_run_contract"), None)
    assert contract is not None and not contract.passed


def test_life_contract_event_mismatch_fails(tmp_path):
    run = _life_run(tmp_path, {"mode": "life", "ticks": 64, "events": [["promote", 0, 0]]})
    ctx = OntosEngine().load_context(run, selection="default")
    results = run_checks(ctx)
    contract = next((r for r in results if r.name == "ontos_run_contract"), None)
    assert contract is not None and not contract.passed


# --- ONT-009: untimed life events must initialize (before tick 1) ---


def test_life_event_at_later_boundary_fails_contract(tmp_path):
    # Move the second scheduled demote from the initialization boundary to
    # the boundary before tick 3: identity and count still match, placement
    # must fail the contract.
    data = bytearray((R_PENTOMINO / "ontos.stream").read_bytes())
    sizes = {1: 9, 2: 9, 3: 17, 4: 10, 5: 34}
    off = 16
    level_offsets = []
    frame_index = 0
    frame_starts = []
    while off < len(data):
        tag = data[off]
        if tag == 4:
            level_offsets.append(off)
        if tag == 1:
            frame_starts.append(off)
            frame_index += 1
        off += sizes[tag]
    assert len(level_offsets) == 2 and len(frame_starts) >= 3
    second_level = bytes(data[level_offsets[1] : level_offsets[1] + 10])
    after_removal = bytes(data[: level_offsets[1]]) + bytes(data[level_offsets[1] + 10 :])
    # The boundary before tick 3 shifted left by one 10-byte record.
    insert_at = frame_starts[2] - 10
    mutated = after_removal[:insert_at] + second_level + after_removal[insert_at:]
    p = tmp_path / "moved.stream"
    p.write_bytes(mutated)

    from simval.ontos import life_contract_problems, parse_stream, verify_stream

    _, records = parse_stream(p)
    summary = verify_stream(p, 42)
    meta = json.loads((R_PENTOMINO / "ontos.json").read_text())
    problems = life_contract_problems(meta, summary, records)
    assert any("must precede tick 1" in p_ for p_ in problems), problems


def test_life_event_placement_clean_fixture_has_no_problems():
    from simval.ontos import life_contract_problems, parse_stream, verify_stream

    for fixture in ("r_pentomino", "promote_roundtrip", "all_fine"):
        run = EXAMPLES / fixture
        _, records = parse_stream(run / "ontos.stream")
        summary = verify_stream(run / "ontos.stream", 42 if fixture != "all_fine" else 7)
        meta = json.loads((run / "ontos.json").read_text())
        assert life_contract_problems(meta, summary, records) == [], fixture


def test_life_event_at_later_boundary_fails_engine_checks(tmp_path):
    import shutil

    run = tmp_path / "ontos_run"
    shutil.copytree(R_PENTOMINO, run)
    data = bytearray((run / "ontos.stream").read_bytes())
    sizes = {1: 9, 2: 9, 3: 17, 4: 10, 5: 34}
    off = 16
    level_offsets = []
    frame_starts = []
    while off < len(data):
        tag = data[off]
        if tag == 4:
            level_offsets.append(off)
        if tag == 1:
            frame_starts.append(off)
        off += sizes[tag]
    second_level = bytes(data[level_offsets[1] : level_offsets[1] + 10])
    after_removal = bytes(data[: level_offsets[1]]) + bytes(data[level_offsets[1] + 10 :])
    insert_at = frame_starts[2] - 10
    (run / "ontos.stream").write_bytes(
        after_removal[:insert_at] + second_level + after_removal[insert_at:]
    )
    ctx = OntosEngine().load_context(run, selection="default")
    results = run_checks(ctx)
    contract = next((r for r in results if r.name == "ontos_run_contract"), None)
    assert contract is not None and not contract.passed


# --- ONT-020: life initialization events are compared as an ORDERED sequence ---


def test_life_reordered_initialization_fails_contract_on_ordering(tmp_path):
    # A self-consistent promote-then-demote tick-1 stream: replay agrees
    # with itself (physics is internally consistent), the event multiset
    # matches the requested pair — only the ORDER differs from the
    # canonical demote-then-promote request. The contract must fail it.
    from simval.ontos import (
        ReferenceWorld,
        life_contract_problems,
        parse_stream,
        verify_stream,
    )

    def build(path, schedule):
        world = ReferenceWorld(seed=42)
        world.seed_r_pentomino()
        _emit_stream(path, world, schedule, 16)

    canonical = tmp_path / "canonical.stream"
    build(canonical, [(1, 1, 0), (1, 1, 1)])  # demote then promote
    swapped = tmp_path / "swapped.stream"
    build(swapped, [(1, 1, 1), (1, 1, 0)])  # promote then demote

    meta = {"mode": "life", "ticks": 16, "events": [["demote", 1, 1], ["promote", 1, 1]]}
    for path in (canonical, swapped):
        summary = verify_stream(path, 42)
        assert summary["mismatch_count"] == 0, (path, summary["mismatches"])

    assert life_contract_problems(meta, verify_stream(canonical, 42), parse_stream(canonical)[1]) == []
    problems = life_contract_problems(meta, verify_stream(swapped, 42), parse_stream(swapped)[1])
    assert any("event order mismatch" in p for p in problems), problems


def test_life_order_contract_fails_engine_checks(tmp_path):
    # Through the engine adapter: the reordered stream is replay-clean but
    # must fail the ontos_run_contract diagnostic.
    import shutil

    run = tmp_path / "ontos_run"
    shutil.copytree(EXAMPLES / "r_pentomino", run)
    world = ReferenceWorld(seed=42)
    world.seed_r_pentomino()
    _emit_stream(run / "ontos.stream", world, [(1, 1, 1), (1, 1, 0)], 64)
    meta = {
        "mode": "life",
        "seed": 42,
        "ticks": 64,
        "events": [["demote", 1, 1], ["promote", 1, 1]],
    }
    (run / "ontos.json").write_text(json.dumps(meta))

    ctx = OntosEngine().load_context(run, selection="default")
    results = run_checks(ctx)
    contract = next((r for r in results if r.name == "ontos_run_contract"), None)
    assert contract is not None and not contract.passed
    assert any("event order mismatch" in p for p in contract.detail["problems"])
