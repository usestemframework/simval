"""Ontos gravity adapter tests: independent spec reimplementation vs recorded streams."""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from simval.context import select_engine
from simval.ontos_gravity import (
    MONOPOLE_BASE,
    UNMANAGED,
    WALL_BASE,
    GravityWorld,
    SplitMix64,
    check_bounded_drift,
    check_collapse_energy,
    check_contact_resolution,
    check_multipole_match,
    check_radial_shape,
    check_reconstruction_error,
    check_reference_match_gravity,
    check_shell_shape,
    parse_stream_v2,
    region_at,
    verify_stream_gravity,
)
from simval.pipeline import run_checks

EXAMPLES = Path(__file__).parent.parent / "examples" / "ontos_gravity"

CASES = [
    ("all_fine", 42, 100),
    ("window", 42, 120),
    ("refit", 7, 120),
    ("multi", 3, 150),
    ("observer", 5, 300),
]


@pytest.mark.parametrize("name,seed,ticks", CASES)
def test_reference_matches_stream(name, seed, ticks):
    summary = verify_stream_gravity(EXAMPLES / name / "ontos.stream", seed)
    assert summary["mismatch_count"] == 0
    assert summary["ticks_verified"] == ticks
    assert check_reference_match_gravity(summary).passed
    assert check_bounded_drift(summary).passed


def test_all_fine_conserves_momentum_exactly():
    from simval.ontos_gravity import GravityWorld

    world = GravityWorld(5, 6)
    px0, py0 = world.px, world.py
    for _ in range(60):
        world.step()
    assert world.px == px0
    assert world.py == py0


def test_splitmix64_deterministic_and_spread():
    a = SplitMix64(123)
    b = SplitMix64(123)
    vals = [a.next() for _ in range(64)]
    assert vals == [b.next() for _ in range(64)]
    assert len(set(vals)) == 64


def test_corrupted_body_state_fails(tmp_path):
    data = bytearray((EXAMPLES / "all_fine" / "ontos.stream").read_bytes())
    _, records = __import__("simval.ontos_gravity", fromlist=["parse_stream_v2"]).parse_stream_v2(
        EXAMPLES / "all_fine" / "ontos.stream"
    )
    offset = 20
    for record in records:
        size = {
            "tick": 9,
            "snapshot": 9,
            "flip": 17,
            "level": 10,
            "state": 34,
            "body": 55,
            "totals": 57,
            "multipole": 57,
        }[record[0]]
        if record[0] == "body":
            data[offset + size - 9] ^= 0x01
            break
        offset += size
    corrupt = tmp_path / "ontos.stream"
    corrupt.write_bytes(bytes(data))
    summary = verify_stream_gravity(corrupt, 42)
    assert summary["mismatch_count"] > 0
    assert not check_reference_match_gravity(summary).passed


def test_engine_detect_and_diagnose_gravity(tmp_path):
    import shutil

    run = tmp_path / "ontos_run"
    shutil.copytree(EXAMPLES / "window", run)
    engine = select_engine(run)
    assert engine.name == "ontos"
    ctx = engine.load_context(run, selection="default")
    results = run_checks(ctx)
    names = {r.name for r in results}
    assert "ontos_reference_match" in names
    assert "ontos_window_drift" in names
    assert all(r.passed for r in results if r.name.startswith("ontos_"))


def test_wrong_seed_fails(tmp_path):
    import shutil

    run = tmp_path / "ontos_run"
    shutil.copytree(EXAMPLES / "window", run)
    summary = verify_stream_gravity(run / "ontos.stream", 43)
    assert summary["mismatch_count"] > 0


def test_zoom_policy_matches_and_detects_tampering(tmp_path):
    import shutil

    from simval.ontos_gravity import check_zoom_policy, parse_stream_v2

    _, records = parse_stream_v2(EXAMPLES / "observer" / "ontos.stream")
    result = check_zoom_policy(records, 5, 777)
    assert result.passed, result.detail
    assert result.detail["policy_events"] >= 3

    run = tmp_path / "ontos_run"
    shutil.copytree(EXAMPLES / "observer", run)
    data = bytearray((run / "ontos.stream").read_bytes())
    sizes = {1: 9, 2: 9, 3: 17, 4: 10, 5: 34, 6: 55, 7: 57}
    off = 20
    flipped = False
    while off < len(data):
        tag = data[off]
        if tag == 4 and not flipped:
            data[off + 9] ^= 0x01
            flipped = True
            break
        off += sizes[tag]
    (run / "ontos.stream").write_bytes(bytes(data))
    _, records = parse_stream_v2(run / "ontos.stream")
    result = check_zoom_policy(records, 5, 777)
    assert not result.passed


def test_observer_engine_diagnose(tmp_path):
    import shutil

    from simval.context import select_engine as sel
    from simval.pipeline import run_checks as rc

    run = tmp_path / "ontos_run"
    shutil.copytree(EXAMPLES / "observer", run)
    engine = sel(run)
    ctx = engine.load_context(run, selection="default")
    results = rc(ctx)
    names = {r.name for r in results}
    assert "ontos_zoom_policy" in names
    zoom = next(r for r in results if r.name == "ontos_zoom_policy")
    assert zoom.passed


def test_rebound_anchor_agrees():
    rebound = pytest.importorskip("rebound")
    from simval.ontos_gravity import check_rebound_anchor, parse_stream_v2

    _, records = parse_stream_v2(EXAMPLES / "all_fine" / "ontos.stream")
    result = check_rebound_anchor(records, 42, 8)
    assert result.passed, result.detail
    assert result.value < 1e-4


# --- ONT-011: per-region window deadline, independent of member Fits ---


def test_empty_demote_refits_at_deadline():
    # A demotion with no in-box bodies still owes its t0 + WINDOW
    # re-fit. The deadline must be armed so the empty re-fit path
    # returns the region to Fine when it fires; before the fix the
    # region stayed Coarse forever (no member Fit could carry the
    # obligation).
    w = GravityWorld(1, 1)
    b = w.bodies[0]
    assert not (b["x"] < 64.0 and b["y"] < 64.0), "seed 1 keeps body 0 outside region 0"
    w.schedule(1, 0, 0)
    w.step()
    assert w.tick == 1
    assert w.region_coarse[0]
    assert w.region_window_deadline[0] == 1 + 32
    while w.tick < 32:
        w.step()
    assert w.region_coarse[0], "before the deadline"
    w.step()
    assert w.tick == 33
    assert not w.region_coarse[0], "empty re-fit at deadline"
    assert w.region_window_deadline[0] is None


def test_donor_region_refits_after_foreign_absorption():
    # Foreign-window absorption (section 26 collapse) removes the donor
    # region's only member Fit; the donor's own deadline still fires and
    # the now-memberless donor returns to Fine (fit-scan detection
    # stalled forever before ONT-011).
    w = GravityWorld(11, 8)
    w.bodies[0]["x"] = 63.995
    w.bodies[0]["y"] = 32.0
    w.bodies[0]["vx"] = 0.5
    w.bodies[0]["vy"] = 0.0
    for b in w.bodies[1:]:
        b["x"] = 100.0
        b["y"] = 100.0
    w.schedule(1, 0, 0)
    w.schedule(30, 1, 2)
    while w.tick < 32:
        w.step()
    assert w.region_collapsed[1] is not None
    assert w.body_collapsed[0] is not None, "body 0 absorbed by region 1"
    assert w.coarse[0] is None, "absorption dropped the body fit"
    assert w.region_coarse[0]
    w.step()
    assert w.tick == 1 + 32, "donor deadline was t0 + WINDOW"
    assert not w.region_coarse[0], "donor refits at its deadline with no surviving member fit"
    assert w.region_window_deadline[0] is None


# --- ONT-012: re-demotion materializes and clears the region's old fits ---


def test_re_demote_discards_stale_fits_at_boundary():
    # Re-demoting an already-Coarse region must materialize and discard
    # every old fit of the region before building the new window. An
    # out-of-box member would otherwise keep a fit whose validity ends
    # at its own t0 + WINDOW while the region deadline advances to the
    # new t0 + WINDOW, and the polynomial would be evaluated past its
    # window end.
    w = GravityWorld(11, 8)
    w.bodies[0]["x"] = 127.995
    w.bodies[0]["y"] = 96.0
    w.bodies[0]["vx"] = 0.5
    w.bodies[0]["vy"] = 0.0
    for b in w.bodies[1:]:
        b["x"] = 100.0
        b["y"] = 100.0
    w.schedule(1, 3, 0)
    w.schedule(30, 3, 0)
    while w.tick < 29:
        w.step()
    assert w.coarse[0] is not None, "body 0 carried a fit into tick 30"
    w.step()
    assert w.coarse[0] is None, "out-of-box member thawed at the re-demotion boundary"
    assert w.region_window_deadline[3] == 30 + 32, "one coherent new window"
    while w.tick < 70:
        w.step()
        for i in range(len(w.bodies)):
            if w.coarse[i] is not None:
                assert w.tick <= w.coarse[i]["t0"] + 32, (
                    f"body {i} fit t0 {w.coarse[i]['t0']} evaluated at {w.tick} "
                    "past its window end"
                )


def test_re_demote_excludes_collapsed_bodies_from_membership():
    # Corrected ontos demote membership excludes collapsed bodies (the
    # section 14 rule): a body collapsed into a foreign region whose
    # monopole sits inside this box must not receive a window fit.
    w = GravityWorld(11, 8)
    for b in w.bodies[1:]:
        b["x"] = 100.0
        b["y"] = 100.0
    w.schedule(1, 3, 2)
    w.step()
    assert w.region_collapsed[3] is not None
    assert 0 not in w.region_collapsed[3]["members"], "seed 11 body 0 outside region 3"
    w.bodies[0]["x"] = w.region_collapsed[3]["com_x"]
    w.bodies[0]["y"] = w.region_collapsed[3]["com_y"]
    w.schedule(2, 3, 0)
    w.step()
    assert w.coarse[0] is None, "collapsed body never joins a window"


# --- ONT-013: suppression activates at the first recorded contact ---


def test_first_impulse_records_after_silent_overlap():
    # Seed 416 reproducer (--mode gravity --contacts --seed 416 --bodies
    # 2): the pair first overlaps while receding, joining the touching
    # set without a record, and the first trajectory-changing impulse at
    # tick 13 must still emit the stream's first Contact record —
    # suppression activates only from the first recorded contact onward.
    # Before the fix the tick-13 impulse applied silently and the run
    # diverged from gravity-only with zero records.
    w = GravityWorld(416, 2)
    w.contacts = True
    while w.tick < 12:
        w.step()
        assert not w.last_contacts, f"no impulse before tick 13 (tick {w.tick})"
        w.last_contacts.clear()
    w.step()
    assert w.tick == 13
    assert len(w.last_contacts) == 1, "tick 13 impulse produces the first Contact record"
    c = w.last_contacts[0]
    assert (c["a"], c["b"], c["tick"]) == (0, 1, 13)
    assert c["jn"] > 0.0
    assert (0, 1) in w.touching
    assert w.contact_armed, "suppression live from the next pass"
    w.last_contacts.clear()
    # The pair is now in the touching set: a same-pass-ordered re-approach
    # at the next tick stays silent, matching the post-activation
    # semantics a verifier replays.
    w.step()
    assert w.touching == {(0, 1)}, "pair remains overlapping"


# --- ONT-014: touching invalidated at every fine-membership transition ---


def test_same_tick_demote_promote_invalidates_touching():
    # Invalidation follows applied transitions, not final membership. A
    # body demoted and re-promoted within one boundary keeps no stale
    # touching keys, so the still-overlapping pair begins a fresh contact
    # (record + impulse) even though its Fine status at the pass equals
    # the status before the boundary.
    w = GravityWorld(11, 2)
    w.bodies[0]["mass"] = 2.0
    w.bodies[0]["x"] = 62.0
    w.bodies[0]["y"] = 32.0
    w.bodies[0]["vx"] = 0.0
    w.bodies[0]["vy"] = 0.0
    w.bodies[1]["mass"] = 2.0
    w.bodies[1]["x"] = 66.0
    w.bodies[1]["y"] = 32.0
    w.bodies[1]["vx"] = 0.0
    w.bodies[1]["vy"] = 0.0
    w.contacts = True
    w.contact_params = True
    w.schedule(2, 1, 0)
    w.schedule(2, 1, 1)
    w.step()
    assert len(w.last_contacts) == 1, "contact begins at tick 1"
    w.last_contacts.clear()
    w.step()
    assert len(w.last_contacts) == 1, "demote+promote in one boundary still invalidates the pair"
    c = w.last_contacts[0]
    assert (c["a"], c["b"]) == (0, 1)
    assert c["jn"] > 0.0, "impulse fires and records"


def test_same_tick_expand_recollapse_invalidates_monopole_touching():
    # A region that expands and re-collapses within one boundary must
    # drop its monopole pseudo-id touching keys, so a fine body still
    # overlapping the reborn monopole begins a fresh contact. Final
    # region mode (Collapsed) equals the pre-boundary mode, so a
    # final-vs-initial derivation kept the stale key and suppressed the
    # record forever after.
    w = GravityWorld(11, 5)
    for slot, i in enumerate(range(1, 5)):
        w.bodies[i]["x"] = 66.0 + float(slot % 2)
        w.bodies[i]["y"] = 96.0 + float(slot // 2)
        w.bodies[i]["vx"] = 0.0
        w.bodies[i]["vy"] = 0.0
    w.bodies[0]["x"] = 63.5
    w.bodies[0]["y"] = 96.0
    w.bodies[0]["vx"] = 0.5
    w.bodies[0]["vy"] = 0.0
    w.contacts = True
    w.contact_params = True
    w.schedule(1, 3, 2)
    w.schedule(3, 3, 1)
    w.schedule(3, 3, 2)
    w.step()

    def monopole_records(world):
        return [c for c in world.last_contacts if c["b"] == MONOPOLE_BASE + 3]

    assert len(monopole_records(w)) == 1, "monopole contact begins at tick 1"
    w.last_contacts.clear()
    w.step()
    assert len(monopole_records(w)) == 0, "unchanged boundary stays silent"
    w.last_contacts.clear()
    w.step()
    recs = monopole_records(w)
    assert len(recs) == 1, "expand+recollapse in one boundary invalidates the pseudo key"
    assert recs[0]["jn"] > 0.0


# --- ONT-015: wall overlap holds the touching key without an impulse ---


def test_wall_touching_holds_receding_penetration_without_record():
    # Three phases: (1) penetrating + inbound emits exactly one Contact;
    # (2) penetrating + outbound keeps the wall key in the touching set,
    # no record; (3) inbound again before separating applies the impulse
    # with NO second record — the pair never left the touching set, so
    # re-approach is not a contact beginning.
    w = GravityWorld(3, 1)
    w.contacts = True
    w.contact_params = True
    w.walls = True
    w.restitution = 0.5
    w.friction = 0.0
    w.bodies[0]["x"] = -1.0
    w.bodies[0]["y"] = 50.0
    w.bodies[0]["vx"] = -0.5
    w.bodies[0]["vy"] = 0.0
    w.step()
    assert len(w.last_contacts) == 1, "phase 1: one contact record"
    assert w.last_contacts[0]["b"] == WALL_BASE
    assert w.last_contacts[0]["jn"] > 0.0
    assert w.bodies[0]["vx"] > 0.0, "bounced outbound"
    assert (0, WALL_BASE) in w.touching
    w.last_contacts.clear()
    w.step()
    assert not w.last_contacts, "phase 2: receding penetration emits nothing"
    assert (0, WALL_BASE) in w.touching, "wall key stays in touching while overlapping"
    w.last_contacts.clear()
    w.bodies[0]["vx"] = -0.5
    w.step()
    assert not w.last_contacts, "phase 3: same contact, no second record"
    assert w.bodies[0]["vx"] > 0.0, "the impulse still applies on re-approach"
    assert (0, WALL_BASE) in w.touching


# --- Section 19: collapse and reconstruction ---


def _emit_gravity_stream(
    path,
    seed,
    count,
    schedule,
    ticks,
    contacts=False,
    radial=False,
    shells=False,
    params=None,
    profile=None,
):
    """Serialize a reference gravity run per the spec section 16 emission contract.

    schedule: list of (tick, region_index, level) with level 0 demote,
    1 promote/expand, 2 collapse. Collapse events additionally emit the
    section 19 RegionCollapsed record (and RegionMultipole /
    RegionRadial / RegionShells when the modes are on) before that
    tick's TickHeader.
    contacts: spec section 21 mode — emits Contact records (tag 10) in
    generation order before that tick's TickHeader.
    radial: spec section 23 mode — collapses emit RegionRadial (tag 11).
    shells: spec section 25 mode — collapses emit RegionShells (tag 13).
    params: optional (restitution, friction, walls) tuple — emits
    ContactParams (tag 12) before the first TickHeader.
    profile: optional test-only corpus initial conditions (wallshot /
    coarsehit; mirror of ontos corpus_initial_conditions).
    """
    world = GravityWorld(seed, count, profile)
    world.contacts = contacts
    world.radial_enabled = radial
    world.shells_enabled = shells
    if params is not None:
        world.contact_params = True
        world.restitution, world.friction, walls = params
        world.walls = bool(walls)
    out = bytearray(b"ONTO")
    out += struct.pack("<IIII", 2, 128, 128, count)
    if params is not None:
        out += b"\x0c" + struct.pack("<ddB", params[0], params[1], 1 if params[2] else 0)
    events = {}
    for t, region, level in schedule:
        world.schedule(t, region, level)
        events.setdefault(t, []).append((region, level))
    for _ in range(ticks):
        entering = world.tick + 1
        for region, level in events.get(entering, []):
            out += b"\x04" + struct.pack("<IIB", region % 2, region // 2, level)
        world.step()
        for c in world.last_collapses:
            out += b"\x08" + struct.pack(
                "<QIIQdddddd",
                c["tick"],
                c["region"] % 2,
                c["region"] // 2,
                c["count"],
                c["mass"],
                c["com_x"],
                c["com_y"],
                c["px"],
                c["py"],
                c["energy"],
            )
            if c["multipole"]:
                out += b"\x09" + struct.pack(
                    "<QIIddddd",
                    c["tick"],
                    c["region"] % 2,
                    c["region"] // 2,
                    c["mx"],
                    c["my"],
                    c["qxx"],
                    c["qxy"],
                    c["qyy"],
                )
                if c["radial"]:
                    out += b"\x0b" + struct.pack(
                        "<QIId",
                        c["tick"],
                        c["region"] % 2,
                        c["region"] // 2,
                        c["binding"],
                    )
                if c["shells"]:
                    out += b"\x0d" + struct.pack(
                        "<QIIddddd",
                        c["tick"],
                        c["region"] % 2,
                        c["region"] // 2,
                        c["binding"],
                        c["shell_bindings"][0],
                        c["shell_bindings"][1],
                        c["shell_bindings"][2],
                        c["shell_bindings"][3],
                    )
        world.last_collapses.clear()
        for c in world.last_contacts:
            out += b"\x0a" + struct.pack(
                "<QIIddd",
                c["tick"],
                c["a"],
                c["b"],
                c["jn"],
                c["cx"],
                c["cy"],
            )
        world.last_contacts.clear()
        out += b"\x01" + struct.pack("<Q", world.tick)
        out += b"\x02" + struct.pack("<Q", count)
        fine, coarse_n, mass, px, py, energy = world.totals()
        out += b"\x07" + struct.pack("<QQQdddd", world.tick, fine, coarse_n, mass, px, py, energy)
        for rx, ry in ((0, 0), (1, 0), (0, 1), (1, 1)):
            level, pop, rhash = world.region_hash(ry * 2 + rx)
            out += b"\x05" + struct.pack("<QIIBQQ", world.tick, rx, ry, level, pop, rhash)
        for i in range(count):
            b, region, level = world.emitted_state(i)
            out += b"\x06" + struct.pack(
                "<QIBBddddd", world.tick, i, region, level, b["x"], b["y"], b["vx"], b["vy"], b["mass"]
            )
    Path(path).write_bytes(bytes(out))
    return world


def _region_occupancy(seed, count, at_tick):
    world = GravityWorld(seed, count)
    for _ in range(at_tick):
        world.step()
    counts = [0, 0, 0, 0]
    for b in world.bodies:
        r = region_at(b["x"], b["y"])
        if r != UNMANAGED:
            counts[r] += 1
    return sorted(range(4), key=lambda r: -counts[r]), counts


@pytest.mark.parametrize(
    "name,seed,count,schedule,ticks",
    [
        ("all_fine", 42, 8, [], 100),
        ("window", 42, 8, [(20, 0, 0), (80, 0, 1)], 120),
        ("refit", 7, 12, [(5, 3, 0)], 120),
        ("multi", 3, 10, [(10, 2, 0), (40, 2, 0), (100, 2, 1)], 150),
    ],
)
def test_emitter_reproduces_recorded_streams(tmp_path, name, seed, count, schedule, ticks):
    out = tmp_path / f"{name}.stream"
    _emit_gravity_stream(out, seed, count, schedule, ticks)
    assert out.read_bytes() == (EXAMPLES / name / "ontos.stream").read_bytes()


@pytest.mark.parametrize("seed,count", [(42, 8), (7, 12), (3, 10), (5, 16)])
def test_collapse_expand_roundtrip_self_consistent(tmp_path, seed, count):
    order, counts = _region_occupancy(seed, count, 5)
    region = order[0]
    assert counts[region] >= 2
    stream = tmp_path / f"collapse_{seed}.stream"
    _emit_gravity_stream(stream, seed, count, [(6, region, 2), (36, region, 1)], 80)
    summary = verify_stream_gravity(stream, seed)
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["collapse_events"] == 1
    assert summary["expand_events"] == 1
    assert summary["ticks_verified"] == 80
    assert check_reference_match_gravity(summary).passed
    # Measured across these seeds: post-expansion deviation 10.3-22.2
    # (tolerance 64.0), collapse-energy relative delta 0.46-3.49 (tolerance 8.0),
    # tracked-body window drift pos<=5.3e-5 mom<=4.4e-3 energy<=4.9e-4.
    assert check_reconstruction_error(summary).passed
    assert 0.0 < summary["post_expansion_deviation"] < 64.0
    assert check_collapse_energy(summary).passed
    assert check_collapse_energy(summary).value < 8.0
    assert check_bounded_drift(summary).passed


def test_multi_region_collapse_with_window(tmp_path):
    # seed 42, 8 bodies: occupancy at tick 5 is [3, 0, 3, 2] -> order [0, 2, 3, 1]
    order, _ = _region_occupancy(42, 8, 5)
    schedule = [
        (6, order[0], 2),
        (10, order[1], 2),
        (14, order[2], 0),
        (40, order[0], 1),
    ]
    stream = tmp_path / "multi_collapse.stream"
    _emit_gravity_stream(stream, 42, 8, schedule, 70)
    summary = verify_stream_gravity(stream, 42)
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["collapse_events"] == 2
    assert summary["expand_events"] == 1
    assert check_reconstruction_error(summary).passed
    assert check_collapse_energy(summary).passed
    # 6 of 8 bodies collapsed for most of the run: the two tracked bodies see
    # monopoles instead of real bodies. Measured tracked energy drift 1.2e-3
    # (above the 1e-3 single-window default, still small); bounded here.
    assert summary["energy_drift"] < 5e-3
    assert summary["momentum_drift"] < 5e-2


def test_collapsed_bodies_static_level2_real_mass():
    order, _ = _region_occupancy(42, 8, 5)
    region = order[0]
    world = GravityWorld(42, 8)
    world.schedule(6, region, 2)
    for _ in range(6):
        world.step()
    states = [world.emitted_state(i) for i in range(8)]
    collapsed = [(b, r) for b, r, lv in states if lv == 2]
    assert len(collapsed) >= 2
    for _ in range(12):
        world.step()
    after = [world.emitted_state(i) for i in range(8)]
    for (b1, r1, lv1), (b2, r2, lv2) in zip(states, after):
        if lv1 == 2:
            assert lv2 == 2 and r2 == r1 == region
            assert (b1["x"], b1["y"], b1["vx"], b1["vy"], b1["mass"]) == (
                b2["x"],
                b2["y"],
                b2["vx"],
                b2["vy"],
                b2["mass"],
            )
    ics = {b["id"]: b["mass"] for b in GravityWorld(42, 8).bodies}
    for b, _ in collapsed:
        assert b["mass"] == ics[b["id"]]


def test_expansion_residual_momentum_exact():
    order, _ = _region_occupancy(42, 8, 5)
    region = order[0]
    world = GravityWorld(42, 8)
    world.schedule(6, region, 2)
    for _ in range(6):
        world.step()
    world.schedule(36, region, 1)
    for _ in range(30):
        world.step()
    exp = world.last_expansion
    assert exp["tick"] == 36
    bodies = exp["bodies"]
    assert [b["id"] for b in bodies] == sorted(b["id"] for b in bodies)
    # last body velocity is the exact spec residual, bit for bit
    last = bodies[-1]
    sum_mvx = 0.0
    sum_mvy = 0.0
    for b in bodies[:-1]:
        sum_mvx += b["mass"] * b["vx"]
        sum_mvy += b["mass"] * b["vy"]
    assert struct.pack("<d", last["vx"]) == struct.pack("<d", (exp["target_px"] - sum_mvx) / last["mass"])
    assert struct.pack("<d", last["vy"]) == struct.pack("<d", (exp["target_py"] - sum_mvy) / last["mass"])
    # total reconstructed momentum closes on the collapse totals to rounding
    # (measured: 0.0 or ~1.4e-16 absolute at momentum scale ~0.7)
    px = 0.0
    py = 0.0
    for b in bodies:
        px += b["mass"] * b["vx"]
        py += b["mass"] * b["vy"]
    scale = max(abs(exp["target_px"]), abs(exp["target_py"]), 1e-30)
    assert max(abs(px - exp["target_px"]), abs(py - exp["target_py"])) / scale < 1e-12


def test_collapse_monopole_ledger_drift_bounded():
    # Measured over 30 collapsed ticks across seeds 42/7/3/5 (N=8..16):
    # relative ledger drift 1.0e-3 - 2.1e-3; bound at 1e-2 for margin.
    order, _ = _region_occupancy(42, 8, 5)
    world = GravityWorld(42, 8)
    world.schedule(6, order[0], 2)
    for _ in range(6):
        world.step()
    px0, py0 = world.px, world.py
    scale = max(abs(px0), abs(py0), 1e-30)
    for _ in range(30):
        world.step()
    drift = max(abs(world.px - px0), abs(world.py - py0)) / scale
    assert 0.0 < drift < 1e-2


def test_tampered_collapse_record_fails(tmp_path):
    order, _ = _region_occupancy(42, 8, 5)
    stream = tmp_path / "ontos.stream"
    _emit_gravity_stream(stream, 42, 8, [(6, order[0], 2), (36, order[0], 1)], 40)
    data = bytearray(stream.read_bytes())
    sizes = {1: 9, 2: 9, 3: 17, 4: 10, 5: 34, 6: 55, 7: 57, 8: 73, 9: 57}
    off = 20
    flipped = False
    while off < len(data):
        tag = data[off]
        if tag == 8 and not flipped:
            data[off + 33] ^= 0x01  # low byte of com_x
            flipped = True
            break
        off += sizes[tag]
    assert flipped
    corrupt = tmp_path / "corrupt.stream"
    corrupt.write_bytes(bytes(data))
    summary = verify_stream_gravity(corrupt, 42)
    assert summary["mismatch_count"] > 0
    assert not check_reference_match_gravity(summary).passed


def test_wrong_seed_fails_collapse_stream(tmp_path):
    order, _ = _region_occupancy(42, 8, 5)
    stream = tmp_path / "ontos.stream"
    _emit_gravity_stream(stream, 42, 8, [(6, order[0], 2), (36, order[0], 1)], 40)
    summary = verify_stream_gravity(stream, 43)
    assert summary["mismatch_count"] > 0
    assert not check_reference_match_gravity(summary).passed


def test_engine_diagnose_collapse_run(tmp_path):
    order, _ = _region_occupancy(42, 8, 5)
    run = tmp_path / "ontos_run"
    run.mkdir()
    _emit_gravity_stream(run / "ontos.stream", 42, 8, [(6, order[0], 2), (36, order[0], 1)], 80)
    (run / "ontos.json").write_text(json.dumps({"seed": 42}))
    engine = select_engine(run)
    assert engine.name == "ontos"
    ctx = engine.load_context(run, selection="default")
    results = run_checks(ctx)
    names = {r.name for r in results}
    assert "ontos_reconstruction_error" in names
    assert "ontos_collapse_energy" in names
    assert all(r.passed for r in results if r.name.startswith("ontos_"))


# --- Section 20: multipole reconstruction ---


def test_multipole_roundtrip_self_consistent(tmp_path):
    # The default world is multipole-on (section 20): collapses emit the
    # RegionMultipole record and expansions synthesize under dipole +
    # quadrupole constraints.
    order, counts = _region_occupancy(42, 8, 5)
    region = order[0]
    assert counts[region] >= 3
    stream = tmp_path / "multipole.stream"
    _emit_gravity_stream(stream, 42, 8, [(6, region, 2), (36, region, 1)], 80)
    summary = verify_stream_gravity(stream, 42)
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["collapse_events"] == 1
    assert summary["multipole_events"] == 1
    assert summary["expand_events"] == 1
    result = check_multipole_match(summary)
    assert result.passed, result.detail
    # Measured: dipole closes at 0.0, quadrupole matches ~1e-15.
    assert result.detail["worst_dipole_relative"] <= 1e-12
    assert result.detail["worst_quadrupole_relative"] <= 1e-9
    # Post-expansion deviation stays region-scale (same tier as section 19).
    assert check_reconstruction_error(summary).passed
    assert 0.0 < summary["post_expansion_deviation"] < 64.0


def test_multipole_recorded_stream_verifies():
    summary = verify_stream_gravity(EXAMPLES / "multipole" / "ontos.stream", 17)
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["ticks_verified"] == 240
    assert summary["collapse_events"] == 3
    assert summary["multipole_events"] == 3
    assert summary["expand_events"] == 3
    assert check_multipole_match(summary).passed


def test_legacy_section19_examples_still_verify():
    # Streams whose collapses carry no RegionMultipole record reconstruct
    # per section 19; the committed pre-section-20 example must verify
    # bit-exactly. (collapse_observer was regenerated from corrected
    # ontos for ONT-010 and now carries the section 20 record.)
    for name, seed in (("collapse", 11),):
        summary = verify_stream_gravity(EXAMPLES / name / "ontos.stream", seed)
        assert summary["mismatch_count"] == 0
        assert summary["multipole_events"] == 0
        assert check_multipole_match(summary).passed


def test_multipole_dipole_residual_exact():
    order, _ = _region_occupancy(42, 8, 5)
    region = order[0]
    world = GravityWorld(42, 8)
    world.schedule(6, region, 2)
    world.schedule(36, region, 1)
    for _ in range(36):
        world.step()
    exp = world.last_expansion
    assert exp is not None and exp["multipole"]
    sx = 0.0
    sy = 0.0
    for b in exp["bodies"]:
        sx += b["mass"] * b["x"]
        sy += b["mass"] * b["y"]
    scale = max(abs(exp["target_mx"]), abs(exp["target_my"]), 1e-30)
    assert max(abs(sx - exp["target_mx"]), abs(sy - exp["target_my"])) / scale < 1e-12
    # momentum residual unchanged from section 19
    px = 0.0
    py = 0.0
    for b in exp["bodies"]:
        px += b["mass"] * b["vx"]
        py += b["mass"] * b["vy"]
    scale = max(abs(exp["target_px"]), abs(exp["target_py"]), 1e-30)
    assert max(abs(px - exp["target_px"]), abs(py - exp["target_py"])) / scale < 1e-12


def test_multipole_quadrupole_matches_record():
    order, _ = _region_occupancy(7, 12, 5)
    region = order[0]
    world = GravityWorld(7, 12)
    world.schedule(6, region, 2)
    world.schedule(36, region, 1)
    for _ in range(36):
        world.step()
    exp = world.last_expansion
    assert exp is not None and exp["transformed"]
    qxx = 0.0
    qxy = 0.0
    qyy = 0.0
    for b in exp["bodies"]:
        dx = b["x"] - exp["target_com_x"]
        dy = b["y"] - exp["target_com_y"]
        qxx += b["mass"] * dx * dx
        qxy += b["mass"] * dx * dy
        qyy += b["mass"] * dy * dy
    scale = max(abs(exp["target_qxx"]), abs(exp["target_qyy"]), 1e-30)
    assert max(abs(qxx - exp["target_qxx"]), abs(qxy - exp["target_qxy"]), abs(qyy - exp["target_qyy"])) / scale < 1e-9


def test_multipole_single_body_pins_position():
    world = None
    for seed in range(100_000):
        w = GravityWorld(seed, 1)
        b = w.bodies[0]
        if 0.0 <= b["x"] < 64.0 and 0.0 <= b["y"] < 64.0:
            world = w
            break
    w = world
    w.schedule(1, 0, 2)
    w.schedule(10, 0, 1)
    for _ in range(10):
        w.step()
    exp = w.last_expansion
    assert exp is not None and exp["multipole"] and not exp["transformed"]
    b = exp["bodies"][0]
    scale = max(abs(exp["target_mx"]), abs(exp["target_my"]), 1e-30)
    assert max(abs(b["mass"] * b["x"] - exp["target_mx"]), abs(b["mass"] * b["y"] - exp["target_my"])) / scale < 1e-12


def test_tampered_multipole_record_fails(tmp_path):
    order, _ = _region_occupancy(42, 8, 5)
    stream = tmp_path / "ontos.stream"
    _emit_gravity_stream(stream, 42, 8, [(6, order[0], 2), (36, order[0], 1)], 40)
    data = bytearray(stream.read_bytes())
    sizes = {1: 9, 2: 9, 3: 17, 4: 10, 5: 34, 6: 55, 7: 57, 8: 73, 9: 57}
    off = 20
    flipped = False
    while off < len(data):
        tag = data[off]
        if tag == 9 and not flipped:
            data[off + 17] ^= 0x01  # low byte of mx
            flipped = True
            break
        off += sizes[tag]
    assert flipped
    corrupt = tmp_path / "corrupt.stream"
    corrupt.write_bytes(bytes(data))
    summary = verify_stream_gravity(corrupt, 42)
    assert summary["mismatch_count"] > 0
    assert not check_reference_match_gravity(summary).passed


def test_engine_diagnose_multipole_run(tmp_path):
    import shutil

    run = tmp_path / "ontos_run"
    shutil.copytree(EXAMPLES / "multipole", run)
    engine = select_engine(run)
    assert engine.name == "ontos"
    ctx = engine.load_context(run, selection="default")
    results = run_checks(ctx)
    names = {r.name for r in results}
    assert "ontos_multipole_match" in names
    assert "ontos_reconstruction_error" in names
    assert all(r.passed for r in results if r.name.startswith("ontos_"))


# --- Section 21: contact dynamics ---

CONTACT_EXAMPLES = Path(__file__).parent.parent / "examples" / "ontos_contact"


def test_contact_emitter_reproduces_recorded_stream(tmp_path):
    out = tmp_path / "contact.stream"
    _emit_gravity_stream(out, 11, 32, [], 400, contacts=True)
    assert out.read_bytes() == (CONTACT_EXAMPLES / "contact" / "ontos.stream").read_bytes()


def test_contact_recorded_stream_verifies():
    summary = verify_stream_gravity(CONTACT_EXAMPLES / "contact" / "ontos.stream", 11)
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["ticks_verified"] == 400
    assert summary["contact_events"] == 5
    assert summary["contact_run"] is True
    assert check_contact_resolution(summary).passed
    assert summary["contact_worst_vn_after"] < 1e-12
    assert summary["contact_min_jn"] > 0.0


def test_contact_collapse_composition_verifies():
    summary = verify_stream_gravity(
        CONTACT_EXAMPLES / "contact_collapse" / "ontos.stream", 11
    )
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["contact_events"] == 5
    assert summary["collapse_events"] == 1
    assert summary["expand_events"] == 1
    assert check_contact_resolution(summary).passed
    assert check_multipole_match(summary).passed


def test_contact_ledger_untouched_by_impulses():
    world = GravityWorld(11, 32)
    world.contacts = True
    px0, py0 = world.px, world.py
    impulses = 0
    for _ in range(400):
        world.step()
        impulses += len(world.last_contacts)
        world.last_contacts.clear()
    assert impulses >= 3
    assert world.px == px0
    assert world.py == py0


def test_contact_physical_momentum_drifts_only_by_rounding():
    world = GravityWorld(11, 32)
    world.contacts = True

    def sum_mv(w):
        px = 0.0
        py = 0.0
        for b in w.bodies:
            px += b["mass"] * b["vx"]
            py += b["mass"] * b["vy"]
        return px, py

    px0, py0 = sum_mv(world)
    for _ in range(200):
        world.step()
        world.last_contacts.clear()
    px1, py1 = sum_mv(world)
    scale = max(abs(px0), abs(py0), 1e-30)
    drift = max(abs(px1 - px0), abs(py1 - py0)) / scale
    assert drift < 1e-11, drift


def test_contact_no_records_without_overlaps(tmp_path):
    stream = tmp_path / "sparse.stream"
    _emit_gravity_stream(stream, 1, 8, [], 200, contacts=True)
    summary = verify_stream_gravity(stream, 1)
    assert summary["mismatch_count"] == 0
    assert summary["contact_events"] == 0
    assert summary["contact_run"] is False
    assert check_contact_resolution(summary).passed


def test_tampered_contact_record_fails(tmp_path):
    stream = tmp_path / "ontos.stream"
    _emit_gravity_stream(stream, 11, 32, [], 60, contacts=True)
    data = bytearray(stream.read_bytes())
    sizes = {1: 9, 2: 9, 3: 17, 4: 10, 5: 34, 6: 55, 7: 57, 8: 73, 9: 57, 10: 41}
    off = 20
    flipped = False
    while off < len(data):
        tag = data[off]
        if tag == 10 and not flipped:
            data[off + 17] ^= 0x01  # low byte of jn
            flipped = True
            break
        off += sizes[tag]
    assert flipped
    corrupt = tmp_path / "corrupt.stream"
    corrupt.write_bytes(bytes(data))
    summary = verify_stream_gravity(corrupt, 11)
    assert summary["mismatch_count"] > 0
    assert not check_reference_match_gravity(summary).passed


def test_engine_diagnose_contact_run(tmp_path):
    import shutil

    run = tmp_path / "ontos_run"
    shutil.copytree(CONTACT_EXAMPLES / "contact", run)
    engine = select_engine(run)
    assert engine.name == "ontos"
    ctx = engine.load_context(run, selection="default")
    results = run_checks(ctx)
    names = {r.name for r in results}
    assert "ontos_contact_resolution" in names
    assert "ontos_audio_match" in names
    assert all(r.passed for r in results if r.name.startswith("ontos_"))


# --- Section 23: radial-shape synthesis ---


def test_radial_roundtrip_self_consistent(tmp_path):
    order, _ = _region_occupancy(42, 8, 5)
    region = order[0]
    stream = tmp_path / "radial.stream"
    _emit_gravity_stream(stream, 42, 8, [(6, region, 2), (36, region, 1)], 80, radial=True)
    summary = verify_stream_gravity(stream, 42)
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["collapse_events"] == 1
    assert summary["radial_events"] == 1
    assert summary["expand_events"] == 1
    result = check_radial_shape(summary)
    assert result.passed, result.detail
    assert result.detail["worst_dipole_relative"] <= 1e-12
    assert result.detail["worst_binding_relative"] <= 1e-9
    assert result.detail["worst_energy_relative"] <= 1e-9
    assert result.detail["worst_quadrupole_relative"] <= 4.0


def test_radial_recorded_stream_verifies():
    summary = verify_stream_gravity(EXAMPLES / "radial" / "ontos.stream", 17)
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["ticks_verified"] == 240
    assert summary["radial_events"] == 3
    assert summary["expand_events"] == 3
    result = check_radial_shape(summary)
    assert result.passed, result.detail
    # Measured: energy closes at rounding, binding at bisection precision.
    assert result.detail["worst_energy_relative"] < 1e-6
    assert result.detail["worst_binding_relative"] < 1e-9
    assert check_reconstruction_error(summary).passed


def test_radial_energy_delta_beats_section20():
    # Same seed/schedule with and without --radial: the section 20 energy
    # delta is O(0.1-1) (tensor match does not pin pair distances); the
    # section 23 closure drives it to rounding.
    order, _ = _region_occupancy(7, 12, 5)
    region = order[0]
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        plain = Path(td) / "plain.stream"
        rad = Path(td) / "rad.stream"
        _emit_gravity_stream(plain, 7, 12, [(6, region, 2), (36, region, 1)], 60)
        _emit_gravity_stream(rad, 7, 12, [(6, region, 2), (36, region, 1)], 60, radial=True)
        s20 = verify_stream_gravity(plain, 7)
        s23 = verify_stream_gravity(rad, 7)
    worst_20 = max(d[4] for d in s20["multipole_deltas"])
    worst_23 = max(d[5] for d in s23["radial_deltas"])
    assert worst_23 < 1e-6
    assert worst_20 > 100 * worst_23


def test_tampered_radial_record_fails(tmp_path):
    order, _ = _region_occupancy(42, 8, 5)
    stream = tmp_path / "ontos.stream"
    _emit_gravity_stream(stream, 42, 8, [(6, order[0], 2), (36, order[0], 1)], 40, radial=True)
    data = bytearray(stream.read_bytes())
    sizes = {1: 9, 2: 9, 3: 17, 4: 10, 5: 34, 6: 55, 7: 57, 8: 73, 9: 57, 10: 41, 11: 25}
    off = 20
    flipped = False
    while off < len(data):
        tag = data[off]
        if tag == 11 and not flipped:
            data[off + 17] ^= 0x01  # low byte of binding
            flipped = True
            break
        off += sizes[tag]
    assert flipped
    corrupt = tmp_path / "corrupt.stream"
    corrupt.write_bytes(bytes(data))
    summary = verify_stream_gravity(corrupt, 42)
    assert summary["mismatch_count"] > 0
    assert not check_reference_match_gravity(summary).passed


def test_radial_two_body_cycle_closes():
    world = None
    for seed in range(100_000):
        w = GravityWorld(seed, 2)
        if all(64.0 <= b["x"] < 128.0 and 64.0 <= b["y"] < 128.0 for b in w.bodies):
            world = w
            break
    w = world
    w.radial_enabled = True
    w.schedule(1, 3, 2)
    w.schedule(30, 3, 1)
    for _ in range(30):
        w.step()
    exp = w.last_expansion
    assert exp is not None and exp["radial"]
    b0, b1 = exp["bodies"]
    dx = b1["x"] - b0["x"]
    dy = b1["y"] - b0["y"]
    binding = b0["mass"] * b1["mass"] / math.sqrt(dx * dx + dy * dy + 1.0)
    rel = abs(binding - exp["target_binding"]) / max(abs(exp["target_binding"]), 1e-30)
    assert rel < 1e-9, rel


def test_engine_diagnose_radial_run(tmp_path):
    import shutil

    run = tmp_path / "ontos_run"
    shutil.copytree(EXAMPLES / "radial", run)
    engine = select_engine(run)
    assert engine.name == "ontos"
    ctx = engine.load_context(run, selection="default")
    results = run_checks(ctx)
    names = {r.name for r in results}
    assert "ontos_radial_shape" in names
    assert all(r.passed for r in results if r.name.startswith("ontos_"))


# --- Section 25: per-shell radial synthesis ---


def test_shells_roundtrip_self_consistent(tmp_path):
    order, _ = _region_occupancy(42, 12, 5)
    region = order[0]
    stream = tmp_path / "shells.stream"
    _emit_gravity_stream(stream, 42, 12, [(6, region, 2), (36, region, 1)], 80, shells=True)
    summary = verify_stream_gravity(stream, 42)
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["collapse_events"] == 1
    assert summary["shell_events"] == 1
    assert summary["expand_events"] == 1
    result = check_shell_shape(summary)
    assert result.passed, result.detail
    assert result.detail["worst_dipole_relative"] <= 1e-12
    assert result.detail["worst_shell_binding_relative"] <= 1e-9
    if result.detail["vertex_cycles"] == 0:
        assert result.detail["worst_energy_relative"] <= 1e-9


def test_shells_recorded_stream_verifies():
    summary = verify_stream_gravity(EXAMPLES / "shells" / "ontos.stream", 17)
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["ticks_verified"] == 240
    assert summary["shell_events"] == 3
    assert summary["expand_events"] == 3
    result = check_shell_shape(summary)
    assert result.passed, result.detail
    # Measured: per-shell binding closes at bisection precision, dipole
    # exactly, energy at rounding for every reachable cycle.
    assert result.detail["worst_shell_binding_relative"] < 1e-9
    assert result.detail["worst_dipole_relative"] < 1e-12
    assert check_reconstruction_error(summary).passed


def test_shells_per_shell_binding_exact_across_seeds(tmp_path):
    # The exact invariant of the mode: each solve-time shell's intra
    # binding, measured on the final synthesized positions, closes on
    # its record. Swept seeds keep it at rounding.
    worst = 0.0
    for seed in (7, 11, 13, 19, 23):
        order, _ = _region_occupancy(seed, 16, 5)
        stream = tmp_path / f"shells_{seed}.stream"
        _emit_gravity_stream(
            stream, seed, 16, [(6, order[0], 2), (40, order[0], 1)], 100, shells=True
        )
        summary = verify_stream_gravity(stream, seed)
        assert summary["mismatch_count"] == 0
        for d in summary["shell_deltas"]:
            worst = max(worst, d[4])
    assert worst < 1e-9


def test_shells_classification_precedes_global_scale():
    # OTO-017 mirror (ontos near-tie regression): section 25 step 2
    # classifies the UNSCALED base before step 3 applies the global
    # lambda. The frozen 12-member layout below (adjacent +/- pairs,
    # unit masses, so the mass-weighted mean is exactly zero scaled and
    # unscaled) carries an exact radius tie — members 4/5 (+/-(4,13))
    # and 6/7 (+/-(8,11)) all sit at r = sqrt(185) — straddling the
    # shell boundary. Unscaled, the tie resolves by id (4,5 in shell
    # 0); under the solved lambda (bits pinned below) the scaled radii
    # round apart with (8,11) below (4,13), so classifying after the
    # scale would swap the pairs across the boundary. Constants were
    # hunted once offline against ontos' pinned 128-bisection and
    # verified at the solved lambda and its nextafter neighbors.
    from simval.ontos_gravity import shell_assignment

    w = GravityWorld(9, 12)
    for b in w.bodies:
        b["mass"] = 1.0
    base = {
        0: (1.0, 0.0),
        1: (-1.0, 0.0),
        2: (2.0, 1.0),
        3: (-2.0, -1.0),
        4: (4.0, 13.0),
        5: (-4.0, -13.0),
        6: (8.0, 11.0),
        7: (-8.0, -11.0),
        8: (16.0, 16.0),
        9: (-16.0, -16.0),
        10: (18.0, 18.0),
        11: (-18.0, -18.0),
    }
    rec = {
        "mass": 12.0,
        "binding": 31.449361063651804,
        "shell_bindings": [
            2.8963589184073895,
            1.35434973735769,
            1.193364772721091,
            1.28597484643118,
        ],
    }
    members = list(range(12))
    work = dict(base)
    _, groups = w._shell_scale(members, rec, work)
    # Spec classification: unscaled radii about the unscaled mean.
    swx = 0.0
    swy = 0.0
    for i in members:
        swx += base[i][0]
        swy += base[i][1]
    cx = swx / rec["mass"]
    cy = swy / rec["mass"]
    unscaled = []
    for i in members:
        dx = base[i][0] - cx
        dy = base[i][1] - cy
        unscaled.append(math.sqrt(dx * dx + dy * dy))
    want = shell_assignment(unscaled)
    want_groups = [[] for _ in range(4)]
    for slot, shell in enumerate(want):
        want_groups[shell].append(slot)
    assert groups == want_groups, "pre-scale classification wins"
    # Premise guard: the tie is exact unscaled and breaks under the
    # solved scale, so the post-scale classification differs — the
    # test genuinely pins the order.
    lam = struct.unpack("<d", struct.pack("<Q", 0x3FBE3BCD35A85906))[0]
    rc = math.sqrt((lam * 4.0) ** 2 + (lam * 13.0) ** 2)
    rd = math.sqrt((lam * 8.0) ** 2 + (lam * 11.0) ** 2)
    assert struct.pack("<d", unscaled[4]) == struct.pack("<d", unscaled[6])
    assert rd < rc, f"scaled rounding must break the tie toward (8,11): {rd} !< {rc}"
    scaled = []
    for i in members:
        sx = lam * base[i][0]
        sy = lam * base[i][1]
        scaled.append(math.sqrt(sx * sx + sy * sy))
    assert shell_assignment(scaled) != want, (
        "post-scale classification must differ for the premise to bite"
    )
    # Secondary: the per-shell solves close on the record targets with
    # the pre-scale classification retained.
    for k, group in enumerate(groups):
        binding = 0.0
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                dx = work[group[b]][0] - work[group[a]][0]
                dy = work[group[b]][1] - work[group[a]][1]
                binding += 1.0 / math.sqrt(dx * dx + dy * dy + 1.0)
        target = rec["shell_bindings"][k]
        rel = abs(binding - target) / max(abs(target), 1e-30)
        assert rel < 1e-9, f"shell {k} residual {rel}"


def test_shells_modes_differ_from_radial(tmp_path):
    order, _ = _region_occupancy(42, 12, 5)
    region = order[0]
    rad = tmp_path / "rad.stream"
    shl = tmp_path / "shl.stream"
    _emit_gravity_stream(rad, 42, 12, [(6, region, 2), (36, region, 1)], 80, radial=True)
    _emit_gravity_stream(shl, 42, 12, [(6, region, 2), (36, region, 1)], 80, shells=True)
    assert rad.read_bytes() != shl.read_bytes()


def test_tampered_shells_record_fails(tmp_path):
    order, _ = _region_occupancy(42, 12, 5)
    stream = tmp_path / "ontos.stream"
    _emit_gravity_stream(stream, 42, 12, [(6, order[0], 2), (36, order[0], 1)], 40, shells=True)
    data = bytearray(stream.read_bytes())
    sizes = {1: 9, 2: 9, 3: 17, 4: 10, 5: 34, 6: 55, 7: 57, 8: 73, 9: 57, 10: 41, 11: 25, 13: 57}
    off = 20
    flipped = False
    while off < len(data):
        tag = data[off]
        if tag == 13 and not flipped:
            data[off + 25] ^= 0x01  # low byte of b0
            flipped = True
            break
        off += sizes[tag]
    assert flipped
    corrupt = tmp_path / "corrupt.stream"
    corrupt.write_bytes(bytes(data))
    summary = verify_stream_gravity(corrupt, 42)
    assert summary["mismatch_count"] > 0
    assert not check_reference_match_gravity(summary).passed


def test_engine_diagnose_shells_run(tmp_path):
    import shutil

    run = tmp_path / "ontos_run"
    shutil.copytree(EXAMPLES / "shells", run)
    engine = select_engine(run)
    assert engine.name == "ontos"
    ctx = engine.load_context(run, selection="default")
    results = run_checks(ctx)
    names = {r.name for r in results}
    assert "ontos_shell_shape" in names
    assert all(r.passed for r in results if r.name.startswith("ontos_"))


# --- Section 26: collapse-on-coarse composition ---


def test_coarse_collapse_roundtrip_self_consistent(tmp_path):
    order, _ = _region_occupancy(11, 10, 5)
    region = order[0]
    stream = tmp_path / "coarse.stream"
    _emit_gravity_stream(
        stream, 11, 10, [(10, region, 0), (30, region, 2), (120, region, 1)], 200
    )
    summary = verify_stream_gravity(stream, 11)
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["collapse_events"] == 1
    assert summary["expand_events"] == 1
    assert check_bounded_drift(summary).passed


def test_coarse_collapse_recorded_stream_verifies():
    summary = verify_stream_gravity(EXAMPLES / "coarse_collapse" / "ontos.stream", 23)
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["collapse_events"] == 1
    assert summary["expand_events"] == 1
    assert check_bounded_drift(summary).passed
    assert check_reconstruction_error(summary).passed


def test_coarse_collapse_terminates_window_for_out_of_box_bodies():
    w = GravityWorld(11, 8)
    w.bodies[0]["x"] = 127.995
    w.bodies[0]["y"] = 96.0
    w.bodies[0]["vx"] = 0.5
    w.bodies[0]["vy"] = 0.0
    w.schedule(1, 3, 0)
    w.schedule(30, 3, 2)
    for _ in range(30):
        w.step()
    assert w.region_collapsed[3] is not None
    assert w.region_collapsed[3]["count"] > 0
    assert 0 not in w.region_collapsed[3]["members"]
    assert w.body_collapsed[0] is None
    assert w.coarse[0] is None, "fit discarded at collapse"
    _, region, level = w.emitted_state(0)
    assert (region, level) == (UNMANAGED, 1)
    for _ in range(10):
        w.step()
    _, region, level = w.emitted_state(0)
    assert level == 1, "stays fine after the collapse"


def test_coarse_collapse_absorbs_foreign_window_bodies_by_evaluation():
    w = GravityWorld(11, 8)
    w.bodies[0]["x"] = 63.995
    w.bodies[0]["y"] = 32.0
    w.bodies[0]["vx"] = 0.5
    w.bodies[0]["vy"] = 0.0
    w.schedule(1, 0, 0)
    w.schedule(30, 1, 2)
    for _ in range(30):
        w.step()
    rec = w.region_collapsed[1]
    assert rec is not None and 0 in rec["members"]
    assert w.coarse[0] is None, "old window discarded"
    _, region, level = w.emitted_state(0)
    assert (region, level) == (1, 2)


def test_coarse_collapse_with_shells_verifies(tmp_path):
    order, _ = _region_occupancy(11, 16, 5)
    region = order[0]
    stream = tmp_path / "coarse_shells.stream"
    _emit_gravity_stream(
        stream, 11, 16, [(10, region, 0), (30, region, 2), (120, region, 1)], 200, shells=True
    )
    summary = verify_stream_gravity(stream, 11)
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["shell_events"] == 1
    assert check_shell_shape(summary).passed


# --- Section 24: contact extensions ---


def test_restitution_emitter_reproduces_recorded_stream(tmp_path):
    out = tmp_path / "restitution.stream"
    _emit_gravity_stream(out, 11, 32, [], 400, contacts=True, params=(0.5, 0.25, 0))
    assert out.read_bytes() == (CONTACT_EXAMPLES / "restitution" / "ontos.stream").read_bytes()


def test_walls_emitter_reproduces_recorded_stream(tmp_path):
    out = tmp_path / "walls.stream"
    _emit_gravity_stream(
        out, 22, 24, [(5, 2, 2), (200, 2, 1), (260, 2, 2)], 600, contacts=True, params=(0.0, 0.0, 1)
    )
    assert out.read_bytes() == (CONTACT_EXAMPLES / "walls" / "ontos.stream").read_bytes()


def test_restitution_stream_verifies():
    summary = verify_stream_gravity(CONTACT_EXAMPLES / "restitution" / "ontos.stream", 11)
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["contact_events"] >= 3
    assert summary["contact_restitution"] == 0.5
    assert check_contact_resolution(summary).passed
    assert summary["contact_worst_vn_after"] < 1e-12
    assert summary["contact_min_jn"] > 0.0


def test_walls_stream_verifies_with_static_contacts():
    summary = verify_stream_gravity(CONTACT_EXAMPLES / "walls" / "ontos.stream", 22)
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["contact_events"] >= 4
    assert summary["static_contact_events"] == 2
    assert summary["contact_run"] is True
    assert check_contact_resolution(summary).passed
    assert check_multipole_match(summary).passed


def test_restitution_and_friction_keep_ledger_exact_all_fine():
    world = GravityWorld(11, 32)
    world.contacts = True
    world.contact_params = True
    world.restitution = 0.5
    world.friction = 0.25
    px0, py0 = world.px, world.py
    impulses = 0
    for _ in range(400):
        world.step()
        impulses += len(world.last_contacts)
        world.last_contacts.clear()
    assert impulses >= 3
    assert world.px == px0
    assert world.py == py0


def test_friction_kills_tangential_relative_velocity():
    # Constructed overlapping pair with a tangential approach; friction
    # far inside the Coulomb cone (friction = 10) so the impulse is
    # unclamped and the tangential relative velocity closes on zero to
    # rounding of the construction (gravity perturbs it at ~4e-4).
    world = GravityWorld(3, 2)
    world.contacts = True
    world.contact_params = True
    world.friction = 10.0
    world.bodies[0]["x"] = 40.0
    world.bodies[0]["y"] = 40.0
    world.bodies[1]["x"] = 42.0
    world.bodies[1]["y"] = 40.0
    world.bodies[0]["vx"] = 0.0
    world.bodies[0]["vy"] = 0.0
    world.bodies[1]["vx"] = -0.1
    world.bodies[1]["vy"] = 0.3
    world.step()
    assert len(world.last_contacts) == 1
    dx = world.bodies[1]["x"] - world.bodies[0]["x"]
    dy = world.bodies[1]["y"] - world.bodies[0]["y"]
    d = math.sqrt(dx * dx + dy * dy)
    tx = -(dy / d)
    ty = dx / d
    vt = (world.bodies[1]["vx"] - world.bodies[0]["vx"]) * tx + (
        world.bodies[1]["vy"] - world.bodies[0]["vy"]
    ) * ty
    assert abs(vt) < 1e-3, f"tangential velocity after unclamped friction {vt}"
    world.last_contacts.clear()


def test_wall_bounce_reflects_and_books_ledger():
    world = GravityWorld(3, 1)
    world.contacts = True
    world.contact_params = True
    world.walls = True
    world.restitution = 0.5
    world.friction = 0.25
    world.bodies[0]["x"] = -1.0
    world.bodies[0]["y"] = 50.0
    world.bodies[0]["vx"] = -0.5
    world.bodies[0]["vy"] = 0.25
    m = world.bodies[0]["mass"]
    px0, py0 = world.px, world.py
    world.step()
    events = world.last_contacts
    assert len(events) == 1
    c = events[0]
    assert c["b"] == WALL_BASE
    assert c["jn"] > 0.0
    assert abs(world.bodies[0]["vx"] - 0.25) < 1e-12
    assert abs(world.bodies[0]["vy"] - 0.0625) < 1e-12
    s = (1.0 + 0.5) * -0.5
    assert abs(world.px - (px0 + m * (s * (0.0 - 1.0)))) < 1e-12
    assert abs(world.py - (py0 + 0.1875 * m * (0.0 - 1.0))) < 1e-12
    world.last_contacts.clear()


def test_monopole_contact_one_sided_frozen_totals():
    world = None
    for seed in range(100_000):
        w = GravityWorld(seed, 5)
        in3 = sum(
            1
            for b in w.bodies
            if 64.0 <= b["x"] < 128.0 and 64.0 <= b["y"] < 128.0
        )
        if in3 >= 3 and w.bodies[0]["x"] < 64.0:
            world = w
            break
    w = world
    w.contacts = True
    w.contact_params = True
    w.restitution = 0.5
    w.schedule(1, 3, 2)
    w.step()
    rec = w.region_collapsed[3]
    w.bodies[0]["x"] = rec["com_x"] - 2.0
    w.bodies[0]["y"] = rec["com_y"]
    w.bodies[0]["vx"] = rec["vcom_x"] + 1.0
    w.bodies[0]["vy"] = rec["vcom_y"]
    mi = w.bodies[0]["mass"]
    w.step()
    events = w.last_contacts
    assert len(events) == 1
    c = events[0]
    assert c["b"] == MONOPOLE_BASE + 3
    expected_jn = (0.0 - c["vn"]) * (1.0 + 0.5) * mi
    assert abs(c["jn"] - expected_jn) < 1e-12
    assert abs(c["vn_after"] - (0.0 - c["vn"] * 0.5)) < 1e-12
    after = w.region_collapsed[3]
    assert (after["com_x"], after["com_y"], after["vcom_x"], after["vcom_y"]) == (
        rec["com_x"],
        rec["com_y"],
        rec["vcom_x"],
        rec["vcom_y"],
    )
    w.last_contacts.clear()


def test_contact_params_after_first_tick_rejected(tmp_path):
    stream = tmp_path / "ontos.stream"
    _emit_gravity_stream(stream, 11, 8, [], 10, contacts=True, params=(0.5, 0.25, 0))
    data = bytearray(stream.read_bytes())
    record = bytes(data[20:38])  # the 1 + 17 byte ContactParams record
    del data[20:38]
    data += record  # now after every tick
    moved = tmp_path / "moved.stream"
    moved.write_bytes(bytes(data))
    # The strict grammar rejects ContactParams after the first TickHeader
    # (and the dangling record at EOF) outright.
    with pytest.raises(ValueError, match="ContactParams"):
        verify_stream_gravity(moved, 11)


def test_engine_diagnose_walls_run(tmp_path):
    import shutil

    run = tmp_path / "ontos_run"
    shutil.copytree(CONTACT_EXAMPLES / "walls", run)
    engine = select_engine(run)
    assert engine.name == "ontos"
    ctx = engine.load_context(run, selection="default")
    results = run_checks(ctx)
    names = {r.name for r in results}
    assert "ontos_contact_resolution" in names
    assert "ontos_audio_match" in names
    assert all(r.passed for r in results if r.name.startswith("ontos_"))


# --- Section 24 corpus coverage: wall-hit + fine x coarse static contacts ---


def test_wallshot_emitter_reproduces_recorded_stream(tmp_path):
    out = tmp_path / "wallshot.stream"
    _emit_gravity_stream(
        out, 3, 16, [], 800, contacts=True, params=(0.7, 0.3, 1), profile="wallshot"
    )
    assert out.read_bytes() == (CONTACT_EXAMPLES / "wallshot" / "ontos.stream").read_bytes()


def test_coarsehit_emitter_reproduces_recorded_stream(tmp_path):
    out = tmp_path / "coarsehit.stream"
    _emit_gravity_stream(
        out, 11, 8, [(1, 3, 0)], 500, contacts=True, params=(0.5, 0.25, 0), profile="coarsehit"
    )
    assert out.read_bytes() == (CONTACT_EXAMPLES / "coarsehit" / "ontos.stream").read_bytes()


def test_wallshot_stream_hits_every_wall():
    from simval.ontos_gravity import parse_stream_v2

    summary = verify_stream_gravity(
        CONTACT_EXAMPLES / "wallshot" / "ontos.stream", 3, "wallshot"
    )
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["contact_events"] == 16
    assert summary["static_contact_events"] == 16
    assert summary["contact_run"] is True
    assert check_contact_resolution(summary).passed
    assert summary["contact_worst_vn_after"] < 1e-12
    assert summary["contact_min_jn"] > 0.0
    _, records = parse_stream_v2(CONTACT_EXAMPLES / "wallshot" / "ontos.stream")
    walls = {}
    for record in records:
        if record[0] == "contact" and record[3] >= WALL_BASE:
            walls[record[3] - WALL_BASE] = walls.get(record[3] - WALL_BASE, 0) + 1
    assert walls == {0: 4, 1: 4, 2: 4, 3: 4}, walls


def test_coarsehit_stream_lands_static_contacts():
    summary = verify_stream_gravity(
        CONTACT_EXAMPLES / "coarsehit" / "ontos.stream", 11, "coarsehit"
    )
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["contact_events"] == 4
    assert summary["coarse_static_contact_events"] == 4
    assert summary["contact_run"] is True
    assert check_contact_resolution(summary).passed
    assert summary["contact_worst_vn_after"] < 1e-12
    assert summary["contact_min_jn"] > 0.0


def test_static_contact_fires_when_coarse_body_has_smaller_id():
    # OTO-016 mirror (ontos seed-117 regression): section 24 static
    # contact is positional — the sweep must reach a fine body resolving
    # against an ephemeris-coarse contactant even when the coarse body
    # carries the smaller id. Body 1 sits in region 2 and is demoted at
    # tick 1; fine body 3 outside the box overlaps the sum of radii
    # inbound. The old sweep skipped the pair (the outer index had to
    # be fine), letting body 3 fall through the frozen body.
    def setup(contacts):
        w = GravityWorld(117, 4)
        w.bodies[0]["mass"] = 0.5
        w.bodies[0]["x"] = 16.0
        w.bodies[0]["y"] = 16.0
        w.bodies[0]["vx"] = 0.0
        w.bodies[0]["vy"] = 0.0
        w.bodies[1]["mass"] = 2.0
        w.bodies[1]["x"] = 32.0
        w.bodies[1]["y"] = 65.0
        w.bodies[1]["vx"] = 0.0
        w.bodies[1]["vy"] = 0.0
        w.bodies[2]["mass"] = 0.5
        w.bodies[2]["x"] = 112.0
        w.bodies[2]["y"] = 16.0
        w.bodies[2]["vx"] = 0.0
        w.bodies[2]["vy"] = 0.0
        w.bodies[3]["mass"] = 2.0
        w.bodies[3]["x"] = 32.0
        w.bodies[3]["y"] = 63.0
        w.bodies[3]["vx"] = 0.0
        w.bodies[3]["vy"] = 0.25
        w.contacts = contacts
        w.contact_params = contacts
        w.schedule(1, 2, 0)
        return w

    def same_bits(a, b):
        return struct.pack("<d", a) == struct.pack("<d", b)

    def same_fit(a, b):
        if a is None or b is None:
            return False
        if a["t0"] != b["t0"]:
            return False
        for ca, cb in zip(a["c"], b["c"]):
            if not all(same_bits(u, v) for u, v in zip(ca, cb)):
                return False
        return True

    ctl = setup(False)
    w = setup(True)
    ctl.step()
    w.step()
    assert len(ctl.last_contacts) == 0
    assert len(w.last_contacts) == 1, "the (coarse 1, fine 3) pair fires"
    c = w.last_contacts[0]
    assert (c["a"], c["b"]) == (1, 3), "record ids in pinned (min, max) order"
    assert c["static_pair"] is True
    assert c["jn"] > 0.0
    assert abs(c["vn_after"]) < 1e-12, f"e = 0 closure, vn_after {c['vn_after']}"
    assert (1, 3) in w.touching
    assert same_fit(w.coarse[1], ctl.coarse[1]), "fit identical to the no-contact run"
    # Body 3 receives exactly the one-sided impulse: its pre-impulse
    # state is the control run's (identical history up to the pass).
    s1 = w._state_at(1, 1)
    p3 = ctl.bodies[3]
    dx = s1["x"] - p3["x"]
    dy = s1["y"] - p3["y"]
    dist = math.sqrt(dx * dx + dy * dy)
    nx = dx / dist
    ny = dy / dist
    vrx = s1["vx"] - p3["vx"]
    vry = s1["vy"] - p3["vy"]
    vn = vrx * nx + vry * ny
    s = (1.0 + w.restitution) * vn
    m3 = w.bodies[3]["mass"]
    assert same_bits(w.bodies[3]["vx"], p3["vx"] + s * nx), "body 3 vx changed by s * nx"
    assert same_bits(w.bodies[3]["vy"], p3["vy"] + s * ny), "body 3 vy changed by s * ny"
    # The ledger books exactly the one-sided impulse (the fc-kick
    # bookings of the demoted body are shared with the control run).
    assert same_bits(w.px, ctl.px + m3 * (s * nx)), "ledger px books the static impulse"
    assert same_bits(w.py, ctl.py + m3 * (s * ny)), "ledger py books the static impulse"
    frozen_fit = {"t0": w.coarse[1]["t0"], "c": [list(c) for c in w.coarse[1]["c"]]}
    for _ in range(8):
        w.step()
    assert same_fit(w.coarse[1], frozen_fit), "fit stays frozen"


def test_bare_contacts_skip_fine_coarse_static_pairs():
    # OTO-019 mirror (ontos bare_contacts_skip_fine_coarse_static_pairs):
    # the coarsehit IC with bare --contacts (no tag 12 record) and an
    # early demotion follows section 21 — a pair with a non-fine member
    # is skipped in either id orientation, so the run is bit-identical
    # to one with contacts off.
    def run(contacts):
        w = GravityWorld(11, 8, "coarsehit")
        w.contacts = contacts
        w.schedule(1, 3, 0)
        records = 0
        for _ in range(500):
            w.step()
            records += len(w.last_contacts)
            w.last_contacts.clear()
        return w, records

    ctl, ctl_records = run(False)
    bare, bare_records = run(True)
    assert ctl_records == 0, "contacts off emits nothing"
    assert bare_records == 0, "bare contacts emits no record in this IC"
    for a, b in zip(ctl.bodies, bare.bodies):
        assert struct.pack("<d", a["x"]) == struct.pack("<d", b["x"])
        assert struct.pack("<d", a["y"]) == struct.pack("<d", b["y"])
        assert struct.pack("<d", a["vx"]) == struct.pack("<d", b["vx"])
        assert struct.pack("<d", a["vy"]) == struct.pack("<d", b["vy"])
    assert not bare.touching, "no static pair enters the touching set"


def test_zero_restitution_record_admits_static_contacts():
    # OTO-019 mirror (ontos zero_restitution_record_admits_static_contacts):
    # record presence, not value — an explicit --restitution 0 emits
    # tag 12 with zeros and the fine x coarse static pairs fire,
    # distinguishing record absence from zero-valued parameters.
    w = GravityWorld(11, 8, "coarsehit")
    w.contacts = True
    w.contact_params = True
    w.restitution = 0.0
    w.schedule(1, 3, 0)
    statics = 0
    for _ in range(500):
        w.step()
        for c in w.last_contacts:
            assert c["jn"] > 0.0
            if c["b"] < len(w.bodies) and w.coarse[c["b"]] is not None:
                statics += 1
                assert abs(c["vn_after"]) < 1e-12, (
                    f"e = 0 closure, vn_after {c['vn_after']}"
                )
        w.last_contacts.clear()
    assert statics >= 4, f"expected fine x coarse static contacts, got {statics}"


def test_test_ic_profiles_match_spec_ic_masses():
    from simval.ontos_gravity import initial_conditions, test_initial_conditions

    spec = initial_conditions(5, 6)
    for profile in ("wallshot", "coarsehit"):
        corpus = test_initial_conditions(profile, 5, 6)
        for a, b in zip(spec, corpus):
            assert a["id"] == b["id"]
            assert struct.pack("<d", a["mass"]) == struct.pack("<d", b["mass"])


def test_unknown_profile_rejected():
    from simval.ontos_gravity import test_initial_conditions

    with pytest.raises(ValueError, match="unknown corpus profile"):
        test_initial_conditions("nonsense", 1, 4)


def test_engine_diagnose_corpus_runs(tmp_path):
    import shutil

    for name in ("wallshot", "coarsehit"):
        run = tmp_path / f"ontos_run_{name}"
        shutil.copytree(CONTACT_EXAMPLES / name, run)
        engine = select_engine(run)
        assert engine.name == "ontos"
        ctx = engine.load_context(run, selection="default")
        results = run_checks(ctx)
        names = {r.name for r in results}
        assert "ontos_contact_resolution" in names
        assert "ontos_audio_match" in names
        assert all(r.passed for r in results if r.name.startswith("ontos_")), name


# --- ONT-002/004/005: strict grammar, frame-tick equality, canonical encodings ---

_RECORD_SIZES = {1: 9, 2: 9, 3: 17, 4: 10, 5: 34, 6: 55, 7: 57, 8: 73, 9: 57, 10: 41, 11: 25, 12: 18, 13: 57}


def _record_offsets(data: bytes) -> list[tuple[int, int]]:
    offs = []
    off = 20
    while off < len(data):
        offs.append((data[off], off))
        off += _RECORD_SIZES[data[off]]
    assert off == len(data)
    return offs


def _nth_offset(data: bytes, tag: int, n: int = 0) -> int:
    hits = [off for t, off in _record_offsets(data) if t == tag]
    return hits[n]


def _grammar_stream(tmp_path, name="g.stream", **kw):
    path = tmp_path / name
    _emit_gravity_stream(path, 42, 4, kw.pop("schedule", []), kw.pop("ticks", 6), **kw)
    return path.read_bytes()


def test_v2_header_only_rejected(tmp_path):
    p = tmp_path / "h.stream"
    p.write_bytes(b"ONTO" + struct.pack("<IIII", 2, 128, 128, 4))
    with pytest.raises(ValueError, match="no tick frames"):
        parse_stream_v2(p)


def test_v2_tickheader_only_rejected(tmp_path):
    data = _grammar_stream(tmp_path)
    p = tmp_path / "t.stream"
    p.write_bytes(data[:20] + b"\x01" + struct.pack("<Q", 1))
    with pytest.raises(ValueError, match="incomplete tick frame"):
        parse_stream_v2(p)


def test_v2_drop_one_snapshot_rejected(tmp_path):
    data = _grammar_stream(tmp_path)
    off = _nth_offset(data, 2)
    mutated = data[:off] + data[off + 9 :]
    p = tmp_path / "m.stream"
    p.write_bytes(mutated)
    with pytest.raises(ValueError, match="TotalsState|incomplete|outside"):
        parse_stream_v2(p)


def test_v2_drop_one_body_rejected(tmp_path):
    data = _grammar_stream(tmp_path)
    off = _nth_offset(data, 6, 2)
    mutated = data[:off] + data[off + 55 :]
    p = tmp_path / "m.stream"
    p.write_bytes(mutated)
    with pytest.raises(ValueError, match="body|incomplete"):
        parse_stream_v2(p)


def test_v2_drop_tickheader_rejected(tmp_path):
    data = _grammar_stream(tmp_path)
    off = _nth_offset(data, 1, 1)
    mutated = data[:off] + data[off + 9 :]
    p = tmp_path / "m.stream"
    p.write_bytes(mutated)
    with pytest.raises(ValueError, match="outside an open tick frame"):
        parse_stream_v2(p)


def test_v2_duplicate_snapshot_rejected(tmp_path):
    data = _grammar_stream(tmp_path)
    off = _nth_offset(data, 2)
    mutated = data[: off + 9] + data[off : off + 9] + data[off + 9 :]
    p = tmp_path / "m.stream"
    p.write_bytes(mutated)
    with pytest.raises(ValueError):
        parse_stream_v2(p)


def test_v2_duplicate_whole_tick_frame_rejected(tmp_path):
    data = bytearray(_grammar_stream(tmp_path))
    start = _nth_offset(bytes(data), 1, 0)
    frame = bytes(data[start:])
    p = tmp_path / "m.stream"
    p.write_bytes(bytes(data) + frame)
    with pytest.raises(ValueError, match="non-consecutive"):
        parse_stream_v2(p)


def test_v2_append_after_last_tick_rejected(tmp_path):
    data = _grammar_stream(tmp_path)
    stray = b"\x05" + struct.pack("<QIIBQQ", 6, 0, 0, 1, 0, 0)
    p = tmp_path / "m.stream"
    p.write_bytes(data + stray)
    with pytest.raises(ValueError, match="outside an open tick frame"):
        parse_stream_v2(p)


def test_v2_dangling_level_at_eof_rejected(tmp_path):
    data = _grammar_stream(tmp_path)
    p = tmp_path / "m.stream"
    p.write_bytes(data + b"\x04" + struct.pack("<IIB", 0, 0, 0))
    with pytest.raises(ValueError, match="dangling"):
        parse_stream_v2(p)


@pytest.mark.parametrize("tag,payload_start", [(7, 1), (5, 1), (6, 1)])
def test_v2_frame_record_tick_mismatch_rejected(tmp_path, tag, payload_start):
    # ONT-004: totals/state/body records must repeat the open TickHeader tick.
    data = bytearray(_grammar_stream(tmp_path))
    off = _nth_offset(bytes(data), tag, 0)
    data[off + payload_start] += 1  # low byte of the record's u64 tick
    p = tmp_path / "m.stream"
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="does not match the open frame tick"):
        parse_stream_v2(p)


def test_v2_pre_tick_record_tick_mismatch_rejected(tmp_path):
    # A RegionCollapsed tick must equal the TickHeader it precedes.
    order, _ = _region_occupancy(42, 4, 3)
    data = bytearray(_grammar_stream(tmp_path, schedule=[(4, order[0], 2)], ticks=8))
    off = _nth_offset(bytes(data), 8, 0)
    data[off + 1] -= 1
    p = tmp_path / "m.stream"
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="does not match the following TickHeader"):
        parse_stream_v2(p)


def test_v2_region_coord_rewrite_rejected(tmp_path):
    # ONT-005: (0,1) rewritten as (2,0) must not alias region 2.
    data = bytearray(_grammar_stream(tmp_path, schedule=[(2, 2, 0)], ticks=4))
    off = _nth_offset(bytes(data), 4, 0)
    struct.pack_into("<II", data, off + 1, 2, 0)
    p = tmp_path / "m.stream"
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="noncanonical region coordinates"):
        parse_stream_v2(p)


def test_v2_collapsed_region_coord_rewrite_rejected(tmp_path):
    order, _ = _region_occupancy(42, 4, 3)
    data = bytearray(_grammar_stream(tmp_path, schedule=[(4, order[0], 2)], ticks=8))
    off = _nth_offset(bytes(data), 8, 0)
    struct.pack_into("<II", data, off + 9, 0, 3)
    p = tmp_path / "m.stream"
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="noncanonical region coordinates"):
        parse_stream_v2(p)


# --- ONT-003: Snapshot population must be compared, not just counted ---


def test_snapshot_payload_bitflip_produces_mismatch(tmp_path):
    corpus = CONTACT_EXAMPLES / "contact" / "ontos.stream"
    data = bytearray(corpus.read_bytes())
    off = _nth_offset(bytes(data), 2, 0)
    data[off + 1] ^= 0x02  # single bit in the Snapshot population payload
    p = tmp_path / "m.stream"
    p.write_bytes(bytes(data))
    summary = verify_stream_gravity(p, 11)
    assert summary["mismatch_count"] > 0
    assert not check_reference_match_gravity(summary).passed


def test_snapshot_population_compared_against_body_count(tmp_path):
    path = tmp_path / "s.stream"
    _emit_gravity_stream(path, 42, 4, [], 5)
    data = bytearray(path.read_bytes())
    off = _nth_offset(bytes(data), 2, 0)
    struct.pack_into("<Q", data, off + 1, 3)  # wrong population
    p = tmp_path / "m.stream"
    p.write_bytes(bytes(data))
    summary = verify_stream_gravity(p, 42)
    assert summary["mismatch_count"] > 0
    assert any(m["field"] == "snapshot_population" for m in summary["mismatches"])


# --- ONT-006: externally-supplied floats must be finite; checks fail closed ---


def _params_stream(tmp_path, restitution, friction, walls=0):
    """Stream whose ContactParams payload is surgically overwritten (the
    replay itself cannot run with non-finite params — that is the point)."""
    path = tmp_path / "p.stream"
    _emit_gravity_stream(path, 11, 8, [], 4, contacts=True, params=(0.5, 0.25, walls))
    data = bytearray(path.read_bytes())
    off = _nth_offset(bytes(data), 12, 0)
    struct.pack_into("<dd", data, off + 1, restitution, friction)
    path.write_bytes(bytes(data))
    return path


@pytest.mark.parametrize(
    "restitution,friction",
    [
        (float("nan"), 0.25),
        (float("inf"), 0.25),
        (float("-inf"), 0.25),
        (0.5, float("nan")),
        (0.5, float("inf")),
        (0.5, float("-inf")),
    ],
)
def test_nonfinite_contact_params_rejected_by_stream_verifier(tmp_path, restitution, friction):
    p = _params_stream(tmp_path, restitution, friction)
    summary = verify_stream_gravity(p, 11)
    assert summary["mismatch_count"] > 0
    assert any(m["field"] == "contact_params" for m in summary["mismatches"])
    assert not check_reference_match_gravity(summary).passed


def test_valid_contact_params_still_verify(tmp_path):
    p = _params_stream(tmp_path, 0.5, 0.25)
    summary = verify_stream_gravity(p, 11)
    assert summary["mismatch_count"] == 0


@pytest.mark.parametrize(
    "field,bad",
    [
        ("restitution", float("nan")),
        ("restitution", float("inf")),
        ("friction", float("nan")),
        ("friction", float("-inf")),
    ],
)
def test_nonfinite_grid_floats_rejected_by_normalizer(field, bad):
    from simval.orchestrate import normalize_spec

    spec = {"name": "x", "contacts": True, field: bad}
    with pytest.raises(ValueError, match=field):
        normalize_spec(spec)


# --- ONT-001: the expected-run contract drives verification, not the stream ---


def test_contract_from_metadata_validation():
    from simval.ontos_gravity import GravityContract

    with pytest.raises(ValueError, match="events entries"):
        GravityContract.from_metadata({"events": [[1, 0]]})
    with pytest.raises(ValueError, match="outside the 2x2 grid"):
        GravityContract.from_metadata({"events": [[1, 2, 0, 0]]})
    with pytest.raises(ValueError, match="friction"):
        GravityContract.from_metadata({"friction": float("nan")})
    with pytest.raises(ValueError, match="restitution"):
        GravityContract.from_metadata({"restitution": 1.5})
    c = GravityContract.from_metadata(
        {"bodies": 8, "ticks": 10, "events": [[4, 1, 1, 2]], "radial": True}
    )
    assert c.body_count == 8 and c.ticks == 10
    assert c.events == ((4, 3, 2),)


def _contract(ticks=6, **kw):
    from simval.ontos_gravity import GravityContract

    base = {"bodies": 4, "ticks": ticks, "events": []}
    base.update(kw)
    return GravityContract.from_metadata(base)


def test_contract_body_count_mismatch_fails_before_replay(tmp_path):
    p = tmp_path / "s.stream"
    _emit_gravity_stream(p, 42, 4, [], 6)
    summary = verify_stream_gravity(p, 42, expected=_contract(bodies=8))
    assert summary["mismatch_count"] > 0
    assert any(m["field"] == "contract_body_count" for m in summary["mismatches"])


def test_contract_horizon_truncated_stream_fails(tmp_path):
    # A stream cut short used to verify clean: replay matched every
    # present record. The contract pins the requested final tick.
    p = tmp_path / "s.stream"
    _emit_gravity_stream(p, 42, 4, [], 6)
    summary = verify_stream_gravity(p, 42, expected=_contract(ticks=10))
    assert summary["mismatch_count"] > 0
    assert any(m["field"] == "contract_ticks" for m in summary["mismatches"])


def test_contract_missing_scheduled_event_fails(tmp_path):
    order, _ = _region_occupancy(42, 4, 3)
    p = tmp_path / "s.stream"
    _emit_gravity_stream(p, 42, 4, [], 8)
    summary = verify_stream_gravity(
        p, 42, expected=_contract(ticks=8, events=[[2, order[0] % 2, order[0] // 2, 0]])
    )
    assert any(m["field"] == "contract_event" for m in summary["mismatches"])


def test_contract_unscheduled_event_fails_without_observer(tmp_path):
    order, _ = _region_occupancy(42, 4, 3)
    p = tmp_path / "s.stream"
    _emit_gravity_stream(p, 42, 4, [(2, order[0], 0)], 8)
    summary = verify_stream_gravity(p, 42, expected=_contract(ticks=8, events=[]))
    assert any(m["field"] == "contract_unscheduled_event" for m in summary["mismatches"])


def test_contract_declared_event_verifies(tmp_path):
    order, _ = _region_occupancy(42, 4, 3)
    p = tmp_path / "s.stream"
    _emit_gravity_stream(p, 42, 4, [(2, order[0], 0)], 8)
    summary = verify_stream_gravity(
        p, 42, expected=_contract(ticks=8, events=[[2, order[0] % 2, order[0] // 2, 0]])
    )
    assert summary["mismatch_count"] == 0, summary["mismatches"]


def test_contract_radial_mode_required_at_collapses(tmp_path):
    # Stream collapses without radial records; contract says the run
    # was requested with --radial: the radial synthesis evidence is
    # mandatory, and the replay must synthesize radially.
    order, _ = _region_occupancy(42, 4, 3)
    p = tmp_path / "s.stream"
    _emit_gravity_stream(p, 42, 4, [(4, order[0], 2), (6, order[0], 1)], 10)
    summary = verify_stream_gravity(p, 42, expected=_contract(ticks=10, radial=True))
    assert any(m["field"] == "contract_radial_records" for m in summary["mismatches"])


def test_contract_radial_run_verifies(tmp_path):
    order, _ = _region_occupancy(42, 4, 3)
    p = tmp_path / "s.stream"
    _emit_gravity_stream(p, 42, 4, [(4, order[0], 2), (6, order[0], 1)], 10, radial=True)
    summary = verify_stream_gravity(
        p, 42, expected=_contract(ticks=10, radial=True, events=[[4, order[0] % 2, order[0] // 2, 2], [6, order[0] % 2, order[0] // 2, 1]])
    )
    assert summary["mismatch_count"] == 0, summary["mismatches"]
    assert summary["radial_events"] == 1


def test_contract_multipole_off_rejects_multipole_records(tmp_path):
    order, _ = _region_occupancy(42, 4, 3)
    p = tmp_path / "s.stream"
    _emit_gravity_stream(p, 42, 4, [(4, order[0], 2), (6, order[0], 1)], 10)
    summary = verify_stream_gravity(p, 42, expected=_contract(ticks=10, multipole=False))
    assert any(m["field"] == "contract_multipole_records" for m in summary["mismatches"])


def test_contract_contacts_off_rejects_contact_stream(tmp_path):
    p = tmp_path / "s.stream"
    _emit_gravity_stream(p, 11, 8, [], 6, contacts=True)
    summary = verify_stream_gravity(p, 11, expected=_contract(ticks=6, bodies=8, contacts=False))
    assert any(m["field"] == "contract_contacts" for m in summary["mismatches"])


def test_contract_params_values_must_match(tmp_path):
    p = tmp_path / "s.stream"
    _emit_gravity_stream(p, 11, 8, [], 6, contacts=True, params=(0.7, 0.3, 0))
    summary = verify_stream_gravity(
        p, 11, expected=_contract(ticks=6, bodies=8, contacts=True, restitution=0.5, friction=0.25)
    )
    assert any(m["field"] == "contract_contact_params" for m in summary["mismatches"])


def test_engine_mode_mismatch_rejected(tmp_path):
    import shutil

    run = tmp_path / "mixed"
    shutil.copytree(EXAMPLES / "all_fine", run)
    (run / "ontos.json").write_text(json.dumps({"mode": "life", "seed": 42}))
    engine = select_engine(run)
    with pytest.raises(ValueError, match="mode=life"):
        engine.load_context(run, selection="default")


def test_engine_contract_metadata_tamper_fails_checks(tmp_path):
    import shutil

    run = tmp_path / "tampered_meta"
    shutil.copytree(EXAMPLES / "all_fine", run)
    meta = json.loads((run / "ontos.json").read_text())
    meta["ticks"] = 50  # claims a shorter horizon than the stream carries
    (run / "ontos.json").write_text(json.dumps(meta))
    ctx = select_engine(run).load_context(run, selection="default")
    results = run_checks(ctx)
    ref = next(r for r in results if r.name == "ontos_reference_match")
    assert not ref.passed


# --- ONT-007: standalone CLI requires the expected-run contract ---


def test_module_cli_gravity_contract_from_metadata(tmp_path, capsys):
    from simval.ontos_gravity import _main

    run = EXAMPLES / "window"
    rc = _main(
        [str(run / "ontos.stream"), "42", "--metadata", str(run / "ontos.json")]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out and "[NO CONTRACT]" not in out


def test_module_cli_gravity_bare_stream_rejected(tmp_path, capsys):
    from simval.ontos_gravity import _main

    run = EXAMPLES / "window"
    rc = _main([str(run / "ontos.stream"), "42"])
    assert rc == 2
    assert "no expected-run contract" in capsys.readouterr().err


def test_module_cli_gravity_explicit_flags_contract(tmp_path, capsys):
    from simval.ontos_gravity import _main

    run = EXAMPLES / "window"
    rc = _main(
        [str(run / "ontos.stream"), "42", "--mode", "gravity", "--ticks", "120",
         "--bodies", "8", "--demote-at", "20", "0", "0", "--promote-at", "80", "0", "0"]
    )
    assert rc == 0
    assert "[NO CONTRACT]" not in capsys.readouterr().out


def test_module_cli_gravity_contract_flags_catch_horizon_lie(tmp_path, capsys):
    from simval.ontos_gravity import _main

    run = EXAMPLES / "window"
    rc = _main(
        [str(run / "ontos.stream"), "42", "--mode", "gravity", "--ticks", "90",
         "--bodies", "8"]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "contract_ticks" in out


# --- ONT-008: event multiplicity enforced for observer runs ---


def _level_record_offsets(data: bytes):
    sizes = {1: 9, 2: 9, 3: 17, 4: 10, 5: 34, 6: 55, 7: 57, 8: 73, 9: 57,
             10: 41, 11: 25, 12: 18, 13: 57}
    off = 20
    offsets = []
    while off < len(data):
        tag = data[off]
        offsets.append((off, tag))
        off += sizes[tag]
    return offsets


def test_observer_duplicate_requested_event_fails_contract(tmp_path):
    # Duplicate the CLI collapse event record: membership-set comparison
    # let it pass; the multiset contract must reject it.
    import shutil

    run = tmp_path / "ontos_run"
    shutil.copytree(EXAMPLES / "collapse_observer", run)
    data = bytearray((run / "ontos.stream").read_bytes())
    level_offsets = [off for off, tag in _level_record_offsets(bytes(data)) if tag == 4]
    assert level_offsets
    rec = bytes(data[level_offsets[0] : level_offsets[0] + 10])
    dup = bytes(data[: level_offsets[0]]) + rec + bytes(data[level_offsets[0]:])
    (run / "ontos.stream").write_bytes(dup)

    from simval.context import select_engine
    from simval.pipeline import run_checks as rc

    ctx = select_engine(run).load_context(run, selection="default")
    results = rc(ctx)
    ref = next(r for r in results if r.name == "ontos_reference_match")
    assert not ref.passed
    assert any(
        m["field"] == "contract_unscheduled_event"
        for m in ref.detail["first_mismatches"]
    )


def test_observer_duplicate_policy_event_fails(tmp_path):
    import shutil

    from simval.ontos_gravity import check_zoom_policy, parse_stream_v2

    run = tmp_path / "ontos_run"
    shutil.copytree(EXAMPLES / "observer", run)
    data = bytearray((run / "ontos.stream").read_bytes())
    level_offsets = [off for off, tag in _level_record_offsets(bytes(data)) if tag == 4]
    assert level_offsets
    rec = bytes(data[level_offsets[0] : level_offsets[0] + 10])
    dup = bytes(data[: level_offsets[0]]) + rec + bytes(data[level_offsets[0]:])
    (run / "ontos.stream").write_bytes(dup)
    _, records = parse_stream_v2(run / "ontos.stream")
    result = check_zoom_policy(records, 5, 777)
    assert not result.passed
    assert result.detail["unexpected"]


def test_observer_clean_run_event_multiset_matches(tmp_path):
    from simval.ontos_gravity import check_zoom_policy, parse_stream_v2

    for name, seed, offset in (("observer", 5, 777), ("collapse_observer", 13, 42)):
        _, records = parse_stream_v2(EXAMPLES / name / "ontos.stream")
        meta = json.loads((EXAMPLES / name / "ontos.json").read_text())
        cli = [(int(t), ry * 2 + rx, int(lv)) for t, rx, ry, lv in meta.get("events", [])]
        result = check_zoom_policy(records, seed, offset, cli_events=cli)
        assert result.passed, (name, result.detail)


def test_zoom_policy_never_promotes_collapsed_region():
    # ONT-010: seed 13 / observer 42 / collapse-at 40 0 0 — the focus
    # enters region 0's promote band at tick 49 while region 0 carries a
    # CLI collapse. The policy gates promotion on Coarse only; the old
    # `modes[region] != 0` arm promoted the Collapsed region.
    from simval.ontos_gravity import expected_zoom_policy

    events = expected_zoom_policy(13, 42, 300, cli_events=[(40, 0, 2)])
    assert (49, 0, 1) not in events
    assert not any(t == 49 and r == 0 and lv == 1 for t, r, lv in events)


def test_contract_observer_without_ticks_rejected():
    from simval.ontos_gravity import GravityContract

    with pytest.raises(ValueError, match="observer runs must carry ticks"):
        GravityContract.from_metadata({"observer": 777, "bodies": 8})


def test_observer_duplicate_event_fails_zoom_contract_in_verifier(tmp_path):
    # End-to-end through verify_stream_gravity with the metadata contract:
    # a duplicated RegionLevel record must produce a contract mismatch.
    import shutil

    from simval.ontos_gravity import GravityContract, verify_stream_gravity

    run = tmp_path / "ontos_run"
    shutil.copytree(EXAMPLES / "collapse_observer", run)
    data = bytearray((run / "ontos.stream").read_bytes())
    level_offsets = [off for off, tag in _level_record_offsets(bytes(data)) if tag == 4]
    rec = bytes(data[level_offsets[0] : level_offsets[0] + 10])
    (run / "ontos.stream").write_bytes(
        bytes(data[: level_offsets[0]]) + rec + bytes(data[level_offsets[0]:])
    )
    meta = json.loads((EXAMPLES / "collapse_observer" / "ontos.json").read_text())
    summary = verify_stream_gravity(
        run / "ontos.stream", 13, expected=GravityContract.from_metadata(meta)
    )
    assert summary["mismatch_count"] > 0
    assert any(
        m["field"].startswith("contract_") for m in summary["mismatches"]
    )


# --- ONT-020: boundary events are compared as ORDERED sequences ---


def test_expected_event_sequence_is_producer_canonical():
    from simval.ontos_gravity import expected_event_sequence

    # Scheduled events normalize like the ontos CLI (sort + dedup); the
    # metadata's original order must not leak into the expectation.
    seq = expected_event_sequence([(2, 0, 1), (2, 0, 0), (1, 3, 0), (2, 0, 0)])
    assert seq == [(1, 3, 0), (2, 0, 0), (2, 0, 1)]


def test_same_boundary_order_swap_fails_contract_on_ordering(tmp_path):
    # Same-tick demote,promote vs promote,demote: identical event multiset,
    # different run (replay follows candidate stream order). A
    # self-consistent promote-then-demote stream must fail the canonical
    # contract specifically on ordering.
    from simval.ontos_gravity import GravityContract, verify_stream_gravity

    order, _ = _region_occupancy(42, 4, 3)
    region = order[0]
    rx, ry = region % 2, region // 2
    contract = GravityContract.from_metadata(
        {"bodies": 4, "ticks": 8, "events": [[2, rx, ry, 0], [2, rx, ry, 1]]}
    )

    canonical = tmp_path / "canonical.stream"
    _emit_gravity_stream(canonical, 42, 4, [(2, region, 0), (2, region, 1)], 8)
    assert verify_stream_gravity(canonical, 42, expected=contract)["mismatch_count"] == 0

    swapped = tmp_path / "swapped.stream"
    _emit_gravity_stream(swapped, 42, 4, [(2, region, 1), (2, region, 0)], 8)
    summary = verify_stream_gravity(swapped, 42, expected=contract)
    assert any(m["field"] == "contract_event_order" for m in summary["mismatches"]), (
        summary["mismatches"]
    )


def test_observer_boundary_order_swap_fails_contract_and_zoom_policy(tmp_path):
    # The observer corpus boundary before tick 289 carries two policy
    # events (regions 2 and 3). Swapping the same-size records preserves
    # the multiset and the framing — only the order changes — and must
    # fail both the zoom-policy check and the verifier contract.
    import shutil

    from simval.ontos_gravity import (
        GravityContract,
        check_zoom_policy,
        parse_stream_v2,
        verify_stream_gravity,
    )

    run = tmp_path / "ontos_run"
    shutil.copytree(EXAMPLES / "observer", run)
    data = bytearray((run / "ontos.stream").read_bytes())
    level_offsets = [off for off, tag in _level_record_offsets(bytes(data)) if tag == 4]
    # The last two RegionLevel records are the tick-289 pair.
    a, b = level_offsets[-2], level_offsets[-1]
    rec_a = bytes(data[a : a + 10])
    rec_b = bytes(data[b : b + 10])
    swapped = bytes(data[:a]) + rec_b + rec_a + bytes(data[b + 10 :])
    (run / "ontos.stream").write_bytes(swapped)

    _, records = parse_stream_v2(run / "ontos.stream")
    result = check_zoom_policy(records, 5, 777)
    assert not result.passed
    assert result.detail["order"]["note"].startswith("boundary RegionLevel records")

    meta = json.loads((EXAMPLES / "observer" / "ontos.json").read_text())
    summary = verify_stream_gravity(
        run / "ontos.stream", 5, expected=GravityContract.from_metadata(meta)
    )
    assert any(m["field"] == "contract_event_order" for m in summary["mismatches"])
