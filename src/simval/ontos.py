"""Ontos adapter: independent Phase-0 reference implementation and stream verifier.

Implements the Ontos stream format and simulation rules strictly from
ontos docs/STREAM_SPEC.md (the normative contract) and verifies recorded
ontos streams against this reimplementation. Pure stdlib, no numpy —
`python3 -m simval.ontos <stream> <seed>` must run on a bare interpreter.
The engine adapter (numpy-adjacent plumbing) lives in simval.ontos_eng.

The reference deliberately shares no code with the Rust simulator: any
divergence between the two is a finding, not a nuisance.
"""
from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

from simval.result import DiagnosticResult

REGION_FINE = 64
REGIONS_PER_AXIS = 2
COARSE_FACTOR = 2
WORLD = REGIONS_PER_AXIS * REGION_FINE
FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MAGIC = b"ONTO"


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET_BASIS
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


@dataclass
class Region:
    level: str  # "fine" | "coarse"
    cells: bytearray

    @classmethod
    def fine(cls) -> "Region":
        return cls("fine", bytearray(REGION_FINE * REGION_FINE))

    @classmethod
    def coarse(cls) -> "Region":
        n = REGION_FINE // COARSE_FACTOR
        return cls("coarse", bytearray(n * n))

    def axis(self) -> int:
        return REGION_FINE if self.level == "fine" else REGION_FINE // COARSE_FACTOR

    def population(self) -> int:
        return sum(self.cells)

    def hash(self) -> int:
        level_byte = 0 if self.level == "coarse" else 1
        return fnv1a64(bytes([level_byte]) + bytes(self.cells))


@dataclass
class ReferenceWorld:
    seed: int
    tick: int = 0
    regions: list = field(default_factory=lambda: [Region.fine() for _ in range(4)])

    def region_index(self, rx: int, ry: int) -> int:
        return ry * REGIONS_PER_AXIS + rx

    def set_fine(self, fx: int, fy: int, alive: bool) -> None:
        rx, ry = fx // REGION_FINE, fy // REGION_FINE
        region = self.regions[self.region_index(rx, ry)]
        if region.level == "fine":
            idx = (fy % REGION_FINE) * REGION_FINE + (fx % REGION_FINE)
        else:
            n = region.axis()
            idx = ((fy % REGION_FINE) // COARSE_FACTOR) * n + ((fx % REGION_FINE) // COARSE_FACTOR)
        region.cells[idx] = 1 if alive else 0

    def seed_r_pentomino(self) -> None:
        c = WORLD // 2
        for dx, dy in ((0, 0), (1, 0), (0, 1), (-1, 1), (0, 2)):
            self.set_fine((c + dx) % WORLD, (c + dy) % WORLD, True)

    def read(self, fx: int, fy: int) -> int:
        fx, fy = fx % WORLD, fy % WORLD
        rx, ry = fx // REGION_FINE, fy // REGION_FINE
        region = self.regions[self.region_index(rx, ry)]
        if region.level == "fine":
            return region.cells[(fy % REGION_FINE) * REGION_FINE + (fx % REGION_FINE)]
        n = region.axis()
        cx, cy = (fx % REGION_FINE) // COARSE_FACTOR, (fy % REGION_FINE) // COARSE_FACTOR
        return region.cells[cy * n + cx]

    def read_block(self, cx: int, cy: int) -> int:
        coarse_world = WORLD // COARSE_FACTOR
        cx, cy = cx % coarse_world, cy % coarse_world
        rx, ry = (cx * COARSE_FACTOR) // REGION_FINE, (cy * COARSE_FACTOR) // REGION_FINE
        region = self.regions[self.region_index(rx, ry)]
        if region.level == "coarse":
            n = region.axis()
            return region.cells[(cy % n) * n + (cx % n)]
        base_fx, base_fy = cx * COARSE_FACTOR, cy * COARSE_FACTOR
        value = 0
        for dy in range(COARSE_FACTOR):
            for dx in range(COARSE_FACTOR):
                value |= self.read(base_fx + dx, base_fy + dy)
        return value

    def expansion_pick(self, gx: int, gy: int) -> int:
        payload = struct.pack("<QII", self.seed, gx, gy)
        return fnv1a64(payload) % 4

    def set_level(self, rx: int, ry: int, level: str) -> None:
        region = self.regions[self.region_index(rx, ry)]
        if region.level == level:
            return
        if level == "fine":
            fine = Region.fine()
            n = region.axis()
            offsets = ((0, 0), (1, 0), (0, 1), (1, 1))
            for cy in range(n):
                for cx in range(n):
                    if region.cells[cy * n + cx] == 0:
                        continue
                    gx = rx * REGION_FINE + cx * COARSE_FACTOR
                    gy = ry * REGION_FINE + cy * COARSE_FACTOR
                    dx, dy = offsets[self.expansion_pick(gx, gy)]
                    fine.cells[(cy * COARSE_FACTOR + dy) * REGION_FINE + (cx * COARSE_FACTOR + dx)] = 1
            self.regions[self.region_index(rx, ry)] = fine
        else:
            coarse = Region.coarse()
            n = coarse.axis()
            for cy in range(n):
                for cx in range(n):
                    alive = 0
                    for dy in range(COARSE_FACTOR):
                        for dx in range(COARSE_FACTOR):
                            alive |= region.cells[
                                (cy * COARSE_FACTOR + dy) * REGION_FINE + (cx * COARSE_FACTOR + dx)
                            ]
                    coarse.cells[cy * n + cx] = alive
            self.regions[self.region_index(rx, ry)] = coarse

    def region_key(self, fx: int, fy: int) -> tuple:
        rx, ry = fx // REGION_FINE, fy // REGION_FINE
        if self.regions[self.region_index(rx, ry)].level == "fine":
            return (0, fx, fy)
        return (1, fx // COARSE_FACTOR, fy // COARSE_FACTOR)

    def fine_neighbors(self, gx: int, gy: int) -> int:
        count = 0
        seen = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (gx + dx) % WORLD, (gy + dy) % WORLD
                if self.read(nx, ny) == 0:
                    continue
                key = self.region_key(nx, ny)
                if key in seen:
                    continue
                seen.add(key)
                count += 1
        return count

    def coarse_neighbors(self, gx: int, gy: int) -> int:
        coarse_world = WORLD // COARSE_FACTOR
        count = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                count += self.read_block((gx + dx) % coarse_world, (gy + dy) % coarse_world)
        return count

    def step(self) -> None:
        next_regions = []
        for index, region in enumerate(self.regions):
            rx, ry = index % REGIONS_PER_AXIS, index // REGIONS_PER_AXIS
            nxt = Region.fine() if region.level == "fine" else Region.coarse()
            if region.level == "fine":
                for fy in range(REGION_FINE):
                    for fx in range(REGION_FINE):
                        gx, gy = rx * REGION_FINE + fx, ry * REGION_FINE + fy
                        alive = region.cells[fy * REGION_FINE + fx] == 1
                        n = self.fine_neighbors(gx, gy)
                        nxt.cells[fy * REGION_FINE + fx] = 1 if (alive and n in (2, 3)) or (not alive and n == 3) else 0
            else:
                n_axis = region.axis()
                for cy in range(n_axis):
                    for cx in range(n_axis):
                        gx = rx * (REGION_FINE // COARSE_FACTOR) + cx
                        gy = ry * (REGION_FINE // COARSE_FACTOR) + cy
                        alive = region.cells[cy * n_axis + cx] == 1
                        n = self.coarse_neighbors(gx, gy)
                        nxt.cells[cy * n_axis + cx] = 1 if (alive and n in (2, 3)) or (not alive and n == 3) else 0
            next_regions.append(nxt)
        self.regions = next_regions
        self.tick += 1

    def population(self) -> int:
        return sum(region.population() for region in self.regions)

    def region_hash(self, rx: int, ry: int) -> int:
        return self.regions[self.region_index(rx, ry)].hash()

    def world_hash(self) -> int:
        payload = struct.pack("<Q", self.tick)
        for ry in range(REGIONS_PER_AXIS):
            for rx in range(REGIONS_PER_AXIS):
                payload += struct.pack("<Q", self.region_hash(rx, ry))
        return fnv1a64(payload)


@dataclass
class StreamHeader:
    world_w: int
    world_h: int
    version: int


_V1_REGION_ORDER = ((0, 0), (1, 0), (0, 1), (1, 1))


def _check_region_xy(tag: int, rx: int, ry: int, offset: int) -> None:
    if rx not in (0, 1) or ry not in (0, 1):
        raise ValueError(
            f"noncanonical region coordinates ({rx},{ry}) in record tag {tag} at offset {offset}: "
            "rx and ry must each be 0 or 1"
        )


def parse_stream(path) -> tuple:
    """Strict version-1 parser (spec section 9 emission contract).

    Grammar: RegionLevel records only between tick frames (a dangling level
    record at EOF is rejected); each frame is exactly TickHeader, Snapshot,
    RegionState for regions (0,0), (1,0), (0,1), (1,1) in that order;
    timestamped records must repeat the open frame's tick. Anything else —
    duplicates, omissions, records outside an open frame, non-monotonic
    ticks, a zero-record stream — is a parse error.
    """
    data = Path(path).read_bytes()
    if len(data) < 16 or data[:4] != MAGIC:
        raise ValueError("not an ontos stream: bad magic")
    version, world_w, world_h = struct.unpack_from("<III", data, 4)
    if version != 1:
        raise ValueError(f"unsupported ontos stream version {version}")
    records = []
    offset = 16
    n = len(data)
    # state: "boundary" (between frames) | "snapshot" | ("region", i)
    state = "boundary"
    last_tick = 0
    pending_boundary = False
    while offset < n:
        rec_off = offset
        tag = data[offset]
        offset += 1
        if tag == 1:
            if state != "boundary":
                raise ValueError(
                    f"TickHeader at offset {rec_off} inside an incomplete tick frame (tick {last_tick})"
                )
            (tick,) = struct.unpack_from("<Q", data, offset)
            offset += 8
            if tick != last_tick + 1:
                raise ValueError(
                    f"non-consecutive TickHeader tick {tick} at offset {rec_off}: expected {last_tick + 1}"
                )
            last_tick = tick
            pending_boundary = False
            state = "snapshot"
            records.append(("tick", tick))
        elif tag == 2:
            if state != "snapshot":
                raise ValueError(f"Snapshot outside an open tick frame at offset {rec_off}")
            (population,) = struct.unpack_from("<Q", data, offset)
            offset += 8
            state = ("region", 0)
            records.append(("snapshot", population))
        elif tag == 3:
            if state == "boundary":
                raise ValueError(f"CellFlipped outside an open tick frame at offset {rec_off}")
            tick, x, y = struct.unpack_from("<QII", data, offset)
            offset += 16
            if tick != last_tick:
                raise ValueError(
                    f"CellFlipped tick {tick} does not match the open frame tick {last_tick} at offset {rec_off}"
                )
            if x >= world_w or y >= world_h:
                raise ValueError(
                    f"CellFlipped coordinates ({x},{y}) outside the {world_w}x{world_h} world at offset {rec_off}"
                )
            records.append(("flip", tick, x, y))
        elif tag == 4:
            if state != "boundary":
                raise ValueError(f"RegionLevel inside a tick frame at offset {rec_off}")
            region_x, region_y, level = struct.unpack_from("<IIB", data, offset)
            offset += 9
            _check_region_xy(4, region_x, region_y, rec_off)
            if level not in (0, 1):
                raise ValueError(
                    f"invalid version-1 region level {level} at offset {rec_off}: expected 0 (coarse) or 1 (fine)"
                )
            pending_boundary = True
            records.append(("level", region_x, region_y, level))
        elif tag == 5:
            if not (isinstance(state, tuple) and state[0] == "region"):
                raise ValueError(f"RegionState outside an open tick frame at offset {rec_off}")
            tick, rx, ry, level, population, rhash = struct.unpack_from("<QIIBQQ", data, offset)
            offset += 33
            _check_region_xy(5, rx, ry, rec_off)
            expected = _V1_REGION_ORDER[state[1]]
            if (rx, ry) != expected:
                raise ValueError(
                    f"RegionState for region ({rx},{ry}) at offset {rec_off}: expected region "
                    f"{expected} next (order/duplicate violation)"
                )
            if tick != last_tick:
                raise ValueError(
                    f"RegionState tick {tick} does not match the open frame tick {last_tick} at offset {rec_off}"
                )
            if level not in (0, 1):
                raise ValueError(
                    f"invalid version-1 region-state level {level} at offset {rec_off}: expected 0 or 1"
                )
            nxt = state[1] + 1
            state = "boundary" if nxt == 4 else ("region", nxt)
            records.append(("state", tick, rx, ry, level, population, rhash))
        else:
            raise ValueError(f"unknown record tag {tag} at offset {rec_off}")
    if state != "boundary":
        raise ValueError(
            "stream ends inside an incomplete tick frame "
            f"({state if not isinstance(state, tuple) else state[0] + ' ' + str(state[1])}, tick {last_tick})"
        )
    if pending_boundary:
        raise ValueError("dangling RegionLevel record(s) at end of stream with no following TickHeader")
    if last_tick == 0:
        raise ValueError("stream carries no tick frames")
    return StreamHeader(world_w, world_h, version), records


def verify_stream(path, seed: int) -> dict:
    """Replay a stream against the independent reference implementation."""
    header, records = parse_stream(path)
    if header.world_w != WORLD or header.world_h != WORLD:
        raise ValueError(f"unsupported world size {header.world_w}x{header.world_h}")
    world = ReferenceWorld(seed=seed)
    world.seed_r_pentomino()

    mismatches = []
    records_compared = 0
    last_tick = 0
    tick_monotonic = True

    for record in records:
        if record[0] == "level":
            _, rx, ry, level = record
            world.set_level(rx, ry, "fine" if level == 1 else "coarse")
        elif record[0] == "tick":
            _, tick = record
            if tick != last_tick + 1:
                tick_monotonic = False
            last_tick = tick
            world.step()
            if tick != world.tick:
                mismatches.append({"tick": tick, "region": None, "field": "tick",
                                   "expected": tick, "actual": world.tick})
        elif record[0] == "snapshot":
            _, population = record
            records_compared += 1
            if population != world.population():
                mismatches.append({"tick": world.tick, "region": None, "field": "population",
                                   "expected": population, "actual": world.population()})
        elif record[0] == "state":
            _, tick, rx, ry, level, population, rhash = record
            records_compared += 1
            region = world.regions[world.region_index(rx, ry)]
            actual_level = 1 if region.level == "fine" else 0
            if level != actual_level:
                mismatches.append({"tick": tick, "region": (rx, ry), "field": "level",
                                   "expected": level, "actual": actual_level})
            if population != region.population():
                mismatches.append({"tick": tick, "region": (rx, ry), "field": "population",
                                   "expected": population, "actual": region.population()})
            if rhash != region.hash():
                mismatches.append({"tick": tick, "region": (rx, ry), "field": "hash",
                                   "expected": rhash, "actual": region.hash()})

    return {
        "ticks_verified": last_tick,
        "records_compared": records_compared,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "tick_monotonic": tick_monotonic,
        "final_population": world.population(),
        "final_world_hash": world.world_hash(),
    }


def check_reference_match(summary: dict, *, threshold: float = 0.0) -> DiagnosticResult:
    count = summary["mismatch_count"]
    # Independent of mismatch_count: a stream that compared zero (or
    # tick-free) records must never read as a pass.
    insufficient = summary["records_compared"] < 1 or summary["ticks_verified"] < 1
    return DiagnosticResult(
        name="ontos_reference_match",
        passed=count <= threshold and not insufficient,
        threshold=float(threshold),
        value=float(count),
        detail={
            "ticks_verified": summary["ticks_verified"],
            "records_compared": summary["records_compared"],
            "first_mismatches": summary["mismatches"],
            "insufficient_records": insufficient,
        },
    )


def check_tick_monotonicity(summary: dict) -> DiagnosticResult:
    ok = summary["tick_monotonic"]
    return DiagnosticResult(
        name="ontos_tick_monotonicity",
        passed=ok,
        threshold=1.0,
        value=1.0 if ok else 0.0,
        detail={"ticks_verified": summary["ticks_verified"]},
    )


def life_contract_problems(meta: dict, summary: dict, records: list) -> list[str]:
    """Validate a life run's ontos.json metadata against the stream.

    Shared by the engine adapter and the standalone CLI so both paths
    enforce the same run contract (audit ONT-007/009). Untimed life events
    are initialization semantics: every requested event must sit in the
    boundary before TickHeader 1 — an occurrence at any later boundary is
    a contract violation even though the event identity and count match.
    The comparison is ORDERED (audit ONT-020): the producer initializes in
    demotes-then-promotes CLI order and replay applies boundary records in
    stream order, so a promote->demote stream is a different run than the
    requested demote->promote even though the event multisets match."""
    problems = []
    meta_ticks = meta.get("ticks")
    if meta_ticks is not None and int(meta_ticks) != summary["ticks_verified"]:
        problems.append(
            f"ontos.json ticks={meta_ticks} but stream verified {summary['ticks_verified']} ticks"
        )
    placements: list[tuple[int, tuple]] = []
    pending: list[tuple] = []
    for record in records:
        if record[0] == "level":
            pending.append((record[1], record[2], record[3]))
        elif record[0] == "tick":
            for key in pending:
                placements.append((record[1], key))
            pending.clear()
    if "events" in meta:
        from collections import Counter

        want_seq: list[tuple] = []
        want = Counter()
        for ev in meta["events"]:
            if len(ev) != 3 or ev[0] not in _LIFE_EVENT_KINDS:
                problems.append(f"ontos.json life events must be [kind, rx, ry], got {list(ev)}")
                continue
            kind, rx, ry = ev[0], int(ev[1]), int(ev[2])
            if rx not in (0, 1) or ry not in (0, 1):
                problems.append(f"ontos.json life event region ({rx},{ry}) outside the 2x2 grid")
                continue
            key = (rx, ry, _LIFE_EVENT_KINDS[kind])
            want_seq.append(key)
            want[key] += 1
        got_seq = [key for _, key in placements]
        got = Counter(got_seq)
        for key in sorted((want - got).elements()):
            problems.append(f"missing scheduled event {key}")
        for key in sorted((got - want).elements()):
            problems.append(f"unscheduled event {key} in stream")
        if want == got and got_seq != want_seq:
            i = next(idx for idx in range(len(got_seq)) if got_seq[idx] != want_seq[idx])
            problems.append(
                f"event order mismatch at initialization position {i}: requested "
                f"{want_seq[i]} but the stream carries {got_seq[i]} (boundary events "
                "are order-sensitive: replay applies them in stream order)"
            )
        for tick, key in placements:
            if tick != 1:
                problems.append(
                    f"event {key} at the boundary before tick {tick}: "
                    "requested untimed life events are initialization semantics "
                    "and must precede tick 1"
                )
    return problems


_LIFE_EVENT_KINDS = {"demote": 0, "promote": 1}


def check_life_contract(meta: dict, summary: dict, records: list) -> DiagnosticResult | None:
    """Engine-adapter wrapper: a failed life run contract as a DiagnosticResult."""
    problems = life_contract_problems(meta, summary, records)
    if not problems:
        return None
    return DiagnosticResult(
        name="ontos_run_contract",
        passed=False,
        threshold=0.0,
        value=float(len(problems)),
        detail={"problems": problems[:10]},
    )


def _main(argv=None) -> int:
    from simval.ontos_gravity import _main as gravity_aware_main

    return gravity_aware_main(argv)


if __name__ == "__main__":
    raise SystemExit(_main())
