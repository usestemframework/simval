"""Ontos gravity reference: independent version-2 stream verifier.

Implements the gravity epoch of the ontos stream spec (ontos
docs/STREAM_SPEC.md Part II) strictly from the spec text and verifies
recorded streams against this reimplementation. Pure stdlib. Float ops
stay in the spec closure (+ - * / sqrt) with fixed order, so this
reference is bit-compatible with the Rust simulator.
"""
from __future__ import annotations

import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from simval.ontos import MAGIC, fnv1a64, parse_stream
from simval.result import DiagnosticResult

G = 1.0
EPS2 = 1.0
DT = 1.0 / 1024.0
WINDOW = 32
DEGREE = 8
SAMPLES = 33
UNMANAGED = 255
CONTACT_R = 2.0
MONOPOLE_BASE = 0xFF000000
WALL_BASE = 0xFFFFFF00
TWO_POW_NEG64 = 2.0**-64


class SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)


def initial_conditions(seed: int, count: int) -> list:
    rng = SplitMix64(seed)
    bodies = []
    for i in range(count):
        u0 = rng.next()
        u1 = rng.next()
        u2 = rng.next()
        u3 = rng.next()
        u4 = rng.next()
        bodies.append(
            {
                "id": i,
                "mass": 0.5 + u0 * TWO_POW_NEG64 * 2.0,
                "x": 32.0 + u1 * TWO_POW_NEG64 * 64.0,
                "y": 32.0 + u2 * TWO_POW_NEG64 * 64.0,
                "vx": (u3 * TWO_POW_NEG64 - 0.5) * 0.5,
                "vy": (u4 * TWO_POW_NEG64 - 0.5) * 0.5,
            }
        )
    return bodies


def test_initial_conditions(profile: str, seed: int, count: int) -> list:
    """Test-only corpus initial conditions (mirror of ontos
    corpus_initial_conditions; see ontos docs/DESIGN.md corpus
    coverage). wallshot: body i targets wall i % 4, starting near it and
    inbound at 2..5. coarsehit (count 8): bodies 0..3 are interceptors
    outside the region-3 box aimed at a slow target cluster (bodies
    4..7) over shared y lanes. Interceptors carry the smaller ids so
    the corpus exercises the fine-low static arm (fine i, coarse j) of
    the section 24 sweep; the reversed id order (coarse contactant
    below the fine body) is pinned by unit regression instead. Same
    five SplitMix64 draws per body as initial_conditions, so masses
    match the spec ICs of the same seed.
    """
    rng = SplitMix64(seed)
    bodies = []
    for i in range(count):
        u0 = rng.next()
        u1 = rng.next()
        u2 = rng.next()
        u3 = rng.next()
        u4 = rng.next()
        mass = 0.5 + u0 * TWO_POW_NEG64 * 2.0
        along = 16.0 + u1 * TWO_POW_NEG64 * 96.0
        off = u2 * TWO_POW_NEG64 * 2.0
        speed = 2.0 + u3 * TWO_POW_NEG64 * 3.0
        drift = (u4 * TWO_POW_NEG64 - 0.5) * 0.5
        if profile == "wallshot":
            w = i % 4
            if w == 0:
                x, y, vx, vy = 2.0 + off, along, 0.0 - speed, drift
            elif w == 1:
                x, y, vx, vy = 124.0 + off, along, speed, drift
            elif w == 2:
                x, y, vx, vy = along, 2.0 + off, drift, 0.0 - speed
            else:
                x, y, vx, vy = along, 124.0 + off, drift, speed
        elif profile == "coarsehit":
            lane = 77.0 + 8.0 * (i % 4) + u2 * TWO_POW_NEG64 * 2.0
            if i < 4:
                x, y, vx, vy = (
                    56.0 + u1 * TWO_POW_NEG64 * 4.0,
                    lane,
                    56.0 + u3 * TWO_POW_NEG64 * 16.0,
                    (u4 * TWO_POW_NEG64 - 0.5) * 0.5,
                )
            else:
                x, y, vx, vy = (
                    84.0 + u1 * TWO_POW_NEG64 * 4.0,
                    lane,
                    (u3 * TWO_POW_NEG64 - 0.5) * 0.5,
                    (u4 * TWO_POW_NEG64 - 0.5) * 0.5,
                )
        else:
            raise ValueError(f"unknown corpus profile {profile}")
        bodies.append({"id": i, "mass": mass, "x": x, "y": y, "vx": vx, "vy": vy})
    return bodies


def region_at(x: float, y: float) -> int:
    if x < 0.0 or x >= 128.0 or y < 0.0 or y >= 128.0:
        return UNMANAGED
    rx = int(x / 64.0)
    ry = int(y / 64.0)
    return ry * 2 + rx


def clenshaw(c, s: float) -> float:
    b1 = 0.0
    b2 = 0.0
    for j in range(DEGREE, 0, -1):
        b0 = c[j] + 2.0 * s * b1 - b2
        b2 = b1
        b1 = b0
    return c[0] + s * b1 - b2


def cheb_table():
    t = [[0.0] * SAMPLES for _ in range(DEGREE + 1)]
    w = [1.0] * SAMPLES
    w[0] = 0.5
    w[SAMPLES - 1] = 0.5
    for k in range(SAMPLES):
        s = -1.0 + k / 16.0
        t[0][k] = 1.0
        t[1][k] = s
        for j in range(2, DEGREE + 1):
            t[j][k] = 2.0 * s * t[j - 1][k] - t[j - 2][k]
    return t, w


CHEB_T, CHEB_W = cheb_table()


def project(ys):
    g = [[0.0] * (DEGREE + 1) for _ in range(DEGREE + 1)]
    for j in range(DEGREE + 1):
        for l in range(DEGREE + 1):
            total = 0.0
            for k in range(SAMPLES):
                total += CHEB_W[k] * CHEB_T[j][k] * CHEB_T[l][k]
            g[j][l] = total
    b = [0.0] * (DEGREE + 1)
    for j in range(DEGREE + 1):
        total = 0.0
        for k in range(SAMPLES):
            total += CHEB_W[k] * ys[k] * CHEB_T[j][k]
        b[j] = total
    return cholesky_solve(g, b)


def cholesky_solve(g, b):
    n = DEGREE + 1
    l = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = g[i][j]
            for k in range(j):
                total -= l[i][k] * l[j][k]
            if i == j:
                l[i][j] = math.sqrt(total)
            else:
                l[i][j] = total / l[j][j]
    z = [0.0] * n
    for i in range(n):
        total = b[i]
        for k in range(i):
            total -= l[i][k] * z[k]
        z[i] = total / l[i][i]
    c = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = z[i]
        for k in range(i + 1, n):
            total -= l[k][i] * c[k]
        c[i] = total / l[i][i]
    return c


def subset_energy(bodies) -> float:
    """Section 15 energy over a body list (KE in id order, then PE i<j)."""
    ke = 0.0
    for b in bodies:
        ke += 0.5 * b["mass"] * (b["vx"] * b["vx"] + b["vy"] * b["vy"])
    pe = 0.0
    n = len(bodies)
    for i in range(n):
        for j in range(i + 1, n):
            dx = bodies[j]["x"] - bodies[i]["x"]
            dy = bodies[j]["y"] - bodies[i]["y"]
            s2 = dx * dx + dy * dy + EPS2
            pe -= bodies[i]["mass"] * bodies[j]["mass"] / math.sqrt(s2)
    return ke + pe


def shell_count(n: int) -> int:
    """Spec section 25: S = min(4, max(1, n div 3)) for n >= 1."""
    return min(4, max(1, n // 3))


def shell_assignment(radii):
    """Spec section 25: rank shells — sort by (radius, id), split into S
    equal-count groups (first n mod S groups one larger)."""
    n = len(radii)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (radii[i], i))
    s = shell_count(n)
    q, rem = divmod(n, s)
    shells = [0] * n
    pos = 0
    for k in range(s):
        for _ in range(q + (1 if k < rem else 0)):
            shells[order[pos]] = k
            pos += 1
    return shells


class GravityWorld:
    def __init__(self, seed: int, count: int, profile: str | None = None) -> None:
        self.seed = seed
        if profile is None:
            self.bodies = initial_conditions(seed, count)
        else:
            self.bodies = test_initial_conditions(profile, seed, count)
        self.coarse = [None] * count
        self.body_region = [UNMANAGED] * count
        self.region_coarse = [False] * 4
        self.region_window_deadline: list[int | None] = [None] * 4
        self.region_collapsed = [None] * 4
        self.body_collapsed = [None] * count
        self.reconstructed: set[int] = set()
        self.last_collapses = []
        self.last_expansion = None
        self.expand_count = 0
        self.events: dict[int, list[tuple[int, int]]] = {}
        self.tick = 0
        self.mp_enabled = True
        self.radial_enabled = False
        self.shells_enabled = False
        self.contacts = False
        self.contact_params = False
        self.contact_armed = False
        self.restitution = 0.0
        self.friction = 0.0
        self.walls = False
        self.touching: set[tuple[int, int]] = set()
        self.last_contacts: list[dict] = []
        px = 0.0
        py = 0.0
        for b in self.bodies:
            px += b["mass"] * b["vx"]
            py += b["mass"] * b["vy"]
        self.px = px
        self.py = py

    def schedule(self, tick: int, region: int, level: int) -> None:
        self.events.setdefault(tick, []).append((region, level))

    def _state_at(self, i: int, t: int) -> dict:
        collapsed = self.body_collapsed[i]
        if collapsed is not None:
            rec = self.region_collapsed[collapsed]
            jx, jy = rec["jitter"][i]
            return {
                "id": i,
                "mass": self.bodies[i]["mass"],
                "x": rec["com_x"] + jx,
                "y": rec["com_y"] + jy,
                "vx": rec["vcom_x"],
                "vy": rec["vcom_y"],
            }
        b = dict(self.bodies[i])
        fit = self.coarse[i]
        if fit is not None:
            s = -1.0 + (t - fit["t0"]) / 16.0
            b["x"] = clenshaw(fit["c"][0], s)
            b["y"] = clenshaw(fit["c"][1], s)
            b["vx"] = clenshaw(fit["c"][2], s)
            b["vy"] = clenshaw(fit["c"][3], s)
        return b

    def _leapfrog(self, bodies):
        half = DT * 0.5
        ax, ay = self._accumulate(bodies)
        for i, b in enumerate(bodies):
            b["vx"] += ax[i] * half
            b["vy"] += ay[i] * half
        for b in bodies:
            b["x"] += b["vx"] * DT
            b["y"] += b["vy"] * DT
        ax, ay = self._accumulate(bodies)
        for i, b in enumerate(bodies):
            b["vx"] += ax[i] * half
            b["vy"] += ay[i] * half

    @staticmethod
    def _accumulate(bodies):
        n = len(bodies)
        ax = [0.0] * n
        ay = [0.0] * n
        for i in range(n):
            for j in range(i + 1, n):
                dx = bodies[j]["x"] - bodies[i]["x"]
                dy = bodies[j]["y"] - bodies[i]["y"]
                s2 = dx * dx + dy * dy + EPS2
                inv3 = 1.0 / (s2 * math.sqrt(s2))
                fx = G * inv3 * dx
                fy = G * inv3 * dy
                ax[i] += bodies[j]["mass"] * fx
                ay[i] += bodies[j]["mass"] * fy
                ax[j] -= bodies[i]["mass"] * fx
                ay[j] -= bodies[i]["mass"] * fy
        return ax, ay

    def _pre_integrate(self, bodies):
        samples = [[dict(b) for _ in range(SAMPLES)] for b in bodies]
        cur = [dict(b) for b in bodies]
        for k in range(1, SAMPLES):
            self._leapfrog(cur)
            for i, b in enumerate(cur):
                samples[i][k] = dict(b)
        return samples

    def _fit_members(self, members, t0):
        subset = [self._state_at(i, t0) for i in members]
        samples = self._pre_integrate(subset)
        for slot, i in enumerate(members):
            fits = [
                project([s["x"] for s in samples[slot]]),
                project([s["y"] for s in samples[slot]]),
                project([s["vx"] for s in samples[slot]]),
                project([s["vy"] for s in samples[slot]]),
            ]
            self.coarse[i] = {"c": fits, "t0": t0}

    def _demote(self, region: int, t0: int) -> None:
        x0 = (region % 2) * 64.0
        y0 = (region // 2) * 64.0
        # Section 26 materialization applies to re-demotion too (audit
        # ONT-012): every existing window fit of the region is
        # materialized and discarded before membership is re-evaluated,
        # exactly like collapse-on-coarse. An out-of-box member would
        # otherwise keep its old fit while the region deadline advances
        # past that fit's own [t0, t0 + WINDOW] validity end.
        for i in range(len(self.bodies)):
            if self.coarse[i] is not None and self.body_region[i] == region:
                self.bodies[i] = self._state_at(i, t0)
                self.coarse[i] = None
                self.body_region[i] = UNMANAGED
        members = [
            i
            for i in range(len(self.bodies))
            if self.body_collapsed[i] is None
            and (lambda b: b["x"] >= x0 and b["x"] < x0 + 64.0 and b["y"] >= y0 and b["y"] < y0 + 64.0)(
                self._state_at(i, t0)
            )
        ]
        self.region_coarse[region] = True
        # Section 14 owes every demotion its t0 + WINDOW re-fit, empty
        # ones included (audit ONT-011): the per-region deadline is the
        # only carrier of that obligation, independent of member Fits —
        # the empty re-fit path is what returns the region to Fine.
        self.region_window_deadline[region] = t0 + WINDOW
        if not members:
            return
        self._fit_members(members, t0)
        for i in members:
            self.body_region[i] = region

    def _promote(self, region: int, t: int) -> None:
        for i in range(len(self.bodies)):
            if self.coarse[i] is not None and self.body_region[i] == region:
                self.bodies[i] = self._state_at(i, t)
                self.coarse[i] = None
                self.body_region[i] = UNMANAGED
        self.region_coarse[region] = False
        self.region_window_deadline[region] = None

    def _collapse(self, region: int, t: int) -> None:
        x0 = (region % 2) * 64.0
        y0 = (region // 2) * 64.0
        # Spec section 26: collapse terminates the region's own window —
        # every coarse body of the region materializes its evaluation at
        # t (out-of-box bodies leave as unmanaged fine, refit exit rule).
        for i in range(len(self.bodies)):
            if self.coarse[i] is not None and self.body_region[i] == region:
                self.bodies[i] = self._state_at(i, t)
                self.coarse[i] = None
                self.body_region[i] = UNMANAGED
        self.region_coarse[region] = False
        # Spec section 26 membership: evaluated state at t, collapsed
        # bodies excluded (the section 14 demote rule).
        members = []
        for i in range(len(self.bodies)):
            if self.body_collapsed[i] is not None:
                continue
            b = self._state_at(i, t)
            if b["x"] >= x0 and b["x"] < x0 + 64.0 and b["y"] >= y0 and b["y"] < y0 + 64.0:
                members.append(i)
        for i in members:
            if self.coarse[i] is not None:
                self.bodies[i] = self._state_at(i, t)
                self.coarse[i] = None
                self.body_region[i] = UNMANAGED
        mass = 0.0
        mx = 0.0
        my = 0.0
        px = 0.0
        py = 0.0
        for i in members:
            b = self.bodies[i]
            mass += b["mass"]
            mx += b["mass"] * b["x"]
            my += b["mass"] * b["y"]
            px += b["mass"] * b["vx"]
            py += b["mass"] * b["vy"]
        if members:
            com_x = mx / mass
            com_y = my / mass
            vcom_x = px / mass
            vcom_y = py / mass
            qxx = 0.0
            qxy = 0.0
            qyy = 0.0
            for i in members:
                b = self.bodies[i]
                dx = b["x"] - com_x
                dy = b["y"] - com_y
                qxx += b["mass"] * dx * dx
                qxy += b["mass"] * dx * dy
                qyy += b["mass"] * dy * dy
            radii = []
            for i in members:
                b = self.bodies[i]
                dx = b["x"] - com_x
                dy = b["y"] - com_y
                radii.append(math.sqrt(dx * dx + dy * dy))
            shells = shell_assignment(radii)
            shell_bindings = [0.0, 0.0, 0.0, 0.0]
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    if shells[a] != shells[b]:
                        continue
                    ba = self.bodies[members[a]]
                    bb = self.bodies[members[b]]
                    dx = bb["x"] - ba["x"]
                    dy = bb["y"] - ba["y"]
                    shell_bindings[shells[a]] += (
                        ba["mass"] * bb["mass"] / math.sqrt(dx * dx + dy * dy + EPS2)
                    )
        else:
            com_x = 0.0
            com_y = 0.0
            vcom_x = 0.0
            vcom_y = 0.0
            mx = 0.0
            my = 0.0
            qxx = 0.0
            qxy = 0.0
            qyy = 0.0
            shell_bindings = [0.0, 0.0, 0.0, 0.0]
        binding = 0.0
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                ba = self.bodies[members[a]]
                bb = self.bodies[members[b]]
                dx = bb["x"] - ba["x"]
                dy = bb["y"] - ba["y"]
                binding += ba["mass"] * bb["mass"] / math.sqrt(dx * dx + dy * dy + EPS2)
        rng = SplitMix64(self.seed ^ ((region * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF))
        jitter = {}
        for i in members:
            jx = (rng.next() * TWO_POW_NEG64 - 0.5) * 8.0
            jy = (rng.next() * TWO_POW_NEG64 - 0.5) * 8.0
            jitter[i] = (jx, jy)
        spread = {}
        for i in members[:-1]:
            sx = (rng.next() * TWO_POW_NEG64 - 0.5) * 0.1
            sy = (rng.next() * TWO_POW_NEG64 - 0.5) * 0.1
            spread[i] = (sx, sy)
        syn = [
            {
                "id": i,
                "mass": self.bodies[i]["mass"],
                "x": com_x + jitter[i][0],
                "y": com_y + jitter[i][1],
                "vx": vcom_x,
                "vy": vcom_y,
            }
            for i in members
        ]
        rec = {
            "tick": t,
            "region": region,
            "count": len(members),
            "mass": mass,
            "com_x": com_x,
            "com_y": com_y,
            "vcom_x": vcom_x,
            "vcom_y": vcom_y,
            "px": px,
            "py": py,
            "energy": subset_energy([self.bodies[i] for i in members]) if members else 0.0,
            "syn_energy": subset_energy(syn),
            "members": members,
            "jitter": jitter,
            "spread": spread,
            "multipole": self.mp_enabled,
            "radial": self.radial_enabled,
            "shells": self.shells_enabled,
            "mx": mx,
            "my": my,
            "qxx": qxx,
            "qxy": qxy,
            "qyy": qyy,
            "binding": binding,
            "shell_bindings": shell_bindings,
        }
        self.region_collapsed[region] = rec
        self.last_collapses.append(rec)
        self.region_window_deadline[region] = None
        for i in members:
            self.body_collapsed[i] = region
            self.body_region[i] = region

    def _expand(self, region: int, t: int) -> None:
        rec = self.region_collapsed[region]
        members = rec["members"]
        states = []
        transformed = False
        sigma = 1.0
        fl = None
        shell_groups = []
        sigma_vertex = False
        if members:
            if rec["multipole"]:
                base = {i: rec["jitter"][i] for i in members}
                if len(members) >= 3:
                    swx = 0.0
                    swy = 0.0
                    for i in members:
                        swx += self.bodies[i]["mass"] * base[i][0]
                        swy += self.bodies[i]["mass"] * base[i][1]
                    wx = swx / rec["mass"]
                    wy = swy / rec["mass"]
                    dhat = {}
                    jxx = 0.0
                    jxy = 0.0
                    jyy = 0.0
                    for i in members:
                        m = self.bodies[i]["mass"]
                        dx = base[i][0] - wx
                        dy = base[i][1] - wy
                        jxx += m * dx * dx
                        jxy += m * dx * dy
                        jyy += m * dy * dy
                        dhat[i] = (dx, dy)
                    if jxx > 0.0 and rec["qxx"] > 0.0:
                        lj00 = math.sqrt(jxx)
                        lj10 = jxy / lj00
                        jjd = jyy - lj10 * lj10
                        if jjd > 0.0:
                            lq00 = math.sqrt(rec["qxx"])
                            lq10 = rec["qxy"] / lq00
                            qqd = rec["qyy"] - lq10 * lq10
                            if qqd > 0.0:
                                lj11 = math.sqrt(jjd)
                                lq11 = math.sqrt(qqd)
                                u00 = 1.0 / lj00
                                u11 = 1.0 / lj11
                                u10 = -(lj10 / (lj00 * lj11))
                                a00 = lq00 * u00
                                a11 = lq11 * u11
                                a10 = lq10 * u00 + lq11 * u10
                                for i in members:
                                    dx, dy = dhat[i]
                                    base[i] = (a00 * dx, a10 * dx + a11 * dy)
                                transformed = True
                if rec["shells"]:
                    fl, shell_groups = self._shell_scale(members, rec, base)
                    sigma, sigma_vertex = self._solve_sigma(members, rec, rec["energy"] + fl)
                elif rec["radial"]:
                    fl = self._radial_scale(members, rec, base)
                    sigma, _ = self._solve_sigma(members, rec, rec["energy"] + fl)
                sum_mx = 0.0
                sum_my = 0.0
                for i in members[:-1]:
                    self.bodies[i]["x"] = rec["com_x"] + base[i][0]
                    self.bodies[i]["y"] = rec["com_y"] + base[i][1]
                    sum_mx += self.bodies[i]["mass"] * self.bodies[i]["x"]
                    sum_my += self.bodies[i]["mass"] * self.bodies[i]["y"]
                last = members[-1]
                m_last = self.bodies[last]["mass"]
                self.bodies[last]["x"] = (rec["mx"] - sum_mx) / m_last
                self.bodies[last]["y"] = (rec["my"] - sum_my) / m_last
            else:
                for i in members:
                    jx, jy = rec["jitter"][i]
                    self.bodies[i]["x"] = rec["com_x"] + jx
                    self.bodies[i]["y"] = rec["com_y"] + jy
            sum_mvx = 0.0
            sum_mvy = 0.0
            for i in members[:-1]:
                sx, sy = rec["spread"][i]
                self.bodies[i]["vx"] = rec["vcom_x"] + sigma * sx
                self.bodies[i]["vy"] = rec["vcom_y"] + sigma * sy
                sum_mvx += self.bodies[i]["mass"] * self.bodies[i]["vx"]
                sum_mvy += self.bodies[i]["mass"] * self.bodies[i]["vy"]
            last = members[-1]
            m_last = self.bodies[last]["mass"]
            self.bodies[last]["vx"] = (rec["px"] - sum_mvx) / m_last
            self.bodies[last]["vy"] = (rec["py"] - sum_mvy) / m_last
            for i in members:
                self.body_collapsed[i] = None
                self.body_region[i] = UNMANAGED
                self.reconstructed.add(i)
            states = [dict(self.bodies[i]) for i in members]
        self.region_collapsed[region] = None
        self.region_window_deadline[region] = None
        self.expand_count += 1
        self.last_expansion = {
            "tick": t,
            "region": region,
            "target_px": rec["px"],
            "target_py": rec["py"],
            "bodies": [dict(b) for b in states],
            "multipole": bool(rec["multipole"]),
            "transformed": transformed,
            "radial": bool(rec["radial"]),
            "shells": bool(rec["shells"]),
            "shell_groups": [list(g) for g in shell_groups],
            "sigma_vertex": sigma_vertex,
            "target_shell_bindings": list(rec["shell_bindings"]),
            "target_com_x": rec["com_x"],
            "target_com_y": rec["com_y"],
            "target_mx": rec["mx"],
            "target_my": rec["my"],
            "target_qxx": rec["qxx"],
            "target_qxy": rec["qxy"],
            "target_qyy": rec["qyy"],
            "target_energy": rec["energy"],
            "target_binding": rec["binding"],
        }

    def _radial_scale(self, members, rec, base):
        """Spec section 23: binding-matched radial scale; returns F(lambda)."""
        if len(members) < 2:
            return 0.0
        pairs = []
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                w = self.bodies[members[a]]["mass"] * self.bodies[members[b]]["mass"]
                dx = base[members[b]][0] - base[members[a]][0]
                dy = base[members[b]][1] - base[members[a]][1]
                pairs.append((w, dx * dx + dy * dy))

        def f(lam):
            total = 0.0
            for w, d2 in pairs:
                total += w / math.sqrt(lam * lam * d2 + 1.0)
            return total

        target = rec["binding"]
        if target >= f(0.0):
            lam = 0.0
        else:
            hi = 1.0
            doublings = 0
            while f(hi) > target and doublings < 64:
                hi *= 2.0
                doublings += 1
            lo = 0.0
            for _ in range(128):
                mid = (lo + hi) * 0.5
                if f(mid) >= target:
                    lo = mid
                else:
                    hi = mid
            lam = (lo + hi) * 0.5
        for i in members:
            base[i] = (lam * base[i][0], lam * base[i][1])
        swx = 0.0
        swy = 0.0
        for i in members:
            m = self.bodies[i]["mass"]
            swx += m * base[i][0]
            swy += m * base[i][1]
        wx = swx / rec["mass"]
        wy = swy / rec["mass"]
        for i in members:
            base[i] = (base[i][0] - wx, base[i][1] - wy)
        return f(lam)

    def _shell_scale(self, members, rec, base):
        """Spec section 25: global binding scale then per-shell corrections.

        Returns (F_total, solve-time shell groups).
        """
        n = len(members)
        if n < 2:
            return 0.0, []
        all_pairs = []
        for a in range(n):
            for b in range(a + 1, n):
                w = self.bodies[members[a]]["mass"] * self.bodies[members[b]]["mass"]
                dx = base[members[b]][0] - base[members[a]][0]
                dy = base[members[b]][1] - base[members[a]][1]
                all_pairs.append((w, dx * dx + dy * dy))

        def solve(pairs, target):
            def f(lam):
                total = 0.0
                for w, d2 in pairs:
                    total += w / math.sqrt(lam * lam * d2 + 1.0)
                return total

            if target >= f(0.0):
                return 0.0
            hi = 1.0
            doublings = 0
            while f(hi) > target and doublings < 64:
                hi *= 2.0
                doublings += 1
            lo = 0.0
            for _ in range(128):
                mid = (lo + hi) * 0.5
                if f(mid) >= target:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) * 0.5

        # Section 25 pins the order: classify the base (step 2) before
        # the global scale (step 3). The mass-weighted mean and radii
        # come from the UNSCALED section 20 displacements and the
        # shells vector is retained for the per-shell solves — solving
        # and applying lambda first can flip rank near-ties under
        # binary64 rounding (scaled radii about the scaled mean are not
        # exactly the unscaled radii), changing shell membership.
        swx = 0.0
        swy = 0.0
        for i in members:
            m = self.bodies[i]["mass"]
            swx += m * base[i][0]
            swy += m * base[i][1]
        cx = swx / rec["mass"]
        cy = swy / rec["mass"]
        radii = []
        for i in members:
            dx = base[i][0] - cx
            dy = base[i][1] - cy
            radii.append(math.sqrt(dx * dx + dy * dy))
        shells = shell_assignment(radii)
        lam = solve(all_pairs, rec["binding"])
        for i in members:
            base[i] = (lam * base[i][0], lam * base[i][1])
        s = shell_count(n)
        pairs = [[] for _ in range(s)]
        for a in range(n):
            for b in range(a + 1, n):
                if shells[a] != shells[b]:
                    continue
                w = self.bodies[members[a]]["mass"] * self.bodies[members[b]]["mass"]
                dx = base[members[b]][0] - base[members[a]][0]
                dy = base[members[b]][1] - base[members[a]][1]
                pairs[shells[a]].append((w, dx * dx + dy * dy))
        mus = [1.0, 1.0, 1.0, 1.0]
        for k in range(s):
            target = rec["shell_bindings"][k]
            if not pairs[k] or target <= 0.0:
                continue
            mus[k] = solve(pairs[k], target)
        for slot, i in enumerate(members):
            mu = mus[shells[slot]]
            base[i] = (mu * base[i][0], mu * base[i][1])
        swx = 0.0
        swy = 0.0
        for i in members:
            m = self.bodies[i]["mass"]
            swx += m * base[i][0]
            swy += m * base[i][1]
        wx = swx / rec["mass"]
        wy = swy / rec["mass"]
        for i in members:
            base[i] = (base[i][0] - wx, base[i][1] - wy)
        ft = 0.0
        for a in range(n):
            for b in range(a + 1, n):
                w = self.bodies[members[a]]["mass"] * self.bodies[members[b]]["mass"]
                dx = base[members[b]][0] - base[members[a]][0]
                dy = base[members[b]][1] - base[members[a]][1]
                ft += w / math.sqrt(dx * dx + dy * dy + 1.0)
        groups = [[] for _ in range(s)]
        for slot, i in enumerate(members):
            groups[shells[slot]].append(i)
        return ft, groups

    def _synth_velocities(self, members, rec, sigma):
        out = {}
        sum_mv_x = 0.0
        sum_mv_y = 0.0
        for slot, i in enumerate(members):
            if slot + 1 < len(members):
                sx, sy = rec["spread"][i]
                vx = rec["vcom_x"] + sigma * sx
                vy = rec["vcom_y"] + sigma * sy
                sum_mv_x += self.bodies[i]["mass"] * vx
                sum_mv_y += self.bodies[i]["mass"] * vy
                out[i] = (vx, vy)
            else:
                out[i] = (
                    (rec["px"] - sum_mv_x) / self.bodies[i]["mass"],
                    (rec["py"] - sum_mv_y) / self.bodies[i]["mass"],
                )
        return out

    def _synth_ke(self, members, rec, sigma):
        vs = self._synth_velocities(members, rec, sigma)
        ke = 0.0
        for i in members:
            vx, vy = vs[i]
            ke += 0.5 * self.bodies[i]["mass"] * (vx * vx + vy * vy)
        return ke

    def _solve_sigma(self, members, rec, k_target):
        c0 = self._synth_ke(members, rec, 0.0)
        c1 = self._synth_ke(members, rec, 1.0)
        cm = self._synth_ke(members, rec, -1.0)
        a = ((c1 + cm) - (c0 + c0)) * 0.5
        b = (c1 - cm) * 0.5
        disc = b * b - 4.0 * a * (c0 - k_target)
        if a == 0.0:
            return 0.0, True
        if disc < 0.0:
            return (0.0 - b) / (2.0 * a), True
        sq = math.sqrt(disc)
        r1 = ((0.0 - b) + sq) / (2.0 * a)
        r2 = ((0.0 - b) - sq) / (2.0 * a)
        return (r1 if r1 * r1 <= r2 * r2 else r2), False

    def _refit(self, region: int, t: int) -> None:
        # The deadline is consumed on execution and re-armed only when
        # the window keeps members (audit ONT-011).
        self.region_window_deadline[region] = None
        x0 = (region % 2) * 64.0
        y0 = (region // 2) * 64.0
        members = [
            i for i in range(len(self.bodies)) if self.coarse[i] is not None and self.body_region[i] == region
        ]
        for i in members:
            self.bodies[i] = self._state_at(i, t)
        keep = [
            i
            for i in members
            if self.bodies[i]["x"] >= x0
            and self.bodies[i]["x"] < x0 + 64.0
            and self.bodies[i]["y"] >= y0
            and self.bodies[i]["y"] < y0 + 64.0
        ]
        for i in members:
            if i not in keep:
                self.coarse[i] = None
                self.body_region[i] = UNMANAGED
        if not keep:
            self.region_coarse[region] = False
            return
        self._fit_members(keep, t)
        for i in keep:
            self.body_region[i] = region
        self.region_window_deadline[region] = t + WINDOW

    def _static_impulse(self, i, nx, ny, vrx, vry):
        """Spec section 24 one-sided impulse vs a frozen contactant."""
        e = self.restitution
        fr = self.friction
        mi = self.bodies[i]["mass"]
        vn = vrx * nx + vry * ny
        s = (1.0 + e) * vn
        self.bodies[i]["vx"] += s * nx
        self.bodies[i]["vy"] += s * ny
        self.px += mi * (s * nx)
        self.py += mi * (s * ny)
        jn = (0.0 - s) * mi
        if fr > 0.0:
            vt = (0.0 - vrx) * ny + vry * nx
            jt = vt * mi
            jt_max = fr * jn
            if jt > jt_max:
                jt = jt_max
            if jt < 0.0 - jt_max:
                jt = 0.0 - jt_max
            w = jt / mi
            self.bodies[i]["vx"] += w * (0.0 - ny)
            self.bodies[i]["vy"] += w * nx
            self.px += jt * (0.0 - ny)
            self.py += jt * nx
        return vn, jn

    def _contact_pass(self, entering: int, frozen) -> None:
        """Spec sections 21 + 24: single pinned lexicographic impulse pass."""
        radii = [CONTACT_R * math.sqrt(b["mass"]) for b in self.bodies]
        e = self.restitution
        fr = self.friction
        extended = self.contact_params
        # Section 21 activation rule (audit ONT-013): the touching set's
        # record-suppression semantics are live only from the first pass
        # AFTER a Contact record has been emitted. Until then every
        # impulse-producing overlap records — the run's first
        # trajectory-changing impulse is never silent, so a record-only
        # verifier replaying gravity-only up to the first record's tick
        # reproduces the run exactly (a verifier's own touching history
        # also begins at the first record).
        suppress = self.contact_armed
        n = len(self.bodies)
        nxt = set()
        events = []
        # Section 24: the sweep visits every unordered real-body pair
        # once in pinned (i, j) id order and dispatches on membership —
        # a fine body resolving against an ephemeris-coarse contactant
        # is reachable whichever member carries the smaller id. Only
        # (fine, fine), (fine, coarse), and — with the section 24
        # record — (coarse, fine) pairs proceed; collapsed members
        # never contact individually (their region contacts as a
        # monopole), coarse-coarse pairs have no movable member, and
        # without the record non-fine bodies never contact (section 21).
        for i in range(n):
            for j in range(i + 1, n):
                if frozen[j]:
                    if frozen[i] or self.body_collapsed[j] is not None:
                        continue
                elif frozen[i] and not (extended and self.coarse[i] is not None):
                    continue
                # f is the fine member; so is the other member's state
                # at the tick (polynomial evaluation for a coarse
                # contactant, integrated state for a fine pair). The
                # normal points from the fine body toward the contactant.
                if frozen[i]:
                    f, o = j, i
                else:
                    f, o = i, j
                so = self._state_at(o, entering)
                dx = so["x"] - self.bodies[f]["x"]
                dy = so["y"] - self.bodies[f]["y"]
                rs = radii[i] + radii[j]
                d2 = dx * dx + dy * dy
                if d2 >= rs * rs:
                    continue
                pair = (i, j)
                nxt.add(pair)
                if d2 == 0.0:
                    nx = 1.0
                    ny = 0.0
                else:
                    dist = math.sqrt(d2)
                    nx = dx / dist
                    ny = dy / dist
                vrx = so["vx"] - self.bodies[f]["vx"]
                vry = so["vy"] - self.bodies[f]["vy"]
                vn = vrx * nx + vry * ny
                if vn >= 0.0:
                    continue
                mi = self.bodies[i]["mass"]
                mj = self.bodies[j]["mass"]
                cx = (self.bodies[f]["x"] + so["x"]) * 0.5
                cy = (self.bodies[f]["y"] + so["y"]) * 0.5
                fine_pair = not frozen[i] and not frozen[j]
                if fine_pair:
                    inv = 1.0 / (mi + mj)
                    t = vn * inv
                    s = (1.0 + e) * t
                    fi = s * mj
                    fj = s * mi
                    self.bodies[i]["vx"] += fi * nx
                    self.bodies[i]["vy"] += fi * ny
                    self.bodies[j]["vx"] -= fj * nx
                    self.bodies[j]["vy"] -= fj * ny
                    mu = (mi * mj) / (mi + mj)
                    jn = ((0.0 - vn) * (1.0 + e)) * mu
                    if fr > 0.0:
                        vt = (0.0 - vrx) * ny + vry * nx
                        q = vt * inv
                        qmax = (fr * jn) * inv
                        if q > qmax:
                            q = qmax
                        if q < 0.0 - qmax:
                            q = 0.0 - qmax
                        fti = q * mj
                        ftj = q * mi
                        self.bodies[i]["vx"] += fti * (0.0 - ny)
                        self.bodies[i]["vy"] += fti * nx
                        self.bodies[j]["vx"] -= ftj * (0.0 - ny)
                        self.bodies[j]["vy"] -= ftj * nx
                else:
                    _, jn = self._static_impulse(f, nx, ny, vrx, vry)
                    mu = (mi * mj) / (mi + mj)
                if suppress and pair in self.touching:
                    continue
                # The contactant is measured at its post-impulse state for
                # fine pairs and at its (frozen) polynomial evaluation for
                # coarse pairs — never at the stale demote-time slot.
                if fine_pair:
                    vn_after = (
                        (self.bodies[j]["vx"] - self.bodies[i]["vx"]) * nx
                        + (self.bodies[j]["vy"] - self.bodies[i]["vy"]) * ny
                    )
                else:
                    vn_after = (so["vx"] - self.bodies[f]["vx"]) * nx + (
                        so["vy"] - self.bodies[f]["vy"]
                    ) * ny
                events.append(
                    {
                        "tick": entering,
                        "a": i,
                        "b": j,
                        "jn": jn,
                        "cx": cx,
                        "cy": cy,
                        "vn": vn,
                        "vn_after": vn_after,
                        "mu": mu,
                        "static_pair": not fine_pair,
                    }
                )
        if extended:
            for region in range(4):
                rec = self.region_collapsed[region]
                if rec is None or rec["count"] == 0:
                    continue
                big_r = CONTACT_R * math.sqrt(rec["mass"])
                for i in range(n):
                    if frozen[i]:
                        continue
                    dx = rec["com_x"] - self.bodies[i]["x"]
                    dy = rec["com_y"] - self.bodies[i]["y"]
                    rs = radii[i] + big_r
                    d2 = dx * dx + dy * dy
                    if d2 >= rs * rs:
                        continue
                    pair = (i, MONOPOLE_BASE + region)
                    nxt.add(pair)
                    if d2 == 0.0:
                        nx = 1.0
                        ny = 0.0
                    else:
                        dist = math.sqrt(d2)
                        nx = dx / dist
                        ny = dy / dist
                    vrx = rec["vcom_x"] - self.bodies[i]["vx"]
                    vry = rec["vcom_y"] - self.bodies[i]["vy"]
                    vn = vrx * nx + vry * ny
                    if vn >= 0.0:
                        continue
                    mi = self.bodies[i]["mass"]
                    cx = (self.bodies[i]["x"] + rec["com_x"]) * 0.5
                    cy = (self.bodies[i]["y"] + rec["com_y"]) * 0.5
                    _, jn = self._static_impulse(i, nx, ny, vrx, vry)
                    mu = (mi * rec["mass"]) / (mi + rec["mass"])
                    if suppress and pair in self.touching:
                        continue
                    vn_after = (rec["vcom_x"] - self.bodies[i]["vx"]) * nx + (
                        rec["vcom_y"] - self.bodies[i]["vy"]
                    ) * ny
                    events.append(
                        {
                            "tick": entering,
                            "a": i,
                            "b": MONOPOLE_BASE + region,
                            "jn": jn,
                            "cx": cx,
                            "cy": cy,
                            "vn": vn,
                            "vn_after": vn_after,
                            "mu": mu,
                        }
                    )
        if self.walls:
            for i in range(n):
                if frozen[i]:
                    continue
                for wall in range(4):
                    # Section 24 (audit ONT-015): overlap and approach are
                    # separate tests. Touching contains every overlapping
                    # pair, walls included, so a penetrating-but-receding
                    # body keeps its key (the contact has not ended); the
                    # impulse applies only while approaching.
                    if wall == 0:
                        overlap = self.bodies[i]["x"] - radii[i] < 0.0
                        approaching = self.bodies[i]["vx"] < 0.0
                        nx = 0.0 - 1.0
                        ny = 0.0
                        cx = (self.bodies[i]["x"] + 0.0) * 0.5
                        cy = self.bodies[i]["y"]
                    elif wall == 1:
                        overlap = self.bodies[i]["x"] + radii[i] > 128.0
                        approaching = self.bodies[i]["vx"] > 0.0
                        nx = 1.0
                        ny = 0.0
                        cx = (self.bodies[i]["x"] + 128.0) * 0.5
                        cy = self.bodies[i]["y"]
                    elif wall == 2:
                        overlap = self.bodies[i]["y"] - radii[i] < 0.0
                        approaching = self.bodies[i]["vy"] < 0.0
                        nx = 0.0
                        ny = 0.0 - 1.0
                        cx = self.bodies[i]["x"]
                        cy = (self.bodies[i]["y"] + 0.0) * 0.5
                    else:
                        overlap = self.bodies[i]["y"] + radii[i] > 128.0
                        approaching = self.bodies[i]["vy"] > 0.0
                        nx = 0.0
                        ny = 1.0
                        cx = self.bodies[i]["x"]
                        cy = (self.bodies[i]["y"] + 128.0) * 0.5
                    if not overlap:
                        continue
                    pair = (i, WALL_BASE + wall)
                    nxt.add(pair)
                    if not approaching:
                        continue
                    vrx = 0.0 - self.bodies[i]["vx"]
                    vry = 0.0 - self.bodies[i]["vy"]
                    vn = vrx * nx + vry * ny
                    if vn >= 0.0:
                        continue
                    vn, jn = self._static_impulse(i, nx, ny, vrx, vry)
                    if suppress and pair in self.touching:
                        continue
                    vn_after = (0.0 - self.bodies[i]["vx"]) * nx + (0.0 - self.bodies[i]["vy"]) * ny
                    events.append(
                        {
                            "tick": entering,
                            "a": i,
                            "b": WALL_BASE + wall,
                            "jn": jn,
                            "cx": cx,
                            "cy": cy,
                            "vn": vn,
                            "vn_after": vn_after,
                            "mu": self.bodies[i]["mass"],
                        }
                    )
        if events:
            self.contact_armed = True
        self.touching = nxt
        self.last_contacts.extend(events)

    def step(self) -> None:
        entering = self.tick + 1
        n = len(self.bodies)
        # Section 21 (audit ONT-014): a pair leaves the touching set when
        # either member crosses the Fine/non-Fine boundary at any point
        # during the boundary (demote, promote, collapse, expansion,
        # refit thaw), even if the body ends the boundary back at its
        # starting status (e.g. demote+promote or expand+recollapse on
        # one tick), so a body that is Fine again while still overlapping
        # begins a fresh contact. Marks accumulate per applied
        # transition, not from final-vs-initial membership. A coarse
        # body moving between coarse regions (e.g. absorbed into a
        # foreign collapse) stays non-Fine and keeps its keys. Pseudo-id
        # keys of a collapsed region drop when the region leaves
        # collapse during the boundary, even if it re-collapses.
        status = [self.coarse[i] is None and self.body_collapsed[i] is None for i in range(n)]
        left_fine = [False] * n
        left_collapse = [False] * 4

        def mark_fine_moves() -> None:
            for i in range(n):
                fine = self.coarse[i] is None and self.body_collapsed[i] is None
                if fine != status[i]:
                    left_fine[i] = True
                    status[i] = fine

        for region, level in self.events.pop(entering, []):
            was_collapsed = self.region_collapsed[region] is not None
            if level == 0:
                if self.region_collapsed[region] is None:
                    self._demote(region, entering)
            elif level == 1:
                if self.region_collapsed[region] is not None:
                    self._expand(region, entering)
                else:
                    self._promote(region, entering)
            elif self.region_collapsed[region] is None:
                self._collapse(region, entering)
            if was_collapsed and self.region_collapsed[region] is None:
                left_collapse[region] = True
            mark_fine_moves()
        for region in range(4):
            if (
                self.region_coarse[region]
                and self.region_collapsed[region] is None
                and self.region_window_deadline[region] == entering
            ):
                self._refit(region, entering)
                mark_fine_moves()

        if self.touching and (any(left_fine) or any(left_collapse)):

            def _pair_kept(pair):
                a, b = pair
                if (
                    MONOPOLE_BASE <= b < WALL_BASE
                    and (b - MONOPOLE_BASE) < 4
                    and left_collapse[b - MONOPOLE_BASE]
                ):
                    return False
                if a < n and left_fine[a]:
                    return False
                return not (b < MONOPOLE_BASE and b < n and left_fine[b])

            self.touching = {pair for pair in self.touching if _pair_kept(pair)}

        coarse = [self.coarse[i] is not None for i in range(n)]
        frozen = [coarse[i] or self.body_collapsed[i] is not None for i in range(n)]
        monopoles = [
            self.region_collapsed[r]
            for r in range(4)
            if self.region_collapsed[r] is not None and self.region_collapsed[r]["mass"] > 0.0
        ]
        half = DT * 0.5
        view = [self._state_at(i, entering) for i in range(n)]
        ax_ff, ay_ff, ax_fc, ay_fc = accel_split(view, coarse, self.body_collapsed, monopoles)
        for i in range(n):
            if not frozen[i]:
                self.bodies[i]["vx"] += ax_ff[i] * half
                self.bodies[i]["vy"] += ay_ff[i] * half
        for i in range(n):
            if not frozen[i]:
                self.bodies[i]["vx"] += ax_fc[i] * half
                self.bodies[i]["vy"] += ay_fc[i] * half
                self.px += self.bodies[i]["mass"] * (ax_fc[i] * half)
                self.py += self.bodies[i]["mass"] * (ay_fc[i] * half)
        for i in range(n):
            if not frozen[i]:
                self.bodies[i]["x"] += self.bodies[i]["vx"] * DT
                self.bodies[i]["y"] += self.bodies[i]["vy"] * DT
        view = [self._state_at(i, entering) for i in range(n)]
        ax_ff, ay_ff, ax_fc, ay_fc = accel_split(view, coarse, self.body_collapsed, monopoles)
        for i in range(n):
            if not frozen[i]:
                self.bodies[i]["vx"] += ax_ff[i] * half
                self.bodies[i]["vy"] += ay_ff[i] * half
        for i in range(n):
            if not frozen[i]:
                self.bodies[i]["vx"] += ax_fc[i] * half
                self.bodies[i]["vy"] += ay_fc[i] * half
                self.px += self.bodies[i]["mass"] * (ax_fc[i] * half)
                self.py += self.bodies[i]["mass"] * (ay_fc[i] * half)
        if self.contacts:
            self._contact_pass(entering, frozen)
        self.tick = entering

    def totals(self):
        n = len(self.bodies)
        view = [self._state_at(i, self.tick) for i in range(n)]
        fine = 0
        coarse_n = 0
        mass = 0.0
        for b in view:
            if self.coarse[b["id"]] is not None or self.body_collapsed[b["id"]] is not None:
                coarse_n += 1
            else:
                fine += 1
            mass += b["mass"]
        return fine, coarse_n, mass, self.px, self.py, subset_energy(view)

    def state_bytes(self, i: int) -> bytes:
        b = self._state_at(i, self.tick)
        if self.body_collapsed[i] is not None:
            level = 2
        elif self.coarse[i] is not None:
            level = 0
        else:
            level = 1
        return struct.pack(
            "<IdddddB", b["id"], b["x"], b["y"], b["vx"], b["vy"], b["mass"], level
        )

    def emitted_state(self, i: int):
        b = self._state_at(i, self.tick)
        if self.body_collapsed[i] is not None:
            level = 2
            region = self.body_collapsed[i]
        elif self.coarse[i] is not None:
            level = 0
            region = self.body_region[i]
        else:
            level = 1
            region = region_at(b["x"], b["y"])
        return b, region, level

    def region_hash(self, region: int):
        members = []
        for i in range(len(self.bodies)):
            if self.body_collapsed[i] is not None:
                if self.body_collapsed[i] == region:
                    members.append(i)
            elif self.coarse[i] is not None:
                if self.body_region[i] == region:
                    members.append(i)
            else:
                b = self._state_at(i, self.tick)
                if region_at(b["x"], b["y"]) == region:
                    members.append(i)
        if self.region_collapsed[region] is not None:
            level = 2
        elif self.region_coarse[region]:
            level = 0
        else:
            level = 1
        payload = bytes([level])
        for i in sorted(members):
            payload += self.state_bytes(i)
        return level, len(members), fnv1a64(payload)

    def world_hash(self) -> int:
        payload = struct.pack("<Q", self.tick)
        for i in range(len(self.bodies)):
            payload += self.state_bytes(i)
        return fnv1a64(payload)


def accel_split(view, coarse, body_collapsed=None, monopoles=None):
    n = len(view)
    ax_ff = [0.0] * n
    ay_ff = [0.0] * n
    ax_fc = [0.0] * n
    ay_fc = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if body_collapsed is not None and (body_collapsed[i] is not None or body_collapsed[j] is not None):
                continue
            dx = view[j]["x"] - view[i]["x"]
            dy = view[j]["y"] - view[i]["y"]
            s2 = dx * dx + dy * dy + EPS2
            inv3 = 1.0 / (s2 * math.sqrt(s2))
            fx = G * inv3 * dx
            fy = G * inv3 * dy
            ci, cj = coarse[i], coarse[j]
            if not ci and not cj:
                ax_ff[i] += view[j]["mass"] * fx
                ay_ff[i] += view[j]["mass"] * fy
                ax_ff[j] -= view[i]["mass"] * fx
                ay_ff[j] -= view[i]["mass"] * fy
            elif not ci and cj:
                ax_fc[i] += view[j]["mass"] * fx
                ay_fc[i] += view[j]["mass"] * fy
            elif ci and not cj:
                ax_fc[j] -= view[i]["mass"] * fx
                ay_fc[j] -= view[i]["mass"] * fy
    if monopoles:
        for m in monopoles:
            for i in range(n):
                if coarse[i] or (body_collapsed is not None and body_collapsed[i] is not None):
                    continue
                dx = m["com_x"] - view[i]["x"]
                dy = m["com_y"] - view[i]["y"]
                s2 = dx * dx + dy * dy + EPS2
                inv3 = 1.0 / (s2 * math.sqrt(s2))
                fx = G * inv3 * dx
                fy = G * inv3 * dy
                ax_fc[i] += m["mass"] * fx
                ay_fc[i] += m["mass"] * fy
    return ax_ff, ay_ff, ax_fc, ay_fc


def expected_zoom_policy(seed: int, offset: int, ticks: int, cli_events=None):
    """Recompute the spec section 18 zoom-policy event sequence.

    cli_events: scheduled (tick, region, level) triples applied to the
    level-state machine in stream order before the policy evaluates at
    each boundary (levels: 0 fine, 1 coarse, 2 collapsed; demote on a
    collapsed region is a no-op per spec 19).
    """
    rng = SplitMix64(seed ^ offset)
    points = []
    for _ in range(2):
        u0 = rng.next()
        u1 = rng.next()
        points.append((16.0 + u0 * TWO_POW_NEG64 * 96.0, 16.0 + u1 * TWO_POW_NEG64 * 96.0))

    def focus(t):
        k = (t - 1) // 64
        while len(points) < k + 2:
            u0 = rng.next()
            u1 = rng.next()
            points.append((16.0 + u0 * TWO_POW_NEG64 * 96.0, 16.0 + u1 * TWO_POW_NEG64 * 96.0))
        p0, p1 = points[k], points[k + 1]
        f = (t - (1 + 64 * k)) / 64.0
        return (p0[0] + (p1[0] - p0[0]) * f, p0[1] + (p1[1] - p0[1]) * f)

    cli = {}
    for ev in cli_events or []:
        t, region, lv = ev
        cli.setdefault(t, []).append((region, lv))
    modes = [0, 0, 0, 0]
    events = []
    pending_cli = sorted(cli.items())
    for t in range(17, ticks + 1, 16):
        while pending_cli and pending_cli[0][0] <= t:
            for region, lv in pending_cli.pop(0)[1]:
                if lv == 2:
                    modes[region] = 2
                elif lv == 0:
                    if modes[region] == 0:
                        modes[region] = 1
                else:
                    modes[region] = 0
        fx, fy = focus(t)
        for region in range(4):
            x0 = (region % 2) * 64.0
            y0 = (region // 2) * 64.0
            cx = min(max(fx, x0), x0 + 64.0)
            cy = min(max(fy, y0), y0 + 64.0)
            d = math.sqrt((fx - cx) * (fx - cx) + (fy - cy) * (fy - cy))
            if modes[region] == 0 and d > 48.0:
                modes[region] = 1
                events.append((t, region, 0))
            elif modes[region] == 1 and d < 24.0:
                # Promotion gates on Coarse only (audit ONT-010): a
                # Collapsed region is left alone — the zoom policy never
                # expands a monopole.
                modes[region] = 0
                events.append((t, region, 1))
    return events


def check_zoom_policy(records, seed: int, offset: int, *, cli_events=None) -> DiagnosticResult:
    """Verify the stream's RegionLevel sequence matches the zoom policy.

    cli_events: optional iterable of (tick, region, level) events scheduled
    via --demote-at/--promote-at; they are interleaved with policy events
    in stream order and excluded from the policy expectation.

    The comparison is over multisets, not membership sets: the expected
    sequence is policy events + requested CLI events, each exactly once,
    and a duplicated (or dropped) RegionLevel record fails (audit ONT-008).
    """
    from collections import Counter

    stream_events = []
    pending = []
    last_tick = 0
    for record in records:
        if record[0] == "level":
            pending.append((record[1], record[2], record[3]))
        elif record[0] == "tick":
            _, tick = record
            for rx, ry, lv in pending:
                stream_events.append((tick, ry * 2 + rx, lv))
            pending.clear()
            last_tick = tick
    policy = expected_zoom_policy(seed, offset, last_tick, cli_events=cli_events)
    expected = Counter(policy) + Counter(cli_events or [])
    got = Counter(stream_events)
    ok = got == expected
    detail = {
        "stream_events": len(stream_events),
        "policy_events": len(policy),
    }
    if not ok:
        detail["missing"] = sorted((expected - got).elements())[:4]
        detail["unexpected"] = sorted((got - expected).elements())[:4]
    return DiagnosticResult(
        name="ontos_zoom_policy",
        passed=ok,
        threshold=0.0,
        value=0.0 if ok else 1.0,
        detail=detail,
    )


def rebound_positions(seed: int, body_count: int, ticks: int):
    """REBOUND anchor: same ICs, same softening/dt, independent integrator."""
    import rebound as rb

    ics = initial_conditions(seed, body_count)
    sim = rb.Simulation()
    sim.G = 1.0
    sim.softening = 1.0
    sim.dt = DT
    sim.integrator = "LEAPFROG"
    for b in ics:
        sim.add(m=b["mass"], x=b["x"], y=b["y"], z=0.0, vx=b["vx"], vy=b["vy"], vz=0.0)
    sim.integrate(ticks * DT)
    return [(p.x, p.y) for p in sim.particles]


def check_rebound_anchor(records, seed: int, body_count: int, *, ticks: int = None, tol: float = 1e-4) -> DiagnosticResult:
    """Compare the stream's final tick positions against a REBOUND run."""
    last_tick = 0
    positions = {}
    for record in records:
        if record[0] == "tick":
            last_tick = record[1]
        elif record[0] == "body":
            _, tick, bid, region, level, x, y, vx, vy, mass = record
            positions[bid] = (x, y)
    if not positions:
        raise ValueError("stream carries no BodyState records")
    if len(positions) != body_count:
        raise ValueError(f"stream has {len(positions)} bodies, expected {body_count}")
    ref = rebound_positions(seed, body_count, ticks if ticks is not None else last_tick)
    max_dev = 0.0
    scale = 0.0
    for bid, (rx, ry) in enumerate(ref):
        sx, sy = positions[bid]
        max_dev = max(max_dev, abs(sx - rx), abs(sy - ry))
        scale = max(scale, abs(rx), abs(ry))
    rel = max_dev / (scale if scale > 1e-30 else 1e-30)
    return DiagnosticResult(
        name="ontos_rebound_anchor",
        passed=rel <= tol,
        threshold=float(tol),
        value=float(rel),
        detail={"ticks": last_tick, "max_abs_deviation": max_dev, "relative": rel},
    )


_V2_REGION_ORDER = ((0, 0), (1, 0), (0, 1), (1, 1))


def _check_region_xy(tag: int, rx: int, ry: int, offset: int) -> None:
    if rx not in (0, 1) or ry not in (0, 1):
        raise ValueError(
            f"noncanonical region coordinates ({rx},{ry}) in record tag {tag} at offset {offset}: "
            "rx and ry must each be 0 or 1"
        )


def parse_stream_v2(path):
    """Strict version-2 parser (spec section 16 emission contract).

    Grammar: ContactParams only before the first TickHeader; at each tick
    boundary RegionLevel records first, then RegionCollapsed groups
    (each optionally followed by RegionMultipole and then RegionRadial or
    RegionShells), then Contact records; then exactly TickHeader,
    Snapshot, TotalsState, RegionState for regions (0,0), (1,0), (0,1),
    (1,1) in that order, then BodyState for bodies 0..N-1 in id order.
    Every timestamped record must repeat its frame's tick; pre-tick
    records must carry the tick of the TickHeader they precede.
    Duplicates, omissions, records outside an open frame, non-monotonic
    ticks, dangling boundary records at EOF, and a zero-record stream are
    all parse errors.
    """
    data = Path(path).read_bytes()
    if len(data) < 20 or data[:4] != MAGIC:
        raise ValueError("not an ontos stream: bad magic")
    version = struct.unpack_from("<I", data, 4)[0]
    if version != 2:
        raise ValueError(f"not a gravity stream (version {version})")
    world_w, world_h, body_count = struct.unpack_from("<III", data, 8)
    if body_count < 1:
        raise ValueError(f"invalid body_count {body_count} in stream header")
    records = []
    offset = 20
    n = len(data)
    # state: boundary family ("boundary", "collapse", "collapse-mp",
    # "contacts") or frame family ("snapshot", "totals", ("region", i),
    # ("body", i)).
    state = "boundary"
    last_tick = 0
    pending_boundary = False
    pending_pre_tick: list[tuple[str, int]] = []
    while offset < n:
        rec_off = offset
        tag = data[offset]
        offset += 1
        if tag == 1:
            if state not in ("boundary", "collapse", "collapse-mp", "contacts"):
                raise ValueError(
                    f"TickHeader at offset {rec_off} inside an incomplete tick frame (tick {last_tick})"
                )
            (tick,) = struct.unpack_from("<Q", data, offset)
            offset += 8
            if tick != last_tick + 1:
                raise ValueError(
                    f"non-consecutive TickHeader tick {tick} at offset {rec_off}: expected {last_tick + 1}"
                )
            for kind, ptick in pending_pre_tick:
                if ptick != tick:
                    raise ValueError(
                        f"{kind} tick {ptick} at boundary does not match the following TickHeader tick {tick}"
                    )
            pending_pre_tick.clear()
            pending_boundary = False
            last_tick = tick
            state = "snapshot"
            records.append(("tick", tick))
        elif tag == 2:
            if state != "snapshot":
                raise ValueError(f"Snapshot outside an open tick frame at offset {rec_off}")
            (population,) = struct.unpack_from("<Q", data, offset)
            offset += 8
            state = "totals"
            records.append(("snapshot", population))
        elif tag == 3:
            raise ValueError(f"tag 3 (CellFlipped) is reserved and unused in version 2 (offset {rec_off})")
        elif tag == 4:
            if state not in ("boundary",):
                raise ValueError(f"RegionLevel after boundary physics records at offset {rec_off}")
            region_x, region_y, level = struct.unpack_from("<IIB", data, offset)
            offset += 9
            _check_region_xy(4, region_x, region_y, rec_off)
            if level > 2:
                raise ValueError(f"invalid region level {level} at offset {rec_off}")
            pending_boundary = True
            records.append(("level", region_x, region_y, level))
        elif tag == 5:
            if not (isinstance(state, tuple) and state[0] == "region"):
                raise ValueError(f"RegionState outside an open tick frame at offset {rec_off}")
            tick, rx, ry, level, population, rhash = struct.unpack_from("<QIIBQQ", data, offset)
            offset += 33
            _check_region_xy(5, rx, ry, rec_off)
            expected = _V2_REGION_ORDER[state[1]]
            if (rx, ry) != expected:
                raise ValueError(
                    f"RegionState for region ({rx},{ry}) at offset {rec_off}: expected region "
                    f"{expected} next (order/duplicate violation)"
                )
            if tick != last_tick:
                raise ValueError(
                    f"RegionState tick {tick} does not match the open frame tick {last_tick} at offset {rec_off}"
                )
            if level > 2:
                raise ValueError(f"invalid region-state level {level} at offset {rec_off}")
            nxt = state[1] + 1
            state = ("region", nxt) if nxt < 4 else ("body", 0)
            records.append(("state", tick, rx, ry, level, population, rhash))
        elif tag == 6:
            if not (isinstance(state, tuple) and state[0] == "body"):
                raise ValueError(
                    f"BodyState outside the body block of tick {last_tick} at offset {rec_off}"
                )
            tick, bid, region, level, x, y, vx, vy, mass = struct.unpack_from("<QIBBddddd", data, offset)
            offset += 54
            if tick != last_tick:
                raise ValueError(
                    f"BodyState tick {tick} does not match the open frame tick {last_tick} at offset {rec_off}"
                )
            if bid != state[1]:
                raise ValueError(
                    f"BodyState id {bid} at offset {rec_off}: expected body {state[1]} next "
                    "(id order/duplicate violation)"
                )
            if region > 3 and region != UNMANAGED:
                raise ValueError(f"BodyState region {region} at offset {rec_off}: expected 0..3 or {UNMANAGED}")
            if level > 2:
                raise ValueError(f"BodyState level {level} at offset {rec_off}: expected 0..2")
            nxt = state[1] + 1
            state = ("body", nxt) if nxt < body_count else "boundary"
            records.append(("body", tick, bid, region, level, x, y, vx, vy, mass))
        elif tag == 7:
            if state != "totals":
                raise ValueError(f"TotalsState outside an open tick frame at offset {rec_off}")
            vals = struct.unpack_from("<QQQdddd", data, offset)
            offset += 56
            if vals[0] != last_tick:
                raise ValueError(
                    f"TotalsState tick {vals[0]} does not match the open frame tick {last_tick} at offset {rec_off}"
                )
            state = ("region", 0)
            records.append(("totals", *vals))
        elif tag == 8:
            if state not in ("boundary", "collapse", "collapse-mp"):
                raise ValueError(f"RegionCollapsed after Contact records at offset {rec_off}")
            vals = struct.unpack_from("<QIIQdddddd", data, offset)
            offset += 72
            _check_region_xy(8, vals[1], vals[2], rec_off)
            pending_boundary = True
            pending_pre_tick.append(("RegionCollapsed", vals[0]))
            state = "collapse"
            records.append(("collapsed", *vals))
        elif tag == 9:
            if state != "collapse":
                raise ValueError(
                    f"RegionMultipole without a preceding RegionCollapsed at offset {rec_off}"
                )
            vals = struct.unpack_from("<QIIddddd", data, offset)
            offset += 56
            _check_region_xy(9, vals[1], vals[2], rec_off)
            pending_pre_tick.append(("RegionMultipole", vals[0]))
            state = "collapse-mp"
            records.append(("multipole", *vals))
        elif tag == 10:
            if state not in ("boundary", "collapse", "collapse-mp", "contacts"):
                raise ValueError(f"Contact outside a tick boundary at offset {rec_off}")
            vals = struct.unpack_from("<QIIddd", data, offset)
            offset += 40
            body_a, body_b = vals[1], vals[2]
            if body_a >= body_count:
                raise ValueError(
                    f"Contact body_a {body_a} at offset {rec_off}: expected a real body id < {body_count}"
                )
            if body_b >= body_count and body_b < MONOPOLE_BASE:
                raise ValueError(
                    f"Contact body_b {body_b} at offset {rec_off}: expected a real body id < {body_count} "
                    f"or a pseudo id >= {MONOPOLE_BASE}"
                )
            pending_boundary = True
            pending_pre_tick.append(("Contact", vals[0]))
            state = "contacts"
            records.append(("contact", *vals))
        elif tag == 11 or tag == 13:
            if state != "collapse-mp":
                kind = "RegionRadial" if tag == 11 else "RegionShells"
                raise ValueError(f"{kind} without a preceding RegionMultipole at offset {rec_off}")
            if tag == 11:
                vals = struct.unpack_from("<QIId", data, offset)
                offset += 24
            else:
                vals = struct.unpack_from("<QIIddddd", data, offset)
                offset += 56
            _check_region_xy(tag, vals[1], vals[2], rec_off)
            pending_pre_tick.append(("RegionRadial" if tag == 11 else "RegionShells", vals[0]))
            state = "contacts"
            records.append(("radial" if tag == 11 else "shells", *vals))
        elif tag == 12:
            if state != "boundary" or last_tick != 0:
                raise ValueError(
                    f"ContactParams after the first TickHeader at offset {rec_off}: it must precede all ticks"
                )
            vals = struct.unpack_from("<ddB", data, offset)
            offset += 17
            if vals[2] > 1:
                raise ValueError(f"ContactParams walls byte {vals[2]} at offset {rec_off}: expected 0 or 1")
            pending_boundary = True
            records.append(("params", *vals))
        else:
            raise ValueError(f"unknown record tag {tag} at offset {rec_off}")
    if state not in ("boundary", "collapse", "collapse-mp", "contacts"):
        raise ValueError(f"stream ends inside an incomplete tick frame (state {state}, tick {last_tick})")
    if pending_boundary or pending_pre_tick:
        raise ValueError(
            "dangling boundary record(s) at end of stream with no following TickHeader "
            f"(pending ticks: {pending_pre_tick[:4]})"
        )
    if last_tick == 0:
        raise ValueError("stream carries no tick frames")
    return (world_w, world_h, body_count), records


@dataclass(frozen=True)
class GravityContract:
    """Immutable expected-run contract (audit ONT-001).

    Built from the grid spec / ontos.json metadata of the *requested*
    run — never from the candidate stream — and validated against the
    stream before physics replay. Fields left None are not asserted
    (legacy metadata carrying only a seed).
    """

    body_count: int | None = None
    ticks: int | None = None
    events: tuple | None = None  # None = not asserted; () = no events requested
    contacts: bool | None = None
    radial: bool | None = None
    shells: bool | None = None
    multipole: bool | None = None
    restitution: float | None = None
    friction: float | None = None
    walls: bool | None = None
    observer: int | None = None
    test_ic: str | None = None

    @classmethod
    def from_metadata(cls, meta: dict) -> "GravityContract":
        def opt_int(field, minimum):
            v = meta.get(field)
            if v is None:
                return None
            v = int(v)
            if v < minimum:
                raise ValueError(f"ontos.json {field} must be >= {minimum}, got {v}")
            return v

        def opt_float(field, low, high):
            v = meta.get(field)
            if v is None:
                return None
            v = float(v)
            if not (math.isfinite(v) and low <= v <= high):
                raise ValueError(f"ontos.json {field} must be finite in [{low},{high}], got {v}")
            return v

        def opt_bool(field):
            v = meta.get(field)
            return None if v is None else bool(v)

        events = [] if "events" in meta else None
        for ev in meta.get("events", []):
            if len(ev) != 4:
                raise ValueError(f"ontos.json events entries must be [t, rx, ry, level], got {list(ev)}")
            t, rx, ry, level = int(ev[0]), int(ev[1]), int(ev[2]), int(ev[3])
            if t < 1:
                raise ValueError(f"ontos.json event tick must be >= 1, got {t}")
            if rx not in (0, 1) or ry not in (0, 1):
                raise ValueError(f"ontos.json event region ({rx},{ry}) outside the 2x2 grid")
            if level not in (0, 1, 2):
                raise ValueError(f"ontos.json event level {level} not in {{0,1,2}}")
            events.append((t, ry * 2 + rx, level))
        observer = opt_int("observer", 0)
        ticks = opt_int("ticks", 1)
        if observer is not None and ticks is None:
            # The observer zoom policy is evaluated over the requested
            # horizon; without it the event schedule cannot be checked
            # (audit ONT-008).
            raise ValueError("ontos.json observer runs must carry ticks")
        test_ic = meta.get("test_ic")
        if test_ic is not None and test_ic not in ("wallshot", "coarsehit"):
            raise ValueError(f"ontos.json test_ic {test_ic!r} not in ('wallshot', 'coarsehit')")
        return cls(
            body_count=opt_int("bodies", 1),
            ticks=ticks,
            events=None if events is None else tuple(events),
            contacts=opt_bool("contacts"),
            radial=opt_bool("radial"),
            shells=opt_bool("shells"),
            multipole=opt_bool("multipole"),
            restitution=opt_float("restitution", 0.0, 1.0),
            friction=opt_float("friction", 0.0, float("inf")),
            walls=opt_bool("walls"),
            observer=observer,
            test_ic=test_ic,
        )


def verify_stream_gravity(path, seed: int, profile: str | None = None, *, expected: "GravityContract | None" = None) -> dict:
    (world_w, world_h, body_count), records = parse_stream_v2(path)
    if world_w != 128 or world_h != 128:
        raise ValueError(f"unsupported world size {world_w}x{world_h}")
    mismatches = []

    observed_events = []
    _pending_scan = []
    expected_last_tick = 0
    for record in records:
        if record[0] == "level":
            _pending_scan.append((record[1], record[2], record[3]))
        elif record[0] == "tick":
            for rx, ry, lv in _pending_scan:
                observed_events.append((record[1], ry * 2 + rx, lv))
            _pending_scan.clear()
            expected_last_tick = record[1]

    # --- Contract validation (audit ONT-001), before physics replay ---
    if expected is not None:
        if expected.body_count is not None and body_count != expected.body_count:
            mismatches.append(
                {
                    "tick": 0,
                    "field": "contract_body_count",
                    "expected": expected.body_count,
                    "actual": body_count,
                }
            )
        if expected.ticks is not None and expected_last_tick != expected.ticks:
            mismatches.append(
                {
                    "tick": expected_last_tick,
                    "field": "contract_ticks",
                    "expected": expected.ticks,
                    "actual": expected_last_tick,
                }
            )
        if expected.test_ic is not None and profile is not None and expected.test_ic != profile:
            mismatches.append(
                {
                    "tick": 0,
                    "field": "contract_test_ic",
                    "expected": expected.test_ic,
                    "actual": profile,
                }
            )
        if expected.events is not None:
            from collections import Counter

            want = Counter(expected.events)
            got = Counter(observed_events)
            if expected.observer is None:
                expected_total = want
            else:
                # With an observer the deterministic zoom policy contributes
                # its own events (which may legitimately repeat per policy),
                # so the stream multiset must equal policy + requested
                # exactly — a duplicated RegionLevel record must not pass
                # (audit ONT-008).
                expected_total = Counter(
                    expected_zoom_policy(
                        seed, expected.observer, expected.ticks,
                        cli_events=expected.events,
                    )
                ) + want
            for ev in sorted(expected_total - got):
                mismatches.append(
                    {
                        "tick": ev[0],
                        "field": "contract_event",
                        "expected": ev,
                        "actual": None,
                    }
                )
            for ev in sorted(got - expected_total):
                mismatches.append(
                    {
                        "tick": ev[0],
                        "field": "contract_unscheduled_event",
                        "expected": None,
                        "actual": ev,
                    }
                )

    world = GravityWorld(seed, body_count, profile)
    reference = GravityWorld(seed, body_count, profile)
    if expected is not None:
        # Feature modes come from the requested run, never from the
        # presence of candidate records.
        if expected.contacts:
            world.contacts = True
        if expected.radial:
            world.radial_enabled = True
        if expected.shells:
            world.shells_enabled = True
        if expected.restitution is not None:
            world.contact_params = True
            world.restitution = expected.restitution
        if expected.friction is not None:
            world.contact_params = True
            world.friction = expected.friction
        if expected.walls:
            world.contacts = True
            world.contact_params = True
            world.walls = True
        if expected.multipole is not None:
            world.mp_enabled = expected.multipole
    compared = 0
    last_tick = 0
    pending = []
    pending_collapsed = []
    pending_multipole = []
    pending_contact = []
    pending_radial = []
    pending_shells = []
    params_seen = False
    stream_params = None
    max_pos_dev = 0.0
    post_exp_dev = 0.0
    collapse_events = 0
    multipole_events = 0
    radial_events = 0
    shell_events = 0
    contact_events = 0
    contact_records_seen = 0
    contact_worst_vn_after = 0.0
    contact_min_jn = float("inf")
    static_contact_events = 0
    coarse_static_contact_events = 0
    collapse_energy_deltas = []
    multipole_deltas = []
    radial_deltas = []
    shell_deltas = []

    for record in records:
        kind = record[0]
        if kind == "level":
            _, rx, ry, level = record
            pending.append((ry * 2 + rx, level))
        elif kind == "collapsed":
            pending_collapsed.append(record)
        elif kind == "multipole":
            pending_multipole.append(record)
        elif kind == "contact":
            pending_contact.append(record)
        elif kind == "radial":
            pending_radial.append(record)
        elif kind == "shells":
            pending_shells.append(record)
        elif kind == "params":
            _, restitution, friction, walls = record
            if params_seen or last_tick != 0:
                mismatches.append(
                    {
                        "tick": last_tick,
                        "field": "contact_params",
                        "expected": "at most one, before the first tick",
                        "actual": record,
                    }
                )
            # Fail-closed: NaN comparisons are all False, so finiteness is
            # required explicitly and range checks are written positively.
            # A NaN friction used to pass the old `friction < 0.0` check.
            bad_params = not (
                math.isfinite(restitution)
                and math.isfinite(friction)
                and 0.0 <= restitution <= 1.0
                and friction >= 0.0
                and walls <= 1
            )
            if bad_params:
                mismatches.append(
                    {
                        "tick": last_tick,
                        "field": "contact_params",
                        "expected": "finite restitution in [0,1], finite friction >= 0, walls in {0,1}",
                        "actual": record,
                    }
                )
            else:
                world.contact_params = True
                world.restitution = restitution
                world.friction = friction
                world.walls = walls == 1
            params_seen = True
            stream_params = (restitution, friction, walls)
        elif kind == "tick":
            _, tick = record
            # Record-driven mode switches apply only when the contract
            # does not pin the mode (audit ONT-001).
            if expected is None or expected.multipole is None:
                world.mp_enabled = bool(pending_multipole)
            if expected is None or expected.radial is not False:
                world.radial_enabled = world.radial_enabled or bool(pending_radial)
            if expected is None or expected.shells is not False:
                world.shells_enabled = world.shells_enabled or bool(pending_shells)
            if pending_contact and (expected is None or expected.contacts is not False):
                world.contacts = True
            for region, level in pending:
                world.schedule(world.tick + 1, region, level)
            pending.clear()
            world.step()
            reference.step()
            last_tick = tick
            if tick != world.tick:
                mismatches.append({"tick": tick, "field": "tick", "expected": tick, "actual": world.tick})
            local_contacts = world.last_contacts
            world.last_contacts = []
            if len(local_contacts) != len(pending_contact):
                mismatches.append(
                    {
                        "tick": tick,
                        "field": "contact_count",
                        "expected": len(pending_contact),
                        "actual": len(local_contacts),
                    }
                )
            else:
                for rec, local in zip(pending_contact, local_contacts):
                    _, tick_c, body_a, body_b, jn, cx, cy = rec
                    compared += 1
                    contact_events += 1
                    residual = local["vn_after"] + world.restitution * local["vn"]
                    contact_worst_vn_after = max(contact_worst_vn_after, abs(residual))
                    contact_min_jn = min(contact_min_jn, jn)
                    if body_b >= MONOPOLE_BASE:
                        static_contact_events += 1
                    if local.get("static_pair"):
                        coarse_static_contact_events += 1
                    if (
                        tick_c != local["tick"]
                        or body_a != local["a"]
                        or body_b != local["b"]
                        or struct.pack("<ddd", jn, cx, cy)
                        != struct.pack("<ddd", local["jn"], local["cx"], local["cy"])
                    ):
                        mismatches.append(
                            {
                                "tick": tick_c,
                                "field": f"contact_{body_a}_{body_b}",
                                "expected": (jn, cx, cy),
                                "actual": (local["jn"], local["cx"], local["cy"]),
                            }
                        )
            contact_records_seen += len(pending_contact)
            pending_contact = []
            for i in range(len(world.bodies)):
                if world.body_collapsed[i] is not None:
                    continue
                b = world._state_at(i, world.tick)
                r = reference.bodies[i]
                dev = max(abs(b["x"] - r["x"]), abs(b["y"] - r["y"]))
                if world.expand_count > 0 and dev > post_exp_dev:
                    post_exp_dev = dev
                if i not in world.reconstructed and dev > max_pos_dev:
                    max_pos_dev = dev
            for cres in pending_collapsed:
                _, tick_c, rx, ry, count, mass, com_x, com_y, px, py, energy = cres
                compared += 1
                collapse_events += 1
                region = ry * 2 + rx
                local = next((c for c in world.last_collapses if c["region"] == region), None)
                if local is None:
                    mismatches.append(
                        {
                            "tick": tick_c,
                            "field": "collapse_record",
                            "expected": "no local collapse",
                            "actual": None,
                        }
                    )
                    continue
                world.last_collapses.remove(local)
                if count != local["count"]:
                    mismatches.append(
                        {"tick": tick_c, "field": "collapse_count", "expected": count, "actual": local["count"]}
                    )
                for name, got, want in (
                    ("collapse_mass", mass, local["mass"]),
                    ("collapse_com_x", com_x, local["com_x"]),
                    ("collapse_com_y", com_y, local["com_y"]),
                    ("collapse_px", px, local["px"]),
                    ("collapse_py", py, local["py"]),
                    ("collapse_energy", energy, local["energy"]),
                ):
                    if struct.pack("<d", got) != struct.pack("<d", want):
                        mismatches.append({"tick": tick_c, "field": name, "expected": got, "actual": want})
                collapse_energy_deltas.append((tick_c, region, energy, local["syn_energy"]))
                mrec = next(
                    (m for m in pending_multipole if m[1] == tick_c and (m[3] * 2 + m[2]) == region),
                    None,
                )
                if mrec is not None:
                    pending_multipole.remove(mrec)
                    compared += 1
                    multipole_events += 1
                    for name, got, want in (
                        ("multipole_mx", mrec[4], local["mx"]),
                        ("multipole_my", mrec[5], local["my"]),
                        ("multipole_qxx", mrec[6], local["qxx"]),
                        ("multipole_qxy", mrec[7], local["qxy"]),
                        ("multipole_qyy", mrec[8], local["qyy"]),
                    ):
                        if struct.pack("<d", got) != struct.pack("<d", want):
                            mismatches.append({"tick": tick_c, "field": name, "expected": got, "actual": want})
                rrec = next(
                    (m for m in pending_radial if m[1] == tick_c and (m[3] * 2 + m[2]) == region),
                    None,
                )
                if rrec is not None:
                    pending_radial.remove(rrec)
                    compared += 1
                    radial_events += 1
                    if struct.pack("<d", rrec[4]) != struct.pack("<d", local["binding"]):
                        mismatches.append(
                            {
                                "tick": tick_c,
                                "field": "radial_binding",
                                "expected": rrec[4],
                                "actual": local["binding"],
                            }
                        )
                srec = next(
                    (m for m in pending_shells if m[1] == tick_c and (m[3] * 2 + m[2]) == region),
                    None,
                )
                if srec is not None:
                    pending_shells.remove(srec)
                    compared += 1
                    shell_events += 1
                    if struct.pack("<d", srec[4]) != struct.pack("<d", local["binding"]):
                        mismatches.append(
                            {
                                "tick": tick_c,
                                "field": "shells_binding",
                                "expected": srec[4],
                                "actual": local["binding"],
                            }
                        )
                    for slot in range(4):
                        if struct.pack("<d", srec[5 + slot]) != struct.pack(
                            "<d", local["shell_bindings"][slot]
                        ):
                            mismatches.append(
                                {
                                    "tick": tick_c,
                                    "field": f"shells_b{slot}",
                                    "expected": srec[5 + slot],
                                    "actual": local["shell_bindings"][slot],
                                }
                            )
            pending_collapsed.clear()
            for mrec in pending_multipole:
                mismatches.append(
                    {
                        "tick": tick,
                        "field": "multipole_record",
                        "expected": None,
                        "actual": f"RegionMultipole without RegionCollapsed for region {mrec[3] * 2 + mrec[2]}",
                    }
                )
            pending_multipole.clear()
            for rrec in pending_radial:
                mismatches.append(
                    {
                        "tick": tick,
                        "field": "radial_record",
                        "expected": None,
                        "actual": f"RegionRadial without RegionCollapsed for region {rrec[3] * 2 + rrec[2]}",
                    }
                )
            pending_radial.clear()
            for srec in pending_shells:
                mismatches.append(
                    {
                        "tick": tick,
                        "field": "shells_record",
                        "expected": None,
                        "actual": f"RegionShells without RegionCollapsed for region {srec[3] * 2 + srec[2]}",
                    }
                )
            pending_shells.clear()
            for local in world.last_collapses:
                mismatches.append(
                    {
                        "tick": tick,
                        "field": "collapse_record",
                        "expected": None,
                        "actual": f"missing RegionCollapsed for region {local['region']}",
                    }
                )
            world.last_collapses.clear()
            if world.last_expansion is not None:
                exp = world.last_expansion
                world.last_expansion = None
                if exp["multipole"] and exp["bodies"]:
                    bodies = exp["bodies"]
                    sx = 0.0
                    sy = 0.0
                    for b in bodies:
                        sx += b["mass"] * b["x"]
                        sy += b["mass"] * b["y"]
                    scale = max(abs(exp["target_mx"]), abs(exp["target_my"]), 1e-30)
                    dipole = max(abs(sx - exp["target_mx"]), abs(sy - exp["target_my"])) / scale
                    qxx = 0.0
                    qxy = 0.0
                    qyy = 0.0
                    for b in bodies:
                        dx = b["x"] - exp["target_com_x"]
                        dy = b["y"] - exp["target_com_y"]
                        qxx += b["mass"] * dx * dx
                        qxy += b["mass"] * dx * dy
                        qyy += b["mass"] * dy * dy
                    q_scale = max(abs(exp["target_qxx"]), abs(exp["target_qyy"]), 1e-30)
                    quad = (
                        max(
                            abs(qxx - exp["target_qxx"]),
                            abs(qxy - exp["target_qxy"]),
                            abs(qyy - exp["target_qyy"]),
                        )
                        / q_scale
                    )
                    if not exp.get("transformed"):
                        quad = 0.0
                    energy_delta = abs(subset_energy(exp["bodies"]) - exp["target_energy"]) / max(
                        abs(exp["target_energy"]), 1.0
                    )
                    if exp.get("radial"):
                        binding = 0.0
                        for a in range(len(bodies)):
                            for b2 in range(a + 1, len(bodies)):
                                dx = bodies[b2]["x"] - bodies[a]["x"]
                                dy = bodies[b2]["y"] - bodies[a]["y"]
                                binding += (
                                    bodies[a]["mass"]
                                    * bodies[b2]["mass"]
                                    / math.sqrt(dx * dx + dy * dy + EPS2)
                                )
                        binding_rel = abs(binding - exp["target_binding"]) / max(
                            abs(exp["target_binding"]), 1e-30
                        )
                        radial_deltas.append(
                            (exp["tick"], exp["region"], dipole, quad, binding_rel, energy_delta)
                        )
                    elif exp.get("shells"):
                        by_id = {b["id"]: b for b in bodies}
                        worst_group = 0.0
                        for k, group in enumerate(exp["shell_groups"]):
                            gb = [by_id[i] for i in group]
                            if len(gb) < 2:
                                continue
                            group_binding = 0.0
                            for a in range(len(gb)):
                                for b2 in range(a + 1, len(gb)):
                                    dx = gb[b2]["x"] - gb[a]["x"]
                                    dy = gb[b2]["y"] - gb[a]["y"]
                                    group_binding += (
                                        gb[a]["mass"]
                                        * gb[b2]["mass"]
                                        / math.sqrt(dx * dx + dy * dy + EPS2)
                                    )
                            target = exp["target_shell_bindings"][k]
                            rel = abs(group_binding - target) / max(abs(target), 1e-30)
                            worst_group = max(worst_group, rel)
                        total_binding = 0.0
                        for a in range(len(bodies)):
                            for b2 in range(a + 1, len(bodies)):
                                dx = bodies[b2]["x"] - bodies[a]["x"]
                                dy = bodies[b2]["y"] - bodies[a]["y"]
                                total_binding += (
                                    bodies[a]["mass"]
                                    * bodies[b2]["mass"]
                                    / math.sqrt(dx * dx + dy * dy + EPS2)
                                )
                        total_rel = abs(total_binding - exp["target_binding"]) / max(
                            abs(exp["target_binding"]), 1e-30
                        )
                        shell_deltas.append(
                            (
                                exp["tick"],
                                exp["region"],
                                dipole,
                                quad,
                                worst_group,
                                total_rel,
                                energy_delta,
                                bool(exp.get("sigma_vertex")),
                            )
                        )
                    else:
                        multipole_deltas.append((exp["tick"], exp["region"], dipole, quad, energy_delta))
        elif kind == "snapshot":
            _, population = record
            compared += 1
            if population != body_count:
                mismatches.append(
                    {
                        "tick": last_tick,
                        "field": "snapshot_population",
                        "expected": body_count,
                        "actual": population,
                    }
                )
        elif kind == "totals":
            _, tick, fine, coarse_n, mass, px, py, energy = record
            compared += 1
            w_fine, w_coarse, w_mass, w_px, w_py, w_energy = world.totals()
            if fine != w_fine:
                mismatches.append({"tick": tick, "field": "fine_count", "expected": fine, "actual": w_fine})
            if coarse_n != w_coarse:
                mismatches.append({"tick": tick, "field": "coarse_count", "expected": coarse_n, "actual": w_coarse})
            for name, got, want in (
                ("mass", mass, w_mass),
                ("px", px, w_px),
                ("py", py, w_py),
                ("energy", energy, w_energy),
            ):
                if struct.pack("<d", got) != struct.pack("<d", want):
                    mismatches.append({"tick": tick, "field": name, "expected": got, "actual": want})
        elif kind == "state":
            _, tick, rx, ry, level, population, rhash = record
            compared += 1
            w_level, w_pop, w_hash = world.region_hash(ry * 2 + rx)
            if (level, population) != (w_level, w_pop) or rhash != w_hash:
                mismatches.append(
                    {
                        "tick": tick,
                        "field": "region_state",
                        "expected": (level, population, rhash),
                        "actual": (w_level, w_pop, w_hash),
                    }
                )
        elif kind == "body":
            _, tick, bid, region, level, x, y, vx, vy, mass = record
            compared += 1
            b, w_region, w_level = world.emitted_state(bid)
            ok = (
                region == w_region
                and level == w_level
                and struct.pack("<ddddd", x, y, vx, vy, mass)
                == struct.pack("<ddddd", b["x"], b["y"], b["vx"], b["vy"], b["mass"])
            )
            if not ok:
                mismatches.append(
                    {
                        "tick": tick,
                        "field": f"body_{bid}",
                        "expected": (region, level, x, y, vx, vy),
                        "actual": (w_region, w_level, b["x"], b["y"], b["vx"], b["vy"]),
                    }
                )

    tracked = [
        i
        for i in range(len(world.bodies))
        if world.body_collapsed[i] is None and i not in world.reconstructed
    ]
    if world.contacts:
        # The contact-free reference is meaningless for contact runs
        # (contacts are dissipative by design); drift metrics report 0
        # and check_bounded_drift defers to the contact checks instead.
        end_px = ref_px = 0.0
        end_py = ref_py = 0.0
        end_e = ref_e0 = 1.0
        max_pos_dev = 0.0
        post_exp_dev = 0.0
    elif len(tracked) == len(world.bodies):
        _, _, ref_mass, ref_px, ref_py, ref_e0 = reference.totals()
        _, _, _, end_px, end_py, end_e = world.totals()
    else:
        wsub = [world._state_at(i, world.tick) for i in tracked]
        rsub = [reference.bodies[i] for i in tracked]
        end_px = 0.0
        end_py = 0.0
        for b in wsub:
            end_px += b["mass"] * b["vx"]
            end_py += b["mass"] * b["vy"]
        ref_px = 0.0
        ref_py = 0.0
        for b in rsub:
            ref_px += b["mass"] * b["vx"]
            ref_py += b["mass"] * b["vy"]
        end_e = subset_energy(wsub)
        ref_e0 = subset_energy(rsub)
    ref_scale = max(abs(ref_px), abs(ref_py), 1e-30)
    # --- Contract validation, post-replay (audit ONT-001) ---
    # Collected separately and prepended so contract violations (the root
    # cause) survive the truncated first-20 mismatch detail.
    contract_post = []
    if expected is not None:
        if expected.multipole is True and collapse_events > 0 and multipole_events != collapse_events:
            contract_post.append(
                {
                    "tick": last_tick,
                    "field": "contract_multipole_records",
                    "expected": f"{collapse_events} RegionMultipole record(s) for {collapse_events} collapse(s)",
                    "actual": multipole_events,
                }
            )
        if expected.multipole is False and multipole_events > 0:
            contract_post.append(
                {
                    "tick": last_tick,
                    "field": "contract_multipole_records",
                    "expected": 0,
                    "actual": multipole_events,
                }
            )
        for flag, count, kind in (
            (expected.radial, radial_events, "radial"),
            (expected.shells, shell_events, "shells"),
        ):
            if flag is True and collapse_events > 0 and count != collapse_events:
                contract_post.append(
                    {
                        "tick": last_tick,
                        "field": f"contract_{kind}_records",
                        "expected": collapse_events,
                        "actual": count,
                    }
                )
            if flag is False and count > 0:
                contract_post.append(
                    {
                        "tick": last_tick,
                        "field": f"contract_{kind}_records",
                        "expected": 0,
                        "actual": count,
                    }
                )
        stream_has_params = params_seen
        if expected.contacts is False and (contact_records_seen > 0 or stream_has_params):
            contract_post.append(
                {
                    "tick": last_tick,
                    "field": "contract_contacts",
                    "expected": "no contact records or ContactParams in a non-contact run",
                    "actual": f"{contact_records_seen} contact record(s), params={stream_has_params}",
                }
            )
        for field, want in (
            ("restitution", expected.restitution),
            ("friction", expected.friction),
        ):
            if want is None:
                continue
            if stream_params is None:
                contract_post.append(
                    {
                        "tick": last_tick,
                        "field": "contract_contact_params",
                        "expected": f"{field}={want} requires a ContactParams record",
                        "actual": None,
                    }
                )
            elif struct.pack("<d", stream_params[0 if field == "restitution" else 1]) != struct.pack("<d", want):
                contract_post.append(
                    {
                        "tick": last_tick,
                        "field": "contract_contact_params",
                        "expected": f"{field}={want}",
                        "actual": stream_params[0 if field == "restitution" else 1],
                    }
                )
        if expected.walls is True and stream_params is None:
            contract_post.append(
                {
                    "tick": last_tick,
                    "field": "contract_contact_params",
                    "expected": "walls run must carry a ContactParams record with walls=1",
                    "actual": None,
                }
            )
        if stream_params is not None and expected.walls is not None:
            want_byte = 1 if expected.walls else 0
            if stream_params[2] != want_byte:
                contract_post.append(
                    {
                        "tick": last_tick,
                        "field": "contract_contact_params",
                        "expected": f"walls={want_byte}",
                        "actual": stream_params[2],
                    }
                )
    if contract_post:
        mismatches[0:0] = contract_post
    return {
        "ticks_verified": last_tick,
        "records_compared": compared,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "max_position_deviation": max_pos_dev,
        "momentum_drift": max(abs(end_px - ref_px), abs(end_py - ref_py)) / ref_scale,
        "energy_drift": abs(end_e - ref_e0) / abs(ref_e0),
        "collapse_events": collapse_events,
        "expand_events": world.expand_count,
        "collapse_energy_deltas": collapse_energy_deltas,
        "multipole_events": multipole_events,
        "multipole_deltas": multipole_deltas,
        "radial_events": radial_events,
        "radial_deltas": radial_deltas,
        "shell_events": shell_events,
        "shell_deltas": shell_deltas,
        "post_expansion_deviation": post_exp_dev,
        "contact_events": contact_events,
        "contact_run": world.contacts,
        "contact_restitution": world.restitution if params_seen else 0.0,
        "contact_worst_vn_after": contact_worst_vn_after,
        "contact_min_jn": contact_min_jn if contact_events else 0.0,
        "static_contact_events": static_contact_events,
        "coarse_static_contact_events": coarse_static_contact_events,
        "final_world_hash": world.world_hash(),
    }


def check_reference_match_gravity(summary: dict, *, threshold: float = 0.0) -> DiagnosticResult:
    count = summary["mismatch_count"]
    return DiagnosticResult(
        name="ontos_reference_match",
        passed=count <= threshold,
        threshold=float(threshold),
        value=float(count),
        detail={
            "ticks_verified": summary["ticks_verified"],
            "records_compared": summary["records_compared"],
            "first_mismatches": summary["mismatches"],
        },
    )


def check_bounded_drift(summary: dict, *, pos_tol: float = 5e-2, mom_tol: float = 5e-2, energy_tol: float = 1e-3) -> DiagnosticResult:
    pos = summary["max_position_deviation"]
    mom = summary["momentum_drift"]
    energy = summary["energy_drift"]
    contact_run = summary.get("contact_run", False)
    return DiagnosticResult(
        name="ontos_window_drift",
        passed=contact_run or (pos <= pos_tol and mom <= mom_tol and energy <= energy_tol),
        threshold=float(pos_tol),
        value=float(max(pos, mom, energy)),
        detail={
            "max_position_deviation": pos,
            "momentum_drift_relative": mom,
            "energy_drift_relative": energy,
            "tolerances": {"position": pos_tol, "momentum": mom_tol, "energy": energy_tol},
            "note": "contact run: reference drift not applicable (dissipative contact)" if contact_run else None,
        },
    )


def check_reconstruction_error(summary: dict, *, pos_tol: float = 64.0) -> DiagnosticResult:
    """Spec section 19: post-expansion continuation vs the all-fine reference.

    Reconstruction positions are com + jitter, so the honest bound is
    region-scale. Bodies that were never reconstructed are excluded (their
    deviation is window/monopole drift, owned by check_bounded_drift).
    """
    expanded = summary.get("expand_events", 0) > 0
    dev = summary.get("post_expansion_deviation", 0.0)
    return DiagnosticResult(
        name="ontos_reconstruction_error",
        passed=(not expanded) or dev <= pos_tol,
        threshold=float(pos_tol),
        value=float(dev),
        detail={
            "post_expansion_deviation": dev,
            "expansions": summary.get("expand_events", 0),
            "note": None if expanded else "no expansion in stream; reconstruction error unmeasured",
        },
    )


def check_collapse_energy(summary: dict, *, tol: float = 8.0) -> DiagnosticResult:
    """Spec section 19: synthesized-set energy vs the collapse record's energy."""
    worst = 0.0
    deltas = summary.get("collapse_energy_deltas", [])
    for _, _, rec_e, syn_e in deltas:
        rel = abs(syn_e - rec_e) / max(abs(rec_e), 1.0)
        worst = max(worst, rel)
    return DiagnosticResult(
        name="ontos_collapse_energy",
        passed=worst <= tol,
        threshold=float(tol),
        value=float(worst),
        detail={"collapse_events": len(deltas), "worst_relative_delta": worst},
    )


def check_multipole_match(summary: dict, *, dipole_tol: float = 1e-12, quad_tol: float = 1e-9) -> DiagnosticResult:
    """Spec section 20: post-expansion dipole closure and quadrupole match.

    The synthesized set's mass-weighted position sum must close on the
    collapse record's (mx, my) to rounding, and its second central moments
    on (qxx, qxy, qyy) to rounding of the transform arithmetic (measured
    when the transform ran; N < 3 cycles pin the dipole only). The
    expansion-set energy delta vs the collapse record is reported as
    detail (it is not an invariant of the synthesis).
    """
    deltas = summary.get("multipole_deltas", [])
    worst_dipole = 0.0
    worst_quad = 0.0
    worst_energy = 0.0
    for _, _, dipole, quad, energy_delta in deltas:
        worst_dipole = max(worst_dipole, dipole)
        worst_quad = max(worst_quad, quad)
        worst_energy = max(worst_energy, energy_delta)
    expanded = bool(deltas)
    return DiagnosticResult(
        name="ontos_multipole_match",
        passed=(not expanded) or (worst_dipole <= dipole_tol and worst_quad <= quad_tol),
        threshold=float(quad_tol),
        value=float(max(worst_dipole, worst_quad)),
        detail={
            "multipole_expansions": len(deltas),
            "worst_dipole_relative": worst_dipole,
            "worst_quadrupole_relative": worst_quad,
            "worst_energy_relative": worst_energy,
            "tolerances": {"dipole": dipole_tol, "quadrupole": quad_tol},
            "note": None if expanded else "no multipole expansion in stream; match unmeasured",
        },
    )


def check_contact_resolution(summary: dict, *, vn_tol: float = 1e-12) -> DiagnosticResult:
    """Spec sections 21 + 24: contact impulse invariants.

    Every emitted record must carry a positive impulse, and the relative
    normal speed of a resolving pair, measured immediately after its own
    impulse, must close on -e * vn within rounding of the impulse
    arithmetic (later impulses in the same pass may perturb other
    pairs; the next tick's pass resolves those).
    """
    events = summary.get("contact_events", 0)
    has_contacts = events > 0
    e = summary.get("contact_restitution", 0.0)
    worst_vn = summary.get("contact_worst_vn_after", 0.0)
    min_jn = summary.get("contact_min_jn", 0.0)
    ok = (not has_contacts) or (min_jn > 0.0 and worst_vn <= vn_tol)
    return DiagnosticResult(
        name="ontos_contact_resolution",
        passed=ok,
        threshold=float(vn_tol),
        value=float(worst_vn),
        detail={
            "contact_events": events,
            "static_contact_events": summary.get("static_contact_events", 0),
            "restitution": e,
            "min_jn": min_jn,
            "worst_post_impulse_residual": worst_vn,
            "note": None if has_contacts else "no contact records in stream; resolution unmeasured",
        },
    )


def check_radial_shape(
    summary: dict,
    *,
    dipole_tol: float = 1e-12,
    binding_tol: float = 1e-9,
    energy_tol: float = 1e-9,
    quad_tol: float = 4.0,
) -> DiagnosticResult:
    """Spec section 23: binding-matched radial-shape synthesis.

    The synthesized set's internal potential (binding formula over the
    synthesized positions) must close on the RegionRadial record, its
    total energy on the RegionCollapsed energy (the radial scale plus
    the spread-scale quadratic drive both to rounding), and its dipole
    on (mx, my). The quadrupole closes only to lambda^2 of the record
    (the radial scale trades tensor exactness for binding exactness);
    it is reported and bounded, not exact.
    """
    deltas = summary.get("radial_deltas", [])
    worst_dipole = 0.0
    worst_binding = 0.0
    worst_energy = 0.0
    worst_quad = 0.0
    for _, _, dipole, quad, binding, energy in deltas:
        worst_dipole = max(worst_dipole, dipole)
        worst_binding = max(worst_binding, binding)
        worst_energy = max(worst_energy, energy)
        worst_quad = max(worst_quad, quad)
    expanded = bool(deltas)
    return DiagnosticResult(
        name="ontos_radial_shape",
        passed=(
            not expanded
            or (
                worst_dipole <= dipole_tol
                and worst_binding <= binding_tol
                and worst_energy <= energy_tol
                and worst_quad <= quad_tol
            )
        ),
        threshold=float(binding_tol),
        value=float(max(worst_dipole, worst_binding, worst_energy)),
        detail={
            "radial_expansions": len(deltas),
            "worst_dipole_relative": worst_dipole,
            "worst_binding_relative": worst_binding,
            "worst_energy_relative": worst_energy,
            "worst_quadrupole_relative": worst_quad,
            "tolerances": {
                "dipole": dipole_tol,
                "binding": binding_tol,
                "energy": energy_tol,
                "quadrupole": quad_tol,
            },
            "note": None if expanded else "no radial expansion in stream; shape unmeasured",
        },
    )


def check_shell_shape(
    summary: dict,
    *,
    dipole_tol: float = 1e-12,
    binding_tol: float = 1e-9,
    energy_tol: float = 1e-9,
    total_tol: float = 2.0,
    vertex_tol: float = 1.0,
    quad_tol: float = 16.0,
) -> DiagnosticResult:
    """Spec section 25: per-shell radial synthesis.

    Each shell's intra-shell binding, measured on the final synthesized
    positions over the expansion's solve-time shell membership (rank
    shells of the globally scaled base — deterministic replay state),
    closes on its RegionShells record; the synthesized-set total energy
    closes on the RegionCollapsed energy; the dipole closes on (mx, my).
    The total binding is retained (the global scale closes it exactly;
    the per-shell corrections trade a bounded total deviation for
    per-shell exactness) and the quadrupole closes only to the
    per-shell scale mix — both reported and bounded, the honest trades.
    """
    deltas = summary.get("shell_deltas", [])
    worst_dipole = 0.0
    worst_binding = 0.0
    worst_energy = 0.0
    worst_total = 0.0
    worst_quad = 0.0
    worst_vertex_energy = 0.0
    vertex_cycles = 0
    for _, _, dipole, quad, binding, total, energy, vertex in deltas:
        worst_dipole = max(worst_dipole, dipole)
        worst_binding = max(worst_binding, binding)
        worst_quad = max(worst_quad, quad)
        worst_total = max(worst_total, total)
        if vertex:
            vertex_cycles += 1
            worst_vertex_energy = max(worst_vertex_energy, energy)
        else:
            worst_energy = max(worst_energy, energy)
    expanded = bool(deltas)
    return DiagnosticResult(
        name="ontos_shell_shape",
        passed=(
            not expanded
            or (
                worst_dipole <= dipole_tol
                and worst_binding <= binding_tol
                and worst_energy <= energy_tol
                and worst_total <= total_tol
                and worst_vertex_energy <= vertex_tol
                and worst_quad <= quad_tol
            )
        ),
        threshold=float(binding_tol),
        value=float(max(worst_dipole, worst_binding, worst_energy)),
        detail={
            "shell_expansions": len(deltas),
            "worst_dipole_relative": worst_dipole,
            "worst_shell_binding_relative": worst_binding,
            "worst_total_binding_relative": worst_total,
            "worst_energy_relative": worst_energy,
            "worst_quadrupole_relative": worst_quad,
            "vertex_cycles": vertex_cycles,
            "worst_vertex_energy_relative": worst_vertex_energy,
            "tolerances": {
                "dipole": dipole_tol,
                "shell_binding": binding_tol,
                "total_binding": total_tol,
                "energy": energy_tol,
                "vertex_energy": vertex_tol,
                "quadrupole": quad_tol,
            },
            "note": None if expanded else "no shell expansion in stream; shape unmeasured",
        },
    )


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m simval.ontos",
        description="Verify an ontos record stream (life or gravity) against this independent reference",
    )
    parser.add_argument("stream", help="path to a .stream file")
    parser.add_argument("seed", type=int, help="world seed the stream was produced with")
    parser.add_argument(
        "--test-ic",
        choices=["wallshot", "coarsehit"],
        help="test-only corpus initial conditions the stream was produced with "
        "(ontos --test-ic; see ontos docs/DESIGN.md corpus coverage)",
    )
    g = parser.add_argument_group(
        "expected-run contract (audit ONT-007)",
        "the verifier must know what run was requested, not just replay the "
        "stream: pass --metadata ontos.json or the same CLI flags given to ontos",
    )
    g.add_argument("--metadata", metavar="PATH", help="ontos.json run-contract file")
    g.add_argument("--mode", choices=["life", "gravity"])
    g.add_argument("--ticks", type=int)
    g.add_argument("--bodies", type=int)
    g.add_argument("--observer", type=int)
    g.add_argument("--contacts", action="store_true")
    g.add_argument("--radial", action="store_true")
    g.add_argument("--shells", action="store_true")
    g.add_argument("--walls", action="store_true")
    g.add_argument("--restitution", type=float)
    g.add_argument("--friction", type=float)
    g.add_argument("--demote-at", nargs=3, action="append", default=[], metavar=("T", "RX", "RY"))
    g.add_argument("--promote-at", nargs=3, action="append", default=[], metavar=("T", "RX", "RY"))
    g.add_argument("--collapse-at", nargs=3, action="append", default=[], metavar=("T", "RX", "RY"))
    g.add_argument("--expand-at", nargs=3, action="append", default=[], metavar=("T", "RX", "RY"))
    g.add_argument("--demote", nargs=2, action="append", default=[], metavar=("RX", "RY"))
    g.add_argument("--promote", nargs=2, action="append", default=[], metavar=("RX", "RY"))
    g.add_argument(
        "--no-contract",
        action="store_true",
        help="bare-stream replay with NO expected-run contract. Physics-only; "
        "a stream that happens to match its own replay still passes. NOT for CI.",
    )
    args = parser.parse_args(argv)

    gravity_event_levels = {
        "demote-at": 0,
        "promote-at": 1,
        "expand-at": 1,
        "collapse-at": 2,
    }

    def _contract_meta_from_flags(is_gravity_stream: bool) -> dict:
        explicit = any(
            getattr(args, k) is not None
            for k in ("mode", "ticks", "bodies", "observer", "restitution", "friction")
        ) or any(
            getattr(args, k)
            for k in (
                "contacts", "radial", "shells", "walls",
                "demote_at", "promote_at", "collapse_at", "expand_at",
                "demote", "promote",
            )
        )
        if args.metadata and explicit:
            raise ValueError("--metadata and explicit contract flags are mutually exclusive")
        if args.metadata:
            import json as _json

            return _json.loads(Path(args.metadata).read_text())
        if not explicit:
            return {}
        mode = args.mode or ("gravity" if is_gravity_stream else "life")
        meta = {"mode": mode, "seed": args.seed, "ticks": args.ticks}
        if args.test_ic:
            meta["test_ic"] = args.test_ic
        if mode == "gravity":
            if args.ticks is None or args.bodies is None:
                raise ValueError("gravity contract needs --ticks and --bodies (or --metadata)")
            meta["bodies"] = args.bodies
            events = []
            for kind in ("demote-at", "promote-at", "collapse-at", "expand-at"):
                for t, rx, ry in getattr(args, kind.replace("-", "_")):
                    events.append([int(t), int(rx), int(ry), gravity_event_levels[kind]])
            meta["events"] = events
            meta["contacts"] = args.contacts
            meta["radial"] = args.radial
            meta["shells"] = args.shells
            meta["walls"] = args.walls
            meta["multipole"] = True  # the ontos CLI default (spec section 20)
            if args.observer is not None:
                meta["observer"] = args.observer
            if args.restitution is not None:
                meta["restitution"] = args.restitution
            if args.friction is not None:
                meta["friction"] = args.friction
        else:
            events = []
            for kind in ("demote", "promote"):
                for rx, ry in getattr(args, kind):
                    events.append([kind, int(rx), int(ry)])
            meta["events"] = events
        return meta

    try:
        data = Path(args.stream).read_bytes()
        is_gravity_stream = len(data) >= 8 and data[4:8] == struct.pack("<I", 2)
        meta = _contract_meta_from_flags(is_gravity_stream)
        has_contract = bool(meta) or (args.metadata is not None)
        if not has_contract and not args.no_contract:
            raise ValueError(
                "no expected-run contract: pass --metadata ontos.json or the same "
                "flags given to ontos (--ticks/--bodies/--events...), or --no-contract "
                "for an explicit bare-stream replay (not for CI)"
            )
        if is_gravity_stream:
            expected = None
            if has_contract:
                if meta.get("mode", "gravity") != "gravity":
                    raise ValueError("contract declares mode=life but the stream is version 2 (gravity)")
                expected = GravityContract.from_metadata(meta)
            elif args.no_contract:
                print(
                    "simval.ontos: WARNING: bare-stream replay with no expected-run "
                    "contract (audit ONT-007): body count, horizon, event schedule and "
                    "feature modes are NOT validated. Do not use this mode in CI.",
                    file=sys.stderr,
                )
            summary = verify_stream_gravity(
                args.stream, args.seed, args.test_ic, expected=expected
            )
            ok = summary["mismatch_count"] == 0
            contract_note = "" if expected is not None else " [NO CONTRACT]"
            print(
                f"ontos gravity stream: {'OK' if ok else 'MISMATCH'}{contract_note} | "
                f"ticks={summary['ticks_verified']} "
                f"records={summary['records_compared']} mismatches={summary['mismatch_count']} "
                f"max_pos_dev={summary['max_position_deviation']:.3e} "
                f"mom_drift={summary['momentum_drift']:.3e} energy_drift={summary['energy_drift']:.3e} "
                f"final_world_hash={summary['final_world_hash']:016x}"
            )
            for m in summary["mismatches"]:
                print(f"  MISMATCH: {m}")
            return 0 if ok else 1
        from simval.ontos import verify_stream

        if has_contract:
            if meta.get("mode", "life") != "life":
                raise ValueError("contract declares mode=gravity but the stream is version 1 (life)")
        elif args.no_contract:
            print(
                "simval.ontos: WARNING: bare-stream replay with no expected-run "
                "contract (audit ONT-007): the life event schedule is NOT validated. "
                "Do not use this mode in CI.",
                file=sys.stderr,
            )
        summary = verify_stream(args.stream, args.seed)
        problems = []
        if has_contract:
            from simval.ontos import life_contract_problems

            problems = life_contract_problems(meta, summary, parse_stream(args.stream)[1])
        ok = summary["mismatch_count"] == 0 and summary["tick_monotonic"] and not problems
        contract_note = "" if has_contract else " [NO CONTRACT]"
        print(
            f"ontos stream: {'OK' if ok else 'MISMATCH'}{contract_note} | ticks={summary['ticks_verified']} "
            f"records={summary['records_compared']} mismatches={summary['mismatch_count']} "
            f"final_population={summary['final_population']} "
            f"final_world_hash={summary['final_world_hash']:016x}"
        )
        for m in summary["mismatches"]:
            print(f"  MISMATCH: {m}")
        for p in problems:
            print(f"  CONTRACT: {p}")
        return 0 if ok else 1
    except (FileNotFoundError, ValueError) as e:
        print(f"simval.ontos: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
