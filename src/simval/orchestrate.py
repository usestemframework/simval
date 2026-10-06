"""Orchestrator: drive ontos runs across a parameter grid, verify each one.

FUTURE-ARC arc step 2 — simval runs the sweep, not just judges it. Each grid
entry is executed by the ontos binary into its own run-dir, then verified by
the existing oracle (engine adapter -> verify_stream_gravity -> checks) and
tabulated next to its peers; outlier runs are flagged by robust statistics.

Grid file format — a JSON list of run specs (YAML accepted when PyYAML is
installed and the file ends in .yaml/.yml):

    [
      {"name": "all_fine_42", "mode": "gravity", "ticks": 100,
       "seed": 42, "bodies": 8},
      {"name": "window_42", "mode": "gravity", "ticks": 120, "seed": 42,
       "bodies": 8, "events": [["demote-at", 20, 0, 0], ["promote-at", 80, 0, 0]]},
      {"name": "observer_5", "mode": "gravity", "ticks": 300, "seed": 5,
       "bodies": 16, "observer": 777},
      {"name": "collapse_11", "mode": "gravity", "ticks": 200, "seed": 11,
       "bodies": 8, "events": [["collapse-at", 30, 1, 1], ["expand-at", 120, 1, 1]]},
      {"name": "contact_11", "mode": "gravity", "ticks": 400, "seed": 11,
       "bodies": 32, "contacts": true}
    ]

    Fields per spec:
      name     run label + run-dir name (default "run-<index>")
      mode     "gravity" (default) or "life"
      ticks    simulation length (default 100)
      seed     world seed (default 42)
      bodies   gravity only, body count (default 8)
      events   gravity: ["demote-at"|"promote-at"|"collapse-at"|"expand-at", t, rx, ry]
               life:    ["demote"|"promote", rx, ry]
      observer gravity only, observer offset (enables the zoom-policy check)
      contacts gravity only, enable contact mode (spec section 21)

Runs are sequential by design (determinism first); a run that fails to
generate or verify gets an "_error" row instead of aborting the grid.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from simval.report import rule_like

GRAVITY_EVENT_LEVELS = {"demote-at": 0, "promote-at": 1, "expand-at": 1, "collapse-at": 2}
LIFE_EVENTS = ("demote", "promote")

KEY_METRICS = [
    "ticks", "mismatch_count",
    "max_position_deviation", "momentum_drift", "energy_drift",
    "post_expansion_deviation", "collapse_events", "expand_events",
    "contact_events", "checks_failed", "wall_s",
]

# Display-only short names for the table header (full keys stay in the
# result rows and the --out JSON, untouched).
COLUMN_ALIASES = {
    "ticks": "ticks",
    "mismatch_count": "mismatch",
    "max_position_deviation": "pos_dev",
    "momentum_drift": "mom_drift",
    "energy_drift": "en_drift",
    "post_expansion_deviation": "postexp_dev",
    "collapse_events": "collapse",
    "expand_events": "expand",
    "contact_events": "contact",
    "checks_failed": "chk_fail",
    "wall_s": "wall_s",
}

_RUN_W = 20
_STATUS_W = 6
_METRIC_W = 11


def row_ok(row: dict) -> bool:
    """The single clean-row predicate: no error, zero mismatches, zero
    failed checks. A row that does not even report mismatch_count counts
    as dirty (fails closed, like the CLI exit code always has)."""
    mism = row.get("mismatch_count", 1)
    chk = row.get("checks_failed") or 0
    return "_error" not in row and mism == 0 and chk == 0

DRIFT_METRICS = [
    "max_position_deviation", "momentum_drift", "energy_drift",
    "post_expansion_deviation", "collapse_energy_worst", "multipole_worst",
]


def find_ontos_bin(explicit=None) -> Path:
    """Locate the ontos binary: explicit path > $ONTOS_BIN > PATH."""
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"ontos binary not found: {path}")
        return path
    env = os.environ.get("ONTOS_BIN")
    if env:
        path = Path(env)
        if not path.exists():
            raise FileNotFoundError(f"$ONTOS_BIN does not exist: {path}")
        return path
    found = shutil.which("ontos")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "ontos binary not found: pass ontos_bin=, set ONTOS_BIN, or put ontos on PATH"
    )


def load_grid(path) -> list[dict]:
    """Load a grid file: JSON by default, YAML when PyYAML is available."""
    grid_path = Path(path)
    text = grid_path.read_text()
    if grid_path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ValueError(f"PyYAML required for YAML grids: {grid_path}") from e
        specs = yaml.safe_load(text)
    else:
        specs = json.loads(text)
    if not isinstance(specs, list) or not all(isinstance(s, dict) for s in specs):
        raise ValueError(f"grid file must be a list of run specs: {grid_path}")
    if not specs:
        raise ValueError(f"grid file contains no run specs: {grid_path}")
    return specs


def validate_run_names(names: list[str]) -> None:
    """Reject run names that are not safe plain directory names, and
    duplicates — before any filesystem mutation happens (audit ORCH-001).
    Display names never become paths: run directories use internal ids."""
    seen: set[str] = set()
    for name in names:
        candidate = Path(name)
        if (
            not name
            or candidate.is_absolute()
            or candidate.name != name
            or "/" in name
            or "\\" in name
            or ".." in candidate.parts
        ):
            raise ValueError(
                f"unsafe grid run name {name!r}: must be a plain name with no "
                "separators, '..', or absolute components"
            )
        if name in seen:
            raise ValueError(f"duplicate grid run name {name!r}")
        seen.add(name)


def run_grid(
    specs, *, ontos_bin=None, workdir=None, cell_timeout_s: float = 1800.0, progress=None
) -> list[dict]:
    """Generate + verify every spec sequentially; one result row per run.

    cell_timeout_s bounds each ontos invocation (a hung producer must not
    hang the grid): on expiry the child is killed and the cell records an
    `_error` row; there are NO implicit retries (audit ORCH-004).

    progress(row, index, total), when given, is called once per completed
    row (display only — it can print, log, or be ignored).
    """
    import math

    if not (isinstance(cell_timeout_s, (int, float)) and math.isfinite(cell_timeout_s) and cell_timeout_s > 0):
        raise ValueError(f"cell_timeout_s must be a finite positive number, got {cell_timeout_s!r}")
    binary = find_ontos_bin(ontos_bin)
    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="simval-grid-")
        root = Path(tmp.name)
    else:
        root = Path(workdir)
        root.mkdir(parents=True, exist_ok=True)
    names = [
        str(s.get("name") or f"run-{i}") if isinstance(s, dict) else f"run-{i}"
        for i, s in enumerate(specs)
    ]
    # Preflight every name before touching the filesystem.
    validate_run_names(names)
    try:
        rows = []
        for i, raw in enumerate(specs):
            cell = f"cell-{i:04d}"
            run_dir = (root / cell).resolve()
            if root.resolve() not in run_dir.parents:
                raise ValueError(f"internal cell id escaped the grid root: {run_dir}")
            row: dict = {"run": names[i], "cell": cell}
            try:
                # Normalization lives inside the per-cell failure boundary
                # (audit ORCH-002): a malformed cell gets an _error row and
                # the grid continues.
                spec = normalize_spec(raw, i)
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                gen_wall = generate_run(spec, run_dir, binary, timeout_s=cell_timeout_s)
                row.update(verify_run(run_dir))
                row.update(
                    {
                        "ticks": spec["ticks"],
                        "events": len(spec["events"]),
                        "gen_wall_s": gen_wall,
                        "wall_s": round(gen_wall + row["verify_wall_s"], 3),
                    }
                )
                row.pop("verify_wall_s", None)
            except Exception as e:
                row["_error"] = f"cell {i} ({names[i]}): {e}"[:200]
            rows.append(row)
            if progress is not None:
                progress(row, i, len(specs))
        return rows
    finally:
        if tmp is not None:
            tmp.cleanup()


def normalize_spec(spec: dict, index: int = 0) -> dict:
    """Fill defaults and validate one grid entry."""
    import math

    name = spec.get("name") or f"run-{index}"
    mode = spec.get("mode", "gravity")
    if mode not in ("gravity", "life"):
        raise ValueError(f"{name}: unknown mode {mode!r}")
    out = {
        "name": str(name),
        "mode": mode,
        "ticks": int(spec.get("ticks", 100)),
        "seed": int(spec.get("seed", 42)),
        "bodies": int(spec.get("bodies", 8)),
        "events": [],
        "observer": None,
        "contacts": False,
        "radial": False,
        "shells": False,
        "restitution": None,
        "friction": None,
        "walls": False,
    }
    if mode == "gravity":
        out["contacts"] = bool(spec.get("contacts", False))
        out["radial"] = bool(spec.get("radial", False))
        out["shells"] = bool(spec.get("shells", False))
        out["walls"] = bool(spec.get("walls", False))
        if out["walls"]:
            out["contacts"] = True
        if out["radial"] and out["shells"]:
            raise ValueError(f"{name}: radial and shells are exclusive")
        if spec.get("restitution") is not None:
            e = float(spec["restitution"])
            if not (math.isfinite(e) and 0.0 <= e <= 1.0):
                raise ValueError(f"{name}: restitution must be finite in [0,1], got {e}")
            out["restitution"] = e
            out["contacts"] = True
        if spec.get("friction") is not None:
            fr = float(spec["friction"])
            if not (math.isfinite(fr) and fr >= 0.0):
                raise ValueError(f"{name}: friction must be finite and >= 0, got {fr}")
            out["friction"] = fr
            out["contacts"] = True
    if mode == "gravity" and spec.get("observer") is not None:
        out["observer"] = int(spec["observer"])
    for event in spec.get("events", []):
        kind = event[0]
        if mode == "gravity":
            if kind not in GRAVITY_EVENT_LEVELS or len(event) != 4:
                raise ValueError(
                    f"{name}: gravity events must be [kind, t, rx, ry] with kind in "
                    f"{sorted(GRAVITY_EVENT_LEVELS)}; got {list(event)}"
                )
            out["events"].append((kind, int(event[1]), int(event[2]), int(event[3])))
        else:
            if kind not in LIFE_EVENTS or len(event) != 3:
                raise ValueError(
                    f"{name}: life events must be [kind, rx, ry] with kind in "
                    f"{list(LIFE_EVENTS)}; got {list(event)}"
                )
            out["events"].append((kind, int(event[1]), int(event[2])))
    return out


def _ontos_json(spec: dict) -> dict:
    """The run contract artifact: every CLI-affecting field is persisted so
    the verifier can check the stream against what was requested (ONT-001)."""
    meta = {"mode": spec["mode"], "seed": spec["seed"], "ticks": spec["ticks"]}
    if spec["mode"] == "gravity":
        meta["bodies"] = spec["bodies"]
        meta["events"] = [
            [t, rx, ry, GRAVITY_EVENT_LEVELS[kind]] for kind, t, rx, ry in spec["events"]
        ]
        meta["contacts"] = spec["contacts"]
        meta["radial"] = spec["radial"]
        meta["shells"] = spec["shells"]
        meta["walls"] = spec["walls"]
        meta["multipole"] = True  # the ontos CLI default (spec section 20)
        if spec["observer"] is not None:
            meta["observer"] = spec["observer"]
        if spec["restitution"] is not None:
            meta["restitution"] = spec["restitution"]
        if spec["friction"] is not None:
            meta["friction"] = spec["friction"]
    else:
        meta["events"] = [[kind, rx, ry] for kind, rx, ry in spec["events"]]
    return meta


def _cli_args(spec: dict, stream_path: Path) -> list[str]:
    args = ["--mode", spec["mode"], "--ticks", str(spec["ticks"]), "--seed", str(spec["seed"])]
    if spec["mode"] == "gravity":
        args += ["--bodies", str(spec["bodies"])]
        for kind, t, rx, ry in spec["events"]:
            args += [f"--{kind}", str(t), str(rx), str(ry)]
        if spec["observer"] is not None:
            args += ["--observer", str(spec["observer"])]
        if spec["contacts"]:
            args += ["--contacts"]
        if spec["radial"]:
            args += ["--radial"]
        if spec["shells"]:
            args += ["--shells"]
        if spec["restitution"] is not None:
            args += ["--restitution", str(spec["restitution"])]
        if spec["friction"] is not None:
            args += ["--friction", str(spec["friction"])]
        if spec["walls"]:
            args += ["--walls"]
    else:
        for kind, rx, ry in spec["events"]:
            args += [f"--{kind}", str(rx), str(ry)]
    args += ["--out", str(stream_path)]
    return args


def generate_run(spec: dict, run_dir: Path, ontos_bin, *, timeout_s: float | None = None) -> float:
    """Run the ontos binary for one spec into run_dir (ontos.stream + ontos.json).

    timeout_s bounds the invocation: on expiry subprocess kills the child
    and TimeoutExpired is re-raised as a per-cell error. No retries (audit
    ORCH-004) — the caller records the failure and moves on."""
    run_dir.mkdir(parents=True, exist_ok=True)
    stream_path = run_dir / "ontos.stream"
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(ontos_bin)] + _cli_args(spec, stream_path),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"ontos did not finish within {timeout_s}s (killed; no retry)"
        ) from e
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"ontos exited {proc.returncode}: {proc.stderr.strip()[:200]}")
    (run_dir / "ontos.json").write_text(json.dumps(_ontos_json(spec)))
    return wall


def verify_run(run_dir) -> dict:
    """Verify one ontos run-dir through the engine adapter + checks."""
    from simval.context import select_engine
    from simval.pipeline import run_checks

    run = Path(run_dir)
    engine = select_engine(run)
    if engine.name != "ontos":
        raise ValueError(f"not an ontos run-dir: {run}")
    start = time.perf_counter()
    ctx = engine.load_context(run, selection="default")
    checks = run_checks(ctx)
    wall = time.perf_counter() - start

    summary = ctx.extra.get("ontos_gravity_summary") or ctx.extra["ontos_summary"]
    row = {
        "mode": ctx.run_params.get("mode"),
        "seed": ctx.run_params.get("seed"),
        "ticks_verified": summary["ticks_verified"],
        "records_compared": summary["records_compared"],
        "mismatch_count": summary["mismatch_count"],
        "final_world_hash": f"{summary['final_world_hash']:016x}",
        "checks": [{"name": c.name, "passed": c.passed, "value": float(c.value)} for c in checks],
        "checks_failed": sum(not c.passed for c in checks),
        "verify_wall_s": wall,
    }
    if "ontos_gravity_summary" in ctx.extra:
        worst = 0.0
        for _, _, rec_e, syn_e in summary.get("collapse_energy_deltas", []):
            worst = max(worst, abs(syn_e - rec_e) / max(abs(rec_e), 1.0))
        mp_worst = 0.0
        for _, _, dipole, quad, _energy in summary.get("multipole_deltas", []):
            mp_worst = max(mp_worst, dipole, quad)
        rad_worst = 0.0
        for _, _, dipole, quad, binding, energy in summary.get("radial_deltas", []):
            rad_worst = max(rad_worst, binding, energy)
        row.update(
            {
                "max_position_deviation": summary["max_position_deviation"],
                "momentum_drift": summary["momentum_drift"],
                "energy_drift": summary["energy_drift"],
                "post_expansion_deviation": summary["post_expansion_deviation"],
                "collapse_events": summary["collapse_events"],
                "expand_events": summary["expand_events"],
                "collapse_energy_worst": worst,
                "multipole_events": summary.get("multipole_events", 0),
                "multipole_worst": mp_worst,
                "radial_events": summary.get("radial_events", 0),
                "radial_worst": rad_worst,
                "contact_events": summary.get("contact_events", 0),
            }
        )
    else:
        row["final_population"] = summary["final_population"]
    return row


def tabulate(results) -> str:
    """Render result rows as a sweep-style table (one string, newline-terminated).

    Display only: a status column (ok / FAIL / ERROR) derived by row_ok,
    short column aliases, and a dashed rule under the header. The rows
    themselves (and the --out JSON) keep the full metric keys.
    """
    keys = [k for k in KEY_METRICS if any(k in r for r in results)]

    def col(k):
        return COLUMN_ALIASES.get(k, k[:_METRIC_W])

    header = (
        f"  {'run':<{_RUN_W}} {'status':<{_STATUS_W}} "
        + " ".join(f"{col(k):>{_METRIC_W}}" for k in keys)
    ).rstrip()
    lines = [header, rule_like(header)]
    for r in results:
        if "_error" in r:
            lines.append(f"  {r['run']:<{_RUN_W}}  ERROR: {r['_error']}")
            continue
        status = "ok" if row_ok(r) else "FAIL"
        cells = []
        for k in keys:
            v = r.get(k)
            cells.append("-".rjust(_METRIC_W) if v is None else f"{v:>{_METRIC_W}.3g}")
        lines.append(
            f"  {r['run']:<{_RUN_W}} {status:<{_STATUS_W}} " + " ".join(cells)
        )
    return "\n".join(lines) + "\n"


def outliers(results, *, k: float = 3.0) -> list[dict]:
    """Runs whose any drift metric exceeds the grid median by k MADs.

    For each drift metric the median and median absolute deviation are taken
    over the runs that report it (>= 2 runs required); a run is flagged when
    |value - median| > k * MAD. With MAD == 0 (e.g. bit-identical metrics)
    any nonzero deviation flags.
    """
    flags = []
    for metric in DRIFT_METRICS:
        vals = [r[metric] for r in results if isinstance(r.get(metric), (int, float))]
        if len(vals) < 2:
            continue
        med = statistics.median(vals)
        mad = statistics.median(abs(v - med) for v in vals)
        for r in results:
            v = r.get(metric)
            if not isinstance(v, (int, float)) or abs(v - med) <= k * mad:
                continue
            flags.append({"run": r["run"], "metric": metric, "value": v, "median": med, "mad": mad})
    return flags
