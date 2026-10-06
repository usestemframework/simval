"""Plain-text report formatting shared by the CLI surfaces.

Display only: nothing in this module participates in a verdict, a
tolerance decision, or a digest. Keeping the rendering here means the
command handlers stay about orchestration, and every table in the CLI
shares one visual language (flags, alignment, rules).
"""
from __future__ import annotations

PASS_FLAG = "PASS"
FAIL_FLAG = "FAIL"


def fmt_num(v, width: int = 10) -> str:
    """Right-aligned plain number for table cells: ints without decimals,
    floats at 4 significant digits, None (and non-numbers) as a dash or
    their string form."""
    if v is None:
        return "-".rjust(width)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v).rjust(width)
    if isinstance(v, int):
        return str(v).rjust(width)
    return f"{v:.4g}".rjust(width)


def fmt_dur(seconds: float | None) -> str:
    """Human wall-clock duration: 0.42s, 63.4s, 2m 04s, 1h 07m."""
    if seconds is None or seconds < 0:
        return "-"
    if seconds < 60:
        return f"{seconds:.2f}s".replace(".00s", "s")
    total = round(seconds)
    m, s = divmod(total, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def rule_like(line: str) -> str:
    """A dashed rule exactly as wide as `line`'s content, keeping its
    indent — the separator drawn under a table header."""
    stripped = line.rstrip()
    indent = len(stripped) - len(stripped.lstrip())
    return " " * indent + "-" * (len(stripped) - indent)


def tol_str(kind: str, tol) -> str:
    """A tolerance spec as one readable token: 'rel 0.1', 'exact',
    'interval [0, 1]', 'max 1', ..."""
    if kind == "exact" or kind == "ignore":
        return kind
    if kind == "interval":
        lo, hi = tol
        return f"interval [{lo:g}, {hi:g}]"
    if tol is None:
        return kind
    return f"{kind} {tol:g}"


def render_check_row(r: dict) -> str:
    """One diagnostics row of `simval diagnose` / `simval run`.

    PASS rows show value and threshold; FAIL rows additionally show how
    many times the value exceeds the threshold (checks are upper bounds:
    value must be <= threshold); rows that failed by exception show the
    error instead of numbers.
    """
    flag = PASS_FLAG if r["passed"] else FAIL_FLAG
    name = str(r["name"])[:24]
    detail = r.get("detail") or {}
    if not r["passed"] and detail.get("status") == "error":
        return f"  [{flag}] {name:<24} error: {str(detail.get('error', ''))[:90]}"
    value, threshold = r["value"], r["threshold"]
    line = f"  [{flag}] {name:<24} value={fmt_num(value, 10)} threshold={fmt_num(threshold, 10)}"
    if (
        not r["passed"]
        and isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
        and threshold > 0
        and isinstance(value, (int, float)) and not isinstance(value, bool)
        and value > threshold
    ):
        # Factual for any check: the recorded value sits above its recorded
        # threshold. (Compound checks may have failed for more than this.)
        line += f"  exceeded {value / threshold:.2f}x"
    return line


def _drift_reason(m: dict) -> str:
    """Why an oracle metric drifted, phrased against the tolerance that
    should have held — the expected-vs-actual story for one row."""
    kind = m.get("tol_kind")
    tol = m.get("tol")
    ref, cand = m.get("reference"), m.get("candidate")
    if m.get("error"):
        return str(m["error"])
    if cand is None:
        return "missing from candidate"
    if kind == "exact":
        return f"expected exact match, got {cand:.6g} vs {ref:.6g}"
    if kind == "rel":
        drel = m.get("delta_rel")
        ratio = f" ({drel / tol:.1f}x over)" if isinstance(drel, (int, float)) and tol else ""
        return f"drel {drel:.3g} > tol {tol:g}{ratio}"
    if kind == "abs":
        return f"|delta| {abs(cand - ref):.3g} > tol {tol:g}"
    if kind == "max":
        return f"value {cand:.3g} > bound {tol:g}"
    if kind == "min":
        return f"value {cand:.3g} < floor {tol:g}"
    if kind == "interval":
        return f"outside [{tol[0]:g}, {tol[1]:g}]"
    return "drifted"


def render_oracle_row(name: str, m: dict) -> str:
    """One metric row of `simval validate`: reference vs candidate with the
    tolerance in force; drifted metrics append the violated bound."""
    flag = " ok " if m["passed"] else "DRIFT"
    line = (
        f"  [{flag}] {str(name)[:24]:<24} ref={fmt_num(m['reference'], 12)} "
        f"cand={fmt_num(m.get('candidate'), 12)} drel={fmt_num(m.get('delta_rel'), 9)} "
        f"tol={tol_str(m.get('tol_kind', '?'), m.get('tol'))}"
    )
    if not m["passed"]:
        line += f"  -- {_drift_reason(m)}"
    return line


def render_identity(detail: dict) -> list[str]:
    """The scenario-identity block of `simval validate` when the candidate
    is not the golden's scenario: per-input golden vs candidate hashes
    (truncated) with the problem, then a matched-count summary."""
    lines = []
    n_ok = 0
    for name, e in detail.items():
        if name == "__undeclared_inputs__":
            lines.append(f"    [FAIL] undeclared inputs present: {', '.join(map(str, e['problem']))}")
            continue
        golden = str(e.get("golden", ""))[:12]
        cand = e.get("candidate")
        cand_s = "-" if cand is None else str(cand)[:12]
        if "problem" in e:
            lines.append(f"    [FAIL] {name:<20} golden {golden}..  candidate {cand_s}..  ({e['problem']})")
        else:
            n_ok += 1
    if n_ok:
        lines.append(f"    ({n_ok} input{'s' if n_ok != 1 else ''} matched)")
    return lines
