"""Free-energy-perturbation (FEP) domain.

Wraps the standard BAR/MBAR estimators (alchemlyb -> pymbar); it never
reimplements them. Headline checks are the MBAR free-energy difference
(delta_f[-1, 0]), the MBAR overlap-matrix minimum-eigenvalue reliability
gauge, and forward/reverse-leg hysteresis. This proves the EngineAdapter port
accepts yet another physics domain (alchemical thermodynamics)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from simval.context import EngineAdapter, RunContext, register_engine
from simval.result import DiagnosticResult

KT_KJMOL_298K = 2.479
_OVERLAP_MIN_EIGENVALUE = 0.05


def _fit(u_nk):
    """Fit MBAR once and return (delta_f, d_delta_f, mbar_obj) as numpy arrays +
    the underlying pymbar object (for overlap).

    Accepts an alchemlyb u_nk DataFrame (preferred) or a raw K x N numpy array
    interpreted as pymbar ``u_kn`` with equal samples per state."""
    if hasattr(u_nk, "iloc"):
        from alchemlyb.estimators import MBAR

        est = MBAR()
        est.fit(u_nk)
        delta_f = est.delta_f_.to_numpy()
        d_delta_f = est.d_delta_f_.to_numpy()
        mbar = getattr(est, "_mbar", None)
        return delta_f, d_delta_f, mbar
    import pymbar

    arr = np.asarray(u_nk, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"u_nk array must be 2D (K x N), got shape {arr.shape}")
    k_states, n_total = arr.shape
    n_k = np.full(k_states, n_total // k_states, dtype=int)
    mbar = pymbar.MBAR(arr, n_k)
    res = mbar.compute_free_energy_differences()
    return res["Delta_f"], res["dDelta_f"], mbar


def check_free_energy(u_nk, *, kt_kjmol: float = KT_KJMOL_298K) -> DiagnosticResult:
    """MBAR free-energy difference delta_f[-1, 0] (last minus first lambda state).

    Always passes: this is an informational estimator, not a pass/fail invariant."""
    delta_f, d_delta_f, _ = _fit(u_nk)
    dG = float(delta_f[-1, 0])
    dG_unc = float(d_delta_f[-1, 0])
    return DiagnosticResult(
        name="free_energy",
        passed=True,
        threshold=0.0,
        value=dG,
        detail={
            "deltaG_kT": dG,
            "uncertainty_kT": dG_unc,
            "deltaG_kJmol": dG * kt_kjmol,
            "uncertainty_kJmol": dG_unc * kt_kjmol,
            "kt_kjmol": float(kt_kjmol),
        },
    )


def check_overlap(u_nk, *, min_eigenvalue: float = _OVERLAP_MIN_EIGENVALUE) -> DiagnosticResult:
    """MBAR overlap-matrix minimum eigenvalue.

    A small minimum eigenvalue (< ~0.05) flags an under-sampled or poorly
    connected lambda state, making the associated free-energy estimates
    unreliable. Pass when the minimum eigenvalue is at or above the threshold."""
    _, _, mbar = _fit(u_nk)
    if mbar is None:
        return DiagnosticResult(
            name="fep_overlap",
            passed=False,
            threshold=float(min_eigenvalue),
            value=0.0,
            detail={"error": "no MBAR overlap available"},
        )
    eig = np.real(mbar.compute_overlap()["eigenvalues"])
    min_eig = float(eig.min())
    return DiagnosticResult(
        name="fep_overlap",
        passed=min_eig >= min_eigenvalue,
        threshold=float(min_eigenvalue),
        value=min_eig,
        detail={
            "overlap_min_eigenvalue": min_eig,
            "eigenvalues": [float(x) for x in eig],
            "rule": "min eigenvalue >= threshold required for reliable MBAR estimates",
        },
    )


def check_hysteresis(u_nk, u_nk_reverse=None, *, threshold: float = 1.0) -> DiagnosticResult:
    """Forward/reverse-leg hysteresis.

    Both legs estimate the same thermodynamic free-energy difference
    (F_first - F_last via delta_f[-1, 0]); hysteresis is the absolute
    discrepancy between the two estimates, vanishing for a converged,
    reversible transformation. Without a reverse leg the check is skipped
    gracefully but still returns a DiagnosticResult."""
    if u_nk_reverse is None:
        return DiagnosticResult(
            name="fep_hysteresis",
            passed=True,
            threshold=float(threshold),
            value=0.0,
            detail={"skipped": True, "reason": "no reverse leg provided"},
        )
    dG_fwd = float(_fit(u_nk)[0][-1, 0])
    dG_rev = float(_fit(u_nk_reverse)[0][-1, 0])
    hyst = abs(dG_fwd - dG_rev)
    return DiagnosticResult(
        name="fep_hysteresis",
        passed=hyst <= threshold,
        threshold=float(threshold),
        value=hyst,
        detail={
            "deltaG_forward": dG_fwd,
            "deltaG_reverse": dG_rev,
            "hysteresis": hyst,
        },
    )


def synthetic_u_nk(*, seed: int = 42, n_samples: int = 20000,
                   k_first: float = 4.0, k_second: float = 1.0):
    """Deterministic alchemlyb-format u_nk for two 1D harmonic states.

    Analytical reduced free-energy difference F_first - F_second =
    0.5 * ln(k_first / k_second) = ln(2) ~= 0.6931 for the defaults, which the
    MBAR estimator recovers. Backs the oracle reference case ``fep_synthetic``."""
    import pandas as pd

    rng = np.random.default_rng(seed)

    def _sample(k, n):
        return rng.normal(0.0, 1.0 / np.sqrt(k), n)

    def _u(k, x):
        return 0.5 * k * x * x

    a = _sample(k_first, n_samples)
    b = _sample(k_second, n_samples)
    idx_a = pd.MultiIndex.from_arrays(
        [np.arange(n_samples), np.zeros(n_samples)], names=["time", "fep-lambda"])
    idx_b = pd.MultiIndex.from_arrays(
        [np.arange(n_samples, 2 * n_samples), np.ones(n_samples)],
        names=["time", "fep-lambda"])
    dfa = pd.DataFrame({0.0: _u(k_first, a), 1.0: _u(k_second, a)}, index=idx_a)
    dfb = pd.DataFrame({0.0: _u(k_first, b), 1.0: _u(k_second, b)}, index=idx_b)
    return pd.concat([dfa, dfb])


def _read_u_nk_csv(path: Path):
    """Simple CSV reader: first column = sample's fep-lambda, remaining columns
    are reduced potentials at each lambda (column headers parse to floats)."""
    import pandas as pd

    raw = pd.read_csv(path)
    lam_col = raw.columns[0]
    lams = raw[lam_col].to_numpy(dtype=float)
    cols = [float(c) for c in raw.columns[1:]]
    body = raw.iloc[:, 1:].to_numpy(dtype=float)
    n = body.shape[0]
    idx = pd.MultiIndex.from_arrays([np.arange(n), lams], names=["time", "fep-lambda"])
    return pd.DataFrame(body, index=idx, columns=cols)


def _load_reduced_potentials(path: Path, kind: str, temperature: float):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_u_nk_csv(path)
    if suffix == ".xvg":
        from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk

        if kind == "dhdl":
            return extract_dHdl(str(path), T=temperature)
        return extract_u_nk(str(path), T=temperature)
    raise ValueError(f"unsupported FEP file: {path}")


def _concat_sorted(frames):
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    import pandas as pd

    combined = pd.concat(frames)
    return combined.sort_index(level=combined.index.nlevels - 1)


def _has_dhdl(run: Path) -> bool:
    return bool(next(run.glob("*dhdl*.xvg"), None) or next(run.glob("*dhdl*.csv"), None))


def _discover(run: Path, *, reverse: bool) -> list[str]:
    out: list[str] = []
    for pat in ("*dhdl*.xvg", "*dhdl*.csv"):
        for p in sorted(run.glob(pat)):
            if ("reverse" in p.name.lower()) == reverse:
                out.append(p.name)
    return out


class FepEngine(EngineAdapter):
    name = "fep"

    def detect(self, run: Path) -> bool:
        if (run / "fep.json").exists():
            return True
        return _has_dhdl(run)

    def load_context(self, run: Path, selection: str) -> RunContext:
        ctx = RunContext(run_dir=run, engine=self.name, selection=selection)
        manifest = {}
        manifest_path = run / "fep.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            ctx.consumed_inputs.append(manifest_path)
        temperature = float(manifest.get("temperature", 298.15))
        kind = manifest.get("kind", "u_nk")
        fwd_files = manifest.get("files") or _discover(run, reverse=False)
        rev_files = manifest.get("reverse_files") or (
            [] if manifest.get("files") else _discover(run, reverse=True))

        def _load(file_list, leg):
            # Every filename present in a declared list must exist and parse;
            # a declared-but-missing file is a run-contract error, never a
            # silent skip (audit FEP-003). Absence of the whole 'files' /
            # 'reverse_files' key remains the intentional discovery/no-reverse
            # path — an explicit empty list discovers nothing.
            frames = []
            for name in file_list:
                p = run / name
                if not p.exists():
                    raise ValueError(
                        f"fep run contract: declared {leg} file {str(name)!r} is missing "
                        f"from the run directory"
                    )
                ctx.consumed_inputs.append(p)
                frames.append(_load_reduced_potentials(p, kind, temperature))
            return _concat_sorted(frames)

        try:
            loaded = _load(fwd_files, "forward")
        except ValueError as e:
            ctx.extra["fep_run_contract_error"] = str(e)
            loaded = None
        key = "dhdl" if kind == "dhdl" else "u_nk"
        if loaded is not None:
            ctx.extra[key] = loaded

        if rev_files:
            try:
                rev = _load(rev_files, "reverse")
            except ValueError as e:
                ctx.extra.setdefault("fep_run_contract_error", str(e))
                rev = None
            if rev is not None:
                ctx.extra[key + "_reverse"] = rev

        ctx.run_params = {
            "engine": self.name,
            "domain": "fep",
            "temperature_K": temperature,
            "kind": kind,
            "n_files": len(fwd_files),
        }
        return ctx


register_engine(FepEngine())
