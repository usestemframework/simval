from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from simval.diagnostics import rmsd as rmsd_mod
from simval.oracle.cases import ReferenceCase
from simval.result import DiagnosticResult


def compute_metrics(run_dir, *, selection: str = "protein and name CA") -> dict:
    """Scalar metrics for a candidate run, computed the same way as references.
    Dispatches by engine so the oracle is domain-general (MD today, N-body too)."""
    from simval.context import select_engine

    run = Path(run_dir)
    engine = select_engine(run)
    if engine.name == "nbody-rebound":
        return _nbody_metrics(run)
    if engine.name == "wave-fdtd":
        return _wave_metrics(run)
    if engine.name == "fluid-lbm":
        return _fluid_metrics(run)
    if engine.name == "em-fdtd":
        return _em_metrics(run)
    if engine.name == "quantum-spin":
        return _quantum_metrics(run)
    if engine.name == "fep":
        return _fep_metrics(run)
    if engine.name == "qc-pyscf":
        return _pyscf_metrics(run)
    if engine.name == "qc-qiskit":
        return _qiskit_metrics(run)
    if engine.name == "chemical-kinetics":
        return _kinetics_metrics(run)
    if engine.name == "heat-diffusion":
        return _diffusion_metrics(run)
    if engine.name == "relativistic-boris":
        return _relativistic_metrics(run)
    return _md_metrics(run, selection)


def _md_metrics(run: Path, selection: str) -> dict:
    from simval import io
    from simval._util import find_unique, select_trajectory_topology

    top = select_trajectory_topology(run)
    xtc = find_unique(run, "*.xtc", "*.dcd", "*.trr", "*.nc", what="trajectory")
    if not (top and xtc):
        raise FileNotFoundError(f"run-dir {run} needs a trajectory (.xtc/.dcd/.trr/.nc) and topology (.gro/.pdb/.prmtop/.psf/.tpr)")

    positions, reference, _names = io.load_trajectory(xtc, top, selection=selection)
    rseries = rmsd_mod.rmsd_over_time(positions, reference)

    com = positions.mean(axis=1, keepdims=True)
    rg_per_frame = np.sqrt(((positions - com) ** 2).sum(axis=2).mean(axis=1))

    plateau = rmsd_mod.check_rmsd_plateau(positions, reference)

    metrics = {
        "n_frames": int(positions.shape[0]),
        "n_selected_atoms": int(positions.shape[1]),
        "mean_rmsd_nm": float(rseries[1:].mean()) if rseries.size > 1 else 0.0,
        "final_rmsd_nm": float(rseries[-1]),
        "rmsd_tail_drift_fraction": float(plateau.value),
        "mean_rg_nm": float(rg_per_frame.mean()),
        "final_rg_nm": float(rg_per_frame[-1]),
    }

    xvg = find_unique(run, "*.xvg", what="energy file (xvg)")
    if xvg:
        from simval import io as io_mod
        from simval.diagnostics import energy as energy_mod
        try:
            _term, e = io_mod.load_preferred_energy(xvg)
        except io_mod.ConservedEnergyColumnMissing:
            # No labeled conserved-energy column: the metric is not
            # computable from this run and stays absent from the candidate,
            # so any reference that requires it fails (audit IO-002). A
            # malformed file raises XvgParseError and must fail the
            # validation, not skip the metric (audit IO-003).
            pass
        else:
            metrics["energy_relative_range"] = float(energy_mod.check_energy_drift(e).value)

    return metrics


def _nbody_metrics(run: Path) -> dict:
    from simval import nbody as nbody_mod

    data = nbody_mod.integrate_system(run / "system.json", samples=120)
    e = data["energy"]
    L = data["L_magnitude"]
    com = data["com"]
    e_rel = float((e.max() - e.min()) / (abs(e.mean()) + 1e-15))
    L_rel = float((L.max() - L.min()) / (abs(L.mean()) + 1e-15))
    com_drift = float(np.sqrt(((com - com[0]) ** 2).sum(axis=1)).max())
    return {
        "n_samples": int(e.size),
        "energy_relative_range": e_rel,
        "angular_momentum_relative_range": L_rel,
        "com_drift_max": com_drift,
    }


def _wave_metrics(run: Path) -> dict:
    import json

    from simval.wave import check_wave_energy, integrate_wave

    cfg = json.loads((run / "wave.json").read_text())
    data = integrate_wave(cfg)
    growth = check_wave_energy(data["energy"], src_on_index=data["src_on"] // 4)
    return {
        "cfl": float(data["cfl"]),
        "energy_growth": float(growth.value),
        "n_steps": int(data["n_steps"]),
    }


def _fluid_metrics(run: Path) -> dict:
    import json

    from simval.fluid import check_mass_conservation, check_tau_stability, integrate_fluid

    cfg = json.loads((run / "fluid.json").read_text())
    data = integrate_fluid(cfg)
    tau = check_tau_stability(data["tau"])
    mass = check_mass_conservation(data["mass"])
    return {
        "tau": float(data["tau"]),
        "mass_drift": float(mass.value),
        "tau_in_range": 1.0 if tau.passed else 0.0,
    }


def _em_metrics(run: Path) -> dict:
    import json

    from simval.em import check_em_energy, integrate_em

    cfg = json.loads((run / "em.json").read_text())
    data = integrate_em(cfg)
    eg = check_em_energy(data["energy"], src_on_index=data["src_on"] // 10)
    return {
        "courant": float(data["courant"]),
        "em_energy_growth": float(eg.value),
        "n_steps": int(data["n_steps"]),
    }


def _quantum_metrics(run: Path) -> dict:
    import json

    from simval.quantum import check_norm_conservation, evolve_spin

    cfg = json.loads((run / "quantum.json").read_text())
    data = evolve_spin(cfg)
    nc = check_norm_conservation(data["norm"])
    return {
        "norm_drift": float(nc.value),
        "p_up_swing": float(data["p_up"].max() - data["p_up"].min()),
        "n_steps": int(data["n_steps"]),
    }


def _fep_metrics(run: Path) -> dict:
    from simval.fep import FepEngine, check_free_energy, check_overlap

    ctx = FepEngine().load_context(run, "n/a")
    u_nk = ctx.extra["u_nk"]
    fe = check_free_energy(u_nk)
    ov = check_overlap(u_nk)
    return {
        "deltaG_kT": float(fe.detail["deltaG_kT"]),
        "uncertainty_kT": float(fe.detail["uncertainty_kT"]),
        "overlap_min_eigenvalue": float(ov.detail["overlap_min_eigenvalue"]),
    }


def _pyscf_metrics(run: Path) -> dict:
    from simval.pyscf_eng import PyscfEngine

    ctx = PyscfEngine().load_context(run, "n/a")
    energies = ctx.extra["scf_energies"]
    last_delta = abs(energies[-1] - energies[-2]) if len(energies) >= 2 else 0.0
    return {
        "final_energy_hartree": float(ctx.extra["final_energy"]),
        "converged": 1.0 if ctx.extra["converged"] else 0.0,
        "n_electrons": int(ctx.extra["n_electrons"]),
        "n_cycles": len(energies),
        "scf_last_delta": float(last_delta),
    }


def _qiskit_metrics(run: Path) -> dict:
    from simval.qiskit_eng import QiskitEngine, check_norm_conservation

    ctx = QiskitEngine().load_context(run, "n/a")
    nm = check_norm_conservation(ctx.extra["statevector"])
    metrics = {
        "norm_drift": float(nm.value),
        "n_qubits": int(ctx.extra["n_qubits"]),
    }
    expected = ctx.extra.get("expected_probabilities")
    if expected:
        probs = ctx.extra["probabilities"]
        keys = sorted(set(probs) | set(expected))
        tv = 0.5 * sum(abs(float(probs.get(k, 0.0)) - float(expected.get(k, 0.0))) for k in keys)
        metrics["tv_distance"] = float(tv)
    return metrics


def _kinetics_metrics(run: Path) -> dict:
    import json

    from simval.kinetics import check_mass_balance, integrate_kinetics

    cfg = json.loads((run / "kinetics.json").read_text())
    data = integrate_kinetics(cfg)
    mb = check_mass_balance(data["history"])
    return {"mass_balance_drift": float(mb.value), "n_steps": int(data["n_steps"])}


def _diffusion_metrics(run: Path) -> dict:
    import json

    from simval.diffusion import check_heat_conservation, integrate_diffusion

    cfg = json.loads((run / "diffusion.json").read_text())
    data = integrate_diffusion(cfg)
    hc = check_heat_conservation(data["energy"])
    return {"fourier": float(data["fourier"]), "energy_drift": float(hc.value)}


def _relativistic_metrics(run: Path) -> dict:
    import json

    from simval.relativistic import check_relativistic_energy, integrate_relativistic

    cfg = json.loads((run / "relativistic.json").read_text())
    data = integrate_relativistic(cfg)
    re = check_relativistic_energy(data["gamma"])
    return {"gamma_drift": float(re.value), "n_steps": int(data["n_steps"])}


_DEFAULT_TOLERANCES = {
    # Count-like and version-like fields: exact match, always.
    "n_selected_atoms": ("exact",),
    "n_frames": ("exact",),
    "n_steps": ("exact",),
    "n_samples": ("exact",),
    "n_electrons": ("exact",),
    "n_qubits": ("exact",),
    "n_cycles": ("exact",),
    "n_gates": ("exact",),
    "converged": ("exact",),
    "tau_in_range": ("exact",),
    # Physical quantities: delta tolerances against the reference value.
    "mean_rmsd_nm": ("rel", 0.10),
    "final_rmsd_nm": ("rel", 0.10),
    "rmsd_tail_drift_fraction": ("rel", 0.25),
    "mean_rg_nm": ("rel", 0.02),
    "final_rg_nm": ("rel", 0.03),
    "energy_relative_range": ("rel", 0.25),
    "energy_drift": ("rel", 0.10),
    "gamma_drift": ("abs", 1e-9),
    "mass_balance_drift": ("abs", 1e-9),
    "norm_drift": ("abs", 1e-9),
    "tv_distance": ("abs", 1e-9),
    "deltaG_kT": ("abs", 2.0),
    "uncertainty_kT": ("abs", 1.0),
    "overlap_min_eigenvalue": ("min", 0.05),
    "final_energy_hartree": ("abs", 1e-6),
    "scf_last_delta": ("abs", 1e-6),
}


def _within(kind, candidate, reference, tol):
    if kind == "exact":
        return candidate == reference
    if kind == "rel":
        denom = abs(reference) + 1e-12
        return abs(candidate - reference) / denom <= tol
    if kind == "abs":
        return abs(candidate - reference) <= tol
    if kind == "max":
        # Physical ceiling: candidate must be finite and at or below the bound.
        return math.isfinite(candidate) and candidate <= tol
    if kind == "min":
        # Physical floor: candidate must be finite and at or above the bound.
        return math.isfinite(candidate) and candidate >= tol
    if kind == "interval":
        lo, hi = tol
        return math.isfinite(candidate) and lo <= candidate <= hi
    raise ValueError(f"unknown tolerance kind: {kind}")


def _validate_tolerance_spec(metric: str, spec) -> None:
    """Externally supplied comparison rules must be well-formed and bounded
    (audit ORA-004): known kind, finite numeric operands, positive tolerances,
    ordered intervals. An inf/NaN operand would make the rule vacuous."""
    kinds = ("exact", "abs", "rel", "max", "min", "interval")
    if not isinstance(spec, (list, tuple)) or not spec or spec[0] not in kinds:
        raise ValueError(
            f"tolerance rule for {metric!r} must be [kind, ...] with kind in {kinds}, got {spec!r}"
        )
    kind = spec[0]

    def _num(v, what):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"tolerance rule for {metric!r}: {what} must be a finite number, got {v!r}")
        return float(v)

    if kind == "exact":
        if len(spec) != 1:
            raise ValueError(f"tolerance rule for {metric!r}: 'exact' takes no operand, got {spec!r}")
    elif kind == "interval":
        if len(spec) != 3:
            raise ValueError(f"tolerance rule for {metric!r}: 'interval' needs [interval, lo, hi], got {spec!r}")
        lo, hi = _num(spec[1], "interval lo"), _num(spec[2], "interval hi")
        if not lo < hi:
            raise ValueError(f"tolerance rule for {metric!r}: interval lo must be < hi, got [{lo}, {hi}]")
    else:
        if len(spec) != 2:
            raise ValueError(f"tolerance rule for {metric!r}: {kind!r} needs exactly one operand, got {spec!r}")
        tol = _num(spec[1], f"{kind} bound")
        if kind in ("abs", "rel") and tol <= 0:
            raise ValueError(f"tolerance rule for {metric!r}: {kind!r} tolerance must be positive, got {tol}")


def compare_metrics(
    candidate: dict,
    reference: dict,
    tolerances: dict | None = None,
    *,
    ignore: list[str] | None = None,
) -> dict:
    """Pure comparison. Returns {metric: {ref, candidate, delta, passed}} + overall 'passed'.

    Every reference metric is mandatory: a metric missing from the candidate
    fails the comparison, and a metric with no comparison policy is a hard
    configuration error. The only exemption is an explicit per-case `ignore`
    list (empty by default).
    """
    ignore_set = set(ignore or ())
    tols = dict(_DEFAULT_TOLERANCES)
    if tolerances:
        stale = sorted(set(tolerances) - set(reference))
        if stale:
            raise ValueError(f"tolerances configured for unknown reference metrics: {stale}")
        for metric, spec in tolerances.items():
            _validate_tolerance_spec(metric, spec)
        tols.update(tolerances)
    out: dict = {}
    all_pass = True
    for metric, ref_val in reference.items():
        if metric in ignore_set:
            out[metric] = {
                "reference": ref_val,
                "candidate": None,
                "delta_rel": None,
                "tol_kind": "ignore",
                "tol": None,
                "passed": True,
                "ignored": True,
            }
            continue
        if metric not in tols:
            raise ValueError(
                f"reference metric {metric!r} has no comparison policy; "
                "add a tolerance rule or declare it in the case's ignore list"
            )
        if metric not in candidate:
            spec = tols[metric]
            out[metric] = {
                "reference": ref_val,
                "candidate": None,
                "delta_rel": None,
                "tol_kind": spec[0],
                "tol": list(spec[1:3]) if spec[0] == "interval" else (spec[1] if len(spec) > 1 else None),
                "passed": False,
                "error": "missing from candidate",
            }
            all_pass = False
            continue
        kind = tols[metric][0]
        cand_val = candidate[metric]
        tol = tols[metric][1] if len(tols[metric]) > 1 else None
        if kind == "interval":
            tol = tuple(tols[metric][1:3])
        passed = _within(kind, cand_val, ref_val, tol)
        denom = abs(ref_val) + 1e-12
        out[metric] = {
            "reference": ref_val,
            "candidate": cand_val,
            "delta_rel": float(abs(cand_val - ref_val) / denom) if kind != "exact" else (0.0 if passed else 1.0),
            "tol_kind": kind,
            "tol": list(tol) if kind == "interval" else tol,
            "passed": bool(passed),
        }
        all_pass = all_pass and passed
    out["__passed__"] = bool(all_pass)
    return out


def compute_identity(run_dir) -> dict[str, str]:
    """Scenario identity of a candidate run (audit ORA-003): sha256 of every
    scenario-defining input, keyed by file name. Deliberately cheap — file
    selection and hashing only, no heavy engine imports — and computed the
    same way for golden generation and validation."""
    from simval.context import select_engine

    run = Path(run_dir)
    engine = select_engine(run).name

    def sha(p: Path) -> str:
        import hashlib

        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 16), b""):
                h.update(block)
        return h.hexdigest()

    def identity_of(*paths: Path) -> dict[str, str]:
        present = [p for p in paths if p is not None and p.exists()]
        return {p.name: sha(p) for p in sorted(present, key=lambda p: p.name)}

    if engine == "gromacs":
        # ORA-005: identity covers the canonical scenario enumeration
        # (every present role, not just the selected trajectory topology),
        # and the methods.json force-field contract is enforced before
        # any hashing/metrics.
        from simval._util import gromacs_force_field_problem, gromacs_scenario_inputs

        problem = gromacs_force_field_problem(run)
        if problem:
            raise ValueError(problem)
        return identity_of(*gromacs_scenario_inputs(run))
    if engine in ("nbody-rebound",):
        return identity_of(run / "system.json")
    if engine == "wave-fdtd":
        return identity_of(run / "wave.json")
    if engine == "fluid-lbm":
        return identity_of(run / "fluid.json")
    if engine == "em-fdtd":
        return identity_of(run / "em.json")
    if engine == "quantum-spin":
        return identity_of(run / "quantum.json")
    if engine == "heat-diffusion":
        return identity_of(run / "diffusion.json")
    if engine == "chemical-kinetics":
        return identity_of(run / "kinetics.json")
    if engine == "relativistic-boris":
        return identity_of(run / "relativistic.json")
    if engine == "qc-pyscf":
        return identity_of(run / "molecule.json")
    if engine == "qc-qiskit":
        return identity_of(run / "circuit.json")
    if engine == "fep":
        manifest = run / "fep.json"
        files: list[Path] = [manifest]
        if manifest.exists():
            m = json.loads(manifest.read_text())
            for key in ("files", "reverse_files"):
                for name in m.get(key) or []:
                    files.append(run / str(name))
        else:
            from simval.fep import _discover

            files += [run / n for n in _discover(run, reverse=False)]
            files += [run / n for n in _discover(run, reverse=True)]
        return identity_of(*files)
    raise ValueError(f"no scenario-identity rule for engine {engine!r}")


def _identity_result(case, run_engine: str, error: str, identity: dict | None = None) -> DiagnosticResult:
    return DiagnosticResult(
        name="reference_oracle",
        passed=False,
        threshold=0.0,
        value=0.0,
        detail={
            "case": case.name,
            "run_engine": run_engine,
            "error": error,
            "identity": identity or {},
            "n_checked": 0,
            "n_failed": 0,
            "metrics": {},
            "candidate_metrics": {},
        },
    )


def validate(run_dir, case: ReferenceCase | str, *, selection: str | None = None) -> DiagnosticResult:
    if isinstance(case, str):
        from simval.oracle.cases import get_case
        case = get_case(case)
    run = Path(run_dir)
    from simval.context import select_engine
    try:
        run_engine = select_engine(run).name
    except ValueError as e:
        # Engine detection keys off the scenario config file itself; a run
        # whose defining input is gone still fails closed with a clear error.
        return _identity_result(case, "unknown", str(e)[:200])
    if case.engine and case.engine != run_engine:
        return DiagnosticResult(
            name="reference_oracle",
            passed=False,
            threshold=0.0,
            value=0.0,
            detail={
                "case": case.name,
                "run_engine": run_engine,
                "case_engine": case.engine,
                "error": f"domain mismatch: run is '{run_engine}', case is '{case.engine}'",
                "n_checked": 0, "n_failed": 0, "metrics": {}, "candidate_metrics": {},
            },
        )
    sel = selection or case.selection
    # --- Scenario identity gate (audit ORA-003) ---
    # Metric agreement alone must never pass a candidate: the golden's
    # scenario-defining inputs are compared exactly (name + sha256) before
    # any metric is computed. Missing identity on either side fails closed.
    if not case.identity:
        return _identity_result(
            case, run_engine, "golden carries no scenario identity; refusing to validate"
        )
    try:
        candidate_identity = compute_identity(run)
    except Exception as e:
        return _identity_result(
            case, run_engine, f"candidate identity could not be computed: {type(e).__name__}: {e}"[:200]
        )
    identity_detail = {}
    identity_ok = True
    for name, want_hash in sorted(case.identity.items()):
        p = run / name
        if not p.exists():
            identity_detail[name] = {"golden": want_hash, "candidate": None, "problem": "missing input"}
            identity_ok = False
            continue
        if name not in candidate_identity:
            identity_detail[name] = {
                "golden": want_hash, "candidate": None,
                "problem": "present but not part of the candidate's scenario inputs",
            }
            identity_ok = False
            continue
        got_hash = candidate_identity[name]
        entry = {"golden": want_hash, "candidate": got_hash}
        if got_hash != want_hash:
            entry["problem"] = "content mismatch"
            identity_ok = False
        identity_detail[name] = entry
    undeclared = sorted(set(candidate_identity) - set(case.identity))
    if undeclared:
        identity_ok = False
        identity_detail["__undeclared_inputs__"] = {"problem": undeclared}
    if not identity_ok:
        return DiagnosticResult(
            name="reference_oracle",
            passed=False,
            threshold=0.0,
            value=0.0,
            detail={
                "case": case.name,
                "run_engine": run_engine,
                "error": "scenario identity mismatch: the candidate is not the golden's scenario",
                "identity": identity_detail,
                "n_checked": 0,
                "n_failed": 0,
                "metrics": {},
                "candidate_metrics": {},
            },
        )
    candidate = compute_metrics(run_dir, selection=sel)
    compared = compare_metrics(
        candidate, case.reference_metrics, case.tolerances, ignore=case.ignore
    )
    passed = compared.pop("__passed__")
    n_checked = len(compared)
    n_failed = sum(1 for v in compared.values() if not v["passed"])
    if n_checked == 0:
        passed = False
        compared["_domain_mismatch"] = {
            "reference_metrics": sorted(case.reference_metrics),
            "candidate_metrics": sorted(candidate),
            "note": "no overlapping metrics despite matching engine",
        }
    detail = {
        "case": case.name,
        "selection": sel,
        "n_checked": n_checked,
        "n_failed": n_failed,
        "metrics": compared,
        "candidate_metrics": candidate,
    }
    return DiagnosticResult(
        name="reference_oracle",
        passed=passed,
        threshold=0.0,
        value=float(n_failed),
        detail=detail,
    )


def _find(run: Path, *patterns: str):
    from simval._util import find_files
    return find_files(run, *patterns)
