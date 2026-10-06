# simval

Project status: simval is a deterministic verification oracle for computational-physics simulations and is a work in progress.

![License: MIT](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/Python-3.11+-3776AB) ![Status](https://img.shields.io/badge/status-pre--alpha-red)

Deterministic verification + a reference-oracle for computational physics simulations. Local-first, LLM-free core. The moat is the unsexy layer — packaged, consistent diagnostics + provenance + a validation oracle that an AI agent (or human) loops against to know whether a simulation's output is correct.

Working name; the folder may be renamed. See `PLAN.md` for the full thesis and `AUDIT.md` for the 5-pass analysis that shaped it.

## What it does

- **Diagnose** a run-dir: deterministic Tier-1 checks (energy drift, RMSD plateau, structural equilibration, per-residue RMSF, force-field coverage, parameter/unit sanity, prep sanity: box-vs-cutoff, steric clashes, charge state + His-tautomers).
- **Provenance manifest** (`simval.provenance.v1`): params, diagnostic results, SHA256 file hashes, verdict, Tier-2 human sign-off hook, and an **auto-extracted methods paragraph**.
- **Reference-oracle**: compare a candidate run against stored canonical references (`validate`) — the agent-loop teacher. Domain-general.
- **Compare** two runs (the parameter-sweep daily loop).

## Nine physics domains via one port

| Engine | Domain | Solver | Headline check |
|---|---|---|---|
| `gromacs` (format-agnostic) | molecular dynamics | GROMACS, OpenMM | conserved-energy drift, RMSD/RMSF, charge, H-bonds |
| `nbody-rebound` | celestial mechanics | REBOUND (IAS15) | energy + angular-momentum + COM conservation |
| `ontos` | discrete multiscale (life v1) + nbody gravity with Chebyshev ephemeris windows, zoom policy, collapse + multipole + radial-shape + per-shell radial reconstruction, collapse-on-coarse composition, contact dynamics (restitution/friction/static/walls) + modal audio (v2) | independent spec reimplementation | stream records vs reference, bit-exact state hashes, bounded window drift, zoom-policy conformance, collapse reconstruction error, multipole dipole/quadrupole closure, radial binding/energy closure, per-shell binding closure (total/vertex/quad bounded), coarse->collapse seam drift, contact impulse invariants (incl. -e*vn restitution closure), bit-exact WAV/audio-hash resynthesis (incl. pseudo-id static contactants), corpus coverage of wall-hit + fine-x-coarse static contacts via test-only IC profiles (`--test-ic`, `examples/ontos_contact/{wallshot,coarsehit}`), REBOUND anchor (<1e-4) |
| `wave-fdtd` | waves / PDE | built-in leapfrog FDTD | CFL stability + energy boundedness |
| `fluid-lbm` | fluids / CFD | built-in D2Q9 LBM | BGK τ stability + exact mass conservation |
| `em-fdtd` | electromagnetism | built-in 2D TMz Yee | Courant condition + EM energy boundedness |
| `quantum-spin` | quantum dynamics | built-in statevector | norm conservation + Rabi population swing |
| `fep` | alchemical free energy | alchemlyb + pymbar | MBAR ΔG + overlap + hysteresis |
| `qc-pyscf` | quantum chemistry | PySCF (HF/DFT) | SCF convergence + energy sanity |
| `qc-qiskit` | quantum circuits | Qiskit Aer | norm conservation + measurement distribution |

Plus **654 reference anchors** (12 oracle cases + 642 FreeSolv experimental ΔG values, CC-BY-4.0).

A new domain implements `EngineAdapter` (`detect` + `load_context`) and registers itself; universal checks reuse automatically, domain-specific checks fire on `ctx.extra`.

## Install

```
pip install simval                       # core (numpy-only: waves, fluids, EM, quantum, kinetics, diffusion, relativistic)
pip install simval[md]                   # + MDAnalysis — read MD trajectories (GROMACS/OpenMM/AMBER/NAMD)
pip install simval[fep]                  # + alchemlyb + pymbar — free-energy verification (BAR/MBAR)
pip install simval[nbody]                # + REBOUND — celestial mechanics
pip install simval[qc]                   # + PySCF — quantum chemistry (HF/DFT)
pip install simval[quantum]              # + Qiskit — quantum circuits
pip install simval[web]                  # + FastAPI — local dashboard
pip install simval[all]                  # everything
```

Simulation engines (GROMACS, OpenMM) are **not** bundled — the user installs them to *produce* runs. simval reads their output. Structure databases (PDB, AlphaFold DB) are fetched on demand via `simval fetch`. The FreeSolv experimental database (642 compounds, CC-BY-4.0) ships in the package.

## Thresholds

Defaults are starting points, not physics — a flexible loop and a rigid pocket need different RMSD ceilings. Override per-check either with a `thresholds.json` in the run-dir or via `--thresholds overrides.json`:

```json
{"energy_drift": 0.02, "rmsd_plateau": 0.15, "per_residue_rmsf": 0.35}
```

Each check records the threshold it actually used in its `DiagnosticResult` (and thus the manifest), so a verdict is always traceable to the thresholds in force. See `src/simval/thresholds.py` for the full default table.

## CLI

```
simval diagnose <run-dir>                       # run diagnostics -> provenance.json
simval validate <run-dir> --case <name>         # oracle: match/drift vs reference
simval compare  <run-a> <run-b>                  # sweep diff
simval sweep    <folder> [--baseline X]          # diagnose every run-dir, tabulate
simval cases                                     # list reference cases
simval engines                                   # list registered domain adapters
simval inspect  <run-dir>                        # engine + what the run-dir contains
simval verify-manifest <provenance.json>         # re-hash files, detect tampering
simval fetch    <pdb-id|uniprot-id>              # fetch structure from PDB/AlphaFold DB
simval afold    <uniprot-id>                     # AlphaFold pLDDT confidence profile
simval export-omex <run-dir>                     # COMBINE archive (.omex)
simval case-info <name>                          # reference case provenance
simval freesolv <compound-id> [<computed-dG>]    # FreeSolv experimental ΔG lookup/validation
simval-web --port 8765                           # local dashboard (3D rendering, charts)
python -m simval.ontos <stream> <seed>           # verify an ontos record stream (stdlib-only)
python -m simval.ontos_audio <stream> [--wav F] # resynthesize spec-22 modal audio (bit-exact)
simval orchestrate --grid <file> [--ontos-bin P] # generate + verify an ontos run grid, MAD outliers
```

## Examples

- `examples/nbody/two_body/system.json` — Kepler orbit; `simval validate . --case kepler_two_body`
- `examples/wave/pulse/wave.json` — stable wave pulse; `simval validate . --case wave_pulse_stable`
- `pipeline/` — a real GROMACS 1AKI lysozyme run (Dockerfile + `run.sh`).

## Architecture

```
engine adapter (per domain)  ->  RunContext  ->  pipeline.run_checks  ->  manifest
                                          \->  oracle.validate (candidate vs reference)
service.py (stable API)  ->  cli / web / (future agent, notebook)
```

The core is local + LLM-free (verification must be deterministic). The optional NL/agent layer is the only cloud-touching part and is gated (see PLAN §1.5.1).

## Extending — adding a physics domain

The port is the headroom. To add a domain:

1. **Engine adapter** — subclass `EngineAdapter` (`simval/context.py`): implement `detect(run)` (recognize your run-dir) and `load_context(run) -> RunContext` (read your solver's outputs into the context). Call `register_engine(MyEngine())` at import. **Universal checks reuse automatically** — anything you put in `ctx.energy` gets `energy_drift`.
2. **Domain checks** — add branches to `pipeline.run_checks` that fire on `ctx.extra` (your domain's observables). Return `DiagnosticResult`s.
3. **Oracle (optional)** — add a `_xx_metrics(run)` to `oracle/validate.py` + the dispatch line, compute reference metrics from a canonical case, drop a `references/xx.json`.

**Minimal reference template:** `src/simval/wave.py` (~110 lines) — a complete domain (FDTD solver, 2 checks, engine, self-registration) and `references/wave_pulse_stable.json`. `src/simval/nbody.py` is the REBOUND-wrapped variant. Copy the shape, swap the physics.

## Status

Pre-alpha, no users yet. Phase 1 = plain-MD verification + provenance + oracle, with the multi-domain port proven. FEP/ΔΔG (the core pharma deliverable) is Phase 3 — a separate, months-long investment gated on a real user + domain partner. See PLAN §7.

## License

MIT (simval core). The verification layer invokes solvers **arm's-length** (subprocess / container / MCP) and does not link them, so wrapping GPL solvers (GROMACS, OpenFOAM) does not propagate copyleft — this is what keeps a future paid layer legal (PLAN §11). Upstream solvers keep their own licenses.

## Security

simval's current scope is **RCE-free**: it reads simulation output files and runs internal numeric code (numpy / REBOUND / a built-in FDTD). It does not execute untrusted code, and the web dashboard is local-first (`127.0.0.1`) reading user-supplied paths. The audit's Critical RCE finding (A7) applied to the original `bpy` + agent-generated-code vision — that risk materializes **only if** an agent layer is added that runs LLM-generated code; at that point it requires the audit's mitigations (sandboxed/rootless containers, `--network=none --read-only`, digest-pinned images, **human approval before any generated code runs**, MCP tool output treated as untrusted). Not present today; gated with the agent layer.
