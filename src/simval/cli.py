from __future__ import annotations

import argparse
import time
from pathlib import Path

from simval import __version__
from simval.pipeline import diagnose as run_diagnose
from simval.report import (
    fmt_dur,
    render_check_row,
    render_identity,
    render_oracle_row,
    rule_like,
)


def _safe(fn):
    """Run an engine/oracle call; on a missing/unrecognized run-dir print a clean
    error and return None instead of a traceback."""
    try:
        return fn()
    except (FileNotFoundError, ValueError) as e:
        print(f"simval: error: {e}")
        return None


def _print_manifest_report(manifest, *, elapsed_s: float, provenance_path: Path | None = None):
    """Verdict header + one aligned row per check + skipped checks.

    Shared by `diagnose` and `run` so both report identically."""
    diags = manifest["diagnostics"]
    n_pass = sum(1 for d in diags if d["passed"])
    print(
        f"simval {__version__} | verdict: {manifest['verdict'].upper()} | "
        f"{n_pass}/{len(diags)} checks passed | {fmt_dur(elapsed_s)}"
    )
    for d in diags:
        print(render_check_row(d))
    skipped = (manifest.get("params") or {}).get("skipped") or {}
    for name in sorted(skipped):
        print(f"  [SKIP] {str(name)[:24]:<24} {skipped[name]}")
    if provenance_path is not None:
        print(f"  provenance -> {provenance_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="simval", description="Deterministic MD verification + reference oracle")
    parser.add_argument("--version", action="version", version=f"simval {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="run diagnostics on a run directory; writes provenance.json")
    d.add_argument("run_dir")
    d.add_argument("--out", default="provenance.json")
    d.add_argument("--selection", default="protein", help="MDAnalysis selection (default: protein)")
    d.add_argument("--thresholds", default=None, help="path to a JSON of per-check threshold overrides")

    ins = sub.add_parser("inspect", help="show engine + what a run-dir contains (no checks)")
    ins.add_argument("run_dir")
    ins.add_argument("--selection", default="protein")

    v = sub.add_parser("validate", help="compare a run against a stored reference case (oracle)")
    v.add_argument("run_dir")
    v.add_argument("--case", required=True)
    v.add_argument("--selection", default=None)

    sub.add_parser("cases", help="list available reference cases")
    sub.add_parser("engines", help="list registered engine adapters")
    cmp = sub.add_parser("compare", help="compare two runs on key metrics (sweep view)")
    cmp.add_argument("run_a")
    cmp.add_argument("run_b")
    cmp.add_argument("--selection", default="protein and name CA")

    sw = sub.add_parser("sweep", help="diagnose every run-dir under a folder; tabulate")
    sw.add_argument("folder")
    sw.add_argument("--baseline", default=None)
    sw.add_argument("--selection", default="protein and name CA")

    orc = sub.add_parser("orchestrate", help="run an ontos parameter grid; verify + tabulate every run")
    orc.add_argument("--grid", required=True, help="JSON (or YAML) list of run specs; format in simval.orchestrate")
    orc.add_argument("--ontos-bin", default=None, help="path to the ontos binary (default: $ONTOS_BIN or PATH)")
    orc.add_argument("--out", default=None, help="write the results dict as JSON to this path")
    orc.add_argument(
        "--cell-timeout", type=float, default=1800.0, dest="cell_timeout_s", metavar="SECONDS",
        help="per-cell ontos invocation timeout in seconds (default 1800; expiry kills the "
        "child and records an _error row — no retries)",
    )

    vm = sub.add_parser("verify-manifest", help="re-hash files; confirm they match a provenance.json")
    vm.add_argument("manifest")

    ft = sub.add_parser("fetch", help="fetch a structure by PDB ID or UniProt ID (RCSB / AlphaFold DB)")
    ft.add_argument("identifier")
    ft.add_argument("out_dir", nargs="?", default=".")
    ft.add_argument("--source", default=None, choices=["pdb", "alphafold"])

    rn = sub.add_parser("run", help="run MD via OpenMM + auto-verify (simulate -> diagnose)")
    rn.add_argument("pdb")
    rn.add_argument("--out", default="runs/md")
    rn.add_argument("--steps", type=int, default=5000)
    rn.add_argument("--dt", type=float, default=0.002)
    rn.add_argument("--temp", type=float, default=300.0)

    af = sub.add_parser("afold", help="AlphaFold pLDDT confidence profile for a UniProt id")
    af.add_argument("uniprot_id")
    af.add_argument("out_dir", nargs="?", default=".")

    omx = sub.add_parser("export-omex", help="package a run's provenance + artifacts into a COMBINE archive (.omex)")
    omx.add_argument("run_dir")
    omx.add_argument("out_path", nargs="?", default=None)

    ci = sub.add_parser("case-info", help="show provenance + reference metrics for a stored case")
    ci.add_argument("name")

    fs = sub.add_parser("freesolv", help="FreeSolv experimental dG database (642 compounds)")
    fs.add_argument("compound_id", nargs="?", default=None)
    fs.add_argument("computed_dG", nargs="?", type=float, default=None)
    fs.add_argument("--search", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "diagnose":
        overrides = None
        if args.thresholds:
            import json
            overrides = json.loads(Path(args.thresholds).read_text())
        start = time.perf_counter()
        manifest = _safe(lambda: run_diagnose(
            args.run_dir, out=args.out, selection=args.selection, thresholds=overrides))
        if manifest is None:
            return 1
        _print_manifest_report(
            manifest, elapsed_s=time.perf_counter() - start,
            provenance_path=Path(args.run_dir) / args.out,
        )
        return 0 if manifest["verdict"] == "pass" else 1

    if args.cmd == "inspect":
        from simval import service
        snap = _safe(lambda: service.inspect(args.run_dir, selection=args.selection))
        if snap is None:
            return 1
        print(f"simval {__version__} | engine: {snap['engine']} | selection: {snap['selection']}")
        for k, v in snap["has"].items():
            print(f"  {'has' if v else 'no '}  {k}")
        if snap.get("metadata"):
            ff = snap["metadata"].get("force_field") or "-"
            wm = snap["metadata"].get("water_model") or "-"
            print(f"  force_field={ff}  water={wm}")
        return 0

    if args.cmd == "cases":
        from simval.oracle import list_cases
        cases = list_cases()
        print(f"simval {__version__} | {len(cases)} reference cases")
        for name in cases:
            print(f"  {name}")
        return 0

    if args.cmd == "engines":
        from simval import service
        names = service.list_engines()
        print(f"simval {__version__} | {len(names)} engines")
        for name in names:
            print(f"  {name}")
        return 0

    if args.cmd == "compare":
        from simval.compare import compare_runs, largest_deltas
        from simval.report import fmt_num
        comp = _safe(lambda: compare_runs(args.run_a, args.run_b, selection=args.selection))
        if comp is None:
            return 1
        print(f"simval {__version__} | compare {args.run_a}  vs  {args.run_b}")
        ranked = largest_deltas(comp, n=8)
        header = f"  {'metric':<24} {'A':>12} {'B':>12} {'drel':>10}"
        print(header)
        print(rule_like(header))
        for name, drel in ranked:
            d = comp["deltas"][name]
            print(f"  {str(name)[:24]:<24} {fmt_num(d['a'], 12)} {fmt_num(d['b'], 12)} {fmt_num(drel, 10)}")
        n_total = len(comp["deltas"])
        if n_total > len(ranked):
            print(f"  (top {len(ranked)} of {n_total} metrics by drel)")
        return 0

    if args.cmd == "sweep":
        from simval.sweep import KEY_METRICS, sweep
        aliases = {
            "mean_rmsd_nm": "rmsd_mean",
            "final_rmsd_nm": "rmsd_final",
            "mean_rg_nm": "rg_mean",
            "energy_relative_range": "en_rel_range",
            "angular_momentum_relative_range": "L_rel_range",
        }
        out = sweep(args.folder, selection=args.selection, baseline=args.baseline)
        n_err = sum(1 for r in out["runs"] if "_error" in r)
        summary = f"simval {__version__} | sweep {args.folder} | {out['n']} runs"
        if n_err:
            summary += f" ({n_err} error)"
        print(summary)
        keys = [k for k in KEY_METRICS if any(k in r for r in out["runs"])]
        hdr = f"  {'run':<20} " + " ".join(f"{aliases.get(k, k[:14]):>14}" for k in keys)
        print(hdr)
        print(rule_like(hdr))
        base = out.get("baseline") or {}
        for r in out["runs"]:
            if "_error" in r:
                print(f"  {r['run']:<20}  ERROR: {r['_error']}")
                continue
            cells = []
            for k in keys:
                v = r.get(k)
                if v is None:
                    cells.append(f"{'-':>14}")
                elif base and k in base:
                    d = (v - base[k]) / (abs(base[k]) + 1e-12)
                    cells.append(f"{v:>9.3g}({d:+.0%})")
                else:
                    cells.append(f"{v:>14.3g}")
            print(f"  {r['run']:<20} " + " ".join(cells))
        if base:
            print(f"  (deltas vs baseline {args.baseline})")
        return 0

    if args.cmd == "orchestrate":
        import json

        from simval.manifest import canonical_digest
        from simval.orchestrate import load_grid, outliers, row_ok, run_grid, tabulate
        specs = _safe(lambda: load_grid(args.grid))
        if specs is None:
            return 1
        print(f"simval {__version__} | orchestrate {args.grid} | {len(specs)} runs")

        def _progress(row, i, total):
            name = row.get("run", f"run-{i}")
            if "_error" in row:
                print(f"[{i + 1}/{total}] {name}: ERROR {row['_error']}")
            elif row_ok(row):
                print(
                    f"[{i + 1}/{total}] {name}: ok "
                    f"(mismatch=0 checks_failed=0 wall={row.get('wall_s', 0)}s)"
                )
            else:
                print(
                    f"[{i + 1}/{total}] {name}: FAIL "
                    f"(mismatch={row.get('mismatch_count')} "
                    f"checks_failed={row.get('checks_failed')})"
                )

        start = time.perf_counter()
        results = _safe(lambda: run_grid(
            specs, ontos_bin=args.ontos_bin, cell_timeout_s=args.cell_timeout_s,
            progress=_progress,
        ))
        if results is None:
            return 1
        print(tabulate(results), end="")
        ol = outliers(results)
        if ol:
            print("  outliers (drift metric > median + k*MAD):")
            for o in ol:
                print(f"    {o['run']:<20} {o['metric']:<26} value={o['value']:.3g} "
                      f"median={o['median']:.3g} mad={o['mad']:.3g}")
        else:
            print("  outliers: none")
        n_clean = sum(1 for r in results if row_ok(r))
        n_err = sum(1 for r in results if "_error" in r)
        verdict = "CLEAN" if n_clean == len(results) else "FAILED"
        parts = [f"{n_clean}/{len(results)} runs clean"]
        if n_err:
            parts.append(f"{n_err} error")
        print(f"  grid: {verdict} | {' | '.join(parts)} | elapsed {fmt_dur(time.perf_counter() - start)}")
        if args.out:
            Path(args.out).write_text(
                json.dumps(
                    {
                        "grid": args.grid,
                        "runs": results,
                        "outliers": ol,
                        "canonical_digest": canonical_digest(
                            {"runs": results, "outliers": ol}
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"  results -> {args.out}")
        clean = bool(results) and all(row_ok(r) for r in results)
        return 0 if clean else 1

    if args.cmd == "verify-manifest":
        from simval.manifest import verify_manifest
        out = verify_manifest(args.manifest)
        verdict = "OK" if out["ok"] else "MISMATCH"
        print(f"simval {__version__} | manifest {verdict} | "
              f"{len(out['verified'])} verified, {len(out['tampered'])} tampered, {len(out['missing'])} missing "
              f"(original verdict: {out['verdict']})")
        for f in out["tampered"]:
            print(f"  [TAMPERED] {f}")
        for f in out["missing"]:
            print(f"  [MISSING]  {f}")
        return 0 if out["ok"] else 1

    if args.cmd == "fetch":
        from simval.fetch import fetch_structure
        info = _safe(lambda: fetch_structure(args.identifier, args.out_dir, source=args.source))
        if info is None:
            return 1
        print(f"simval {__version__} | fetched {info['id']} from {info['source']} -> {info['path']} ({info['bytes']} bytes)")
        return 0

    if args.cmd == "run":
        from simval.mdrun import run_md
        pdb = args.pdb
        if pdb.endswith(".pdb") or "/" in pdb:
            pdb_path = Path(pdb)
        else:
            from simval.fetch import fetch_structure
            info = _safe(lambda: fetch_structure(pdb, args.out, source="pdb"))
            if info is None:
                return 1
            pdb_path = Path(info["path"])
        out = _safe(lambda: run_md(pdb_path, args.out, steps=args.steps, dt=args.dt, temp=args.temp))
        if out is None:
            return 1
        print(f"simval {__version__} | MD complete -> {out}")
        start = time.perf_counter()
        manifest = _safe(lambda: run_diagnose(str(out), selection="protein and name CA"))
        if manifest is None:
            return 1
        _print_manifest_report(
            manifest, elapsed_s=time.perf_counter() - start,
            provenance_path=Path(out) / "provenance.json",
        )
        return 0 if manifest["verdict"] == "pass" else 1

    if args.cmd == "afold":
        from simval.afold import check_plddt_profile, fetch_plddt
        fetched = _safe(lambda: fetch_plddt(args.uniprot_id, args.out_dir))
        if fetched is None:
            return 1
        plddt, path = fetched
        r = check_plddt_profile(plddt)
        flag = "CONFIDENT" if r.passed else "LOW-CONFIDENCE"
        print(f"simval {__version__} | AlphaFold pLDDT {args.uniprot_id} | {flag}")
        print(f"  mean pLDDT={r.detail['mean_plddt']:.1f}  residues={r.detail['n_residues']}")
        print(f"  <50: {r.detail['fraction_below_50']*100:.1f}%  <70: {r.detail['fraction_below_70']*100:.1f}%  >90: {r.detail['fraction_above_90']*100:.1f}%")
        if r.detail["low_confidence_residue_indices"]:
            print(f"  low-confidence residues: {r.detail['low_confidence_residue_indices'][:20]}")
        return 0 if r.passed else 1

    if args.cmd == "export-omex":
        from simval.omex import export_omex
        out_path = args.out_path or f"{Path(args.run_dir).name}.omex"
        info = _safe(lambda: export_omex(args.run_dir, out_path))
        if info is None:
            return 1
        print(f"simval {__version__} | OMEX archive -> {info['path']} ({info['bytes']} bytes, {len(info['entries'])} entries)")
        return 0

    if args.cmd == "case-info":
        from simval.oracle import get_case
        from simval.report import fmt_num, tol_str
        case = _safe(lambda: get_case(args.name))
        if case is None:
            return 1
        print(f"simval {__version__} | case: {case.name}")
        print(f"  engine: {case.engine}  | ff: {case.force_field} | selection: {case.selection}")
        print(f"  description: {case.description}")
        print(f"  source: {case.source}")
        print("  reference metrics:")
        for name in sorted(case.reference_metrics):
            print(f"    {name:<28} {fmt_num(case.reference_metrics[name], 14)}")
        print("  tolerances:")
        if case.tolerances:
            for name in sorted(case.tolerances):
                spec = case.tolerances[name]
                print(f"    {name:<28} {tol_str(spec[0], spec[1] if len(spec) > 1 else None)}")
        else:
            print("    (defaults)")
        return 0

    if args.cmd == "freesolv":
        from simval.freesolv import check_against_experiment, list_compounds, lookup, search
        if args.search:
            hits = search(args.search)
            print(f"simval {__version__} | {len(hits)} FreeSolv matches for '{args.search}'")
            for h in hits:
                print(f"  {h['compound_id']}: {h['iupac'][:40]:40}  expt={h['expt_dG_kcal_mol']:+.2f} ± {h['expt_uncertainty']}")
        elif args.compound_id and args.computed_dG is not None:
            r = check_against_experiment(args.computed_dG, args.compound_id)
            flag = "PASS" if r.passed else "FAIL"
            print(f"simval {__version__} | FreeSolv {flag} | {r.detail['iupac'][:40]}")
            print(f"  computed: {r.detail['computed_dG_kcal_mol']:+.2f}  expt: {r.detail['experimental_dG_kcal_mol']:+.2f} ± {r.detail['experimental_uncertainty']}")
            print(f"  deviation: {r.detail['deviation_kcal_mol']:+.2f} kcal/mol  (tol {r.detail['tolerance_kcal_mol']:.2f})")
            return 0 if r.passed else 1
        elif args.compound_id:
            info = lookup(args.compound_id)
            print(f"simval {__version__} | FreeSolv {info['compound_id']}")
            print(f"  {info['iupac']}")
            print(f"  expt dG = {info['expt_dG_kcal_mol']:+.2f} ± {info['expt_uncertainty']} kcal/mol")
            print(f"  SMILES: {info['smiles']}")
        else:
            n = len(list_compounds())
            print(f"simval {__version__} | FreeSolv database: {n} compounds (CC-BY-4.0)")
            print("  simval freesolv <compound_id>           lookup")
            print("  simval freesolv <compound_id> <dG>      validate")
            print("  simval freesolv --search <name>         search")
        return 0

    if args.cmd == "validate":
        from simval.oracle import validate as oracle_validate
        result = _safe(lambda: oracle_validate(args.run_dir, args.case, selection=args.selection))
        if result is None:
            return 1
        verdict = "MATCH" if result.passed else "DRIFT"
        print(f"simval {__version__} | oracle case={args.case} | {verdict} | "
              f"{result.detail['n_checked']} metrics, {result.detail['n_failed']} drifted")
        if result.detail.get("error"):
            print(f"  error: {result.detail['error']}")
            if result.detail.get("identity"):
                print("  identity:")
                for line in render_identity(result.detail["identity"]):
                    print(line)
        for name, m in result.detail["metrics"].items():
            print(render_oracle_row(name, m))
        return 0 if result.passed else 1
    return 0
