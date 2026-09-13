"""Internal helpers shared across modules."""
from __future__ import annotations

from pathlib import Path


def find_files(run: Path, *patterns: str):
    """First file in `run` matching any pattern (in order), or None.

    Only for detection heuristics - semantic input selection must use
    find_unique (audit IO-001)."""
    for pat in patterns:
        hit = next(run.glob(pat), None)
        if hit:
            return hit
    return None


def find_unique(run: Path, *patterns: str, what: str = "input"):
    """The exactly-one file in `run` matching any pattern (sorted for a
    deterministic error), or None when nothing matches.

    Ambiguity is an error, not a first-match: two candidate trajectories
    (or topologies) in one run-dir used to silently pick whichever the
    filesystem returned first (audit IO-001)."""
    hits = sorted({hit for pat in patterns for hit in run.glob(pat)}, key=str)
    if len(hits) > 1:
        raise ValueError(
            f"ambiguous {what}: expected exactly one match for {list(patterns)} in {run}, "
            f"found {[str(h) for h in hits]}"
        )
    return hits[0] if hits else None


# --- MD input roles (audit GROM-001) -------------------------------------
#
# A normal GROMACS run-dir carries a structure (.gro/.pdb), a run topology
# (.tpr), and possibly an Amber/CHARMM topology (.prmtop/.psf) — these are
# ROLES, not mutually-exclusive alternatives. Only multiple candidates for
# the SAME role are ambiguous. When several roles can serve as the
# trajectory topology, the precedence below is deterministic and documented.

STRUCTURE_PATTERNS = ("*.gro", "*.pdb")
RUN_TOPOLOGY_PATTERNS = ("*.tpr",)
ALTERNATE_TOPOLOGY_PATTERNS = ("*.prmtop", "*.psf")
TRAJECTORY_PATTERNS = ("*.xtc", "*.dcd", "*.trr", "*.nc")


def select_structure(run: Path):
    """Structure role (.gro/.pdb): exactly one, or None."""
    return find_unique(run, *STRUCTURE_PATTERNS, what="structure")


def select_run_topology(run: Path):
    """Run-topology role (.tpr): exactly one, or None."""
    return find_unique(run, *RUN_TOPOLOGY_PATTERNS, what="run topology (tpr)")


def select_alternate_topology(run: Path):
    """Amber/CHARMM topology role (.prmtop/.psf): exactly one, or None."""
    return find_unique(run, *ALTERNATE_TOPOLOGY_PATTERNS, what="topology (prmtop/psf)")


def select_trajectory_topology(run: Path):
    """Topology used to load the trajectory.

    Documented precedence: structure (.gro/.pdb) > run topology (.tpr) >
    Amber/CHARMM (.prmtop/.psf). Ambiguity is only an error WITHIN a role.
    """
    for select in (select_structure, select_run_topology, select_alternate_topology):
        hit = select(run)
        if hit is not None:
            return hit
    return None


def gromacs_scenario_inputs(run: Path) -> list[Path]:
    """Every PRESENT GROMACS scenario role (audit ORA-005).

    One canonical enumerator shared by provenance (consumed inputs) and
    reference identity (the golden's scenario pin): structure, run
    topology (.tpr), alternate topology, trajectory, energy xvg, mdp,
    topology .top, params.json, methods.json — whichever are present.
    Identity used to hash only the selected trajectory topology (so a
    conf.gro hid topol.tpr) plus trajectory and xvg; .mdp/.top/
    params.json/methods.json were omitted entirely. Role selection keeps
    find_unique semantics: two candidates for one role are an error.
    """
    roles = [
        select_structure(run),
        select_run_topology(run),
        select_alternate_topology(run),
        find_unique(run, *TRAJECTORY_PATTERNS, what="trajectory"),
        find_unique(run, "*.xvg", what="energy file (xvg)"),
        find_unique(run, "mdout.mdp", "*.mdp", what="run parameters (mdp)"),
        find_unique(run, "*.top", what="topology file (top)"),
        run / "params.json",
        run / "methods.json",
    ]
    return sorted(
        {p for p in roles if p is not None and p.exists()}, key=lambda p: p.name
    )


def gromacs_force_field_problem(run: Path) -> str | None:
    """Force-field contract vs methods metadata (audit ORA-005): when both
    a methods.json and a topology .top are present, the declared force
    field must agree with what the .top derives. None = no complaint
    (nothing to cross-check, or they agree)."""
    methods = run / "methods.json"
    if not methods.exists():
        return None
    top = find_unique(run, "*.top", what="topology file (top)")
    if top is None:
        return None
    import json

    try:
        declared = (json.loads(methods.read_text()) or {}).get("force_field")
    except ValueError:
        return f"methods.json is not valid JSON: {methods}"
    if not declared:
        return None
    from simval.metadata import extract_force_field

    derived = extract_force_field(top)
    if derived:
        # GROMACS include dirs carry a ".ff" suffix ("amber99sb-ildn.ff")
        # that methods metadata drops ("amber99sb-ildn").
        derived = derived[:-3] if derived.endswith(".ff") else derived
    if derived and derived != declared:
        return (
            f"force-field contract violation: methods.json declares {declared!r} "
            f"but the run topology {top.name} derives {derived!r}"
        )
    return None
