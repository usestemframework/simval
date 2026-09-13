import numpy as np
from pathlib import Path

import pytest

from simval.diagnostics.energy import check_energy_drift
from simval.fixtures import drifting_energy_series, good_energy_series
from simval.manifest import build_manifest, compute_hashes, load_manifest, write_manifest


def test_pass_manifest_verdict_pass():
    result = check_energy_drift(good_energy_series())
    manifest = build_manifest({}, [result])
    assert manifest["verdict"] == "pass"
    assert manifest["schema"] == "simval.provenance.v1"
    assert manifest["tier2_signed_off"] is False


def test_fail_manifest_verdict_fail():
    result = check_energy_drift(drifting_energy_series())
    manifest = build_manifest({}, [result])
    assert manifest["verdict"] == "fail"
    assert manifest["diagnostics"][0]["name"] == "energy_drift"


def test_round_trip_with_hashes(tmp_path):
    f = tmp_path / "energy.npy"
    np.save(f, good_energy_series())
    result = check_energy_drift(good_energy_series())
    manifest = build_manifest({}, [result], files=[f])
    out = tmp_path / "provenance.json"
    write_manifest(manifest, out)
    loaded = load_manifest(out)
    assert loaded["verdict"] == manifest["verdict"]
    assert str(f) in loaded["files"]
    assert loaded["files"][str(f)] == compute_hashes([f])[str(f)]


def test_verify_manifest_detects_tampering(tmp_path):
    import numpy as np

    f = tmp_path / "energy.npy"
    np.save(f, good_energy_series())
    result = check_energy_drift(good_energy_series())
    manifest = build_manifest({}, [result], files=[f])
    out = tmp_path / "provenance.json"
    write_manifest(manifest, out)

    from simval.manifest import verify_manifest
    ok = verify_manifest(out)
    assert ok["ok"] is True

    np.save(f, drifting_energy_series())  # tamper
    tampered = verify_manifest(out)
    assert tampered["ok"] is False
    assert tampered["tampered"]


# --- PIPE-001: errored applicable diagnostics block the verdict ---


def _ctx(**over):
    from pathlib import Path

    from simval.context import RunContext

    ctx = RunContext(run_dir=Path("/tmp/x"), engine="gromacs", selection="protein")
    for k, v in over.items():
        setattr(ctx, k, v)
    return ctx


def test_errored_rmsf_check_fails_and_names_diagnostic(monkeypatch):
    from simval import pipeline
    from simval.diagnostics import rmsf as rmsf_mod

    def boom(*a, **k):
        raise RuntimeError("rmsf exploded")

    monkeypatch.setitem(pipeline._OPTIONAL_CHECK_DEPS, "per_residue_rmsf", ("json",))
    monkeypatch.setattr(rmsf_mod, "check_rmsf", boom)
    import numpy as np

    ctx = _ctx(
        ca_positions=np.zeros((2, 3, 3)),
        ca_reference=np.zeros((3, 3)),
    )
    results = pipeline.run_checks(ctx)
    errored = [r for r in results if r.name == "per_residue_rmsf"]
    assert errored and not errored[0].passed
    assert errored[0].detail["status"] == "error"
    assert "rmsf exploded" in errored[0].detail["error"]
    manifest = build_manifest({}, results)
    assert manifest["verdict"] == "fail"


def test_ca_load_error_fails_manifest_naming_rmsf():
    from simval import pipeline

    ctx = _ctx()
    ctx.extra["ca_load_error"] = "RuntimeError: trajectory exploded"
    results = pipeline.run_checks(ctx)
    errored = [r for r in results if r.name == "per_residue_rmsf"]
    assert errored and not errored[0].passed
    assert build_manifest({}, results)["verdict"] == "fail"


def test_errored_charge_state_and_hbonds_fail_manifest(monkeypatch, tmp_path):
    from simval import pipeline
    from simval.diagnostics import hbonds as hbonds_mod
    from simval.diagnostics import prep as prep_mod

    def boom(*a, **k):
        raise RuntimeError("diagnostic exploded")

    # Capabilities declared present so the checks are applicable and the
    # monkeypatched explosions become error results (PIPE-002 contract).
    monkeypatch.setitem(pipeline._OPTIONAL_CHECK_DEPS, "charge_state", ("json",))
    monkeypatch.setitem(pipeline._OPTIONAL_CHECK_DEPS, "hydrogen_bonds", ("json",))
    monkeypatch.setattr(prep_mod, "check_charge_state", boom)
    monkeypatch.setattr(hbonds_mod, "check_hydrogen_bonds", boom)
    ctx = _ctx(
        structure_path=tmp_path / "s.gro",
        tpr_path=tmp_path / "t.tpr",
        trajectory_path=tmp_path / "t.xtc",
    )
    results = pipeline.run_checks(ctx)
    names = {r.name for r in results if not r.passed}
    assert {"charge_state", "hydrogen_bonds"} <= names
    assert build_manifest({}, results)["verdict"] == "fail"


def test_declared_capability_absence_skips_without_invoking(monkeypatch, tmp_path):
    # PIPE-002: the only skip is a pre-declared absent optional dependency,
    # probed BEFORE the check runs — the check itself must never execute.
    from simval import pipeline
    from simval.diagnostics import prep as prep_mod

    def must_not_run(*a, **k):
        raise AssertionError("check invoked although its capability is absent")

    monkeypatch.setitem(pipeline._OPTIONAL_CHECK_DEPS, "charge_state", ("simval_no_such_dep_xyz",))
    monkeypatch.setattr(prep_mod, "check_charge_state", must_not_run)
    ctx = _ctx(structure_path=tmp_path / "s.gro", tpr_path=tmp_path / "t.tpr")
    results = pipeline.run_checks(ctx)
    assert not any(r.name == "charge_state" for r in results)
    assert "charge_state" in ctx.skipped
    assert "not applicable" in ctx.skipped["charge_state"]


def test_import_error_from_invoked_check_is_failing(monkeypatch, tmp_path):
    # PIPE-002 (regression): an ImportError raised by an APPLICABLE check
    # (capability present, lazy import broken inside the check) must be a
    # failing error result, never a skip.
    from simval import pipeline
    from simval.diagnostics import prep as prep_mod

    def broken_import(*a, **k):
        raise ImportError("No module named 'gmx'")

    monkeypatch.setitem(pipeline._OPTIONAL_CHECK_DEPS, "charge_state", ("json",))
    monkeypatch.setattr(prep_mod, "check_charge_state", broken_import)
    ctx = _ctx(structure_path=tmp_path / "s.gro", tpr_path=tmp_path / "t.tpr")
    results = pipeline.run_checks(ctx)
    errored = [r for r in results if r.name == "charge_state"]
    assert errored and not errored[0].passed
    assert errored[0].detail["status"] == "error"
    assert "ImportError" in errored[0].detail["error"]
    assert "charge_state" not in ctx.skipped
    assert build_manifest({}, results)["verdict"] == "fail"


# --- PYS-001: solver convergence is a mandatory, verdict-bearing check ---
# Stub pattern: the pyscf engine module imports pyscf lazily, so these
# pipeline-wiring tests run without the optional dependency installed.


def test_scf_converged_flag_check_passes_and_fails():
    from simval.pyscf_eng import check_scf_converged

    ok = check_scf_converged(True)
    assert ok.passed and ok.name == "scf_converged"
    bad = check_scf_converged(False)
    assert not bad.passed and bad.value == 0.0


def test_unconverged_scf_fails_pipeline_regardless_of_delta():
    from simval.pipeline import run_checks

    ctx = _ctx()
    # A flat energy sequence (final delta 0.0) with converged=False: the
    # delta check passes, the mandatory flag check must still fail.
    ctx.extra = {
        "scf_energies": [-1.1167593073964255, -1.1167593073964255],
        "final_energy": -1.1167593073964255,
        "n_electrons": 2,
        "converged": False,
    }
    results = run_checks(ctx)
    by_name = {r.name: r for r in results}
    assert by_name["scf_convergence"].passed is True
    assert by_name["scf_converged"].passed is False
    assert by_name["energy_sane"].passed is True
    assert build_manifest({}, results)["verdict"] == "fail"


def test_converged_scf_passes_pipeline_checks():
    from simval.pipeline import run_checks

    ctx = _ctx()
    ctx.extra = {
        "scf_energies": [-1.1167593073964255, -1.1167593073964255],
        "final_energy": -1.1167593073964255,
        "n_electrons": 2,
        "converged": True,
    }
    results = run_checks(ctx)
    assert all(
        r.passed for r in results if r.name.startswith("scf_") or r.name == "energy_sane"
    )


# --- IO-001: unique input selection + consumed-input provenance ---


def test_two_candidate_trajectories_rejected(tmp_path):
    # Filesystem-order dependent first-match used to pick one silently.
    from simval.context import GromacsEngine

    run = tmp_path / "amb"
    run.mkdir()
    (run / "a.xtc").write_bytes(b"0")
    (run / "b.xtc").write_bytes(b"0")
    (run / "conf.gro").write_bytes(b"0")
    with pytest.raises(ValueError, match="ambiguous trajectory"):
        GromacsEngine().load_context(run, selection="protein")


def test_two_topologies_rejected(tmp_path):
    from simval.context import GromacsEngine

    run = tmp_path / "amb"
    run.mkdir()
    (run / "a.gro").write_bytes(b"0")
    (run / "b.pdb").write_bytes(b"0")
    (run / "traj.xtc").write_bytes(b"0")
    # GROM-001: two candidates for the SAME role (structure) are still
    # ambiguous; roles themselves are no longer mutually exclusive.
    with pytest.raises(ValueError, match="ambiguous structure"):
        GromacsEngine().load_context(run, selection="protein")


def test_manifest_hashes_all_consumed_synthetic_inputs(tmp_path):
    import numpy as np

    from simval.fixtures import make_run_dir
    from simval.manifest import verify_manifest, write_manifest
    from simval.pipeline import diagnose

    run = make_run_dir(tmp_path / "good", good=True)
    manifest = diagnose(run)
    hashed = set(manifest["files"])
    assert hashed >= {
        str(run / "energy.npy"),
        str(run / "positions.npy"),
        str(run / "reference.npy"),
        str(run / "params.json"),
    }
    out = tmp_path / "prov.json"
    write_manifest(manifest, out)
    assert verify_manifest(out)["ok"] is True

    np.save(run / "positions.npy", np.zeros((4, 3, 3)))  # tamper a consumed input
    tampered = verify_manifest(out)
    assert tampered["ok"] is False
    assert str(run / "positions.npy") in tampered["tampered"]


def test_artifact_paths_canonically_sorted(tmp_path):
    from simval.fixtures import make_run_dir
    from simval.pipeline import diagnose

    run = make_run_dir(tmp_path / "s", good=True)
    manifest = diagnose(run)
    assert list(manifest["files"]) == sorted(manifest["files"])


def test_ontos_run_tracks_stream_and_meta_inputs(tmp_path):
    import shutil

    from simval.context import select_engine

    run = tmp_path / "ontos_run"
    shutil.copytree(
        Path(__file__).parent.parent / "examples" / "ontos" / "r_pentomino", run
    )
    ctx = select_engine(run).load_context(run, selection="default")
    assert {p.name for p in ctx.consumed_inputs} == {"ontos.stream", "ontos.json"}


# --- DET-001: canonical digest independent of volatile execution metadata ---


def test_identical_verifications_share_canonical_digest(tmp_path):
    from simval.fixtures import make_run_dir
    from simval.pipeline import diagnose

    run = make_run_dir(tmp_path / "a", good=True, seed=7)
    m1 = diagnose(run)
    m2 = diagnose(run)  # identical verification, different created_at
    assert m1["created_at"] is not None
    assert m1["canonical_digest"] == m2["canonical_digest"]
    from simval.manifest import canonical_digest

    assert canonical_digest(m1) == m1["canonical_digest"]
    m1["files"]["fake.npy"] = "0" * 64
    assert canonical_digest(m1) != m1["canonical_digest"]


def test_orchestrate_results_digest_ignores_wall_times():
    from simval.manifest import canonical_digest

    rows_a = [{"run": "x", "mismatch_count": 0, "wall_s": 1.5, "gen_wall_s": 0.2}]
    rows_b = [{"run": "x", "mismatch_count": 0, "wall_s": 9.9, "gen_wall_s": 3.0}]
    assert canonical_digest({"runs": rows_a}) == canonical_digest({"runs": rows_b})
    rows_c = [{"run": "x", "mismatch_count": 1, "wall_s": 1.5}]
    assert canonical_digest({"runs": rows_a}) != canonical_digest({"runs": rows_c})


# --- MAN-001: the canonical digest is verified, not just stored ---


def test_verify_manifest_detects_verdict_flip(tmp_path):
    # Files untouched, verdict flipped pass->fail after signing: the file
    # hashes still match, the digest must not.
    import numpy as np

    f = tmp_path / "energy.npy"
    np.save(f, good_energy_series())
    manifest = build_manifest({}, [check_energy_drift(good_energy_series())], files=[f])
    assert manifest["verdict"] == "pass"
    manifest["verdict"] = "fail"
    out = tmp_path / "prov.json"
    write_manifest(manifest, out)

    from simval.manifest import verify_manifest

    result = verify_manifest(out)
    assert result["ok"] is False
    assert "canonical_digest mismatch" in result["manifest_tampered"]
    assert not result["tampered"] and not result["missing"]


def test_verify_manifest_detects_diagnostics_edit(tmp_path):
    import numpy as np

    f = tmp_path / "energy.npy"
    np.save(f, good_energy_series())
    manifest = build_manifest({}, [check_energy_drift(good_energy_series())], files=[f])
    manifest["diagnostics"][0]["passed"] = not manifest["diagnostics"][0]["passed"]
    out = tmp_path / "prov.json"
    write_manifest(manifest, out)

    from simval.manifest import verify_manifest

    result = verify_manifest(out)
    assert result["ok"] is False
    assert "canonical_digest mismatch" in result["manifest_tampered"]


def test_verify_manifest_rejects_unsigned_manifest(tmp_path):
    import numpy as np

    f = tmp_path / "energy.npy"
    np.save(f, good_energy_series())
    manifest = build_manifest({}, [check_energy_drift(good_energy_series())], files=[f])
    del manifest["canonical_digest"]
    out = tmp_path / "prov.json"
    write_manifest(manifest, out)

    from simval.manifest import verify_manifest

    result = verify_manifest(out)
    assert result["ok"] is False
    assert "no canonical_digest" in result["manifest_tampered"]


# --- MAN-002: the digest covers the complete payload incl. metadata/methods ---


def test_digest_of_returned_manifest_matches_stored():
    from simval.manifest import canonical_digest

    meta = {"force_field": "amber99sb-ildn", "water_model": "tip3p",
            "methods": "MD was performed with GROMACS."}
    manifest = build_manifest(
        {}, [check_energy_drift(good_energy_series())], metadata=meta
    )
    assert canonical_digest(manifest) == manifest["canonical_digest"]


def test_mutating_force_field_or_methods_changes_digest():
    base_meta = {"force_field": "amber99sb-ildn", "methods": "MD was performed with GROMACS."}
    m1 = build_manifest({}, [check_energy_drift(good_energy_series())], metadata=dict(base_meta))
    changed_ff = dict(base_meta, force_field="charmm36")
    m2 = build_manifest({}, [check_energy_drift(good_energy_series())], metadata=changed_ff)
    changed_methods = dict(base_meta, methods="MD was performed with AMBER.")
    m3 = build_manifest({}, [check_energy_drift(good_energy_series())], metadata=changed_methods)
    assert len({m1["canonical_digest"], m2["canonical_digest"], m3["canonical_digest"]}) == 3


def test_diagnose_passes_metadata_into_build_manifest(tmp_path, monkeypatch):
    # Wiring check independent of engine: diagnose must hand ctx.metadata
    # to build_manifest (the digest covers it there), never append it after.
    from simval import pipeline
    from simval.fixtures import make_run_dir

    captured = {}
    real_build = pipeline.build_manifest

    def spy(params, results, *, files=None, image_digest=None, notes="",
            tier2_signed_off=False, metadata=None):
        captured["metadata"] = metadata
        return real_build(
            params, results, files=files, image_digest=image_digest,
            notes=notes, tier2_signed_off=tier2_signed_off, metadata=metadata,
        )

    monkeypatch.setattr(pipeline, "build_manifest", spy)
    run = make_run_dir(tmp_path / "good", good=True)
    (run / "methods.json").write_text('{"force_field": "amber14", "water": "tip3p"}')
    manifest = pipeline.diagnose(run)
    assert "metadata" in captured
    # The synthetic engine does not read methods.json; the None here proves
    # no post-hoc append happened (manifest carries no metadata key at all).
    assert captured["metadata"] is None
    assert "metadata" not in manifest


# --- PROV-001: every engine registers its consumed inputs ---


DOMAIN_EXAMPLES = [
    ("wave.json", "wave"),
    ("fluid.json", "fluid"),
    ("em.json", "em"),
    ("quantum.json", "quantum"),
    ("diffusion.json", "diffusion"),
    ("kinetics.json", "kinetics"),
    ("relativistic.json", "relativistic"),
]


@pytest.mark.parametrize("config,domain", DOMAIN_EXAMPLES)
def test_every_json_domain_registers_and_hashes_its_config(tmp_path, config, domain):
    import json
    import shutil

    from simval.manifest import verify_manifest, write_manifest
    from simval.pipeline import diagnose

    src = Path(__file__).parent.parent / "examples" / domain
    fixture = next(p for p in src.iterdir() if p.is_dir())
    run = tmp_path / domain
    shutil.copytree(fixture, run)
    manifest = diagnose(run)
    assert str(run / config) in manifest["files"], manifest["files"].keys()

    out = tmp_path / "prov.json"
    write_manifest(manifest, out)
    assert verify_manifest(out)["ok"] is True

    cfg = json.loads((run / config).read_text())
    key = next(iter(cfg))
    cfg[key] = cfg[key] * 2 if isinstance(cfg[key], (int, float)) else "mutated"
    (run / config).write_text(json.dumps(cfg))
    tampered = verify_manifest(out)
    assert tampered["ok"] is False
    assert str(run / config) in tampered["tampered"]


def test_engine_with_no_consumed_inputs_is_a_contract_error():
    from simval.pipeline import _artifact_files

    ctx = _ctx()
    with pytest.raises(ValueError, match="no consumed inputs"):
        _artifact_files(ctx)


# --- GROM-001: MD inputs are roles; a normal gro+tpr+xtc dir loads ---


def test_md_input_roles_are_not_mutually_exclusive(tmp_path):
    # conf.gro + topol.tpr + traj.xtc used to be "ambiguous topology".
    from simval._util import (
        select_run_topology,
        select_structure,
        select_trajectory_topology,
    )

    run = tmp_path / "normal"
    run.mkdir()
    (run / "conf.gro").write_bytes(b"g")
    (run / "topol.tpr").write_bytes(b"t")
    (run / "traj.xtc").write_bytes(b"x")
    assert select_structure(run).name == "conf.gro"
    assert select_run_topology(run).name == "topol.tpr"
    # Documented precedence: structure > run topology > prmtop/psf.
    assert select_trajectory_topology(run).name == "conf.gro"

    (run / "conf.gro").unlink()
    assert select_trajectory_topology(run).name == "topol.tpr"

    (run / "sys.prmtop").write_bytes(b"p")
    (run / "topol.tpr").unlink()
    assert select_trajectory_topology(run).name == "sys.prmtop"


def test_same_role_ambiguity_still_rejected(tmp_path):
    from simval._util import select_structure, select_trajectory_topology

    run = tmp_path / "amb"
    run.mkdir()
    (run / "a.gro").write_bytes(b"0")
    (run / "b.gro").write_bytes(b"0")
    with pytest.raises(ValueError, match="ambiguous structure"):
        select_structure(run)
    with pytest.raises(ValueError, match="ambiguous"):
        select_trajectory_topology(run)


# --- PROV-002: the tpr role is consumed regardless of optional extraction ---


def ctx_paths(manifest):
    from pathlib import Path

    return [Path(k) for k in manifest["files"]]


def test_tpr_consumed_when_atom_type_extraction_not_available(tmp_path, monkeypatch):
    # The tpr used to enter consumed_inputs only when MDAnalysis atom-type
    # extraction succeeded; in the typed not-available branch (unsupported
    # TPR version -> []) the tpr was omitted while charge_state still
    # consumed it via gmx dump. Provenance must register the role when it
    # is SELECTED, not when the optional extraction wins.
    pytest.importorskip("MDAnalysis")
    datafiles = pytest.importorskip("MDAnalysisTests.datafiles")
    import shutil

    from simval import io
    from simval.manifest import verify_manifest, write_manifest
    from simval.pipeline import diagnose

    run = tmp_path / "adk"
    run.mkdir()
    shutil.copy(datafiles.XTC, run / "traj.xtc")
    shutil.copy(datafiles.GRO, run / "conf.gro")
    (run / "topol.tpr").write_bytes(b"pretend tpr bytes")

    monkeypatch.setattr(io, "load_atom_types", lambda *a, **k: [])  # typed not-available
    from simval.diagnostics import prep as prep_mod

    monkeypatch.setattr(prep_mod, "net_charge_from_tpr", lambda p: 0.0)  # working gmx dump

    manifest = diagnose(run, selection="protein and name CA")
    assert any(p.name == "topol.tpr" for p in ctx_paths(manifest))

    charge = next(d for d in manifest["diagnostics"] if d["name"] == "charge_state")
    assert charge["passed"] is True  # the gmx-dump path really consumed the tpr

    out = run / "provenance.json"
    write_manifest(manifest, out)
    (run / "topol.tpr").write_bytes(b"tampered tpr bytes")
    tampered = verify_manifest(out)
    assert tampered["ok"] is False
    assert any(entry.endswith("topol.tpr") for entry in tampered["tampered"])
