"""Reference-case registry tests: version gate + every shipped metric resolves
to exactly one comparison rule (audit ORA-001/GOLD-001 corpus property)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from simval.oracle import cases as cases_mod
from simval.oracle import load_all
from simval.oracle.cases import ReferenceCase
from simval.oracle.validate import _DEFAULT_TOLERANCES, compare_metrics


def test_every_reference_metric_resolves_to_exactly_one_rule():
    problems = []
    for name, case in load_all().items():
        for metric in case.reference_metrics:
            if metric in case.ignore:
                continue
            if metric not in case.tolerances and metric not in _DEFAULT_TOLERANCES:
                problems.append(f"{name}: {metric} has no rule")
        for metric in case.tolerances:
            if metric not in case.reference_metrics:
                problems.append(f"{name}: tolerance for unknown metric {metric}")
    assert not problems, problems


def test_no_undeclared_ignores_in_shipped_references():
    for name, case in load_all().items():
        assert case.ignore == [], f"{name}: unexpected ignore list {case.ignore}"


def test_reference_version_gate_rejects_unsupported(tmp_path):
    src = cases_mod._REFERENCES_DIR / "adk_morph.json"
    d = json.loads(src.read_text())
    d["name"] = "future_version_probe"  # not a shipped reference: no manifest pin
    d["reference_version"] = "99.0.0"
    path = tmp_path / "future_version_probe.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="unsupported reference_version"):
        cases_mod.load_case_unpinned(path)


def test_reference_version_gate_rejects_missing(tmp_path):
    src = cases_mod._REFERENCES_DIR / "adk_morph.json"
    d = json.loads(src.read_text())
    d["name"] = "noversion_probe"  # not a shipped reference: no manifest pin
    del d["reference_version"]
    path = tmp_path / "noversion_probe.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="missing reference_version"):
        cases_mod.load_case_unpinned(path)


def test_all_shipped_references_parse_under_version_gate():
    all_cases = load_all()
    # 14 after benzene_hydration_fep was retired (FEP-001, AUDIT.md §G).
    assert len(all_cases) >= 14
    assert all(c.reference_version.startswith("0.1.") for c in all_cases.values())


def test_case_roundtrip_carries_new_fields():
    case = load_all()["adk_morph"]
    d = case.to_dict()
    assert d["reference_version"] == "0.1.0"
    assert d["ignore"] == []
    restored = ReferenceCase(**{k: v for k, v in d.items()})
    assert restored.reference_version == case.reference_version


def test_compare_metrics_missing_candidate_metric_fails():
    ref = {"n_frames": 10, "mean_rg_nm": 2.2}
    cand = {"mean_rg_nm": 2.2}
    out = compare_metrics(cand, ref)
    assert out["__passed__"] is False
    assert out["n_frames"]["passed"] is False
    assert "missing from candidate" in out["n_frames"]["error"]


def test_compare_metrics_missing_policy_is_config_error():
    with pytest.raises(ValueError, match="'weird_metric'.*no comparison policy"):
        compare_metrics({"weird_metric": 1.0}, {"weird_metric": 1.0})


def test_compare_metrics_ignore_exempts_metric():
    ref = {"n_frames": 10}
    out = compare_metrics({}, ref, ignore=["n_frames"])
    assert out["__passed__"] is True
    assert out["n_frames"]["tol_kind"] == "ignore"


# --- ORA-002: bounds are bounds, not distances from the golden ---


def _case_tols(name):
    return load_all()[name].tolerances


def test_wave_cfl_ceiling_rejects_supercritical():
    # The old abs-1.0 rule let CFL 1.4 pass against a 0.5 golden.
    ref = {"cfl": 0.5, "energy_growth": 1.007, "n_steps": 2000}
    tols = _case_tols("wave_pulse_stable")
    bad = compare_metrics({"cfl": 1.4, "energy_growth": 1.007, "n_steps": 2000}, ref, tols)
    assert bad["cfl"]["passed"] is False
    ok = compare_metrics({"cfl": 0.99, "energy_growth": 1.007, "n_steps": 2000}, ref, tols)
    assert ok["cfl"]["passed"] is True
    assert ok["__passed__"] is True


def test_em_courant_ceiling_rejects_supercritical():
    ref = {"courant": 0.7071, "em_energy_growth": 1.014, "n_steps": 800}
    tols = _case_tols("em_pulse_stable")
    bad = compare_metrics({"courant": 1.2, "em_energy_growth": 1.0, "n_steps": 800}, ref, tols)
    assert bad["courant"]["passed"] is False
    assert bad["em_energy_growth"]["passed"] is True  # 1.0 is bounded energy, not drift


def test_bound_kinds_require_finite_candidates():
    ref = {"cfl": 0.5}
    out = compare_metrics(
        {"cfl": float("nan")}, ref, {"cfl": ["max", 1.0]}
    )
    assert out["cfl"]["passed"] is False
    out = compare_metrics({"cfl": float("inf")}, ref, {"cfl": ["max", 1.0]})
    assert out["cfl"]["passed"] is False
    out = compare_metrics({"p": float("nan")}, {"p": 0.99}, {"p": ["min", 0.9]})
    assert out["p"]["passed"] is False


def test_interval_and_min_bounds():
    assert compare_metrics({"tau": 0.8}, {"tau": 0.8}, {"tau": ["interval", 0.5, 2.0]})["tau"]["passed"]
    assert not compare_metrics({"tau": 2.8}, {"tau": 0.8}, {"tau": ["interval", 0.5, 2.0]})["tau"]["passed"]
    assert not compare_metrics({"tau": 0.3}, {"tau": 0.8}, {"tau": ["interval", 0.5, 2.0]})["tau"]["passed"]
    assert compare_metrics({"p": 0.9999}, {"p": 0.9999}, {"p": ["min", 0.9]})["p"]["passed"]
    assert not compare_metrics({"p": 0.5}, {"p": 0.9999}, {"p": ["min", 0.9]})["p"]["passed"]


# --- ORA-004: tolerance rules are externally supplied too ---


@pytest.mark.parametrize(
    "spec",
    [
        ["abs", float("inf")],
        ["abs", float("nan")],
        ["abs", 0.0],
        ["abs", -1.0],
        ["rel", float("inf")],
        ["max", float("inf")],
        ["min", float("nan")],
        ["interval", 0.5, float("inf")],
        ["interval", 2.0, 0.5],
        ["exact", 1.0],
        ["vacuous", 1.0],
        ["abs"],
        ["abs", 1.0, 2.0],
    ],
)
def test_malformed_tolerance_rules_rejected(spec):
    with pytest.raises(ValueError, match="tolerance rule"):
        compare_metrics({"m": 1.0}, {"m": 1.0}, {"m": spec})


def test_wellformed_tolerance_rules_accepted():
    assert compare_metrics({"m": 1.0}, {"m": 1.0}, {"m": ["abs", 0.5]})["__passed__"]
    assert compare_metrics({"m": 1.0}, {"m": 1.0}, {"m": ["interval", 0.0, 2.0]})["__passed__"]


# --- ORA-003: scenario identity is required and enforced ---


def test_every_shipped_golden_carries_identity():
    for name, case in load_all().items():
        assert case.identity, f"{name}: no scenario identity"
        for path, digest in case.identity.items():
            assert "/" not in path and case.identity[path] == digest


def test_golden_without_identity_fails_closed(tmp_path):
    src = cases_mod._REFERENCES_DIR / "wave_pulse_stable.json"
    d = json.loads(src.read_text())
    d["name"] = "noidentity_probe"  # not a shipped reference: no manifest pin
    del d["identity"]
    path = tmp_path / "noidentity_probe.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="no scenario 'identity'"):
        cases_mod.load_case_unpinned(path)


def test_golden_with_malformed_identity_rejected(tmp_path):
    src = cases_mod._REFERENCES_DIR / "wave_pulse_stable.json"
    d = json.loads(src.read_text())
    d["name"] = "badhash_probe"  # not a shipped reference: no manifest pin
    d["identity"] = {"wave.json": "not-a-hash"}
    path = tmp_path / "badhash_probe.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="sha256"):
        cases_mod.load_case_unpinned(path)
    d["name"] = "empty_probe"
    d["identity"] = {}
    path = tmp_path / "empty_probe.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="no scenario 'identity'"):
        cases_mod.load_case_unpinned(path)


def test_validate_missing_declared_input_fails_closed(tmp_path):
    # The fep engine still detects the run via its dhdl.csv glob when the
    # manifest is gone, but the golden declares fep.json: identity must
    # fail closed before any metric runs (no alchemlyb needed).
    import shutil

    from simval.oracle import validate

    src = Path(__file__).parent.parent / "examples" / "fep" / "synthetic"
    run = tmp_path / "synthetic"
    shutil.copytree(src, run)
    (run / "fep.json").unlink()
    result = validate(run, "fep_synthetic")
    assert result.passed is False
    assert result.detail["identity"]["fep.json"]["problem"] == "missing input"


def test_validate_undeclared_scenario_input_fails_closed(tmp_path):
    # A manifest that pulls in an extra data file the golden never declared
    # changes the scenario; the identity gate must reject it even though
    # every declared file is byte-identical.
    import json
    import shutil

    from simval.oracle import validate

    src = Path(__file__).parent.parent / "examples" / "fep" / "synthetic"
    run = tmp_path / "synthetic"
    shutil.copytree(src, run)
    (run / "extra.csv").write_text("0.0\n")
    manifest = json.loads((run / "fep.json").read_text())
    manifest["files"] = manifest["files"] + ["extra.csv"]
    (run / "fep.json").write_text(json.dumps(manifest))
    result = validate(run, "fep_synthetic")
    assert result.passed is False
    assert result.detail["identity"]["__undeclared_inputs__"]["problem"] == ["extra.csv"]


# --- GOLD-002: shipped goldens are pinned by an independent manifest ---


def test_manifest_covers_exactly_the_shipped_references():
    manifest = cases_mod._load_manifest()
    shipped = set(cases_mod.list_cases())
    assert shipped, "no shipped references found"
    assert set(manifest) == shipped


def test_unmodified_references_load():
    # Every shipped golden loads through the manifest-verified path.
    cases = load_all()
    assert cases
    assert all(c.source_hash is not None for c in cases.values())


def test_tampered_tolerance_rejected_without_manifest_update(tmp_path, monkeypatch):
    # A tolerance loosened in a shipped golden without regenerating the
    # manifest must fail closed at load, naming the case.
    import shutil

    refs = tmp_path / "references"
    refs.mkdir()
    shipped = sorted(cases_mod._REFERENCES_DIR.glob("*.json"))
    for p in shipped:
        shutil.copy(p, refs / p.name)
    monkeypatch.setattr(cases_mod, "_REFERENCES_DIR", refs)
    victim = refs / "wave_pulse_stable.json"
    d = json.loads(victim.read_text())
    d["tolerances"]["cfl"] = ["max", 99.0]
    victim.write_text(json.dumps(d, indent=2))
    with pytest.raises(ValueError, match="wave_pulse_stable.*does not match the pinned MANIFEST"):
        cases_mod._load(victim)


def test_layout_only_edit_is_inert(tmp_path, monkeypatch):
    # The pin covers canonical content, not file layout: re-indenting a
    # golden without touching content must still load.
    import shutil

    refs = tmp_path / "references"
    refs.mkdir()
    for p in sorted(cases_mod._REFERENCES_DIR.glob("*.json")):
        shutil.copy(p, refs / p.name)
    monkeypatch.setattr(cases_mod, "_REFERENCES_DIR", refs)
    victim = refs / "wave_pulse_stable.json"
    d = json.loads(victim.read_text())
    victim.write_text(json.dumps(d, indent=4, sort_keys=False))
    case = cases_mod._load(victim)
    assert case.name == "wave_pulse_stable"


def test_shipped_file_without_manifest_entry_fails_closed(tmp_path, monkeypatch):
    import shutil

    refs = tmp_path / "references"
    refs.mkdir()
    for p in sorted(cases_mod._REFERENCES_DIR.glob("*.json")):
        shutil.copy(p, refs / p.name)
    manifest = json.loads((refs / "MANIFEST.json").read_text())
    del manifest["wave_pulse_stable"]
    (refs / "MANIFEST.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(cases_mod, "_REFERENCES_DIR", refs)
    with pytest.raises(ValueError, match="no MANIFEST.json entry pins its content"):
        cases_mod._load(refs / "wave_pulse_stable.json")


def test_missing_manifest_fails_closed(tmp_path, monkeypatch):
    import shutil

    refs = tmp_path / "references"
    refs.mkdir()
    for p in sorted(cases_mod._REFERENCES_DIR.glob("*.json")):
        if p.name == "MANIFEST.json":
            continue
        shutil.copy(p, refs / p.name)
    monkeypatch.setattr(cases_mod, "_REFERENCES_DIR", refs)
    with pytest.raises(ValueError, match="manifest"):
        cases_mod._load(refs / "wave_pulse_stable.json")


def test_malformed_manifest_entry_fails_closed(tmp_path, monkeypatch):
    import shutil

    refs = tmp_path / "references"
    refs.mkdir()
    for p in sorted(cases_mod._REFERENCES_DIR.glob("*.json")):
        shutil.copy(p, refs / p.name)
    manifest = json.loads((refs / "MANIFEST.json").read_text())
    manifest["wave_pulse_stable"] = "not-a-hash"
    (refs / "MANIFEST.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(cases_mod, "_REFERENCES_DIR", refs)
    with pytest.raises(ValueError, match="manifest entries must be lowercase sha256"):
        cases_mod._load(refs / "wave_pulse_stable.json")


def test_manifest_not_listed_as_a_case():
    assert "MANIFEST" not in cases_mod.list_cases()


# --- GOLD-003: get_case is a name registry, not a path loader ---


def test_get_case_rejects_path_traversal_and_absolute_names():
    refs = cases_mod._REFERENCES_DIR
    for bad in (
        "../wave_pulse_stable",
        "references/../wave_pulse_stable",
        "sub/wave_pulse_stable",
        str(refs / "wave_pulse_stable.json"),
        "/etc/passwd",
        ".",
        "..",
        "",
    ):
        with pytest.raises(KeyError, match="plain registered case names|unknown reference case"):
            cases_mod.get_case(bad)


def test_byte_copied_golden_rejected_on_name_stem_mismatch(tmp_path):
    # Byte-copy pinned case B to a file named like anything else: the
    # embedded name no longer matches the stem, so the impostor must be
    # rejected before any pin comparison could be confused.
    src = cases_mod._REFERENCES_DIR / "fep_synthetic.json"
    impostor = tmp_path / "impostor.json"
    impostor.write_bytes(src.read_bytes())
    with pytest.raises(ValueError, match="embedded name 'fep_synthetic'.*stem 'impostor'"):
        cases_mod.load_case_unpinned(impostor)


def test_byte_copied_golden_rejected_inside_references_dir(tmp_path, monkeypatch):
    # Same byte-copy, but posed inside the references directory under a
    # shipped case's name: get_case must refuse it on the name/stem rule.
    import shutil

    refs = tmp_path / "references"
    refs.mkdir()
    for p in sorted(cases_mod._REFERENCES_DIR.glob("*.json")):
        shutil.copy(p, refs / p.name)
    monkeypatch.setattr(cases_mod, "_REFERENCES_DIR", refs)
    (refs / "wave_pulse_stable.json").write_bytes(
        (refs / "fep_synthetic.json").read_bytes()
    )
    with pytest.raises(ValueError, match="embedded name 'fep_synthetic'"):
        cases_mod.get_case("wave_pulse_stable")


def test_get_case_loads_every_shipped_case():
    for name in cases_mod.list_cases():
        assert cases_mod.get_case(name).name == name


def test_load_case_unpinned_verifies_pinned_impostor(tmp_path):
    # An external file claiming a PINNED case name is still digest-checked:
    # a content-modified impostor must not ride the shipped pin.
    src = cases_mod._REFERENCES_DIR / "wave_pulse_stable.json"
    d = json.loads(src.read_text())
    d["reference_metrics"]["cfl"] = 0.9
    impostor = tmp_path / "wave_pulse_stable.json"
    impostor.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="does not match the pinned MANIFEST"):
        cases_mod.load_case_unpinned(impostor)
