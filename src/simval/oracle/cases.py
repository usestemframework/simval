from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

_REFERENCES_DIR = Path(__file__).parent / "references"
_MANIFEST_NAME = "MANIFEST.json"

# Supported reference-format versions: the current 0.1.x series. Anything
# else (or a missing version) fails closed — a golden of unknown vintage
# must not silently flow into verdicts.
SUPPORTED_REFERENCE_SERIES = (0, 1)

_SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")


def canonical_reference_bytes(d) -> bytes:
    """Canonical serialization every reference digest is computed over:
    sorted keys, compact separators — layout edits are inert, content
    edits change the digest (audit GOLD-002)."""
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


def reference_digest(d) -> str:
    return hashlib.sha256(canonical_reference_bytes(d)).hexdigest()


def _load_manifest() -> dict[str, str]:
    """Load references/MANIFEST.json — the pinned record of every shipped
    golden's canonical-content digest (audit GOLD-002).

    Trust model, stated honestly: the manifest is the pinned artifact and
    is itself a checked-in file not covered by any further hash. It
    raises the cost of tampering from editing one file to editing two
    coordinated files (golden + manifest) in the same commit; a reviewer
    is asked to notice either. Missing or malformed = fail closed.
    """
    p = _REFERENCES_DIR / _MANIFEST_NAME
    if not p.exists():
        raise ValueError(
            f"reference manifest {_MANIFEST_NAME} is missing from {_REFERENCES_DIR}: "
            "every shipped golden must be pinned (regenerate deliberately via "
            "scripts/regen_reference_manifest.py)"
        )
    m = json.loads(p.read_text())
    if not isinstance(m, dict) or not m:
        raise ValueError(f"reference manifest {_MANIFEST_NAME} must be a non-empty name -> sha256 map")
    bad = sorted(k for k, v in m.items() if not isinstance(k, str) or not isinstance(v, str))
    if bad:
        raise ValueError(f"reference manifest entries must be strings: {bad}")
    bad_hashes = sorted(k for k, v in m.items() if not _SHA256_RE.match(v))
    if bad_hashes:
        raise ValueError(f"reference manifest entries must be lowercase sha256 hex: {bad_hashes}")
    return dict(m)


def _verify_pinned(path: Path, d, name: str) -> None:
    """Content-pin enforcement (audit GOLD-002) + name/stem binding (GOLD-003).

    A golden living in the references directory must be pinned and match
    the manifest digest. A file elsewhere whose embedded name is pinned is
    verified against that pin (a same-named impostor must not ride it);
    an unpinned external file is loadable ONLY through the explicit
    load_case_unpinned() escape hatch, never via get_case()."""
    if name != path.stem:
        raise ValueError(
            f"reference case at {path}: embedded name {name!r} does not match the "
            f"file stem {path.stem!r} — a byte-copied golden must not pose as "
            "another case (audit GOLD-003)"
        )
    pinned = _load_manifest().get(name)
    if pinned is None:
        if path.parent == _REFERENCES_DIR:
            raise ValueError(
                f"reference case {name!r}: no {_MANIFEST_NAME} entry pins its content — "
                "an unpinned golden must not flow into verdicts (regenerate the manifest "
                "deliberately via scripts/regen_reference_manifest.py)"
            )
        return
    digest = reference_digest(d)
    if digest != pinned:
        raise ValueError(
            f"reference case {name!r}: content digest {digest} does not match the pinned "
            f"{_MANIFEST_NAME} entry {pinned} — the golden was modified without regenerating "
            "the manifest. If the change is intentional, rerun scripts/regen_reference_manifest.py"
        )


def _check_identity(name: str, identity) -> dict[str, str]:
    """Scenario identity (audit ORA-003): a required, non-empty map of
    run-relative input path -> sha256(hex). A golden without identity is
    malformed and fails closed at load time."""
    if not isinstance(identity, dict) or not identity:
        raise ValueError(
            f"reference case {name!r}: golden carries no scenario 'identity' "
            "(required: {input path: sha256} for every scenario-defining input)"
        )
    bad_types = sorted(k for k, v in identity.items() if not isinstance(k, str) or not isinstance(v, str))
    if bad_types:
        raise ValueError(f"reference case {name!r}: identity keys/values must be strings: {bad_types}")
    bad_hashes = sorted(k for k, v in identity.items() if not _SHA256_RE.match(v))
    if bad_hashes:
        raise ValueError(
            f"reference case {name!r}: identity entries must be lowercase sha256 hex: {bad_hashes}"
        )
    return dict(identity)


def _check_reference_version(name: str, version) -> str:
    if not isinstance(version, str) or not version:
        raise ValueError(f"reference case {name!r}: missing reference_version")
    try:
        parts = tuple(int(p) for p in version.split("."))
    except ValueError:
        raise ValueError(f"reference case {name!r}: malformed reference_version {version!r}") from None
    if len(parts) != 3 or parts[:2] != SUPPORTED_REFERENCE_SERIES:
        raise ValueError(
            f"reference case {name!r}: unsupported reference_version {version!r} "
            f"(supported series: {'.'.join(str(p) for p in SUPPORTED_REFERENCE_SERIES)}.x)"
        )
    return version


@dataclass
class ReferenceCase:
    name: str
    description: str
    engine: str
    force_field: str
    selection: str
    reference_metrics: dict
    tolerances: dict
    source: str
    source_hash: str
    reference_version: str = ""
    ignore: list[str] = field(default_factory=list)
    identity: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "engine": self.engine,
            "force_field": self.force_field,
            "selection": self.selection,
            "reference_metrics": self.reference_metrics,
            "tolerances": self.tolerances,
            "source": self.source,
            "source_hash": self.source_hash,
            "reference_version": self.reference_version,
            "ignore": list(self.ignore),
            "identity": dict(self.identity),
        }


def _load(path: Path) -> ReferenceCase:
    d = json.loads(path.read_text())
    name = d.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"reference case at {path}: missing or invalid 'name'")
    _verify_pinned(path, d, name)
    ignore = d.get("ignore", [])
    if not isinstance(ignore, list) or not all(isinstance(m, str) for m in ignore):
        raise ValueError(f"reference case {name!r}: 'ignore' must be a list of metric names")
    unknown_ignores = sorted(set(ignore) - set(d["reference_metrics"]))
    if unknown_ignores:
        raise ValueError(
            f"reference case {name!r}: ignore lists unknown metrics {unknown_ignores}"
        )
    return ReferenceCase(
        name=name,
        description=d.get("description", ""),
        engine=d.get("engine", "gromacs"),
        force_field=d.get("force_field", ""),
        selection=d.get("selection", "protein and name CA"),
        reference_metrics=d["reference_metrics"],
        tolerances=d.get("tolerances", {}),
        source=d.get("source", ""),
        source_hash=d.get("source_hash", ""),
        reference_version=_check_reference_version(name, d.get("reference_version")),
        ignore=ignore,
        identity=_check_identity(name, d.get("identity")),
    )


def _reference_files() -> list[Path]:
    return [p for p in _REFERENCES_DIR.glob("*.json") if p.name != _MANIFEST_NAME]


def list_cases() -> list[str]:
    if not _REFERENCES_DIR.exists():
        return []
    return sorted(p.stem for p in _reference_files())


def get_case(name: str) -> ReferenceCase:
    """Load a SHIPPED golden by its plain registered name (audit GOLD-003).

    Only plain names resolve: path-like, absolute, and dot names are
    rejected before any filesystem use, and the resolved path must stay
    inside the references directory. External/custom references go through
    load_case_unpinned(), which is explicit about its weaker guarantees."""
    if (
        not isinstance(name, str)
        or name in ("", ".", "..")
        or "/" in name
        or "\\" in name
        or Path(name).is_absolute()
    ):
        raise KeyError(
            f"invalid reference case name {name!r}: get_case accepts only plain "
            f"registered case names; available: {list_cases()}"
        )
    path = _REFERENCES_DIR / f"{name}.json"
    if path.resolve().parent != _REFERENCES_DIR.resolve():
        raise KeyError(
            f"reference case {name!r} resolves outside the references directory"
        )
    if not path.exists():
        raise KeyError(f"unknown reference case: {name!r}; available: {list_cases()}")
    return _load(path)


def load_case_unpinned(path) -> ReferenceCase:
    """Load a reference JSON from an explicit path WITHOUT requiring it to
    be a shipped, manifest-pinned golden (audit GOLD-003).

    Test/tool escape hatch, clearly named: a file whose embedded name IS
    pinned is still digest-verified (the impostor rule); only genuinely
    unpinned external references load through this door. Verdict-bearing
    code must use get_case()/load_all() instead."""
    return _load(Path(path))


def load_all() -> dict[str, ReferenceCase]:
    if not _REFERENCES_DIR.exists():
        return {}
    return {p.stem: _load(p) for p in _reference_files()}
