from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from simval import __version__
from simval.result import DiagnosticResult

SCHEMA = "simval.provenance.v1"

# Volatile execution metadata: excluded from the canonical digest so two
# identical verifications produce byte-identical canonical payloads even
# though their wall-clock timestamps and timings differ (audit DET-001).
VOLATILE_KEYS = ("created_at", "canonical_digest", "wall_s", "gen_wall_s", "verify_wall_s")


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def canonical_digest(payload: dict) -> str:
    """SHA-256 over the canonical JSON of a payload (sorted keys, tight
    separators) after recursively stripping volatile execution metadata."""
    import json as _json

    blob = _json.dumps(_strip_volatile(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def compute_hashes(paths: Iterable, *, chunk: int = 1 << 16, base=None) -> dict[str, str]:
    """sha256 of every path, keyed canonically. With `base`, relative
    entries resolve under it and the KEY stays the run-relative form
    (audit MAN-003); absolute entries pass through unchanged."""
    out: dict[str, str] = {}
    for p in paths:
        path = Path(p)
        key_path = path
        if base is not None and not path.is_absolute():
            key_path = Path(path)
            path = Path(base) / path
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        out[str(key_path)] = h.hexdigest()
    return out


def build_manifest(
    params,
    results,
    *,
    files=None,
    files_base=None,
    image_digest=None,
    notes="",
    tier2_signed_off: bool = False,
    metadata: dict | None = None,
) -> dict:
    verdict = all(r.passed for r in results) if results else False
    diagnostics = [r.to_dict() if isinstance(r, DiagnosticResult) else r for r in results]
    payload = {
        "schema": SCHEMA,
        "simval_version": __version__,
        "verdict": "pass" if verdict else "fail",
        "params": params,
        "diagnostics": diagnostics,
        "files": compute_hashes(files, base=files_base) if files else {},
        "image_digest": image_digest,
        "tier2_signed_off": bool(tier2_signed_off),
        "notes": notes,
    }
    # The complete payload — metadata/methods included — is assembled BEFORE
    # the digest is computed, exactly once, immediately before serialization
    # (audit MAN-002): nothing may ride outside the signed body.
    if metadata:
        payload["metadata"] = metadata
        if "methods" in metadata:
            payload["methods"] = metadata["methods"]
    # Volatile envelope (created_at) rides alongside; the digest covers
    # only the canonical payload.
    return {
        **payload,
        "canonical_digest": canonical_digest(payload),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_manifest(manifest: dict, path) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text())


def verify_manifest(path) -> dict:
    """Re-hash every file the manifest references and confirm it still
    matches, and recompute the canonical digest over the manifest payload
    (audit MAN-001): verdict/diagnostics edits after signing must not pass.
    Closes the provenance loop: a manifest is not just written, it can be
    checked later for tampering or drift.

    Relative file entries resolve against the MANIFEST'S OWN DIRECTORY,
    never the process CWD (audit MAN-003): a shadow file at the same
    relative path in another directory must not be able to stand in for
    the tampered original."""
    manifest = load_manifest(path)
    base = Path(path).parent
    stored = manifest.get("files", {})
    out = {"verified": [], "tampered": [], "missing": [], "verdict": manifest.get("verdict")}
    stored_digest = manifest.get("canonical_digest")
    if not stored_digest:
        out["manifest_tampered"] = (
            "manifest carries no canonical_digest (unsigned/pre-DET-001 artifact)"
        )
    else:
        recomputed = canonical_digest(manifest)
        if recomputed != stored_digest:
            out["manifest_tampered"] = (
                f"canonical_digest mismatch: stored {stored_digest[:12]}... != "
                f"recomputed {recomputed[:12]}... — the payload was edited after signing"
            )
    for rel, expected in stored.items():
        p = Path(rel)
        if not p.is_absolute():
            p = base / p
        if not p.exists():
            out["missing"].append(rel)
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 16), b""):
                h.update(block)
        actual = h.hexdigest()
        (out["verified"] if actual == expected else out["tampered"]).append(rel)
    out["ok"] = not out["tampered"] and not out["missing"] and "manifest_tampered" not in out
    return out
