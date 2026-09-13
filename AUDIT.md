# Omnilator — Audit v1 (5-pass synthesis)

> **Register note (2026-09-11):** §A–E below are the historical planning audit
> (pre-rename; kept verbatim). The verification-oracle audit register — the
> authoritative record of oracle-hardening findings against HEAD 7495370 —
> is §F at the bottom of this file. Batch 2 (HEAD 0bdb2d3) is §G; batch 3
> (HEAD 74e8ee6) is §H; batch 4 (HEAD f636c8c) is §I; batch 5 (HEAD 6ed8f28)
> is §J.

> Synthesis of five independent analysis passes: competitive positioning, technical architecture, risk/oversight hunt, Hive codebase audit, product/UX/scope.
> Convergent findings (flagged by ≥2 passes) are high-confidence. The audit's own conclusion is in §D.

---

## A. Convergent findings (≥2 independent passes agree)

**A1. Two products, not one — cut creative as a vertical.** *(A, D-implicit, E)*
Generative Blender/Godot art and GROMACS MD share only the Blender binary — different users, workflows, willingness to pay, retention loops. **blender-mcp already ships the creative half for free** (23k★, MIT, opencode/Cursor/Claude-native). Recommendation: Blender = render target for sim output *only*; move generative-creative to Out-of-Scope. **This reopens the naming decision** — "Omnilator" (omni) contradicts a sim-only thesis.

**A2. No user named; and NL-first fights the paying user.** *(A, C, E)*
Experts (comp chemists, CFD engineers, biotech R&D) pay and don't churn — but they want determinism and reproducibility, which NL+LLM actively undermines. NL solves a *novice* problem; novices churn. The expert's actual complaint about GROMACS is "setup is boilerplate and sweeps are hard to manage," not "I wish I could type English at it." That's a tooling problem, not a conversation problem.

**A3. Chat UI contradicts the reproducibility principle.** *(B, E)*
§9 promises reproducibility-from-day-one; §3's chat UI hides provenance. The truthful shape of reproducible science is a **notebook** (Jupyter-shape) with NL-accepting cells + an always-on provenance panel, not "chat + canvas." Chat is the viral demo and the worse product.

**A4. Verification is the real moat — and the biggest hand-wave.** *(A, B, C, E — all four non-codebase passes)*
"Validator agent" must become: **deterministic diagnostics** (energy drift from `.edr`, RMSD/Rg via MDAnalysis, unit checks, equilibration gates) + **human sign-off** for semantic correctness. An LLM-validator checking an LLM-executor is *correlated failure* (same training biases). This is the one thing the free agent+MCP+Docker combo can't do (rated ~5% solved) — so it must be the **headline**, not a Phase-3 line.

**A5. GROMACS→USD is fiction — use the existing stack.** *(A, B)*
No USD/glTF schema for particles/bonds/PBC/thermodynamics; building it is a multi-month second product. Drop it for MD. Use **MDAnalysis** (reads `.xtc/.trr/.edr` natively) + **NGLView** (live) + **Molecular Nodes** (Blender render addon). **HDF5/numpy as interchange.** Reopen USD only if a non-MD vertical needs scene interchange.

**A6. Phase 1 is ~3× over-built.** *(A, B, C, E)*
Gateway/Registry (Anthropic + GitHub now run MCP registries — redundant), the 5-agent graph, K8s — all wrong for solo Phase 1. **Minimal Phase-1 stack:**
```
notebook UI (defer canvas/timeline)
  → one ReAct agent (LangGraph create_react_agent, cloud model via LiteLLM)
  → MCP client (langchain_mcp_adapters, stdio) — no gateway
  → tools.yaml registry (name/transport/command)
  → docker run + mounted ./workspace volume
  → GROMACS MCP server (you write it)
  → MDAnalysis → HDF5/numpy (no USD)
  → NGLView (live) | Molecular Nodes (render)
  → deterministic diagnostics (gmx energy + RMSD + thresholds → pass/fail)
  → JSON provenance manifest (params, image digest, hashes, diagnostics)
```

**A7. Security / RCE surface is unacknowledged — CRITICAL.** *(C primary; D found Hive's eval-skills as a liability to avoid)*
Arbitrary Python via `bpy` = **remote code execution as a product feature.** Containers aren't a default boundary (shared kernel); community images are unscanned; MCP has had prompt-injection / tool-poisoning CVEs. The word "security" appears nowhere in the plan. Mitigations: rootless/gVisor/Kata containers, `--network=none --read-only`, no-default-egress, digest-pinned images, **human approval gate before any generated code runs**, treat MCP tool results/descriptions as untrusted input.

**A8. GPL licensing constrains the paid/cloud layer — unanalyzed.** *(A, C)*
GROMACS (LGPL/GPL), Blender (GPLv3), OpenFOAM (GPLv3). **The plan's own "no forks, wrappers-only" rule is what makes open-core legal** — arm's-length invocation (subprocess/container/MCP socket) doesn't propagate copyleft. This must be stated explicitly + backed by a **license-audit task (R9)**. "Deep integration" must mean *deep-at-the-boundary*, never *by-linking*, or the proprietary tier becomes GPL-derivative.

**A9. The reproducibility claim is internally contradicted.** *(B, C)*
LLM planning is non-deterministic (same prompt → different `.mdp` → different trajectory); GPU/CUDA results differ across hardware; images drift. Honest claim: **captured artifacts** (params, scripts, image digest, seed, hardware/CUDA record) **are re-runnable**; the NL→artifact step is **not** bit-reproducible. Drop "bit-for-bit" from any external-facing language.

**A10. "Local-first" conflates two things.** *(B, C, E)*
- **Execution/data local-first** = achievable, worth defending.
- **LLM local-first** = impossible on 4GB N5000s; marginal even on Proxmox without a 40GB+ VRAM GPU (→ only 7B–32B models, too weak for reliable bpy/GROMACS codegen; they hallucinate `.mdp` params).
**Cloud LLM is effectively required for Phase 1.** Reframe: *"local-first where it matters (data, control, compute); cloud-assisted where it must (LLM)."* Abstract the model behind LiteLLM so local stays a config swap.

**A11. Greenfield Python, not Hive reuse — decided with evidence.** *(B, C, D-audit)*
Hive (`aispace`) is TS/Next.js/Vercel-AI-SDK + hardcoded code-tools + business-SaaS schema (Operator/Inbox/BusinessTemplate, LEGAL/MARKETING agent roles), **dormant ~4.5 months** (last real feature commit 2026-02-11; the May commit is a WIP snapshot). Its agent runtime — the heart of Omnilator — doesn't transfer to Python/LangGraph/MCP. Reuse tax > rewrite cost. The salvageable language-agnostic bits (Docker pool, realtime, queue) are each <400 lines and trivially portable. **`openclaw/` is an unrelated vendored messaging-assistant — ignore it.**
→ Steal 4 *patterns* (not code): (1) container-pool + sandbox flags (`container-pool.ts`, `docker.ts`), (2) circuit breaker, (3) review/approval + LLM quality-gate (maps 1:1 to sim-correctness human-in-loop), (4) realtime event vocabulary + tool-policy model. **§4 #2 and #3 → RESOLVED: greenfield Python.**

**A12. Meta-risk: over-planning may already be the failure mode.** *(C, E)*
A session was spent on naming. Phase 0's R1/R2/R5/R6 are best answered **by building the slice**, not by more research. For a solo dev running ~10 projects, permanent pre-build is the classic death. Highest-leverage move: collapse Phase 0+1 into a **2–3 week timebox with a kill-by date.**

**A13. Competitive landscape is empty; the existence-proof threat is real.** *(A, C)*
- **Surrogate vendors** (PhysicsX, BeyondMath, Navista SimAI) build ML models that *skip the solvers Omnilator wraps* — a strictly better tech vector. Need a thesis for why "orchestrate slow legacy solvers" wins (candidate: "trustworthy ground-truth while surrogates mature").
- **Incumbents** (Schrödinger, Rescale, SimScale, Cadence) own the buyer *and* the natural paid layer (managed reproducible compute). Rescale's model = your paid layer; the wedge is the sub-$1k/mo indie/academic price point they ignore + local-first.
- **Jupyter ecosystem** is where the scientist user already lives and already does CAD+MD+CFD+provenance. "New workbench" is the wrong framing for that user.
- **The composite substitute** (AI coding agent + community MCP servers + Docker) gets ~60–70% for throwaway single-tool work. Omnilator's 30–40% gap is *entirely the unsexy layer*: converters, provenance, verification, job survival. That gap is the product. Lead with it.

---

## B. Decisions the audit forces (the forks)

| # | Decision | Audit recommendation |
|---|---|---|
| D1 | **Scope & user** | Narrow to **one vertical (GROMACS MD) for one user (biotech/pharma R&D; secondary: rusty PI)**. Cut generative-creative. This cascades to name, UI, and registry abstraction. |
| D2 | **Runtime** (was §4 #2/#3) | **Greenfield Python.** Hive = reference only. RESOLVED. |
| D3 | **LLM sourcing** | **Cloud LLM for Phase 1** (Claude/GPT via LiteLLM); local execution/data. Resolves the local-first ambiguity. |
| D4 | **Verification model** | **Deterministic diagnostics + human sign-off**, not an agent. Promote to Phase-1 exit criterion. |
| D5 | **Process** | **Collapse Phase 0 into Phase 1**, 2–3 week timebox, kill-by date. Stop researching, build. |
| D6 | **Name** (was §4 #5) | **Reopen.** If scope narrows to simulation (recommended), "Omnilator/omni" is dissonance. The folder rename shouldn't anchor you. |

---

## C. Recommended concrete edits to PLAN.md (propose as v0.3)

1. **Rewrite §1 vision line** → "reproducible simulation workbench with NL assist, provenance, and shared viz." Drop "creative tools" from the vision; Blender = viz implementation detail.
2. **Add §1.5 "Primary user + JTBD."** Named persona, ARPU, current alternative (Schrödinger/CHARMM-GUI/GROMACS+VMD), daily loop = **parameter sweep + compare (RMSD/Rg/energy) + export reproducible bundle.**
3. **Add §1.6 "Existence proof."** blender-mcp, Jupyter-AI, the agent+MCP+Docker composite, and the % gap table. Lead the pitch with the gap.
4. **Add §2.5 "Competitive landscape"** — surrogates / incumbents / Jupyter / composite, with a one-line wedge each.
5. **Rewrite §3 architecture diagram** to the minimal Phase-1 stack (A6). Soften "Blender universal viz" from axiom to Phase-2 review point; for MD, NGLView/ChimeraX/OVITO are purpose-built.
6. **Delete the MCP Gateway/Registry box** (or reduce to "consume Anthropic/GitHub registries; do not build one").
7. **§3 UI line** → "notebook with NL-accepting cells + always-on provenance panel + comparison view" (not chat+canvas).
8. **Resolve §4 #2/#3 → greenfield Python** (A11). Resolve §4 #5 → **reopened** (D6). Add §4 #6: "paid layer + buyer + price band."
9. **Rewrite R6** → "size integration of MDAnalysis/NGLView/Molecular Nodes" (drop USD research for MD). **Add R9 (license audit per tool)** as a pre-paid-layer gate.
10. **§6 risk register** — add rows: Security/RCE (Critical), GPL/cloud licensing (High), Meta-risk: over-planning & solo-bandwidth (High, *up-rated*), Cost/context-window (Med-High). **Split "sim-correctness" into ≥4 sub-risks**; strike "Validator agent," replace with "deterministic linters + assertions; LLM summarizes only; human sign-off on export."
11. **§7** — collapse Phase 0 into Phase 1 (2–3 wk timebox + kill-by date). Phase-1 exit criterion: *"a target user completes a real weekly task unaided, the diagnostics flag a deliberately-broken run as failed while passing a sane one, and exports a reproducible bundle."*
12. **§9 reproducibility line** → honest two-tier claim (A9).
13. **Add §10 "Kill criterion"** — one paragraph: what observation, on what date, kills Omnilator.
14. **Add §11 "Licensing & open-core boundary"** (A8).

---

## D. What to do RIGHT NOW (the audit's own conclusion)

The five passes agree on one meta-point: **the planning process is at risk of becoming the product.** Naming got a session; Phase 0 as written could eat months for a solo dev who context-switches across ~10 projects — and R1/R2/R5/R6 are lower-information than just building the slice.

So the real fork is binary:

- **(a) Commit and build.** Approve D1–D5, apply the §C edits as v0.3, and start the GROMACS thin slice inside a 2–3 week timebox with a kill-by date. No more research passes.
- **(b) Shelve honestly.** If you can't name (i) the one user, (ii) your own day-2 loop, or (iii) why this beats Tebian/Neutron/Teploy for your bandwidth — Omnilator is architecture-as-fun, and the kind move is to park it until there's a real first user.

Doing another planning round is *not* on this list. The highest-quality next action is a decision, not more analysis.

---

## E. Questions only Tyler can answer (sharpest across all passes)

1. **Opportunity-cost kill criterion:** why does Omnilator get bandwidth over Tebian / Neutron / Teploy? Which of those slips so this ships?
2. **Have you talked to a single comp chemist / CFD engineer / biotech researcher about NL control specifically?** If not, that's task #1 — ahead of any code.
3. **Unfair advantage:** domain knowledge of MD/CFD? Agent orchestration? Viz? If none, a specialist out-executes you on any single vertical. Where do you win?
4. **Would you use this weekly yourself, for what?** If you can't name your own day-2 loop, the target user can't either.
5. **Is "local-first" a religious constraint or a marketing differentiator?** (Audit says: for sim on your hardware it's *factually* cloud-assisted. Are you willing to say that out loud in the positioning?)
6. **Is the paid layer real or hypothetical?** If hypothetical, GPL-licensing risk drops a tier. If real/near-term, R9 (license audit) is the most urgent task in the doc.
7. **Did the naming session feel like progress or procrastination?** (Honest calibration of whether A12 is already active.)

---

## Per-pass one-liners

- **A (positioning):** blender-mcp already ships the creative half free; the only defensible product is the hard half — trusted, reproducible, multi-tool *scientific* sim. Lead with the gap, not "NL workbench."
- **B (architecture):** 3× over-built for solo Phase 1; kill gateway/5-agent/K8s; verification + GROMACS→USD are the two load-bearing hand-waves.
- **C (risk):** §6 covers ~30% of the real surface — security, GPL, silent-wrong-physics, and the meta-risk of over-planning are all missing.
- **D (Hive audit):** greenfield Python; Hive is TS/AI-SDK/code-tools, dormant 4.5 mo; steal 4 patterns, ignore `openclaw/`.
- **E (product/UX):** engineering architecture wearing a product's clothes — no user named, and its three loudest choices (NL-first, chat UI, creative+science) each contradict its own reproducibility principle.

---

## F. Verification-oracle audit register (2026-09-11)

Findings verified against HEAD `7495370` by an external review; implemented
in the commits below. simval is the trust anchor — a false pass is the worst
failure mode, so every fix errs toward failing closed.

| ID | Severity | Area | Finding | Status |
|---|---|---|---|---|
| ORA-001 | P0 | oracle/validate.py | `compare_metrics` silently skipped reference metrics absent from the candidate or lacking a tolerance rule; `all_pass` stayed true for unexamined fields | Fixed — every reference metric is mandatory (missing-from-candidate = FAIL, missing-policy = hard config error); per-case `ignore` list (default empty); count/version-like metrics exact by default; corpus test proves every shipped metric resolves to exactly one rule |
| ORA-002 | P0 | oracle + references | Physical ceilings encoded as abs-distance-from-golden (wave CFL 1.4 passed vs ref 0.5 tol 1.0) | Fixed — `max`/`min`/`interval` bound kinds with finiteness required; wave CFL max 1.0, EM Courant max 1.0, energy-growth ceilings as max, tau interval [0.5,2.0], fourier max 0.5, p_up_swing min 0.9 |
| ONT-001 | P0 | ontos_gravity/orchestrate | Oracle derived body count, horizon, feature modes and events from the candidate stream; orchestrator did not persist radial/restitution/friction/walls | Fixed — immutable `GravityContract` from metadata validated before replay (body count, exact horizon, event schedule) and after (multipole/radial/shells record presence, contact params bit-exact); replay modes driven by the contract; every CLI-affecting field persisted to ontos.json; all 20 corpus examples carry full contract metadata |
| ONT-002 | P0 | ontos.py, ontos_gravity.py | No strict grammar/cardinality/EOF finalization; a header-only v1 stream passed with zero mismatches | Fixed — strict per-version parser state machines (per-tick record order and cardinality, duplicate/omission rejection, no timestamped records outside an open frame, dangling boundary records at EOF rejected, non-consecutive ticks rejected); reference-match independently fails on zero compared records; mutation suite: header-only, TickHeader-only, drop-one, duplicate-one, duplicate-frame, append-after-last-tick |
| ONT-003 | P0 | ontos_gravity.py | v2 Snapshot population parsed but only counted as compared | Fixed — compared against the stream header body count; single-bit flip in a corpus tag-2 payload produces nonzero mismatch_count |
| ONT-004 | P0 | ontos.py, ontos_gravity.py | Totals/State/Body records never checked their tick against the active TickHeader | Fixed — every timestamped record must repeat its frame tick, including pre-tick records vs the TickHeader they precede; tick +/-1 mutations on Totals/State/Body/pre-tick records all rejected |
| ONT-005 | P0 | ontos.py, ontos_gravity.py | Noncanonical encodings aliased (v1 level != 1 became coarse; region coords ry*2+rx unvalidated so (2,0) aliased (0,1)) | Fixed — v1 levels strictly {0,1}; v2 levels documented sets; rx/ry each validated in {0,1} on every region-carrying record; body ids strictly sequenced; body region/level bytes and contact pseudo-id ranges validated |
| ONT-006 | P0 | orchestrate.py, ontos_gravity.py | Friction validated only as `friction < 0.0` so NaN passed | Fixed — all externally-supplied floats (grid spec + ContactParams in streams) require isfinite with positively-expressed range checks at both layers; NaN/+inf/-inf restitution and friction rejected |
| GOLD-001 | P1 | oracle/cases.py | `reference_version` present in goldens but never parsed or gated | Fixed — parsed into ReferenceCase; 0.1.x series accepted; missing/unsupported fails closed before metric comparison (99.0.0 fixture rejected) |
| AUDIO-001 | P1 | ontos_audio.py | Monopole contacts resolved against FINAL collapse mass, not the mass at contact time | Fixed — collapse-mass timeline captured in one stream-order pass at each Contact record (body masses immutable, resolve post-pass); falsifying recollapse test (mass 10 -> contact -> recollapse 20 -> contact); corpus WAVs unchanged (walls corpus carries identical recollapse masses); the ontos Rust CLI already computed mu at contact time |
| PIPE-001 | P1 | context.py, pipeline.py, manifest.py | Applicable diagnostics raising Exception were silently skipped and absent from the all() verdict | Fixed — ImportError from known optional deps is an explicit skip; any other raise from an applicable check (charge_state, hydrogen_bonds, per_residue_rmsf, box_cutoff, steric_clashes) is a failing status=error DiagnosticResult that blocks the verdict; CA-load distinguishes not-applicable (ValueError) from errored |
| ORCH-001 | P1 | orchestrate.py | Grid run names used as filesystem paths, recursively deleted if present | Fixed — names preflight-validated (no absolute/separators/../duplicates) before any filesystem mutation; run directories use internal `cell-<index>` ids with display names kept separate; sentinel-file and ../escape tests |
| ORCH-002 | P1 | orchestrate.py | normalize_spec ran outside the per-cell try, aborting the whole grid | Fixed — normalization inside the failure boundary; errors attributed to cell index/name in an `_error` row; [valid, invalid, valid] yields 3 rows with the third executed |
| ORCH-003 | P1 | orchestrate.py, cli.py | Empty grid loaded fine and `all([])` exited 0 | Fixed — load_grid rejects an empty list; CLI success requires bool(results) |
| IO-001 | P1 | _util.py, validate.py, pipeline.py, context.py | Input selection used first glob match (filesystem-order dependent); provenance hashed one file per pattern | Fixed — semantic inputs require exactly one match (ambiguity = error); RunContext.consumed_inputs tracked by the synthetic/Gromacs/ontos engines; manifest hashes ALL consumed inputs, canonically sorted; mutating any consumed .npy fails verify-manifest |
| DET-001 | P2 | manifest.py, orchestrate.py | Wall-clock timestamps/timings embedded in reports break byte-determinism | Fixed — canonical_digest over the payload with volatile keys (created_at, wall timings) recursively stripped; identical verifications produce identical digests; orchestrate --out carries the canonical digest alongside raw rows |

Golden/reference files touched, and why:

- `references/h2_rhf.json` — `final_energy` renamed to `final_energy_hartree`
  (and its tolerance entry) to match the metric the oracle actually computes;
  `_pyscf_metrics` now also reports `n_cycles`/`scf_last_delta`.
- `references/wave_pulse_stable.json`, `em_pulse_stable.json` — ceilings
  re-encoded as bounds (ORA-002), counts exact.
- `references/fluid_flow_stable.json` — tau as interval [0.5, 2.0] mirroring
  check_tau_stability; tau_in_range exact.
- `references/quantum_spin.json` — p_up_swing as min 0.9 floor.
- `references/diffusion_heat.json` — fourier as max 0.5 mirroring
  check_fourier_stability.
- `examples/ontos*/**/ontos.json` — all 20 corpus runs upgraded with full
  run-contract metadata (mode, ticks, bodies, events, observer, contact
  flags/params, multipole) derived from each stream's own records; the two
  legacy pre-spec-20 examples (`collapse`, `collapse_observer`) marked
  multipole=false. No stream or WAV bytes changed; every example still
  verifies through the full engine path.

---

## G. Verification-oracle audit register, batch 2 (2026-09-12)

Findings verified against HEAD `0bdb2d3` by an external review; implemented
in the commits below. Same posture as §F: simval is the trust anchor, every
fix errs toward failing closed. Three findings (GROM-001, FEP-002,
PIPE-002-residual) are regressions from the §F batch — the locking tests and
half-fixes are corrected this time.

| ID | Severity | Area | Finding | Status |
|---|---|---|---|---|
| FEP-001 | P0 | fep.py, oracle/validate.py, references | MBAR overlap compared as abs-distance from the golden (golden stored 0.0), so a zero-overlap candidate passed while check_overlap() declared it unreliable | Fixed — overlap is a `min` bound invariant (>= 0.05) mirroring check_overlap. `benzene_hydration_fep` RETIRED: the alchemtest fixture's actual MBAR overlap min-eigenvalues (Coulomb leg 7.4e-4, VDW leg 1.4e-8, pymbar 4.0.3) are far below the 0.05 invariant, so the golden as shipped contradicted the domain check; regeneration impossible, case removed (owner decision, documented here — not silently dropped). fep_synthetic remains the shipped FEP reference |
| ONT-007 | P0 | ontos_gravity._main, ci.yml | Standalone/cross-verify CLI path called verify_stream_gravity with expected=None, bypassing the expected-run contract; CI cross-verification used that path | Fixed — the CLI requires the contract: `--metadata ontos.json` or the same flags given to ontos (--ticks/--bodies/--demote-at/.../--restitution/--friction/--test-ic), translated into a GravityContract (gravity) or the life contract check; bare-stream replay exists ONLY behind `--no-contract` with a loud stderr warning and a `[NO CONTRACT]` output marker, and is not used by CI; ci.yml cross-verify now passes the full contract (life cases too) |
| ONT-008 | P0 | ontos_gravity.py | Observer-run contract validation dropped event multiplicity (extra got-want events skipped when observer exists; check_zoom_policy used membership sets), so duplicated RegionLevel records passed | Fixed — both layers compare multisets: check_zoom_policy requires stream events == Counter(policy) + Counter(requested); the verifier contract requires the same equality, computing the deterministic policy from (seed, observer, ticks, requested); metadata carrying an observer but no ticks now fails closed (the policy needs the requested horizon). Duplicate-request and duplicate-policy-record mutation tests reject |
| ONT-009 | P0 | ontos.py, ontos_eng.py, orchestrate.py | Life-mode contract checked event identity/counts but not placement — untimed initialization events accepted at any frame boundary | Fixed — life_contract_problems (shared by the engine adapter and the CLI) attributes every RegionLevel record to the boundary tick it precedes and requires requested events to precede TickHeader 1; any later-boundary occurrence is a contract violation. Moving a scheduled demote to a later boundary fails the ontos_run_contract check |
| PAR-001 | P0 | diagnostics/params.py, context.py | NaN passed the `value <= 0` positivity checks (params.json JSON accepts the NaN literal) | Fixed — every spec-covered quantity is checked for finiteness BEFORE dimensional/range checks, with a clear violation; NaN/+inf/-inf on dt or ref_t fails the params diagnostic and the verdict |
| PIPE-002 | P0 | pipeline.py, tests/test_manifest.py (REGRESSION from PIPE-001) | `_guarded()` treated ANY ImportError as a successful omission and the locking test enshrined it | Fixed — the only skip is a pre-declared absent optional dependency, probed via find_spec BEFORE the check runs (`_OPTIONAL_CHECK_DEPS` allowlist); once an applicable check is invoked, ANY exception — ImportError included — is a failing error result. Locking test rewritten to the new contract; monkeypatch-ImportError test added; the two PIPE-001 error tests now declare the capability present (deliberate test update) |
| ORA-003 | P0 | oracle/cases.py, oracle/validate.py, all references | Golden `source_hash`/scenario identity was unenforced — validation keyed off conservation metrics alone, so a different scenario with matching metrics passed | Fixed — golden schema gains a required `identity` map {input name: sha256}; a golden without identity (or malformed hashes) fails closed at load. validate() computes the candidate's scenario identity (engine-aware canonical inputs; cheap file selection + hashing, no heavy deps) and requires exact equality — missing input, content mismatch, or undeclared scenario input fails before any metric is computed; an undetectable run-dir (defining config deleted) also fails closed. All 14 shipped goldens now carry identity computed from their fixtures |
| MAN-001 | P1 | manifest.py | verify_manifest() re-hashed files but never recomputed canonical_digest — verdict/diagnostics edits passed | Fixed — verify_manifest recomputes the canonical digest over the loaded payload and requires equality; a mismatch (or a manifest with no digest at all, e.g. the pre-DET-001 provenance artifacts shipped under examples/) reports `manifest_tampered` and ok=False. Verdict-flip and diagnostics-edit tests with files untouched fail |
| MAN-002 | P1 | manifest.py, pipeline.py | Digest computed before metadata/methods were appended, so the stored digest never covered them | Fixed — build_manifest takes the complete payload (metadata/methods included) and computes the digest exactly once, immediately before serialization; diagnose passes ctx.metadata in at build time instead of appending after. Digest-of-returned == stored, force_field/methods mutations change the digest, and the gromacs diagnose path carries metadata inside the signed body |
| PROV-001 | P1 | wave/fluid/em/quantum/kinetics/diffusion/relativistic/nbody/fep/pyscf/qiskit engines, pipeline.py | JSON-driven domains never registered consumed inputs (wave.json etc.); fallback artifact globs missed them | Fixed — every engine adapter registers every consumed input (config JSON, data files, manifest); `_artifact_files` drops the legacy glob fallback and an engine registering nothing is an explicit engine-contract error. Parameterized mutation test: mutating each domain's config JSON fails verify_manifest |
| GROM-001 | P1 | context.py, oracle/validate.py, _util.py (REGRESSION from IO-001) | `_find_unique` treated all topology-capable formats as mutually exclusive, rejecting a normal conf.gro + topol.tpr + traj.xtc directory | Fixed — MD inputs are modeled as roles: structure (.gro/.pdb), run topology (.tpr), alternate topology (.prmtop/.psf), trajectory; only multiple candidates for the SAME role are ambiguous. The trajectory topology uses the documented precedence structure > tpr > prmtop/psf (same selection in the oracle's `_md_metrics`). conf.gro + topol.tpr + traj.xtc loads deterministically and hashes both files; the two-structures ambiguity test moved from "ambiguous topology" to "ambiguous structure" (deliberate test update); the three MDAnalysisTests-gated tests that the regression broke now pass |
| FEP-002 | P1 | oracle/validate.py, references/fep_synthetic.json, tests/test_fep.py (REGRESSION from ORA-001) | `_fep_metrics()` returned deltaG/overlap_min_eig while the golden required deltaG_kT/overlap_min_eigenvalue/uncertainty_kT — validate("fep_synthetic") failed on all of them; the old test masked it by renaming keys before compare_metrics | Fixed — `_fep_metrics` emits the canonical names (deltaG_kT, uncertainty_kT, overlap_min_eigenvalue; the fep.py detail-dict names every other path already uses); the retired benzene golden's nonstandard names are gone with it. The test now goes through validate() on the committed fixture. The golden was also regenerated from the committed fixture (it was recorded from synthetic_u_nk(seed=42, n=20000) but the shipped dhdl.csv is synthetic_u_nk(seed=7, n=500) — a scenario mismatch ORA-003 now makes visible) with tolerances recalibrated to that fixture's sampling noise (deltaG abs 0.12 ~ 5 sigma, uncertainty abs 0.05) |
| FF-001 | P1 | io.py, context.py, pipeline.py | load_atom_types() caught every exception returning [], so the engine saw None and ff_coverage silently never ran | Fixed — [] only for the typed not-available case (an IOError naming the tpr/gromacs, how MDAnalysis signals unsupported TPR versions); unexpected parse errors propagate, are recorded as ff_load_error on the context, and become a FAILING ff_coverage diagnostic whenever the check is applicable (ff_atom_types.txt present). Without an ff list the check stays not-applicable. Stub tests for both branches + the forced-parser-exception diagnosis test |
| ORCH-004 | P1 | orchestrate.py, cli.py | subprocess.run without timeout — a hung producer hung the whole grid | Fixed — per-cell `cell_timeout_s` (default 1800 s, CLI `--cell-timeout`): expiry kills the child and records an `_error` row for that cell; the grid continues and there are NO implicit retries (documented in run_grid/generate_run). [valid, hung, valid] with a 5 s timeout yields 3 rows, the hung row carries the timeout error, the third cell executes and verifies; non-finite/non-positive timeouts are rejected |

Golden/reference files touched in batch 2, and why:

- `references/benzene_hydration_fep.json` — RETIRED (FEP-001); the
  alchemtest fixture's real MBAR overlap (7.4e-4 / 1.4e-8) contradicts the
  >= 0.05 invariant.
- `references/lysozyme_openmm.json` — `energy_relative_range` removed
  (IO-002: the committed energy.xvg carries only a `Potential` column —
  no positional fallback, so the metric is not computable from this
  fixture) and metrics regenerated from the committed fixture under
  ORA-003 (identity pins the input bytes; the previous values came from
  an older MDAnalysis DCD reader).
- `references/h2_rhf.json` — `converged: 1.0` (exact) added (PYS-001).
- `references/fep_synthetic.json` — overlap tolerance re-encoded as
  `["min", 0.05]` (FEP-001); metrics + tolerances regenerated from the
  committed fixture (FEP-002: the golden had been recorded from
  synthetic_u_nk(seed=42, n=20000) while the shipped dhdl.csv is
  seed=7/n=500; tolerances recalibrated to that fixture's sampling noise).
- all 14 references — `identity` block added (ORA-003): config/data-file
  sha256s from each case's fixture (`adk_morph` from the MDAnalysisTests
  datafiles; `lysozyme_nvt_30ps` from the canonical conf.gro/traj.xtc/
  energy.xvg of pipeline/runs/lysozyme — note that raw dir also carries
  duplicate nvt.xtc/nvt.tpr copies, so validate() on it correctly fails
  IO-001 ambiguity; validate against a copy with the duplicates removed).

Deliberate test-behavior updates (each locked in the old behavior):

- `tests/test_manifest.py::test_optional_dependency_absence_is_explicit_skip`
  → split into skip-without-invoking and ImportError-fails (PIPE-002).
- `tests/test_manifest.py` PIPE-001 error tests declare capabilities
  present via the new allowlist (PIPE-002).
- `tests/test_manifest.py::test_two_topologies_rejected` — "ambiguous
  topology" → "ambiguous structure" (GROM-001 role semantics).
- `tests/test_ontos.py::test_module_cli_verifies_and_rejects` — now runs
  with `--metadata` (ONT-007 contract requirement).
- `tests/test_reference_rules.py` — shipped-case count 15 → 14 (benzene
  retirement) and every golden must carry identity (ORA-003).
- `tests/test_pyscf_eng.py::test_h2_reference_case_exists_and_matches_fresh_run`
  — used the pre-ORA-001 metric key `final_energy`; fixed to
  `final_energy_hartree` (stale since the §F rename, only visible with
  pyscf installed).


---

## H. Verification-oracle audit register, batch 3 (2026-09-12)

Findings verified against HEAD `74e8ee6` by an external review; implemented
in the commits below. Same posture as §F/§G: simval is the trust anchor,
every fix errs toward failing closed. The six ONT findings were producer
bugs mirrored into the reference: the corrected semantics live in the
ontos clone at `usestemframework/ontos` befff49 (OTO-002/003/008/009/012/
013/014 there), and the reference plus its corpus now track those.

| ID | Severity | Area | Finding | Status |
|---|---|---|---|---|
| ONT-010 | P0 | ontos_gravity.expected_zoom_policy | Promotion arm fired for any non-Fine mode (`modes != 0`), so the observer zoom policy expanded a Collapsed region | Fixed — promotion gates on Coarse only (`modes == 1`); a collapsed region leaves collapse solely via an explicit expansion event. Corpus resync below |
| ONT-011 | P0 | ontos_gravity.GravityWorld | Refit detection scanned surviving member Fits for `entering == t0 + WINDOW`: an empty demotion armed nothing (region stayed Coarse forever) and foreign-window absorption (section 26 collapse consuming the donor's last Fit) stalled the donor's re-fit | Fixed — per-region `region_window_deadline` armed on every demotion (empty ones included) and on every re-fit that keeps members; consumed at refit execution; cleared on promote, collapse and expansion (mirrors ontos 555cecd/d8aa3e1) |
| ONT-012 | P0 | ontos_gravity.GravityWorld._demote | Re-demotion reselected membership without touching the region's existing Fits, so a stale out-of-box member Fit survived past its own validity window | Fixed — every Fit owned by the region is materialized into bodies and cleared at t0 before membership is recomputed (collapsed bodies excluded, the section 14 rule) and the new window fitted (mirrors ontos 7546061) |
| ONT-013 | P0 | ontos_gravity._contact_pass | Touching-based record suppression consulted the initially-empty touching set from pass 1, so the run's first trajectory-changing impulse could be silent | Fixed — `contact_armed` latch: False initially, suppression consulted only when armed, armed after a pass emits at least one Contact (mirrors ontos e733200). Seed-416 reproducer locked in as a test |
| ONT-014 | P0 | ontos_gravity.step | Level transitions applied without invalidating touching — fresh-contact-on-return-to-Fine suppressed, including same-boundary Fine->non-Fine->Fine (demote+promote, expand+recollapse) | Fixed — changed members accumulate after EACH individual transition (scheduled events and window refits), their touching entries drop before the contact pass, and a collapsed region's monopole pseudo-ids drop whenever the region leaves collapse during the boundary (mirrors ontos 5a45877/13d7d1d) |
| ONT-015 | P0 | ontos_gravity._contact_pass | Wall guards combined overlap && approach and bailed before inserting into the next touching set, so a receding-but-overlapping body dropped its wall key and re-approach emitted a duplicate contact beginning | Fixed — the wall pair enters the next touching set on ANY overlap; the approach test gates only the impulse/record (mirrors ontos 38cf8a3, spec section 24) |
| GOLD-002 | P0 | oracle/cases.py, references/MANIFEST.json (new), scripts/ | Reference JSONs trusted as-is; `source_hash` lived inside the same mutable file | Fixed — `references/MANIFEST.json` pins each golden's canonical-content sha256 (sorted keys, compact separators: layout edits inert, content edits detected); `_load()`/`load_all()` verify before use and fail closed naming the mismatched case; an unpinned golden inside references/ is rejected; a pinned name is verified wherever the file lives. Trust model documented honestly in `_load_manifest`: the manifest is itself the pinned artifact — tampering requires editing two coordinated files in one commit. `scripts/regen_reference_manifest.py` regenerates it deliberately (refuses name/stem mismatches and duplicate names) |
| IO-003 | P0 | io.py, context.py, pipeline.py, oracle/validate.py | Non-numeric xvg data rows silently discarded; a ragged numeric table raised ValueError which GromacsEngine labeled energy-check-not-applicable | Fixed — malformed non-directive data raises `XvgParseError` with line/column context, ragged rows name both lines; the genuinely-missing-label case raises the narrow typed `ConservedEnergyColumnMissing` — the only path that may become an explicit skip. Parse/shape failures surface as a failing `energy_drift` diagnostic (FF-001 pattern) and the oracle metric path no longer swallows them |

Golden/reference/corpus files touched in batch 3, and why:

- `references/MANIFEST.json` — NEW (GOLD-002): canonical-content pins for
  all 14 shipped goldens, generated by `scripts/regen_reference_manifest.py`.
- `examples/ontos_gravity/collapse_observer/{ontos.stream,ontos.json,provenance.json}`
  — regenerated from the corrected ontos clone at befff49 (ONT-010): the
  seed-13/observer-42/collapse-at-40 run no longer carries the wrong
  tick-49 promotion of collapsed region 0. The regenerated stream carries
  the section 20 RegionMultipole record (ontos CLI default), so the
  example metadata moves to `multipole: true` and it leaves the legacy
  pre-spec-20 set (`collapse`, seed 11, remains). No other corpus stream
  changed: all six corrected behaviors are byte-identical on every other
  committed corpus run (the emitter-reproduction tests pin this), matching
  the ontos-side golden analysis.
- `tests/test_ontos_gravity.py::test_legacy_section19_examples_still_verify`
  — collapse_observer removed from the legacy pair (deliberate; it is now
  a section 20 corpus run).
- `tests/test_reference_rules.py` version-gate and identity fixtures —
  renamed their synthetic cases to probe names so the GOLD-002 pin does
  not shadow the validation under test (deliberate test update).

---

## I. Verification-oracle audit register, batch 4 (2026-09-12)

Findings verified against HEAD `f636c8c` by an external review; implemented
in the commits below. Same posture as §F/§G/§H: simval is the trust anchor,
every fix errs toward failing closed. Both findings are producer bugs
mirrored into the reference: the corrected semantics live in the ontos
clone at `usestemframework/ontos` 89afa2e (OTO-016) and edff03b (OTO-017,
HEAD aa305ee), and the reference now tracks them. No existing corpus
exercises the fixed orientations (all cross-checks were already
bit-matching), so no golden/corpus bytes changed — the emitter-reproduction
tests pin this.

| ID | Severity | Area | Finding | Status |
|---|---|---|---|---|
| ONT-016 | P0 | ontos_gravity._contact_pass | The pair sweep skipped any non-fine outer i and only considered j > i, so a fine x ephemeris-coarse static pair with the COARSE body at the smaller id never fired its one-sided impulse — the oracle would have blessed the wrong producer behavior once new corpus cases exercised that orientation | Fixed — the sweep visits every unordered real-body pair once in pinned (i, j) order and dispatches on membership: (fine, fine) two-sided, (fine, coarse) static on i, and — with the section 24 record — (coarse, fine) a one-sided impulse on the fine body j against i's frozen polynomial state, normal pointing fine->coarse (mirrors ontos 89afa2e exactly, including the ungated (0,1) arm; the §21-only contacts+demote CI case cross-verifies bit-exact). Touching keys and Contact record ids stay (min, max) for real-body pairs per STREAM_SPEC §24; static_pair marks every one-sided pair. The coarsehit test-IC docstring drops its fine-low rationale (it was hiding the bug, not a spec choice; the reversed id order is now pinned by the mirrored seed-117 regression: record ids (1, 3), fit frozen vs a no-contact control, body 3 changed by exactly s*n, ledger booking exactly m*s*n) |
| ONT-017 | P0 | ontos_gravity._shell_scale | The global radial scale was applied BEFORE classifying shell membership; §25 pins classify-unscaled-base-then-scale. In exact arithmetic rank survives a uniform scale, but binary64 rounding on near-ties can flip shell membership (changing mu_k solves, positions, hashes, stream bytes) | Fixed — mass-weighted mean, radii, and shell_assignment are computed from the UNSCALED section 20 base and retained; lambda is then solved/applied and the per-shell pairs gathered from the scaled displacements per step 4 (mirrors ontos edff03b). The mirrored near-tie regression pins an exact sqrt(185) radius tie that the solved lambda breaks the wrong way (asserted to fail under the old order) plus per-shell closure on the record targets |

Verification: full pytest suite (392 passed), `simval diagnose` on all 20
ontos examples (PASS, provenance byte-stable), and the complete CI
cross-verification against the local ontos clone at aa305ee (23 distinct
stream contracts incl. wallshot/coarsehit test-ICs + 3 modal-audio cases)
— every stream and WAV bit-exact, zero mismatches.

---

## J. Verification-oracle audit register, batch 5 (2026-09-12)

Findings verified against HEAD `6ed8f28` by an external review; implemented
in the commit below. Same posture as §F–§I: simval is the trust anchor,
every fix errs toward failing closed. This finding is the residual of the
batch 4 ONT-016 mirror: the corrected semantics live in the ontos clone at
`usestemframework/ontos` a0b59c4 (OTO-019 there, HEAD 1792cae), and the
reference now tracks them. No committed corpus case exercises the ungated
orientation with a bare-contacts stream, so no golden/corpus bytes changed
— the emitter-reproduction tests pin this.

| ID | Severity | Area | Finding | Status |
|---|---|---|---|---|
| ONT-019 | P0 | ontos_gravity._contact_pass | The ONT-016 sweep mirror admitted (fine i, coarse j) pairs unconditionally while gating only (coarse i, fine j) on the section 24 record — pre-ONT-016 both orientations required it. Sections 21/24 pin static real-body contact on ContactParams record presence regardless of id orientation, so the oracle would have blessed static contacts in a bare `--contacts` stream (no tag 12) whenever the coarse body carried the larger id | Fixed — the dispatch mirrors ontos a0b59c4 exactly: `(fine, fine)` unconditional, `(fine, coarse) \| (coarse, fine)` only with the record (`extended`). The mirrored coarsehit pair locks both halves: bare contacts + early demote is bit-identical to contacts off (zero records, zero static impulses, empty touching set, bit-equal body states), while an explicit `--restitution 0` — record present, zero-valued — fires the fine x coarse statics with e = 0 closure (the bare-contacts half fails on the ungated code; the corpus emitter-reproduction tests confirm no committed stream changed) |

Verification: full pytest suite (452 passed, 4 skipped, 1 network test
deselected; optional-engine deps installed), `simval diagnose` on all 20
ontos examples (PASS, provenance byte-stable), and the complete CI
cross-verification against the local ontos clone at 1792cae (25 stream
contracts incl. wallshot/coarsehit test-ICs + 3 modal-audio cases) —
every stream and WAV bit-exact, zero mismatches.
