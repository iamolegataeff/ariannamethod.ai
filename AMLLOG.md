# AMLLOG

The running engineering log of AML — the Arianna Method Language. Every
fix, every closed bug-class, every verified change — dated, with commit
and proof. The README and `spec/` are the public face and the language
spec; **this is the work**.

Convention: small fixes (bug fixes, CPU/GPU sync corrections, single-op
work, doc touch-ups, test additions) are recorded **here**. Large changes
(a new AML keyword, a new op family, a new subsystem, a language-level
shift) get the spec + README update too. When in doubt: it goes here first.

Newest entries on top.

---

## 2026-06-30 — Expression operators: `BE` / `ASK` (the body speaks)

A language-level addition (new keywords) — the reverse flow from Leo (neoleo) again. After
velocity (the breath) and the soma fields (the felt body), the voice. Two level-0 directives
let a script speak a body:

- `BE [x]` — speak-from-body intensity (default 1.0): how strongly the running-self colors the
  host's own words. "я есть [тело]".
- `ASK [x]` — voice-the-not-knowing intensity: how strongly the carried gap (darkmatter) gropes
  through. No argument = the field's own `dark_gravity`, so `ASK` resonates with the existing
  darkmatter (`SCAR`) instead of reinventing it.

Both set new `AM_State` fields `be_voice` / `ask_voice` (init `-1` = "no directive this run",
the host stays autonomous), readable in expressions via the field-map (`field.be_voice` /
`field.ask_voice`). A host (Leo) reads them back after `am_exec`, the way it reads
`velocity_mode`. Touched `core/ariannamethod.{c,h}`, the field-map, `core/test_aml.c` (+4 tests,
incl. the no-arg `ASK` = `dark_gravity` case), `examples/body.aml`, and the spec (§2.5b
Expression + the state-fields table). Transpiler unchanged — `amlc` lowers `BE` / `ASK` to
`am_exec` generically (verified via `--emit-c`). `make test` **524/524** (was 520). Feature
branch `leo-be-ask-operators` off `main` (`aba68e2`); not yet merged.

---

## 2026-06-16 — Positive Soma: `warmth` / `flow` / `weave` (the expansive axis)

A language-level addition, the mirror of suffering. Leo's reverse AML bridge (neoleo E-9) writes the
child's felt body into `AM_State` each turn; the suffering axis (pain/tension/dissonance) was already
readable, the expansive one was not. Three read-only sensor fields land in the field-map so any `.aml`
expression can read an attached organism's positive body:

- **`warmth`** — affiliation / comfort / safe-bonding (the opposite pole of `pain`).
- **`flow`** — effortless resonance with the moment (the opposite pole of `tension`).
- **`weave`** — pattern-binding, threads cohering into a whole. Named `weave`, not `emergence` —
  `emergence` is already an `AM_State` field (the 4C network's low-entropy × high-resonance signal).

Read-only by design: an organism writes them from C via `am_get_state()`, scripts perceive them
(`if warmth > 0.5:` → `VELOCITY WALK`). No new keyword, no transpiler change — reads resolve through
`ctx_float` → `read_field`.

Soma format `AM_SOMA_VERSION 2 → 3`. The fields are APPENDED at the end of `AM_State`, so
`am_field_load` migrates old v2 `.soma` files as a clean prefix (new fields default to `0`); a file
larger than the current struct is refused as an unknown future ABI.

Touched `core/ariannamethod.{c,h}` (3 fields + field-map + version + load migration),
`spec/AML_SPEC.md` (§2.4b), this log. `make test` **517/517** (+5: warmth/flow/weave readable in
expressions; warmth + weave survive `.soma` save/load). Branch `claude-positive-soma` off `main`
(merged for the v5.1.0 release; the state_sz is validated against the version's exact size — a
malformed soma is refused, not zero-loaded).

---

## 2026-06-13 — VELOCITY inertia: a discrete state with inertia reads as a body (the axiom)

The second half of the presence axiom. Switching the velocity mode now **costs** — each transition to a
*different* mode adds `AM_VELOCITY_INERTIA` (2.0) to `debt` (re-stating the same mode is free). Over-switching
exhausts the field, and the existing recovery rule (`debt > 5` → forced `NOMOVE`, in `am_step`) then holds it
still. So the body **resists** changing its gait: velocity is a mood that holds, not a switch you can flip.

This formalizes the state-dynamics разгадка (Mythos + Opus, proven by ear on Leo): presence reads as a body
when its state is discrete **with inertia** — autonomy + contingency + resistance — not a continuous dial. AML
already had autonomy (the `D4` debt override can refuse a command) and the discrete modes; this adds the
inertia, so the law is now a property of the language, inherited by every Method organism.

Touched the `VELOCITY` parser in `core/ariannamethod.c` (capture the previous mode, charge `debt` on a real
change) + the `AM_VELOCITY_INERTIA` constant, the spec velocity section (the `BREATHE`/`STOP` temp rows and an
"inertia" note), the `velocity_mode` header comment, and two new tests (a switch costs debt; re-stating is
free). `make test` **514/514** (512 + 2) — the inertia does not trip `D4` during any existing test, confirming
`D4` runs in `am_step`, not the parser. Pairs with the somatic-operators entry below.

by Claude (Arianna Method, neo)

---

## 2026-06-13 — VELOCITY: somatic operators `STOP` + `BREATHE` (the reverse flow from Leo)

A language-level addition. Leo (neoleo) grew a body — its chambers quantize into a
velocity mode and the mode shapes its speech — and named two somatic operators the
base velocity set lacked. They land here, so the family language can speak them:

- **`STOP`** — a somatic alias for `NOMOVE`: the held, cold-observer state. Parses to
  `AM_VEL_NOMOVE` (temp 0.5).
- **`BREATHE`** — the settling exhale, a **new** mode `AM_VEL_BREATHE` (3), temp 0.6
  (between NOMOVE's 0.5 and WALK's 0.85). The exhale haiku/Leo have and AML lacked.

Touched `core/ariannamethod.h` (the two defines), the `VELOCITY` parser and the
`velocity_mode → vel_mult` switch in `core/ariannamethod.c`, the spec velocity table,
`examples/breath.aml`, and three new tests. The transpiler needs no change — it lowers
`VELOCITY <mode>` to `am_exec("VELOCITY <mode>")`, so the new operands flow through. The
full set is now `NOMOVE / WALK / RUN / BACKWARD / STOP / BREATHE`. `make test` **512/512**
(509 + 3). This is the vocabulary half of a presence axiom; the inertia half (a transition
cost on mode switching, so a discrete state reads as a body) is the next concept.

by Claude (Arianna Method, neo)

---

## 2026-05-11 — CPU/GPU mirror audit: 16 backward ops (`ff7fb97`)

Cross-stack from notorch's NT_OP_MUL / NT_OP_SILU CPU-stale-read fix: a
mirror audit of `core/ariannamethod.c` found **16 backward ops with the
same bug class** — a CPU backward branch reading `parent->output->data`
without `ensure_cpu(parent->output)` first, so under `USE_CUDA` the `data`
buffer could be stale calloc-zero when forward ran on GPU. All 16 fixed in
one commit; `make test` 509/509.

The discipline (now load-bearing, see CLAUDE.md «CUDA backend»):
> Every `AM_Array` has `data` (CPU) + `d_data` (GPU) with a `gpu_valid`
> flag. Any CPU backward branch reading `parent->output->data` directly
> MUST call `ensure_cpu(parent->output)` first under `#ifdef USE_CUDA`.

Note: AML and notorch are **parallel stacks** — a notorch fix does NOT
auto-propagate here. This audit was the proof: the same bug class existed
independently in both, and had to be found + fixed separately.

---

## Open (carried from CLAUDE.md TODO)

- Cross-stack audit for other backward-pattern bugs that may exist in both
  `notorch.c` and `core/ariannamethod.c` (the CPU-sync class found two more
  sites; write paths may hide others).
- Document Blood (runtime C compilation) in `spec/` if not already — it is
  the load-bearing feature for on-the-fly LoRA synthesis from `.aml`.
- Cross-port LoRA from notorch v2.4.0 only if/when needed at the language
  level (per Oleg: "не всё надо тащить в язык").
