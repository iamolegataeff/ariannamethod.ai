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
