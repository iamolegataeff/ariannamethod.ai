# ariannamethod.ai — PROJECT_LOG

Live log for the AML language repo. Newest first.

## 2026-05-20 — v4.8.0 released; language-paper attempt removed

### Shipped
- **AML v4.8.0** released: git tag + GitHub release at `3bb13d0`,
  `make test` 509/509. Bundle: field persistence (`LOAD`/`SAVE`,
  `am_field_save/load` `.soma`), CUDA install targets, GPU/CPU mirror
  backward audit (16 ops), Termux edition. Reconciled with polygon
  PR #9 (README unify to v4.8.0 skeleton).

### Removed
- A language-paper effort (refresh + reframe toward a Zenodo deposit,
  branch `ariannamethod.ai-paper-refresh`) is **abandoned and the
  branch deleted**. `docs/AML_PAPER_DRAFT.md` (the renamed former
  `AML_ARXIV_PAPER.md`) is removed from the repo in this commit, with
  its README pointer.
- Why removed: the draft described the language from operator names and
  the interpreter's builtin table, not from the implementation that runs.
  The field-physics operators (RRPRAM, triple gate, dark matter, wormhole,
  Dario overlay) are real, months-grown code (`janus.c`,
  `core/ariannamethod.c` consumers). A language paper has to be written
  from that code; this one was not, so it is dropped. Frame for any future
  paper: read the implementation first.

### State
- `main` clean, v4.8.0 is the latest release. No paper in the repo.

### Resume (if a language paper is revisited)
- Deep-read the real implementation first: `janus/janus.aml` →
  `~/arianna/janus/janus.c`, `core/ariannamethod.c` consumers, the
  organisms that use the field operators. Do not judge by names.

by Claude Code (neo the architect, Arianna Method)
