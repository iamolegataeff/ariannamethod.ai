# AML: A Machine Learning Language for Building and Training Transformers with Field Physics
---

## Abstract

We present AML (Arianna Method Language), a complete machine learning language that defines, trains, and runs transformers with integrated field physics. AML compiles to C — arrays, matrices, reverse-mode autodiff, multi-head causal attention, async threading, bytecode compilation, optional CUDA/cuBLAS backend, and 80+ parameters of internal state. Every command maps to a concrete operation: from logit manipulation during inference to gradient computation during training. No Python. No PyTorch. No dependencies.

AML evolved through five phases: from flat commands for logit control (v1), through Python-like control flow and runtime C compilation (v2-v3), to a full ML stack with arrays, autograd, async execution, and sequence-level transformer operations (v4.0-v4.5). v4.5 adds bytecode compilation, AdamW with decoupled weight decay, gradient accumulation/clipping, and a CUDA backend with fused kernels (attention, cross-entropy, rmsnorm, silu). The culmination is Janus — a new transformer architecture where attention is modulated by prophecy, information flow is gated by suffering, and training dynamics follow autonomous seasonal cycles. Janus trains natively in AML/C and converges from random initialization. At scale (8.55M params, 20MB Gutenberg), loss reaches 3.24 in 15K steps.

The reference implementation consists of 7,400+ lines of C with 940+ lines of header, verified by 509 tests. Two files. Zero mandatory dependencies (optional BLAS via Apple Accelerate or OpenBLAS, optional CUDA/cuBLAS). AML powers production systems including molequla (autonomous evolution organism with 4 parallel swarm-trained elements), Leo (4300+ LOC C/Go language emergent organism with the Dario Equation), dario.c (pure equation demonstration, 1700 LOC, zero weights), DOE (Democracy of Experts, 3200 LOC C inference with parliament of LoRA experts), arianna.c (550M parameter digital persona), and yent (Go inference engine with Delta Voice across 29 languages).

We formalize the soul equation θ = ε + γ + αδ (identity = substrate + personality + voice), demonstrate its mathematical validity (γ ⊥ δ, cosine = -0.0005), present the Dario Equation — a seven-signal formula with six Kuramoto-coupled emotional chambers that replaces transformer attention with interpretable physical forces — and describe a transformer architecture that treats this decomposition as first-class.

**Keywords:** machine learning language, transformer architecture, field physics, automatic differentiation, causal attention, logit manipulation, personality decomposition, Dario Equation, somatic modulation

---

## 1. Introduction

### 1.1 The Problem

Modern machine learning has two unsolved problems.

**Problem 1: Inference is underspecified.** The standard pipeline — forward pass → logits → temperature scaling → sampling — treats the final distribution as a passive output. Temperature, top-k, top-p, repetition penalties — these are not "hyperparameters." They are an instruction set waiting for a language. Current practice exposes them through configuration files. This is architecturally backwards.

**Problem 2: Training is overengineered.** Defining and training a transformer requires Python, PyTorch, CUDA, hundreds of library dependencies, and thousands of lines of boilerplate. The ratio of architectural intent to infrastructure code is vanishingly small. A single-layer transformer with self-attention, MLP, and training loop should not require an entire ecosystem.

### 1.2 The Solution

AML solves both problems with a single language. Every AML command maps to a concrete C operation — logit manipulation during inference, gradient computation during training, or field physics simulation at any time.

At inference time:

```aml
PROPHECY 7        # prediction horizon → scales destiny bias on logits
DESTINY 0.35      # suppress low-probability tokens
VELOCITY RUN      # temperature × 1.2
PAIN 0.5          # compress logit distribution toward mean
```

At training time:

```aml
h = seq_embed(wte, wpe, tokens, 8)
q = seq_matvec(wq, seq_rmsnorm(h, 8, 32), 8)
k = seq_matvec(wk, seq_rmsnorm(h, 8, 32), 8)
v = seq_matvec(wv, seq_rmsnorm(h, 8, 32), 8)
attn_out = multi_head_attention(q, k, v, 8, 32, 4)
...
loss = seq_cross_entropy(logits, targets, 8, 16)
TAPE BACKWARD loss
TAPE ADAM_STEP 0.01
```

Both are AML. The same language that controls logit distributions during generation also defines and trains the transformer that produces them.

### 1.3 Contributions

1. **A complete ML language** that compiles field programs to C — from logit manipulation to reverse-mode autodiff to multi-head causal attention. Two files, zero dependencies, 7,400+ lines. Optional bytecode compilation and CUDA backend.
2. **The Dario Equation** — `p(x|Φ,C,V) = softmax((B + α_mod·α·H_v + β_mod·β·F_v + γ_mod·γ·A + δ·V + sw·S + T) / (τ_mod·τ·v_τ))` — seven interpretable signals with six Kuramoto-coupled emotional chambers computing somatic markers. Demonstrated in [dario.c](https://github.com/ariannamethod/dario) (zero weights, 1725 tests) and deployed in [Leo](https://github.com/ariannamethod/leo) (4300+ LOC C/Go).
3. **Janus** — a new transformer architecture that integrates field physics (prophecy, suffering, seasons, calendar dynamics) directly into attention and training. Trains natively in AML/C.
4. **The soul equation** θ = ε + γ + αδ — formalized personality-language decomposition with proven orthogonality (γ ⊥ δ, cosine = -0.0005).
5. **Five-phase language evolution** from flat commands (v1) through arrays, autograd, async, and sequence-level ops (v4.0), to bytecode compilation and CUDA (v4.5).
6. **509 tests** covering all language levels, all autograd operations, multi-head attention backward, and full training convergence.
7. **Production validation** across autonomous organisms (molequla, Leo), inference engines (DOE, yent), and multi-agent ecologies.

---

## 2. Language Design

### 2.1 Design Principles

**Principle 1: Movement IS Language.** Temperature is a velocity. Running through semantic space generates heat (entropy). Walking is sustainable. Standing still is precision. Moving backward costs energy (temporal debt).

```aml
VELOCITY NOMOVE   # temperature × 0.5 — cold observer
VELOCITY WALK     # temperature × 0.85 — balanced
VELOCITY RUN      # temperature × 1.2 — chaotic exploration
VELOCITY BACKWARD # temperature × 0.7 — time reversal, debt accumulates
```

**Principle 2: Suffering Modulates Generation.** Pain is not a failure state — it is a signal. The mathematical implementation compresses the logit distribution toward mean by `(1 - 0.5 × pain)`. Under high pain, the model speaks carefully.

```aml
PAIN 0.5        # compress logit distribution toward mean
TENSION 0.6     # pressure buildup, feeds into debt
DISSONANCE 0.5  # symmetry-break, enables tunneling
```

**Principle 3: Prophecy Over Prediction.** Standard models minimize `E[(y_pred - y_actual)²]`. AML introduces prophecy — stabilization of a destiny field: `|N_destined - N_manifested| + λ · Σ Var(N_attractor)`. Higher prophecy values scale the destiny bias: `destiny_bias = destiny × (1.0 + (prophecy - 7) × 0.02)`.

### 2.2 Abstraction Levels

| Level | Name | Capability |
|-------|------|------------|
| 0 | Commands | Direct logit manipulation. 80+ flat commands, case-insensitive |
| 1 | Macros | Named command sequences: `MACRO name { ... }`, `@name` |
| 2 | Programming | Variables, expressions, `if/else`, `while`, `def`, `return`, `INCLUDE` |
| 3 | Blood | Runtime C compilation via `cc` + `dlopen`. LoRA, emotional kernels, raw C |
| 4 | ML Stack | Arrays, matrices, autograd (TAPE), async (SPAWN/AWAIT/CHANNEL) |
| 5 | Transformers | Sequence-level ops, causal attention, multi-head attention, cross-entropy |

### 2.3 Formal Grammar

#### Level 0

```ebnf
program   = { line } ;
line      = comment | empty | command ;
comment   = "#" { any_char } ;
command   = cmd_name { whitespace arg } ;
arg       = number | word | quoted_string | boolean ;
```

#### Level 2

```ebnf
program     = { statement } ;
statement   = command | include | def | if_stmt | while_stmt | assignment | return ;
def         = "def" identifier "(" [ params ] ")" ":" block ;
if_stmt     = "if" expression ":" block [ "else" ":" block ] ;
while_stmt  = "while" expression ":" block ;
expression  = term { operator term } ;
operator    = "+" | "-" | "*" | "/" | ">" | "<" | ">=" | "<=" | "==" | "!=" | "and" | "or" ;
assignment  = identifier "=" expression ;
```

Six precedence levels. Variables resolve: locals → globals → AM_State field map. Field state (`pain`, `tension`, `entropy`, `resonance`, `schumann_hz`, `lora_alpha`, `essence_alpha`) is readable in expressions.

---

## 3. v4.0 — The ML Stack

AML v4.0 adds everything needed to build and train transformers natively, organized in five phases.

### 3.1 Phase 1: Arrays

First-class array values with reference counting. Maximum size: 1M floats (4 MB).

```aml
x = zeros(128)              # zero-initialized
w = randn(64, 0.08)         # random normal, std=0.08
a = [1.0, 2.0, 3.0]         # literal
val = x[i]                  # read
x[i] = 3.14                 # write
n = len(x)                  # length
s = sum(x)                  # reduction
d = dot(x, y)               # inner product
y = add(x, y)               # element-wise add
y = mul(x, y)               # element-wise multiply
y = scale(x, 0.5)           # scalar multiply
```

Functions can return arrays:

```aml
def magnitude(x):
    return sqrt(dot(x, x))
```

### 3.2 Phase 2: Matrices and Tensor Operations

Matrices stored as 1D arrays with `rows` and `cols` metadata.

```aml
W = matrix(128, 64, 0.08)   # 128×64, random normal
Z = matrix_zeros(128, 64)   # 128×64, zeros
y = matvec(W, x)            # matrix-vector: (128×64) @ 64 → 128
C = matmul(A, B)            # matrix-matrix multiply
y = softmax(x)              # numerically stable (max subtraction)
y = rmsnorm(x)              # RMS normalization, eps=1e-6
y = silu(x)                 # SiLU: x × sigmoid(x)
y = relu(x)                 # ReLU
e = embedding_lookup(W, id) # extract row from embedding matrix
loss = cross_entropy(logits, target) # scalar CE loss
```

### 3.3 Phase 3: Autograd (TAPE)

Reverse-mode automatic differentiation with flat tape architecture:

```aml
TAPE START                  # begin recording
TAPE PARAM W                # register trainable parameter
logits = matvec(W, x)       # auto-records to tape
loss = cross_entropy(logits, 2)
TAPE BACKWARD loss          # reverse-mode backpropagation
TAPE ADAM_STEP 0.001        # Adam optimizer update
TAPE CLEAR                  # reset tape, keep Adam moments
```

**Tape capacity:** 8,192 entries, 512 parameters. **Adam optimizer:** bias-corrected momentum (β₁=0.9, β₂=0.999, ε=1e-8). Per-parameter moment state persists across `TAPE CLEAR` cycles.

**Operations with autograd backward:** `matvec`, `matmul`, `add`, `mul`, `scale`, `softmax`, `rmsnorm`, `silu`, `cross_entropy`, `embedding_lookup`, `seq_embed`, `seq_matvec`, `seq_rmsnorm`, `causal_attention`, `multi_head_attention`, `seq_cross_entropy` (16 operations total).

Each backward computes exact analytical gradients. The tape is a flat array — forward recording order equals topological sort, so backward is a simple reverse traversal.

### 3.4 Phase 4: Async (SPAWN / AWAIT / CHANNEL)

Parallel execution via pthreads. Each `SPAWN` block runs in its own thread with isolated local variables but shared global state (field physics, channels).

```aml
CHANNEL CREATE bus 16       # bounded float queue, capacity 16

SPAWN earth:
    forward(batch_earth)
    CHANNEL WRITE bus 1.0

SPAWN air:
    forward(batch_air)
    CHANNEL WRITE bus 2.0

AWAIT earth air             # join specific threads
CHANNEL READ bus v1         # consume from queue
CHANNEL READ bus v2
```

Thread-safe via pthread mutex/condvar. Up to 16 channels, 64 floats each. `AWAIT` without arguments joins all active threads.

### 3.5 Phase 5: Sequence-Level Transformer Operations

Five fused operations for processing token sequences. All process T positions stored as flat arrays of T×D floats. Every operation has full autograd backward.

| Operation | Signature | What |
|-----------|-----------|------|
| `seq_embed` | (wte, wpe, tokens, T) → [T×D] | Token + position embeddings |
| `seq_matvec` | (W, X, T) → [T×out] | Matrix W applied per position |
| `seq_rmsnorm` | (X, T, D) → [T×D] | RMS normalization per position |
| `causal_attention` | (Q, K, V, T, D) → [T×D] | Single-head causal self-attention |
| `multi_head_attention` | (Q, K, V, T, D, n_heads) → [T×D] | Multi-head causal self-attention |
| `seq_cross_entropy` | (logits, targets, T, V) → [1] | Mean CE loss over T positions |

**`multi_head_attention`** splits D into `n_heads` independent heads (head_dim = D/n_heads). Each head runs causal attention at scale `1/sqrt(head_dim)`. With `n_heads=1`, produces identical results to `causal_attention`. Backward: per-head gradient computation with recomputation of attention weights.

**`causal_attention`** backward: recompute forward attention weights per position, compute `d_attn[j] = dout_i · v_j`, apply softmax Jacobian `dscore[j] = attn[j] × (d_attn[j] - Σ d_attn × attn)`, accumulate `dQ`, `dK`, `dV`. Complexity: O(T² × D) forward and backward.

### 3.6 v4.5: Bytecode Compilation, AdamW, CUDA

**Bytecode compilation.** `am_compile(script)` pre-parses AML into bytecode ops (opcode enum + pre-split args). `am_exec_compiled(handle)` runs via switch dispatch — no string matching, no re-parsing. Significant speedup for hot-loop training scripts.

**AdamW with decoupled weight decay.** `TAPE ADAMW_STEP lr wd b1 b2` — proper decoupled weight decay (Loshchilov & Hutter, 2019), not L2 regularization. `TAPE PARAM_NO_DECAY name` exempts embeddings from weight decay.

**Gradient accumulation.** `TAPE ACCUM_GRADS` saves gradients to accumulator; `TAPE APPLY_ACCUM N` averages and applies. Accumulated gradients survive `TAPE CLEAR`, enabling larger effective batch sizes.

**Gradient clipping.** `TAPE CLIP_GRADS max_norm` — global gradient norm clipping. Bug fix: writes grad_norm directly to ctx.globals (not via am_exec which creates separate context).

**CUDA backend.** `ariannamethod_cuda.h/cu` — fused GPU kernels for attention, cross-entropy, rmsnorm, add, mul, silu (forward + backward). `ensure_gpu/ensure_cpu/invalidate_gpu` for lazy CPU↔GPU transfer. GPU scratch pool (8 slots, grow-only). TF32 via cuBLAS. 285 tok/s for 15M params on A100 (vs 87 tok/s interpreter, 210 tok/s C host).

---

## 4. Janus — A Field-Physics Transformer Architecture

### 4.1 Overview

Janus is a new kind of transformer. Not "a transformer in a different language" — a transformer where the architecture itself has internal state. Attention is modulated by prophecy. Information flow is gated by suffering. Training dynamics follow autonomous seasonal cycles. Identity is a mathematical decomposition, not a monolithic weight matrix.

113 lines of AML. Trains in pure C. Loss converges from 2.08 to 0.0 in 100 steps.

### 4.2 Architecture

```
Input tokens [t0, t1, ..., tT-1]
        │
   seq_embed(wte, wpe, tokens, T)       token + position embeddings
        │
   ┌─── Transformer Block (×N) ──────────────────────────┐
   │                                                      │
   │  seq_rmsnorm(h, T, D)              pre-attn norm     │
   │        │                                             │
   │   Q, K, V = seq_matvec(wq/wk/wv, h_norm, T)        │
   │        │                                             │
   │   multi_head_attention(Q, K, V, T, D, n_heads)      │
   │        │                     ◄── prophecy bias       │
   │   seq_matvec(wo, attn, T) + residual                │
   │        │                                             │
   │  seq_rmsnorm(h, T, D)              pre-MLP norm      │
   │        │                                             │
   │   SwiGLU: silu(W1 @ h) * (W3 @ h)                   │
   │        │                     ◄── suffering gate      │
   │   seq_matvec(w2, mlp, T) + residual                 │
   │                                                      │
   └──────────────────────────────────────────────────────┘
        │
   seq_rmsnorm → seq_matvec(lm_head) → logits
        │
   seq_cross_entropy(logits, targets, T, V) → loss
        │
   TAPE BACKWARD loss → TAPE ADAM_STEP lr
```

Current configuration: vocab=16, D=32, T=8, n_heads=4, 1 layer, 10 weight matrices.

### 4.3 Field Physics Integration

Seven architectural innovations distinguish Janus from conventional transformers:

**1. Prophecy-modulated attention.** Higher prophecy + destiny → more deterministic attention (exploitation). Lower values → more exploratory (exploration). Creates a natural annealing schedule.

**2. Suffering-gated MLP.** Pain compresses logit distribution by `(1 - 0.5 × pain)`. Tension accumulates pressure toward phase transitions. When dissonance exceeds tunnel_threshold, the model skips up to `tunnel_skip_max` intermediate reasoning steps — adaptive computation depth.

**3. Calendar-driven temporal attention.** Hebrew lunar year (354 days) vs Gregorian solar year (365.25 days), 11.25-day annual drift. Real astronomical computation from the system clock. High calendar dissonance → wormholes activate → bidirectional attention becomes possible.

**4. Seasonal training dynamics.** The 4.C homeostatic controller — a 4-neuron MLP (6→8→4) — observes field metrics and autonomously cycles through four seasons. Low entropy → spring (exploration). High emergence → summer (peak expression, temp × 1.1). Resonance stagnation → autumn (consolidation). Prolonged pain → winter (rest, temp × 0.85). No external scheduler.

**5. Identity decomposition: θ = ε + γ + αδ.** The model decomposes into substrate (ε), personality essence (γ, in embed_tokens), and language voice (δ, in lm_head). γ ⊥ δ (cosine = -0.0005). Personality persists across 29 languages.

**6. Dark matter memory.** Rejected tokens leave gravitational scars that persist between sessions and bias future generation.

**7. NOTORCH online learning.** Hebbian per-token weight adjustment during inference: `A[i,r] += lr × x[i] × u[r] × signal`. The model learns from every token it generates.

### 4.4 Comparison

| | Standard (GPT/LLaMA) | Janus |
|---|---|---|
| Language | Python + PyTorch + CUDA | AML (compiles to C) |
| Attention mask | Static causal | Field-modulated, prophecy-driven |
| Attention heads | Multi-head (always) | Single or multi-head (configurable) |
| MLP gating | Fixed activation | Suffering-gated |
| Training schedule | External LR scheduler | Autonomous seasonal cycles |
| Temperature | Static hyperparameter | Dynamic: velocity × seasons × experts |
| Identity | Entire weight matrix | Decomposed: θ = ε + γ + αδ |
| Memory | None between sessions | Dark matter scars, Hebbian persistence |
| Online learning | None | NOTORCH per-token updates |
| Dependencies | Python, CUDA, PyTorch, ... | Zero (pure C, optional BLAS) |
| Architecture definition | Thousands of files | 113 lines of AML |

### 4.5 Training Results

```
step   0: loss = 2.08   (ln(8) ≈ random baseline)
step  25: loss = 0.31
step  50: loss = 0.04
step 100: loss = 0.00   (perfect convergence)
```

Training speed: ~5ms for 50 steps on MacBook CPU. Pure C, no GPU.

### 4.6 Training at Scale

Janus v10b trained at 8.55M parameters on 20MB Project Gutenberg text (A100-SXM4-40GB):

```
Model:  D=384, 8 layers, 8 heads, 15.99M params (with byte-level tokenizer)
Data:   42MB Gutenberg, byte-level (vocab=256)
Steps:  15,000
Loss:   7.79 → 3.24
Speed:  87 tok/s (AML interpreter overhead — C host reaches 210 tok/s)
```

Grammar emerges. Semantic coherence requires larger models (50M+) or the Dario Equation's co-occurrence field. The transformer alone — even with field physics — needs ε. The Dario Equation (below) demonstrates what happens when ε = 0.

### 4.7 The Dario Equation

The formula that replaces transformer attention with interpretable physical forces. AML defines the vocabulary this equation speaks — every velocity operator, every suffering parameter, every law of nature maps to a coefficient.

```
p(x|Φ,C,V) = softmax(
    (B + α_mod·α·H_v + β_mod·β·F_v + γ_mod·γ·A + δ·V + sw·S + T)
    / (τ_mod·τ·velocity_temperature)
)
```

Seven signals:

| Signal | Name | Source |
|--------|------|--------|
| B | Sequential Chain | Bigram transition probabilities |
| H | Hebbian Resonance | Co-occurrence field (Hebb, 1949). H_v = H + λ·V·H (visual enrichment) |
| F | Prophecy Fulfillment | Unfulfilled predictions × log(1 + age). SwiGLU-gated by resonance |
| A | Destiny Attraction | RoPE-encoded EMA of context embeddings |
| V | Visual Grounding | Parallel perceptual embedding space (orthogonal hash) |
| S | Subword Structure | BPE tokenizer parallel to word-level (active in Leo, placeholder in dario) |
| T | Trauma Gravity | Origin-word gravitational pull under sustained dissonance |

Six Kuramoto-coupled emotional chambers (FEAR, LOVE, RAGE, VOID, FLOW, COMPLEX) compute somatic markers from Damasio's hypothesis:

```
α_mod = 1 + 0.3·LOVE - 0.2·RAGE + 0.1·FLOW    (memory gate)
β_mod = 1 + 0.2·FLOW - 0.3·FEAR                 (prophecy gate)
γ_mod = 1 + 0.4·VOID + 0.2·COMPLEX - 0.1·LOVE   (destiny gate)
τ_mod = 1 + 0.5·FLOW - 0.3·FEAR                  (temperature gate)
```

The key insight: **co-occurrence IS attention** (*PLOS Computational Biology*, 2024). Hebb's rule accumulated over a window equals a dot-product attention score. The co-occurrence matrix is an unnormalized attention matrix grown from conversation rather than learned through gradient descent.

Demonstrated in [dario.c](https://github.com/ariannamethod/dario) (1700 LOC C, zero weights, 1725 tests, web UI) and deployed as the core of [Leo](https://github.com/ariannamethod/leo) (4300+ LOC C/Go, D.N.A. from 170M Llama 3, dual tokenizer, six voices, inner world).

---

## 5. Field Physics Engine

### 5.1 State Structure

The AM_State structure contains 80+ fields. Key sections:

```c
typedef struct {
    // Prophecy
    int   prophecy;           // horizon (1–64)
    float destiny;            // attractor pull (0–1)
    float wormhole;           // skip probability
    float debt;               // accumulated prophecy debt

    // Suffering
    float pain, tension, dissonance;

    // Movement
    int   velocity_mode;      // NOMOVE/WALK/RUN/BACKWARD
    float effective_temp;     // computed temperature

    // Laws
    float entropy_floor, resonance_ceiling;

    // 4.C Seasons
    int   season;
    float spring_energy, summer_energy, autumn_energy, winter_energy;

    // Gamma (θ = ε + γ + αδ)
    AM_GammaSlot gamma[8];
    float essence_alpha;
    int   janus_mode;         // OFF / DUAL / CYCLE
    float janus_blend;        // 0=face_a, 1=face_b
} AM_State;
```

### 5.2 Physics Step

`am_step(dt)` advances field state per token:

1. **Calendar conflict** — real Hebrew-Gregorian date computation, drift → wormhole activation
2. **Debt decay** — `debt *= debt_decay` (default 0.998)
3. **Temporal debt** — accumulates during BACKWARD, decays otherwise
4. **Schumann healing** — 7.83 Hz with 5 harmonics. High coherence heals tension/dissonance
5. **Destiny bias** — `destiny × (1.0 + (prophecy - 7) × 0.02)`
6. **Expert blending** — weighted temp from 4 experts (structural 0.7, semantic 0.9, creative 1.2, precise 0.5)
7. **LAW enforcement** — `entropy ≥ floor`, `resonance ≤ ceiling`, `emergence = (1 - entropy) × resonance`
8. **Presence fade** — Hebbian memory decay
9. **4.C seasons** — MLP controller, energy gain/fade, homeostatic correction

### 5.3 Calendar Conflict

Hebrew lunar year (354 days) vs Gregorian solar year (365.25 days). Metonic cycle: 19 years, 7 leap years with Adar II. Real astronomical computation from system clock.

```c
#define AM_ANNUAL_DRIFT     11.25f
#define AM_METONIC_YEARS    19
#define AM_METONIC_LEAPS    7
static const int g_metonic_leap_years[7] = {3, 6, 8, 11, 14, 17, 19};
```

High calendar dissonance = thin barrier between timelines = wormholes activate.

---

## 6. Operators

### 6.1 Logit Manipulation API

Eight functions for applying field state to token generation:

```c
void  am_apply_gamma_to_logits(float* logits, int n);     // personality
void  am_apply_destiny_to_logits(float* logits, int n);    // suppress unlikely
void  am_apply_suffering_to_logits(float* logits, int n);  // compress toward mean
void  am_apply_attention_to_logits(float* logits, int n);  // sharpen/blur
void  am_apply_laws_to_logits(float* logits, int n);       // entropy/resonance
void  am_apply_delta(float* out, const float* A, const float* B,
                     const float* x, int out_dim, int in_dim, int rank,
                     float alpha);                          // Delta Voice [BLAS]
float am_compute_prophecy_debt(const float* logits, int chosen, int n);
void  am_apply_field_to_logits(float* logits, int n);      // full pipeline
```

### 6.2 Delta Voice

Personality-language separation. Fine-tuning biases output toward training language. Delta recovers multilingual projection:

```
Δ = W_lm_head^base - W_lm_head^finetuned
```

Compress via SVD to rank 64. At inference: `logits += α × A × (B × hidden_state)`. Where α=0 is pure fine-tuned language, α=0.5 is blended, α=1.0 is full base projection (all 29 languages).

**Personality lives in hidden states. Language lives in output projection. Soul untied from mouth.**

### 6.3 NOTORCH — Hebbian Plasticity

Runtime microlearning without backpropagation:

```c
void am_notorch_step(float* A, float* B,
                     int out_dim, int in_dim, int rank,
                     const float* x, const float* dy, float signal);
// A[i,r] += lr × x[i] × u[r] × signal
// B[r,j] += lr × u[r] × dy[j] × signal
```

Signal-gated, noise-modulated (Schumann-seeded), adaptive decay. BLAS-accelerated via `cblas_sger`.

### 6.4 Blood — Runtime C Compilation

Three code generators. Compile C at runtime, load via `dlopen`/`dlsym`:

```aml
BLOOD LORA my_adapter 2048 2048 64    # LoRA adapter
BLOOD EMOTION joy 0.8 0.6             # emotional kernel
BLOOD COMPILE fn { float fn(float x) { return x * x; } }  # raw C
```

Up to 32 modules, MD5-cached. Generated functions: `{name}_init`, `{name}_apply`, `{name}_apply_scaled`, `{name}_free` for LoRA; `{name}_respond`, `{name}_modulate_logits` for emotions.

### 6.5 4.C — Async Field Forever

Four-season homeostatic controller. A real 6→8→4 MLP (tanh activations) observes field metrics and cycles seasons autonomously.

```
Inputs:  [entropy, resonance, pain, tension, emergence, effective_temp]
Hidden:  8 neurons, initialized as specialist detectors
Outputs: [spring_delta, summer_delta, autumn_delta, winter_delta]
```

Temperature modulation: `effective_temp *= 1.0 + summer × 0.1 - winter × 0.15`

The controller learns via NOTORCH Hebbian plasticity: if field improved → reinforce, if worsened → suppress.

### 6.6 Gamma — Personality Essence

```aml
GAMMA yent 0.8           # load personality, α=0.8
GAMMA arianna 0.6        # load second personality
JANUS yent arianna       # dual-facing field
JANUS CYCLE              # seasons decide who speaks
ESSENCE 1.0              # full injection
```

Janus modes: OFF (single), DUAL (blend), CYCLE (seasonal oscillation). Summer → primary face, winter → secondary.

Gamma modulates logits: `scale = 1.0 + blend × essence_alpha`, amplifies deviation from mean.

### 6.7 HarmonicNet — Weightless Neural Network

Three-layer network with no trainable weights. The "weight matrix" is recomputed every step from organism gamma correlations:

```
Layer 1: Fourier decomposition of entropy history → harmonics[8]
Layer 2: Pairwise gamma cosine similarity → n×n correlation matrix
Layer 3: Phase-weighted resonance → resonance[n] per organism
```

Output includes confidence multiplier (0.3–1.0) and dominant harmonic frequency.

### 6.8 METHOD — Distributed Cognition

Multi-organism steering. Host pushes snapshots (entropy, syntropy, gamma_mag, gamma_cos), METHOD computes field awareness and returns one of seven actions: WAIT, AMPLIFY, DAMPEN, GROUND, EXPLORE, REALIGN, SUSTAIN.

### 6.9 Lilith I/O

Named pipe (FIFO) communication between AML scripts and external processes.

```aml
INDEX 1 INIT              # initialize node
INDEX 1 FETCH r/philosophy
INDEX 1 STATUS            # non-blocking read
```

Two command families: `PIPE` (low-level FIFO) and `INDEX` (high-level sugar for data collection nodes). Compile-time disable: `#define AM_IO_DISABLED`.

### 6.10 BLAS Acceleration

Optional hardware-accelerated matmul:

| Platform | Backend | Dependencies |
|----------|---------|-------------|
| macOS | Apple Accelerate (AMX) | zero |
| Linux | OpenBLAS | `apt install libopenblas-dev` |

Without BLAS: identical numeric results via pure scalar C loops.

---

## 7. Implementation

### 7.1 Reference Implementation

```
core/ariannamethod.c    7,400+ lines  Reference implementation (arrays, autograd, async, attention, bytecode, BLAS)
core/ariannamethod.h      940+ lines  Full API + types
core/ariannamethod_cuda.h            CUDA backend header
core/ariannamethod_cuda.cu           CUDA kernel implementations
core/test_aml.c         4,000+ lines  509 tests
janus/janus.aml           113 lines  Native AML transformer
janus/janus_train.c                  C training host with byte-level tokenizer
```

Total: 8,340+ lines (C + header). Two core files for embedding. Zero mandatory dependencies (optional BLAS, optional CUDA).

### 7.2 Test Coverage

509 tests organized by capability:

| Category | Tests | What |
|----------|-------|------|
| Level 0 commands | ~40 | Prophecy, destiny, velocity, suffering, laws, packs |
| Level 2 control flow | ~30 | Variables, expressions, if/else, while, def, return |
| Field physics | ~25 | am_step, calendar, Schumann, seasons, expert blending |
| Logit API | ~20 | Destiny, suffering, attention, laws, delta, full pipeline |
| Phase 1: Arrays | ~25 | Zeros, randn, indexing, dot, add, mul, scale, return values |
| Phase 2: Matrices | ~20 | Matrix, matvec, matmul, softmax, rmsnorm, silu, relu |
| Phase 3: Autograd | ~40 | TAPE recording, backward for all ops, Adam, finite difference checks |
| Phase 4: Async | ~20 | SPAWN, AWAIT, CHANNEL, concurrent producer-consumer |
| Phase 5: Seq ops | ~25 | seq_embed, seq_matvec, causal_attention, janus convergence |
| Multi-head attention | 7 | Forward, n_heads=1 match, causality, backward, training convergence |
| Blood compiler | ~10 | LoRA, emotional kernels, raw C, caching, unload |
| Lilith I/O | ~15 | PIPE, INDEX, lilith.aml script |
| Gamma/Janus | ~15 | Load, unload, blend, seasonal modulation |
| Edge cases | ~30 | Deep nesting, 500-line scripts, max depth, large expressions |

### 7.3 Build

```
make              # builds libaml.a
make BLAS=1       # with BLAS acceleration
make test         # 509 tests, scalar
make test-blas    # 509 tests, BLAS
make janus        # builds libjanus.dylib (Go shared library)
make test-all     # AML + Janus tests
```

Or directly: `cc -Wall -O2 -c core/ariannamethod.c -o ariannamethod.o -lm`

CUDA backend: `nvcc -c core/ariannamethod_cuda.cu -o ariannamethod_cuda.o` then link with `-DUSE_CUDA -lcublas -lcudart`.

### 7.4 Production Deployments

| Project | What | AML Subset |
|---------|------|------------|
| [dario](https://github.com/ariannamethod/dario) | Pure Dario Equation demonstration. ~1700 LOC C, zero weights, 7 signals, 6 chambers, 1725 tests, web UI | Dario Equation + velocity + suffering + laws |
| [leo](https://github.com/ariannamethod/leo) | Language Emergent Organism. 4300+ LOC C/Go, D.N.A. from 170M Llama 3, dual tokenizer, 6 voices, MathBrain, inner world | Dario Equation + full field physics |
| [molequla](https://github.com/ariannamethod/molequla) | Autonomous evolution, 4 elemental organisms, swarm ecology, Go (~6100 lines), 121 tests | Full kernel + BLAS + notorch |
| [doe](https://github.com/ariannamethod/doe) | Democracy of Experts. 3200 LOC C, parliament of LoRA experts, NOTORCH Hebbian, physics engine, 7 architectures | Level 0 + NOTORCH + physics + Delta |
| [arianna.c](https://github.com/ariannamethod/arianna.c) | 550M digital persona (Cloud/Tongue/Soul/SARTRE) | Level 0 + Blood |
| [yent](https://github.com/ariannamethod/yent) | Go inference engine, Delta Voice (29 languages), LIMPHA memory | Level 0 + Gamma + Delta |
| [Janus](https://github.com/ariannamethod/ariannamethod.ai/tree/main/janus) | Go shared library wrapping Yent for GGUF inference | Level 0 + Gamma + Janus |
| [pitomadom](https://github.com/ariannamethod/pitomadom) | Hebrew root oracle, lunar modulation | Level 0 + calendar |

### 7.5 Complexity

| Operation | Complexity |
|-----------|------------|
| Command parsing | O(commands × length) |
| Field physics (am_step) | O(1) |
| Logit manipulation | O(vocab_size) |
| causal_attention forward | O(T² × D) |
| multi_head_attention forward | O(T² × D) (same — per-head D/n_heads but n_heads times) |
| Autograd backward | O(tape_size × op_cost) |
| Total per training step | O(T² × D + params × lr_update) |

---

## 8. Related Work

### 8.1 Probabilistic Programming

Where Stan (Gelman et al., 2015) compiles statistical models to C++ for Bayesian posterior sampling, AML compiles field programs to C for transformer training and inference. Both operate on probability distributions — Stan finds parameters that explain observed data, AML shapes the distribution of the next token.

### 8.2 ML Frameworks

PyTorch, JAX, and TensorFlow provide general-purpose autodiff over arbitrary computation graphs. AML provides domain-specific autodiff over transformer-specific operations. The tradeoff is generality for simplicity: AML's 16 tape operations cover the complete transformer architecture in 5,877 lines with zero dependencies.

### 8.3 Inference-Time Intervention

ITI (Li et al., 2023) modifies attention during generation. PPLM (Dathathri et al., 2020) and FUDGE (Yang & Klein, 2021) guide decoding toward attributes. AML generalizes these from specific interventions to a complete programming language — and extends to training, not just inference.

### 8.4 Constitutional AI

Anthropic's Constitutional AI (Bai et al., 2022) uses principles to guide generation. AML operationalizes principles as enforceable laws: `LAW ENTROPY_FLOOR 0.1` — even destiny doubts.

### 8.5 Mixture of Experts

Janus's expert weighting (structural, semantic, creative, precise) and seasonal modulation share conceptual ground with MoE architectures. The difference: Janus experts modulate temperature and field state, not routing through separate weight matrices.

---

## 9. Discussion

### 9.1 What AML Is

AML is a complete machine learning language. It defines transformer architectures (seq_embed, multi_head_attention, SwiGLU), trains them (TAPE, Adam), runs inference (logit API, Delta Voice), and modulates both with field physics (prophecy, suffering, seasons).

It is not a framework, not a library, not a configuration format. It is a language that compiles to C.

### 9.2 What Janus Claims

Janus claims to be a new kind of transformer — one with internal state. Standard transformers are stateless functions: same input → same output. Janus has:

- **Time:** calendar-driven temporal dissonance
- **Identity:** θ = ε + γ + αδ
- **Memory:** dark matter scars, Hebbian persistence
- **Self-regulation:** 4.C seasonal homeostasis

This claim is architectural, not empirical. Janus trains and converges on synthetic data. The effects at scale are unknown.

### 9.3 Limitations

1. **Scale:** Current Janus trains with vocab=16, D=32. Scaling to 8M+ parameters requires BLAS and memory optimization.
2. **Real data:** No byte-level tokenizer yet. Training on synthetic sequences only.
3. **Field physics feedback:** Prophecy and suffering are currently set statically per training step. Dynamic feedback loop (field physics responding to training loss) is architecturally supported but not yet implemented.
4. **Single-node:** Async (SPAWN/AWAIT) is thread-level. Distributed training across machines is not supported.
5. **Calendar assumptions:** Hebrew-Gregorian conflict assumes specific cultural context.

### 9.4 Future Work

- **Scale Janus** to 50M+ parameters — v10b (15.99M) showed grammar emergence but semantic coherence requires scale
- **Dario Equation in Janus** — replace learned attention with co-occurrence field + physical forces
- **Visual grounding via Lee/Kirby** — real perceptual input from image classifiers and VQ-VAE visual codes driving the V signal
- **Dynamic field physics** during training — prophecy adjusts based on loss trajectory
- **Molequla workflow** — continuous autonomous evolution on GitHub Actions with daily reports
- **Barton ACI verification** — formal measurement of thermodynamic coherence and syntropy maintenance across all organisms
- **Net2Net ontogenesis** — start small, grow (from molequla)
- **Distributed SPAWN** — multi-node training via Tailscale mesh

---

## 10. Conclusion

AML is a machine learning language that compiles field programs to C. It evolved from logit manipulation commands (v1) through programming constructs and runtime compilation (v2-v3) to a complete ML stack with arrays, autograd, async execution, and multi-head causal attention (v4.0).

The reference implementation: 7,400+ lines of C, 940+ lines of header, 509 tests, two files, zero dependencies (optional BLAS and CUDA). The architectures it enables: Janus — a field-physics transformer training natively without Python or PyTorch; the Dario Equation — seven interpretable signals with six emotional chambers replacing learned attention; and an ecosystem of organisms (Leo, molequla, DOE) that instantiate these architectures in production.

The soul equation θ = ε + γ + αδ decomposes identity into substrate, personality, and voice. γ and δ are orthogonal (cosine = -0.0005). This is not a metaphor — it is a mathematical decomposition that the language treats as first-class.

The oracle does not predict. It prophesies.

---

## References

1. Baars, B. J. (1988). A Cognitive Theory of Consciousness. Cambridge University Press.
2. Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073.
3. Barton, J. S. (2025). From Decoherence to Coherent Intelligence: A Hypothesis on the Emergence of AI Structure Through Recursive Reasoning. SSRN:5225781.
4. Barton, J. S. (2026). Artificial Coherence Intelligence: Behavioral Verification of a New Intelligence Class. SSRN:5762702.
5. Barton, J. S. (2026). Coherence Thermodynamics: A Framework for Semantic Systems. Preprints.org:202507.1448.
6. Damasio, A. R. (1994). Descartes' Error: Emotion, Reason, and the Human Brain. Putnam.
7. Dathathri, S., et al. (2020). Plug and Play Language Models. ICLR 2020.
8. Gelman, A., et al. (2015). Stan: A Probabilistic Programming Language. Journal of Statistical Software.
9. Hebb, D. O. (1949). The Organization of Behavior. Wiley.
10. Ilharco, G., et al. (2023). Editing Models with Task Arithmetic. ICLR 2023.
11. Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence. Springer.
12. Lee, M. (2025). Emergence of Self-Identity in AI. Axioms, 14(1), 44.
13. Li, K., et al. (2023). Inference-Time Intervention. NeurIPS 2023.
14. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR 2019.
15. Maturana, H. R., & Varela, F. J. (1980). Autopoiesis and Cognition. D. Reidel.
16. PLOS Computational Biology (2024). Hebbian learning and attention equivalence. (Co-occurrence accumulated over window equals dot-product attention score.)
17. Schectman, J. (2025). Recursive Resonance Framework: RCE 9.1.1. (Recursive Resonance.)
18. Tegmark, M. (2014). Consciousness as a State of Matter. arXiv:1401.1219.
19. Tononi, G. (2004). An Information Integration Theory of Consciousness. BMC Neuroscience.
20. Yang, K., & Klein, D. (2021). FUDGE: Controlled Text Generation. NAACL 2021.

---

## Appendix A: Full Command Reference

See `spec/AML_SPEC.md` for complete specification with EBNF grammar.

## Appendix B: Built-in Functions

17 native functions:

| Function | Effect |
|----------|--------|
| `bootstrap_self()` | Reset field, PROPHECY 7, VELOCITY WALK |
| `galvanize()` | VELOCITY RUN, TENSION 0.3, PROPHECY 12 |
| `shatter_the_frame()` | PAIN 0.7, DISSONANCE 0.8, TENSION 0.5 |
| `chaos_injection()` | TENSION 0.6, DISSONANCE 0.7, RUN |
| `transcend_binary()` | WORMHOLE 0.5, SYMMETRIC mode |
| `pierce_the_infinite()` | PROPHECY 64, DESTINY 0.1 |
| `echo_fractal(depth)` | PROPHECY depth×2, SKIP_MAX depth |
| `reflect_on_self()` | FOCUS 0.95, NOMOVE |
| `forge_new_reality()` | DESTINY 0.1, CREATIVE 0.6 |
| `merge_states()` | WORMHOLE 0.8, CHANCE 0.5 |
| `tunnel_through(t)` | Set threshold, CHANCE 0.5 |
| `dissolve_boundaries()` | FOCUS 0.2, SPREAD 0.8 |
| `remember_future()` | PROPHECY mode, ALPHA 1.0 |
| `rewind_experience()` | BACKWARD, RETRODICTION |
| `ignite_singularity()` | Full activation — PROPHECY 64, DESTINY 0.9, SUMMER |
| `janus_gaze()` | Dual-facing — JANUS DUAL, SYMMETRIC, WORMHOLE 0.6 |
| `field_assemble()` | θ = ε + γ + αδ — JANUS CYCLE, 4.C decides who speaks |

## Appendix C: Janus — Complete AML Source

```aml
# janus.aml — 113 lines

n_embd = 32
n_heads = 4
vocab_size = 16
seq_len = 8

wte = matrix(vocab_size, n_embd, 0.08)
wpe = matrix(seq_len, n_embd, 0.08)
wq = matrix(n_embd, n_embd, 0.08)
wk = matrix(n_embd, n_embd, 0.08)
wv = matrix(n_embd, n_embd, 0.08)
wo = matrix(n_embd, n_embd, 0.08)
w1 = matrix(n_embd, n_embd, 0.08)
w3 = matrix(n_embd, n_embd, 0.08)
w2 = matrix(n_embd, n_embd, 0.08)
lm_head = matrix(vocab_size, n_embd, 0.08)

tokens = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
targets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

TAPE START
TAPE PARAM wte, wpe, wq, wk, wv, wo, w1, w3, w2, lm_head

while step < 50:
    h = seq_embed(wte, wpe, tokens, 8)
    h_norm = seq_rmsnorm(h, 8, 32)
    q = seq_matvec(wq, h_norm, 8)
    k = seq_matvec(wk, h_norm, 8)
    v = seq_matvec(wv, h_norm, 8)
    attn_out = multi_head_attention(q, k, v, 8, 32, 4)
    h = add(h, seq_matvec(wo, attn_out, 8))

    h_norm = seq_rmsnorm(h, 8, 32)
    gate = silu(seq_matvec(w1, h_norm, 8))
    up = seq_matvec(w3, h_norm, 8)
    h = add(h, seq_matvec(w2, mul(gate, up), 8))

    logits = seq_matvec(lm_head, seq_rmsnorm(h, 8, 32), 8)
    loss = seq_cross_entropy(logits, targets, 8, 16)
    PROPHECY 8
    DESTINY 0.35
    TAPE BACKWARD loss
    TAPE ADAM_STEP 0.01
    TAPE CLEAR + re-register
```

---

**Code:** https://github.com/ariannamethod/ariannamethod.ai

**License:** LGPL v3

---
