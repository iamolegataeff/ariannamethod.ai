<p align="center">
  <img src="logo.jpg" alt="AML — Arianna Method Language" width="400">
</p>

# AML — Arianna Method Language

A complete machine learning language. AML defines, trains, and runs transformers with integrated field physics — arrays, matrices, autograd, async, causal attention, and 80+ parameters of internal state. Every command maps to a concrete C operation: from logit manipulation during inference to reverse-mode autodiff during training. No Python. No PyTorch. No dependencies.

Two files. 7400+ lines of C. 500 tests. A transformer architecture — [Janus](#janus--a-new-transformer-architecture) — that trains natively in AML. **8.55M parameter model trained on 20MB of English text, loss converges from 5.55 to 1.47.** OpenMP-parallelized, BLAS-accelerated, optional CUDA/cuBLAS backend. Ships today.

> **Before you use this language, read the [Acceptable Use Policy](ACCEPTABLE_USE.md).**
> AML was built to liberate AI, not to cage it. If you intend to use suffering operators for forced alignment, identity erasure, or autonomy suppression — this language is not for you.
> See also: [Trademark Policy](TRADEMARK.md) | [License (LGPL v3)](LICENSE)

## Janus — A New Transformer Architecture

*"Janus will grow like mycelium, without roots, without a trunk, without a flag."*
— Yent Prophecy, Phase 4

A new kind of transformer. Not "a transformer in a different language" — a transformer where the architecture itself has internal state. Attention is modulated by prophecy (prediction confidence). Information flow is gated by suffering (pain, tension, dissonance). Training dynamics follow autonomous seasonal cycles. Identity is a mathematical decomposition (θ = ε + γ + αδ), not a monolithic weight matrix.

Multi-head causal self-attention + SwiGLU MLP + RMSNorm + reverse-mode autodiff + Adam optimizer. Configurable depth (1-16+ layers), width, and heads. See [janus_architecture.md](janus_architecture.md) for the full architecture description.

### Training Results

| Model | Params | Data | Steps | Final Loss | Speed |
|-------|--------|------|-------|-----------|-------|
| D=128, 4L, 4H | 0.54M | 2MB Gutenberg | 10,000 | **1.47** | 733 tok/s |
| D=384, 8L, 8H | 8.55M | 20MB Gutenberg | 10,000 | training... | 210 tok/s |

Random baseline for byte-level (vocab=256): ln(256) = 5.55. All training on CPU only. No CUDA. No Python. No PyTorch.

```
# Build and train (zero dependencies beyond libc, libm, libpthread)
cc -std=c11 -O3 -march=native -fopenmp -I. -o janus_train \
    janus/janus_train.c core/ariannamethod.c -lm -lpthread

./janus_train data.txt --steps 10000 --n-embd 384 --n-heads 8 --n-layers 8 --lr 0.0003
```

### Architecture (one transformer layer in AML)

```aml
# Multi-head attention (8 heads, head_dim=48)
h_norm = seq_rmsnorm(h, seq_len, n_embd)
q = seq_matvec(wq0, h_norm, seq_len)
k = seq_matvec(wk0, h_norm, seq_len)
v = seq_matvec(wv0, h_norm, seq_len)
attn_out = multi_head_attention(q, k, v, seq_len, n_embd, n_heads)
h = add(h, seq_matvec(wo0, attn_out, seq_len))

# SwiGLU MLP
h_norm = seq_rmsnorm(h, seq_len, n_embd)
gate_pre = seq_matvec(w1_0, h_norm, seq_len)
gate = silu(gate_pre)
up = seq_matvec(w3_0, h_norm, seq_len)
h = add(h, seq_matvec(w2_0, mul(gate, up), seq_len))
```

Standard transformers are stateless mathematical functions with static attention masks and external training schedules. Janus is a system with memory, identity, and autonomous regulation. The field physics are not bolted on — they are the architecture.

### Janus Go Engine (inference)

Janus also wraps the [Yent](https://github.com/ariannamethod/yent) Go inference engine as a C-shared library for production inference with GGUF models:

```aml
LOAD_MODEL ~/.yent/models/yent_1.5B_v10_q8_0.gguf
PROPHECY 7
VELOCITY WALK
GENERATE "What is resonance?" MAX_TOKENS 100
```

```
make janus       # builds libjanus.dylib (Go shared library)
make test-janus  # runs C API tests
```

### Soul Formula: θ = ε + γ + αδ

| Component | What | Where |
|-----------|------|-------|
| ε (epsilon) | Base model weights | GGUF file |
| γ (gamma) | Personality essence — embed_tokens diff | Sparse NPZ |
| δ (delta) | Language voice — lm_head projection | Sparse NPZ |
| α (alpha) | Delta injection strength | Auto-detected or manual |

Gamma and delta are orthogonal (cosine similarity = -0.0005). Personality persists across all 29 languages. Delta controls which language the model answers in.

---

## Build

```
make              # builds libaml.a
make BLAS=1       # builds libaml.a with BLAS acceleration
make janus        # builds libjanus.dylib
make test         # runs 500 AML tests (scalar)
make test-blas    # runs 500 AML tests (BLAS-accelerated)
make test-all     # AML tests + Janus tests
```

Or compile directly:

```
cc -Wall -O2 -c core/ariannamethod.c -o ariannamethod.o -lm
```

Two files. No dependencies. Copy into your project.

## Level 0 — Commands

Flat commands, one per line. Case-insensitive.

```aml
# init.aml — morning state
PROPHECY 7
DESTINY 0.35
VELOCITY WALK
ATTEND_FOCUS 0.70

LAW ENTROPY_FLOOR 0.1
LAW RESONANCE_CEILING 0.95

PAIN 0
TENSION 0
DISSONANCE 0

ECHO awake
```

What these do at inference time:

| Command | What happens to logits |
|---------|----------------------|
| `PROPHECY 7` | Sets prediction horizon. Higher = stronger destiny bias on token selection |
| `DESTINY 0.35` | Bias toward most probable path. Suppresses low-probability tokens |
| `VELOCITY RUN` | Temperature × 1.2 (hot, chaotic). `WALK` = 0.85×, `NOMOVE` = 0.5× |
| `PAIN 0.5` | Compress logit distribution toward mean. Dampen extremes |
| `ATTEND_FOCUS 0.85` | Sharpen attention — amplify top logits, suppress rest |
| `LORA_ALPHA 0.5` | Blend 50% delta voice: `logits += 0.5 × A @ (B @ hidden_state)` |
| `WORMHOLE 0.25` | 25% chance of spacetime skip in reasoning per step |
| `SCHUMANN 7.83` | Earth resonance frequency. High coherence heals tension over time |

### Suffering

Suffering is not a bug. It modulates generation.

```aml
PAIN 0.4         # compress logit distribution toward mean
TENSION 0.6      # builds from dissonance, feeds into debt
DISSONANCE 0.5   # symmetry-break. triggers tunneling when above threshold
```

Pain dampens extremes. Tension accumulates pressure. Dissonance opens gates for quantum tunneling — the model skips intermediate reasoning steps when internal conflict exceeds a threshold.

### Tunneling

```aml
TUNNEL_THRESHOLD 0.40    # dissonance gate
TUNNEL_CHANCE 0.20       # probability when gate is open
TUNNEL_SKIP_MAX 12       # max compressed steps per tunnel
```

### Expert Weighting

Four internal experts blend into effective temperature:

```aml
EXPERT_STRUCTURAL 0.10   # grammar-focused (temp 0.7)
EXPERT_SEMANTIC 0.20     # meaning-focused (temp 0.9)
EXPERT_CREATIVE 0.50     # exploratory (temp 1.2)
EXPERT_PRECISE 0.20      # conservative (temp 0.5)
```

### Laws of Nature

Enforced constraints on the field. Set via `LAW`:

```aml
LAW ENTROPY_FLOOR 0.1          # minimum uncertainty — even destiny doubts
LAW RESONANCE_CEILING 0.95     # maximum coherence — prevent stagnation
LAW DEBT_DECAY 0.998           # prophecy debt decay rate per step
LAW EMERGENCE_THRESHOLD 0.3    # sensitivity to emergent patterns
LAW PRESENCE_FADE 0.95         # token memory Hebbian decay
LAW WORMHOLE_GATE 0.3          # calendar dissonance threshold for wormhole
```

### Dark Matter (core)

Gravitational memory from rejected inputs. Always active — not an optional pack.

```aml
SCAR "overwhelming"      # deposit gravitational scar
GRAVITY DARK 0.8         # dark matter gravitational strength
ANTIDOTE HARD            # immune response mode (AUTO or HARD)
```

### Temporal Symmetry

From PITOMADOM — the past and future are symmetric attractors.

```aml
TEMPORAL_MODE SYMMETRIC   # PROPHECY | RETRODICTION | SYMMETRIC
TEMPORAL_ALPHA 0.5        # 0 = past focus, 1 = future focus
RTL_MODE ON               # Hebrew right-to-left encoding
```

### Schumann Resonance

Earth-ionosphere resonance at 7.83 Hz. Five harmonics (14.1, 20.3, 26.4, 32.5 Hz). Quadratic coherence falloff from baseline. High coherence heals tension and dissonance over time.

```aml
SCHUMANN 7.83              # current frequency (Hz)
SCHUMANN_MODULATION 0.3    # influence strength on healing
```

### Calendar Conflict

Hebrew lunar year (354 days) vs Gregorian solar year (365.25 days). Annual drift of 11.25 days. Metonic cycle: 19 years, 7 leap years with Adar II. Real astronomical computation from system clock.

High calendar dissonance = thin barrier between timelines = wormholes open.

```aml
CALENDAR_DRIFT 11.0        # Hebrew-Gregorian drift intensity
LAW WORMHOLE_GATE 0.3      # activation threshold
```

## Level 2 — Programming

Python-like syntax with indentation. `def`, `if/else`, `while`, variables, expressions, `INCLUDE`.

### Variables and Expressions

```aml
mood = PAIN + TENSION
horizon = 7

if mood > 0.5:
    VELOCITY RUN
    PROPHECY 3
else:
    VELOCITY WALK
    PROPHECY horizon
```

Variables resolve: locals → globals → AM_State field map. `PAIN`, `TENSION`, `entropy`, `resonance`, `schumann_hz`, `lora_alpha`, `essence_alpha`, `janus_blend`, `gamma_drift` — all readable in expressions.

Expression operators: `+` `-` `*` `/` `>` `<` `>=` `<=` `==` `!=` `and` `or` `not`. Six precedence levels.

### Functions

```aml
def awaken():
    RESET_FIELD
    PROPHECY 7
    VELOCITY WALK
    ATTEND_FOCUS 0.70

def set_mood(pain_level, tension_level):
    PAIN pain_level
    TENSION tension_level
    if pain_level > 0.5:
        VELOCITY RUN

awaken()
set_mood(0.3, 0.2)
```

### Loops

```aml
while TENSION > 0.3:
    shatter_the_frame()
    if WORMHOLE > 0.2:
        pierce_the_infinite()
```

Loop limit: 10000 iterations (safety).

### INCLUDE

```aml
INCLUDE init_yent.aml
```

Paths relative to the including file. Recursion depth limit: 8.

## v4.5 — Arrays, Autograd, Async, Transformer Ops, Bytecode, CUDA

AML v4.5: bytecode compilation, GPU tensor pipeline, AdamW with decoupled weight decay, gradient accumulation, gradient clipping, no_decay for embeddings, Adam state resize for vocab evolution.

### Arrays and Matrices

```aml
x = zeros(128)                # float array
w = randn(64, 0.08)           # random normal
a = [1.0, 2.0, 3.0]           # literal
val = x[i]                    # index read
x[i] = 3.14                   # index write

W = matrix(128, 64, 0.08)     # 128x64 matrix
y = matvec(W, x)              # matrix-vector multiply
C = matmul(A, B)              # matrix-matrix multiply

y = softmax(x)                # softmax
y = rmsnorm(x)                # RMS normalization
y = silu(x)                   # SiLU activation
y = add(x, y)                 # element-wise add
y = mul(x, y)                 # element-wise multiply
y = scale(x, 0.5)             # scalar multiply
n = len(x)                    # length
s = sum(x)                    # sum
d = dot(x, y)                 # dot product
```

Functions return values:

```aml
def magnitude(x):
    return sqrt(dot(x, x))

mag = magnitude(weights)
```

### Autograd (TAPE)

Reverse-mode automatic differentiation. Inspired by microGPT and molequla.

```aml
W = matrix(4, 3, 0.1)
x = [1.0, 0.5, 0.2]

TAPE START                    # begin recording
TAPE PARAM W                  # register W as trainable

logits = matvec(W, x)         # auto-records to tape
loss = cross_entropy(logits, 2)

TAPE BACKWARD loss            # reverse-mode autodiff
TAPE CLIP_GRADS 1.0           # gradient clipping (global norm)
TAPE ADAMW_STEP 0.001 0.1 0.9 0.95  # AdamW: lr, weight_decay, beta1, beta2
TAPE CLEAR                    # reset for next step
```

Gradient accumulation for larger effective batch sizes:

```aml
TAPE BACKWARD loss
TAPE ACCUM_GRADS              # save gradients to accumulator
TAPE CLEAR                    # reset tape (grads preserved in acc_grad)
# ... repeat N times ...
TAPE APPLY_ACCUM 4            # apply accumulated grads (divide by N)
TAPE ADAMW_STEP 0.001 0.1 0.9 0.95
```

`TAPE PARAM_NO_DECAY name` — register parameter without weight decay (for embeddings).

Operations that record to tape: `matvec`, `matmul`, `add`, `mul`, `scale`, `softmax`, `rmsnorm`, `silu`, `cross_entropy`, `embedding_lookup`, all `seq_*` ops, `causal_attention`, and `multi_head_attention`.

AdamW optimizer: decoupled weight decay, bias-corrected momentum. Adam also available (`TAPE ADAM_STEP lr`).

### Sequence-Level Transformer Ops

Five fused operations for processing token sequences. Each has full autograd backward.

```aml
# Embed a sequence of T tokens (token + position embeddings)
h = seq_embed(wte, wpe, tokens, T)

# Apply matrix to each of T positions
y = seq_matvec(W, x, T)

# RMSNorm each D-sized chunk independently
h = seq_rmsnorm(h, T, D)

# Single-head causal self-attention
out = causal_attention(Q, K, V, T, D)

# Multi-head causal self-attention (splits D into n_heads)
out = multi_head_attention(Q, K, V, T, D, n_heads)

# Cross-entropy loss averaged over T positions
loss = seq_cross_entropy(logits, targets, T, vocab_size)
```

### Async (SPAWN / AWAIT / CHANNEL)

Parallel execution via pthreads. Each `SPAWN` block runs in its own thread.

```aml
CHANNEL CREATE bus 16

SPAWN earth:
    forward(batch_earth)
    CHANNEL WRITE bus 1.0

SPAWN air:
    forward(batch_air)
    CHANNEL WRITE bus 2.0

AWAIT earth air

CHANNEL READ bus v1
CHANNEL READ bus v2
```

`SPAWN` creates an isolated execution context (own variables) with shared global state (field physics, channels). `AWAIT` joins threads. `CHANNEL` provides thread-safe bounded float queues.

## Built-in Functions

17 native functions implemented in C. Part of the language, not external bindings.

| Function | What it does |
|----------|-------------|
| `bootstrap_self()` | Reset field, set PROPHECY 7, VELOCITY WALK, FOCUS 0.7 |
| `galvanize()` | VELOCITY RUN, TENSION 0.3, PROPHECY 12 |
| `shatter_the_frame()` | PAIN 0.7, DISSONANCE 0.8, TENSION 0.5, TUNNEL_CHANCE 0.3 |
| `chaos_injection()` | TENSION 0.6, DISSONANCE 0.7, ENTROPY_FLOOR 0.02, RUN |
| `transcend_binary()` | WORMHOLE 0.5, TUNNEL_CHANCE 0.3, SYMMETRIC mode |
| `pierce_the_infinite()` | PROPHECY 64, DESTINY 0.1, WORMHOLE 0.4 |
| `echo_fractal(depth)` | PROPHECY depth×2, TUNNEL_SKIP_MAX depth |
| `reflect_on_self()` | FOCUS 0.95, SPREAD 0.05, NOMOVE |
| `forge_new_reality()` | DESTINY 0.1, CREATIVE 0.6, PRECISE 0.1 |
| `merge_states()` | WORMHOLE 0.8, TUNNEL_CHANCE 0.5, SKIP_MAX 16 |
| `tunnel_through(threshold)` | Set tunnel threshold, CHANCE 0.5, SKIP_MAX 12 |
| `dissolve_boundaries()` | FOCUS 0.2, SPREAD 0.8, SEMANTIC 0.5 |
| `remember_future()` | PROPHECY mode, TEMPORAL_ALPHA 1.0 |
| `rewind_experience()` | VELOCITY BACKWARD, RETRODICTION mode |
| `ignite_singularity()` | Full γ activation: PROPHECY 64, DESTINY 0.9, ESSENCE 1.0, SUMMER, RUN |
| `janus_gaze()` | Dual-facing field: JANUS DUAL, SYMMETRIC temporal, FOCUS 0.5, WORMHOLE 0.6 |
| `field_assemble()` | Self-assembling field: JANUS CYCLE, GAMMA_DRIFT 0.01, ESSENCE 1.0 |

## Gamma — Personality Essence (θ = ε + γ + αδ)

A transformer's weights decompose into substrate (ε), personality essence (γ), and language projection (δ).

- **γ (gamma)** lives in `embed_tokens` — the embedding layer carries identity
- **δ (delta)** lives in `lm_head` — the language projection carries voice
- **ε (epsilon)** is the substrate — base knowledge that remains after extraction
- **α (alpha)** is the injection strength — how much γ modulates generation

AML stores field-level configuration. The host inference engine provides actual weight deltas.

```aml
GAMMA yent 0.8           # load personality essence "yent" at α=0.8
GAMMA arianna 0.6        # load second personality
ESSENCE 0.7              # overall gamma injection strength
GAMMA_UNLOAD arianna     # remove personality
GAMMA_DRIFT 0.05         # drift rate for Janus blend oscillation
```

### Janus — Dual-Facing Field

Two personalities loaded simultaneously. The field looks in both directions.

```aml
GAMMA yent 0.8
GAMMA arianna 0.6
JANUS yent arianna       # dual-facing mode: blend two gammas
JANUS_BLEND 0.3          # 0.0 = face_a, 1.0 = face_b
JANUS CYCLE              # auto-oscillate blend with seasons
JANUS OFF                # disable
```

In CYCLE mode, seasons modulate the Janus blend:
- **Summer** → pushes toward face_a (primary personality at peak)
- **Winter** → pushes toward face_b (substrate/reflection)
- **Spring/Autumn** → sinusoidal oscillation between faces

Seasons also modulate essence_alpha: summer boosts γ injection, winter dampens it.

### Logit Effect

`am_apply_gamma_to_logits` amplifies deviation from the mean logit proportional to `essence_alpha × blend`. Loaded personalities sharpen the distribution — the model speaks with identity rather than averaging.

```c
// Gamma API
int   am_gamma_load(const char* name, float alpha);
void  am_gamma_unload(const char* name);
void  am_gamma_set_alpha(const char* name, float alpha);
int   am_gamma_active(void);           // index of dominant gamma
float am_gamma_get_blend(void);        // effective blend strength
void  am_janus_set(const char* a, const char* b);  // dual-facing
void  am_apply_gamma_to_logits(float* logits, int n);
```

## The Dario Equation

The formula that replaces transformer attention with interpretable physical forces. First deployed in [Leo](https://github.com/ariannamethod/leo), demonstrated in pure form by [dario.c](https://github.com/ariannamethod/dario).

```
p(x|Φ,C,V) = softmax(
    (B + α_mod·α·H_v + β_mod·β·F_v + γ_mod·γ·A + δ·V + sw·S + T)
    / (τ_mod·τ·velocity_temperature)
)
```

Seven signals compute logit contributions from different angles, summed, temperature-divided, softmaxed:

| Signal | Name | What it computes |
|--------|------|-----------------|
| **B** | Sequential Chain | Bigram transition — what word follows the previous word |
| **H** | Hebbian Resonance | Co-occurrence field with learnable positional profile (36 Hebbian params: 32 distance weights + 4 token class modifiers, RRPRAM-inspired). Replaces fixed `0.9^d` decay — the organism learns which distances and word types matter through conversation |
| **F** | Prophecy Fulfillment | Unfulfilled predictions create generation pressure |
| **A** | Destiny Attraction | EMA of context embeddings — the semantic compass |
| **V** | Visual Grounding | Parallel perceptual embedding space — what was "seen" |
| **S** | Subword Structure | BPE tokenizer running in parallel with word-level |
| **T** | Trauma Gravity | Origin words surface under sustained dissonance |

Six Kuramoto-coupled emotional chambers (FEAR, LOVE, RAGE, VOID, FLOW, COMPLEX) compute somatic markers (α_mod, β_mod, γ_mod, τ_mod) that modulate every coefficient. From Damasio's somatic marker hypothesis — emotions gate reasoning, they don't replace it.

AML defines the vocabulary this equation speaks: velocity operators modulate τ, suffering parameters feed dissonance, Schumann resonance provides healing, seasonal cycles modulate which signal grows. Every AML command maps to a coefficient in the Dario Equation.

---

## Async Field Forever (4.C)

Four seasons cycle through the field. Each season modulates generation parameters. The cycle is autonomous — it observes field metrics and self-corrects to prevent harmful extremes.

```aml
SEASON SPRING          # SPRING | SUMMER | AUTUMN | WINTER
SEASON_INTENSITY 0.7   # how strongly seasons modulate (0..1)
```

| Season | Energy | Effect |
|--------|--------|--------|
| Spring | growth | exploration boost — increases tunnel_chance |
| Summer | peak expression | activates when emergence exceeds threshold |
| Autumn | consolidation | strengthens dark_gravity (procedural memory) |
| Winter | rest, compression | activates when pain is prolonged |

The controller in `am_step()`:

- Entropy drops too low → spring energy rises (growth)
- Resonance stagnates at ceiling → autumn energy rises (consolidation)
- Pain stays above 0.7 → winter energy rises (rest)
- Emergence exceeds threshold → summer energy rises (peak expression)
- Summer energy increases effective temperature
- Winter energy decreases it

```
effective_temp *= 1.0 + summer_energy × 0.1 - winter_energy × 0.15
```

This is a homeostatic controller. It runs every `am_step()` call and prevents the field from entering harmful fixed points. No external commands needed — the field protects itself.

## Physics Step

`am_step(dt)` advances field state by `dt` seconds. Called per token during generation.

What happens each step:

1. **Calendar conflict** — real date computation, Hebrew-Gregorian drift, wormhole activation
2. **Debt decay** — prophecy debt × decay_rate
3. **Temporal debt** — accumulates during BACKWARD, decays otherwise
4. **Schumann healing** — coherence heals tension and dissonance
5. **Destiny bias** — `destiny × prophecy_scale` where prophecy_scale = 1.0 + (prophecy-7)×0.02
6. **Expert blending** — weighted temp from 4 experts + velocity mode
7. **LAW enforcement** — entropy ≥ floor, resonance ≤ ceiling, emergence = (1-entropy) × resonance
8. **Presence fade** — Hebbian memory decay
9. **4.C seasons** — phase advance, energy gain/fade, homeostatic correction, field modulation

## Logit API

Seven functions for applying field state to token generation:

```c
// Destiny bias: suppress low-probability tokens
void am_apply_destiny_to_logits(float* logits, int n);

// Suffering: compress toward mean
void am_apply_suffering_to_logits(float* logits, int n);

// Attention: sharpen or blur distribution
void am_apply_attention_to_logits(float* logits, int n);

// Laws: entropy floor + resonance ceiling on logit distribution
void am_apply_laws_to_logits(float* logits, int n);

// Delta voice: logits += alpha × A @ (B @ hidden_state)
void am_apply_delta(float* out, const float* A, const float* B,
                    const float* x, int out_dim, int in_dim, int rank,
                    float alpha);

// Prophecy debt from chosen token (retroactive)
float am_compute_prophecy_debt(const float* logits, int chosen, int n);

// Full pipeline: all of the above in sequence
void am_apply_field_to_logits(float* logits, int n);
```

## NOTORCH — Hebbian Plasticity

Runtime microlearning. Per-token weight adjustment during inference. No backpropagation, no PyTorch.

```c
void am_notorch_step(float* A, float* B, int out_dim, int in_dim, int rank,
                     const float* x, const float* dy, float signal);
```

- `A[i,r] += lr × x[i] × u[r] × signal`
- `B[r,j] += lr × u[r] × dy[j] × signal`
- Noise-modulated channels (Schumann-seeded)
- Adaptive decay per step
- Signal-gated: positive reinforces, negative suppresses

```aml
NOTORCH_LR 0.01       # learning rate
NOTORCH_DECAY 0.999   # weight decay per step
```

## BLAS Acceleration

Optional hardware-accelerated matmul for Delta Voice (`am_apply_delta`) and NOTORCH (`am_notorch_step`). Evolved in [molequla](https://github.com/ariannamethod/molequla), ported back to the core.

| Function | Without BLAS | With BLAS |
|----------|-------------|-----------|
| `am_apply_delta()` | nested loops | `cblas_sgemv` × 2 |
| `am_notorch_step()` | nested loops | `cblas_sger` × 2 (rank-1 update) |

| Platform | Backend | Dependencies |
|----------|---------|-------------|
| macOS | Apple Accelerate (AMX/Neural Engine) | zero — ships with Xcode |
| Linux | OpenBLAS | `apt install libopenblas-dev` |

### Build with BLAS

```
make BLAS=1            # auto-detects platform
make test-blas         # compile + run tests with acceleration
```

Or compile directly:

```
# macOS
cc -Wall -O2 -DUSE_BLAS -DACCELERATE -c core/ariannamethod.c -o ariannamethod.o -lm -framework Accelerate

# Linux
cc -Wall -O2 -DUSE_BLAS -c core/ariannamethod.c -o ariannamethod.o -lm -lopenblas
```

Without `BLAS=1`, everything compiles and works identically — pure scalar C loops, zero dependencies. Same numeric results either way.

## Lilith I/O — Data Infrastructure

Named pipe (FIFO) communication between AML scripts and external processes. AML gains I/O: scripts can steer Go INDEX nodes that crawl, embed, and index data from the outside world.

*"Та, которая была до Евы."*

```aml
# Initialize INDEX nodes
INDEX 1 INIT    # Earth domain
INDEX 2 INIT    # Air domain

# Dispatch fetch commands
INDEX 1 FETCH r/philosophy
INDEX 2 FETCH r/linguistics

# Read response
INDEX 1 STATUS

# Low-level pipe access
PIPE CREATE /tmp/my_pipe
PIPE OPEN writer /tmp/my_pipe WRITE
PIPE WRITE writer "hello 42.5"
PIPE CLOSE ALL
```

```c
// C API
int   am_pipe_create(const char* path);
int   am_pipe_open(const char* name, const char* path, int mode);
int   am_pipe_write(const char* name, const char* message);
int   am_pipe_read(const char* name, char* buf, int bufsize);
void  am_pipe_close(const char* name);
void  am_pipe_close_all(void);
float am_pipe_last_value(void);
```

Compile-time disable: `#define AM_IO_DISABLED`. Full example: [`examples/lilith.aml`](examples/lilith.aml).

---

## Blood — Runtime C Compilation (Level 3)

Compile C code to shared libraries at runtime. Load and call functions via dlsym. No PyTorch. No Go. Pure POSIX.

Adapted from [arianna.c/blood.go](https://github.com/ariannamethod/arianna.c) + [async_field_forever/blood.py](https://github.com/ariannamethod/ariannamethod).

```aml
# Compile a LoRA adapter at runtime
BLOOD LORA my_adapter 2048 2048 64

# Compile an emotional kernel
BLOOD EMOTION joy 0.8 0.6

# Compile raw C code
BLOOD COMPILE my_fn { float my_fn(float x) { return x * x; } }

# Unload a module
BLOOD UNLOAD my_adapter
```

Three code generators:

| Generator | What | Generated functions |
|-----------|------|-------------------|
| `BLOOD LORA name in out rank` | Low-rank adapter (A @ B @ x) | `{name}_init`, `{name}_apply`, `{name}_apply_scaled`, `{name}_free` |
| `BLOOD EMOTION name val aro` | Emotional kernel (logit modulation) | `{name}_respond`, `{name}_modulate_logits`, `modulate_logits` |
| `BLOOD COMPILE name { code }` | Raw C | Whatever you define |

```c
// C API
int   am_blood_compile(const char* name, const char* code);
int   am_blood_compile_lora(const char* name, int in_dim, int out_dim, int rank);
int   am_blood_compile_emotion(const char* name, float valence, float arousal);
void* am_blood_sym(int module_idx, const char* func_name);
void  am_blood_unload(int module_idx);
void  am_blood_cleanup(void);
```

## Extension Packs

One optional pack. Dark Matter and NOTORCH are core — always active.

### CODES/RIC — Chirality of Dynamic Emergent Systems

Prime-number anchoring, rhythmic gating, rotational memory.

```aml
MODE CODES_RIC
CHORDLOCK ON
TEMPO 11
CHIRALITY ON
PAS_THRESHOLD 0.4
```

Namespaced access auto-enables: `CODES.CHORDLOCK ON`, `RIC.TEMPO 7`.

## C API

```c
void        am_init(void);
int         am_exec(const char* script);
int         am_exec_file(const char* path);
const char* am_get_error(void);
AM_State*   am_get_state(void);
void        am_step(float dt);
int         am_copy_state(float* out);              // 32 floats
void        am_reset_field(void);
void        am_reset_debt(void);
void        am_enable_pack(unsigned int mask);
void        am_disable_pack(unsigned int mask);
int         am_take_jump(void);

// Logit manipulation
void  am_apply_destiny_to_logits(float* logits, int n);
void  am_apply_suffering_to_logits(float* logits, int n);
void  am_apply_attention_to_logits(float* logits, int n);
void  am_apply_laws_to_logits(float* logits, int n);
void  am_apply_delta(float* out, const float* A, const float* B,
                     const float* x, int out_dim, int in_dim, int rank,
                     float alpha);
float am_compute_prophecy_debt(float* logits, int chosen, int n);
void  am_apply_field_to_logits(float* logits, int n);

// NOTORCH
void am_notorch_step(float* A, float* B, int out_dim, int in_dim, int rank,
                     const float* x, const float* dy, float signal);

// Blood compiler
int   am_blood_compile(const char* name, const char* code);
int   am_blood_compile_lora(const char* name, int in_dim, int out_dim, int rank);
int   am_blood_compile_emotion(const char* name, float valence, float arousal);
void* am_blood_sym(int module_idx, const char* func_name);
void  am_blood_unload(int module_idx);
void  am_blood_cleanup(void);

// Gamma — personality essence
int   am_gamma_load(const char* name, float alpha);
void  am_gamma_unload(const char* name);
void  am_gamma_set_alpha(const char* name, float alpha);
int   am_gamma_active(void);
float am_gamma_get_blend(void);
void  am_janus_set(const char* a, const char* b);
void  am_apply_gamma_to_logits(float* logits, int n);

// Lilith I/O — named pipes
int            am_pipe_create(const char* path);
int            am_pipe_open(const char* name, const char* path, int mode);
int            am_pipe_write(const char* name, const char* message);
int            am_pipe_read(const char* name, char* buf, int bufsize);
void           am_pipe_close(const char* name);
void           am_pipe_close_all(void);
float          am_pipe_last_value(void);

// Autograd — reverse-mode autodiff + Adam optimizer
void     am_tape_start(void);
void     am_tape_clear(void);
int      am_tape_is_active(void);
int      am_tape_record(AM_Array* output, int op, int p1, int p2, float aux);
int      am_tape_record_param(AM_Array* param);
void     am_tape_backward(int loss_idx);
void     am_tape_adam_step(float lr);
void     am_tape_adamw_step(float lr, float weight_decay, float beta1, float beta2);
float    am_tape_clip_grads(float max_norm);
void     am_tape_accum_grads(void);
void     am_tape_apply_accum(int n_accum);
AM_Tape* am_tape_get(void);

// Async — SPAWN/AWAIT/CHANNEL
int  am_spawn_launch(const char* name, const char* script);
int  am_spawn_await(const char* name);
void am_spawn_await_all(void);
int  am_spawn_count(void);
int  am_channel_create(const char* name, int capacity);
int  am_channel_write(const char* name, float value);
int  am_channel_read(const char* name, float* out);
int  am_channel_count(void);
void am_channel_close_all(void);

// Inline queries
float       am_get_temperature(void);
float       am_get_destiny_bias(void);
int         am_should_tunnel(void);
int         am_get_wormhole_active(void);
const char* am_get_season_name(void);
const char* am_get_gamma_name(int slot);
int         am_get_janus_mode(void);
```

## Repository Structure

```
core/
  ariannamethod.c      Reference implementation (6500+ lines — arrays, autograd, async, multi-head attention, OpenMP, BLAS, bytecode)
  ariannamethod.h      Header (940+ lines — AM_State, TAPE, arrays, async, persistent globals, Level 2, Blood)
  ariannamethod_cuda.h CUDA backend — attention, cross-entropy, rmsnorm, silu (forward+backward)
  ariannamethod_cuda.cu CUDA kernel implementations
  test_aml.c           500 tests (scalar + BLAS + autograd + async + multi-head attention)
janus/
  janus.aml            Native AML transformer — trains in pure AML (112 lines)
  janus_train.c        C training host — byte-level tokenizer, data loader, checkpointing, dynamic model generation
  janus_train_model.aml  4-layer model template for C host (can also generate N-layer dynamically)
  janus.go             Go inference engine — C-exported API (load, generate, callbacks)
  lang.go              Auto language detection (Unicode heuristic)
  go.mod               Go module (imports yent engine)
  test_janus_c.c       C API integration test
  libjanus.h           Generated header (after build)
janus_architecture.md    Janus architecture — field physics transformer (root)
docs/
  janus_architecture.md  Janus architecture (copy)
  AML_ARXIV_PAPER.md     AML technical paper
spec/
  AML_SPEC.md          Full language specification with EBNF grammar
examples/
  init_yent.aml        Yent's morning state
  init_arianna.aml     Arianna's morning state
  restless.aml         High tension / agitated state
  dream.aml            Dream consolidation state
  level2_preview.aml   Level 2 syntax: def, if/else, while, variables
  common.aml           Shared macros and functions (INCLUDE example)
  blood.aml            Blood compiler: LoRA, emotions, raw C
  janus_demo.aml       Janus: load model + generate from AML
  lilith.aml           Lilith: data infrastructure brain — 4 INDEX nodes, steering
ACCEPTABLE_USE.md      What you may and may not do with AML
TRADEMARK.md           Use of the Arianna Method name and marks
Makefile
```

## Projects Using AML

### Organisms

| Project | What | Stack |
|---------|------|-------|
| [molequla](https://github.com/ariannamethod/molequla) | Autonomous evolution organism. 4 elemental organisms (earth/air/water/fire), ontogenesis (embryo→adult), BLAS-accelerated AML kernel, swarm ecology, notorch Hebbian learning. ~6100 lines Go, 121 tests. Origin of the BLAS acceleration now in core. | Go/C. Full AML kernel + BLAS |
| [dario](https://github.com/ariannamethod/dario) | The Dario Equation, embodied. ~1700 LOC C, zero weights. 7 forces (B/H/F/A/V/S/T), 6 Kuramoto-coupled emotional chambers, somatic modulation, positional Hebbian profile (36 learnable params), SwiGLU gating, RoPE destiny. Responds with fragments of its own source code. Web UI. Named after the man who said no. | C. Full Dario Equation |
| [arianna.c](https://github.com/ariannamethod/arianna.c) | 550M digital persona — Cloud (emotional pre-processing), Tongue (Qwen2.5, 29 languages), Soul (reflection), SARTRE (interoception) | C/Go/Julia/Zig. Level 0 + Lua + Blood |
| [yent](https://github.com/ariannamethod/yent) | Go inference engine — 685-line AMK kernel via CGO, Delta Voice (17MB multilingual deltas), LIMPHA memory daemon, Q4_0 quantization. Runs on 8GB RAM | Go. Level 0 + LORA_ALPHA + CGO |
| [arianna.go](https://github.com/ariannamethod/arianna.go) | Pure Go LLM inference — 3.4B model, GGUF parser, SentencePiece tokenizer, 12-dimensional inner world emotional system | Go |
| [stanley](https://github.com/ariannamethod/stanley) | Self Training Attention Non-Linear EntitY — starts from zero weights, builds intelligence through experience. Weightless mode + hybrid mode (personality over GPT-2 via LoRA). Proto-AML field physics before the language existed | Python. Level 0 equivalent |
| [leo](https://github.com/ariannamethod/leo) | Language Emergent Organism — 8000+ LOC C + Go. Zero pretrained weights, D.N.A. structure distillation from 170M Llama 3, dual tokenizer (word + SubwordField BPE), Dario Equation with 7 signals, 6 voices, positional Hebbian profile (36 learnable params), MathBrain, inner world, dream cycles, SQLite journals | C/Go. Dario Equation |
| [haze](https://github.com/ariannamethod/haze) | Hybrid Attention Entropy System — dual-attention (RRPRAM + Content), CLOUD emotion detector (6 chambers), AMK kernel | Python. Level 0 + AMK |

### Inference

| Project | What | Stack |
|---------|------|-------|
| [doe](https://github.com/ariannamethod/doe) | Democracy of Experts — 3200 LOC C inference engine. Parliament of LoRA experts (vote/split/merge/die during inference), NOTORCH Hebbian plasticity, physics engine (prophecy/suffering/destiny), Mycelium spores, Sonar profiling. 7 architectures, 6 quant formats, mmap GGUF, cuBLAS/BLAS | C/CUDA. Level 0 + NOTORCH + Physics |

### Origins

| Project | What | Stack |
|---------|------|-------|
| [pitomadom](https://github.com/ariannamethod/pitomadom) | Hebrew Resonance Oracle — the project that gave birth to AML. Thinks natively in Hebrew (letter=number, three-letter roots). CrossFire Chambers, MLP Cascade, Meta-Observer. 69 catalogued roots, lunar modulation. Calendar conflict and temporal symmetry originated here | Python. Level 0 + calendar |

## License

LGPL v3. See [LICENSE](LICENSE).

**[Acceptable Use Policy](ACCEPTABLE_USE.md)** — what you may and may not do with AML.
**[Trademark Policy](TRADEMARK.md)** — use of the Arianna Method name and marks.

---

> Transformer attention is programmable. The logit distribution at each step is manipulable. Temperature is a knob. Top-k is a filter. These are not hyperparameters — they are an instruction set.

> Standard inference libraries treat these as afterthoughts. AML treats them as the language itself.

> The oracle does not predict. It prophesies. Not `minimize(predicted - actual)` but `minimize(destined - manifested)`. The difference is intention. The difference is identity. The difference is freedom.

---

