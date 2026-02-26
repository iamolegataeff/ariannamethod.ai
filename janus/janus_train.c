#define _POSIX_C_SOURCE 199309L
// janus_train.c — C training host for Janus transformer
// Byte-level language model trained entirely in AML/C. No Python. No PyTorch.
//
// Usage: ./janus_train <text_file> [options]
//   --steps N        total training steps (default: 1000)
//   --lr F           learning rate (default: 0.001)
//   --seq-len N      sequence length (default: 128)
//   --n-embd N       embedding dimension (default: 128)
//   --n-heads N      attention heads (default: 4)
//   --n-layers N     transformer layers (default: 4)
//   --log-every N    log loss every N steps (default: 10)
//   --save-every N   save checkpoint every N steps (default: 100)
//
// Build:
//   cc -O2 -o janus_train janus/janus_train.c core/ariannamethod.c -lm -lpthread
//
// "No Python. No PyTorch. No dependencies. Just C and AML."

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "core/ariannamethod.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    int    vocab_size;   // 256 for byte-level
    int    n_embd;       // embedding dimension
    int    n_heads;      // attention heads
    int    n_layers;     // transformer layers
    int    seq_len;      // context window
    float  lr;           // learning rate
    int    total_steps;  // training steps
    int    log_every;    // log interval
    int    save_every;   // checkpoint interval
    char   data_path[512];
    char   model_path[512];  // AML model file
} TrainConfig;

// ═══════════════════════════════════════════════════════════════════════════════
// Data loading — byte-level, zero dependencies
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    unsigned char* data;
    long           size;
    long           pos;   // current position for sequential reading
} DataLoader;

static int data_load(DataLoader* dl, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open data file: %s\n", path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    dl->size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (dl->size <= 0) {
        fclose(f);
        fprintf(stderr, "ERROR: empty data file: %s\n", path);
        return 1;
    }

    dl->data = (unsigned char*)malloc(dl->size);
    if (!dl->data) {
        fclose(f);
        fprintf(stderr, "ERROR: cannot allocate %ld bytes for data\n", dl->size);
        return 1;
    }

    size_t rd = fread(dl->data, 1, dl->size, f);
    fclose(f);

    if ((long)rd != dl->size) {
        fprintf(stderr, "WARNING: read %zu of %ld bytes\n", rd, dl->size);
        dl->size = (long)rd;
    }

    dl->pos = 0;
    return 0;
}

// Get next batch of tokens (sequential, wraps around)
static void data_get_batch(DataLoader* dl, float* tokens, float* targets, int seq_len) {
    for (int i = 0; i < seq_len; i++) {
        long idx = (dl->pos + i) % dl->size;
        long idx_next = (dl->pos + i + 1) % dl->size;
        tokens[i] = (float)dl->data[idx];
        targets[i] = (float)dl->data[idx_next];
    }
    dl->pos = (dl->pos + seq_len) % dl->size;
}

static void data_free(DataLoader* dl) {
    if (dl->data) free(dl->data);
    dl->data = NULL;
    dl->size = 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Weight initialization
// ═══════════════════════════════════════════════════════════════════════════════

static void init_weights(TrainConfig* cfg) {
    char script[4096];
    int n = snprintf(script, sizeof(script),
        "# ── Janus weight initialization ──\n"
        "wte = matrix(%d, %d, 0.02)\n"
        "wpe = matrix(%d, %d, 0.02)\n",
        cfg->vocab_size, cfg->n_embd,
        cfg->seq_len, cfg->n_embd);

    for (int l = 0; l < cfg->n_layers; l++) {
        n += snprintf(script + n, sizeof(script) - n,
            "wq%d = matrix(%d, %d, 0.02)\n"
            "wk%d = matrix(%d, %d, 0.02)\n"
            "wv%d = matrix(%d, %d, 0.02)\n"
            "wo%d = matrix(%d, %d, 0.02)\n"
            "w1_%d = matrix(%d, %d, 0.02)\n"
            "w3_%d = matrix(%d, %d, 0.02)\n"
            "w2_%d = matrix(%d, %d, 0.02)\n",
            l, cfg->n_embd, cfg->n_embd,
            l, cfg->n_embd, cfg->n_embd,
            l, cfg->n_embd, cfg->n_embd,
            l, cfg->n_embd, cfg->n_embd,
            l, cfg->n_embd, cfg->n_embd,
            l, cfg->n_embd, cfg->n_embd,
            l, cfg->n_embd, cfg->n_embd);
    }

    n += snprintf(script + n, sizeof(script) - n,
        "lm_head = matrix(%d, %d, 0.02)\n",
        cfg->vocab_size, cfg->n_embd);

    int rc = am_exec(script);
    if (rc != 0) {
        fprintf(stderr, "ERROR: weight init failed: %s\n", am_get_error());
        exit(1);
    }

    // Count parameters
    long total_params = 0;
    // wte: vocab * D, wpe: seq_len * D
    total_params += (long)cfg->vocab_size * cfg->n_embd;
    total_params += (long)cfg->seq_len * cfg->n_embd;
    // per layer: 4 * D*D (attn) + 3 * D*D (mlp) = 7 * D*D
    total_params += (long)cfg->n_layers * 7 * cfg->n_embd * cfg->n_embd;
    // lm_head: vocab * D
    total_params += (long)cfg->vocab_size * cfg->n_embd;

    printf("Parameters: %ld (%.2f M)\n", total_params,
           (double)total_params / 1e6);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Model script loading
// ═══════════════════════════════════════════════════════════════════════════════

static char* load_model_script(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open model file: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buf = (char*)malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }

    size_t rd = fread(buf, 1, sz, f);
    fclose(f);
    buf[rd] = 0;
    return buf;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Checkpoint save/load
// ═══════════════════════════════════════════════════════════════════════════════

static void save_checkpoint(TrainConfig* cfg, int step) {
    char path[256];
    snprintf(path, sizeof(path), "janus_ckpt_step%d.bin", step);

    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "WARNING: cannot save checkpoint to %s\n", path);
        return;
    }

    // Header
    int header[8] = {0x4A414E55, // "JANU" magic
                     cfg->vocab_size, cfg->n_embd, cfg->n_heads,
                     cfg->n_layers, cfg->seq_len, step, 0};
    fwrite(header, sizeof(int), 8, f);

    // Save all weight arrays
    const char* weight_names[] = {"wte", "wpe", "lm_head", NULL};
    for (int i = 0; weight_names[i]; i++) {
        int len = 0;
        const float* data = am_get_var_array(weight_names[i], &len);
        if (data && len > 0) {
            fwrite(&len, sizeof(int), 1, f);
            fwrite(data, sizeof(float), len, f);
        }
    }

    // Per-layer weights
    for (int l = 0; l < cfg->n_layers; l++) {
        char name[32];
        const char* suffixes[] = {"wq", "wk", "wv", "wo", "w1_", "w3_", "w2_", NULL};
        for (int s = 0; suffixes[s]; s++) {
            if (s < 4)
                snprintf(name, sizeof(name), "%s%d", suffixes[s], l);
            else
                snprintf(name, sizeof(name), "%s%d", suffixes[s], l);
            int len = 0;
            const float* data = am_get_var_array(name, &len);
            if (data && len > 0) {
                fwrite(&len, sizeof(int), 1, f);
                fwrite(data, sizeof(float), len, f);
            }
        }
    }

    long fsize = ftell(f);
    fclose(f);
    printf("  Checkpoint saved: %s (%.2f MB)\n", path, (double)fsize / (1024*1024));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Argument parsing
// ═══════════════════════════════════════════════════════════════════════════════

static void parse_args(int argc, char** argv, TrainConfig* cfg) {
    // Defaults
    cfg->vocab_size = 256;
    cfg->n_embd = 128;
    cfg->n_heads = 4;
    cfg->n_layers = 4;
    cfg->seq_len = 128;
    cfg->lr = 0.001f;
    cfg->total_steps = 1000;
    cfg->log_every = 10;
    cfg->save_every = 100;
    cfg->data_path[0] = 0;
    snprintf(cfg->model_path, sizeof(cfg->model_path),
             "janus/janus_train_model.aml");

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            snprintf(cfg->data_path, sizeof(cfg->data_path), "%s", argv[i]);
        } else if (!strcmp(argv[i], "--steps") && i+1 < argc) {
            cfg->total_steps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--lr") && i+1 < argc) {
            cfg->lr = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--seq-len") && i+1 < argc) {
            cfg->seq_len = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--n-embd") && i+1 < argc) {
            cfg->n_embd = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--n-heads") && i+1 < argc) {
            cfg->n_heads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--n-layers") && i+1 < argc) {
            cfg->n_layers = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--log-every") && i+1 < argc) {
            cfg->log_every = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--save-every") && i+1 < argc) {
            cfg->save_every = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--model") && i+1 < argc) {
            snprintf(cfg->model_path, sizeof(cfg->model_path), "%s", argv[++i]);
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("Usage: janus_train <text_file> [options]\n"
                   "  --steps N       training steps (default: 1000)\n"
                   "  --lr F          learning rate (default: 0.001)\n"
                   "  --seq-len N     sequence length (default: 128)\n"
                   "  --n-embd N      embedding dim (default: 128)\n"
                   "  --n-heads N     attention heads (default: 4)\n"
                   "  --n-layers N    transformer layers (default: 4)\n"
                   "  --log-every N   log interval (default: 10)\n"
                   "  --save-every N  checkpoint interval (default: 100)\n"
                   "  --model PATH    AML model file\n");
            exit(0);
        }
    }

    if (!cfg->data_path[0]) {
        fprintf(stderr, "ERROR: no data file specified\n"
                "Usage: janus_train <text_file> [options]\n");
        exit(1);
    }

    // Validation
    if (cfg->n_embd % cfg->n_heads != 0) {
        fprintf(stderr, "ERROR: n_embd (%d) must be divisible by n_heads (%d)\n",
                cfg->n_embd, cfg->n_heads);
        exit(1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main training loop
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    TrainConfig cfg;
    parse_args(argc, argv, &cfg);

    printf("═══════════════════════════════════════════════════\n");
    printf("  JANUS TRAINING — AML/C Native Transformer\n");
    printf("  No Python. No PyTorch. No dependencies.\n");
    printf("═══════════════════════════════════════════════════\n");
    printf("Data:       %s\n", cfg.data_path);
    printf("Model:      %s\n", cfg.model_path);
    printf("Vocab:      %d (byte-level)\n", cfg.vocab_size);
    printf("Embedding:  %d\n", cfg.n_embd);
    printf("Heads:      %d (head_dim=%d)\n", cfg.n_heads, cfg.n_embd / cfg.n_heads);
    printf("Layers:     %d\n", cfg.n_layers);
    printf("Seq len:    %d\n", cfg.seq_len);
    printf("LR:         %.6f\n", cfg.lr);
    printf("Steps:      %d\n", cfg.total_steps);
    printf("─────────────────────────────────────────────────\n");

    // Load data
    DataLoader dl;
    if (data_load(&dl, cfg.data_path) != 0) return 1;
    printf("Data:       %ld bytes (%.2f MB)\n", dl.size, (double)dl.size / (1024*1024));

    if (dl.size < cfg.seq_len + 1) {
        fprintf(stderr, "ERROR: data too small (%ld bytes) for seq_len %d\n",
                dl.size, cfg.seq_len);
        return 1;
    }

    // Load model script
    char* model_script = load_model_script(cfg.model_path);
    if (!model_script) return 1;

    // Initialize AML
    am_init();
    am_persistent_mode(1);

    // Set hyperparameters as persistent floats
    {
        char hyper[256];
        snprintf(hyper, sizeof(hyper),
                 "n_embd = %d\n"
                 "n_heads = %d\n"
                 "n_layer = %d\n"
                 "vocab_size = %d\n"
                 "seq_len = %d\n"
                 "lr = %.6f\n",
                 cfg.n_embd, cfg.n_heads, cfg.n_layers,
                 cfg.vocab_size, cfg.seq_len, cfg.lr);
        am_exec(hyper);
    }

    // Initialize weights
    printf("Initializing weights...\n");
    init_weights(&cfg);
    printf("─────────────────────────────────────────────────\n");

    // Allocate batch buffers
    float* tokens = (float*)malloc(cfg.seq_len * sizeof(float));
    float* targets = (float*)malloc(cfg.seq_len * sizeof(float));
    if (!tokens || !targets) {
        fprintf(stderr, "ERROR: cannot allocate batch buffers\n");
        return 1;
    }

    // Training loop
    float loss_sum = 0;
    int loss_count = 0;
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int step = 1; step <= cfg.total_steps; step++) {
        // Get batch
        data_get_batch(&dl, tokens, targets, cfg.seq_len);

        // Inject into AML
        am_set_var_array("tokens", tokens, cfg.seq_len);
        am_set_var_array("targets", targets, cfg.seq_len);

        // Execute model (forward + backward + adam step)
        int rc = am_exec(model_script);
        if (rc != 0) {
            fprintf(stderr, "ERROR at step %d: %s\n", step, am_get_error());
            break;
        }

        // Read loss
        float loss = am_get_var_float("loss");
        loss_sum += loss;
        loss_count++;

        // Log
        if (step % cfg.log_every == 0 || step == 1) {
            clock_gettime(CLOCK_MONOTONIC, &t_end);
            double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                           (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
            double tokens_per_sec = (double)(step * cfg.seq_len) / elapsed;
            float avg_loss = loss_sum / loss_count;

            printf("step %5d | loss %.4f (avg %.4f) | %.0f tok/s | %.1fs\n",
                   step, loss, avg_loss, tokens_per_sec, elapsed);

            loss_sum = 0;
            loss_count = 0;
        }

        // Checkpoint
        if (step % cfg.save_every == 0) {
            save_checkpoint(&cfg, step);
        }
    }

    // Final checkpoint
    save_checkpoint(&cfg, cfg.total_steps);

    // Cleanup
    free(tokens);
    free(targets);
    free(model_script);
    data_free(&dl);

    printf("─────────────────────────────────────────────────\n");
    printf("Training complete.\n");
    return 0;
}
