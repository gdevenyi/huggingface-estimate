# GGUF Memory Estimator

Estimate VRAM and RAM requirements for any GGUF model from HuggingFace — no download required.

## Quick start

### Browser

```bash
cd huggingface-estimate
python3 -m http.server 8000
```

Open `http://localhost:8000`. Enter a HuggingFace path (e.g. `bartowski/Llama-3.1-8B-Instruct-GGUF`) and the tool finds the `.gguf` files, parses metadata via HTTP Range requests, and computes memory estimates.

**Note:** Must be served via HTTP — `file://` blocks the CORS headers needed to fetch GGUF metadata from HuggingFace.

### CLI

```bash
node run-calc.js bartowski/Llama-3.1-8B-Instruct-GGUF --ctx 8192 --kvTypeK Q8_0
node run-calc.js unsloth/Qwen3-30B-A3B-GGUF --ctx 8192 --vram 24 --ram 64
node run-calc.js --batch testModels.list
```

Options: `--ctx N`, `--batchSize N`, `--kvTypeK TYPE`, `--kvTypeV TYPE`, `--vram GB`, `--ram GB`, `--mmproj FILE`, `--mmprojDevice vram|ram`. Passing `--vram` / `--ram` adds a fit check to the JSON output. Batch file has one HF repo per line (`#` for comments). KV cache types include F16, F32, BF16, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1, Q8_KV, Q8_KV_R8, plus rotorquant (TURBO2_0/3_0/4_0, PLANAR3_0/4_0, ISO3_0/4_0).

## What it calculates

| Component | Where | Details |
|-----------|-------|---------|
| **Model weights** | Dense: VRAM / MoE: VRAM (active) + RAM (inactive) | Sum of all tensor sizes from GGUF metadata, exact bytes-per-element per quantization type |
| **KV cache** | VRAM | Separate K and V quantization. `layers × kv_heads × head_size × context × bytes_per_elem` per cache |
| **Activations** | VRAM | `layers × batch × (hidden + ff_size) × 4` bytes (FP32). For MoE, uses `expert_used_count × expert_ff_size` |
| **Multimodal projector (mmproj)** | VRAM default, RAM with `--no-mmproj-offload` | Weights + per-image output activation (`n_output_tokens × projection_dim × 4` bytes). `n_output_tokens` is arch-specific, mirroring `clip_n_output_tokens()` from llama.cpp |

## Multimodal projector (mmproj) support

When a HuggingFace repo contains both a language-model GGUF and an `mmproj-*.gguf` companion (vision/audio encoder), the tool detects and lists mmproj files separately from the main-model dropdown. Pick "None" to ignore, or select one to fold its memory contribution into the VRAM or RAM total (toggle via the placement select). The CLI exposes the same via `--mmproj <filename>` and `--mmprojDevice vram|ram`. Detection is filename-based (`/mmproj/i` on basename). Activation output tokens are estimated per projector type (gemma3, qwen2vl_merger, resampler, etc.); audio projectors report weights only because their patch count depends on runtime audio length.

## Dense vs MoE

- **Dense**: all weights + KV cache + activations in VRAM
- **MoE**: only active experts in VRAM, inactive experts in RAM. KV cache and activations always in VRAM

## Example: Mixtral 8x7B (MoE)

| | Size |
|---|---|
| Total parameters | 467B |
| **Active parameters** | **12.3B** |
| In VRAM (Q4_K + F16 KV) | ~10 GB |
| In RAM (inactive experts) | ~37 GB |
| KV cache (4096 ctx) | ~2.5 GB |

The model fits on a 24GB GPU even though total weight size is ~22GB — only the active experts need to be in VRAM.

## Supported architectures

llama, mistral3/4, qwen2/3/3.5/3next, qwen2vl, qwen3vl, qwen3moe, qwen35moe, qwen3vlmoe, phi3, gemma2/3/3n, gemma-embedding, olmo2, granite, granitehybrid, glm4, glm4moe, glm-dsa, falcon-h1, cohere2, smollm3, ernie4_5, ernie4_5_moe, grok, nemotron_h, nemotron_h_moe, lfm2, lfm2_moe, minimax-m2, seed_oss, apertus, dots1, afmoe, deci, flux, ltxv, lumina2, qwen_image, wan, mimo2, hunyuan-dense, hunyuan_moe, bailingmoe2, deepseek2 (MLA), gemma4 (ISWA), gpt-oss (ISWA), llama4 (ISWA)

## Quantization types

101 types supported (34 standard + 60 from [ik_llama.cpp](ik_llama.cpp) + 7 rotorquant KV cache):

**Standard:** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, IQ1_M, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, I8, I16, I32, I64, F64

**ik_llama.cpp extensions:** I2_S (MS BitNet ternary), Q6_0, Q8_KV, Q8_KV_R8, Q8_K_R16 (KV cache), Q8_K64, Q8_K16, Q8_K32, Q8_KR8, Q8_K128, Q8_K_R8, Q8_0_X4, Q8_1_X4, Q8_2_X4, Q4_0_4_4, Q4_0_4_8, Q4_0_8_8, IQ1_BN, IQ2_BN, IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K, IQ4_KS, IQ2_KS, IQ4_KSS, IQ5_KS, IQ2_KT, IQ3_KT, IQ4_KT, IQ3_KS, IQ2_KL, IQ1_KT, Q4_0_R8, Q5_0_R4, Q8_0_R8, Q2_K_R4, Q3_K_R4, Q4_K_R4, Q5_K_R4, Q6_K_R4, IQ2_XXS_R4, IQ2_XS_R4, IQ3_XXS_R4, IQ1_S_R4, IQ4_NL_R4, IQ3_S_R4, IQ2_S_R4, IQ4_XS_R8, IQ1_M_R4, Q6_0_R4, IQ2_BN_R4, IQ2_K_R4, IQ3_K_R4, IQ4_K_R4, IQ5_K_R4, IQ4_KS_R4, IQ5_KS_R4, BF16_R16

**rotorquant KV cache:** TURBO2_0, TURBO3_0, TURBO4_0, PLANAR3_0, PLANAR4_0, ISO3_0, ISO4_0

## See also

- [`AGENTS.md`](AGENTS.md) — Developer notes: adding architectures, BigInt gotchas, tensor matching patterns
