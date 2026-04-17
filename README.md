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
node run-calc.js bartowski/Llama-8B-Instruct-GGUF --ctx 8192 --kvTypeK Q8_0
node run-calc.js --batch testmodels.list
```

Options: `--ctx N`, `--batchSize N`, `--kvTypeK TYPE`, `--kvTypeV TYPE`. Batch file has one HF repo per line.

## What it calculates

| Component | Where | Details |
|-----------|-------|---------|
| **Model weights** | Dense: VRAM / MoE: VRAM (active) + RAM (inactive) | Sum of all tensor sizes from GGUF metadata, exact bytes-per-element per quantization type |
| **KV cache** | VRAM | Separate K and V quantization. `layers × kv_heads × head_size × context × bytes_per_elem` per cache |
| **Activations** | VRAM | `layers × batch × context × (hidden + ff_size) × 4` bytes (FP32). For MoE, uses `expert_used_count × expert_ff_size` |

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

llama, mistral, qwen2/3/3.5/3next, phi3, gemma2/3, olmo2, glm4, falcon-h1, cohere2, smollm3, ernie4_5, grok, nemotron_h, lfm2, minimax-m2, seed_oss, apertus, dots1, flux, ltxv, lumina2, qwen_image, wan, mimo2, hunyuan-dense, qwen3vlmoe, ernie4_5_moe, hunyuan_moe, bailingmoe2, deepseek2 (MLA), gemma4 (ISWA), gpt-oss (ISWA), llama4 (ISWA)

## Quantization types

34 types supported: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, IQ1_M, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, I8, I16, I32, I64, F64

## See also

- [`AGENTS.md`](AGENTS.md) — Developer notes: adding architectures, BigInt gotchas, tensor matching patterns
