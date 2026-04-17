# GGUF Memory Estimator

Estimate VRAM and RAM requirements for any GGUF model from HuggingFace — no download required.

## Features

- **Resolve any HF model** — enter a HuggingFace path (e.g. `bartowski/Llama-3.1-8B-Instruct-GGUF`) and the tool finds the `.gguf` files automatically
- **Parse GGUF metadata remotely** — reads only the first ~2MB of the file via HTTP Range requests; never downloads the full model
- **Separate K/V cache quantization** — choose independent quantization for K and V caches (F16, Q8_0, Q4_0, Q4_K, etc.)
- **Dense vs MoE handling** — correctly estimates memory for both architectures:
  - **Dense**: all weights + KV cache + activations in VRAM
  - **MoE**: only active experts in VRAM, inactive experts in RAM
- **34 quantization types** — exact bytes-per-element for all GGML quantization schemes (Q4_0 through NVFP4)
- **Context length clamping** — auto-limits context to the model's maximum supported length
- **VRAM fit check** — color-coded bar showing whether your GPU can run the model

## Usage

```bash
cd huggingface-estimate
python3 -m http.server 8000
```

Then open `http://localhost:8000` in a browser.

**Note:** Must be served via HTTP — the `file://` protocol blocks the CORS headers needed to fetch GGUF metadata from HuggingFace.

## What it calculates

| Component | Where | Details |
|-----------|-------|---------|
| **Model weights** | Dense: VRAM / MoE: VRAM (active) + RAM (inactive) | Sum of all tensor sizes from GGUF metadata, using exact bytes-per-element per quantization type |
| **KV cache** | VRAM | Separate K and V quantization. `layers × kv_heads × head_size × context × bytes_per_elem` per cache |
| **Activations** | VRAM | `layers × batch × context × (hidden + ff_size) × 4` bytes (FP32). For MoE, uses `expert_used_count × expert_ff_size` |

## Architecture

Single `index.html` file — no build step, no dependencies. Uses [`@huggingface/gguf`](https://www.npmjs.com/package/@huggingface/gguf) from CDN for GGUF parsing.

- **HF path resolution** — queries `/api/models/{path}` to find `.gguf` files
- **GGUF parsing** — `gguf(url)` fetches the header + metadata + tensor info via Range requests
- **Weight calculation** — iterates `tensorInfos`, multiplies shape product by bytes-per-element from `GGML_QUANT_SIZES`
- **MoE detection** — checks `expert_count > 0`, groups `*_exps.*` tensors, separates active vs inactive

## Quantization types supported

F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, IQ1_M, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, I8, I16, I32, I64, F64

## Example: Mixtral 8x7B (MoE)

| | Size |
|---|---|
| Total parameters | 467B |
| **Active parameters** | **12.3B** |
| In VRAM (Q4_K + F16 KV) | ~10 GB |
| In RAM (inactive experts) | ~37 GB |
| KV cache (4096 ctx) | ~2.5 GB |

The model fits on a 24GB GPU even though total weight size is ~22GB — because only the active experts need to be in VRAM.
