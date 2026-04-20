# AGENTS.md

## Structure

No build step, no framework. ESM throughout (`package.json` has `"type": "module"`).

**Single source of truth** — one copy of `calculations.js` and `parsing.js` runs in both browser and Node:
- `index.html` — UI: HTML/CSS, result rendering. Declares an `<script type="importmap">` that remaps the bare specifier `@huggingface/gguf` to the jsDelivr CDN URL.
- `calculations.js` — Architecture registry, KV cache, activations, MoE, weight calculations. Imports `@huggingface/gguf` as a bare specifier.
- `parsing.js` — GGUF metadata parsing + HF URL resolution. Same import convention.
- `run-calc.js` — Node CLI entry point. Resolves the bare specifier via Node's normal package lookup (`node_modules/@huggingface/gguf`).

**Import map requirement**: the `<script type="importmap">` in `index.html` must be emitted before any `<script type="module">`. Supported in Chromium ≥89, Firefox ≥108, Safari ≥16.4.

**Gitignored reference dirs**: `llama.cpp/`, `ik_llama.cpp/`, `gguf-parser-go/` — local clones for quantization type reference, not part of the app.

## Must serve via HTTP

`file://` blocks CORS headers needed to fetch GGUF metadata from HuggingFace:
```bash
python3 -m http.server 8000
```
Then visit `http://localhost:8000`.

## CLI usage

```bash
node run-calc.js bartowski/Llama-3.1-8B-Instruct-GGUF --ctx 8192 --kvTypeK Q8_0
node run-calc.js --batch testModels.list
```

Options: `--ctx N` (default 4096), `--batchSize N` (default 1), `--kvTypeK TYPE` (default F16), `--kvTypeV TYPE` (default F16), `--mmproj FILE`, `--mmprojDevice vram|ram` (default vram). Batch file has one HF repo per line, `#` comments supported. Outputs JSON to stdout, progress to stderr.

Performance flags (add a `performance` block to the JSON): `--gpu <name|id>` (fuzzy-matches `gpu-data.json`), `--cpu <name|id>` (matches `hardware-presets.js`), `--gpu-flops`, `--gpu-bw`, `--cpu-flops`, `--ram-bw` for manual overrides, `--ngl <n|auto>`, `--cpu-moe`, `--n-cpu-moe N`. Omit all GPU flags to disable the performance block (preserves the pre-feature output shape).

## BigInt gotcha

`@huggingface/gguf` returns `tensorInfos[].shape` as `bigint[]`. Never multiply a `bigint` by a `number` (BPE values are `number`). Always convert first:
```js
// WRONG: t.shape.reduce((a, b) => a * Number(b), 1n)
// CORRECT: t.shape.map(Number).reduce((a, b) => a * b, 1)
```

## HF URL normalization

HF's `/blob/` endpoint lacks CORS headers. Always normalize:
```js
url = path.replace(/\/blob\//, '/resolve/').replace(/#.*$/, '');
```

## MoE VRAM/RAM split

Two views, both matching llama.cpp:

1. **`calcMemoryBreakdown` (top "VRAM/RAM" card)** — ideal placement assuming the model fully fits, used to compute the `vramBytes` displayed.
   - Dense: all weights + KV + activations → VRAM
   - MoE default (no flags): all expert weights in VRAM
   - `--cpu-moe`: all expert weights → RAM, rest → VRAM
   - `--n-cpu-moe N`: expert weights for layers 0..N-1 → RAM
2. **`computeOffloadSplit` / `calcActualMemory` (fit check, perf)** — actual placement given a finite VRAM budget. Mirrors llama.cpp's `--fit on` algorithm in `src/llama.cpp` (`llama_params_fit_impl`):
   - Layer modes: `gpu` (full layer in VRAM, including all experts), `hybrid` (non-expert + KV in VRAM, experts in RAM, expert matmul on CPU — llama.cpp's "dense-only" layer), `cpu` (everything on CPU).
   - Layers offloaded back-to-front (`i_gpu_start = max(n_layer + 1 - n_gpu_layers, 0)`).
   - **Two-pass auto-fit always runs for MoE models** (regardless of `--cpu-moe` / `--n-cpu-moe`):
     - Pass 1: fill all layers dense-only back-to-front (matches llama.cpp step 3).
     - Pass 2: convert dense-only layers to full front-to-back, skipping layers forced hybrid by `--cpu-moe` (all expert layers) or `--n-cpu-moe N` (layers 0..N-1) (matches llama.cpp step 4).
   - With a manual `--ngl N`: last N layers placed back-to-front, then expert-placement overrides applied. No auto-fit.
- KV cache always per-layer (lives wherever the layer lives).
- Performance: `activeExpertWeightBytes` (active fraction = `expertUsedCount / expertCount`) drives per-token bandwidth/compute regardless of where experts are stored — sparse MoE only reads active experts each step.

## Bytes-per-element hardcoded

`GGML_QUANT_SIZES` is NOT exported from the browser build. BPE values are hardcoded as the `BPE` object in `calculations.js`. Standard types use `GGMLQuantizationType` enum keys as indices; ik_llama.cpp extensions use numeric IDs (e.g., `151` for Q8_KV). The `BPE` object is the sole source of truth for bytes-per-element.

## CDN version pin

The importmap in `index.html` pins `@huggingface/gguf` to `https://cdn.jsdelivr.net/npm/@huggingface/gguf@0.4.2/+esm`. Node loads the same version from `package.json`. Check updates at `https://data.jsdelivr.com/v1/package/npm/@huggingface/gguf`.

## Sharded GGUF

Files matching `*-of-*.gguf` are auto-detected as shards in `parseGGUF()`. Calls `ggufAllShards()`, merges tensor infos from all shards, takes metadata from the first shard.

## Multimodal projector (mmproj)

`resolveHFModel()` partitions the repo's `.gguf` siblings by filename: anything matching `/mmproj/i` on the basename goes to `mmProjFiles`, the rest to `ggufFiles`. Detection is filename-only (no metadata prefetch). A repo containing only mmproj files throws — nothing to estimate. `buildResolveUrl()` is exported for constructing the mmproj download URL.

`calcMmProj(metadata, tensorInfos)` in `calculations.js` returns `null` if the parsed GGUF is not a CLIP/mtmd projector (checks `clip.has_vision_encoder`, `clip.has_audio_encoder`, `general.architecture === 'clip'`). Otherwise it computes:

- **Weights** — delegated to `calcWeightSize()`, identical to the main-model weight math.
- **Per-image output activation** — `n_output_tokens × projection_dim × 4` bytes (fp32). `n_output_tokens` mirrors `clip_n_output_tokens()` at `llama.cpp/tools/mtmd/clip.cpp:2829` via `estimateOutputTokens(projType, ...)`. Projector types map to a handful of formulas keyed on `clip.vision.image_size`, `clip.vision.patch_size`, `clip.vision.spatial_merge_size`, and (for resampler) `clip.minicpmv_query_num` / `clip.minicpmv_version`. Audio projectors (ultravox, voxtral, qwen2a, etc.) return 0 — their patch count depends on runtime audio length.
- **Metadata keys read**: `clip.has_vision_encoder`, `clip.has_audio_encoder`, `clip.projector_type` (plus `clip.vision.projector_type` / `clip.audio.projector_type` fallbacks), `clip.vision.image_size`, `clip.vision.patch_size`, `clip.vision.embedding_length`, `clip.vision.block_count`, `clip.vision.projection_dim`, `clip.vision.spatial_merge_size`, `clip.minicpmv_query_num`, `clip.minicpmv_version`. Source of truth: `llama.cpp/tools/mtmd/clip-impl.h`.

**Placement** — `--mmprojDevice vram` (default) mirrors llama.cpp's `mmproj_use_gpu=true`; `ram` mirrors `--no-mmproj-offload` (see `llama.cpp/common/common.h:544`). When selected, the combined `weightBytes + perImageActBytes` is folded into either `vramBytes` or `ramBytes`.

## Adding a new architecture

Add to the `ARCHITECTURES` registry in `calculations.js`. Each entry declares categories and provides handlers for KV cache, activations, and MoE weights.

### Step 1: Identify the architecture

```bash
node --experimental-vm-modules -e "
import { gguf } from '@huggingface/gguf';
const r = await gguf('https://huggingface.co/owner/model/resolve/main/model.gguf');
console.log(r.metadata['general.architecture']);
console.log(Object.keys(r.metadata).filter(k => k.startsWith(r.metadata['general.architecture'] + '.')));
"
```

### Step 2: Determine categories

| Category | Trigger | What changes |
|----------|---------|-------------|
| `mla` | Has `attention.kv_lora_rank` + `attention.key_length_mla` | KV cache uses compressed latent dimensions; activations use `q_lora_rank` + `kv_lora_rank` |
| `iswa` | Has `attention.sliding_window` or per-layer `head_count_kv` array | Per-layer GQA; SWA layers use `min(sliding_window, ctxSize)` |
| `moe` | Has `expert_count > 0` | Expert tensor grouping, VRAM/RAM split for inactive experts |
| `vl` / `embedding` / `diffusion` | Model type markers | Calculation-only markers, no handler changes |

### Step 3: Check for aliases

If the GGUF-returned architecture name differs from the registry key (e.g., hyphens vs underscores), add a mapping to `ARCH_ALIASES`:
```js
ARCH_ALIASES = {
  'ernie4_5-moe': 'ernie4_5_moe',
  'hunyuan-moe':  'hunyuan_moe',
  'lfm2moe':      'lfm2_moe',
}
```

### Step 4: Add registry entry

Handlers are assembled from shared builders near the top of `calculations.js`. For a standard transformer, point at the canonical triple:

```js
myarch: {
  name: 'myarch',
  categories: ['transformer', 'moe'],
  kvCache: llamaKvCache,
  activations: buildActivations,
  moe: moeNoShared,           // or moeShexpOnly, or llamaMoe
  tensorGroups: LLAMA_TENSOR_GROUPS,
},
```

The available builders:

| Builder | Use for |
|---------|---------|
| `llamaKvCache` | Standard (and GQA) KV cache |
| `buildKvCache(meta, ctx, kK, kV, opts)` | ISWA, per-layer filter, effective-layer override, etc. |
| `mlaKvCache` | DeepSeek2 / GLM-DSA latent attention |
| `buildActivations` | Standard transformer activations (incl. MoE-gated) |
| `leadingDenseActivations` | MoE with `leading_dense_block_count` |
| `buildMoe(meta, ti, predicates)` | MoE tensor accounting with custom `isExpert`/`isRouter`/`isShared` |
| `llamaMoe` | Default MoE with `_exps.` / `ffn_gate_inp` / `_shexp.|_chexp.` |
| `moeNoShared` | MoE with no shared experts |
| `moeShexpOnly` | MoE with `_shexp.` shared experts only (no `_chexp.`) |

### Key patterns for tensor matching

- **Expert weights**: `_exps.` in name
- **Router/gate**: `ffn_gate_inp` or `attn_sinks` or `altup_router`
- **Shared experts**: `_shexp.` or `_chexp.`
- **MoE bias**: `exp_probs_b`

Patterns use glob → regex conversion via `globMatch()`.

### `buildKvCache` options

All fields optional. Compose them in the handler: `kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { ... })`.

| Option | Use for |
|--------|---------|
| `iswa: true` | Read `sliding_window` / `sliding_window_pattern`, shrink SWA layer contexts |
| `swaPeriodDefault: N` | Fallback SWA period when metadata doesn't supply one (gpt-oss: 2, llama4: 4, gemma3n: 5) |
| `swaDefault: N` | Fallback `sliding_window` value (llama4: 8192) |
| `effectiveLayers(meta, n_block)` | Override iteration count (gemma4 `shared_kv_layers`, gemma3n `layer_kv_from_start`) |
| `layerFilter(i)` | Skip layers (qwen35moe keeps only every `full_attention_interval`-th layer) |

Per-layer `head_count_kv == 0` is treated as "no KV cache on this layer" across all options. This handles hybrid recurrent/attention (`lfm2_moe`, `nemotron_h_moe`) without extra config — just declare `kvCache: llamaKvCache`.

### Bespoke activations

Architectures with attention shapes that don't match `buildActivations` or `leadingDenseActivations` keep inline handlers:

- **deepseek2**: MLA attention output uses `q_lora_rank + kv_lora_rank`, not `n_embd`.
- **qwen35moe**: residual + shared + routed experts — `2 * n_embd + expertUsedCount * expertFF`.
- **glm-dsa**: adds `indexerTopK * 256` for the sparse indexer state.
- **gemma3n**: multiplies `n_embd` by `altup_num_inputs` (typically 4).

When adding one of these, write the body inline in the registry entry rather than extending `buildActivations` with more options.

### Step 5: Test

```bash
node --experimental-vm-modules -e "
import { gguf } from '@huggingface/gguf';
const r = await gguf('https://huggingface.co/owner/model/resolve/main/model.gguf');
const arch = r.metadata['general.architecture'];
console.log('Arch:', arch);
console.log('Expert tensors:', r.tensorInfos.filter(t => t.name.includes('_exps.')).length);
console.log('Router tensors:', r.tensorInfos.filter(t => t.name.includes('ffn_gate_inp')).length);
"
```

Then open in browser and load the model to verify.

### Fallback

Unknown architectures fall back to the `llama` handler. A warning logs to the console.

## Hardware presets (tokens/sec estimator)

Three sources feed the estimator:

- **`gpu-data.json`** — generated from `gpu_1986-2026.csv` by `scripts/build-gpu-list.js`. Regenerate after editing the CSV: `node scripts/build-gpu-list.js`. Filters to NVIDIA 1000-series+, AMD RX 5000+/MI, Intel Arc. Apple M-series GPU isn't in the CSV uniformly — the per-CPU `defaultRamBwGBps` value covers unified memory.
- **`hardware-presets.js`** — hand-curated `CPU_PRESETS` and `RAM_PRESETS`. CPU FP16 TFLOPS = `cores × boost_GHz × fp16_per_cycle` (AVX2 → 16, AVX-512 → 32, NEON → 8). Pessimistic vs. actual tensor-optimized kernels but bandwidth is typically the decode bottleneck anyway.
- **`calcPerLayerFootprint` / `estimatePerformance`** in `calculations.js` — group tensors by `/^blk\.(\d+)\./`, fold active-expert fraction into per-layer byte totals for MoE layers, then iterate `max(FLOPs/FLOPS, bytes/BW)` per layer. Bottleneck label compares aggregate compute-time vs. BW-time per device side; `cpu-dram-spill` fires when CPU layers exceed 50% of total decode time.

**FP16 rate caveat for data-center cards**: the CSV reports shader-rate FP16 (often 1:64 of tensor-core rate) for H100 / B200. Preprocessor uses `max(BF16, FP16 if ≥ 1.5× FP32, FP32 × 2)` to approximate the tensor-core path, but the result is conservative for Hopper/Blackwell — users can override with `--gpu-flops`.

### Example model and quantization refrence code

The directores `ik_llama.cpp/` `llama.cpp/` and `ik_llama.cpp/` contain variants of the llama.cpp inference engine
with quantization and KV cache quantization implementations. Use these as reference code for proper model calcuations.

The directory `gguf-parser-go/` contains an alternative memory calculator implemenmtation which can be used for
comaparisons. Do not presume it is 100% correct.
