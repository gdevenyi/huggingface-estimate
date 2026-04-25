# AGENTS.md

## Structure

No build step, no framework. ESM throughout (`package.json` has `"type": "module"`).

**Single source of truth** — one copy of `calculations.js` and `parsing.js` runs in both browser and Node:
- `index.html` — UI: HTML/CSS, result rendering. Declares an `<script type="importmap">` that remaps the bare specifier `@huggingface/gguf` to the jsDelivr CDN URL.
- `calculations.js` — Architecture registry, KV cache, activations, MoE, weight calculations, performance estimator. Imports `@huggingface/gguf` as a bare specifier.
- `parsing.js` — GGUF metadata parsing + HF URL resolution + fork detection. Same import convention.
- `run-calc.js` — Node CLI entry point. Resolves the bare specifier via Node's normal package lookup (`node_modules/@huggingface/gguf`).
- `scan-metadata.js` — Node CLI tool that scans GGUF metadata across repos and reports unknown architectures, unhandled metadata keys, and missing BPE entries. Used for regression testing and coverage audits. Supports `--batch`, `--json`, `--summary`, `--unknown-only`, `--concurrency N`.
- `ui.js` — Browser UI logic: preset loading, form handling, result rendering. Loaded as `<script type="module">`.
- `style.css` — Dark-theme-only styles (GitHub-dark palette). Responsive two-column grid, sticky sidebar, SlimSelect overrides.

**Import map requirement**: the `<script type="importmap">` in `index.html` must be emitted before any `<script type="module">`. Supported in Chromium ≥89, Firefox ≥108, Safari ≥16.4.

**Gitignored reference dirs**: `resources/` (entire directory) — contains local clones of llama.cpp forks and vendor hardware CSVs for quantization type reference. Not part of the app. Active forks: `llama.cpp/`, `ik_llama.cpp/`, `llama.cpp-tq3/`, `llama-cpp-turboquant/`, `llama-cpp-rotorquant/`, `gguf-parser-go/`. Hardware data: `gpu_1986-2026.csv`, `apple_silicon_macs.csv`, `amd/`, `intel/`.

**Tracked data**: `specs/` — GPU CSV source data used by build scripts.

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

`QUANT_NAMES` maps each quantization type ID (enum key or numeric string) to a human-readable display name. Auto-populated from `GGMLQuantizationType` plus manual entries for ik_llama.cpp, turboquant, rotorquant, and tq3 types.

All tensor size calculations go through `tensorBpe(t)` which checks for a per-tensor `_bpeOverride` (set by fork detection) before falling back to `BPE[t.dtype]`. Display names similarly use `tensorQuantName(t)` which checks `_nameOverride`.

## Fork-aware BPE overrides

GGUF type IDs 44 and 46 collide between the turboquant and tq3 forks (different types, different BPE). `parseGGUF()` in `parsing.js` detects the source fork via `general.file_type` and tensor dtype presence, then stamps `_bpeOverride` and `_nameOverride` on affected tensors. All downstream size calculations use `tensorBpe(t)` which checks `_bpeOverride` before falling back to `BPE[t.dtype]`.

**Detection heuristics** (first match wins):
1. Any tensor with dtype 200, or `general.file_type == 200` or `45` → tq3 fork
2. Any tensor with dtype 42 or 43 → turboquant fork (TURBO2_0/TURBO3_0 are turboquant-exclusive KV types)
3. `general.file_type == 43` with tensor dtype 44 present → tq3; dtype 45 present → turboquant

**Colliding type IDs**:

| ID | Default (turboquant) | tq3 fork |
|----|----------------------|----------|
| 44 | TURBO4_0 (KV cache, 0.5313 BPE) | TQ3_1S (weight, 0.5 BPE) |
| 45 | TQ3_1S (weight, 0.5 BPE) | TQ3_1S (weight, 0.5 BPE — same) |
| 46 | TQ4_1S (weight, 0.625 BPE) | TQ3_4S (weight, 0.5 BPE) |

ID 200 (TQ3_0, KV-only, 0.4375 BPE) is unique to the tq3 fork and does not collide.

**TQ3 fork types** (`resources/llama.cpp-tq3/`):
| Type | ID | BPE | bpw | KV cache | Weight |
|------|----|-----|-----|----------|--------|
| TQ3_1S | 44 | 0.5 (16/32) | 4.0 | No | Yes |
| TQ3_4S | 46 | 0.5 (16/32) | 4.0 | No | Yes |
| TQ3_0 | 200 | 0.4375 (14/32) | 3.5 | Yes (KV-only) | No |

## Utility exports from `calculations.js`

- `globMatch(pattern, str)` — glob-pattern matching (`*` → `.*` regex) for tensor group matching.
- `getModelArch(metadata)` — returns `general.architecture` from metadata.
- `getMeta(metadata, key, fallback=0)` — safe metadata accessor with numeric coercion.
- `getArchHandler(arch)` — returns architecture handler from registry (with alias resolution + fallback to `llama`).
- `formatBytes(bytes)` / `formatElements(n)` — human-readable formatters (KiB/MiB/GiB/TiB, K/M/B/T).
- `TQ3_FORK_BPE` — BPE override map for tq3 fork collision resolution.
- `TQ3_QUANT_NAMES` — display name overrides for tq3 fork types.
- `calcRecurrentState(metadata, nSeqMax=1)` — computes SSM/recurrent state memory for architectures with recurrent layers. Returns `{ convStateBytes, ssmStateBytes, totalBytes, recurrentLayers, n_embd_r, n_embd_s }` or `null` if the architecture has no SSM hparams. State is F32, indexed by sequence count (not context length). Matches llama.cpp's `n_embd_r()` and `n_embd_s()` in `llama-hparams.cpp`. Supports KDA (Kimi-Linear), RWKV, LFM2 shortconv, and Mamba/Mamba2-style SSM state. Used by pure-recurrent archs (mamba, rwkv*), hybrid attention/SSM archs (jamba, falcon-h1, qwen35/moe, nemotron_h*), and KDA hybrids (kimi-linear).

## Utility exports from `parsing.js`

- `KV_VALID_QUANTS` — union of valid KV cache quantization types across all supported forks. Standard (F32, F16, BF16, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1), ik_llama.cpp (Q6_0 via ID 133, Q8_KV via 151), turboquant (TURBO2_0/3_0/4_0), rotorquant (PLANAR3_0/4_0, ISO3_0/4_0 — also includes turboquant types since rotorquant is a superset), tq3 (TQ3_0). Used to populate KV cache type dropdowns in the UI and validate CLI `--kvTypeK`/`--kvTypeV` args.
- `KV_FORK_GROUPS` — optgroup labels for the UI dropdown. Each fork lists all KV types it supports, excluding the 9 mainline types already shown ungrouped. Rotorquant inherits turboquant's 3 types plus its own 4.
- `parseGGUF(url)` — parses GGUF metadata, detects fork, applies BPE/name overrides. Returns `{ metadata, tensorInfos, fork? }`.
- `resolveHFModel(path)` — resolves HF repo/path to GGUF URLs.
- `buildResolveUrl(path, filename)` — constructs HF resolve URLs.

## CDN version pin

The importmap in `index.html` pins `@huggingface/gguf` to `https://cdn.jsdelivr.net/npm/@huggingface/gguf@0.4.2/+esm`. Node loads the same version from `package.json`. Check updates at `https://data.jsdelivr.com/v1/package/npm/@huggingface/gguf`.

## Sharded GGUF

Files matching `*-of-*.gguf` are auto-detected as shards in `parseGGUF()`. Calls `ggufAllShards()`, merges tensor infos from all shards, takes metadata from the first shard. Fork detection runs after the merge so all shard tensors get overrides.

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
| `recurrent` | Pure SSM/RWKV architectures (mamba, rwkv, etc.) | No KV cache; recurrent state computed via `calcRecurrentState` |
| `audio` | Audio generation architectures (acestep-lm, wavtokenizer-dec) | Calculation-only marker, standard handlers |
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
| `noKvCache` | Encoder-only architectures with no KV cache (gemma-embedding, t5encoder, modern-bert, bert, nomic-bert, neo-bert, jina-bert-v2/v3, eurobert) |
| `llamaKvCache` | Standard (and GQA) KV cache |
| `buildKvCache(meta, ctx, kK, kV, opts)` | ISWA, per-layer filter, effective-layer override, etc. |
| `mlaKvCache` | DeepSeek2 / GLM-DSA latent attention |
| `buildActivations` | Standard transformer activations (incl. MoE-gated) |
| `sharedExpertActivations` | MoE with shared experts that have a separate FFN dim (`expert_shared_feed_forward_length`); residual + shared FFN + routed experts (qwen35moe, qwen3next) |
| `leadingDenseActivations` | MoE with `leading_dense_block_count` |
| `mlaActivations` | MLA attention (non-MoE): uses `q_lora_rank` + `kv_lora_rank` instead of full `n_embd` |
| `buildMoe(meta, ti, predicates)` | MoE tensor accounting with custom `isExpert`/`isRouter`/`isShared` |
| `llamaMoe` | Default MoE with `_exps.` / `ffn_gate_inp` / `_shexp.\|_chexp.` |
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
| `denseFirst: true` | With ISWA, first layer in each period is dense (smallthinker pattern; matches llama.cpp `set_swa_pattern(N, true)`) |
| `swaPeriodDefault: N` | Fallback SWA period when metadata doesn't supply one (gpt-oss: 2, llama4: 4, gemma3n: 5) |
| `swaDefault: N` | Fallback `sliding_window` value (llama4: 8192) |
| `effectiveLayers(meta, n_block)` | Override iteration count (gemma4 `shared_kv_layers`, gemma3n `layer_kv_from_start`, bailingmoe2 `nextn_predict_layers`) |
| `layerFilter(i)` | Skip layers (qwen35moe keeps only every `full_attention_interval`-th layer) |

Per-layer `head_count_kv == 0` is treated as "no KV cache on this layer" across all options. This handles hybrid recurrent/attention (`lfm2_moe`, `nemotron_h_moe`) without extra config — just declare `kvCache: llamaKvCache`.

### Bespoke activations

Architectures with attention shapes that don't match `buildActivations` or `leadingDenseActivations` keep inline handlers:

- **deepseek2**: MLA + MoE with `leading_dense_block_count`. Dense layers use `n_ff`, MoE layers use `expertUsedCount * expertFF`.
- **mistral4**: Same pattern as deepseek2 — MLA + MoE with `leading_dense_block_count`.
- **qwen35moe**: residual + shared + routed experts — `2 * n_embd + expertUsedCount * expertFF`.
- **glm-dsa**: adds `indexerTopK * 256` for the sparse indexer state.
- **gemma3n**: multiplies `n_embd` by `altup_num_inputs` (typically 4).

Note: `sharedExpertActivations` is a reusable builder (not bespoke inline) used by qwen35moe and qwen3next. It handles the residual + shared FFN + routed expert pattern where shared experts may have a different FFN dim.

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

- **Per-vendor GPU presets** — `<vendor>-gpu-presets.json` files loaded at runtime. `nvidia-gpu-presets.json` is generated from `gpu_1986-2026.csv` by `scripts/build-gpu-list.js`. Jetson SoCs with LPDDR5X get `unifiedMemory: true`. `amd-gpu-presets.json` is generated from AMD CSVs by `scripts/build-amd-gpu-list.js` (sections 1–3: discrete GPUs; section 4: consumer APU iGPUs from `Processor Specifications.csv`; section 5: embedded APU iGPUs from `Embedded Processor Specifications.csv`). `intel-gpu-presets.json` is generated from Intel CSVs by `scripts/build-intel-gpu-presets.js`. `apple-gpu-presets.json` is generated from `apple_silicon_macs.csv` by `scripts/build-apple-presets.js`. Regenerate all: `node scripts/build-amd-gpu-list.js && node scripts/build-amd-cpu-presets.js && node scripts/build-intel-gpu-presets.js && node scripts/build-intel-cpu-presets.js && node scripts/build-apple-presets.js && node scripts/build-gpu-list.js`. A merged `gpu-data.json` is also produced for backward compat.

**AMD APU iGPU presets** (sections 4–5 of `build-amd-gpu-list.js`): filter `Graphics Core Count >= 4` / `GPU CU >= 4`, skip entries with no iGPU or "Discrete Graphics Card Required". FP16 TFLOPS = `CUs × boost_GHz × 128 / 1000` (all AMD iGPUs from Vega through RDNA 3.5: 128 FP16 FLOPS/cycle/CU via packed math). Memory bandwidth computed from `System Memory Type` + `System Memory Specification`: handles `256-bit LPDDR5x` / `128-bit LPDDR5x` bus-width patterns, dual-type entries (`DDR5 + LPDDR5X`), and standard channel-based DDR5/DDR4. Default `vramGB`: 64 for Ryzen AI Max series, 32 otherwise (user-adjustable in UI). All entries set `memType: 'Unified'` and `unifiedMemory: true`. Generic GPU names (just "Radeon" after cleaning) are replaced with `"Radeon {CU} CU"`. Cross-CSV dedup skips Embedded CSV entries whose processor base name matches a Processor CSV entry. OEM-only markers (`(OEM Only)`) are stripped from processor names.

**Intel iGPU presets** (section 2 of `build-intel-gpu-presets.js`): all entries from `intel_cpu_specs.csv` with `Xe-cores > 0`. FP16 TFLOPS = `Xe-cores × 128 × gpuClockGHz / 1000`. Memory bandwidth from CPU's `Memory Types` + `Max # of Memory Channels`. Default `vramGB`: 32 (user-adjustable). All entries set `memType: 'Unified'` and `unifiedMemory: true`.
- **Per-vendor CPU presets** — `<vendor>-cpu-presets.json` files loaded at runtime. `intel-cpu-presets.json` is generated from Intel CSVs by `scripts/build-intel-cpu-presets.js`. `amd-cpu-presets.json` is generated from AMD CSVs by `scripts/build-amd-cpu-presets.js`. `apple-cpu-presets.json` is generated from `apple_silicon_macs.csv` by `scripts/build-apple-presets.js`. CPU FP16 TFLOPS = `cores × boost_GHz × fp32_per_cycle` using 2x FMA datapath throughput: Zen 5 (2×512-bit FMA) → 64, Zen 4 (2×256-bit FMA, AVX-512 double-pumped) → 32, Zen 3/2 (2×256-bit FMA) → 32, Zen 1/+ (2×128-bit FMA) → 16; Intel AVX-512 → 32, AVX2 → 16; Apple NEON → 8. Apple CPU FP16 is derived from the CSV's `CPU_FP32_TFLOPS_NEON × 2`. Pessimistic vs. actual tensor-optimized kernels but bandwidth is typically the decode bottleneck anyway.
- **`hardware-presets.js`** — `RAM_PRESETS` (kept here), plus `mergeCpuPresets()`/`mergeGpuPresets()` functions to load the vendor JSON files, and `findCpuPreset()`/`findRamPreset()` lookup functions.
- **`calcPerLayerFootprint` / `estimatePerformance`** in `calculations.js` — group tensors by `/^blk\.(\d+)\./`, fold active-expert fraction into per-layer byte totals for MoE layers, then iterate `max(FLOPs/FLOPS, bytes/BW)` per layer. Bottleneck label compares aggregate compute-time vs. BW-time per device side; `cpu-dram-spill` fires when CPU layers exceed 50% of total decode time.

**FP16 rate caveat for data-center cards**: the CSV reports shader-rate FP16 (often 1:64 of tensor-core rate) for H100 / B200. Preprocessor uses `max(BF16, FP16 if ≥ 1.5× FP32, FP32 × 2)` to approximate the tensor-core path, but the result is conservative for Hopper/Blackwell — users can override with `--gpu-flops`.

### Mobile / server group split

Preset entries carry optional `mobile`, `server`, and `desktop` boolean flags. The UI partitions each vendor's dropdown into up to three `<optgroup>` sections: `"<Vendor>"` (desktop), `"<Vendor> (mobile)"`, and `"<Vendor> (server)"`. Empty groups are suppressed.

Detection by source:
- **AMD CPU**: CSV `Form Factor` column — contains "Laptop" or "Handheld" → `mobile: true`; contains "Desktop" or "Boxed" → eligible for main group; both present → `desktop: true` too (dual-form-factor, appears in both groups). Server CSV entries and EPYC → `server: true`.
- **AMD GPU (consumer)**: name matches `/\d[MS]\b/` or `/(Mobile)/` → `mobile: true`.
- **AMD GPU (PRO/FirePro)**: CSV `GPU Form Factor` column contains "MXM", or name matches `/\d[MS]\b/` or `/(Mobile)/` → `mobile: true`.
- **AMD GPU (Instinct)**: all tagged `server: true`.
- **NVIDIA GPU**: bare letter+number data center names (`A100`, `H100`, `B200`, `L40`, `T4`, etc., excluding `RTX`/`Quadro`/`GeForce`/`Titan` prefixes and M-suffix like `A10M`) → `server: true`. "Tesla" prefix or "Server" in name → `server: true`. Jetson (LPDDR5X memType) → `mobile: true`.
- **Intel GPU**: Arc name matches `/\dM\b/` (A370M, A550M, etc.) → `mobile: true`.
- **Apple CPU/GPU**: CSV `Form_Factor` column — `"Mobile"` → `mobile: true`; `"Desktop"` → `desktop: true`; `"Both"` → both flags. Apple Silicon entries appear in both CPU and GPU preset lists (unified memory architecture).
- **Intel CPU**: Xeon entries tagged `server: true` (hand-curated).

### Example model and quantization reference code

The `resources/` directory contains local clones of llama.cpp forks with quantization and KV cache implementations. Use these as reference code for proper model calculations:

- `resources/llama.cpp/` — upstream llama.cpp (standard quant types)
- `resources/ik_llama.cpp/` — ik_llama.cpp extensions (60+ extra types)
- `resources/llama-cpp-turboquant/` — turboquant fork (TURBO2_0/3_0/4_0 KV, TQ3_1S/TQ4_1S weights)
- `resources/llama.cpp-tq3/` — tq3 fork (TQ3_0 KV, TQ3_1S/TQ3_4S weights; ID collision with turboquant)
- `resources/llama-cpp-rotorquant/` — rotorquant fork (superset of turboquant + PLANAR/ISO KV types)
- `resources/gguf-parser-go/` — alternative memory calculator implementation which can be used for comparisons. Do not presume it is 100% correct.

### Adding a new fork's quantization types

1. Identify new type IDs and BPE values from the fork's `ggml.h` and `ggml-common.h`.
2. Add BPE entries to the `BPE` object in `calculations.js`. If IDs collide with an existing fork, only add non-colliding entries to the base `BPE`; put colliding entries in a new `<FORK>_FORK_BPE` export.
3. Add display names to `QUANT_NAMES` via a new `<FORK>_QUANT_NAMES` object (non-colliding IDs only; colliding IDs get `_nameOverride` at parse time).
4. Add KV cache types to `KV_VALID_QUANTS` and `KV_FORK_GROUPS` in `parsing.js`.
5. Add fork detection logic to `detectFork()` in `parsing.js` — use `general.file_type` and unique tensor dtype IDs.
6. Add override application in `applyForkOverrides()` — stamp `_bpeOverride` and `_nameOverride`.
7. Document the new fork and its detection heuristics in AGENTS.md under "Fork-aware BPE overrides".
