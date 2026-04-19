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

- **Dense**: all weights + KV + activations → VRAM
- **MoE**: only `expert_used_count` experts in VRAM, `(expert_count - expert_used_count)` in RAM
- KV cache and activations always in VRAM
- VRAM fit check compares against `vramBytes` (not `totalBytes`)

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

### Example model and quantization refrence code

The directores `ik_llama.cpp/` `llama.cpp/` and `ik_llama.cpp/` contain variants of the llama.cpp inference engine
with quantization and KV cache quantization implementations. Use these as reference code for proper model calcuations.

The directory `gguf-parser-go/` contains an alternative memory calculator implemenmtation which can be used for
comaparisons. Do not presume it is 100% correct.
