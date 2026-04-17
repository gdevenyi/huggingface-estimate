# AGENTS.md

## Structure

Two-file browser app, no build step, no dependencies. Loads `@huggingface/gguf` from jsDelivr CDN (`0.4.2`).

- **`calculations.js`** — Pure calculation module: architecture registry, KV cache, activations, MoE, weight calculations
- **`index.html`** — Display layer: HTML/CSS, GGUF parsing, HF API resolution, result rendering

Node CLI entry points (in `node_modules/`): `node-calculations.js`, `node-parsing.js`, `run-calc.js`

## Must serve via HTTP

`file://` blocks CORS headers needed to fetch GGUF metadata from HuggingFace:
```bash
python3 -m http.server 8000
```
Then visit `http://localhost:8000`.

## CLI usage

```bash
node run-calc.js bartowski/Llama-3.1-8B-Instruct-GGUF --ctx 8192 --kvTypeK Q8_0
node run-calc.js --batch testmodels.list
```

Options: `--ctx N`, `--batchSize N`, `--kvTypeK TYPE`, `--kvTypeV TYPE`. Batch file has one HF repo per line.

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

## CDN version pin

`@huggingface/gguf` from `https://cdn.jsdelivr.net/npm/@huggingface/gguf@0.4.2/+esm`. Check updates at `https://data.jsdelivr.com/v1/package/npm/@huggingface/gguf`.

## Bytes-per-element hardcoded

`GGML_QUANT_SIZES` is NOT exported from the browser build. BPE values are hardcoded as the `BPE` object in `calculations.js` (lines 5–40). Use `GGMLQuantizationType` enum keys as indices.

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
| `iswa` | Has `attention.sliding_window` or per-layer `head_count_kv` array | Reads `head_count_kv` as array for per-layer GQA |
| `moe` | Has `expert_count > 0` | Expert tensor grouping, VRAM/RAM split for inactive experts |

### Step 3: Add registry entry

```js
myarch: {
  name: 'myarch',
  categories: ['transformer', 'moe', 'iswa'],
  kvCache(meta, ctxSize, kvTypeK, kvTypeV) { /* { bytesK, bytesV, totalBytes, layers, headsK, headsV, totalHeadsKV, avgHeadsKV } */ },
  activations(meta, ctxSize, batchSize) { /* { totalBytes, perLayerBytes, isMoe, expertCount, expertUsedCount, expertFF } */ },
  moe(meta, tensorInfos) { /* { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, ... } */ },
  tensorGroups: {
    expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'],
    router: ['*ffn_gate_inp*'],
    shared: ['*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'],
  },
},
```

### Key patterns for tensor matching

- **Expert weights**: `_exps.` in name
- **Router/gate**: `ffn_gate_inp` or `attn_sinks`
- **Shared experts**: `_shexp.` or `_chexp.`
- **MoE bias**: `exp_probs_b`

Patterns use glob → regex conversion via `globMatch()`.

### Special cases

**MLA (DeepSeek2-style)**: KV cache uses `kv_lora_rank` for K, `key_length_mla` for V:
```js
const totalElemsK = n_layer * kv_lora_rank * ctxSize;
const totalElemsV = n_layer * key_length_mla * ctxSize;
```

**ISWA (Gemma4/Llama4-style)**: `head_count_kv` may be an array (per-layer):
```js
const n_head_kv_arr = Array.isArray(n_head_kv)
  ? (() => { const a = Array(n_layer).fill(n_head[0] || 1); for (let i = 0; i < n_layer; i++) if (n_head_kv[i]) a[i] = Number(n_head_kv[i]); return a; })()
  : Array(n_layer).fill(n_head_kv);
```

**Multiple router tensors**: Gemma4/GPT-OSS have 2 `ffn_gate_inp` per block. Use `.filter()`, not `.find()`.

### Step 4: Test

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
