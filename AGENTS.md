# AGENTS.md

## Structure

Single `index.html` — no build step, no dependencies. Loads `@huggingface/gguf` from jsDelivr CDN (`0.4.2`).

## Must serve via HTTP

`file://` protocol blocks CORS headers needed to fetch GGUF metadata from HuggingFace. Run:
```bash
python3 -m http.server 8000
```
Then visit `http://localhost:8000`.

## BigInt gotcha

`@huggingface/gguf` returns `tensorInfos[].shape` as `bigint[]`. Never multiply a `bigint` by a `number` (BPE values are `number`). Always convert first:
```js
// WRONG: t.shape.reduce((a, b) => a * Number(b), 1n)  // 1n * Number → TypeError
// CORRECT: t.shape.map(Number).reduce((a, b) => a * b, 1)
```

## HF URL normalization

HF's `/blob/` endpoint lacks CORS headers. Always normalize user-pasted URLs:
```js
url = path.replace(/\/blob\//, '/resolve/').replace(/#.*$/, '');
```

## MoE VRAM/RAM split

- **Dense**: all weights + KV + activations → VRAM
- **MoE**: only `expert_used_count` experts in VRAM, `(expert_count - expert_used_count)` experts in RAM
- KV cache and activations always in VRAM
- VRAM fit check compares against `vramBytes` (not `totalBytes`)

## CDN version pin

`@huggingface/gguf` imported from `https://cdn.jsdelivr.net/npm/@huggingface/gguf@0.4.2/+esm`. Check for updates via `https://data.jsdelivr.com/v1/package/npm/@huggingface/gguf`.

## Key exports from @huggingface/gguf

- `gguf(url)` — parses header + metadata + tensor info (~2MB fetch)
- `GGMLQuantizationType` — enum of all 34 quant types
- `GGUF_QUANT_ORDER` — quant ranking (largest to smallest)
- `GGUF_QUANT_DESCRIPTIONS` — human-readable descriptions

Note: `GGML_QUANT_SIZES` is NOT exported from the browser build. Bytes-per-element values are hardcoded in `index.html` (lines ~591-626).
