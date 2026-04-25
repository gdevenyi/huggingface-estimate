import { GGMLQuantizationType } from '@huggingface/gguf';

// ── Bytes-per-element for each quantization type ──
// Exact values from GGML quantization block structures
export const BPE = {
  [GGMLQuantizationType.F32]: 4.0,
  [GGMLQuantizationType.F16]: 2.0,
  [GGMLQuantizationType.BF16]: 2.0,
  [GGMLQuantizationType.Q4_0]: 18 / 32,
  [GGMLQuantizationType.Q4_1]: 20 / 32,
  [GGMLQuantizationType.Q5_0]: 22 / 32,
  [GGMLQuantizationType.Q5_1]: 24 / 32,
  [GGMLQuantizationType.Q8_0]: 34 / 32,
  [GGMLQuantizationType.Q8_1]: 36 / 32,
  [GGMLQuantizationType.Q2_K]: 84 / 256,
  [GGMLQuantizationType.Q3_K]: 110 / 256,
  [GGMLQuantizationType.Q4_K]: 144 / 256,
  [GGMLQuantizationType.Q5_K]: 176 / 256,
  [GGMLQuantizationType.Q6_K]: 210 / 256,
  [GGMLQuantizationType.Q8_K]: 292 / 256,
  [GGMLQuantizationType.IQ2_XXS]: 66 / 256,
  [GGMLQuantizationType.IQ2_XS]: 74 / 256,
  [GGMLQuantizationType.IQ3_XXS]: 98 / 256,
  [GGMLQuantizationType.IQ1_S]: 50 / 256,
  [GGMLQuantizationType.IQ4_NL]: 18 / 32,
  [GGMLQuantizationType.IQ3_S]: 110 / 256,
  [GGMLQuantizationType.IQ2_S]: 82 / 256,
  [GGMLQuantizationType.IQ4_XS]: 136 / 256,
  [GGMLQuantizationType.I8]: 1.0,
  [GGMLQuantizationType.I16]: 2.0,
  [GGMLQuantizationType.I32]: 4.0,
  [GGMLQuantizationType.I64]: 8.0,
  [GGMLQuantizationType.F64]: 8.0,
  [GGMLQuantizationType.IQ1_M]: 56 / 256,
  [GGMLQuantizationType.TQ1_0]: 54 / 256,
  [GGMLQuantizationType.TQ2_0]: 66 / 256,
  [GGMLQuantizationType.MXFP4]: 17 / 32,
  [GGMLQuantizationType.NVFP4]: 36 / 64,
  [GGMLQuantizationType.Q1_0]: 18 / 128,
  36: 1.0,         // I2_S (MS BitNet)
  133: 26 / 32,    // Q6_0
  // ── ik_llama.cpp extensions (not in @huggingface/gguf@0.4.2) ──
  // KV cache quantizations
  151: 32 / 32,   // Q8_KV
  398: 32 / 32,   // Q8_KV_R8
  // Q8 variants (different block sizes)
  136: 68 / 64,   // Q8_K64
  147: 64 / 64,   // Q8_K16
  148: 296 / 256, // Q8_K32
  149: 296 / 256, // Q8_KR8
  150: 140 / 128, // Q8_K128
  399: 258 / 256, // Q8_K_R8
  // X4 row-interleaved variants
  97: 34 / 32,    // Q8_0_X4
  98: 36 / 32,    // Q8_1_X4
  99: 36 / 32,    // Q8_2_X4
  // Legacy interleaved-GEMM Q4_0 variants — REMOVED in mainline ggml
  // (ggml/src/ggml.c:873–891 marks type_size=0 and replaces them with runtime
  // repacking from Q4_0). BPE values retained because old GGUFs still encode
  // them at 18 bytes per 32-element block.
  31: 18 / 32,    // Q4_0_4_4 (legacy)
  32: 18 / 32,    // Q4_0_4_8 (legacy)
  33: 18 / 32,    // Q4_0_8_8 (legacy)
  // Bitnet ternary quantizations
  134: 13 / 64,   // IQ1_BN
  135: 16 / 64,   // IQ2_BN
  // K-extension IQ variants
  137: 76 / 256,  // IQ2_K
  138: 110 / 256, // IQ3_K
  139: 144 / 256, // IQ4_K
  140: 176 / 256, // IQ5_K
  141: 212 / 256, // IQ6_K
  144: 136 / 256, // IQ4_KS
  145: 70 / 256,  // IQ2_KS
  146: 128 / 256, // IQ4_KSS
  152: 168 / 256, // IQ5_KS
  153: 68 / 256,  // IQ2_KT
  154: 100 / 256, // IQ3_KT
  155: 128 / 256, // IQ4_KT
  156: 102 / 256, // IQ3_KS
  157: 86 / 256,  // IQ2_KL
  158: 56 / 256,  // IQ1_KT
  // Row-interleaved R4 variants
  202: 18 / 32,   // Q4_0_R8
  206: 22 / 32,   // Q5_0_R4
  208: 34 / 32,   // Q8_0_R8
  210: 84 / 256,  // Q2_K_R4
  211: 110 / 256, // Q3_K_R4
  212: 144 / 256, // Q4_K_R4
  213: 176 / 256, // Q5_K_R4
  214: 210 / 256, // Q6_K_R4
  216: 66 / 256,  // IQ2_XXS_R4
  217: 74 / 256,  // IQ2_XS_R4
  218: 98 / 256,  // IQ3_XXS_R4
  219: 6 / 32,    // IQ1_S_R4
  220: 18 / 32,   // IQ4_NL_R4
  221: 110 / 256, // IQ3_S_R4
  222: 82 / 256,  // IQ2_S_R4
  223: 136 / 256, // IQ4_XS_R8
  229: 7 / 32,    // IQ1_M_R4
  233: 26 / 32,   // Q6_0_R4
  335: 16 / 64,   // IQ2_BN_R4
  337: 76 / 256,  // IQ2_K_R4
  338: 110 / 256, // IQ3_K_R4
  339: 144 / 256, // IQ4_K_R4
  340: 176 / 256, // IQ5_K_R4
  344: 136 / 256, // IQ4_KS_R4
  352: 168 / 256, // IQ5_KS_R4
  // Other
  230: 2 / 1,     // BF16_R16
  397: 258 / 256, // Q8_K_R16
  // turboquant weight quantization (numeric IDs from llama-cpp-turboquant ggml.h)
  45: 16 / 32, // TQ3_1S
  46: 20 / 32, // TQ4_1S
  // turboquant KV cache quantization (string keys used by KV_VALID_QUANTS dropdown)
  TQ3_1S: 16 / 32,
  TQ4_1S: 20 / 32,
  TURBO2_0: 34 / 128,
  TURBO3_0: 50 / 128,
  TURBO4_0: 68 / 128,
  // rotorquant KV cache quantization
  PLANAR3_0: 50 / 128,
  PLANAR4_0: 68 / 128,
  ISO3_0: 50 / 128,
  ISO4_0: 68 / 128,
  // llama.cpp-tq3 KV cache quantization (string key for KV_VALID_QUANTS dropdown)
  TQ3_0: 14 / 32,
};

// Quantization type names for display
// Auto-populated from @huggingface/gguf package, plus manual entries for ik_llama.cpp extensions
export const QUANT_NAMES = {};
for (const [key, val] of Object.entries(GGMLQuantizationType)) {
  if (typeof val === 'number') QUANT_NAMES[val] = key;
}
// ── ik_llama.cpp extension names (labeled for clarity) ──
const IK_LLAMA_QUANT_NAMES = {
  151: 'Q8_KV',
  398: 'Q8_KV_R8 (ik_llama)',
  136: 'Q8_K64 (ik_llama)',
  147: 'Q8_K16 (ik_llama)',
  148: 'Q8_K32 (ik_llama)',
  149: 'Q8_KR8 (ik_llama)',
  150: 'Q8_K128 (ik_llama)',
  399: 'Q8_K_R8 (ik_llama)',
  97: 'Q8_0_X4 (ik_llama)',
  98: 'Q8_1_X4 (ik_llama)',
  99: 'Q8_2_X4 (ik_llama)',
  31: 'Q4_0_4_4 (legacy)',
  32: 'Q4_0_4_8 (legacy)',
  33: 'Q4_0_8_8 (legacy)',
  134: 'IQ1_BN (ik_llama)',
  135: 'IQ2_BN (ik_llama)',
  137: 'IQ2_K (ik_llama)',
  138: 'IQ3_K (ik_llama)',
  139: 'IQ4_K (ik_llama)',
  140: 'IQ5_K (ik_llama)',
  141: 'IQ6_K (ik_llama)',
  144: 'IQ4_KS (ik_llama)',
  145: 'IQ2_KS (ik_llama)',
  146: 'IQ4_KSS (ik_llama)',
  152: 'IQ5_KS (ik_llama)',
  153: 'IQ2_KT (ik_llama)',
  154: 'IQ3_KT (ik_llama)',
  155: 'IQ4_KT (ik_llama)',
  156: 'IQ3_KS (ik_llama)',
  157: 'IQ2_KL (ik_llama)',
  158: 'IQ1_KT (ik_llama)',
  202: 'Q4_0_R8 (ik_llama)',
  206: 'Q5_0_R4 (ik_llama)',
  208: 'Q8_0_R8 (ik_llama)',
  210: 'Q2_K_R4 (ik_llama)',
  211: 'Q3_K_R4 (ik_llama)',
  212: 'Q4_K_R4 (ik_llama)',
  213: 'Q5_K_R4 (ik_llama)',
  214: 'Q6_K_R4 (ik_llama)',
  216: 'IQ2_XXS_R4 (ik_llama)',
  217: 'IQ2_XS_R4 (ik_llama)',
  218: 'IQ3_XXS_R4 (ik_llama)',
  219: 'IQ1_S_R4 (ik_llama)',
  220: 'IQ4_NL_R4 (ik_llama)',
  221: 'IQ3_S_R4 (ik_llama)',
  222: 'IQ2_S_R4 (ik_llama)',
  223: 'IQ4_XS_R8 (ik_llama)',
  229: 'IQ1_M_R4 (ik_llama)',
  233: 'Q6_0_R4 (ik_llama)',
  335: 'IQ2_BN_R4 (ik_llama)',
  337: 'IQ2_K_R4 (ik_llama)',
  338: 'IQ3_K_R4 (ik_llama)',
  339: 'IQ4_K_R4 (ik_llama)',
  340: 'IQ5_K_R4 (ik_llama)',
  344: 'IQ4_KS_R4 (ik_llama)',
  352: 'IQ5_KS_R4 (ik_llama)',
  230: 'BF16_R16 (ik_llama)',
  36: 'I2_S (ik_llama)',
  133: 'Q6_0',
  397: 'Q8_K_R16 (ik_llama)',
};
Object.assign(QUANT_NAMES, IK_LLAMA_QUANT_NAMES);
const TURBOQUANT_QUANT_NAMES = {
  TQ3_1S: 'TQ3_1S (turboquant)',
  TQ4_1S: 'TQ4_1S (turboquant)',
  TURBO2_0: 'TURBO2_0',
  TURBO3_0: 'TURBO3_0',
  TURBO4_0: 'TURBO4_0',
  42: 'TURBO2_0',
  43: 'TURBO3_0',
  44: 'TURBO4_0',
  45: 'TQ3_1S (turboquant)',
  46: 'TQ4_1S (turboquant)',
};
Object.assign(QUANT_NAMES, TURBOQUANT_QUANT_NAMES);
const ROTORQUANT_QUANT_NAMES = {
  PLANAR3_0: 'PLANAR3_0',
  PLANAR4_0: 'PLANAR4_0',
  ISO3_0: 'ISO3_0',
  ISO4_0: 'ISO4_0',
};
Object.assign(QUANT_NAMES, ROTORQUANT_QUANT_NAMES);
export const TQ3_QUANT_NAMES = {
  44: 'TQ3_1S (tq3)',
  45: 'TQ3_1S (tq3)',
  46: 'TQ3_4S (tq3)',
  200: 'TQ3_0 (tq3 KV)',
};
QUANT_NAMES[200] = 'TQ3_0 (tq3 KV)';

// BPE overrides applied when fork detection identifies a llama.cpp-tq3 model.
// IDs 44, 45, 46 collide with turboquant (TURBO4_0 / TQ3_1S / TQ4_1S).
export const TQ3_FORK_BPE = {
  44: 16 / 32,
  45: 16 / 32,
  46: 16 / 32,
  200: 14 / 32,
};

// ── Tensor size helpers ──
function tensorElems(t) {
  return t.shape.map(Number).reduce((a, b) => a * b, 1);
}
function tensorBpe(t) {
  return t._bpeOverride ?? BPE[t.dtype] ?? 0;
}
function tensorQuantName(t) {
  return t._nameOverride ?? QUANT_NAMES[t.dtype] ?? `type_${t.dtype}`;
}
function sumBytes(tensors) {
  let s = 0;
  for (const t of tensors) s += tensorElems(t) * tensorBpe(t);
  return s;
}
function sumElems(tensors) {
  let s = 0;
  for (const t of tensors) s += tensorElems(t);
  return s;
}

// ── Standard transformer KV cache (parameterized) ──
// opts.iswa               — arch has interleaved sliding-window layers
// opts.denseFirst         — when ISWA, first layer in each period is dense (smallthinker pattern)
// opts.swaPeriodDefault   — fallback for integer sliding_window_pattern
// opts.swaDefault         — fallback for missing sliding_window
// opts.effectiveLayers    — override layer count (gemma4 shared_kv_layers, gemma3n layer_kv_from_start)
// opts.layerFilter        — predicate skipping layers (qwen35moe full_attention_interval)
function buildKvCache(meta, ctxSize, kvTypeK, kvTypeV, opts = {}) {
  const arch = meta['general.architecture'];
  const n_embd = getMeta(meta, `${arch}.embedding_length`);
  const n_head = getMeta(meta, `${arch}.attention.head_count`);
  const headDimK = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
  const headDimV = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
  const headDimK_swa = opts.iswa ? (getMeta(meta, `${arch}.attention.key_length_swa`) || headDimK) : headDimK;
  const headDimV_swa = opts.iswa ? (getMeta(meta, `${arch}.attention.value_length_swa`) || headDimV) : headDimV;
  const n_head_kv_raw = getMeta(meta, `${arch}.attention.head_count_kv`);
  const n_block = getMeta(meta, `${arch}.block_count`);
  const n_layer = opts.effectiveLayers ? opts.effectiveLayers(meta, n_block) : n_block;
  const n_head_kv_arr = Array.isArray(n_head_kv_raw)
    ? (() => {
        const a = Array(n_layer).fill(0);
        for (let i = 0; i < n_layer; i++) if (n_head_kv_raw[i]) a[i] = Number(n_head_kv_raw[i]);
        return a;
      })()
    : Array(n_layer).fill(n_head_kv_raw);
  const n_swa = getMeta(meta, `${arch}.attention.sliding_window`) || opts.swaDefault || 0;
  const swa_pattern_raw = meta[`${arch}.attention.sliding_window_pattern`];
  const swa_arr = Array.isArray(swa_pattern_raw) ? swa_pattern_raw.map(v => Number(v) !== 0) : null;
  const swa_period = swa_arr ? 0 : (swa_pattern_raw != null ? Number(swa_pattern_raw) : (opts.swaPeriodDefault || 0));

  // SWA cache cells = GGML_PAD(min(ctx, n_swa + n_ubatch), 256). See
  // llama-kv-cache-iswa.cpp:46–50. Default n_ubatch in llama.cpp is 512.
  // n_seq_max=1 + unified=false (the typical inference case) is assumed.
  const N_UBATCH_DEFAULT = 512;
  const KV_CELL_PAD = 256;
  const swa_cells = n_swa > 0
    ? Math.min(ctxSize, Math.ceil((n_swa + N_UBATCH_DEFAULT) / KV_CELL_PAD) * KV_CELL_PAD)
    : ctxSize;

  let totalElemsK = 0, totalElemsV = 0, activeLayers = 0, activeHeadsKV = 0;
  for (let i = 0; i < n_layer; i++) {
    if (opts.layerFilter && !opts.layerFilter(i)) continue;
    const heads = n_head_kv_arr[i] || 0;
    if (heads <= 0) continue;
    const isSwa = opts.iswa
      ? (swa_arr ? !!swa_arr[i] : (swa_period > 0 && (
          opts.denseFirst ? (i % swa_period !== 0) : (i % swa_period < (swa_period - 1))
        )))
      : false;
    const layerCtx = isSwa ? swa_cells : ctxSize;
    const hK = isSwa ? headDimK_swa : headDimK;
    const hV = isSwa ? headDimV_swa : headDimV;
    totalElemsK += hK * heads * layerCtx;
    totalElemsV += hV * heads * layerCtx;
    activeLayers++;
    activeHeadsKV += heads;
  }
  return {
    bytesK: totalElemsK * (BPE[kvTypeK] || 0),
    bytesV: totalElemsV * (BPE[kvTypeV] || 0),
    layers: n_block,
    headDimK, headDimV,
    totalHeadsKV: activeHeadsKV,
    avgHeadsKV: activeLayers > 0 ? activeHeadsKV / activeLayers : 0,
  };
}

// ── MLA KV cache (DeepSeek2 / GLM-DSA style) ──
function mlaKvCache(meta, ctxSize, kvTypeK, kvTypeV) {
  const arch = meta['general.architecture'];
  const kv_lora_rank = getMeta(meta, `${arch}.attention.kv_lora_rank`);
  const n_rot = getMeta(meta, `${arch}.rope.dimension_count`);
  const n_layer = getMeta(meta, `${arch}.block_count`);
  const nextn = getMeta(meta, `${arch}.nextn_predict_layers`);
  const n_layer_kv = Math.max(0, n_layer - nextn);
  const totalElemsK = n_layer_kv * (kv_lora_rank + n_rot) * ctxSize;
  return {
    bytesK: totalElemsK * (BPE[kvTypeK] || 0),
    bytesV: 0,
    layers: n_layer,
    headDimK: kv_lora_rank + n_rot,
    headDimV: 0,
    totalHeadsKV: n_layer_kv * (kv_lora_rank + n_rot),
    avgHeadsKV: kv_lora_rank + n_rot,
  };
}

// Sum of feed_forward_length across layers. Some arches (gemma3n, nemotron_h)
// emit `feed_forward_length` as a per-layer array; treat scalar as uniform.
function ffSumOverLayers(meta, arch, n_layer, fromLayer = 0, toLayer = n_layer) {
  const v = meta[`${arch}.feed_forward_length`];
  if (Array.isArray(v)) {
    let s = 0;
    const lo = Math.max(0, fromLayer);
    const hi = Math.min(v.length, toLayer);
    for (let i = lo; i < hi; i++) s += Number(v[i] || 0);
    return s;
  }
  const n = Number(v);
  return Number.isFinite(n) ? n * Math.max(0, toLayer - fromLayer) : 0;
}

// ── Standard transformer activations ──
function buildActivations(meta, batchSize) {
  const arch = meta['general.architecture'];
  const n_embd = getMeta(meta, `${arch}.embedding_length`);
  const n_layer = getMeta(meta, `${arch}.block_count`);
  const expertCount = getMeta(meta, `${arch}.expert_count`);
  const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
  const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
  const isMoe = expertCount > 0;
  const ffTotal = (isMoe && expertUsedCount > 0 && expertFF > 0)
    ? expertUsedCount * expertFF * n_layer
    : ffSumOverLayers(meta, arch, n_layer);
  const totalBytes = batchSize * (n_embd * n_layer + ffTotal) * 4;
  return {
    totalBytes,
    perLayerBytes: n_layer > 0 ? totalBytes / n_layer : 0,
    isMoe, expertCount, expertUsedCount, expertFF,
  };
}

// ── Shared-expert MoE activations: residual + shared FFN + routed experts ──
// Used by qwen35moe, qwen3next (shared expert may have different FFN dim)
function sharedExpertActivations(meta, batchSize) {
  const arch = meta['general.architecture'];
  const n_embd = getMeta(meta, `${arch}.embedding_length`);
  const n_layer = getMeta(meta, `${arch}.block_count`);
  const expertCount = getMeta(meta, `${arch}.expert_count`);
  const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
  const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
  const expertSharedFF = getMeta(meta, `${arch}.expert_shared_feed_forward_length`) || n_embd;
  const isMoe = expertCount > 0;
  const ffTotal = (isMoe && expertUsedCount > 0 && expertFF > 0)
    ? (expertSharedFF + expertUsedCount * expertFF) * n_layer
    : ffSumOverLayers(meta, arch, n_layer);
  const totalBytes = batchSize * (n_embd * n_layer + ffTotal) * 4;
  return {
    totalBytes,
    perLayerBytes: n_layer > 0 ? totalBytes / n_layer : 0,
    isMoe, expertCount, expertUsedCount, expertFF,
  };
}

// ── Leading-dense activations: dense FFN for first N layers, MoE afterwards ──
function leadingDenseActivations(meta, batchSize) {
  const arch = meta['general.architecture'];
  const n_embd = getMeta(meta, `${arch}.embedding_length`);
  const n_layer = getMeta(meta, `${arch}.block_count`);
  const expertCount = getMeta(meta, `${arch}.expert_count`);
  const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
  const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
  const leadingDense = getMeta(meta, `${arch}.leading_dense_block_count`);
  const denseFF = ffSumOverLayers(meta, arch, n_layer, 0, leadingDense);
  const denseBytes = batchSize * (leadingDense * n_embd + denseFF) * 4;
  const moeLayers = n_layer - leadingDense;
  const moeFF = (expertUsedCount > 0 && expertFF > 0)
    ? expertUsedCount * expertFF * moeLayers
    : ffSumOverLayers(meta, arch, n_layer, leadingDense, n_layer);
  const moeBytes = batchSize * (moeLayers * n_embd + moeFF) * 4;
  return {
    totalBytes: denseBytes + moeBytes,
    perLayerBytes: 0,
    isMoe: expertCount > 0,
    expertCount, expertUsedCount, expertFF, leadingDense,
  };
}

// MLA + leading-dense MoE activations (deepseek2, mistral4). Mirrors the
// inline math previously in the deepseek2 activations method.
function deepseek2MlaMoeActivations(meta, batchSize) {
  const arch = meta['general.architecture'];
  const n_embd = getMeta(meta, `${arch}.embedding_length`);
  const n_layer = getMeta(meta, `${arch}.block_count`);
  const expertCount = getMeta(meta, `${arch}.expert_count`);
  const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
  const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
  const q_lora_rank = getMeta(meta, `${arch}.attention.q_lora_rank`);
  const kv_lora_rank = getMeta(meta, `${arch}.attention.kv_lora_rank`);
  const isMoe = expertCount > 0;
  const leadingDense = getMeta(meta, `${arch}.leading_dense_block_count`);
  const denseFF = ffSumOverLayers(meta, arch, n_layer, 0, leadingDense);
  const moeLayers = n_layer - leadingDense;
  const moeFF = (isMoe && expertUsedCount > 0 && expertFF > 0)
    ? expertUsedCount * expertFF * moeLayers
    : ffSumOverLayers(meta, arch, n_layer, leadingDense, n_layer);
  const perLayerAttn = n_embd + q_lora_rank + kv_lora_rank;
  const denseBytes = batchSize * (leadingDense * perLayerAttn + denseFF) * 4;
  const moeBytes = batchSize * (moeLayers * perLayerAttn + moeFF) * 4;
  return { totalBytes: denseBytes + moeBytes, perLayerBytes: 0, isMoe, expertCount, expertUsedCount, expertFF, leadingDense };
}

function mlaActivations(meta, batchSize) {
  const arch = meta['general.architecture'];
  const n_embd = getMeta(meta, `${arch}.embedding_length`);
  const n_layer = getMeta(meta, `${arch}.block_count`);
  const q_lora_rank = getMeta(meta, `${arch}.attention.q_lora_rank`);
  const kv_lora_rank = getMeta(meta, `${arch}.attention.kv_lora_rank`);
  const ffTotal = ffSumOverLayers(meta, arch, n_layer);
  const totalBytes = batchSize * (n_layer * (n_embd + q_lora_rank + kv_lora_rank) + ffTotal) * 4;
  return { totalBytes, perLayerBytes: n_layer > 0 ? totalBytes / n_layer : 0, isMoe: false, expertCount: 0, expertUsedCount: 0, expertFF: 0 };
}

// ── MoE weight accounting (parameterized) ──
// Default filters match the llama handler; per-arch overrides supply their
// own predicates for expert / router / shared tensor classification.
function buildMoe(meta, tensorInfos, {
  isExpert = (t) => t.name.includes('_exps.') || t.name.includes('exp_probs_b'),
  isRouter = (t) => t.name.includes('ffn_gate_inp'),
  isShared = (t) => t.name.includes('_shexp.') || t.name.includes('_chexp.'),
} = {}) {
  const arch = meta['general.architecture'];
  const expertCount = getMeta(meta, `${arch}.expert_count`);
  const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
  if (expertCount === 0) return null;
  const expertTensors = tensorInfos.filter(isExpert);
  const routerTensors = tensorInfos.filter(isRouter);
  const sharedTensors = tensorInfos.filter(isShared);
  const expertWeightBytes = sumBytes(expertTensors);
  const routerBytes = sumBytes(routerTensors);
  const sharedBytes = sumBytes(sharedTensors);
  const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
  return {
    expertCount, expertUsedCount,
    expertWeightBytes, routerBytes, sharedBytes,
    totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes,
    totalModelParams: sumElems(tensorInfos),
    expertParams: sumElems(expertTensors),
    activeExpertWeightBytes: perExpertWeightBytes * expertUsedCount,
  };
}

// ── No-op KV cache for encoder-only architectures (gemma-embedding, t5encoder, modern-bert, etc.) ──
const noKvCache = () => ({
  bytesK: 0, bytesV: 0, layers: 0,
  headDimK: 0, headDimV: 0, totalHeadsKV: 0, avgHeadsKV: 0,
});

// ── Canonical handler triple for standard llama-family transformers ──
const llamaKvCache = (m, c, kK, kV) => buildKvCache(m, c, kK, kV);
const llamaActivations = buildActivations;
const llamaMoe = buildMoe;
const LLAMA_TENSOR_GROUPS = {
  expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*', '*exp_probs_b*'],
  router: ['*ffn_gate_inp*'],
  shared: ['*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'],
};

// Common "no shared tensors" predicate (used by many MoE archs that have no shared experts)
const noShared = () => false;
const shexpOnly = (t) => t.name.includes('_shexp.');
const moeNoShared = (m, ti) => buildMoe(m, ti, { isShared: noShared });
const moeShexpOnly = (m, ti) => buildMoe(m, ti, { isShared: shexpOnly });

// nemotron_h family: a layer is recurrent only when both head_count_kv == 0
// AND feed_forward_length == 0 (llama-model.cpp:2257). Bare head_count_kv == 0
// is not enough — pure-FFN layers also have head_count_kv == 0.
function nemotronHRecurrentLayers(meta) {
  const arch = meta['general.architecture'];
  const n_layer = getMeta(meta, `${arch}.block_count`);
  const head_count_kv = meta[`${arch}.attention.head_count_kv`];
  const ff = meta[`${arch}.feed_forward_length`];
  if (!Array.isArray(head_count_kv) || !Array.isArray(ff)) return 0;
  let count = 0;
  for (let i = 0; i < n_layer; i++) {
    if (Number(head_count_kv[i] || 0) === 0 && Number(ff[i] || 0) === 0) count++;
  }
  return count;
}

// ── Architecture Registry ──
// Each architecture declares its categories and provides specialized handlers
// for KV cache, activations, and MoE weight calculations.

export const ARCHITECTURES = {
  // ── Default: standard transformer (llama, mistral, qwen2, phi3, etc.) ──
  llama: {
    name: 'llama',
    categories: ['transformer'],
    fallback: true,
    kvCache: llamaKvCache,
    activations: llamaActivations,
    moe: llamaMoe,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── DeepSeek2: MLA (Multi-Head Latent Attention) + MoE ──
  deepseek2: {
    name: 'deepseek2',
    categories: ['transformer', 'moe', 'mla'],
    kvCache: mlaKvCache,
    activations: deepseek2MlaMoeActivations,
    moe: moeShexpOnly,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── MiniCPM3: MLA (lite) ──
  minicpm3: {
    name: 'minicpm3',
    categories: ['transformer', 'mla'],
    kvCache: mlaKvCache,
    activations: mlaActivations,
    moe: llamaMoe,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── PLM: simplified MLA (only KV compressed, not Q) ──
  plm: {
    name: 'plm',
    categories: ['transformer', 'mla'],
    kvCache: mlaKvCache,
    activations: mlaActivations,
    moe: llamaMoe,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── Kimi-Linear: MLA + MoE + KDA hybrid ──
  'kimi-linear': {
    name: 'kimi-linear',
    categories: ['transformer', 'moe', 'mla'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const kv_lora_rank = getMeta(meta, `${arch}.attention.kv_lora_rank`);
      const n_rot = getMeta(meta, `${arch}.rope.dimension_count`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const n_head_kv_raw = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_head_kv_arr = Array.isArray(n_head_kv_raw) ? n_head_kv_raw : Array(n_layer).fill(Number(n_head_kv_raw));
      let totalElemsK = 0;
      for (let i = 0; i < n_layer; i++) {
        if (n_head_kv_arr[i] > 0) totalElemsK += (kv_lora_rank + n_rot) * ctxSize;
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: 0,
        layers: n_layer,
        headDimK: kv_lora_rank + n_rot,
        headDimV: 0,
        totalHeadsKV: totalElemsK / ctxSize,
        avgHeadsKV: kv_lora_rank + n_rot,
      };
    },
    activations: leadingDenseActivations,
    moe: moeShexpOnly,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── Gemma4: ISWA + MoE ──
  gemma4: {
    name: 'gemma4',
    categories: ['transformer', 'moe', 'iswa'],
    kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, {
      iswa: true,
      effectiveLayers: (meta, n_block) => {
        const arch = meta['general.architecture'];
        const n_kv_shared = getMeta(meta, `${arch}.attention.shared_kv_layers`);
        return n_kv_shared > 0 ? n_block - n_kv_shared : n_block;
      },
    }),
    activations: buildActivations,
    moe: moeNoShared,
    tensorGroups: { expert: ['*ffn_gate_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] },
  },

  // ── GPT-OSS: ISWA + MoE (no shared experts) ──
  'gpt-oss': {
    name: 'gpt-oss',
    categories: ['transformer', 'moe', 'iswa'],
    kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 2 }),
    activations: buildActivations,
    moe: moeNoShared,
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] },
  },

  // ── Llama4: ISWA + MoE with shared experts ──
  llama4: {
    name: 'llama4',
    categories: ['transformer', 'moe', 'iswa'],
    kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 4, swaDefault: 8192 }),
    activations: buildActivations,
    moe: moeShexpOnly,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── Qwen3 MoE: standard MoE ──
  qwen3moe: {
    name: 'qwen3moe',
    categories: ['transformer', 'moe'],
    kvCache: llamaKvCache,
    activations: buildActivations,
    moe: moeNoShared,
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] },
  },

  // ── Qwen3.6 MoE: mixed DeltaNet/attention + MoE with shared experts ──
  // Only every Nth layer has full attention (the rest are DeltaNet with no KV cache).
  qwen35moe: {
    name: 'qwen35moe',
    categories: ['transformer', 'moe'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const interval = getMeta(meta, `${arch}.full_attention_interval`) || 4;
      return buildKvCache(meta, ctxSize, kvTypeK, kvTypeV, {
        layerFilter: (i) => ((i + 1) % interval === 0),
      });
    },
    activations: sharedExpertActivations,
    moe: (m, ti) => buildMoe(m, ti, {
      isRouter: (t) => t.name.includes('ffn_gate_inp') && !t.name.includes('shexp'),
      isShared: (t) => t.name.includes('_shexp.') || t.name.includes('ffn_gate_inp_shexp'),
    }),
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_gate_inp_shexp*', '*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'] },
  },

  // ── Standard transformers (reuse llama handlers) ──
  qwen2:          { name: 'qwen2',          categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen3:          { name: 'qwen3',          categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen35:         { name: 'qwen35',         categories: ['transformer'],      kvCache(meta, ctxSize, kvTypeK, kvTypeV) { const arch = meta['general.architecture']; const interval = getMeta(meta, `${arch}.full_attention_interval`) || 4; return buildKvCache(meta, ctxSize, kvTypeK, kvTypeV, { layerFilter: (i) => ((i + 1) % interval === 0) }); }, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen3next:      { name: 'qwen3next',      categories: ['transformer', 'moe'],kvCache(meta, ctxSize, kvTypeK, kvTypeV) { const arch = meta['general.architecture']; const interval = getMeta(meta, `${arch}.full_attention_interval`) || 4; return buildKvCache(meta, ctxSize, kvTypeK, kvTypeV, { layerFilter: (i) => ((i + 1) % interval === 0) }); }, activations: sharedExpertActivations, moe: (m, ti) => buildMoe(m, ti, { isRouter: (t) => t.name.includes('ffn_gate_inp') && !t.name.includes('shexp'), isShared: (t) => t.name.includes('_shexp.') || t.name.includes('ffn_gate_inp_shexp'), }), tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_gate_inp_shexp*', '*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'] } },
  qwen2vl:        { name: 'qwen2vl',        categories: ['transformer', 'vl'],kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen3vl:        { name: 'qwen3vl',        categories: ['transformer', 'vl'],kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  gemma3:         { name: 'gemma3',         categories: ['transformer', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 6 }), activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  gemma2:         { name: 'gemma2',         categories: ['transformer', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 2, swaDefault: 4096 }), activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  olmo2:          { name: 'olmo2',          categories: ['transformer', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 4 }), activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  phi3:           { name: 'phi3',           categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  granite:        { name: 'granite',        categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  granitehybrid:  { name: 'granitehybrid',  categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mistral3:       { name: 'mistral3',       categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mistral4:       { name: 'mistral4',       categories: ['transformer', 'moe', 'mla'], kvCache: mlaKvCache, activations: deepseek2MlaMoeActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  glm4:           { name: 'glm4',           categories: ['transformer'],      kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { effectiveLayers: (meta, n_block) => n_block - getMeta(meta, `${meta['general.architecture']}.nextn_predict_layers`) }), activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'falcon-h1':    { name: 'falcon-h1',      categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  deci:           { name: 'deci',           categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  cohere2:        { name: 'cohere2',        categories: ['transformer', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 4 }), activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  smollm3:        { name: 'smollm3',        categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  ernie4_5:       { name: 'ernie4_5',       categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  grok:           { name: 'grok',           categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'gemma-embedding':{ name: 'gemma-embedding', categories: ['embedding'],     kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  nemotron_h:     { name: 'nemotron_h',     categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, recurrentLayers: nemotronHRecurrentLayers, tensorGroups: LLAMA_TENSOR_GROUPS },
  lfm2:           { name: 'lfm2',           categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'minimax-m2':   { name: 'minimax-m2',     categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  seed_oss:       { name: 'seed_oss',       categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  apertus:        { name: 'apertus',        categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  dots1:          { name: 'dots1',          categories: ['transformer', 'moe'],kvCache: llamaKvCache, activations: leadingDenseActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  flux:           { name: 'flux',           categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  ltxv:           { name: 'ltxv',           categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  lumina2:        { name: 'lumina2',        categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen_image:     { name: 'qwen_image',     categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  wan:            { name: 'wan',            categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'acestep-lm':   { name: 'acestep-lm',    categories: ['audio'],            kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  t5encoder:      { name: 't5encoder',      categories: ['transformer'],      kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mimo2:          { name: 'mimo2',          categories: ['transformer', 'moe', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true }), activations: buildActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'hunyuan-dense':{ name: 'hunyuan-dense',  categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  // exaone4: only the 64-layer (32B) variant has SWA; smaller variants are dense.
  // llama-model.cpp:2287–2299 — the SWA branch is gated on n_layer==64.
  exaone4:        { name: 'exaone4',        categories: ['transformer', 'iswa'], kvCache: (m, c, kK, kV) => {
    const arch = m['general.architecture'];
    const n_layer = getMeta(m, `${arch}.block_count`);
    if (n_layer === 64) return buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 4, swaDefault: 4096 });
    return buildKvCache(m, c, kK, kV);
  }, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  plamo3:         { name: 'plamo3',         categories: ['transformer', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 8 }), activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  // smallthinker: llama-model.cpp:2697–2704 forces n_swa = 4096 when the SWA key is present.
  smallthinker:   { name: 'smallthinker',   categories: ['transformer', 'moe', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 4, swaDefault: 4096, denseFirst: true }), activations: buildActivations, moe: moeNoShared, tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] } },
  qwen2moe:       { name: 'qwen2moe',       categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: buildActivations, moe: moeShexpOnly, tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_gate_inp_shexp*', '*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'] } },
  'modern-bert':  { name: 'modern-bert',    categories: ['embedding'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  // ── BERT-family encoders: llama-model.cpp:8430–8443 returns res = nullptr (no KV cache) ──
  bert:           { name: 'bert',           categories: ['embedding'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'nomic-bert':   { name: 'nomic-bert',     categories: ['embedding'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'nomic-bert-moe':{ name: 'nomic-bert-moe',categories: ['embedding', 'moe'], kvCache: noKvCache,    activations: buildActivations, moe: moeNoShared, tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] } },
  'neo-bert':     { name: 'neo-bert',       categories: ['embedding'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'jina-bert-v2': { name: 'jina-bert-v2',   categories: ['embedding'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'jina-bert-v3': { name: 'jina-bert-v3',   categories: ['embedding'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  eurobert:       { name: 'eurobert',       categories: ['embedding'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'wavtokenizer-dec':{ name: 'wavtokenizer-dec', categories: ['audio'],       kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  // ── Diffusion-LM family: also res = nullptr (parallel decoding, no autoregressive KV cache) ──
  dream:          { name: 'dream',          categories: ['transformer'],      kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  llada:          { name: 'llada',          categories: ['transformer'],      kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'llada-moe':    { name: 'llada-moe',      categories: ['transformer', 'moe'], kvCache: noKvCache,  activations: buildActivations, moe: moeNoShared, tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] } },
  rnd1:           { name: 'rnd1',           categories: ['transformer', 'moe'], kvCache: noKvCache,  activations: buildActivations, moe: moeNoShared, tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] } },

  // ── Pure-recurrent architectures (no attention KV; recurrent state per layer) ──
  // llama-model.cpp:8451–8459 wires these to llama_memory_recurrent.
  mamba:          { name: 'mamba',          categories: ['recurrent'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mamba2:         { name: 'mamba2',         categories: ['recurrent'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  rwkv6:          { name: 'rwkv6',          categories: ['recurrent'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  rwkv6qwen2:     { name: 'rwkv6qwen2',     categories: ['recurrent'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  rwkv7:          { name: 'rwkv7',          categories: ['recurrent'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  arwkv7:         { name: 'arwkv7',         categories: ['recurrent'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },

  // ── Hybrid (per-layer attention + Mamba SSM, head_count_kv array) ──
  // llamaKvCache already skips layers with heads<=0; calcRecurrentState picks up
  // the SSM state for the complementary set.
  jamba:          { name: 'jamba',          categories: ['transformer', 'moe'],      kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  plamo2:         { name: 'plamo2',         categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },

  // ── MoE architectures that reuse the standard llama KV cache + std activations ──
  qwen3vlmoe:  { name: 'qwen3vlmoe',  categories: ['transformer', 'moe', 'vl'], kvCache: llamaKvCache, activations: buildActivations, moe: moeNoShared,   tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] } },
  bailingmoe2: { name: 'bailingmoe2', categories: ['transformer', 'moe'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { effectiveLayers: (meta, n_block) => n_block - getMeta(meta, `${meta['general.architecture']}.nextn_predict_layers`) }), activations: leadingDenseActivations, moe: (m, ti) => buildMoe(m, ti, { isShared: (t) => t.name.includes('_shexp.') }), tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'] } },
  // nemotron_h_moe: recurrent layers are those with both head_count_kv == 0 AND
  // feed_forward_length == 0 (llama-model.cpp:2257). Different predicate from
  // other hybrids — pure-FFN layers also have head_count_kv == 0 but are not
  // recurrent.
  nemotron_h_moe: { name: 'nemotron_h_moe', categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: buildActivations, moe: moeShexpOnly, recurrentLayers: nemotronHRecurrentLayers, tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_up_shexp*', '*ffn_down_shexp*'] } },
  dbrx: { name: 'dbrx', categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: buildActivations, moe: moeNoShared, tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] } },
  grovemoe:      { name: 'grovemoe',       categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: buildActivations, moe: (m, ti) => buildMoe(m, ti, { isExpert: (t) => t.name.includes('_exps.') || t.name.includes('_chexps.') || t.name.includes('exp_probs_b'), isShared: noShared }),   tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*', '*ffn_gate_chexps*', '*ffn_up_chexps*', '*ffn_down_chexps*'], router: ['*ffn_gate_inp*'], shared: [] } },

  // ── MoE architectures with leading dense blocks ──
  ernie4_5_moe: { name: 'ernie4_5-moe', categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: (m, ti) => buildMoe(m, ti, { isShared: (t) => t.name.includes('_shexp.') }), tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'] } },
  hunyuan_moe:  { name: 'hunyuan-moe',  categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  lfm2_moe:     { name: 'lfm2moe',      categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeNoShared,   tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] } },
  afmoe:        { name: 'afmoe',        categories: ['transformer', 'moe', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 4 }), activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  deepseek:     { name: 'deepseek',     categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  'deepseek2-ocr': { name: 'deepseek2-ocr', categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  bailingmoe:   { name: 'bailingmoe',   categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },

  // ── ISWA + MoE architectures ──
  // exaone-moe: llama-model.cpp:2310–2314 hardcodes n_swa = 128.
  'exaone-moe': { name: 'exaone-moe',   categories: ['transformer', 'moe', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 4, swaDefault: 128, effectiveLayers: (meta, n_block) => n_block - getMeta(meta, `${meta['general.architecture']}.nextn_predict_layers`) }), activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  step35:       { name: 'step35',        categories: ['transformer', 'moe', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true }), activations: buildActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },

  // ── GLM4 MoE: gate_up_exps fused pattern ──
  glm4moe: {
    name: 'glm4moe',
    categories: ['transformer', 'moe'],
    kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { effectiveLayers: (meta, n_block) => n_block - getMeta(meta, `${meta['general.architecture']}.nextn_predict_layers`) }),
    activations: buildActivations,
    moe: (m, ti) => buildMoe(m, ti, {
      isExpert: (t) => t.name.includes('_exps.') || t.name.includes('gate_up_exps'),
      isShared: noShared,
    }),
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_gate_up_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] },
  },

  // ── DSA (DeepSeek Sparse Attention) — MLA + MoE (GLM-5 family) ──
  'glm-dsa': {
    name: 'glm-dsa',
    categories: ['transformer', 'moe', 'mla'],
    kvCache: mlaKvCache,
    activations(meta, batchSize) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
      const leadingDense = getMeta(meta, `${arch}.leading_dense_block_count`);
      const q_lora_rank = getMeta(meta, `${arch}.attention.q_lora_rank`);
      const kv_lora_rank = getMeta(meta, `${arch}.attention.kv_lora_rank`);
      const indexerTopK = getMeta(meta, `${arch}.attention.indexer.top_k`);
      const nextn = getMeta(meta, `${arch}.nextn_predict_layers`);
      const isMoe = expertCount > 0;
      const n_layer_kv = Math.max(0, n_layer - nextn);
      const indexerBytes = indexerTopK * 256;
      const n_dense = Math.min(leadingDense, n_layer_kv);
      const moeLayers = Math.max(0, n_layer_kv - leadingDense);
      const perLayerAttn = n_embd + q_lora_rank + kv_lora_rank + indexerBytes;
      const denseFF = ffSumOverLayers(meta, arch, n_layer_kv, 0, n_dense);
      const moeFF = (isMoe && expertUsedCount > 0 && expertFF > 0)
        ? expertUsedCount * expertFF * moeLayers
        : ffSumOverLayers(meta, arch, n_layer_kv, leadingDense, leadingDense + moeLayers);
      const denseBytes = batchSize * (n_dense * perLayerAttn + denseFF) * 4;
      const moeBytes = batchSize * (moeLayers * perLayerAttn + moeFF) * 4;
      return {
        totalBytes: denseBytes + moeBytes,
        perLayerBytes: 0,
        isMoe,
        expertCount, expertUsedCount, expertFF, leadingDense,
      };
    },
    moe: moeShexpOnly,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── Gemma3N: ISWA + altup mechanism + per-layer tensors ──
  gemma3n: {
    name: 'gemma3n',
    categories: ['transformer', 'moe', 'iswa'],
    kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, {
      iswa: true,
      swaPeriodDefault: 5,
      effectiveLayers: (meta, n_block) => {
        // llama-model.cpp:1616 hardcodes n_layer_kv_from_start = 20 for gemma3n.
        // The convert script does not emit this key, so fall back to 20 when absent.
        const arch = meta['general.architecture'];
        const n_layer_kv = getMeta(meta, `${arch}.attention.layer_kv_from_start`) || 20;
        return Math.min(n_layer_kv, n_block);
      },
    }),
    activations(meta, batchSize) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
      const n_altup = getMeta(meta, `${arch}.altup_num_inputs`) || 4;
      const isMoe = expertCount > 0;
      // altup mechanism multiplies the residual stream. gemma3n's
      // feed_forward_length is a per-layer array, so sum across layers.
      const ffTotal = (isMoe && expertUsedCount > 0 && expertFF > 0)
        ? expertUsedCount * expertFF * n_layer
        : ffSumOverLayers(meta, arch, n_layer);
      const totalBytes = batchSize * (n_embd * n_altup * n_layer + ffTotal) * 4;
      return { totalBytes, perLayerBytes: n_layer > 0 ? totalBytes / n_layer : 0, isMoe, expertCount, expertUsedCount, expertFF, n_altup };
    },
    // altup_router plays the ffn_gate_inp role; per_layer_*/altup_* are shared scaffolding.
    moe: (m, ti) => buildMoe(m, ti, {
      isRouter: (t) => t.name.includes('altup_router') || t.name.includes('ffn_gate_inp'),
      isShared: (t) => t.name.includes('per_layer_') || (t.name.includes('altup_') && !t.name.includes('altup_router')),
    }),
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*altup_router*'], shared: ['*per_layer_*', '*altup_*'] },
  },
};

// ── Alias map: GGUF-returned names → registry keys ──
export const ARCH_ALIASES = {
  'ernie4_5-moe': 'ernie4_5_moe',
  'hunyuan-moe': 'hunyuan_moe',
  'lfm2moe': 'lfm2_moe',
};

// ── Get architecture handler with fallback ──
export function getArchHandler(arch) {
  const aliasKey = ARCH_ALIASES[arch];
  if (aliasKey && ARCHITECTURES[aliasKey]) return ARCHITECTURES[aliasKey];
  if (ARCHITECTURES[arch]) return ARCHITECTURES[arch];
  console.warn(`Unknown architecture "${arch}", falling back to llama handler`);
  return ARCHITECTURES.llama;
}

// ── Pattern matching for tensor groups ──
export function globMatch(pattern, str) {
  const regex = pattern
    .replace(/[.+?^${}()|[\]\\]/g, '\\$&')
    .replace(/\*/g, '.*');
  return new RegExp('^' + regex + '$').test(str);
}



// ── Memory calculations ──
export function getModelArch(metadata) {
  return metadata['general.architecture'] || 'unknown';
}

export function getMeta(metadata, key, fallback = 0) {
  const val = metadata[key];
  if (val === undefined || val === null) return fallback;
  if (Array.isArray(val)) return val;
  const n = Number(val);
  return Number.isNaN(n) ? fallback : n;
}

export function calcWeightSize(tensorInfos) {
  let total = 0;
  const byQuant = {};

  for (const t of tensorInfos) {
    const nElem = tensorElems(t);
    const bpe = tensorBpe(t);
    const size = nElem * bpe;
    total += size;

    const name = tensorQuantName(t);
    if (!byQuant[name]) {
      byQuant[name] = { count: 0, elements: 0, bytes: 0 };
    }
    byQuant[name].count++;
    byQuant[name].elements += nElem;
    byQuant[name].bytes += size;
  }

  return { total, byQuant };
}

// Compute n_embd_r / n_embd_s following llama-hparams.cpp:155–194 — branches
// by which family of recurrent state the model uses (KDA → Kimi-Linear,
// shortconv → LFM2, wkv → RWKV, ssm → Mamba/Mamba2/granite-hybrid/falcon-h1
// /nemotron_h*). Returns { n_embd_r, n_embd_s } or null when the model has no
// recurrent state.
function recurrentEmbedSizes(metadata) {
  const arch = getModelArch(metadata);
  const n_embd = getMeta(metadata, `${arch}.embedding_length`);

  // KDA (Kimi-Linear): llama-hparams.cpp:166–171 + 185–190.
  const head_dim_kda = getMeta(metadata, `${arch}.kda.head_dim`);
  if (head_dim_kda > 0) {
    const n_head = getMeta(metadata, `${arch}.attention.head_count`);
    const ssm_d_conv = getMeta(metadata, `${arch}.ssm.conv_kernel`);
    const d_inner = n_head * head_dim_kda;
    const conv_minus_1 = ssm_d_conv > 0 ? ssm_d_conv - 1 : 3;
    return {
      n_embd_r: 3 * conv_minus_1 * d_inner,
      n_embd_s: head_dim_kda * head_dim_kda * n_head,
    };
  }

  // RWKV: llama-hparams.cpp:156–159 + 180–183.
  const wkv_head_size = getMeta(metadata, `${arch}.wkv.head_size`);
  if (wkv_head_size > 0) {
    const token_shift_count = getMeta(metadata, `${arch}.token_shift_count`) || 2;
    return {
      n_embd_r: token_shift_count * n_embd,
      n_embd_s: n_embd * wkv_head_size,
    };
  }

  // LFM2 shortconv: llama-hparams.cpp:161–164 — n_embd_s = 0 (no SSM state).
  const shortconv_l = getMeta(metadata, `${arch}.shortconv.l_cache`);
  if (shortconv_l > 0) {
    return { n_embd_r: n_embd * (shortconv_l - 1), n_embd_s: 0 };
  }

  // Mamba/Mamba2-style SSM (default): llama-hparams.cpp:175–177 + 192–193.
  const ssm_d_conv = getMeta(metadata, `${arch}.ssm.conv_kernel`);
  const ssm_d_inner = getMeta(metadata, `${arch}.ssm.inner_size`);
  const ssm_d_state = getMeta(metadata, `${arch}.ssm.state_size`);
  const ssm_n_group = getMeta(metadata, `${arch}.ssm.group_count`);
  if (ssm_d_conv > 0 && ssm_d_inner > 0 && ssm_d_state > 0) {
    return {
      n_embd_r: (ssm_d_conv - 1) * (ssm_d_inner + 2 * ssm_n_group * ssm_d_state),
      n_embd_s: ssm_d_state * ssm_d_inner,
    };
  }
  return null;
}

// Default recurrent-layer count. Mirrors the per-arch recurrent_layer_arr
// initialization in llama-model.cpp. An arch handler can override via a
// `recurrentLayers(meta) -> int` function.
function defaultRecurrentLayers(metadata) {
  const arch = getModelArch(metadata);
  const n_layer = getMeta(metadata, `${arch}.block_count`);
  if (!n_layer) return 0;

  // Pure recurrent (mamba/mamba2/rwkv*/arwkv*) — every layer is recurrent.
  if (/^(mamba|rwkv|arwkv)/.test(arch)) return n_layer;

  // falcon-h1: all layers carry both attention and recurrent state
  // (llama-model.cpp:2573, std::fill(recurrent_layer_arr.begin(), end, true)).
  if (arch === 'falcon-h1') return n_layer;

  // Per-layer head_count_kv array — recurrent where heads == 0
  // (jamba/plamo2/lfm2/lfm2moe/granitehybrid/kimi-linear/nemotron_h*).
  const head_count_kv = metadata[`${arch}.attention.head_count_kv`];
  if (Array.isArray(head_count_kv)) {
    let count = 0;
    for (let i = 0; i < Math.min(head_count_kv.length, n_layer); i++) {
      if (Number(head_count_kv[i]) === 0) count++;
    }
    return count;
  }

  // qwen35-family: recurrent unless (i+1) % full_attention_interval == 0.
  const interval = getMeta(metadata, `${arch}.full_attention_interval`);
  if (interval > 0) return n_layer - Math.floor(n_layer / interval);
  return 0;
}

export function calcRecurrentState(metadata, nSeqMax = 1) {
  const sizes = recurrentEmbedSizes(metadata);
  if (!sizes) return null;
  const handler = getArchHandler(getModelArch(metadata));
  const n_recurrent = (handler && typeof handler.recurrentLayers === 'function')
    ? handler.recurrentLayers(metadata)
    : defaultRecurrentLayers(metadata);
  if (!n_recurrent || n_recurrent <= 0) return null;
  const { n_embd_r, n_embd_s } = sizes;
  // llama-model.cpp:8452–8459 / 8488–8495 — recurrent state always uses F32.
  const convStateBytes = n_recurrent * n_embd_r * nSeqMax * 4;
  const ssmStateBytes = n_recurrent * n_embd_s * nSeqMax * 4;
  return {
    convStateBytes, ssmStateBytes,
    totalBytes: convStateBytes + ssmStateBytes,
    recurrentLayers: n_recurrent,
    n_embd_r, n_embd_s,
  };
}

export function calcKVCache(metadata, ctxSize, kvTypeK, kvTypeV) {
  const arch = getModelArch(metadata);
  const handler = getArchHandler(arch);
  const result = handler.kvCache(metadata, ctxSize, kvTypeK, kvTypeV);
  const recurrent = calcRecurrentState(metadata);
  result.bytesRecurrent = recurrent ? recurrent.totalBytes : 0;
  result.recurrentLayers = recurrent ? recurrent.recurrentLayers : 0;
  result.totalBytes = result.bytesK + result.bytesV + result.bytesRecurrent;
  return result;
}

export function calcActivations(metadata, batchSize) {
  const arch = getModelArch(metadata);
  const handler = getArchHandler(arch);
  return handler.activations(metadata, batchSize);
}

export function calcMoEInfo(metadata, tensorInfos) {
  const arch = getModelArch(metadata);
  const handler = getArchHandler(arch);
  return handler.moe(metadata, tensorInfos);
}

// ── Multimodal projector (mmproj) ──
// Mirrors llama.cpp/tools/mtmd/clip.cpp:2829 clip_n_output_tokens() — a static
// estimate of how many tokens the projector emits per image. Assumes a square
// input at the model's declared image_size; real inputs may differ for
// dynamic-resolution projectors. Audio projectors return 0 because their
// patch count depends on the runtime audio length.
const AUDIO_PROJ_TYPES = new Set([
  'ultravox', 'voxtral', 'qwen2a', 'qwen3a', 'glma', 'lfm2a',
  'gemma4a', 'gemma3na', 'meralion', 'musicflamingo',
]);

function estimateOutputTokens(projType, p) {
  const { imageSize, patchSize, nMerge, minicpmvQ, minicpmvV } = p;
  const type = (projType || '').toLowerCase();
  if (AUDIO_PROJ_TYPES.has(type)) return 0; // runtime-dependent
  if (type === 'resampler') {
    if (minicpmvQ > 0) return minicpmvQ;
    if (minicpmvV === 2) return 96;
    if (minicpmvV >= 3) return 64;
    return 64;
  }
  if (!imageSize || !patchSize) return 0;
  const perSide = Math.floor(imageSize / patchSize);
  const nPatches = perSide * perSide;
  const merge = nMerge > 0 ? nMerge : 1;

  switch (type) {
    case 'mlp':
    case 'mlp_norm':
    case 'phi4':
    case 'pixtral':
    case 'lightonocr':
    case 'janus_pro':
      return nPatches;
    case 'dots_ocr':
    case 'paddleocr': {
      const stride = merge * merge;
      return Math.ceil(nPatches / stride);
    }
    case 'ldp':
    case 'ldpv2':
    case 'adapter':
      return Math.floor(nPatches / 4);
    case 'qwen2vl_merger':
    case 'qwen2.5vl_merger':
    case 'qwen3vl_merger':
    case 'glm4v':
    case 'youtuvl': {
      const x = Math.floor(imageSize / (patchSize * 2));
      return x * x;
    }
    case 'step3vl': {
      const x = Math.floor(imageSize / (patchSize * merge));
      return x * x;
    }
    case 'gemma3':
    case 'gemma4v':
    case 'idefics3':
    case 'internvl':
    case 'nemotron_v2_vl':
    case 'llama4':
      return Math.floor(nPatches / (merge * merge));
    case 'gemma3nv':
      return perSide;
    case 'lfm2':
    case 'kimivl':
    case 'kimik25': {
      const out = patchSize * merge;
      const x = Math.ceil(imageSize / out);
      return x * x;
    }
    case 'cogvlm':
      return nPatches + 2;
    case 'deepseekocr': {
      const reduced = Math.floor(nPatches / 16);
      const h = Math.floor(Math.sqrt(reduced));
      return h * (h + 1) + 1;
    }
    case 'hunyuanocr': {
      const ow = Math.floor(perSide / merge);
      const oh = ow;
      return (ow + 1) * oh + 2;
    }
    case 'yasa2':
      return 64;
    default:
      return nPatches; // generic fallback
  }
}

const KNOWN_PROJ_TYPES = new Set([
  'mlp', 'mlp_norm', 'phi4', 'pixtral', 'lightonocr', 'janus_pro',
  'dots_ocr', 'paddleocr', 'ldp', 'ldpv2', 'adapter', 'resampler',
  'qwen2vl_merger', 'qwen2.5vl_merger', 'qwen3vl_merger', 'glm4v', 'youtuvl',
  'step3vl', 'gemma3', 'gemma4v', 'idefics3', 'internvl', 'nemotron_v2_vl',
  'llama4', 'gemma3nv', 'lfm2', 'kimivl', 'kimik25', 'cogvlm', 'deepseekocr',
  'hunyuanocr', 'yasa2', 'qwen2.5o', ...AUDIO_PROJ_TYPES,
]);

export function calcMmProj(metadata, tensorInfos) {
  const arch = metadata['general.architecture'];
  const hasVision = !!metadata['clip.has_vision_encoder'];
  const hasAudio = !!metadata['clip.has_audio_encoder'];
  const projType = metadata['clip.projector_type']
    || metadata['clip.vision.projector_type']
    || metadata['clip.audio.projector_type']
    || null;
  if (!hasVision && !hasAudio && arch !== 'clip') return null;

  const weights = calcWeightSize(tensorInfos);
  const imageSize = getMeta(metadata, 'clip.vision.image_size');
  const patchSize = getMeta(metadata, 'clip.vision.patch_size');
  const nEmbdV = getMeta(metadata, 'clip.vision.embedding_length');
  const nLayerV = getMeta(metadata, 'clip.vision.block_count');
  const projDim = getMeta(metadata, 'clip.vision.projection_dim') || nEmbdV;
  const nMerge = getMeta(metadata, 'clip.vision.spatial_merge_size') || getMeta(metadata, 'clip.vision.projector.scale_factor') || 1;
  const minicpmvQ = getMeta(metadata, 'clip.minicpmv_query_num');
  const minicpmvV = getMeta(metadata, 'clip.minicpmv_version');

  const projTypeKnown = projType != null && KNOWN_PROJ_TYPES.has(projType.toLowerCase());
  const isAudioProj = projType != null && AUDIO_PROJ_TYPES.has(projType.toLowerCase());
  const nOutputTokens = estimateOutputTokens(projType, {
    imageSize, patchSize, nMerge, minicpmvQ, minicpmvV,
  });
  const perImageActBytes = nOutputTokens > 0 && projDim > 0
    ? nOutputTokens * projDim * 4
    : 0;

  return {
    isMmProj: true,
    hasVision, hasAudio, isAudioProj,
    projType, projTypeKnown,
    name: metadata['general.name'] || null,
    weightBytes: weights.total,
    byQuant: weights.byQuant,
    imageSize, patchSize, nEmbdV, nLayerV, projDim, nMerge,
    nOutputTokens, perImageActBytes,
  };
}

// ── Performance (tokens/sec) estimator ──
// Speed-of-light per-layer throughput model: each token's per-layer latency is
// max(FLOPs/FLOPS, bytes/BW) on the device hosting that layer. Total decode
// latency is the sum across layers (dense partial offload and MoE active-
// expert offload both fall out of this formulation). See
// gguf-parser-go/file_estimate__llamacpp.go:883-909 for the reference.

function layerIndexFromTensorName(name) {
  const m = /^blk\.(\d+)\./.exec(name);
  return m ? parseInt(m[1], 10) : -1;
}

// Compute per-layer byte/element footprint and global (non-block) output
// tensors. For MoE layers, expert tensors are split into full (all experts,
// used for storage fit) vs active (expertUsedCount / expertCount, used for
// per-token streamed totals). Non-expert tensors (attn, router, shared, norms,
// non-MoE FFN) are tracked separately so callers can model the llama.cpp
// `--n-cpu-moe` mode where experts live in RAM but non-expert parts run on GPU.
export function calcPerLayerFootprint(metadata, tensorInfos, kv, moe) {
  const arch = getModelArch(metadata);
  const handler = getArchHandler(arch);
  const nLayers = getMeta(metadata, `${arch}.block_count`) || (kv ? kv.layers : 0);
  const expertCount = moe ? moe.expertCount : 0;
  const expertUsed = moe ? moe.expertUsedCount : 0;
  const expertFrac = expertCount > 0 ? expertUsed / expertCount : 0;
  const expertPatterns = handler.tensorGroups ? (handler.tensorGroups.expert || []) : [];
  const isExpertTensor = (name) => expertPatterns.some(p => globMatch(p, name));

  const layerBytes = new Array(nLayers).fill(0);
  const layerElems = new Array(nLayers).fill(0);
  const layerNonExpertBytes = new Array(nLayers).fill(0);
  const layerNonExpertElems = new Array(nLayers).fill(0);
  const layerExpertBytesFull = new Array(nLayers).fill(0);
  const layerExpertElemsFull = new Array(nLayers).fill(0);
  const layerExpertBytesActive = new Array(nLayers).fill(0);
  const layerExpertElemsActive = new Array(nLayers).fill(0);
  let outputBytes = 0, outputElems = 0;

  for (const t of tensorInfos) {
    const idx = layerIndexFromTensorName(t.name);
    const elems = tensorElems(t);
    const bytes = elems * tensorBpe(t);
    if (idx < 0 || idx >= nLayers) {
      outputBytes += bytes;
      outputElems += elems;
      continue;
    }
    layerBytes[idx] += bytes;
    layerElems[idx] += elems;
    if (expertCount > 0 && isExpertTensor(t.name)) {
      layerExpertBytesFull[idx] += bytes;
      layerExpertElemsFull[idx] += elems;
      layerExpertBytesActive[idx] += bytes * expertFrac;
      layerExpertElemsActive[idx] += elems * expertFrac;
    } else {
      layerNonExpertBytes[idx] += bytes;
      layerNonExpertElems[idx] += elems;
    }
  }

  // Derived convenience arrays: per-token streamed bytes/elems (non-expert +
  // active experts). Kept for callers and bottleneck-diagnostic aggregates.
  const layerActiveBytes = new Array(nLayers);
  const layerActiveElems = new Array(nLayers);
  for (let i = 0; i < nLayers; i++) {
    layerActiveBytes[i] = layerNonExpertBytes[i] + layerExpertBytesActive[i];
    layerActiveElems[i] = layerNonExpertElems[i] + layerExpertElemsActive[i];
  }

  const kvOnlyBytes = kv ? (kv.bytesK + kv.bytesV) : 0;
  const kvBytesPerLayer = nLayers > 0 ? kvOnlyBytes / nLayers : 0;
  const recurrentBytesPerLayer = kv && kv.bytesRecurrent ? kv.bytesRecurrent / nLayers : 0;

  return {
    nLayers,
    layerBytes, layerActiveBytes,
    layerElems, layerActiveElems,
    layerNonExpertBytes, layerNonExpertElems,
    layerExpertBytesFull, layerExpertElemsFull,
    layerExpertBytesActive, layerExpertElemsActive,
    kvBytesPerLayer, recurrentBytesPerLayer,
    outputBytes, outputElems,
    hasExperts: expertCount > 0,
  };
}

// Greedy VRAM fill matching llama.cpp's layer offloading semantics
// (`llama_params_fit_impl` in llama.cpp/src/llama.cpp):
//   'gpu'    — full layer (including all experts) resident in VRAM
//   'hybrid' — non-expert weights + KV in VRAM, ALL expert weights in RAM;
//              expert matmul runs on CPU (llama.cpp `--cpu-moe` / `--n-cpu-moe`)
//   'cpu'    — whole layer spills to CPU
//
// Layers are offloaded back-to-front (highest-numbered layers first), matching
// llama.cpp's `i_gpu_start = max(n_layer + 1 - n_gpu_layers, 0)`.
//
// cpuMoe:    if true, ALL MoE expert tensors across all layers go to CPU RAM
//            (llama.cpp `--cpu-moe` / `-cmoe`)
// nCpuMoe:   if > 0, MoE expert tensors for layers 0..N-1 go to CPU RAM
//            (llama.cpp `--n-cpu-moe N` / `-ncmoe N`).
//
// Auto-fit (`--fit on`, the llama.cpp default) for MoE models always runs a
// two-pass algorithm — `--cpu-moe` / `--n-cpu-moe` only restrict pass 2:
//   Pass 1: Fill all layers as dense-only back-to-front (experts on CPU,
//           non-expert + KV in VRAM). Stops at the first layer that doesn't
//           fit; earlier layers spill to CPU entirely.
//   Pass 2: Convert dense-only layers to full (move experts back to VRAM)
//           front-to-back, skipping layers forced hybrid by --cpu-moe /
//           --n-cpu-moe.
export function computeOffloadSplit({
  vramBytes, footprint, activationBytes = 0, nLayerOverride,
  cpuMoe = false, nCpuMoe = 0, unifiedMemory = false,
}) {
  const {
    nLayers, kvBytesPerLayer, recurrentBytesPerLayer, outputBytes,
    layerNonExpertBytes, layerExpertBytesFull,
  } = footprint;

  const modes = new Array(nLayers).fill('cpu');

  if (unifiedMemory) {
    const n = nLayerOverride != null && nLayerOverride !== 'auto'
      ? Math.max(0, Math.min(nLayers, Number(nLayerOverride)))
      : nLayers;
    const gpuStart = nLayers - n;
    for (let i = gpuStart; i < nLayers; i++) modes[i] = 'gpu';
    return {
      nGpuLayers: n, nHybridLayers: 0, nCpuLayers: nLayers - n,
      auto: nLayerOverride == null || nLayerOverride === 'auto',
      modes, cpuMoe: false, nCpuMoe: 0,
    };
  }
  const hasExpert = (i) => (layerExpertBytesFull[i] || 0) > 0;

  const shouldHybridize = (i) => {
    if (!hasExpert(i)) return false;
    if (cpuMoe) return true;
    if (nCpuMoe > 0 && i < nCpuMoe) return true;
    return false;
  };

  const rB = recurrentBytesPerLayer || 0;
  const gpuNeed = (i) => layerNonExpertBytes[i] + layerExpertBytesFull[i] + kvBytesPerLayer + rB;
  const hybridNeed = (i) => layerNonExpertBytes[i] + kvBytesPerLayer + rB;

  // Manual --ngl override: last N layers to GPU (back-to-front), then apply
  // expert placement overrides for cpuMoe / nCpuMoe.
  if (nLayerOverride != null && nLayerOverride !== 'auto') {
    const n = Math.max(0, Math.min(nLayers, Number(nLayerOverride)));
    const gpuStart = nLayers - n;
    for (let i = gpuStart; i < nLayers; i++) {
      modes[i] = shouldHybridize(i) ? 'hybrid' : 'gpu';
    }
    const nGpu = modes.filter(m => m === 'gpu').length;
    const nHyb = modes.filter(m => m === 'hybrid').length;
    return {
      nGpuLayers: nGpu, nHybridLayers: nHyb, nCpuLayers: nLayers - nGpu - nHyb,
      auto: false, modes, cpuMoe, nCpuMoe,
    };
  }

  if (!vramBytes || vramBytes <= 0) {
    return {
      nGpuLayers: 0, nHybridLayers: 0, nCpuLayers: nLayers,
      auto: true, modes, cpuMoe, nCpuMoe,
    };
  }

  const reserved = outputBytes + activationBytes;
  const hasExperts = footprint.hasExperts;

  if (!hasExperts) {
    // Dense model: simple back-to-front fill, no hybridization possible.
    let remaining = vramBytes - reserved;
    let nGpu = 0;
    for (let i = nLayers - 1; i >= 0; i--) {
      const need = gpuNeed(i);
      if (remaining >= need) {
        modes[i] = 'gpu';
        remaining -= need;
        nGpu++;
      } else {
        break;
      }
    }
    return {
      nGpuLayers: nGpu, nHybridLayers: 0, nCpuLayers: nLayers - nGpu,
      auto: true, modes, cpuMoe, nCpuMoe,
    };
  }

  // MoE model — always run llama.cpp's two-pass `--fit on` algorithm:
  //   Pass 1: dense-only fill back-to-front.
  //   Pass 2: upgrade non-forced layers to full gpu front-to-back.
  // --cpu-moe / --n-cpu-moe only restrict which layers pass 2 may upgrade;
  // the same algorithm runs whether or not those flags are set.
  let remaining = vramBytes - reserved;

  // Pass 1: dense-only fill back-to-front. Dense layers use gpuNeed,
  // MoE layers use hybridNeed (experts not counted toward VRAM).
  const layerModes = new Array(nLayers).fill('cpu');
  for (let i = nLayers - 1; i >= 0; i--) {
    const need = hasExpert(i) ? hybridNeed(i) : gpuNeed(i);
    if (remaining >= need) {
      layerModes[i] = hasExpert(i) ? 'hybrid' : 'gpu';
      remaining -= need;
    } else {
      break;
    }
  }

  // Pass 2: upgrade hybrid layers to full gpu (add experts back to VRAM)
  // front-to-back, but only for layers NOT forced by --cpu-moe / --n-cpu-moe.
  for (let i = 0; i < nLayers; i++) {
    if (layerModes[i] !== 'hybrid') continue;
    if (shouldHybridize(i)) continue;
    const expertBytes = layerExpertBytesFull[i];
    if (remaining >= expertBytes) {
      remaining -= expertBytes;
      layerModes[i] = 'gpu';
    }
  }

  // Count and assign
  for (let i = 0; i < nLayers; i++) modes[i] = layerModes[i];
  const nGpu = modes.filter(m => m === 'gpu').length;
  const nHyb = modes.filter(m => m === 'hybrid').length;

  return {
    nGpuLayers: nGpu, nHybridLayers: nHyb, nCpuLayers: nLayers - nGpu - nHyb,
    auto: true, modes, cpuMoe, nCpuMoe,
  };
}

// Top-level VRAM/RAM memory breakdown for display, matching llama.cpp's
// default behaviour: ALL weights (including all experts) go to VRAM for a
// fully-offloaded model. The cpuMoe / nCpuMoe flags shift expert weights
// to RAM, mirroring llama.cpp's --cpu-moe / --n-cpu-moe flags.
export function calcMemoryBreakdown({ weights, moe, kv, activations, footprint, cpuMoe = false, nCpuMoe = 0 }) {
  let vramWeightBytes, ramExpertBytes = 0;

  if (!moe) {
    vramWeightBytes = weights.total;
  } else if (cpuMoe) {
    vramWeightBytes = weights.total - moe.expertWeightBytes;
    ramExpertBytes = moe.expertWeightBytes;
  } else if (nCpuMoe > 0 && footprint) {
    const expertPerLayer = [];
    for (let i = 0; i < footprint.nLayers; i++) {
      expertPerLayer.push(footprint.layerExpertBytesFull[i] || 0);
    }
    const cpuExpertLayers = Math.min(nCpuMoe, footprint.nLayers);
    for (let i = 0; i < cpuExpertLayers; i++) {
      ramExpertBytes += expertPerLayer[i];
    }
    vramWeightBytes = weights.total - ramExpertBytes;
  } else {
    vramWeightBytes = weights.total;
    ramExpertBytes = 0;
  }

  const vramBytes = vramWeightBytes + kv.totalBytes + activations.totalBytes;

  return {
    vramBytes,
    ramBytes: ramExpertBytes,
    vramWeightBytes,
    ramExpertBytes,
  };
}

// Given a VRAM budget, compute the actual offload split and derive real
// VRAM/RAM usage. Unlike calcMemoryBreakdown (theoretical full-offload),
// this answers "given X GiB VRAM, what actually goes where?"
export function calcActualMemory({ vramBytes, footprint, activationBytes = 0, nLayerOverride, cpuMoe = false, nCpuMoe = 0, unifiedMemory = false }) {
  const split = computeOffloadSplit({ vramBytes, footprint, activationBytes, nLayerOverride, cpuMoe, nCpuMoe, unifiedMemory });

  let actualVram = footprint.outputBytes + activationBytes;
  let actualRam = 0;

  const rB = footprint.recurrentBytesPerLayer || 0;
  for (let i = 0; i < split.modes.length; i++) {
    const mode = split.modes[i];
    if (mode === 'gpu') {
      actualVram += footprint.layerNonExpertBytes[i] + footprint.layerExpertBytesFull[i] + footprint.kvBytesPerLayer + rB;
    } else if (mode === 'hybrid') {
      actualVram += footprint.layerNonExpertBytes[i] + footprint.kvBytesPerLayer + rB;
      actualRam += footprint.layerExpertBytesFull[i];
    } else {
      actualRam += footprint.layerNonExpertBytes[i] + footprint.layerExpertBytesFull[i] + footprint.kvBytesPerLayer + rB;
    }
  }

  return {
    ...split,
    actualVram,
    actualRam,
  };
}

// Estimate decode / prefill / TTFT for a given hardware topology.
//
// device = {
//   gpu: { flopsFp16Tflops, bwGBps, vramBytes },
//   cpu: { flopsFp16Tflops, bwGBps } | null,
//   nGpuLayers: number | 'auto',
//   cpuMoe: boolean   (llama.cpp --cpu-moe: all experts to CPU)
//   nCpuMoe: number   (llama.cpp --n-cpu-moe N: first N layers' experts to CPU)
// }
function layerTiming(params, wBytes, kvB, ctx, flopsRate, bwRate) {
  const flopsDec = 2 * params;
  const bytesDec = wBytes + kvB;
  const flopsPre = 2 * params * ctx;
  const bytesPre = wBytes + kvB;
  return {
    tDecode: Math.max(flopsDec / flopsRate, bytesDec / bwRate),
    tPrefill: Math.max(flopsPre / flopsRate, bytesPre / bwRate),
    flopsTime: flopsDec / flopsRate,
    bwTime: bytesDec / bwRate,
  };
}

export function estimatePerformance({
  metadata, tensorInfos, ctx, batchSize = 1,
  kv, moe, activations, mmproj, device,
}) {
  const footprint = calcPerLayerFootprint(metadata, tensorInfos, kv, moe);
  const actBytes = activations ? activations.totalBytes : 0;
  const mmprojBytes = mmproj ? (mmproj.weightBytes + (mmproj.perImageActBytes || 0)) : 0;

  const gpuFlops = device.gpu.flopsFp16Tflops * 1e12;
  const gpuBw = device.gpu.bwGBps * 1e9;
  const cpu = device.cpu;
  const cpuFlops = cpu ? cpu.flopsFp16Tflops * 1e12 : 0;
  const cpuBw = cpu ? cpu.bwGBps * 1e9 : 0;

  const vramBytes = device.gpu.vramBytes || 0;
  const reservedGpuBytes = actBytes + (device.mmprojOnGpu !== false ? mmprojBytes : 0);
  const unifiedMemory = !!device.unifiedMemory;
  const split = computeOffloadSplit({
    vramBytes, footprint,
    activationBytes: reservedGpuBytes,
    nLayerOverride: device.nGpuLayers,
    cpuMoe: device.cpuMoe || false,
    nCpuMoe: device.nCpuMoe || 0,
    unifiedMemory,
  });
  const { nGpuLayers, nHybridLayers, nCpuLayers, auto, modes } = split;

  const {
    nLayers, layerActiveBytes, layerActiveElems,
    layerNonExpertBytes, layerNonExpertElems,
    layerExpertBytesActive, layerExpertElemsActive,
    layerExpertBytesFull,
    kvBytesPerLayer, outputBytes, outputElems,
  } = footprint;

  let tDecodeGpu = 0, tDecodeCpu = 0, tDecodeHybridGpu = 0, tDecodeHybridCpu = 0;
  let tPrefillGpu = 0, tPrefillCpu = 0, tPrefillHybridGpu = 0, tPrefillHybridCpu = 0;
  let gpuFlopsTime = 0, gpuBwTime = 0, cpuFlopsTime = 0, cpuBwTime = 0;

  const cpuAvailable = cpu && cpuFlops > 0 && cpuBw > 0;
  const kvB = kvBytesPerLayer;

  for (let i = 0; i < nLayers; i++) {
    const mode = modes[i];
    if (mode === 'gpu') {
      const t = layerTiming(layerActiveElems[i], layerActiveBytes[i], kvB, ctx, gpuFlops, gpuBw);
      tDecodeGpu += t.tDecode;
      tPrefillGpu += t.tPrefill;
      gpuFlopsTime += t.flopsTime;
      gpuBwTime += t.bwTime;
    } else if (mode === 'hybrid' && cpuAvailable) {
      const g = layerTiming(layerNonExpertElems[i], layerNonExpertBytes[i], kvB, ctx, gpuFlops, gpuBw);
      tDecodeHybridGpu += g.tDecode;
      tPrefillHybridGpu += g.tPrefill;
      gpuFlopsTime += g.flopsTime;
      gpuBwTime += g.bwTime;
      const c = layerTiming(layerExpertElemsActive[i], layerExpertBytesActive[i], 0, ctx, cpuFlops, cpuBw);
      tDecodeHybridCpu += c.tDecode;
      tPrefillHybridCpu += c.tPrefill;
      cpuFlopsTime += c.flopsTime;
      cpuBwTime += c.bwTime;
    } else if (mode === 'cpu' && cpuAvailable) {
      const t = layerTiming(layerActiveElems[i], layerActiveBytes[i], kvB, ctx, cpuFlops, cpuBw);
      tDecodeCpu += t.tDecode;
      tPrefillCpu += t.tPrefill;
      cpuFlopsTime += t.flopsTime;
      cpuBwTime += t.bwTime;
    }
  }

  const hasGpuOffload = nGpuLayers + nHybridLayers > 0;
  const outFlops = hasGpuOffload ? gpuFlops : (cpuAvailable ? cpuFlops : gpuFlops);
  const outBw = hasGpuOffload ? gpuBw : (cpuAvailable ? cpuBw : gpuBw);
  const tOutDec = Math.max((2 * outputElems) / outFlops, outputBytes / outBw);
  const tOutPre = Math.max((2 * outputElems * ctx) / outFlops, outputBytes / outBw);

  const n_embd = getMeta(metadata, `${getModelArch(metadata)}.embedding_length`) || 0;
  const boundaryBytes = n_embd * 2;
  const boundaryBw = Math.min(32e9, cpuBw || 32e9);
  const spillBoundaryHops = (nGpuLayers + nHybridLayers > 0 && nCpuLayers > 0) ? 1 : 0;
  const hybridHops = 2 * nHybridLayers;
  const totalHops = spillBoundaryHops + hybridHops;
  const tBoundaryDec = totalHops > 0 ? (totalHops * boundaryBytes) / boundaryBw : 0;
  const tBoundaryPre = tBoundaryDec * ctx;

  const tDecode = tDecodeGpu + tDecodeCpu + tDecodeHybridGpu + tDecodeHybridCpu + tOutDec + tBoundaryDec;
  const tPrefill = tPrefillGpu + tPrefillCpu + tPrefillHybridGpu + tPrefillHybridCpu + tOutPre + tBoundaryPre;

  const gpuBottleneck = (nGpuLayers + nHybridLayers) > 0
    ? (gpuFlopsTime > gpuBwTime ? 'compute' : 'bandwidth')
    : null;
  const cpuBottleneck = (nCpuLayers > 0 || nHybridLayers > 0) && cpuAvailable
    ? (cpuFlopsTime > cpuBwTime ? 'compute' : 'bandwidth')
    : null;
  let overall;
  if ((nCpuLayers > 0 || nHybridLayers > 0) && !cpuAvailable) overall = 'cpu-layers-unrun';
  else if (nCpuLayers > 0 && cpuAvailable && tDecodeCpu > 0.5 * tDecode) overall = 'cpu-dram-spill';
  else if (nHybridLayers > 0 && cpuAvailable && tDecodeHybridCpu > 0.5 * tDecode) overall = 'cpu-experts';
  else overall = gpuBottleneck || 'n/a';

  const tHybridTotal = tDecodeHybridGpu + tDecodeHybridCpu;

  return {
    decodeTPS: tDecode > 0 ? 1 / tDecode : 0,
    prefillTPS: tPrefill > 0 ? ctx / tPrefill : 0,
    ttftSec: tPrefill,
    nGpuLayers, nHybridLayers, nCpuLayers, autoSplit: auto, cpuMoe: split.cpuMoe, nCpuMoe: split.nCpuMoe,
    perLayerMs: {
      gpu: nGpuLayers > 0 ? (tDecodeGpu / nGpuLayers) * 1000 : 0,
      hybrid: nHybridLayers > 0 ? (tHybridTotal / nHybridLayers) * 1000 : 0,
      cpu: nCpuLayers > 0 ? (tDecodeCpu / nCpuLayers) * 1000 : 0,
    },
    bottleneck: { gpu: gpuBottleneck, cpu: cpuBottleneck, overall },
    timing: {
      decodeGpu: tDecodeGpu, decodeCpu: tDecodeCpu,
      decodeHybridGpu: tDecodeHybridGpu, decodeHybridCpu: tDecodeHybridCpu,
      output: tOutDec, boundary: tBoundaryDec,
      decodeTotal: tDecode, prefillTotal: tPrefill,
    },
    footprint: {
      nLayers, outputBytes,
      avgLayerActiveBytes: nLayers > 0 ? layerActiveBytes.reduce((a, b) => a + b, 0) / nLayers : 0,
      avgLayerFullBytes: nLayers > 0 ? (layerNonExpertBytes.reduce((a, b) => a + b, 0) + layerExpertBytesFull.reduce((a, b) => a + b, 0)) / nLayers : 0,
      kvBytesPerLayer,
    },
  };
}

// ── Format helpers ──
// Uses base-2 units (1 GiB = 2^30 bytes) to match VRAM/RAM inputs.
export function formatBytes(bytes) {
  const KiB = 1024;
  const MiB = 1024 ** 2;
  const GiB = 1024 ** 3;
  const TiB = 1024 ** 4;
  if (bytes < MiB) return `${(bytes / KiB).toFixed(1)} KiB`;
  if (bytes < GiB) return `${(bytes / MiB).toFixed(1)} MiB`;
  if (bytes < TiB) return `${(bytes / GiB).toFixed(2)} GiB`;
  return `${(bytes / TiB).toFixed(2)} TiB`;
}

export function formatElements(n) {
  if (n >= 1e12) return `${(Number(n) / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(Number(n) / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(Number(n) / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(Number(n) / 1e3).toFixed(1)}K`;
  return n.toString();
}
