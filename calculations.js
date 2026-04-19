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
  // Interleaved GEMM variants
  31: 18 / 32,    // Q4_0_4_4
  32: 18 / 32,    // Q4_0_4_8
  33: 18 / 32,    // Q4_0_8_8
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
  // rotorquant KV cache quantization
  TURBO3_0: 50 / 128,
  TURBO4_0: 68 / 128,
  TURBO2_0: 34 / 128,
  PLANAR3_0: 50 / 128,
  ISO3_0: 50 / 128,
  PLANAR4_0: 68 / 128,
  ISO4_0: 68 / 128,
};

// Quantization type names for display
// Auto-populated from @huggingface/gguf package, plus manual entries for ik_llama.cpp extensions
export const QUANT_NAMES = {};
for (const [key, val] of Object.entries(GGMLQuantizationType)) {
  if (typeof val === 'number') QUANT_NAMES[val] = key;
}
// ── ik_llama.cpp extension names (labeled for clarity) ──
const IK_LLAMA_QUANT_NAMES = {
  151: 'Q8_KV (ik_llama)',
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
  31: 'Q4_0_4_4 (ik_llama)',
  32: 'Q4_0_4_8 (ik_llama)',
  33: 'Q4_0_8_8 (ik_llama)',
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
  133: 'Q6_0 (ik_llama)',
  397: 'Q8_K_R16 (ik_llama)',
};
Object.assign(QUANT_NAMES, IK_LLAMA_QUANT_NAMES);
const ROTORQUANT_QUANT_NAMES = {
  TURBO3_0: 'TURBO3_0 (rotorquant)',
  TURBO4_0: 'TURBO4_0 (rotorquant)',
  TURBO2_0: 'TURBO2_0 (rotorquant)',
  PLANAR3_0: 'PLANAR3_0 (rotorquant)',
  ISO3_0: 'ISO3_0 (rotorquant)',
  PLANAR4_0: 'PLANAR4_0 (rotorquant)',
  ISO4_0: 'ISO4_0 (rotorquant)',
};
Object.assign(QUANT_NAMES, ROTORQUANT_QUANT_NAMES);

// ── Tensor size helpers ──
function tensorElems(t) {
  return t.shape.map(Number).reduce((a, b) => a * b, 1);
}
function sumBytes(tensors) {
  let s = 0;
  for (const t of tensors) s += tensorElems(t) * (BPE[t.dtype] || 0);
  return s;
}
function sumElems(tensors) {
  let s = 0;
  for (const t of tensors) s += tensorElems(t);
  return s;
}

// ── Standard transformer KV cache (parameterized) ──
// opts.iswa               — arch has interleaved sliding-window layers
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
  const swa_pattern_raw = getMeta(meta, `${arch}.attention.sliding_window_pattern`);
  const swa_arr = Array.isArray(swa_pattern_raw) ? swa_pattern_raw.map(v => Number(v) !== 0) : null;
  const swa_period = typeof swa_pattern_raw === 'number' ? swa_pattern_raw : opts.swaPeriodDefault;

  let totalElemsK = 0, totalElemsV = 0, activeLayers = 0, activeHeadsKV = 0;
  for (let i = 0; i < n_layer; i++) {
    if (opts.layerFilter && !opts.layerFilter(i)) continue;
    const heads = n_head_kv_arr[i] || 0;
    if (heads <= 0) continue;
    const isSwa = opts.iswa
      ? (swa_arr ? !!swa_arr[i] : (swa_period > 0 && (i % swa_period < (swa_period - 1))))
      : false;
    const layerCtx = (isSwa && n_swa > 0) ? Math.min(n_swa, ctxSize) : ctxSize;
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
  const totalElemsK = n_layer * (kv_lora_rank + n_rot) * ctxSize;
  return {
    bytesK: totalElemsK * (BPE[kvTypeK] || 0),
    bytesV: 0,
    layers: n_layer,
    headDimK: kv_lora_rank + n_rot,
    headDimV: 0,
    totalHeadsKV: kv_lora_rank + n_rot,
    avgHeadsKV: (kv_lora_rank + n_rot) / n_layer,
  };
}

// ── Standard transformer activations ──
function buildActivations(meta, batchSize) {
  const arch = meta['general.architecture'];
  const n_embd = getMeta(meta, `${arch}.embedding_length`);
  const n_ff = getMeta(meta, `${arch}.feed_forward_length`);
  const n_layer = getMeta(meta, `${arch}.block_count`);
  const expertCount = getMeta(meta, `${arch}.expert_count`);
  const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
  const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
  const isMoe = expertCount > 0;
  const perLayerBytes = (isMoe && expertUsedCount > 0 && expertFF > 0)
    ? batchSize * (n_embd + expertUsedCount * expertFF)
    : batchSize * (n_embd + n_ff);
  return {
    totalBytes: perLayerBytes * n_layer * 4,
    perLayerBytes: perLayerBytes * 4,
    isMoe, expertCount, expertUsedCount, expertFF,
  };
}

// ── Leading-dense activations: dense FFN for first N layers, MoE afterwards ──
function leadingDenseActivations(meta, batchSize) {
  const arch = meta['general.architecture'];
  const n_embd = getMeta(meta, `${arch}.embedding_length`);
  const n_ff = getMeta(meta, `${arch}.feed_forward_length`);
  const n_layer = getMeta(meta, `${arch}.block_count`);
  const expertCount = getMeta(meta, `${arch}.expert_count`);
  const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
  const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
  const leadingDense = getMeta(meta, `${arch}.leading_dense_block_count`);
  const denseBytes = leadingDense * batchSize * (n_embd + n_ff) * 4;
  const moeBytes = (n_layer - leadingDense) * batchSize * (n_embd + expertUsedCount * (expertFF || 1)) * 4;
  return {
    totalBytes: denseBytes + moeBytes,
    perLayerBytes: 0,
    isMoe: expertCount > 0,
    expertCount, expertUsedCount, expertFF, leadingDense,
  };
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
    totalParams: sumElems(tensorInfos),
    expertParams: sumElems(expertTensors),
    activeExpertWeightBytes: perExpertWeightBytes * expertUsedCount,
  };
}

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

// ── Architecture Registry ──
// Each architecture declares its categories and provides specialized handlers
// for KV cache, activations, and MoE weight calculations.

const ARCHITECTURES = {
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
    activations(meta, batchSize) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_ff = getMeta(meta, `${arch}.feed_forward_length`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
      const q_lora_rank = getMeta(meta, `${arch}.attention.q_lora_rank`);
      const kv_lora_rank = getMeta(meta, `${arch}.attention.kv_lora_rank`);
      const isMoe = expertCount > 0;
      // MLA: attention output uses compressed kv_lora_rank, not n_embd
      const perLayerBytes = (isMoe && expertUsedCount > 0 && expertFF > 0)
        ? batchSize * (n_embd + q_lora_rank + kv_lora_rank + expertUsedCount * expertFF)
        : batchSize * (n_embd + q_lora_rank + kv_lora_rank + n_ff);
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF };
    },
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
      const interval = getMeta(meta, `${arch}.attention.full_attention_interval`) || 4;
      return buildKvCache(meta, ctxSize, kvTypeK, kvTypeV, {
        layerFilter: (i) => ((i + 1) % interval === 0),
      });
    },
    activations(meta, batchSize) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_ff = getMeta(meta, `${arch}.feed_forward_length`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
      const isMoe = expertCount > 0;
      // Residual + shared expert + routed experts
      const perLayerBytes = (isMoe && expertUsedCount > 0 && expertFF > 0)
        ? batchSize * (2 * n_embd + expertUsedCount * expertFF)
        : batchSize * (n_embd + n_ff);
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF };
    },
    moe: (m, ti) => buildMoe(m, ti, {
      isRouter: (t) => t.name.includes('ffn_gate_inp') && !t.name.includes('shexp'),
      isShared: (t) => t.name.includes('_shexp.') || t.name.includes('ffn_gate_inp_shexp'),
    }),
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_gate_inp_shexp*', '*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'] },
  },

  // ── Standard transformers (reuse llama handlers) ──
  qwen2:          { name: 'qwen2',          categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen3:          { name: 'qwen3',          categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen35:         { name: 'qwen35',         categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen3next:      { name: 'qwen3next',      categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen2vl:        { name: 'qwen2vl',        categories: ['transformer', 'vl'],kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen3vl:        { name: 'qwen3vl',        categories: ['transformer', 'vl'],kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  gemma3:         { name: 'gemma3',         categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  gemma2:         { name: 'gemma2',         categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  olmo2:          { name: 'olmo2',          categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  phi3:           { name: 'phi3',           categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  granite:        { name: 'granite',        categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  granitehybrid:  { name: 'granitehybrid',  categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mistral3:       { name: 'mistral3',       categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mistral4:       { name: 'mistral4',       categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  glm4:           { name: 'glm4',           categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'falcon-h1':    { name: 'falcon-h1',      categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  deci:           { name: 'deci',           categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  cohere2:        { name: 'cohere2',        categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  smollm3:        { name: 'smollm3',        categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  ernie4_5:       { name: 'ernie4_5',       categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  grok:           { name: 'grok',           categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'gemma-embedding':{ name: 'gemma-embedding', categories: ['embedding'],     kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  nemotron_h:     { name: 'nemotron_h',     categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  lfm2:           { name: 'lfm2',           categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'minimax-m2':   { name: 'minimax-m2',     categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  seed_oss:       { name: 'seed_oss',       categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  apertus:        { name: 'apertus',        categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  dots1:          { name: 'dots1',          categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  flux:           { name: 'flux',           categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  ltxv:           { name: 'ltxv',           categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  lumina2:        { name: 'lumina2',        categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen_image:     { name: 'qwen_image',     categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  wan:            { name: 'wan',            categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mimo2:          { name: 'mimo2',          categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'hunyuan-dense':{ name: 'hunyuan-dense',  categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },

  // ── MoE architectures that reuse the standard llama KV cache + std activations ──
  qwen3vlmoe:  { name: 'qwen3vlmoe',  categories: ['transformer', 'moe', 'vl'], kvCache: llamaKvCache, activations: buildActivations, moe: moeNoShared,   tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] } },
  bailingmoe2: { name: 'bailingmoe2', categories: ['transformer', 'moe'],       kvCache: llamaKvCache, activations: buildActivations, moe: moeNoShared,   tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] } },
  nemotron_h_moe: { name: 'nemotron_h_moe', categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: buildActivations, moe: moeShexpOnly, tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_up_shexp*', '*ffn_down_shexp*'] } },

  // ── MoE architectures with leading dense blocks ──
  ernie4_5_moe: { name: 'ernie4_5-moe', categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeNoShared,   tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] } },
  hunyuan_moe:  { name: 'hunyuan-moe',  categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  lfm2_moe:     { name: 'lfm2moe',      categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeNoShared,   tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] } },
  afmoe:        { name: 'afmoe',        categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },

  // ── GLM4 MoE: gate_up_exps fused pattern ──
  glm4moe: {
    name: 'glm4moe',
    categories: ['transformer', 'moe'],
    kvCache: llamaKvCache,
    activations: buildActivations,
    moe: (m, ti) => buildMoe(m, ti, {
      isExpert: (t) => t.name.includes('_exps.') || t.name.includes('gate_up_exps'),
      isShared: noShared,
    }),
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_gate_up_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] },
  },

  // ── DSA (DeepSeek Sparse Attention) — shares MLA KV cache with DeepSeek2 ──
  'glm-dsa': {
    name: 'glm-dsa',
    categories: ['transformer', 'mla'],
    kvCache: mlaKvCache,
    activations(meta, batchSize) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_ff = getMeta(meta, `${arch}.feed_forward_length`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const indexerTopK = getMeta(meta, `${arch}.attention.indexer.top_k`);
      // DSA has additional indexer state (kv_cache_indexer_k, kv_cache_indexer_v)
      const perLayerBytes = batchSize * (n_embd + n_ff + indexerTopK * 256);
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe: false, expertCount: 0, expertUsedCount: 0, expertFF: 0 };
    },
    moe: () => null,
    tensorGroups: { expert: [], router: [], shared: [] },
  },

  // ── Gemma3N: ISWA + altup mechanism + per-layer tensors ──
  gemma3n: {
    name: 'gemma3n',
    categories: ['transformer', 'moe', 'iswa'],
    kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, {
      iswa: true,
      swaPeriodDefault: 5,
      effectiveLayers: (meta, n_block) => {
        const arch = meta['general.architecture'];
        const n_layer_kv = getMeta(meta, `${arch}.attention.layer_kv_from_start`);
        return n_layer_kv > 0 ? Math.min(n_layer_kv, n_block) : n_block;
      },
    }),
    activations(meta, batchSize) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_ff = getMeta(meta, `${arch}.feed_forward_length`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
      const n_altup = getMeta(meta, `${arch}.altup_num_inputs`) || 4;
      const isMoe = expertCount > 0;
      // altup mechanism multiplies the residual stream
      const perLayerBytes = (isMoe && expertUsedCount > 0 && expertFF > 0)
        ? batchSize * (n_embd * n_altup + expertUsedCount * expertFF)
        : batchSize * (n_embd * n_altup + n_ff);
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF, n_altup };
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
const ARCH_ALIASES = {
  'qwen_image': 'qwen_image',
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

export function matchTensorGroups(tensorInfos, groups) {
  const result = { expert: [], router: [], shared: [] };
  for (const t of tensorInfos) {
    for (const [group, patterns] of Object.entries(groups)) {
      for (const p of patterns) {
        if (globMatch(p, t.name)) {
          result[group].push(t);
          break;
        }
      }
    }
  }
  return result;
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
    const bpe = BPE[t.dtype] || 0;
    const size = nElem * bpe;
    total += size;

    const name = QUANT_NAMES[t.dtype] || `type_${t.dtype}`;
    if (!byQuant[name]) {
      byQuant[name] = { count: 0, elements: 0, bytes: 0 };
    }
    byQuant[name].count++;
    byQuant[name].elements += nElem;
    byQuant[name].bytes += size;
  }

  return { total, byQuant };
}

export function calcKVCache(metadata, ctxSize, kvTypeK, kvTypeV) {
  const arch = getModelArch(metadata);
  if (arch.startsWith('mamba') || arch.startsWith('rwkv')) {
    throw new Error(`Memory estimation for architecture "${arch}" is not supported`);
  }
  const handler = getArchHandler(arch);
  const result = handler.kvCache(metadata, ctxSize, kvTypeK, kvTypeV);
  result.totalBytes = result.bytesK + result.bytesV;
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
  'hunyuanocr', ...AUDIO_PROJ_TYPES,
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
  const nMerge = getMeta(metadata, 'clip.vision.spatial_merge_size') || 1;
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

// ── Format helpers ──
// Uses base-10 units (1 GB = 1e9 bytes) to match vram/ram inputs; don't "fix" to base-2.
export function formatBytes(bytes) {
  if (bytes < 1e6) return `${(bytes / 1e3).toFixed(1)} KB`;
  if (bytes < 1e9) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes < 1e12) return `${(bytes / 1e9).toFixed(2)} GB`;
  return `${(bytes / 1e12).toFixed(2)} TB`;
}

export function formatElements(n) {
  if (n >= 1e12) return `${(Number(n) / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(Number(n) / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(Number(n) / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(Number(n) / 1e3).toFixed(1)}K`;
  return n.toString();
}
