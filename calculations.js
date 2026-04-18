import { GGMLQuantizationType } from 'https://cdn.jsdelivr.net/npm/@huggingface/gguf@0.4.2/+esm';

// ── Bytes-per-element for each quantization type ──
// Exact values from GGML quantization block structures
const BPE = {
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
  148: 292 / 256, // Q8_K32
  149: 292 / 256, // Q8_KR8
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

// ── Standalone handler functions (reused by multiple architectures) ──
function llamaKvCache(meta, ctxSize, kvTypeK, kvTypeV) {
  const arch = meta['general.architecture'];
  const n_embd = getMeta(meta, `${arch}.embedding_length`);
  const n_head = getMeta(meta, `${arch}.attention.head_count`);
  const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
  const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
  const n_head_kv = getMeta(meta, `${arch}.attention.head_count_kv`);
  const n_layer = getMeta(meta, `${arch}.block_count`);
  const n_head_kv_arr = Array(n_layer).fill(n_head_kv);
  let totalElemsK = 0, totalElemsV = 0;
  for (let i = 0; i < n_layer; i++) {
    totalElemsK += n_embd_head_k * n_head_kv_arr[i] * ctxSize;
    totalElemsV += n_embd_head_v * n_head_kv_arr[i] * ctxSize;
  }
  return {
    bytesK: totalElemsK * (BPE[kvTypeK] || 0),
    bytesV: totalElemsV * (BPE[kvTypeV] || 0),
    totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
    headsV: n_embd_head_v,
    totalHeadsKV: n_head_kv_arr.reduce((a, b) => a + b, 0),
    avgHeadsKV: n_head_kv_arr.length > 0 ? n_head_kv_arr.reduce((a, b) => a + b, 0) / n_head_kv_arr.length : 0,
  };
}

function llamaActivations(meta, batchSize) {
  const arch = meta['general.architecture'];
  const n_embd = getMeta(meta, `${arch}.embedding_length`);
  const n_ff = getMeta(meta, `${arch}.feed_forward_length`);
  const n_layer = getMeta(meta, `${arch}.block_count`);
  const expertCount = getMeta(meta, `${arch}.expert_count`);
  const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
  const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
  const isMoe = expertCount > 0;
  let perLayerBytes;
  if (isMoe && expertUsedCount > 0 && expertFF > 0) {
    perLayerBytes = batchSize * (n_embd + expertUsedCount * expertFF);
  } else {
    perLayerBytes = batchSize * (n_embd + n_ff);
  }
  return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF };
}

function llamaMoe(meta, tensorInfos) {
  const arch = meta['general.architecture'];
  const expertCount = getMeta(meta, `${arch}.expert_count`);
  const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
  if (expertCount === 0) return null;
  const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.') || t.name.includes('exp_probs_b'));
  const routerTensor = tensorInfos.find(t => t.name.includes('ffn_gate_inp'));
  const sharedTensors = tensorInfos.filter(t => t.name.includes('_shexp.') || t.name.includes('_chexp.'));
  let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
  for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
  if (routerTensor) { const n = routerTensor.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes = n * (BPE[routerTensor.dtype] || 0); }
  for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
  const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
  const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
  const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
  const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
  return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
}

const LLAMA_TENSOR_GROUPS = { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*', '*exp_probs_b*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'] };

// ── Standalone MLA KV cache handler (shared by deepseek2 and glm-dsa) ──
function mlaKvCache(meta, ctxSize, kvTypeK, kvTypeV) {
  const arch = meta['general.architecture'];
  const kv_lora_rank = getMeta(meta, `${arch}.attention.kv_lora_rank`);
  const n_rot = getMeta(meta, `${arch}.rope.dimension_count`);
  const n_layer = getMeta(meta, `${arch}.block_count`);
  const totalElemsK = n_layer * (kv_lora_rank + n_rot) * ctxSize;
  return {
    bytesK: totalElemsK * (BPE[kvTypeK] || 0),
    bytesV: 0,
    totalBytes: 0, layers: n_layer, headsK: kv_lora_rank + n_rot,
    headsV: 0,
    totalHeadsKV: kv_lora_rank + n_rot,
    avgHeadsKV: (kv_lora_rank + n_rot) / n_layer,
  };
}

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
      // MLA: attention output = kv_lora_rank (compressed), not n_embd
      let perLayerBytes;
      if (isMoe && expertUsedCount > 0 && expertFF > 0) {
        perLayerBytes = batchSize * (n_embd + q_lora_rank + kv_lora_rank + expertUsedCount * expertFF);
      } else {
        perLayerBytes = batchSize * (n_embd + q_lora_rank + kv_lora_rank + n_ff);
      }
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      // DeepSeek2 MoE: _exps. for expert weights, _shexp. for shared, exp_probs_b bias
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.') || t.name.includes('exp_probs_b'));
      const routerTensor = tensorInfos.find(t => t.name.includes('ffn_gate_inp'));
      const sharedTensors = tensorInfos.filter(t => t.name.includes('_shexp.'));
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      if (routerTensor) { const n = routerTensor.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes = n * (BPE[routerTensor.dtype] || 0); }
      for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*', '*exp_probs_b*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'] },
  },

  // ── Gemma4: ISWA (Interleaved Sliding Window Attention) + MoE ──
  gemma4: {
    name: 'gemma4',
    categories: ['transformer', 'moe', 'iswa'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
      const n_embd_head_k_swa = getMeta(meta, `${arch}.attention.key_length_swa`) || n_embd_head_k;
      const n_embd_head_v_swa = getMeta(meta, `${arch}.attention.value_length_swa`) || n_embd_head_v;
      const n_head_kv_raw = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const n_swa = getMeta(meta, `${arch}.attention.sliding_window`);
      const swa_pattern_raw = getMeta(meta, `${arch}.attention.sliding_window_pattern`);
      const n_kv_shared = getMeta(meta, `${arch}.attention.shared_kv_layers`);
      const n_layer_kv = n_kv_shared > 0 ? n_layer - n_kv_shared : n_layer;
      const n_head_kv_arr = Array.isArray(n_head_kv_raw)
        ? (() => { const a = Array(n_layer_kv).fill(n_head[0] || 1); for (let i = 0; i < n_layer_kv; i++) if (n_head_kv_raw[i]) a[i] = Number(n_head_kv_raw[i]); return a; })()
        : Array(n_layer_kv).fill(n_head_kv_raw);
      const swa_arr = Array.isArray(swa_pattern_raw)
        ? swa_pattern_raw.map(v => Number(v) !== 0)
        : null;
      let totalElemsK = 0, totalElemsV = 0;
      for (let i = 0; i < n_layer_kv; i++) {
        const isSwa = swa_arr ? !!swa_arr[i] : false;
        const layerCtx = isSwa ? Math.min(n_swa || ctxSize, ctxSize) : ctxSize;
        const headK = isSwa ? n_embd_head_k_swa : n_embd_head_k;
        const headV = isSwa ? n_embd_head_v_swa : n_embd_head_v;
        totalElemsK += headK * n_head_kv_arr[i] * layerCtx;
        totalElemsV += headV * n_head_kv_arr[i] * layerCtx;
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: totalElemsV * (BPE[kvTypeV] || 0),
        totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
        headsV: n_embd_head_v,
        totalHeadsKV: n_head_kv_arr.reduce((a, b) => a + b, 0),
        avgHeadsKV: n_head_kv_arr.length > 0 ? n_head_kv_arr.reduce((a, b) => a + b, 0) / n_head_kv_arr.length : 0,
      };
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
      let perLayerBytes;
      if (isMoe && expertUsedCount > 0 && expertFF > 0) {
        perLayerBytes = batchSize * (n_embd + expertUsedCount * expertFF);
      } else {
        perLayerBytes = batchSize * (n_embd + n_ff);
      }
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      // Gemma4 MoE: ffn_gate_up_exps, ffn_down_exps (2 tensors per gate), ffn_gate_inp (2 tensors)
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.'));
      const routerTensors = tensorInfos.filter(t => t.name.includes('ffn_gate_inp'));
      const sharedTensors = [];
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      for (const t of routerTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes += n * (BPE[t.dtype] || 0); }
      for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
    tensorGroups: { expert: ['*ffn_gate_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] },
  },

  // ── GPT-OSS: ISWA + MoE with attn_sinks ──
  'gpt-oss': {
    name: 'gpt-oss',
    categories: ['transformer', 'moe', 'iswa'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
      const n_head_kv = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const n_swa = getMeta(meta, `${arch}.attention.sliding_window`);
      const swa_period = getMeta(meta, `${arch}.attention.sliding_window_pattern`) || 2;
      const n_head_kv_arr = Array.isArray(n_head_kv)
        ? n_head_kv.map(v => Number(v))
        : Array(n_layer).fill(n_head_kv);
      let totalElemsK = 0, totalElemsV = 0;
      for (let i = 0; i < n_layer; i++) {
        const isSwa = swa_period > 0 && (i % swa_period < (swa_period - 1));
        const layerCtx = isSwa ? Math.min(n_swa || ctxSize, ctxSize) : ctxSize;
        totalElemsK += n_embd_head_k * n_head_kv_arr[i] * layerCtx;
        totalElemsV += n_embd_head_v * n_head_kv_arr[i] * layerCtx;
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: totalElemsV * (BPE[kvTypeV] || 0),
        totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
        headsV: n_embd_head_v,
        totalHeadsKV: n_head_kv_arr.reduce((a, b) => a + b, 0),
        avgHeadsKV: n_head_kv_arr.length > 0 ? n_head_kv_arr.reduce((a, b) => a + b, 0) / n_head_kv_arr.length : 0,
      };
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
      let perLayerBytes;
      if (isMoe && expertUsedCount > 0 && expertFF > 0) {
        perLayerBytes = batchSize * (n_embd + expertUsedCount * expertFF);
      } else {
        perLayerBytes = batchSize * (n_embd + n_ff);
      }
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      // GPT-OSS: ffn_gate_exps (2 per block), ffn_up_exps (2), ffn_down_exps (2)
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.'));
      const routerTensors = tensorInfos.filter(t => t.name.includes('ffn_gate_inp'));
      const sharedTensors = [];
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      for (const t of routerTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes += n * (BPE[t.dtype] || 0); }
      for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] },
  },

  // ── Llama4: ISWA + MoE ──
  llama4: {
    name: 'llama4',
    categories: ['transformer', 'moe', 'iswa'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
      const n_head_kv = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const n_swa = getMeta(meta, `${arch}.attention.sliding_window`) || 8192;
      const swa_period = getMeta(meta, `${arch}.attention.sliding_window_pattern`) || 4;
      const n_head_kv_arr = Array.isArray(n_head_kv)
        ? n_head_kv.map(v => Number(v))
        : Array(n_layer).fill(n_head_kv);
      let totalElemsK = 0, totalElemsV = 0;
      for (let i = 0; i < n_layer; i++) {
        const isSwa = swa_period > 0 && (i % swa_period < (swa_period - 1));
        const layerCtx = isSwa ? Math.min(n_swa, ctxSize) : ctxSize;
        totalElemsK += n_embd_head_k * n_head_kv_arr[i] * layerCtx;
        totalElemsV += n_embd_head_v * n_head_kv_arr[i] * layerCtx;
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: totalElemsV * (BPE[kvTypeV] || 0),
        totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
        headsV: n_embd_head_v,
        totalHeadsKV: n_head_kv_arr.reduce((a, b) => a + b, 0),
        avgHeadsKV: n_head_kv_arr.length > 0 ? n_head_kv_arr.reduce((a, b) => a + b, 0) / n_head_kv_arr.length : 0,
      };
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
      let perLayerBytes;
      if (isMoe && expertUsedCount > 0 && expertFF > 0) {
        perLayerBytes = batchSize * (n_embd + expertUsedCount * expertFF);
      } else {
        perLayerBytes = batchSize * (n_embd + n_ff);
      }
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      // Llama4 Scout: ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.'));
      const routerTensor = tensorInfos.find(t => t.name.includes('ffn_gate_inp'));
      const sharedTensors = tensorInfos.filter(t => t.name.includes('_shexp.'));
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      if (routerTensor) { const n = routerTensor.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes = n * (BPE[routerTensor.dtype] || 0); }
      for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'] },
  },

  // ── Qwen3 MoE: standard MoE with k/q norm ──
  qwen3moe: {
    name: 'qwen3moe',
    categories: ['transformer', 'moe'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
      const n_head_kv = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const n_head_kv_arr = Array(n_layer).fill(n_head_kv);
      let totalElemsK = 0, totalElemsV = 0;
      for (let i = 0; i < n_layer; i++) {
        totalElemsK += n_embd_head_k * n_head_kv_arr[i] * ctxSize;
        totalElemsV += n_embd_head_v * n_head_kv_arr[i] * ctxSize;
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: totalElemsV * (BPE[kvTypeV] || 0),
        totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
        headsV: n_embd_head_v,
        totalHeadsKV: n_head_kv_arr.reduce((a, b) => a + b, 0),
        avgHeadsKV: n_head_kv_arr.length > 0 ? n_head_kv_arr.reduce((a, b) => a + b, 0) / n_head_kv_arr.length : 0,
      };
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
      let perLayerBytes;
      if (isMoe && expertUsedCount > 0 && expertFF > 0) {
        perLayerBytes = batchSize * (n_embd + expertUsedCount * expertFF);
      } else {
        perLayerBytes = batchSize * (n_embd + n_ff);
      }
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.'));
      const routerTensor = tensorInfos.find(t => t.name.includes('ffn_gate_inp'));
      const sharedTensors = [];
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      if (routerTensor) { const n = routerTensor.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes = n * (BPE[routerTensor.dtype] || 0); }
      for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] },
  },

  // ── Qwen3.6 MoE (qwen35moe): mixed DeltaNet/attention + MoE with shared experts ──
  qwen35moe: {
    name: 'qwen35moe',
    categories: ['transformer', 'moe'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
      const n_head_kv_raw = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const full_attn_interval = getMeta(meta, `${arch}.attention.full_attention_interval`) || 4;
      const n_head_kv_arr = Array.isArray(n_head_kv_raw)
        ? n_head_kv_raw.map(v => Number(v))
        : Array(n_layer).fill(n_head_kv_raw);
      let totalElemsK = 0, totalElemsV = 0, activeLayers = 0, activeHeadsKV = 0;
      for (let i = 0; i < n_layer; i++) {
        const isFullAttn = (i + 1) % full_attn_interval === 0;
        if (isFullAttn && n_head_kv_arr[i] > 0) {
          totalElemsK += n_embd_head_k * n_head_kv_arr[i] * ctxSize;
          totalElemsV += n_embd_head_v * n_head_kv_arr[i] * ctxSize;
          activeLayers++;
          activeHeadsKV += n_head_kv_arr[i];
        }
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: totalElemsV * (BPE[kvTypeV] || 0),
        totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
        headsV: n_embd_head_v,
        totalHeadsKV: activeHeadsKV,
        avgHeadsKV: activeLayers > 0 ? activeHeadsKV / activeLayers : 0,
      };
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
      let perLayerBytes;
      if (isMoe && expertUsedCount > 0 && expertFF > 0) {
        // Shared expert (1) + routed experts (expertUsedCount), plus residual connection
        perLayerBytes = batchSize * (2 * n_embd + expertUsedCount * expertFF);
      } else {
        perLayerBytes = batchSize * (n_embd + n_ff);
      }
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.'));
      const routerTensor = tensorInfos.find(t => t.name.includes('ffn_gate_inp') && !t.name.includes('shexp'));
      const sharedTensors = tensorInfos.filter(t => t.name.includes('_shexp.') || t.name.includes('ffn_gate_inp_shexp'));
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      if (routerTensor) { const n = routerTensor.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes = n * (BPE[routerTensor.dtype] || 0); }
      for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_gate_inp_shexp*', '*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'] },
  },

  // ── Standard transformers (use llama handler logic) ──
  qwen2: { name: 'qwen2', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen3: { name: 'qwen3', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen35: { name: 'qwen35', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen3next: { name: 'qwen3next', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen2vl: { name: 'qwen2vl', categories: ['transformer', 'vl'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen3vl: { name: 'qwen3vl', categories: ['transformer', 'vl'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  gemma3: { name: 'gemma3', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  gemma2: { name: 'gemma2', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  olmo2: { name: 'olmo2', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  phi3: { name: 'phi3', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  granite: { name: 'granite', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  granitehybrid: { name: 'granitehybrid', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mistral3: { name: 'mistral3', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mistral4: { name: 'mistral4', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  glm4: { name: 'glm4', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'falcon-h1': { name: 'falcon-h1', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  deci: { name: 'deci', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  cohere2: { name: 'cohere2', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  smollm3: { name: 'smollm3', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  ernie4_5: { name: 'ernie4_5', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  grok: { name: 'grok', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'gemma-embedding': { name: 'gemma-embedding', categories: ['embedding'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  nemotron_h: { name: 'nemotron_h', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  lfm2: { name: 'lfm2', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'minimax-m2': { name: 'minimax-m2', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  seed_oss: { name: 'seed_oss', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  apertus: { name: 'apertus', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  dots1: { name: 'dots1', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  flux: { name: 'flux', categories: ['diffusion'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  ltxv: { name: 'ltxv', categories: ['diffusion'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  lumina2: { name: 'lumina2', categories: ['diffusion'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen_image: { name: 'qwen_image', categories: ['diffusion'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  wan: { name: 'wan', categories: ['diffusion'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mimo2: { name: 'mimo2', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'hunyuan-dense': { name: 'hunyuan-dense', categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },

  // ── MoE architectures ──
  qwen3vlmoe: {
    name: 'qwen3vlmoe',
    categories: ['transformer', 'moe', 'vl'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
      const n_head_kv = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const n_head_kv_arr = Array(n_layer).fill(n_head_kv);
      let totalElemsK = 0, totalElemsV = 0;
      for (let i = 0; i < n_layer; i++) {
        totalElemsK += n_embd_head_k * n_head_kv_arr[i] * ctxSize;
        totalElemsV += n_embd_head_v * n_head_kv_arr[i] * ctxSize;
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: totalElemsV * (BPE[kvTypeV] || 0),
        totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
        headsV: n_embd_head_v,
        totalHeadsKV: n_head_kv_arr.reduce((a, b) => a + b, 0),
        avgHeadsKV: n_head_kv_arr.length > 0 ? n_head_kv_arr.reduce((a, b) => a + b, 0) / n_head_kv_arr.length : 0,
      };
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
      let perLayerBytes;
      if (isMoe && expertUsedCount > 0 && expertFF > 0) {
        perLayerBytes = batchSize * (n_embd + expertUsedCount * expertFF);
      } else {
        perLayerBytes = batchSize * (n_embd + n_ff);
      }
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.'));
      const routerTensor = tensorInfos.find(t => t.name.includes('ffn_gate_inp'));
      const sharedTensors = [];
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      if (routerTensor) { const n = routerTensor.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes = n * (BPE[routerTensor.dtype] || 0); }
      for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] },
  },

  ernie4_5_moe: {
    name: 'ernie4_5-moe',
    categories: ['transformer', 'moe'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
      const n_head_kv = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const n_head_kv_arr = Array(n_layer).fill(n_head_kv);
      let totalElemsK = 0, totalElemsV = 0;
      for (let i = 0; i < n_layer; i++) {
        totalElemsK += n_embd_head_k * n_head_kv_arr[i] * ctxSize;
        totalElemsV += n_embd_head_v * n_head_kv_arr[i] * ctxSize;
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: totalElemsV * (BPE[kvTypeV] || 0),
        totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
        headsV: n_embd_head_v,
        totalHeadsKV: n_head_kv_arr.reduce((a, b) => a + b, 0),
        avgHeadsKV: n_head_kv_arr.length > 0 ? n_head_kv_arr.reduce((a, b) => a + b, 0) / n_head_kv_arr.length : 0,
      };
    },
    activations(meta, batchSize) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_ff = getMeta(meta, `${arch}.feed_forward_length`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
      const leadingDense = getMeta(meta, `${arch}.leading_dense_block_count`);
      const isMoe = expertCount > 0;
      // Dense lead layers use full FFN, MoE layers use only active experts
      const denseBytes = leadingDense * batchSize * (n_embd + n_ff) * 4;
      const moeBytes = (n_layer - leadingDense) * batchSize * (n_embd + expertUsedCount * (expertFF || 1)) * 4;
      return { totalBytes: denseBytes + moeBytes, perLayerBytes: 0, isMoe, expertCount, expertUsedCount, expertFF, leadingDense };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.'));
      const routerTensor = tensorInfos.find(t => t.name.includes('ffn_gate_inp'));
      const sharedTensors = [];
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      if (routerTensor) { const n = routerTensor.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes = n * (BPE[routerTensor.dtype] || 0); }
      for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] },
  },

  hunyuan_moe: {
    name: 'hunyuan-moe',
    categories: ['transformer', 'moe'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
      const n_head_kv = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const n_head_kv_arr = Array(n_layer).fill(n_head_kv);
      let totalElemsK = 0, totalElemsV = 0;
      for (let i = 0; i < n_layer; i++) {
        totalElemsK += n_embd_head_k * n_head_kv_arr[i] * ctxSize;
        totalElemsV += n_embd_head_v * n_head_kv_arr[i] * ctxSize;
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: totalElemsV * (BPE[kvTypeV] || 0),
        totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
        headsV: n_embd_head_v,
        totalHeadsKV: n_head_kv_arr.reduce((a, b) => a + b, 0),
        avgHeadsKV: n_head_kv_arr.length > 0 ? n_head_kv_arr.reduce((a, b) => a + b, 0) / n_head_kv_arr.length : 0,
      };
    },
    activations(meta, batchSize) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_ff = getMeta(meta, `${arch}.feed_forward_length`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
      const leadingDense = getMeta(meta, `${arch}.leading_dense_block_count`);
      const isMoe = expertCount > 0;
      const denseBytes = leadingDense * batchSize * (n_embd + n_ff) * 4;
      const moeBytes = (n_layer - leadingDense) * batchSize * (n_embd + expertUsedCount * (expertFF || 1)) * 4;
      return { totalBytes: denseBytes + moeBytes, perLayerBytes: 0, isMoe, expertCount, expertUsedCount, expertFF, leadingDense };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.'));
      const routerTensor = tensorInfos.find(t => t.name.includes('ffn_gate_inp'));
      const sharedTensors = tensorInfos.filter(t => t.name.includes('_shexp.'));
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      if (routerTensor) { const n = routerTensor.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes = n * (BPE[routerTensor.dtype] || 0); }
      for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'] },
  },

  bailingmoe2: {
    name: 'bailingmoe2',
    categories: ['transformer', 'moe'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
      const n_head_kv = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const n_head_kv_arr = Array(n_layer).fill(n_head_kv);
      let totalElemsK = 0, totalElemsV = 0;
      for (let i = 0; i < n_layer; i++) {
        totalElemsK += n_embd_head_k * n_head_kv_arr[i] * ctxSize;
        totalElemsV += n_embd_head_v * n_head_kv_arr[i] * ctxSize;
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: totalElemsV * (BPE[kvTypeV] || 0),
        totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
        headsV: n_embd_head_v,
        totalHeadsKV: n_head_kv_arr.reduce((a, b) => a + b, 0),
        avgHeadsKV: n_head_kv_arr.length > 0 ? n_head_kv_arr.reduce((a, b) => a + b, 0) / n_head_kv_arr.length : 0,
      };
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
      let perLayerBytes;
      if (isMoe && expertUsedCount > 0 && expertFF > 0) {
        perLayerBytes = batchSize * (n_embd + expertUsedCount * expertFF);
      } else {
        perLayerBytes = batchSize * (n_embd + n_ff);
      }
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.'));
      const routerTensor = tensorInfos.find(t => t.name.includes('ffn_gate_inp'));
      const sharedTensors = [];
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      if (routerTensor) { const n = routerTensor.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes = n * (BPE[routerTensor.dtype] || 0); }
      for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] },
  },

  lfm2_moe: {
    name: 'lfm2moe',
    categories: ['transformer', 'moe'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
      const n_head_kv = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      // LFM2/LFM2MoE: recurrent layers have n_head_kv == 0 (no KV cache)
      const n_head_kv_arr = Array.isArray(n_head_kv)
        ? n_head_kv.map(v => Number(v))
        : Array(n_layer).fill(n_head_kv);
      let totalElemsK = 0, totalElemsV = 0, activeLayers = 0, activeHeadsKV = 0;
      for (let i = 0; i < n_layer; i++) {
        if (n_head_kv_arr[i] > 0) {
          totalElemsK += n_embd_head_k * n_head_kv_arr[i] * ctxSize;
          totalElemsV += n_embd_head_v * n_head_kv_arr[i] * ctxSize;
          activeLayers++;
          activeHeadsKV += n_head_kv_arr[i];
        }
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: totalElemsV * (BPE[kvTypeV] || 0),
        totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
        headsV: n_embd_head_v,
        totalHeadsKV: activeHeadsKV,
        avgHeadsKV: activeLayers > 0 ? activeHeadsKV / activeLayers : 0,
      };
    },
    activations(meta, batchSize) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_ff = getMeta(meta, `${arch}.feed_forward_length`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
      const leadingDense = getMeta(meta, `${arch}.leading_dense_block_count`);
      const isMoe = expertCount > 0;
      const denseBytes = leadingDense * batchSize * (n_embd + n_ff) * 4;
      const moeBytes = (n_layer - leadingDense) * batchSize * (n_embd + expertUsedCount * (expertFF || 1)) * 4;
      return { totalBytes: denseBytes + moeBytes, perLayerBytes: 0, isMoe, expertCount, expertUsedCount, expertFF, leadingDense };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.'));
      const routerTensor = tensorInfos.find(t => t.name.includes('ffn_gate_inp'));
      const sharedTensors = [];
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      if (routerTensor) { const n = routerTensor.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes = n * (BPE[routerTensor.dtype] || 0); }
      for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] },
  },

  nemotron_h_moe: {
    name: 'nemotron_h_moe',
    categories: ['transformer', 'moe'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
      const n_head_kv = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const n_head_kv_arr = Array.isArray(n_head_kv)
        ? n_head_kv.map(v => Number(v))
        : Array(n_layer).fill(n_head_kv);
      let totalElemsK = 0, totalElemsV = 0, activeLayers = 0, activeHeadsKV = 0;
      for (let i = 0; i < n_layer; i++) {
        if (n_head_kv_arr[i] > 0) {
          totalElemsK += n_embd_head_k * n_head_kv_arr[i] * ctxSize;
          totalElemsV += n_embd_head_v * n_head_kv_arr[i] * ctxSize;
          activeLayers++;
          activeHeadsKV += n_head_kv_arr[i];
        }
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: totalElemsV * (BPE[kvTypeV] || 0),
        totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
        headsV: n_embd_head_v,
        totalHeadsKV: activeHeadsKV,
        avgHeadsKV: activeLayers > 0 ? activeHeadsKV / activeLayers : 0,
      };
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
      let perLayerBytes;
      if (isMoe && expertUsedCount > 0 && expertFF > 0) {
        perLayerBytes = batchSize * (n_embd + expertUsedCount * expertFF);
      } else {
        perLayerBytes = batchSize * (n_embd + n_ff);
      }
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.'));
      const routerTensor = tensorInfos.find(t => t.name.includes('ffn_gate_inp'));
      const sharedTensors = tensorInfos.filter(t => t.name.includes('_shexp.'));
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      if (routerTensor) { const n = routerTensor.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes = n * (BPE[routerTensor.dtype] || 0); }
      for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_up_shexp*', '*ffn_down_shexp*'] },
  },

  afmoe: {
    name: 'afmoe',
    categories: ['transformer', 'moe'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
      const n_head_kv = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const n_head_kv_arr = Array(n_layer).fill(n_head_kv);
      let totalElemsK = 0, totalElemsV = 0;
      for (let i = 0; i < n_layer; i++) {
        totalElemsK += n_embd_head_k * n_head_kv_arr[i] * ctxSize;
        totalElemsV += n_embd_head_v * n_head_kv_arr[i] * ctxSize;
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: totalElemsV * (BPE[kvTypeV] || 0),
        totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
        headsV: n_embd_head_v,
        totalHeadsKV: n_head_kv_arr.reduce((a, b) => a + b, 0),
        avgHeadsKV: n_head_kv_arr.length > 0 ? n_head_kv_arr.reduce((a, b) => a + b, 0) / n_head_kv_arr.length : 0,
      };
    },
    activations(meta, batchSize) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_ff = getMeta(meta, `${arch}.feed_forward_length`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
      const leadingDense = getMeta(meta, `${arch}.leading_dense_block_count`);
      const isMoe = expertCount > 0;
      const denseBytes = leadingDense * batchSize * (n_embd + n_ff) * 4;
      const moeBytes = (n_layer - leadingDense) * batchSize * (n_embd + expertUsedCount * (expertFF || 1)) * 4;
      return { totalBytes: denseBytes + moeBytes, perLayerBytes: 0, isMoe, expertCount, expertUsedCount, expertFF, leadingDense };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.'));
      const routerTensor = tensorInfos.find(t => t.name.includes('ffn_gate_inp'));
      const sharedTensors = tensorInfos.filter(t => t.name.includes('_shexp.'));
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      if (routerTensor) { const n = routerTensor.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes = n * (BPE[routerTensor.dtype] || 0); }
      for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'] },
  },

  // ── GLM4 MoE: uses ffn_gate_up_exps pattern ──
  glm4moe: {
    name: 'glm4moe',
    categories: ['transformer', 'moe'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
      const n_head_kv = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const n_head_kv_arr = Array(n_layer).fill(n_head_kv);
      let totalElemsK = 0, totalElemsV = 0;
      for (let i = 0; i < n_layer; i++) {
        totalElemsK += n_embd_head_k * n_head_kv_arr[i] * ctxSize;
        totalElemsV += n_embd_head_v * n_head_kv_arr[i] * ctxSize;
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: totalElemsV * (BPE[kvTypeV] || 0),
        totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
        headsV: n_embd_head_v,
        totalHeadsKV: n_head_kv_arr.reduce((a, b) => a + b, 0),
        avgHeadsKV: n_head_kv_arr.length > 0 ? n_head_kv_arr.reduce((a, b) => a + b, 0) / n_head_kv_arr.length : 0,
      };
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
      let perLayerBytes;
      if (isMoe && expertUsedCount > 0 && expertFF > 0) {
        perLayerBytes = batchSize * (n_embd + expertUsedCount * expertFF);
      } else {
        perLayerBytes = batchSize * (n_embd + n_ff);
      }
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.') || t.name.includes('gate_up_exps'));
      const routerTensor = tensorInfos.find(t => t.name.includes('ffn_gate_inp'));
      const sharedTensors = [];
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      if (routerTensor) { const n = routerTensor.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes = n * (BPE[routerTensor.dtype] || 0); }
      for (const t of sharedTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
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
    moe(meta, tensorInfos) {
      return null;
    },
    tensorGroups: { expert: [], router: [], shared: [] },
  },

  // ── Gemma3N: ISWA + complex altup mechanism ──
  gemma3n: {
    name: 'gemma3n',
    categories: ['transformer', 'moe', 'iswa'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const n_embd_head_k = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_embd_head_v = getMeta(meta, `${arch}.attention.value_length`) || (n_embd / n_head);
      const n_head_kv = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const n_swa = getMeta(meta, `${arch}.attention.sliding_window`);
      const n_layer_kv = getMeta(meta, `${arch}.attention.layer_kv_from_start`);
      const swa_period = getMeta(meta, `${arch}.attention.sliding_window_pattern`) || 5;
      const effectiveLayers = n_layer_kv > 0 ? Math.min(n_layer_kv, n_layer) : n_layer;
      const n_head_kv_arr = Array.isArray(n_head_kv)
        ? (() => { const a = Array(effectiveLayers).fill(n_head[0] || 1); for (let i = 0; i < effectiveLayers; i++) if (n_head_kv[i]) a[i] = Number(n_head_kv[i]); return a; })()
        : Array(effectiveLayers).fill(n_head_kv);
      let totalElemsK = 0, totalElemsV = 0;
      for (let i = 0; i < effectiveLayers; i++) {
        const isSwa = swa_period > 0 && (i % swa_period < (swa_period - 1));
        const layerCtx = isSwa ? Math.min(n_swa || ctxSize, ctxSize) : ctxSize;
        totalElemsK += n_embd_head_k * n_head_kv_arr[i] * layerCtx;
        totalElemsV += n_embd_head_v * n_head_kv_arr[i] * layerCtx;
      }
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: totalElemsV * (BPE[kvTypeV] || 0),
        totalBytes: 0, layers: n_layer, headsK: n_embd_head_k,
        headsV: n_embd_head_v,
        totalHeadsKV: n_head_kv_arr.reduce((a, b) => a + b, 0),
        avgHeadsKV: n_head_kv_arr.length > 0 ? n_head_kv_arr.reduce((a, b) => a + b, 0) / n_head_kv_arr.length : 0,
      };
    },
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
      // Gemma3n has altup mechanism that multiplies activation memory
      let perLayerBytes;
      if (isMoe && expertUsedCount > 0 && expertFF > 0) {
        perLayerBytes = batchSize * (n_embd * n_altup + expertUsedCount * expertFF);
      } else {
        perLayerBytes = batchSize * (n_embd * n_altup + n_ff);
      }
      return { totalBytes: perLayerBytes * n_layer * 4, perLayerBytes: perLayerBytes * 4, isMoe, expertCount, expertUsedCount, expertFF, n_altup };
    },
    moe(meta, tensorInfos) {
      const arch = meta['general.architecture'];
      const expertCount = getMeta(meta, `${arch}.expert_count`);
      const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
      if (expertCount === 0) return null;
      const expertTensors = tensorInfos.filter(t => t.name.includes('_exps.'));
      const routerTensors = tensorInfos.filter(t => t.name.includes('altup_router') || t.name.includes('ffn_gate_inp'));
      const sharedTensors = [];
      let expertWeightBytes = 0, routerBytes = 0, sharedBytes = 0;
      for (const t of expertTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); expertWeightBytes += n * (BPE[t.dtype] || 0); }
      for (const t of routerTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); routerBytes += n * (BPE[t.dtype] || 0); }
      // Gemma3n has per-layer tensors
      const perLayerTensors = tensorInfos.filter(t => t.name.includes('per_layer_') || t.name.includes('altup_'));
      for (const t of perLayerTensors) { const n = t.shape.map(Number).reduce((a, b) => a * b, 1); sharedBytes += n * (BPE[t.dtype] || 0); }
      const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const expertParams = expertTensors.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);
      const perExpertWeightBytes = expertCount > 0 ? expertWeightBytes / expertCount : 0;
      const activeExpertWeightBytes = perExpertWeightBytes * expertUsedCount;
      return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, totalWeightBytes: expertWeightBytes + routerBytes + sharedBytes, totalParams, expertParams, activeExpertWeightBytes };
    },
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
  // Check aliases first
  const aliasKey = ARCH_ALIASES[arch];
  if (aliasKey && ARCHITECTURES[aliasKey]) return ARCHITECTURES[aliasKey];
  if (ARCHITECTURES[arch]) return ARCHITECTURES[arch];
  // Fall back to llama (standard transformer)
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
    const nElem = t.shape.map(Number).reduce((a, b) => a * b, 1);
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

// ── Format helpers ──
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
