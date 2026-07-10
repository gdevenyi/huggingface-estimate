// quant-types.js is the single source of truth for BPE/QUANT_NAMES and the
// per-fork override maps. calculations.js re-exports them for backwards
// compatibility; downstream code (parsing.js, scan-metadata.js, tests) imports
// from quant-types.js directly.
import {
  BPE, QUANT_NAMES,
  IK_LLAMA_QUANT_NAMES, IK_LLAMA_FORK_BPE,
  TQ3_QUANT_NAMES, TQ3_FORK_BPE,
  BUUN_QUANT_NAMES, BUUN_FORK_BPE,
  PRISM_ML_QUANT_NAMES, PRISM_ML_FORK_BPE,
  BEELLAMA_QUANT_NAMES, BEELLAMA_FORK_BPE,
  tensorBpe, tensorElems, tensorQuantName, sumBytes, sumElems,
} from './quant-types.js';
export {
  BPE, QUANT_NAMES,
  IK_LLAMA_QUANT_NAMES, IK_LLAMA_FORK_BPE,
  TQ3_QUANT_NAMES, TQ3_FORK_BPE,
  BUUN_QUANT_NAMES, BUUN_FORK_BPE,
  PRISM_ML_QUANT_NAMES, PRISM_ML_FORK_BPE,
  BEELLAMA_QUANT_NAMES, BEELLAMA_FORK_BPE,
} from './quant-types.js';

// ── Standard transformer KV cache (parameterized) ──
// opts.iswa               — arch has interleaved sliding-window layers
// opts.denseFirst         — when ISWA, first layer in each period is dense (smallthinker pattern)
// opts.swaPeriodDefault   — fallback for integer sliding_window_pattern
// opts.swaDefault         — fallback for missing sliding_window
// opts.effectiveLayers    — override layer count (gemma4 shared_kv_layers, gemma3n layer_kv_from_start)
// opts.layerFilter        — predicate skipping layers (qwen35moe full_attention_interval)
// SWA layer classifier. Returns true iff layer `i` uses the sliding-window
// context (vs the full context). Extracted from buildKvCache's dense 4-level
// nested ternary (Phase 11 deferred work). Reads pre-computed swa_arr /
// swa_period / opts.denseFirst to decide.
function isSwaLayer(opts, swa_arr, swa_period, i) {
  if (!opts.iswa) return false;
  if (swa_arr) return !!swa_arr[i];
  if (swa_period <= 0) return false;
  return opts.denseFirst
    ? (i % swa_period !== 0)
    : (i % swa_period < (swa_period - 1));
}

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
    const isSwa = isSwaLayer(opts, swa_arr, swa_period, i);
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

// ── T5 encoder-decoder KV cache ──
// The decoder has two KV stores per layer: self-attention (grows with generated
// tokens) and cross-attention over the encoder output (fixed once the prompt is
// encoded). The encoder itself is bidirectional and has no KV cache. We assume
// the encoder output length equals ctxSize (the estimator treats ctx as the full
// sequence budget), so cross-attn doubles the per-layer decoder KV. decoder layer
// count comes from `decoder_block_count` (defaults to `block_count` when absent,
// matching t5.cpp:12). See resources/llama.cpp/src/models/t5.cpp:55-57.
function t5KvCache(meta, ctxSize, kvTypeK, kvTypeV) {
  const arch = meta['general.architecture'];
  const n_head_kv = getMeta(meta, `${arch}.attention.head_count_kv`);
  const headDimK = getMeta(meta, `${arch}.attention.key_length`) || getMeta(meta, `${arch}.embedding_length`);
  const headDimV = getMeta(meta, `${arch}.attention.value_length`) || headDimK;
  const n_block = getMeta(meta, `${arch}.block_count`);
  const dec_n_layer = getMeta(meta, `${arch}.decoder_block_count`) || n_block;
  // self-attn KV + cross-attn KV (both scale with ctxSize under our assumption)
  const totalElemsK = 2 * dec_n_layer * n_head_kv * headDimK * ctxSize;
  const totalElemsV = 2 * dec_n_layer * n_head_kv * headDimV * ctxSize;
  return {
    bytesK: totalElemsK * (BPE[kvTypeK] || 0),
    bytesV: totalElemsV * (BPE[kvTypeV] || 0),
    layers: n_block,
    headDimK, headDimV,
    totalHeadsKV: 2 * dec_n_layer * n_head_kv,
    avgHeadsKV: 2 * n_head_kv,
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
// nextn_predict_layers (MTP) blocks are loaded but skipped by the main decoder
// graph in llama.cpp (`for (il = 0; il < n_layer - nextn_predict_layers; ++il)`
// in src/models/qwen35.cpp:161 and similar), so they don't contribute
// activations. Subtracting reduces to a no-op for non-MTP arches (nextn=0).
// ── Shared activation metadata loader ──
// Five activation builders (buildActivations, sharedExpertActivations,
// leadingDenseActivations, deepseek2MlaMoeActivations, mlaActivations)
// all started with the same ~7-line preamble reading arch-scoped hparams.
// Centralizing eliminates the duplication and gives one site to update
// when a new hparam joins the standard set.
function loadActivationMeta(meta) {
  const arch = meta['general.architecture'];
  const n_embd = getMeta(meta, `${arch}.embedding_length`);
  const n_layer = getMeta(meta, `${arch}.block_count`);
  const nextn = getMeta(meta, `${arch}.nextn_predict_layers`);
  const n_main = Math.max(0, n_layer - nextn);
  const expertCount = getMeta(meta, `${arch}.expert_count`);
  const expertUsedCount = getMeta(meta, `${arch}.expert_used_count`);
  const expertFF = getMeta(meta, `${arch}.expert_feed_forward_length`);
  const isMoe = expertCount > 0;
  return { arch, n_embd, n_layer, nextn, n_main, expertCount, expertUsedCount, expertFF, isMoe };
}

// Common ternary: how many expert-FFN element-positions across N layers?
// Returns expertUsedCount * expertFF * layers when the MoE shape is known,
// otherwise falls back to per-layer ffSumOverLayers for the same range.
function routedExpertFF(m, expertUsedCount, expertFF, isMoe, layers) {
  return (isMoe && expertUsedCount > 0 && expertFF > 0)
    ? expertUsedCount * expertFF * layers
    : ffSumOverLayers(m, m['general.architecture'], m[`${m['general.architecture']}.block_count`], 0, layers);
}

function buildActivations(meta, batchSize) {
  const { n_embd, n_main, expertCount, expertUsedCount, expertFF, isMoe } = loadActivationMeta(meta);
  const ffTotal = routedExpertFF(meta, expertUsedCount, expertFF, isMoe, n_main);
  const totalBytes = batchSize * (n_embd * n_main + ffTotal) * 4;
  return {
    totalBytes,
    perLayerBytes: n_main > 0 ? totalBytes / n_main : 0,
    isMoe, expertCount, expertUsedCount, expertFF,
  };
}

// ── Shared-expert MoE activations: residual + shared FFN + routed experts ──
// Used by qwen35moe, qwen3next (shared expert may have different FFN dim)
function sharedExpertActivations(meta, batchSize) {
  const { arch, n_embd, n_main, expertCount, expertUsedCount, expertFF, isMoe } = loadActivationMeta(meta);
  const expertSharedFF = getMeta(meta, `${arch}.expert_shared_feed_forward_length`) || n_embd;
  const ffTotal = (isMoe && expertUsedCount > 0 && expertFF > 0)
    ? (expertSharedFF + expertUsedCount * expertFF) * n_main
    : ffSumOverLayers(meta, arch, getMeta(meta, `${arch}.block_count`), 0, n_main);
  const totalBytes = batchSize * (n_embd * n_main + ffTotal) * 4;
  return {
    totalBytes,
    perLayerBytes: n_main > 0 ? totalBytes / n_main : 0,
    isMoe, expertCount, expertUsedCount, expertFF,
  };
}

// ── Leading-dense activations: dense FFN for first N layers, MoE afterwards ──
function leadingDenseActivations(meta, batchSize) {
  const { arch, n_embd, n_layer, n_main, expertCount, expertUsedCount, expertFF } = loadActivationMeta(meta);
  const leadingDense = getMeta(meta, `${arch}.leading_dense_block_count`);
  const denseEnd = Math.min(leadingDense, n_main);
  const denseFF = ffSumOverLayers(meta, arch, n_layer, 0, denseEnd);
  const denseBytes = batchSize * (denseEnd * n_embd + denseFF) * 4;
  const moeLayers = Math.max(0, n_main - leadingDense);
  const moeFF = (expertUsedCount > 0 && expertFF > 0)
    ? expertUsedCount * expertFF * moeLayers
    : ffSumOverLayers(meta, arch, n_layer, leadingDense, n_main);
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
  const { arch, n_embd, n_layer, expertCount, expertUsedCount, expertFF, isMoe } = loadActivationMeta(meta);
  const q_lora_rank = getMeta(meta, `${arch}.attention.q_lora_rank`);
  const kv_lora_rank = getMeta(meta, `${arch}.attention.kv_lora_rank`);
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
  const { arch, n_embd, n_layer } = loadActivationMeta(meta);
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
  // Exclude tensors in MTP / nextn blocks (idx >= n_main) — their experts are
  // loaded but never routed during the main forward pass, so they don't belong
  // in active-expert accounting. They surface in footprint.mtpBytes instead.
  const n_block = getMeta(meta, `${arch}.block_count`);
  const nextn = getMeta(meta, `${arch}.nextn_predict_layers`);
  const n_main = Math.max(0, n_block - nextn);
  const inMainBlock = (t) => {
    const idx = layerIndexFromTensorName(t.name);
    return idx < 0 || idx < n_main;
  };
  const expertTensors = tensorInfos.filter(t => inMainBlock(t) && isExpert(t));
  const routerTensors = tensorInfos.filter(t => inMainBlock(t) && isRouter(t));
  const sharedTensors = tensorInfos.filter(t => inMainBlock(t) && isShared(t));
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

// Common tensor-group shapes. Promoted to module constants (Phase 5 cleanup):
// previously each registry entry inlined one of these literal shapes — 14× for
// STANDARD_MOE_TENSOR_GROUPS, 4× for SHEXP_MOE_TENSOR_GROUPS.
const STANDARD_MOE_TENSOR_GROUPS = {
  expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'],
  router: ['*ffn_gate_inp*'],
  shared: [],
};
const SHEXP_MOE_TENSOR_GROUPS = {
  expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'],
  router: ['*ffn_gate_inp*'],
  shared: ['*ffn_gate_inp_shexp*', '*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'],
};

// MTP effective-layers helper: returns n_block minus trailing nextn_predict_layers
// (MTP blocks loaded into memory but skipped by the main decoder graph).
// Used by 6 archs (qwen35, qwen35moe, qwen3next [partial], glm4, glm4moe, bailingmoe2,
// exaone-moe, glm-dsa). Previously inlined as the same lambda 6×.
const mtpEffectiveLayers = (m, n_block) =>
  n_block - getMeta(m, `${m['general.architecture']}.nextn_predict_layers`);

// qwen35-family KV cache: only every Nth layer is full-attention (rest are
// linear/recurrent), then MTP tail is excluded. Used by qwen35, qwen35moe,
// and qwen3next (qwen3next without MTP — its block_count doesn't carry nextn).
// Previously inlined as the same 4-line body 3×.
const QWEN35_FULL_ATTN_INTERVAL_DEFAULT = 4;
function qwen35KvCache(meta, ctxSize, kvTypeK, kvTypeV, opts = {}) {
  const arch = meta['general.architecture'];
  const interval = getMeta(meta, `${arch}.full_attention_interval`) || QWEN35_FULL_ATTN_INTERVAL_DEFAULT;
  return buildKvCache(meta, ctxSize, kvTypeK, kvTypeV, {
    layerFilter: (i) => ((i + 1) % interval === 0),
    effectiveLayers: mtpEffectiveLayers,
    ...opts,
  });
}

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
    categories: ['transformer'],
    fallback: true,
    kvCache: llamaKvCache,
    activations: llamaActivations,
    moe: llamaMoe,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── DeepSeek2: MLA (Multi-Head Latent Attention) + MoE ──
  deepseek2: {
    categories: ['transformer', 'moe', 'mla'],
    kvCache: mlaKvCache,
    activations: deepseek2MlaMoeActivations,
    moe: moeShexpOnly,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── MiniCPM3: MLA (lite) ──
  minicpm3: {
    categories: ['transformer', 'mla'],
    kvCache: mlaKvCache,
    activations: mlaActivations,
    moe: llamaMoe,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── PLM: simplified MLA (only KV compressed, not Q) ──
  plm: {
    categories: ['transformer', 'mla'],
    kvCache: mlaKvCache,
    activations: mlaActivations,
    moe: llamaMoe,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── Kimi-Linear: MLA + MoE + KDA hybrid ──
  'kimi-linear': {
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
    categories: ['transformer', 'moe', 'iswa'],
    kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 2 }),
    activations: buildActivations,
    moe: moeNoShared,
    tensorGroups: STANDARD_MOE_TENSOR_GROUPS,
  },

  // ── Llama4: ISWA + MoE with shared experts ──
  llama4: {
    categories: ['transformer', 'moe', 'iswa'],
    kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 4, swaDefault: 8192 }),
    activations: buildActivations,
    moe: moeShexpOnly,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── Qwen3 MoE: standard MoE ──
  qwen3moe: {
    categories: ['transformer', 'moe'],
    kvCache: llamaKvCache,
    activations: buildActivations,
    moe: moeNoShared,
    tensorGroups: STANDARD_MOE_TENSOR_GROUPS,
  },

  // ── Qwen3.6 MoE: mixed DeltaNet/attention + MoE with shared experts ──
  // Only every Nth layer has full attention (the rest are DeltaNet with no KV cache).
  // Trailing nextn_predict_layers blocks are MTP — loaded but not executed by the
  // main decoder, so excluded from KV iteration via effectiveLayers.
  qwen35moe: {
    categories: ['transformer', 'moe'],
    kvCache: qwen35KvCache,
    activations: sharedExpertActivations,
    moe: (m, ti) => buildMoe(m, ti, {
      isRouter: (t) => t.name.includes('ffn_gate_inp') && !t.name.includes('shexp'),
      isShared: (t) => t.name.includes('_shexp.') || t.name.includes('ffn_gate_inp_shexp'),
    }),
    tensorGroups: SHEXP_MOE_TENSOR_GROUPS,
  },

  // ── Standard transformers (reuse llama handlers) ──
  qwen2:          {          categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen3:          {          categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen35:         {         categories: ['transformer'],      kvCache: qwen35KvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen3next:      {      categories: ['transformer', 'moe'],kvCache: (meta, ctxSize, kvTypeK, kvTypeV) => { const arch = meta['general.architecture']; const interval = getMeta(meta, `${arch}.full_attention_interval`) || QWEN35_FULL_ATTN_INTERVAL_DEFAULT; return buildKvCache(meta, ctxSize, kvTypeK, kvTypeV, { layerFilter: (i) => ((i + 1) % interval === 0) }); }, activations: sharedExpertActivations, moe: (m, ti) => buildMoe(m, ti, { isRouter: (t) => t.name.includes('ffn_gate_inp') && !t.name.includes('shexp'), isShared: (t) => t.name.includes('_shexp.') || t.name.includes('ffn_gate_inp_shexp'), }), tensorGroups: SHEXP_MOE_TENSOR_GROUPS },
  qwen2vl:        {        categories: ['transformer', 'vl'],kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen3vl:        {        categories: ['transformer', 'vl'],kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  gemma3:         {         categories: ['transformer', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 6 }), activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  gemma2:         {         categories: ['transformer', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 2, swaDefault: 4096 }), activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  olmo2:          {          categories: ['transformer', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 4 }), activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  phi3:           {           categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  granite:        {        categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  granitehybrid:  {  categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mistral3:       {       categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mistral4:       {       categories: ['transformer', 'moe', 'mla'], kvCache: mlaKvCache, activations: deepseek2MlaMoeActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  glm4:           {           categories: ['transformer'],      kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { effectiveLayers: mtpEffectiveLayers }), activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'falcon-h1':    {      categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  deci:           {           categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  cohere2:        {        categories: ['transformer', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 4 }), activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  smollm3:        {        categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  ernie4_5:       {       categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  grok:           {           categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'gemma-embedding':{ categories: ['embedding'],     kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  nemotron_h:     {     categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, recurrentLayers: nemotronHRecurrentLayers, tensorGroups: LLAMA_TENSOR_GROUPS },
  lfm2:           {           categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'minimax-m2':   {     categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  seed_oss:       {       categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  apertus:        {        categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  dots1:          {          categories: ['transformer', 'moe'],kvCache: llamaKvCache, activations: leadingDenseActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  flux:           {           categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  ltxv:           {           categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  lumina2:        {        categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen_image:     {     categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  wan:            {            categories: ['diffusion'],        kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'acestep-lm':   {    categories: ['audio'],            kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  t5encoder:      {      categories: ['transformer'],      kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mimo2:          {          categories: ['transformer', 'moe', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true }), activations: buildActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'hunyuan-dense':{  categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  // exaone4: only the 64-layer (32B) variant has SWA; smaller variants are dense.
  // llama-model.cpp:2287–2299 — the SWA branch is gated on n_layer==64.
  exaone4:        {        categories: ['transformer', 'iswa'], kvCache: (m, c, kK, kV) => {
    const arch = m['general.architecture'];
    const n_layer = getMeta(m, `${arch}.block_count`);
    if (n_layer === 64) return buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 4, swaDefault: 4096 });
    return buildKvCache(m, c, kK, kV);
  }, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  plamo3:         {         categories: ['transformer', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 8 }), activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  // smallthinker: llama-model.cpp:2697–2704 forces n_swa = 4096 when the SWA key is present.
  smallthinker:   {   categories: ['transformer', 'moe', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 4, swaDefault: 4096, denseFirst: true }), activations: buildActivations, moe: moeNoShared, tensorGroups: STANDARD_MOE_TENSOR_GROUPS },
  qwen2moe:       {       categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: buildActivations, moe: moeShexpOnly, tensorGroups: SHEXP_MOE_TENSOR_GROUPS },
  'modern-bert':  {    categories: ['embedding'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  // ── BERT-family encoders: llama-model.cpp:8430–8443 returns res = nullptr (no KV cache) ──
  bert:           {           categories: ['embedding'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'nomic-bert':   {     categories: ['embedding'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'nomic-bert-moe':{ categories: ['embedding', 'moe'], kvCache: noKvCache,    activations: buildActivations, moe: moeNoShared, tensorGroups: STANDARD_MOE_TENSOR_GROUPS },
  'neo-bert':     {       categories: ['embedding'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'jina-bert-v2': {   categories: ['embedding'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'jina-bert-v3': {   categories: ['embedding'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  eurobert:       {       categories: ['embedding'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'wavtokenizer-dec':{ categories: ['audio'],       kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  // ── Diffusion-LM family: also res = nullptr (parallel decoding, no autoregressive KV cache) ──
  dream:          {          categories: ['transformer'],      kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  llada:          {          categories: ['transformer'],      kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'llada-moe':    {      categories: ['transformer', 'moe'], kvCache: noKvCache,  activations: buildActivations, moe: moeNoShared, tensorGroups: STANDARD_MOE_TENSOR_GROUPS },
  rnd1:           {           categories: ['transformer', 'moe'], kvCache: noKvCache,  activations: buildActivations, moe: moeNoShared, tensorGroups: STANDARD_MOE_TENSOR_GROUPS },

  // ── Pure-recurrent architectures (no attention KV; recurrent state per layer) ──
  // llama-model.cpp:8451–8459 wires these to llama_memory_recurrent.
  mamba:          {          categories: ['recurrent'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mamba2:         {         categories: ['recurrent'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  rwkv6:          {          categories: ['recurrent'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  rwkv6qwen2:     {     categories: ['recurrent'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  rwkv7:          {          categories: ['recurrent'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  arwkv7:         {         categories: ['recurrent'],        kvCache: noKvCache,    activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },

  // ── Hybrid (per-layer attention + Mamba SSM, head_count_kv array) ──
  // llamaKvCache already skips layers with heads<=0; calcRecurrentState picks up
  // the SSM state for the complementary set.
  jamba:          {          categories: ['transformer', 'moe'],      kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  plamo2:         {         categories: ['transformer'],      kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },

  // ── MoE architectures that reuse the standard llama KV cache + std activations ──
  qwen3vlmoe:  {  categories: ['transformer', 'moe', 'vl'], kvCache: llamaKvCache, activations: buildActivations, moe: moeNoShared,   tensorGroups: STANDARD_MOE_TENSOR_GROUPS },
  bailingmoe2: { categories: ['transformer', 'moe'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { effectiveLayers: mtpEffectiveLayers }), activations: leadingDenseActivations, moe: (m, ti) => buildMoe(m, ti, { isShared: (t) => t.name.includes('_shexp.') }), tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'] } },
  // nemotron_h_moe: recurrent layers are those with both head_count_kv == 0 AND
  // feed_forward_length == 0 (llama-model.cpp:2257). Different predicate from
  // other hybrids — pure-FFN layers also have head_count_kv == 0 but are not
  // recurrent.
  nemotron_h_moe: { categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: buildActivations, moe: moeShexpOnly, recurrentLayers: nemotronHRecurrentLayers, tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_up_shexp*', '*ffn_down_shexp*'] } },
  dbrx: { categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: buildActivations, moe: moeNoShared, tensorGroups: STANDARD_MOE_TENSOR_GROUPS },
  grovemoe:      {       categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: buildActivations, moe: (m, ti) => buildMoe(m, ti, { isExpert: (t) => t.name.includes('_exps.') || t.name.includes('_chexps.') || t.name.includes('exp_probs_b'), isShared: noShared }),   tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*', '*ffn_gate_chexps*', '*ffn_up_chexps*', '*ffn_down_chexps*'], router: ['*ffn_gate_inp*'], shared: [] } },

  // ── MoE architectures with leading dense blocks ──
  ernie4_5_moe: { categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: (m, ti) => buildMoe(m, ti, { isShared: (t) => t.name.includes('_shexp.') }), tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: ['*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'] } },
  hunyuan_moe:  {  categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  lfm2_moe:     {      categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeNoShared,   tensorGroups: STANDARD_MOE_TENSOR_GROUPS },
  afmoe:        {        categories: ['transformer', 'moe', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 4 }), activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  deepseek:     {     categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  'deepseek2-ocr': { categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  bailingmoe:   {   categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },

  // ── ISWA + MoE architectures ──
  // exaone-moe: llama-model.cpp:2310–2314 hardcodes n_swa = 128.
  'exaone-moe': {   categories: ['transformer', 'moe', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true, swaPeriodDefault: 4, swaDefault: 128, effectiveLayers: mtpEffectiveLayers }), activations: leadingDenseActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },
  step35:       {        categories: ['transformer', 'moe', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true }), activations: buildActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },

  // ── GLM4 MoE: gate_up_exps fused pattern ──
  glm4moe: {
    categories: ['transformer', 'moe'],
    kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { effectiveLayers: mtpEffectiveLayers }),
    activations: buildActivations,
    moe: (m, ti) => buildMoe(m, ti, {
      isExpert: (t) => t.name.includes('_exps.') || t.name.includes('gate_up_exps'),
      isShared: noShared,
    }),
    tensorGroups: { expert: ['*ffn_gate_exps*', '*ffn_gate_up_exps*', '*ffn_up_exps*', '*ffn_down_exps*'], router: ['*ffn_gate_inp*'], shared: [] },
  },

  // ── DSA (DeepSeek Sparse Attention) — MLA + MoE (GLM-5 family) ──
  'glm-dsa': {
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

  // ── Cohere2 MoE: ISWA (denseFirst) + leading-dense MoE + MTP + shared experts ──
  // Mirrors afmoe plus the MTP effectiveLayers hook. set_swa_pattern(N, true)
  // in cohere2moe.cpp:33 makes the first layer of each SWA period dense.
  cohere2moe: {
    categories: ['transformer', 'moe', 'iswa'],
    kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, {
      iswa: true, swaPeriodDefault: 4, denseFirst: true,
      effectiveLayers: mtpEffectiveLayers,
    }),
    activations: leadingDenseActivations,
    moe: moeShexpOnly,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── Explicit registrations for remaining llama.cpp architectures ──
  // These are served correctly by the llama handler triple below; registering
  // them removes the unknown-architecture console warning and documents that
  // they have been audited against resources/llama.cpp/src/models/<arch>.cpp.
  // Grouped by family. All are standard causal transformers unless noted.

  // Legacy / classic decoder-only transformers (vanilla; llama fallback is exact)
  falcon:        {        categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  gpt2:          {          categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  gptj:          {          categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  gptneox:       {       categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  mpt:           {           categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  baichuan:      {      categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  starcoder:     {     categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  starcoder2:    {    categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  refact:        {        categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  bloom:         {         categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  stablelm:      {      categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  qwen:          {          categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  phi2:          {          categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  plamo:         {         categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  codeshell:     {     categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  orion:         {         categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  internlm2:     {     categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  gemma:         {         categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  xverse:        {        categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'command-r':   {     categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  olmo:          {          categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  openelm:       {       categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  chatglm:       {       categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  bitnet:        {        categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  jais:          {          categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  jais2:         {         categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  nemotron:      {      categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  exaone:        {        categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  chameleon:     {     categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  arcee:         {         categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'pangu-embedded': { categories: ['embedding'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'llama-embed': {   categories: ['embedding'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  maincoder:     {     categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  talkie:        {        categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  cogvlm:        {        categories: ['transformer', 'vl'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  paddleocr:     {     categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  // hunyuan_vl is a dense VLM (reuses hunyuan-dense graph in llama.cpp); llama fallback is exact
  hunyuan_vl:    {    categories: ['transformer', 'vl'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  // eagle3 is a 1-layer speculative draft head; it has a real self-attention KV cache
  eagle3:        {        categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },

  // arctic: dense + routed MoE per layer (outputs added). The llama fallback
  // counts both the dense ffn_* and the _exps. tensors as resident weights
  // (correct) and the expert activation; it omits the parallel dense-FFN
  // activation term (~0.5 MB on the 10B model, negligible).
  arctic:        {        categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },

  // minicpm / refact have an optional MoE branch (dense when n_expert==0);
  // llamaMoe + buildActivations handle both branches correctly via the
  // expert_count>0 check.
  minicpm:       {       categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },

  // Mellum: optional SWA (reads sliding_window; swa_type only set when present).
  // Register with iswa so SWA layers are shrunk correctly when metadata is set;
  // harmless (no-op) when SWA metadata is absent.
  mellum:        {        categories: ['transformer', 'moe', 'iswa'], kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true }), activations: buildActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },

  // ── Standard MoE (no shared experts): moeNoShared + buildActivations ──
  olmoe:         {         categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: buildActivations, moe: moeNoShared, tensorGroups: STANDARD_MOE_TENSOR_GROUPS },
  phimoe:        {        categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: buildActivations, moe: moeNoShared, tensorGroups: STANDARD_MOE_TENSOR_GROUPS },

  // ── MoE with shared experts: moeShexpOnly + sharedExpertActivations ──
  // granitemoe: optional shared experts (the "Shared" variant has _shexp.).
  granitemoe:    {    categories: ['transformer', 'moe'], kvCache: llamaKvCache, activations: sharedExpertActivations, moe: moeShexpOnly, tensorGroups: LLAMA_TENSOR_GROUPS },

  // ── T5: encoder-decoder (bespoke KV; see t5KvCache above) ──
  t5: {
    categories: ['transformer', 'encdec'],
    kvCache: t5KvCache,
    activations: llamaActivations,
    moe: llamaMoe,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── gemma4-assistant: all-MTP draft head, not a standalone target ──
  // Every layer is a nextn block (n_layer_nextn == n_layer_all), and the KV
  // cache is shared with a parent gemma4 context via ctx_other. Standalone
  // estimation is weights-only; registered with noKvCache to reflect that.
  'gemma4-assistant': {
    categories: ['transformer', 'vl', 'draft'],
    kvCache: noKvCache,
    activations: llamaActivations,
    moe: llamaMoe,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── DeepSeek4: DSV4 sparse attention + MLA-like Q + MoE with shared experts ──
  // Uses a custom multi-level KV cache (raw SWA + CSA/HCA compressed). The raw
  // SWA cache is K-only and dominates at typical context sizes; compressed
  // caches are much smaller (ctx/4 and ctx/128 resolution). This handler
  // approximates the raw SWA component, which is an upper bound.
  // Verified against resources/llama.cpp/src/models/deepseek4.cpp.
  deepseek4: {
    categories: ['transformer', 'moe', 'mla'],
    kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
      const arch = meta['general.architecture'];
      const n_embd = getMeta(meta, `${arch}.embedding_length`);
      const n_head = getMeta(meta, `${arch}.attention.head_count`);
      const headDimK = getMeta(meta, `${arch}.attention.key_length`) || (n_embd / n_head);
      const n_head_kv_raw = getMeta(meta, `${arch}.attention.head_count_kv`);
      const n_head_kv = Array.isArray(n_head_kv_raw)
        ? (Number(n_head_kv_raw[0]) || Number(n_head_kv_raw[n_head_kv_raw.length - 1]) || 0)
        : n_head_kv_raw;
      const n_layer = getMeta(meta, `${arch}.block_count`);
      const n_swa = getMeta(meta, `${arch}.attention.sliding_window`) || 0;
      // DSV4 set_swa_pattern(0) makes ALL layers SWA. K-only cache
      // (dsv4_make_k_only sets V size to 0). SWA cell padding matches
      // buildKvCache's GGML_PAD(min(ctx, n_swa + n_ubatch), 256).
      const N_UBATCH_DEFAULT = 512;
      const KV_CELL_PAD = 256;
      const layerCtx = n_swa > 0
        ? Math.min(ctxSize, Math.ceil((n_swa + N_UBATCH_DEFAULT) / KV_CELL_PAD) * KV_CELL_PAD)
        : ctxSize;
      const totalElemsK = n_layer * n_head_kv * headDimK * layerCtx;
      return {
        bytesK: totalElemsK * (BPE[kvTypeK] || 0),
        bytesV: 0,
        layers: n_layer,
        headDimK,
        headDimV: 0,
        totalHeadsKV: n_layer * n_head_kv,
        avgHeadsKV: n_head_kv,
      };
    },
    activations: deepseek2MlaMoeActivations,
    moe: moeShexpOnly,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── DFlash: cross-attention speculative decoding (beellama.cpp / buun-llama-cpp) ──
  // Both dflash and dflash-draft get res=nullptr in llama-model.cpp:2013-2016
  // (no autoregressive KV cache). DFlash uses cross-attention to the target
  // model's hidden states via a ring buffer, not standard KV.
  dflash: {
    categories: ['transformer', 'draft'],
    kvCache: noKvCache,
    activations: llamaActivations,
    moe: llamaMoe,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },
  'dflash-draft': {
    categories: ['transformer', 'draft'],
    kvCache: noKvCache,
    activations: llamaActivations,
    moe: llamaMoe,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },
  // gemma4-dflash-draft: buun-specific Gemma-4 DFlash speculative drafter.
  // Same no-KV-cache pattern as dflash-draft (cross-attention to target model).
  'gemma4-dflash-draft': {
    categories: ['transformer', 'draft'],
    kvCache: noKvCache,
    activations: llamaActivations,
    moe: llamaMoe,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // ── ik_llama.cpp-specific architectures ──

  // bitnet-25 / bitnet-b1.58: dense ternary-weight transformers (I2_S quants).
  // Both share the same graph (build_bitnet_158); vanilla causal LM with standard
  // KV cache and GQA. The I2_S BPE (ID 36) is already in the base BPE object.
  'bitnet-25':    {    categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },
  'bitnet-b1.58': { categories: ['transformer'], kvCache: llamaKvCache, activations: llamaActivations, moe: llamaMoe, tensorGroups: LLAMA_TENSOR_GROUPS },

  // gemma4_mtp: MTP assistant/draft head for gemma4. Reuses the target model's
  // frozen KV cache (build_gemma4.cpp:640-643) rather than allocating its own.
  // Standalone estimation is weights-only, matching gemma4-assistant / eagle3.
  gemma4_mtp: {
    categories: ['transformer', 'draft'],
    kvCache: noKvCache,
    activations: buildActivations,
    moe: llamaMoe,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // laguna: ISWA + MoE with shared experts (separate FFN dim via
  // expert_shared_feed_forward_length). Shares create_step35_tensors.
  // leading_dense_block_count defaults to 0 (all-MoE); if a real GGUF sets it
  // >0, the leadingDenseActivations split would be needed instead.
  laguna: {
    categories: ['transformer', 'moe', 'iswa'],
    kvCache: (m, c, kK, kV) => buildKvCache(m, c, kK, kV, { iswa: true }),
    activations: sharedExpertActivations,
    moe: moeShexpOnly,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },

  // minimax-m3: leading-dense MoE with shared experts (n_ff_exp * n_shared,
  // not a separate shared FFN dim). Structurally "deepseek2 minus MLA".
  // Mirrors dots1 / hunyuan_moe / bailingmoe (leading-dense + shexp, no MLA).
  'minimax-m3': {
    categories: ['transformer', 'moe'],
    kvCache: llamaKvCache,
    activations: leadingDenseActivations,
    moe: moeShexpOnly,
    tensorGroups: LLAMA_TENSOR_GROUPS,
  },
};

// ── Alias map: GGUF-returned names → registry keys ──
export const ARCH_ALIASES = {
  'ernie4_5-moe': 'ernie4_5_moe',
  'hunyuan-moe':  'hunyuan_moe',
  'lfm2moe':      'lfm2_moe',
  // deepseek32 (DeepSeek V3.2) is structurally identical to glm-dsa: MLA +
  // leading-dense MoE + DSA "lightning indexer" + nextn_predict_layers (MTP).
  // Same kv class (llama_kv_cache_dsa), same hparams, same tensor names.
  // Verified against resources/llama.cpp/src/models/deepseek32.cpp.
  'deepseek32':   'glm-dsa',
  // ik_llama.cpp uses underscore GGUF strings where upstream uses hyphen/no-sep
  'cohere2_moe':  'cohere2moe',
  'gemma4_assistant': 'gemma4-assistant',
};

// ── Get architecture handler with fallback ──
// Single warn per unknown arch (was fired 5× per estimate before Phase 7
// because every calc function calls getArchHandler independently).
const _warnedArchs = new Set();
export function getArchHandler(arch) {
  const aliasKey = ARCH_ALIASES[arch];
  if (aliasKey && ARCHITECTURES[aliasKey]) return ARCHITECTURES[aliasKey];
  if (ARCHITECTURES[arch]) return ARCHITECTURES[arch];
  if (!_warnedArchs.has(arch)) {
    _warnedArchs.add(arch);
    console.warn(`Unknown architecture "${arch}", falling back to llama handler`);
  }
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
  // Resolve via registry categories instead of a hardcoded arch-name regex:
  // any arch declaring `categories: ['recurrent']` is fully recurrent.
  // (The regex fallback below catches legacy names that lack the category.)
  const handler = ARCHITECTURES[ARCH_ALIASES[arch] || arch];
  if (handler?.categories?.includes('recurrent')) return n_layer;
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

export function calcKVCache(metadata, ctxSize, kvTypeK, kvTypeV, nSeqMax = 1) {
  const arch = getModelArch(metadata);
  const handler = getArchHandler(arch);
  const result = handler.kvCache(metadata, ctxSize, kvTypeK, kvTypeV);
  const recurrent = calcRecurrentState(metadata, nSeqMax);
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
    if (minicpmvV >= 3) return 64;  // v3/v4/v5/v6/100045 all use 64
    return 96;  // legacy (v1) / unknown → 96 (mirrors llama.cpp clip.cpp:1187)
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
  // MTP (nextn_predict_layers) blocks live at the tail of block_count but are
  // skipped by the main decoder graph in llama.cpp, so they don't contribute to
  // per-token decode work. Their tensors fall through the idx >= nLayers branch
  // below into outputBytes — correctly counted as resident weights but excluded
  // from the per-layer streaming/compute path.
  const nextn = getMeta(metadata, `${arch}.nextn_predict_layers`);
  const nLayersTotal = getMeta(metadata, `${arch}.block_count`) || (kv ? kv.layers : 0);
  const nLayers = Math.max(0, nLayersTotal - nextn);
  const expertCount = moe ? moe.expertCount : 0;
  const expertUsed = moe ? moe.expertUsedCount : 0;
  const expertFrac = expertCount > 0 ? expertUsed / expertCount : 0;
  const expertPatterns = handler.tensorGroups ? (handler.tensorGroups.expert || []) : [];
  const isExpertTensor = (name) => expertPatterns.some(p => globMatch(p, name));

  // Single per-layer object array. Previously 8 parallel arrays coordinated
  // by index; the object form makes per-layer inspection in consumers far
  // cleaner (footprint.layers[i].nonExpertBytes vs footprint.layerNonExpertBytes[i]).
  const newLayer = () => ({
    bytes: 0, elems: 0,
    nonExpertBytes: 0, nonExpertElems: 0,
    expertBytesFull: 0, expertElemsFull: 0,
    expertBytesActive: 0, expertElemsActive: 0,
    activeBytes: 0, activeElems: 0,
  });
  const layers = Array.from({ length: nLayers }, newLayer);
  let outputBytes = 0, outputElems = 0;
  let mtpBytes = 0, mtpElems = 0;

  for (const t of tensorInfos) {
    const idx = layerIndexFromTensorName(t.name);
    const elems = tensorElems(t);
    const bytes = elems * tensorBpe(t);
    if (idx < 0) {
      // Global / output-pool tensors (token_embd, output, output_norm, ...).
      outputBytes += bytes;
      outputElems += elems;
      continue;
    }
    if (idx >= nLayers) {
      // MTP / nextn block — loaded weights, not on the per-token decode path.
      // Tracked separately so the UI can surface MTP VRAM cost, and so the
      // perf model doesn't bill them into the per-token output stream.
      mtpBytes += bytes;
      mtpElems += elems;
      continue;
    }
    const L = layers[idx];
    L.bytes += bytes;
    L.elems += elems;
    if (expertCount > 0 && isExpertTensor(t.name)) {
      L.expertBytesFull += bytes;
      L.expertElemsFull += elems;
      L.expertBytesActive += bytes * expertFrac;
      L.expertElemsActive += elems * expertFrac;
    } else {
      L.nonExpertBytes += bytes;
      L.nonExpertElems += elems;
    }
  }

  // Derived per-token streamed bytes/elems: non-expert + active experts.
  for (const L of layers) {
    L.activeBytes = L.nonExpertBytes + L.expertBytesActive;
    L.activeElems = L.nonExpertElems + L.expertElemsActive;
  }

  const kvOnlyBytes = kv ? (kv.bytesK + kv.bytesV) : 0;
  const kvBytesPerLayer = nLayers > 0 ? kvOnlyBytes / nLayers : 0;
  const recurrentBytesPerLayer = kv && kv.bytesRecurrent ? kv.bytesRecurrent / nLayers : 0;

  return {
    nLayers,
    layers,
    kvBytesPerLayer, recurrentBytesPerLayer,
    outputBytes, outputElems,
    mtpBytes, mtpElems,
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
// Build the standard return shape from a modes array + flags. The counts
// are derived from modes (no caller-side bookkeeping). Eliminates the 5×
// repeated 7-field object construction in computeOffloadSplit.
function makeSplitResult(modes, { auto, cpuMoe, nCpuMoe }) {
  let nGpu = 0, nHyb = 0;
  for (const m of modes) {
    if (m === 'gpu') nGpu++;
    else if (m === 'hybrid') nHyb++;
  }
  return {
    nGpuLayers: nGpu,
    nHybridLayers: nHyb,
    nCpuLayers: modes.length - nGpu - nHyb,
    auto, modes, cpuMoe, nCpuMoe,
  };
}

export function computeOffloadSplit({
  vramBytes, footprint, activationBytes = 0, nLayerOverride,
  cpuMoe = false, nCpuMoe = 0, unifiedMemory = false,
}) {
  const {
    nLayers, layers, kvBytesPerLayer, recurrentBytesPerLayer, outputBytes, mtpBytes = 0,
  } = footprint;

  const modes = new Array(nLayers).fill('cpu');

  if (unifiedMemory) {
    const n = nLayerOverride != null && nLayerOverride !== 'auto'
      ? Math.max(0, Math.min(nLayers, Number(nLayerOverride)))
      : nLayers;
    const gpuStart = nLayers - n;
    for (let i = gpuStart; i < nLayers; i++) modes[i] = 'gpu';
    return makeSplitResult(modes, {
      auto: nLayerOverride == null || nLayerOverride === 'auto',
      cpuMoe: false, nCpuMoe: 0,
    });
  }
  const hasExpert = (i) => (layers[i]?.expertBytesFull || 0) > 0;

  const shouldHybridize = (i) => {
    if (!hasExpert(i)) return false;
    if (cpuMoe) return true;
    if (nCpuMoe > 0 && i < nCpuMoe) return true;
    return false;
  };

  const rB = recurrentBytesPerLayer || 0;
  const gpuNeed = (i) => layers[i].nonExpertBytes + layers[i].expertBytesFull + kvBytesPerLayer + rB;
  const hybridNeed = (i) => layers[i].nonExpertBytes + kvBytesPerLayer + rB;

  // Manual --ngl override: last N layers to GPU (back-to-front), then apply
  // expert placement overrides for cpuMoe / nCpuMoe.
  if (nLayerOverride != null && nLayerOverride !== 'auto') {
    const n = Math.max(0, Math.min(nLayers, Number(nLayerOverride)));
    const gpuStart = nLayers - n;
    for (let i = gpuStart; i < nLayers; i++) {
      modes[i] = shouldHybridize(i) ? 'hybrid' : 'gpu';
    }
    return makeSplitResult(modes, { auto: false, cpuMoe, nCpuMoe });
  }

  if (!vramBytes || vramBytes <= 0) {
    return makeSplitResult(modes, { auto: true, cpuMoe, nCpuMoe });
  }

  // MTP weights (when present) are loaded into VRAM by llama.cpp alongside the
  // output pool — reserve their bytes before offloading per-layer weights.
  const reserved = outputBytes + mtpBytes + activationBytes;
  const hasExperts = footprint.hasExperts;

  if (!hasExperts) {
    // Dense model: simple back-to-front fill, no hybridization possible.
    let remaining = vramBytes - reserved;
    for (let i = nLayers - 1; i >= 0; i--) {
      const need = gpuNeed(i);
      if (remaining >= need) {
        modes[i] = 'gpu';
        remaining -= need;
      } else {
        break;
      }
    }
    return makeSplitResult(modes, { auto: true, cpuMoe, nCpuMoe });
  }

  // MoE model — always run llama.cpp's two-pass `--fit on` algorithm:
  //   Pass 1: dense-only fill back-to-front.
  //   Pass 2: upgrade non-forced layers to full gpu front-to-back.
  // --cpu-moe / --n-cpu-moe only restrict which layers pass 2 may upgrade;
  // the same algorithm runs whether or not those flags are set.
  let remaining = vramBytes - reserved;

  // Pass 1: dense-only fill back-to-front. Dense layers use gpuNeed,
  // MoE layers use hybridNeed (experts not counted toward VRAM).
  for (let i = nLayers - 1; i >= 0; i--) {
    const need = hasExpert(i) ? hybridNeed(i) : gpuNeed(i);
    if (remaining >= need) {
      modes[i] = hasExpert(i) ? 'hybrid' : 'gpu';
      remaining -= need;
    } else {
      break;
    }
  }

  // Pass 2: upgrade hybrid layers to full gpu (add experts back to VRAM)
  // front-to-back, but only for layers NOT forced by --cpu-moe / --n-cpu-moe.
  for (let i = 0; i < nLayers; i++) {
    if (modes[i] !== 'hybrid') continue;
    if (shouldHybridize(i)) continue;
    const expertBytes = layers[i].expertBytesFull;
    if (remaining >= expertBytes) {
      remaining -= expertBytes;
      modes[i] = 'gpu';
    }
  }

  return makeSplitResult(modes, { auto: true, cpuMoe, nCpuMoe });
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
    const expertPerLayer = footprint.layers.map(L => L.expertBytesFull || 0);
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

  let actualVram = footprint.outputBytes + (footprint.mtpBytes || 0) + activationBytes;
  let actualRam = 0;

  const rB = footprint.recurrentBytesPerLayer || 0;
  for (let i = 0; i < split.modes.length; i++) {
    const mode = split.modes[i];
    const L = footprint.layers[i];
    if (mode === 'gpu') {
      actualVram += L.nonExpertBytes + L.expertBytesFull + footprint.kvBytesPerLayer + rB;
    } else if (mode === 'hybrid') {
      actualVram += L.nonExpertBytes + footprint.kvBytesPerLayer + rB;
      actualRam += L.expertBytesFull;
    } else {
      actualRam += L.nonExpertBytes + L.expertBytesFull + footprint.kvBytesPerLayer + rB;
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
  const flopsDec = FLOPS_PER_PARAM * params;
  const bytesDec = wBytes + kvB;
  const flopsPre = FLOPS_PER_PARAM * params * ctx;
  const bytesPre = wBytes + kvB;
  return {
    tDecode: Math.max(flopsDec / flopsRate, bytesDec / bwRate),
    tPrefill: Math.max(flopsPre / flopsRate, bytesPre / bwRate),
    flopsTime: flopsDec / flopsRate,
    bwTime: bytesDec / bwRate,
  };
}

// ── Bottleneck classification ──
// Ordered predicate list — first match wins. Extracted from estimatePerformance
// (Phase 7) so the classification logic is testable in isolation and easier
// to extend with new labels.
const CPU_DOMINANCE_THRESHOLD = 0.5;  // CPU side dominates decode if > 50%
const PCIE_BOUNDARY_BW_BYTES = 32e9;  // ~PCIe Gen4 x16 fallback when CPU BW unknown
const FLOPS_PER_PARAM = 2;            // 1× params = MACs, 2× = FLOPs (multiply + add)
const ACTIVATION_BYTES_PER_ELEMENT = 2;  // F16 activations crossing GPU<->CPU boundary
const HYBRID_HOPS_PER_LAYER = 2;         // GPU->CPU + CPU->GPU round trip per hybrid layer

export function classifyBottleneck({
  nGpuLayers, nHybridLayers, nCpuLayers,
  cpuAvailable, tDecodeCpu, tDecodeHybridCpu, tDecode,
  gpuBottleneck,
}) {
  if ((nCpuLayers > 0 || nHybridLayers > 0) && !cpuAvailable) return 'cpu-layers-unrun';
  if (nCpuLayers > 0 && cpuAvailable && tDecodeCpu > CPU_DOMINANCE_THRESHOLD * tDecode) return 'cpu-dram-spill';
  if (nHybridLayers > 0 && cpuAvailable && tDecodeHybridCpu > CPU_DOMINANCE_THRESHOLD * tDecode) return 'cpu-experts';
  return gpuBottleneck || 'n/a';
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
    nLayers, layers,
    kvBytesPerLayer, outputBytes, outputElems,
  } = footprint;

  let tDecodeGpu = 0, tDecodeCpu = 0, tDecodeHybridGpu = 0, tDecodeHybridCpu = 0;
  let tPrefillGpu = 0, tPrefillCpu = 0, tPrefillHybridGpu = 0, tPrefillHybridCpu = 0;
  let gpuFlopsTime = 0, gpuBwTime = 0, cpuFlopsTime = 0, cpuBwTime = 0;

  const cpuAvailable = cpu && cpuFlops > 0 && cpuBw > 0;
  const kvB = kvBytesPerLayer;

  for (let i = 0; i < nLayers; i++) {
    const mode = modes[i];
    const L = layers[i];
    if (mode === 'gpu') {
      const t = layerTiming(L.activeElems, L.activeBytes, kvB, ctx, gpuFlops, gpuBw);
      tDecodeGpu += t.tDecode;
      tPrefillGpu += t.tPrefill;
      gpuFlopsTime += t.flopsTime;
      gpuBwTime += t.bwTime;
    } else if (mode === 'hybrid' && cpuAvailable) {
      const g = layerTiming(L.nonExpertElems, L.nonExpertBytes, kvB, ctx, gpuFlops, gpuBw);
      tDecodeHybridGpu += g.tDecode;
      tPrefillHybridGpu += g.tPrefill;
      gpuFlopsTime += g.flopsTime;
      gpuBwTime += g.bwTime;
      const c = layerTiming(L.expertElemsActive, L.expertBytesActive, 0, ctx, cpuFlops, cpuBw);
      tDecodeHybridCpu += c.tDecode;
      tPrefillHybridCpu += c.tPrefill;
      cpuFlopsTime += c.flopsTime;
      cpuBwTime += c.bwTime;
    } else if (mode === 'cpu' && cpuAvailable) {
      const t = layerTiming(L.activeElems, L.activeBytes, kvB, ctx, cpuFlops, cpuBw);
      tDecodeCpu += t.tDecode;
      tPrefillCpu += t.tPrefill;
      cpuFlopsTime += t.flopsTime;
      cpuBwTime += t.bwTime;
    }
  }

  const hasGpuOffload = nGpuLayers + nHybridLayers > 0;
  const outFlops = hasGpuOffload ? gpuFlops : (cpuAvailable ? cpuFlops : gpuFlops);
  const outBw = hasGpuOffload ? gpuBw : (cpuAvailable ? cpuBw : gpuBw);
  const tOutDec = Math.max((FLOPS_PER_PARAM * outputElems) / outFlops, outputBytes / outBw);
  const tOutPre = Math.max((FLOPS_PER_PARAM * outputElems * ctx) / outFlops, outputBytes / outBw);

  // Boundary bytes: F16 activation crossing GPU<->CPU at each hop (2 B/element).
  // Hybrid hops: 2 per layer (one GPU->CPU for expert input, one CPU->GPU for output).
  const n_embd = getMeta(metadata, `${getModelArch(metadata)}.embedding_length`) || 0;
  const boundaryBytes = n_embd * ACTIVATION_BYTES_PER_ELEMENT;
  const boundaryBw = Math.min(PCIE_BOUNDARY_BW_BYTES, cpuBw || PCIE_BOUNDARY_BW_BYTES);
  const spillBoundaryHops = (nGpuLayers + nHybridLayers > 0 && nCpuLayers > 0) ? 1 : 0;
  const hybridHops = HYBRID_HOPS_PER_LAYER * nHybridLayers;
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
  const overall = classifyBottleneck({
    nGpuLayers, nHybridLayers, nCpuLayers,
    cpuAvailable, tDecodeCpu, tDecodeHybridCpu, tDecode,
    gpuBottleneck,
  });

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
      avgLayerActiveBytes: nLayers > 0 ? layers.reduce((s, L) => s + L.activeBytes, 0) / nLayers : 0,
      avgLayerFullBytes: nLayers > 0 ? layers.reduce((s, L) => s + L.nonExpertBytes + L.expertBytesFull, 0) / nLayers : 0,
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
