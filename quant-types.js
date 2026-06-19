// Quantization type registry: bytes-per-element (BPE) values, display names,
// and per-fork override maps for resolving type-ID collisions across
// llama.cpp forks. Also exposes the tensor-size helpers (tensorBpe,
// tensorQuantName, sumBytes, sumElems) that consume those maps.
//
// This module is the single source of truth for "what BPE does this tensor
// have?" and "what should this quant ID be displayed as?". Both calculations.js
// (memory math) and parsing.js (fork-aware override application) import it.
//
// GGMLQuantizationType is re-exported for downstream convenience.

import { GGMLQuantizationType } from '@huggingface/gguf';
export { GGMLQuantizationType };

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
  // Row-interleaved R4 variants (ik_llama.cpp weight types).
  // NOTE: ID 202 (Q4_0_R8) collides with tq3's TURBO4_0 KV cache type —
  // see line below and IK_LLAMA_FORK_BPE / TQ3 KV comment. The ik_llama
  // tensor BPE for 202 is exposed via IK_LLAMA_FORK_BPE; the base BPE[202]
  // holds tq3's value to preserve --kvTypeK=202 KV-cache semantics.
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
  // Default BPE for colliding IDs 42/43 when no fork is detected. These IDs
  // appear in tensorInfos (model weights) from tq3 (Q1_0/TQ3_1S), prism-ml
  // (Q2_0), or buun (TURBO3_0/TURBO4_0). When fork detection succeeds, the
  // fork-specific override maps take precedence. The defaults below match
  // turboquant's string-keyed values (34/128, 50/128), which coincidentally
  // equal prism-ml's Q2_0 BPE — the most common undetected case.
  42: 34 / 128, // default (prism-ml Q2_0 / turboquant TURBO2_0)
  43: 50 / 128, // default (turboquant TURBO3_0)
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
  // llama.cpp-tq3 additional KV cache types (numeric IDs; BPE differs from
  // turboquant's string-keyed TURBO3_0/TURBO4_0 — tq3 uses QK=32 for TURBO3_0,
  // giving 14/32 vs turboquant's 50/128. TURBO4_0 happens to match at 68/128.)
  201: 14 / 32,  // TURBO3_0 (tq3)
  // ID 202 collides with ik_llama.cpp's Q4_0_R8 weight type (18/32).
  // For tq3 KV cache use (--kvTypeK=202), this entry (68/128) is correct
  // and matches llama.cpp-tq3's kv_cache_type. For ik_llama tensor sizing
  // of dtype-202 weights, IK_LLAMA_FORK_BPE below overrides to 18/32.
  202: 68 / 128, // TURBO4_0 (tq3 KV; see IK_LLAMA_FORK_BPE for tensor override)
  // buun-llama-cpp unique type (ID 47 is unique to buun; IDs 42-46 collide and
  // are handled via BUUN_FORK_BPE overrides at parse time)
  47: 130 / 128, // TURBO8_0 (buun)
  // buun-llama-cpp KV cache quantization (string keys for KV_VALID_QUANTS dropdown;
  // use BUUN_ prefix because numeric IDs 45/46 collide with TheTom's TQ3_1S/TQ4_1S)
  BUUN_TURBO3_TCQ: 52 / 128,
  BUUN_TURBO2_TCQ: 36 / 128,
};

// Quantization type names for display
// Auto-populated from @huggingface/gguf package, plus manual entries for ik_llama.cpp extensions
export const QUANT_NAMES = {};
for (const [key, val] of Object.entries(GGMLQuantizationType)) {
  if (typeof val === 'number') QUANT_NAMES[val] = key;
}
// ── ik_llama.cpp extension names (labeled for clarity) ──
export const IK_LLAMA_QUANT_NAMES = {
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
// BPE overrides applied when fork detection identifies an ik_llama.cpp model.
// Only ID 202 collides with another fork (tq3's TURBO4_0 KV cache type).
// All other ik_llama dtype IDs are unique and resolve correctly via base BPE.
export const IK_LLAMA_FORK_BPE = {
  202: 18 / 32,  // Q4_0_R8 (weight type)
};
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
  42: 'Q1_0 (tq3)',
  44: 'TQ3_1S (tq3)',
  45: 'TQ3_1S (tq3)',
  46: 'TQ3_4S (tq3)',
  200: 'TQ3_0 (tq3 KV)',
};
QUANT_NAMES[200] = 'TQ3_0 (tq3 KV)';
QUANT_NAMES[201] = 'TURBO3_0 (tq3 KV)';
QUANT_NAMES[202] = 'TURBO4_0 (tq3 KV)';
QUANT_NAMES[47] = 'TURBO8_0 (buun)';
QUANT_NAMES['BUUN_TURBO3_TCQ'] = 'TURBO3_TCQ (buun)';
QUANT_NAMES['BUUN_TURBO2_TCQ'] = 'TURBO2_TCQ (buun)';

// BPE overrides applied when fork detection identifies a llama.cpp-tq3 model.
// IDs 44, 45, 46 collide with turboquant (TURBO4_0 / TQ3_1S / TQ4_1S).
// ID 42 (Q1_0, 18/128) is a tq3-exclusive weight type that also collides with
// turboquant's TURBO2_0 KV type — detection in parsing.js must resolve tq3 first.
export const TQ3_FORK_BPE = {
  42: 18 / 128,  // Q1_0 (tq3 weight type, 1.125 bpw)
  44: 16 / 32,
  45: 16 / 32,
  46: 16 / 32,
  200: 14 / 32,
};

// BPE overrides for buun-llama-cpp fork. buun uses IDs 42-46 with different
// type assignments AND different BPEs than TheTom's turboquant fork, because
// buun uses QK_TURBO2=32 and QK_TURBO3=32 (vs TheTom's QK=128 for all).
// Detection: dtype 47 (TURBO8_0) is unique to buun.
export const BUUN_FORK_BPE = {
  42: 14 / 32,   // TURBO3_0 (buun; QK_TURBO3=32 → 14/32 vs TheTom's 50/128)
  43: 66 / 128,  // TURBO4_0 (buun; 66/128 vs TheTom's 68/128)
  44: 10 / 32,   // TURBO2_0 (buun; QK_TURBO2=32 → 10/32 vs TheTom's 34/128)
  45: 52 / 128,  // TURBO3_TCQ (buun; unique type, collides with TheTom's TQ3_1S at 16/32)
  46: 36 / 128,  // TURBO2_TCQ (buun; unique type, collides with TheTom's TQ4_1S at 20/32)
};
export const BUUN_QUANT_NAMES = {
  42: 'TURBO3_0 (buun)',
  43: 'TURBO4_0 (buun)',
  44: 'TURBO2_0 (buun)',
  45: 'TURBO3_TCQ (buun)',
  46: 'TURBO2_TCQ (buun)',
  47: 'TURBO8_0 (buun)',
};

// BPE overrides for prism-ml-llama.cpp fork (PrismML/Bonsai).
// Q2_0 (ID 42) has BPE 34/128 — coincidentally the same as TheTom's TURBO2_0,
// so byte counts are correct even without detection. The override exists to
// stamp the correct display name. Detection: ftype 28 (MOSTLY_Q2_0) is unique.
export const PRISM_ML_FORK_BPE = {
  42: 34 / 128,  // Q2_0 (prism-ml; same BPE as TheTom's TURBO2_0)
};
export const PRISM_ML_QUANT_NAMES = {
  42: 'Q2_0 (prism-ml)',
};

// ── Fork registry ──
// Single source of truth for "given a detected fork, which BPE/name maps apply?"
// Each entry's `bpe` and `names` map dtype ID → override values stamped on
// per-tensor _bpeOverride / _nameOverride at parse time. Adding a new fork
// means appending one entry here (plus its detectFork predicate in parsing.js).
//
// Individual maps are still exported above for direct consumers (tests, docs).
export const FORK_OVERRIDES = {
  tq3:      { bpe: TQ3_FORK_BPE,           names: TQ3_QUANT_NAMES      },
  buun:     { bpe: BUUN_FORK_BPE,          names: BUUN_QUANT_NAMES     },
  'prism-ml': { bpe: PRISM_ML_FORK_BPE,    names: PRISM_ML_QUANT_NAMES },
  ik_llama: { bpe: IK_LLAMA_FORK_BPE,      names: IK_LLAMA_QUANT_NAMES },
};

// ── Tensor size helpers ──
// Consume the BPE/QUANT_NAMES maps and the per-tensor _bpeOverride/_nameOverride
// fields that parsing.js's fork detection stamps on tensorInfos.
export function tensorElems(t) {
  return t.shape.map(Number).reduce((a, b) => a * b, 1);
}
export function tensorBpe(t) {
  return t._bpeOverride ?? BPE[t.dtype] ?? 0;
}
export function tensorQuantName(t) {
  return t._nameOverride ?? QUANT_NAMES[t.dtype] ?? `type_${t.dtype}`;
}
export function sumBytes(tensors) {
  let s = 0;
  for (const t of tensors) s += tensorElems(t) * tensorBpe(t);
  return s;
}
export function sumElems(tensors) {
  let s = 0;
  for (const t of tensors) s += tensorElems(t);
  return s;
}
