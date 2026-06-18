import { gguf, ggufAllShards, GGMLQuantizationType } from '@huggingface/gguf';
import {
  TQ3_FORK_BPE, TQ3_QUANT_NAMES,
  BUUN_FORK_BPE, BUUN_QUANT_NAMES,
  PRISM_ML_FORK_BPE, PRISM_ML_QUANT_NAMES,
} from './calculations.js';

export { GGMLQuantizationType };

// Union of allowed --cache-type-k / --cache-type-v values across supported forks:
//   llama.cpp       — F32, F16, BF16, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1
//   ik_llama.cpp    — adds Q6_0 (133), Q8_KV (151)
//   llama-cpp-turboquant  — adds TURBO2_0, TURBO3_0, TURBO4_0
//   llama-cpp-rotorquant  — adds PLANAR3_0, PLANAR4_0, ISO3_0, ISO4_0
//   llama.cpp-tq3   — adds TQ3_0 (200, KV cache only), TURBO3_0 (201), TURBO4_0 (202)
//   buun-llama-cpp  — adds TURBO8_0 (47), TURBO3_TCQ (45 buun), TURBO2_TCQ (46 buun)
//   prism-ml-llama.cpp — no KV-specific types (Q2_0 is weight-only)
// Source of truth: each fork's `kv_cache_types` / `kv_cache_type_from_str`.
export const KV_VALID_QUANTS = [
  GGMLQuantizationType.F32,
  GGMLQuantizationType.F16,
  GGMLQuantizationType.BF16,
  GGMLQuantizationType.Q8_0,
  GGMLQuantizationType.Q4_0,
  GGMLQuantizationType.Q4_1,
  GGMLQuantizationType.IQ4_NL,
  GGMLQuantizationType.Q5_0,
  GGMLQuantizationType.Q5_1,
  // ik_llama.cpp KV cache quantizations
  133,  // Q6_0
  151,  // Q8_KV
  // turboquant KV cache quantizations
  'TURBO2_0', 'TURBO3_0', 'TURBO4_0',
  // rotorquant KV cache quantizations
  'PLANAR3_0', 'PLANAR4_0', 'ISO3_0', 'ISO4_0',
  // tq3 KV cache quantizations
  'TQ3_0',
  201,  // TURBO3_0 (tq3; numeric ID because BPE differs from turboquant's string-keyed TURBO3_0)
  202,  // TURBO4_0 (tq3)
  // buun-llama-cpp KV cache quantizations
  47,   // TURBO8_0 (unique to buun)
  'BUUN_TURBO3_TCQ',  // ID 45 in buun (collides with TQ3_1S in other forks)
  'BUUN_TURBO2_TCQ',  // ID 46 in buun (collides with TQ4_1S in other forks)
];

// Fork-specific KV quant groups for UI optgroup rendering.
// Standard (llama.cpp) types are ungrouped; fork-exclusive types appear in labeled optgroups.
export const KV_FORK_GROUPS = [
  { label: 'ik_llama.cpp', quants: [133, 151] },
  { label: 'turboquant', quants: ['TURBO2_0', 'TURBO3_0', 'TURBO4_0'] },
  { label: 'rotorquant', quants: ['TURBO2_0', 'TURBO3_0', 'TURBO4_0', 'PLANAR3_0', 'PLANAR4_0', 'ISO3_0', 'ISO4_0'] },
  { label: 'tq3', quants: ['TQ3_0', 201, 202] },
  { label: 'buun', quants: [47, 'BUUN_TURBO3_TCQ', 'BUUN_TURBO2_TCQ'] },
];

function detectFork(metadata, tensorInfos) {
  const ftype = Number(metadata['general.file_type'] ?? -1);
  const dtypeSet = new Set(tensorInfos.map((t) => t.dtype));
  // tq3-unique signals first: dtype 200 (TQ3_0 KV), ftype 200/45/40.
  // ftype 40 = MOSTLY_Q1_0; its weight tensors carry dtype 42, which collides
  // with turboquant's TURBO2_0 KV type, so tq3 must be resolved before the
  // generic dtype-42 turboquant heuristic below.
  if (dtypeSet.has(200) || ftype === 200 || ftype === 45 || ftype === 40) return 'tq3';
  // prism-ml: ftype 28 (MOSTLY_Q2_0) is the canonical signal. Some prism-ml
  // quant tools set file_type to the type ID (41) instead of the ftype (28),
  // so also check: dtype 42 present (Q2_0 weight) without any tq3 signal above.
  if (ftype === 28 || (dtypeSet.has(42) && ftype === 41)) return 'prism-ml';
  // buun: dtype 47 (TURBO8_0) is unique to buun. Must be detected before the
  // generic dtype-42/43 turboquant check because buun uses IDs 42-46 with
  // different type assignments and BPEs than TheTom's turboquant fork.
  if (dtypeSet.has(47)) return 'buun';
  // turboquant (TheTom): its IDs 42/43/44 are KV cache types that never appear
  // in model weights (tensorInfos). Detection is based on weight types 45/46
  // (TQ3_1S/TQ4_1S) when no other fork signal fired. A plain dtype-42 in
  // tensorInfos without tq3/prism-ml/buun signals defaults to prism-ml naming
  // (handled by the default BPE[42] = 34/128 and PRISM_ML_QUANT_NAMES fallback).
  if (dtypeSet.has(45) || dtypeSet.has(46)) return 'turboquant';
  if (ftype === 43) {
    if (dtypeSet.has(44)) return 'tq3';
    if (dtypeSet.has(45)) return 'turboquant';
  }
  // Fallback: dtype 42 present without any fork-specific signal → prism-ml
  // (its Q2_0 BPE 34/128 matches the default BPE[42], so byte counts are right
  // regardless; the fork label just provides the correct display name).
  if (dtypeSet.has(42)) return 'prism-ml';
  return null;
}

function applyForkOverrides(tensorInfos, fork) {
  if (fork === 'tq3') {
    for (const t of tensorInfos) {
      if (TQ3_FORK_BPE[t.dtype] !== undefined) {
        t._bpeOverride = TQ3_FORK_BPE[t.dtype];
        t._nameOverride = TQ3_QUANT_NAMES[t.dtype];
      }
    }
  } else if (fork === 'buun') {
    for (const t of tensorInfos) {
      if (BUUN_FORK_BPE[t.dtype] !== undefined) {
        t._bpeOverride = BUUN_FORK_BPE[t.dtype];
        t._nameOverride = BUUN_QUANT_NAMES[t.dtype];
      }
    }
  } else if (fork === 'prism-ml') {
    for (const t of tensorInfos) {
      if (PRISM_ML_FORK_BPE[t.dtype] !== undefined) {
        t._bpeOverride = PRISM_ML_FORK_BPE[t.dtype];
        t._nameOverride = PRISM_ML_QUANT_NAMES[t.dtype];
      }
    }
  }
}

/**
 * Parse a GGUF file and return metadata + tensor infos.
 * Handles sharded GGUF files (e.g. -00001-of-00002.gguf).
 * @param {string} url - Direct URL to a GGUF file
 * @returns {Promise<{ metadata: Record<string, any>, tensorInfos: any[] }>}
 */
export async function parseGGUF(url) {
  let result;
  if (/-\d+-of-\d+\.gguf(?:[?#]|$)/i.test(url)) {
    const shards = await ggufAllShards(url);
    result = {
      metadata: shards.shards[0].metadata,
      tensorInfos: shards.shards.flatMap((s) => s.tensorInfos),
    };
  } else {
    result = await gguf(url);
  }
  const fork = detectFork(result.metadata, result.tensorInfos);
  if (fork) {
    applyForkOverrides(result.tensorInfos, fork);
    result.fork = fork;
  }
  return result;
}

const MMPROJ_RE = /mmproj/i;
const isMmProjName = (f) => MMPROJ_RE.test(f.replace(/^.*\//, ''));

/**
 * Resolve a HuggingFace path or URL to a GGUF file URL.
 * Splits repo GGUFs into main models and mmproj files (filename contains "mmproj").
 * If the repo has multiple main GGUFs, returns { url: null, ggufFiles: [...] }
 * so the caller can prompt the user to pick one. mmProjFiles is always returned
 * when present so the caller can offer a companion-projector selector.
 * @param {string} path - HuggingFace path (e.g. "owner/model") or URL
 * @returns {Promise<{ url: string | null, ggufFiles?: string[], mmProjFiles?: string[] }>}
 */
export async function resolveHFModel(path) {
  // HF page URL → extract owner/model slug and fall through to the API lookup
  if (path.match(/^https?:\/\/huggingface\.co\//i)) {
    const match = path.match(/^https?:\/\/huggingface\.co\/([^/?#]+\/[^/?#]+)/i);
    if (match) {
      const slug = match[1];
      const fileInfo = path.match(/[?&]show_file_info=([^&#]+)/);
      if (fileInfo && fileInfo[1].toLowerCase().endsWith('.gguf')) {
        return { url: `https://huggingface.co/${slug}/resolve/main/${decodeURIComponent(fileInfo[1])}` };
      }
      path = slug;
    }
  }

  // Direct URL to a .gguf file → normalize /blob/ → /resolve/, strip query/fragment
  if (path.match(/^https?:\/\/.*\.gguf/i)) {
    const url = path.replace(/\/blob\//, '/resolve/').replace(/[?#].*$/, '');
    return { url };
  }

  const apiRes = await fetch(`https://huggingface.co/api/models/${path}`, {
    headers: { Accept: 'application/json' },
  });
  if (!apiRes.ok) {
    throw new Error(`HF API returned ${apiRes.status}: ${apiRes.statusText}`);
  }
  const model = await apiRes.json();

  const sortByShardsThenAlpha = (a, b) => {
    const aFirst = /-0*1-of-\d+\.gguf$/i.test(a) ? 0 : 1;
    const bFirst = /-0*1-of-\d+\.gguf$/i.test(b) ? 0 : 1;
    return aFirst - bFirst || a.localeCompare(b);
  };

  const allGguf = (model.siblings || [])
    .map((s) => s.rfilename)
    .filter((f) => f && f.toLowerCase().endsWith('.gguf'))
    .sort(sortByShardsThenAlpha);

  const shardRe = /-\d+-of-\d+\.gguf$/i;
  const shardFirstRe = /-0*1-of-\d+\.gguf$/i;
  const ggufFiles = allGguf.filter((f) => !isMmProjName(f))
    .filter((f) => !shardRe.test(f) || shardFirstRe.test(f));
  const mmProjFiles = allGguf.filter(isMmProjName);

  if (ggufFiles.length === 0) {
    if (mmProjFiles.length > 0) {
      throw new Error('This repository only contains mmproj files; no main GGUF model to estimate.');
    }
    throw new Error('No .gguf files found in this model repository.');
  }

  const result = { url: null };
  if (ggufFiles.length === 1) {
    result.url = `https://huggingface.co/${path}/resolve/main/${ggufFiles[0]}`;
  } else {
    result.ggufFiles = ggufFiles;
  }
  if (mmProjFiles.length > 0) result.mmProjFiles = mmProjFiles;
  return result;
}

/**
 * Build a resolve URL from a model path and selected GGUF filename.
 */
export function buildResolveUrl(path, filename) {
  let modelPath = path;
  if (path.match(/^https?:\/\/huggingface\.co\//i)) {
    const match = path.match(/^https?:\/\/huggingface\.co\/([^/?#]+\/[^/?#]+)/i);
    if (match) modelPath = match[1];
  }
  return `https://huggingface.co/${modelPath}/resolve/main/${filename}`;
}
