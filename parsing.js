import { gguf, ggufAllShards, GGMLQuantizationType } from '@huggingface/gguf';
import { TQ3_FORK_BPE, TQ3_QUANT_NAMES } from './calculations.js';

export { GGMLQuantizationType };

// Union of allowed --cache-type-k / --cache-type-v values across supported forks:
//   llama.cpp       — F32, F16, BF16, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1
//   ik_llama.cpp    — adds Q6_0 (133), Q8_KV (151)
//   llama-cpp-turboquant  — adds TURBO2_0, TURBO3_0, TURBO4_0
//   llama-cpp-rotorquant  — adds PLANAR3_0, PLANAR4_0, ISO3_0, ISO4_0
//   llama.cpp-tq3   — adds TQ3_0 (200, KV cache only)
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
];

// Fork-specific KV quant groups for UI optgroup rendering.
// Standard (llama.cpp) types are ungrouped; fork-exclusive types appear in labeled optgroups.
export const KV_FORK_GROUPS = [
  { label: 'ik_llama.cpp', quants: [133, 151] },
  { label: 'turboquant', quants: ['TURBO2_0', 'TURBO3_0', 'TURBO4_0'] },
  { label: 'rotorquant', quants: ['TURBO2_0', 'TURBO3_0', 'TURBO4_0', 'PLANAR3_0', 'PLANAR4_0', 'ISO3_0', 'ISO4_0'] },
  { label: 'tq3', quants: ['TQ3_0'] },
];

function detectFork(metadata, tensorInfos) {
  const ftype = Number(metadata['general.file_type'] ?? -1);
  const dtypeSet = new Set(tensorInfos.map((t) => t.dtype));
  if (dtypeSet.has(200) || ftype === 200 || ftype === 45) return 'tq3';
  if (dtypeSet.has(42) || dtypeSet.has(43)) return 'turboquant';
  if (ftype === 43) {
    if (dtypeSet.has(44)) return 'tq3';
    if (dtypeSet.has(45)) return 'turboquant';
  }
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
  // Direct URL to a .gguf file → normalize /blob/ → /resolve/, strip query/fragment
  if (path.match(/^https?:\/\/.*\.gguf/i)) {
    const url = path.replace(/\/blob\//, '/resolve/').replace(/[?#].*$/, '');
    return { url };
  }

  // HF page URL → extract owner/model slug and fall through to the API lookup
  if (path.match(/^https?:\/\/huggingface\.co\//i)) {
    const match = path.match(/^https?:\/\/huggingface\.co\/([^/]+\/[^/]+)/i);
    if (match) path = match[1];
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
    const match = path.match(/^https?:\/\/huggingface\.co\/([^/]+\/[^/]+)/i);
    if (match) modelPath = match[1];
  }
  return `https://huggingface.co/${modelPath}/resolve/main/${filename}`;
}
