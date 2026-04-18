import { gguf, ggufAllShards, GGMLQuantizationType } from '@huggingface/gguf';

export { GGMLQuantizationType };

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
  151,  // Q8_KV
  398,  // Q8_KV_R8
  // rotorquant KV cache quantizations
  'TURBO3_0', 'TURBO4_0', 'TURBO2_0',
  'PLANAR3_0', 'ISO3_0', 'PLANAR4_0', 'ISO4_0',
];

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
  return result;
}

/**
 * Resolve a HuggingFace path or URL to a GGUF file URL.
 * If the model has multiple GGUF files, returns { url: null, ggufFiles: [...] }
 * so the caller can prompt the user to pick one.
 * @param {string} path - HuggingFace path (e.g. "owner/model") or URL
 * @returns {Promise<{ url: string | null, ggufFiles?: string[] }>}
 */
export async function resolveHFModel(path) {
  let url;

  // Direct URL
  if (path.match(/^https?:\/\/.*\.gguf/i)) {
    url = path.replace(/\/blob\//, '/resolve/').replace(/#.*$/, '');
  } else if (path.match(/^https?:\/\/huggingface\.co\//i)) {
    const match = path.match(/^https?:\/\/huggingface\.co\/([^/]+\/[^/]+)/i);
    if (match) {
      path = match[1];
    }
  }

  if (!url) {
    const apiRes = await fetch(`https://huggingface.co/api/models/${path}`, {
      headers: { Accept: 'application/json' },
    });
    if (!apiRes.ok) {
      throw new Error(`HF API returned ${apiRes.status}: ${apiRes.statusText}`);
    }
    const model = await apiRes.json();

    const ggufFiles = (model.siblings || [])
      .map((s) => s.rfilename)
      .filter((f) => f && f.toLowerCase().endsWith('.gguf'))
      .sort((a, b) => {
        const aFirst = /-0*1-of-\d+\.gguf$/i.test(a) ? 0 : 1;
        const bFirst = /-0*1-of-\d+\.gguf$/i.test(b) ? 0 : 1;
        return aFirst - bFirst || a.localeCompare(b);
      });

    if (ggufFiles.length === 0) {
      throw new Error('No .gguf files found in this model repository.');
    }

    if (ggufFiles.length === 1) {
      url = `https://huggingface.co/${path}/resolve/main/${ggufFiles[0]}`;
    } else {
      return { url: null, ggufFiles };
    }
  }

  return { url };
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
