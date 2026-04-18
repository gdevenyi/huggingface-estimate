import { resolveHFModel, parseGGUF, GGMLQuantizationType } from './node-parsing.js';
import {
  getArchHandler,
  getModelArch,
  calcWeightSize,
  calcKVCache,
  calcActivations,
  calcMoEInfo,
  formatBytes,
  formatElements,
  QUANT_NAMES,
  getMeta,
  BPE,
} from './node-calculations.js';

// ── CLI argument parsing ──
function parseKvType(val, flag) {
  if (GGMLQuantizationType[val] !== undefined) return GGMLQuantizationType[val];
  if (BPE[val] !== undefined) return val;
  const num = parseInt(val, 10);
  if (!Number.isNaN(num) && BPE[num] !== undefined) return num;
  const validNames = Object.keys(GGMLQuantizationType).filter(k => isNaN(Number(k))).sort().join(', ');
  console.error(`Error: invalid ${flag} value "${val}". Valid: ${validNames}`);
  process.exit(1);
}

function parseArgs(argv) {
  const args = {
    repo: null,
    batch: null,
    ctx: 4096,
    batchSize: 1,
    kvTypeK: GGMLQuantizationType.F16,
    kvTypeV: GGMLQuantizationType.F16,
    vram: 0,
    ram: 0,
  };

  let i = 2;
  while (i < argv.length) {
    const arg = argv[i];
    if (arg === '--batch') {
      args.batch = argv[++i];
    } else if (arg === '--ctx') {
      args.ctx = parseInt(argv[++i], 10);
    } else if (arg === '--batchSize') {
      args.batchSize = parseInt(argv[++i], 10);
    } else if (arg === '--kvTypeK') {
      args.kvTypeK = parseKvType(argv[++i], '--kvTypeK');
    } else if (arg === '--kvTypeV') {
      args.kvTypeV = parseKvType(argv[++i], '--kvTypeV');
    } else if (arg === '--vram') {
      args.vram = Math.max(0, parseFloat(argv[++i]) || 0);
    } else if (arg === '--ram') {
      args.ram = Math.max(0, parseFloat(argv[++i]) || 0);
    } else if (!arg.startsWith('-')) {
      args.repo = arg;
    }
    i++;
  }

  return args;
}

// ── Main calculation for a single model ──
async function calcModel(repo) {
  const result = await resolveHFModel(repo);

  if (!result.url) {
    const err = new Error(`Model "${repo}" has multiple GGUF files. Auto-selecting first: ${result.ggufFiles[0]}`);
    console.error(err.message);
    const url = `https://huggingface.co/${repo}/resolve/main/${result.ggufFiles[0]}`;
    result.url = url;
  }

  const parsed = await parseGGUF(result.url);
  const metadata = parsed.metadata;
  const tensorInfos = parsed.tensorInfos;

  const arch = getModelArch(metadata);
  const handler = getArchHandler(arch);

  // Weight size
  const weightInfo = calcWeightSize(tensorInfos);

  // Determine primary quant (largest by bytes)
  let primaryQuant = 'unknown';
  let primaryQuantKey = 0;
  let maxBytes = 0;
  for (const [name, info] of Object.entries(weightInfo.byQuant)) {
    if (info.bytes > maxBytes) {
      maxBytes = info.bytes;
      primaryQuant = name;
      primaryQuantKey = Object.keys(QUANT_NAMES).find(k => QUANT_NAMES[k] === name) || 0;
    }
  }

  // KV cache
  const kvCache = calcKVCache(metadata, args.ctx, args.kvTypeK, args.kvTypeV);

  // Activations
  const activations = calcActivations(metadata, args.batchSize);

  // MoE info
  const moeInfo = calcMoEInfo(metadata, tensorInfos);

  // VRAM / RAM breakdown
  const isMoe = moeInfo !== null;
  const nonMoEWeightBytes = isMoe
    ? weightInfo.total - moeInfo.expertWeightBytes - moeInfo.routerBytes - moeInfo.sharedBytes
    : 0;
  const vramWeightBytes = isMoe
    ? nonMoEWeightBytes + moeInfo.activeExpertWeightBytes + moeInfo.routerBytes + moeInfo.sharedBytes
    : weightInfo.total;
  const vramBytes = vramWeightBytes + kvCache.totalBytes + activations.totalBytes;
  const ramBytes = isMoe ? (moeInfo.expertWeightBytes - moeInfo.activeExpertWeightBytes) : 0;

  // Total parameters
  const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);

  return {
    repo,
    url: result.url,
    arch,
    quant: primaryQuant,
    totalParams: Number(totalParams),
    totalParamsFormatted: formatElements(totalParams),
    weightBytes: weightInfo.total,
    weightBytesFormatted: formatBytes(weightInfo.total),
    weightByQuant: Object.fromEntries(
      Object.entries(weightInfo.byQuant).map(([name, info]) => [
        name,
        {
          count: info.count,
          elements: Number(info.elements),
          elementsFormatted: formatElements(info.elements),
          bytes: info.bytes,
          bytesFormatted: formatBytes(info.bytes),
        },
      ])
    ),
    kvCache: {
      bytesK: kvCache.bytesK,
      bytesKFormatted: formatBytes(kvCache.bytesK),
      bytesV: kvCache.bytesV,
      bytesVFormatted: formatBytes(kvCache.bytesV),
      totalBytes: kvCache.totalBytes,
      totalBytesFormatted: formatBytes(kvCache.totalBytes),
      layers: kvCache.layers,
      headsK: kvCache.headsK,
      headsV: kvCache.headsV,
      totalHeadsKV: kvCache.totalHeadsKV,
      avgHeadsKV: kvCache.avgHeadsKV,
      kvTypeK: QUANT_NAMES[args.kvTypeK] || String(args.kvTypeK),
      kvTypeV: QUANT_NAMES[args.kvTypeV] || String(args.kvTypeV),
    },
    activations: {
      totalBytes: activations.totalBytes,
      totalBytesFormatted: formatBytes(activations.totalBytes),
      perLayerBytes: activations.perLayerBytes,
      perLayerBytesFormatted: formatBytes(activations.perLayerBytes),
      isMoe: activations.isMoe,
      expertCount: activations.expertCount,
      expertUsedCount: activations.expertUsedCount,
    },
    moe: isMoe ? {
      expertCount: moeInfo.expertCount,
      expertUsedCount: moeInfo.expertUsedCount,
      expertWeightBytes: moeInfo.expertWeightBytes,
      expertWeightBytesFormatted: formatBytes(moeInfo.expertWeightBytes),
      routerBytes: moeInfo.routerBytes,
      routerBytesFormatted: formatBytes(moeInfo.routerBytes),
      sharedBytes: moeInfo.sharedBytes,
      sharedBytesFormatted: formatBytes(moeInfo.sharedBytes),
      totalWeightBytes: moeInfo.totalWeightBytes,
      totalWeightBytesFormatted: formatBytes(moeInfo.totalWeightBytes),
      totalParams: moeInfo.totalParams,
      totalParamsFormatted: formatElements(moeInfo.totalParams),
      expertParams: moeInfo.expertParams,
      expertParamsFormatted: formatElements(moeInfo.expertParams),
      activeExpertWeightBytes: moeInfo.activeExpertWeightBytes,
      activeExpertWeightBytesFormatted: formatBytes(moeInfo.activeExpertWeightBytes),
    } : null,
    vramBytes,
    vramBytesFormatted: formatBytes(vramBytes),
    ramBytes,
    ramBytesFormatted: formatBytes(ramBytes),
    vramFit: args.vram > 0 ? (() => {
      const vramAvailBytes = args.vram * 1e9;
      const usagePct = vramBytes / vramAvailBytes * 100;
      return {
        availableGB: args.vram,
        requiredGB: +(vramBytes / 1e9).toFixed(2),
        fits: vramBytes <= vramAvailBytes,
        usagePct: +usagePct.toFixed(1),
      };
    })() : null,
    ramFit: args.ram > 0 && ramBytes > 0 ? (() => {
      const ramAvailBytes = args.ram * 1e9;
      const usagePct = ramBytes / ramAvailBytes * 100;
      return {
        availableGB: args.ram,
        requiredGB: +(ramBytes / 1e9).toFixed(2),
        fits: ramBytes <= ramAvailBytes,
        usagePct: +usagePct.toFixed(1),
      };
    })() : null,
  };
}

// ── Batch mode ──
async function runBatch(batchFile) {
  const fs = await import('node:fs');
  const lines = fs.default.readFileSync(batchFile, 'utf-8')
    .split('\n')
    .map(l => l.trim())
    .filter(l => l && !l.startsWith('#'));

  const results = [];
  for (let i = 0; i < lines.length; i++) {
    const repo = lines[i];
    process.stderr.write(`[${i + 1}/${lines.length}] ${repo}... `);
    try {
      const result = await calcModel(repo);
      console.error(`done (${result.arch}, ${result.weightBytesFormatted})`);
      results.push({ success: true, data: result });
    } catch (err) {
      console.error(`failed: ${err.message}`);
      results.push({ success: false, repo, error: err.message });
    }
  }
  console.log(JSON.stringify(results, null, 2));
}

// ── Entry point ──
const args = parseArgs(process.argv);

if (args.batch) {
  runBatch(args.batch).catch(err => {
    console.error(`Error: ${err.message}`);
    process.exit(1);
  });
} else if (args.repo) {
  calcModel(args.repo).then(result => {
    console.log(JSON.stringify(result, null, 2));
  }).catch(err => {
    console.error(`Error: ${err.message}`);
    process.exit(1);
  });
} else {
  console.error(`Usage: node run-calc.js <repo> [--ctx N] [--batchSize N] [--kvTypeK TYPE] [--kvTypeV TYPE] [--vram N] [--ram N]
       node run-calc.js --batch testModels.list

Arguments:
  <repo>          HuggingFace repo (e.g. unsloth/Qwen3-8B-GGUF)
  --batch <file>  Process all repos from a file (one per line)
  --ctx <N>       Context size (default: 4096)
  --batchSize <N> Batch size (default: 1)
  --kvTypeK <T>   KV cache K quantization type (name or number, default: F16)
  --kvTypeV <T>   KV cache V quantization type (name or number, default: F16)
  --vram <N>      Available VRAM in GB (enables VRAM fit check)
  --ram <N>       Available system RAM in GB (enables RAM fit check)

Quantization type names: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, Q8_K, ...
`);
  process.exit(1);
}
