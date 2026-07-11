#!/usr/bin/env node
import { resolveHFModel, parseGGUF, buildResolveUrl, GGMLQuantizationType, KV_VALID_QUANTS, KV_FORK_GROUPS } from './parsing.js';
import {
  getArchHandler,
  getModelArch,
  getMeta,
  calcWeightSize,
  calcKVCache,
  calcActivations,
  calcMoEInfo,
  calcMmProj,
  calcPerLayerFootprint,
  calcMemoryBreakdown,
  calcActualMemory,
  estimatePerformance,
  formatBytes,
  formatElements,
  QUANT_NAMES,
} from './calculations.js';
import { mergeCpuPresets, mergeGpuPresets, findCpuPreset, getGpuPresets, getSlowestCpuPreset, UNIFIED_MEMORY_CPU_PRESET } from './hardware-presets.js';
import { readFileSync } from 'node:fs';
import { readRepoList, parallelMap } from './lib/cli.js';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

const GIB = 1024 ** 3;

const CPU_JSON_FILES = ['intel-cpu-presets.json', 'amd-cpu-presets.json'];
const GPU_JSON_FILES = ['nvidia-gpu-presets.json', 'intel-gpu-presets.json', 'amd-gpu-presets.json', 'apple-gpu-presets.json'];

for (const f of [...CPU_JSON_FILES, ...GPU_JSON_FILES]) {
  try {
    const data = JSON.parse(readFileSync(join(__dirname, f), 'utf8'));
    if (f.includes('-cpu-')) mergeCpuPresets(data);
    else mergeGpuPresets(data);
  } catch (e) {
    if (e.code !== 'ENOENT') console.error(`Warning: failed to load ${f}: ${e.message}`);
  }
}
mergeCpuPresets([UNIFIED_MEMORY_CPU_PRESET]);

function findGpuPreset(query) {
  if (!query) return null;
  const list = getGpuPresets();
  const q = query.toLowerCase();
  const exactId = list.find(g => g.id === q);
  if (exactId) return exactId;
  const exactName = list.find(g => g.name.toLowerCase() === q);
  if (exactName) return exactName;
  const subs = list.filter(g => g.name.toLowerCase().includes(q));
  if (subs.length === 0) return null;
  subs.sort((a, b) => a.name.length - b.name.length);
  return subs[0];
}

// ── Help text ──
// KV quant type listing is generated from KV_VALID_QUANTS + KV_FORK_GROUPS so
// it stays in sync with parsing.js automatically.
const KV_QUANT_HELP = (() => {
  const forkQuantSet = new Set(KV_FORK_GROUPS.flatMap(g => g.quants));
  const standard = KV_VALID_QUANTS
    .filter(k => !forkQuantSet.has(k))
    .map(k => (QUANT_NAMES[k] || String(k)).replace(/\s*\(.*\)$/, '').trim());
  const forkLines = KV_FORK_GROUPS
    .map(g => {
      const quants = [...new Set(g.quants)]
        .map(k => (QUANT_NAMES[k] || String(k)).replace(/\s*\(.*\)$/, '').trim());
      return `    ${g.label}: ${quants.join(', ')}`;
    });
  return `  Standard: ${standard.join(', ')}\n${forkLines.join('\n')}`;
})();

const USAGE = `Usage: node run-calc.js <repo> [options]
       node run-calc.js <repo> --file <name>
       node run-calc.js <repo> --all
       node run-calc.js --batch <file> [options]

Estimate GGUF model memory, VRAM fit, and inference performance.

Model selection:
  <repo>              HuggingFace repo (e.g. unsloth/Qwen3-8B-GGUF) or .gguf URL
  --file <name>       Process a specific GGUF file from a multi-file repo
  --all               Process every GGUF file in the repo (wrapped per-repo in output)
  --batch <file>      Process all repos from a file (one per line, # comments)
  --concurrency <N>   Batch parallelism (default: 1)

Calculation parameters:
  --ctx <N|max>        Context size in tokens, or "max" for model maximum (default: 4096)
  --batchSize <N>     Batch size for activations (default: 2048, matches llama.cpp -b)
  --n-seq <N>         Max concurrent sequences for recurrent state (default: 1)
  --kvTypeK <T>       KV cache K quantization type (default: F16)
  --kvTypeV <T>       KV cache V quantization type (default: F16)
  --swa-full <0|1>    Full-size SWA cache: 1=full (llama.cpp default), 0=memory-saving (default: 1)

Multimodal (mmproj):
  --mmproj <file>     Specific mmproj GGUF filename (otherwise auto-detected)
  --no-mmproj         Suppress mmproj auto-detection
  --mmprojDevice <d>  Where to place mmproj: vram (default) or ram

Memory / VRAM fit check:
  --vram <N>          Available VRAM in GiB (enables VRAM fit check)
  --ram <N>           Available system RAM in GiB (enables RAM fit check)

Performance estimation (supply --gpu or --gpu-flops + --gpu-bw to enable):
  --gpu <name|id>     GPU preset (e.g. "RTX 4090", "nvidia-geforce-rtx-4090")
  --gpu-flops <TF>    Override GPU FP16 TFLOPS
  --gpu-bw <GB/s>     Override GPU memory bandwidth
  --cpu <name|id>     CPU preset (e.g. "Ryzen 9 7950X")
  --cpu-flops <TF>    Override CPU FP16 TFLOPS
  --ram-bw <GB/s>     Override system RAM bandwidth
  --ngl <n|auto>      GPU layer count override (default: auto, sized from --vram)
  --cpu-moe           Keep all MoE expert weights in CPU RAM (llama.cpp --cpu-moe)
  --n-cpu-moe <N>     Keep MoE expert weights of first N layers in CPU (llama.cpp --n-cpu-moe)

Other:
  --help, -h          Show this help message

KV cache quantization types:
${KV_QUANT_HELP}

Examples:
  node run-calc.js unsloth/Qwen3-8B-GGUF
  node run-calc.js unsloth/Qwen3-8B-GGUF --file Qwen3-8B-Q4_K_M.gguf
  node run-calc.js unsloth/Qwen3-8B-GGUF --all
  node run-calc.js unsloth/Qwen3-8B-GGUF --ctx 32768 --kvTypeK Q8_0
  node run-calc.js unsloth/Qwen3-8B-GGUF --gpu "RTX 4090" --vram 24
  node run-calc.js --batch test/baseline.list --concurrency 4`;

// ── CLI argument parsing ──

function parseKvType(val, flag) {
  if (GGMLQuantizationType[val] !== undefined) {
    const id = GGMLQuantizationType[val];
    if (KV_VALID_QUANTS.includes(id)) return id;
  }
  if (KV_VALID_QUANTS.includes(val)) return val;
  const num = parseInt(val, 10);
  if (!Number.isNaN(num) && KV_VALID_QUANTS.includes(num)) return num;
  for (const k of KV_VALID_QUANTS) {
    const name = QUANT_NAMES[k] || '';
    const base = name.replace(/\s*\(.*\)$/, '').trim();
    if (base === val) return k;
  }
  const validNames = KV_VALID_QUANTS
    .map(k => (QUANT_NAMES[k] || String(k)).replace(/\s*\(.*\)$/, '').trim())
    .join(', ');
  console.error(`Error: invalid ${flag} value "${val}". Valid: ${validNames}`);
  process.exit(1);
}

function parseArgs(argv) {
  if (argv.includes('--help') || argv.includes('-h')) {
    console.log(USAGE);
    process.exit(0);
  }

  const args = {
    repo: null,
    batch: null,
    file: null,
    all: false,
    ctx: 4096,
    batchSize: 2048,
    nSeq: 1,
    kvTypeK: GGMLQuantizationType.F16,
    kvTypeV: GGMLQuantizationType.F16,
    swaFull: true,
    vram: 0,
    ram: 0,
    mmproj: null,
    noMmproj: false,
    mmprojDevice: 'vram',
    gpu: null,
    gpuFlops: null,
    gpuBw: null,
    cpu: null,
    cpuFlops: null,
    ramBw: null,
    ngl: 'auto',
    cpuMoe: false,
    nCpuMoe: 0,
    concurrency: 1,
  };

  const needValue = (flag) => {
    const val = argv[++i];
    if (val === undefined || val.startsWith('-')) {
      console.error(`Error: ${flag} requires a value`);
      process.exit(1);
    }
    return val;
  };

  let i = 2;
  while (i < argv.length) {
    const arg = argv[i];
    if (arg === '--batch') {
      args.batch = needValue(arg);
    } else if (arg === '--file') {
      args.file = needValue(arg);
    } else if (arg === '--all') {
      args.all = true;
    } else if (arg === '--ctx') {
      const v = needValue(arg);
      if (v === 'max') { args.ctx = 'max'; }
      else {
        const n = parseInt(v, 10);
        if (Number.isNaN(n) || n < 1) {
          console.error('Error: --ctx requires a positive integer or "max"');
          process.exit(1);
        }
        args.ctx = n;
      }
    } else if (arg === '--batchSize') {
      args.batchSize = parseInt(needValue(arg), 10);
      if (Number.isNaN(args.batchSize) || args.batchSize < 1) {
        console.error('Error: --batchSize requires a positive integer');
        process.exit(1);
      }
    } else if (arg === '--n-seq') {
      args.nSeq = parseInt(needValue(arg), 10);
      if (Number.isNaN(args.nSeq) || args.nSeq < 1) {
        console.error('Error: --n-seq requires a positive integer');
        process.exit(1);
      }
    } else if (arg === '--kvTypeK') {
      args.kvTypeK = parseKvType(needValue(arg), '--kvTypeK');
    } else if (arg === '--kvTypeV') {
      args.kvTypeV = parseKvType(needValue(arg), '--kvTypeV');
    } else if (arg === '--swa-full') {
      args.swaFull = parseInt(needValue(arg), 10) !== 0;
    } else if (arg === '--vram') {
      args.vram = Math.max(0, parseFloat(needValue(arg)) || 0);
    } else if (arg === '--ram') {
      args.ram = Math.max(0, parseFloat(needValue(arg)) || 0);
    } else if (arg === '--mmproj') {
      args.mmproj = needValue(arg);
    } else if (arg === '--no-mmproj') {
      args.noMmproj = true;
    } else if (arg === '--mmprojDevice') {
      const v = needValue(arg);
      if (v !== 'vram' && v !== 'ram') {
        console.error(`Error: --mmprojDevice must be "vram" or "ram" (got "${v}")`);
        process.exit(1);
      }
      args.mmprojDevice = v;
    } else if (arg === '--gpu') {
      args.gpu = needValue(arg);
    } else if (arg === '--gpu-flops') {
      args.gpuFlops = parseFloat(needValue(arg));
    } else if (arg === '--gpu-bw') {
      args.gpuBw = parseFloat(needValue(arg));
    } else if (arg === '--cpu') {
      args.cpu = needValue(arg);
    } else if (arg === '--cpu-flops') {
      args.cpuFlops = parseFloat(needValue(arg));
    } else if (arg === '--ram-bw') {
      args.ramBw = parseFloat(needValue(arg));
    } else if (arg === '--ngl') {
      const v = needValue(arg);
      if (v === 'auto') { args.ngl = 'auto'; }
      else {
        const n = parseInt(v, 10);
        if (Number.isNaN(n) || n < 0) {
          console.error('Error: --ngl requires a non-negative integer or "auto"');
          process.exit(1);
        }
        args.ngl = n;
      }
    } else if (arg === '--cpu-moe') {
      args.cpuMoe = true;
    } else if (arg === '--n-cpu-moe') {
      const v = parseInt(needValue(arg), 10);
      if (Number.isNaN(v) || v < 0) {
        console.error('Error: --n-cpu-moe requires a non-negative integer');
        process.exit(1);
      }
      args.nCpuMoe = v;
    } else if (arg === '--concurrency') {
      const v = parseInt(needValue(arg), 10);
      if (Number.isNaN(v) || v < 1) {
        console.error('Error: --concurrency requires a positive integer');
        process.exit(1);
      }
      args.concurrency = v;
    } else if (!arg.startsWith('-')) {
      args.repo = arg;
    } else {
      console.error(`Error: unknown flag "${arg}"`);
      console.error(USAGE);
      process.exit(1);
    }
    i++;
  }

  if (args.file && args.all) {
    console.error('Error: --file and --all are mutually exclusive');
    process.exit(1);
  }

  return args;
}

// ── Device spec resolution for performance estimator ──
// Returns null if no GPU spec was supplied.
function resolveDevice(args) {
  const gpuPreset = args.gpu ? findGpuPreset(args.gpu) : null;
  if (args.gpu && !gpuPreset && (args.gpuFlops == null || args.gpuBw == null)) {
    console.error(`Warning: GPU preset "${args.gpu}" not found in GPU preset files.`);
  }
  const gpuFlops = args.gpuFlops != null ? args.gpuFlops : (gpuPreset ? gpuPreset.fp16Tflops : null);
  const gpuBw = args.gpuBw != null ? args.gpuBw : (gpuPreset ? gpuPreset.memBwGBps : null);
  if (gpuFlops == null || gpuBw == null) return null;

  const unifiedMemory = !!gpuPreset?.unifiedMemory;

  const cpuPreset = args.cpu ? findCpuPreset(args.cpu) : null;
  if (args.cpu && !cpuPreset && (args.cpuFlops == null || args.ramBw == null)) {
    console.error(`Warning: CPU preset "${args.cpu}" not found in CPU preset files.`);
  }
  const cpuFlops = args.cpuFlops != null ? args.cpuFlops : (cpuPreset ? cpuPreset.fp16Tflops : null);
  const ramBw = args.ramBw != null ? args.ramBw : (cpuPreset ? cpuPreset.defaultRamBwGBps : null);
  let cpu = (cpuFlops != null && ramBw != null) ? { flopsFp16Tflops: cpuFlops, bwGBps: ramBw } : null;
  let cpuFallback = null;

  if (!unifiedMemory && !cpu) {
    const slow = getSlowestCpuPreset();
    if (slow) {
      cpu = { flopsFp16Tflops: slow.fp16Tflops, bwGBps: slow.defaultRamBwGBps };
      cpuFallback = slow;
      console.error(`No CPU specified, falling back to slowest preset: ${slow.name} (${slow.fp16Tflops} TF, ${slow.defaultRamBwGBps} GB/s)`);
    }
  }
  if (unifiedMemory) {
    if (args.cpuMoe || args.nCpuMoe > 0) console.error('Warning: --cpu-moe / --n-cpu-moe ignored for unified memory GPUs');
    if (args.cpu) console.error('Warning: --cpu ignored for unified memory GPUs');
    cpu = null;
  }

  return {
    gpu: {
      flopsFp16Tflops: gpuFlops,
      bwGBps: gpuBw,
      vramBytes: args.vram > 0 ? args.vram * GIB : 0,
      preset: gpuPreset,
    },
    cpu: cpu ? { ...cpu, preset: cpuPreset || cpuFallback, fallback: !!cpuFallback } : null,
    nGpuLayers: args.ngl === 'auto' ? 'auto' : args.ngl,
    mmprojOnGpu: args.mmprojDevice !== 'ram',
    cpuMoe: args.cpuMoe,
    nCpuMoe: args.nCpuMoe,
    unifiedMemory,
  };
}

// ── Repo resolution: determine which GGUF file(s) to process + mmproj ──
async function resolveRepo(repo, args) {
  const result = await resolveHFModel(repo);

  // Build the file list to process
  let files;
  if (result.url) {
    const filename = decodeURIComponent(result.url.split('/').pop().replace(/#.*$/, ''));
    files = [{ filename, url: result.url }];
  } else {
    const allFiles = result.ggufFiles;
    if (args.all) {
      files = allFiles.map(f => ({ filename: f, url: buildResolveUrl(repo, f) }));
    } else if (args.file) {
      const q = args.file.toLowerCase();
      const match = allFiles.find(f => f.toLowerCase() === q);
      if (!match) {
        throw new Error(`File "${args.file}" not found in repo "${repo}". Available:\n  ${allFiles.join('\n  ')}`);
      }
      files = [{ filename: match, url: buildResolveUrl(repo, match) }];
    } else {
      const first = allFiles[0];
      if (allFiles.length > 1) {
        process.stderr.write(`Multiple GGUF files found. Using first: ${first}. Use --file <name> or --all for others.\n`);
      }
      files = [{ filename: first, url: buildResolveUrl(repo, first) }];
    }
  }

  // mmproj resolution: explicit > auto-detect > none
  let mmproj = null;
  let mmprojAutoDetected = false;
  if (args.mmproj) {
    mmproj = { filename: args.mmproj, url: buildResolveUrl(repo, args.mmproj) };
  } else if (!args.noMmproj && result.mmProjFiles && result.mmProjFiles.length > 0) {
    const first = result.mmProjFiles[0];
    mmproj = { filename: first, url: buildResolveUrl(repo, first) };
    mmprojAutoDetected = true;
    process.stderr.write(`Auto-detected mmproj: ${first}. Use --no-mmproj to suppress.\n`);
  }

  // Pre-parse mmproj once so --all mode doesn't re-fetch it per file
  let mmProjInfo = null;
  if (mmproj) {
    const mmParsed = await parseGGUF(mmproj.url);
    mmProjInfo = calcMmProj(mmParsed.metadata, mmParsed.tensorInfos);
  }

  return { files, mmproj, mmprojAutoDetected, mmProjInfo };
}

// ── Main calculation for a single GGUF file ──
async function calcSingleFile(repo, fileInfo, args, resolved) {
  const { filename, url } = fileInfo;

  const parsed = await parseGGUF(url);
  const metadata = parsed.metadata;
  const tensorInfos = parsed.tensorInfos;
  const fork = parsed.fork || null;

  const arch = getModelArch(metadata);
  const handler = getArchHandler(arch);

  // Resolve "max" context to model's declared context_length
  let ctxSize = args.ctx;
  if (ctxSize === 'max') {
    ctxSize = getMeta(metadata, `${arch}.context_length`) || 0;
    if (ctxSize > 0) {
      process.stderr.write(`Using model maximum context: ${ctxSize} tokens.\n`);
    } else {
      ctxSize = 4096;
      process.stderr.write(`Warning: model has no context_length metadata; defaulting to 4096.\n`);
    }
  }

  const weightInfo = calcWeightSize(tensorInfos);

  let primaryQuant = 'unknown';
  let maxBytes = 0;
  for (const [name, info] of Object.entries(weightInfo.byQuant)) {
    if (info.bytes > maxBytes) { maxBytes = info.bytes; primaryQuant = name; }
  }

  const kvCache = calcKVCache(metadata, ctxSize, args.kvTypeK, args.kvTypeV, args.nSeq, args.swaFull);
  const activations = calcActivations(metadata, args.batchSize);
  const moeInfo = calcMoEInfo(metadata, tensorInfos);
  const layerFootprint = calcPerLayerFootprint(metadata, tensorInfos, kvCache, moeInfo);

  const memBreakdown = calcMemoryBreakdown({
    weights: weightInfo, kv: kvCache, activations,
    footprint: layerFootprint,
  });
  let vramBytes = memBreakdown.vramBytes;
  let ramBytes = memBreakdown.ramBytes;

  let mmProjInfo = resolved.mmProjInfo;
  let mmProjUrl = null;
  let mmProjBytes = 0;
  if (mmProjInfo) {
    mmProjUrl = resolved.mmproj.url;
    mmProjBytes = mmProjInfo.weightBytes + (mmProjInfo.perImageActBytes || 0);
    if (args.mmprojDevice === 'ram') ramBytes += mmProjBytes;
    else vramBytes += mmProjBytes;
  }

  const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);

  const device = resolveDevice(args);
  const performance = device ? formatPerformance(device, metadata, tensorInfos, ctxSize, args, kvCache, moeInfo, activations, mmProjInfo) : null;
  const vramRamFit = calcVramRamFit(args, activations, mmProjInfo, layerFootprint, ramBytes, !!device?.unifiedMemory);

  return {
    repo,
    filename,
    url,
    arch,
    fork,
    quant: primaryQuant,
    modelInfo: formatModelInfo(metadata, arch, handler),
    totalParams: Number(totalParams),
    totalParamsFormatted: formatElements(totalParams),
    ...formatWeights(weightInfo),
    ...formatKvCache(kvCache, args),
    ...formatActivations(activations),
    ...formatMoe(moeInfo),
    mtp: {
      weightBytes: layerFootprint.mtpBytes || 0,
      weightBytesFormatted: formatBytes(layerFootprint.mtpBytes || 0),
      elements: Number(layerFootprint.mtpElems || 0),
    },
    ...formatMmProj(mmProjInfo, args, resolved, mmProjUrl, mmProjBytes),
    vramBytes,
    vramBytesFormatted: formatBytes(vramBytes),
    ramBytes,
    ramBytesFormatted: formatBytes(ramBytes),
    ...vramRamFit,
    performance,
  };
}

// ── Process a single repo (one file or all files) ──
async function processRepo(repo, args) {
  const resolved = await resolveRepo(repo, args);

  if (args.all && resolved.files.length > 1) {
    const fileResults = [];
    for (const f of resolved.files) {
      const fileResult = await calcSingleFile(repo, f, args, resolved);
      fileResults.push(fileResult);
    }
    return { repo, files: fileResults };
  }
  return calcSingleFile(repo, resolved.files[0], args, resolved);
}

// ── Output formatters ──

function formatModelInfo(metadata, arch, handler) {
  const info = {
    name: metadata['general.name'] || metadata['general.basename'] || arch,
    architecture: arch,
    categories: handler.categories,
    contextLength: getMeta(metadata, `${arch}.context_length`),
    vocabSize: getMeta(metadata, `${arch}.vocab_size`),
    layers: getMeta(metadata, `${arch}.block_count`),
    mtpLayers: getMeta(metadata, `${arch}.nextn_predict_layers`),
    embeddingLength: getMeta(metadata, `${arch}.embedding_length`),
    headCount: getMeta(metadata, `${arch}.attention.head_count`),
    headCountKv: getMeta(metadata, `${arch}.attention.head_count_kv`),
    feedForwardLength: getMeta(metadata, `${arch}.feed_forward_length`),
  };
  if (handler.categories.includes('mla')) {
    info.kvLoraRank = getMeta(metadata, `${arch}.attention.kv_lora_rank`);
    info.qLoraRank = getMeta(metadata, `${arch}.attention.q_lora_rank`);
    info.keyLengthMla = getMeta(metadata, `${arch}.attention.key_length_mla`);
    info.valueLengthMla = getMeta(metadata, `${arch}.attention.value_length_mla`);
  }
  if (handler.categories.includes('iswa')) {
    info.slidingWindow = getMeta(metadata, `${arch}.attention.sliding_window`) || 'off';
  }
  const expertCount = getMeta(metadata, `${arch}.expert_count`);
  if (expertCount > 0) {
    info.expertCount = expertCount;
    info.expertUsedCount = getMeta(metadata, `${arch}.expert_used_count`);
  }
  return info;
}

function formatWeights(weightInfo) {
  return {
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
  };
}

function formatKvCache(kvCache, args) {
  return {
    kvCache: {
      bytesK: kvCache.bytesK,
      bytesKFormatted: formatBytes(kvCache.bytesK),
      bytesV: kvCache.bytesV,
      bytesVFormatted: formatBytes(kvCache.bytesV),
      bytesRecurrent: kvCache.bytesRecurrent,
      bytesRecurrentFormatted: formatBytes(kvCache.bytesRecurrent),
      totalBytes: kvCache.totalBytes,
      totalBytesFormatted: formatBytes(kvCache.totalBytes),
      layers: kvCache.layers,
      headDimK: kvCache.headDimK,
      headDimV: kvCache.headDimV,
      totalHeadsKV: kvCache.totalHeadsKV,
      avgHeadsKV: kvCache.avgHeadsKV,
      kvTypeK: QUANT_NAMES[args.kvTypeK] || String(args.kvTypeK),
      kvTypeV: QUANT_NAMES[args.kvTypeV] || String(args.kvTypeV),
    },
  };
}

function formatActivations(activations) {
  return {
    activations: {
      totalBytes: activations.totalBytes,
      totalBytesFormatted: formatBytes(activations.totalBytes),
      perLayerBytes: activations.perLayerBytes,
      perLayerBytesFormatted: formatBytes(activations.perLayerBytes),
      isMoe: activations.isMoe,
      expertCount: activations.expertCount,
      expertUsedCount: activations.expertUsedCount,
    },
  };
}

function formatMoe(moeInfo) {
  if (!moeInfo) return { moe: null };
  return {
    moe: {
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
      totalParams: moeInfo.totalModelParams,
      totalParamsFormatted: formatElements(moeInfo.totalModelParams),
      expertParams: moeInfo.expertParams,
      expertParamsFormatted: formatElements(moeInfo.expertParams),
      activeExpertWeightBytes: moeInfo.activeExpertWeightBytes,
      activeExpertWeightBytesFormatted: formatBytes(moeInfo.activeExpertWeightBytes),
    },
  };
}

function formatMmProj(mmProjInfo, args, resolved, mmProjUrl, mmProjBytes) {
  if (!mmProjInfo) return { mmproj: null };
  return {
    mmproj: {
      filename: resolved.mmproj.filename,
      autoDetected: resolved.mmprojAutoDetected,
      url: mmProjUrl,
      placement: args.mmprojDevice,
      hasVision: mmProjInfo.hasVision,
      hasAudio: mmProjInfo.hasAudio,
      isAudioProj: mmProjInfo.isAudioProj,
      projType: mmProjInfo.projType,
      projTypeKnown: mmProjInfo.projTypeKnown,
      imageSize: mmProjInfo.imageSize,
      patchSize: mmProjInfo.patchSize,
      nLayerV: mmProjInfo.nLayerV,
      nEmbdV: mmProjInfo.nEmbdV,
      projDim: mmProjInfo.projDim,
      nMerge: mmProjInfo.nMerge,
      nOutputTokens: mmProjInfo.nOutputTokens,
      weightBytes: mmProjInfo.weightBytes,
      weightBytesFormatted: formatBytes(mmProjInfo.weightBytes),
      perImageActBytes: mmProjInfo.perImageActBytes,
      perImageActBytesFormatted: formatBytes(mmProjInfo.perImageActBytes),
      totalBytes: mmProjBytes,
      totalBytesFormatted: formatBytes(mmProjBytes),
    },
  };
}

function formatPerformance(device, metadata, tensorInfos, ctxSize, args, kvCache, moeInfo, activations, mmProjInfo) {
  const perf = estimatePerformance({
    metadata, tensorInfos, ctx: ctxSize, batchSize: args.batchSize,
    kv: kvCache, moe: moeInfo, activations, mmproj: mmProjInfo,
    device,
  });
  return {
    decodeTPS: +perf.decodeTPS.toFixed(2),
    prefillTPS: +perf.prefillTPS.toFixed(2),
    ttftSec: +perf.ttftSec.toFixed(4),
    nGpuLayers: perf.nGpuLayers,
    nHybridLayers: perf.nHybridLayers || 0,
    nPartialLayers: perf.nPartialLayers || 0,
    nCpuLayers: perf.nCpuLayers,
    autoSplit: perf.autoSplit,
    cpuMoe: perf.cpuMoe,
    nCpuMoe: perf.nCpuMoe,
    perLayerMs: {
      gpu: +perf.perLayerMs.gpu.toFixed(3),
      hybrid: +(perf.perLayerMs.hybrid || 0).toFixed(3),
      partial: +(perf.perLayerMs.partial || 0).toFixed(3),
      cpu: +perf.perLayerMs.cpu.toFixed(3),
    },
    bottleneck: perf.bottleneck,
    timing: perf.timing,
    footprint: {
      nLayers: perf.footprint.nLayers,
      outputBytes: perf.footprint.outputBytes,
      outputBytesFormatted: formatBytes(perf.footprint.outputBytes),
      avgLayerActiveBytes: Math.round(perf.footprint.avgLayerActiveBytes),
      avgLayerActiveBytesFormatted: formatBytes(perf.footprint.avgLayerActiveBytes),
      avgLayerFullBytes: Math.round(perf.footprint.avgLayerFullBytes),
      avgLayerFullBytesFormatted: formatBytes(perf.footprint.avgLayerFullBytes),
      kvBytesPerLayer: Math.round(perf.footprint.kvBytesPerLayer),
      kvBytesPerLayerFormatted: formatBytes(perf.footprint.kvBytesPerLayer),
    },
    gpu: {
      name: device.gpu.preset ? device.gpu.preset.name : 'Custom',
      id: device.gpu.preset ? device.gpu.preset.id : null,
      fp16Tflops: device.gpu.flopsFp16Tflops,
      memBwGBps: device.gpu.bwGBps,
      vramGiB: device.gpu.vramBytes ? +(device.gpu.vramBytes / GIB).toFixed(2) : 0,
    },
    cpu: device.cpu ? {
      name: device.cpu.preset ? (device.cpu.fallback ? `${device.cpu.preset.name} (fallback)` : device.cpu.preset.name) : 'Custom',
      id: device.cpu.preset ? device.cpu.preset.id : null,
      fp16Tflops: device.cpu.flopsFp16Tflops,
      ramBwGBps: device.cpu.bwGBps,
    } : null,
  };
}

function calcVramRamFit(args, activations, mmProjInfo, layerFootprint, ramBytes, unifiedMemory = false) {
  if (args.vram <= 0 && args.ram <= 0) return { vramFit: null, ramFit: null };
  const mmprojActBytes = mmProjInfo ? (mmProjInfo.weightBytes + (mmProjInfo.perImageActBytes || 0)) : 0;
  const reservedBytes = activations.totalBytes + (args.mmprojDevice !== 'ram' ? mmprojActBytes : 0);
  let actualRamTotal = ramBytes;
  let vramFit = null;
  if (args.vram > 0) {
    const vramAvailBytes = args.vram * GIB;
    const actual = calcActualMemory({
      vramBytes: vramAvailBytes,
      footprint: layerFootprint,
      activationBytes: reservedBytes,
      cpuMoe: args.cpuMoe,
      nCpuMoe: args.nCpuMoe,
      unifiedMemory,
    });
    const usagePct = actual.actualVram / vramAvailBytes * 100;
    actualRamTotal = actual.actualRam + (args.mmprojDevice === 'ram' ? mmprojActBytes : 0);
    vramFit = {
      availableGiB: args.vram,
      actualVramGiB: +(actual.actualVram / GIB).toFixed(2),
      actualRamGiB: +(actualRamTotal / GIB).toFixed(2),
      fits: actual.actualVram <= vramAvailBytes,
      usagePct: +usagePct.toFixed(1),
      nGpuLayers: actual.nGpuLayers,
      nHybridLayers: actual.nHybridLayers,
      nPartialLayers: actual.nPartialLayers || 0,
      nCpuLayers: actual.nCpuLayers,
      vramBreakdownGiB: {
        weights: +(actual.vramWeightsBytes / GIB).toFixed(2),
        experts: +(actual.vramExpertBytes / GIB).toFixed(2),
        kv: +(actual.vramKvBytes / GIB).toFixed(2),
        activations: +(actual.vramActivationBytes / GIB).toFixed(2),
        output: +(actual.vramOutputBytes / GIB).toFixed(2),
      },
      ramBreakdownGiB: {
        experts: +(actual.ramExpertBytes / GIB).toFixed(2),
        cpuLayerWeights: +(actual.ramCpuLayerBytes / GIB).toFixed(2),
        cpuLayerKv: +(actual.ramCpuKvBytes / GIB).toFixed(2),
        inputEmb: +(actual.ramInputEmbBytes / GIB).toFixed(2),
      },
    };
  }
  let ramFit = null;
  if (args.ram > 0) {
    const ramAvailBytes = args.ram * GIB;
    const usagePct = actualRamTotal / ramAvailBytes * 100;
    ramFit = {
      availableGiB: args.ram,
      requiredGiB: +(actualRamTotal / GIB).toFixed(2),
      fits: actualRamTotal <= ramAvailBytes,
      usagePct: +usagePct.toFixed(1),
    };
  }
  return { vramFit, ramFit };
}

// ── Batch mode ──
async function runBatch(batchFile, args) {
  const lines = readRepoList(batchFile);
  const concurrency = args.concurrency;

  if (concurrency <= 1) {
    const results = [];
    for (let i = 0; i < lines.length; i++) {
      const repo = lines[i];
      process.stderr.write(`[${i + 1}/${lines.length}] ${repo}... `);
      try {
        const data = await processRepo(repo, args);
        results.push({ success: true, data });
        if (args.all && data.files) {
          console.error(`done (${data.files.length} files)`);
        } else {
          console.error(`done (${data.arch}, ${data.weightBytesFormatted})`);
        }
      } catch (err) {
        console.error(`failed: ${err.message}`);
        results.push({ success: false, repo, error: err.message });
      }
    }
    console.log(JSON.stringify(results, null, 2));
  } else {
    let progress = 0;
    const results = await parallelMap(lines, async (repo) => {
      const data = await processRepo(repo, args);
      progress++;
      const label = args.all && data.files
        ? `${data.files.length} files`
        : data.arch;
      process.stderr.write(`[${progress}/${lines.length}] ${repo} -> ${label}\n`);
      return data;
    }, concurrency);
    console.log(JSON.stringify(results, null, 2));
  }
}

// ── Entry point ──
const args = parseArgs(process.argv);

if (args.batch) {
  runBatch(args.batch, args).catch(err => {
    console.error(`Error: ${err.message}`);
    process.exit(1);
  });
} else if (args.repo) {
  processRepo(args.repo, args).then(result => {
    console.log(JSON.stringify(result, null, 2));
  }).catch(err => {
    console.error(`Error: ${err.message}`);
    process.exit(1);
  });
} else {
  console.error(USAGE);
  process.exit(1);
}
