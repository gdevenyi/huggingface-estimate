import { resolveHFModel, parseGGUF, buildResolveUrl, GGMLQuantizationType, KV_VALID_QUANTS } from './parsing.js';
import {
  getArchHandler,
  getModelArch,
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
import { mergeCpuPresets, mergeGpuPresets, findCpuPreset, getGpuPresets } from './hardware-presets.js';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

const CPU_JSON_FILES = ['apple-cpu-presets.json', 'intel-cpu-presets.json', 'amd-cpu-presets.json'];
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

// ── CLI argument parsing ──
// Accepts: standard quant names (F16, Q8_0, ...), numeric GGML type IDs, and
// fork-specific extension names stripped of their "(ik_llama)" / "(rotorquant)"
// suffix (Q6_0, Q8_KV, TURBO3_0, ...). Only KV-legal types are accepted.
function parseKvType(val, flag) {
  // Standard @huggingface/gguf enum name → numeric ID.
  if (GGMLQuantizationType[val] !== undefined) {
    const id = GGMLQuantizationType[val];
    if (KV_VALID_QUANTS.includes(id)) return id;
  }
  // Direct BPE key (e.g. rotorquant string key "TURBO3_0").
  if (KV_VALID_QUANTS.includes(val)) return val;
  // Numeric ID (e.g. "133" for Q6_0, "151" for Q8_KV).
  const num = parseInt(val, 10);
  if (!Number.isNaN(num) && KV_VALID_QUANTS.includes(num)) return num;
  // Fork-extension name lookup via QUANT_NAMES reverse map ("Q6_0" → 133).
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
  const args = {
    repo: null,
    batch: null,
    ctx: 4096,
    batchSize: 1,
    kvTypeK: GGMLQuantizationType.F16,
    kvTypeV: GGMLQuantizationType.F16,
    vram: 0,
    ram: 0,
    mmproj: null,
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
    } else if (arg === '--ctx') {
      args.ctx = parseInt(needValue(arg), 10);
    } else if (arg === '--batchSize') {
      args.batchSize = parseInt(needValue(arg), 10);
    } else if (arg === '--kvTypeK') {
      args.kvTypeK = parseKvType(needValue(arg), '--kvTypeK');
    } else if (arg === '--kvTypeV') {
      args.kvTypeV = parseKvType(needValue(arg), '--kvTypeV');
    } else if (arg === '--vram') {
      args.vram = Math.max(0, parseFloat(needValue(arg)) || 0);
    } else if (arg === '--ram') {
      args.ram = Math.max(0, parseFloat(needValue(arg)) || 0);
    } else if (arg === '--mmproj') {
      args.mmproj = needValue(arg);
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
      args.ngl = v === 'auto' ? 'auto' : parseInt(v, 10);
    } else if (arg === '--cpu-moe') {
      args.cpuMoe = true;
    } else if (arg === '--n-cpu-moe') {
      const v = parseInt(needValue(arg), 10);
      if (Number.isNaN(v) || v < 0) {
        console.error('Error: --n-cpu-moe requires a non-negative integer');
        process.exit(1);
      }
      args.nCpuMoe = v;
    } else if (!arg.startsWith('-')) {
      args.repo = arg;
    }
    i++;
  }

  return args;
}

// Resolve hardware presets + manual overrides into a device spec for the
// performance estimator. Returns null if no GPU spec was supplied.
function resolveDevice(args) {
  const gpuPreset = args.gpu ? findGpuPreset(args.gpu) : null;
  if (args.gpu && !gpuPreset && (args.gpuFlops == null || args.gpuBw == null)) {
    console.error(`Warning: GPU preset "${args.gpu}" not found in GPU preset files.`);
  }
  const gpuFlops = args.gpuFlops != null ? args.gpuFlops : (gpuPreset ? gpuPreset.fp16Tflops : null);
  const gpuBw = args.gpuBw != null ? args.gpuBw : (gpuPreset ? gpuPreset.memBwGBps : null);
  if (gpuFlops == null || gpuBw == null) return null;

  const cpuPreset = args.cpu ? findCpuPreset(args.cpu) : null;
  if (args.cpu && !cpuPreset && (args.cpuFlops == null || args.ramBw == null)) {
    console.error(`Warning: CPU preset "${args.cpu}" not found in CPU preset files.`);
  }
  const cpuFlops = args.cpuFlops != null ? args.cpuFlops : (cpuPreset ? cpuPreset.fp16Tflops : null);
  const ramBw = args.ramBw != null ? args.ramBw : (cpuPreset ? cpuPreset.defaultRamBwGBps : null);
  const cpu = (cpuFlops != null && ramBw != null) ? { flopsFp16Tflops: cpuFlops, bwGBps: ramBw } : null;

  return {
    gpu: {
      flopsFp16Tflops: gpuFlops,
      bwGBps: gpuBw,
      vramBytes: args.vram > 0 ? args.vram * (1024 ** 3) : 0,
      preset: gpuPreset,
    },
    cpu: cpu ? { ...cpu, preset: cpuPreset } : null,
    nGpuLayers: args.ngl === 'auto' ? 'auto' : args.ngl,
    cpuMoe: args.cpuMoe,
    nCpuMoe: args.nCpuMoe,
  };
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
  let maxBytes = 0;
  for (const [name, info] of Object.entries(weightInfo.byQuant)) {
    if (info.bytes > maxBytes) {
      maxBytes = info.bytes;
      primaryQuant = name;
    }
  }

  // KV cache
  const kvCache = calcKVCache(metadata, args.ctx, args.kvTypeK, args.kvTypeV);

  // Activations
  const activations = calcActivations(metadata, args.batchSize);

  // MoE info
  const moeInfo = calcMoEInfo(metadata, tensorInfos);
  const isMoe = moeInfo !== null;

  // Per-layer footprint for MoE expert distribution
  const layerFootprint = calcPerLayerFootprint(metadata, tensorInfos, kvCache, moeInfo);

  // VRAM / RAM breakdown — default: all weights (including all experts) in VRAM.
  // --cpu-moe: all expert weights in RAM. --n-cpu-moe N: first N layers' experts in RAM.
  const memBreakdown = calcMemoryBreakdown({
    weights: weightInfo,
    moe: moeInfo,
    kv: kvCache,
    activations,
    footprint: layerFootprint,
    cpuMoe: args.cpuMoe,
    nCpuMoe: args.nCpuMoe,
  });
  let vramBytes = memBreakdown.vramBytes;
  let ramBytes = memBreakdown.ramBytes;

  // mmproj (optional multimodal projector)
  let mmProjInfo = null;
  let mmProjUrl = null;
  let mmProjBytes = 0;
  if (args.mmproj) {
    mmProjUrl = buildResolveUrl(repo, args.mmproj);
    const mmParsed = await parseGGUF(mmProjUrl);
    mmProjInfo = calcMmProj(mmParsed.metadata, mmParsed.tensorInfos);
    if (mmProjInfo) {
      mmProjBytes = mmProjInfo.weightBytes + (mmProjInfo.perImageActBytes || 0);
      if (args.mmprojDevice === 'ram') ramBytes += mmProjBytes;
      else vramBytes += mmProjBytes;
    }
  }

  // Total parameters
  const totalParams = tensorInfos.reduce((s, t) => s + t.shape.map(Number).reduce((a, b) => a * b, 1), 0);

  // Performance estimate (optional — omitted when no GPU flags supplied)
  const device = resolveDevice(args);
  let performance = null;
  if (device) {
    const perf = estimatePerformance({
      metadata, tensorInfos, ctx: args.ctx, batchSize: args.batchSize,
      kv: kvCache, moe: moeInfo, activations, mmproj: mmProjInfo,
      device,
    });
    performance = {
      decodeTPS: +perf.decodeTPS.toFixed(2),
      prefillTPS: +perf.prefillTPS.toFixed(2),
      ttftSec: +perf.ttftSec.toFixed(4),
      nGpuLayers: perf.nGpuLayers,
      nHybridLayers: perf.nHybridLayers || 0,
      nCpuLayers: perf.nCpuLayers,
      autoSplit: perf.autoSplit,
      cpuMoe: perf.cpuMoe,
      nCpuMoe: perf.nCpuMoe,
      perLayerMs: {
        gpu: +perf.perLayerMs.gpu.toFixed(3),
        hybrid: +(perf.perLayerMs.hybrid || 0).toFixed(3),
        cpu: +perf.perLayerMs.cpu.toFixed(3),
      },
      bottleneck: perf.bottleneck,
      gpu: {
        name: device.gpu.preset ? device.gpu.preset.name : 'Custom',
        id: device.gpu.preset ? device.gpu.preset.id : null,
        fp16Tflops: device.gpu.flopsFp16Tflops,
        memBwGBps: device.gpu.bwGBps,
        vramGiB: device.gpu.vramBytes ? +(device.gpu.vramBytes / (1024 ** 3)).toFixed(2) : 0,
      },
      cpu: device.cpu ? {
        name: device.cpu.preset ? device.cpu.preset.name : 'Custom',
        id: device.cpu.preset ? device.cpu.preset.id : null,
        fp16Tflops: device.cpu.flopsFp16Tflops,
        ramBwGBps: device.cpu.bwGBps,
      } : null,
    };
  }

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
      headDimK: kvCache.headDimK,
      headDimV: kvCache.headDimV,
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
    mmproj: mmProjInfo ? {
      filename: args.mmproj,
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
    } : null,
    vramBytes,
    vramBytesFormatted: formatBytes(vramBytes),
    ramBytes,
    ramBytesFormatted: formatBytes(ramBytes),
    ...(() => {
      if (args.vram <= 0 && args.ram <= 0) return { vramFit: null, ramFit: null };
      const mmprojActBytes = mmProjInfo ? (mmProjInfo.weightBytes + (mmProjInfo.perImageActBytes || 0)) : 0;
      const reservedBytes = activations.totalBytes + (args.mmprojDevice !== 'ram' ? mmprojActBytes : 0);
      let actualRamTotal = ramBytes;
      let vramFit = null;
      if (args.vram > 0) {
        const vramAvailBytes = args.vram * (1024 ** 3);
        const actual = calcActualMemory({
          vramBytes: vramAvailBytes,
          footprint: layerFootprint,
          activationBytes: reservedBytes,
          cpuMoe: args.cpuMoe,
          nCpuMoe: args.nCpuMoe,
        });
        const usagePct = actual.actualVram / vramAvailBytes * 100;
        actualRamTotal = actual.actualRam + (args.mmprojDevice === 'ram' ? mmprojActBytes : 0);
        vramFit = {
          availableGiB: args.vram,
          actualVramGiB: +(actual.actualVram / (1024 ** 3)).toFixed(2),
          actualRamGiB: +(actualRamTotal / (1024 ** 3)).toFixed(2),
          fits: actual.actualVram <= vramAvailBytes,
          usagePct: +usagePct.toFixed(1),
          nGpuLayers: actual.nGpuLayers,
          nHybridLayers: actual.nHybridLayers,
          nCpuLayers: actual.nCpuLayers,
        };
      }
      let ramFit = null;
      if (args.ram > 0) {
        const ramAvailBytes = args.ram * (1024 ** 3);
        const usagePct = actualRamTotal / ramAvailBytes * 100;
        ramFit = {
          availableGiB: args.ram,
          requiredGiB: +(actualRamTotal / (1024 ** 3)).toFixed(2),
          fits: actualRamTotal <= ramAvailBytes,
          usagePct: +usagePct.toFixed(1),
        };
      }
      return { vramFit, ramFit };
    })(),
    performance,
  };
}

// ── Batch mode ──
async function runBatch(batchFile) {
  const lines = readFileSync(batchFile, 'utf-8')
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
  console.error(`Usage: node run-calc.js <repo> [options]
       node run-calc.js --batch testModels.list

Arguments:
  <repo>             HuggingFace repo (e.g. unsloth/Qwen3-8B-GGUF)
  --batch <file>     Process all repos from a file (one per line)
  --ctx <N>          Context size (default: 4096)
  --batchSize <N>    Batch size (default: 1)
  --kvTypeK <T>      KV cache K quantization type (name or number, default: F16)
  --kvTypeV <T>      KV cache V quantization type (name or number, default: F16)
  --vram <N>         Available VRAM in GiB (enables VRAM fit check + performance split)
  --ram <N>          Available system RAM in GiB (enables RAM fit check)
  --mmproj <file>    mmproj GGUF filename within the repo (e.g. mmproj-F16.gguf)
  --mmprojDevice <d> Where to place mmproj: vram (default) or ram (--no-mmproj-offload)

Performance estimation (supply --gpu or --gpu-flops + --gpu-bw to enable):
  --gpu <name|id>    GPU preset (e.g. "RTX 4090", "nvidia-geforce-rtx-4090")
  --gpu-flops <TF>   Override GPU FP16 TFLOPS
  --gpu-bw <GB/s>    Override GPU memory bandwidth
  --cpu <name|id>    CPU preset from hardware-presets.js (e.g. "Ryzen 9 7950X")
  --cpu-flops <TF>   Override CPU FP16 TFLOPS
  --ram-bw <GB/s>    Override system RAM bandwidth
  --ngl <n|auto>     GPU layer override (default: auto, sized from --vram)
  --cpu-moe          Keep all MoE expert weights in CPU (llama.cpp -cmoe)
  --n-cpu-moe <N>    Keep MoE expert weights of first N layers in CPU (llama.cpp -ncmoe)

Quantization type names: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, Q8_K, ...
`);
  process.exit(1);
}
