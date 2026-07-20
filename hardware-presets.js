// Hardware preset loading/lookup for the tokens/sec estimator.
//
// CPU and GPU presets live in per-vendor JSON files:
//   intel-cpu-presets.json, amd-cpu-presets.json
//   nvidia-gpu-presets.json, intel-gpu-presets.json, amd-gpu-presets.json, apple-gpu-presets.json
//
// The "Unified Memory" CPU preset (used when a unified-memory GPU is selected)
// is defined here as UNIFIED_MEMORY_CPU_PRESET, not in a vendor JSON file.
//
// CPU FP16 FLOPS approximation: cores x boostGHz x FLOPsPerCycle, where
// FLOPsPerCycle is the effective FP16 throughput per core per cycle using
// the widest available SIMD (AVX2 = 16, AVX-512 = 32, NEON = 8, Zen 1/+
// AVX2 on 128-bit FPU = 8). Same formula used by gguf-parser-go's docs
// (README.md section CPU FLOPS), pessimistic vs. real-world tensor-core-
// equivalent CPU kernels but good enough as a decode-time lower bound
// (decode is bandwidth-bound anyway).

// ── CPU ID aliases for backward compat ──
const CPU_ID_ALIASES = {
  'ryzen-1600x': 'ryzen-5-1600x',
  'ryzen-1800x': 'ryzen-7-1800x',
  'ryzen-2600x': 'ryzen-5-2600x',
  'ryzen-2700x': 'ryzen-7-2700x',
  'ryzen-3600': 'ryzen-5-3600',
  'ryzen-3700x': 'ryzen-7-3700x',
  'ryzen-3900x': 'ryzen-9-3900x',
  'ryzen-3950x': 'ryzen-9-3950x',
  'ryzen-5600x': 'ryzen-5-5600x',
  'ryzen-5700x': 'ryzen-7-5700x',
  'ryzen-5800x': 'ryzen-7-5800x',
  'ryzen-5900x': 'ryzen-9-5900x',
  'ryzen-5950x': 'ryzen-9-5950x',
  'ryzen-7600x': 'ryzen-5-7600x',
  'ryzen-7700x': 'ryzen-7-7700x',
  'ryzen-7900x': 'ryzen-9-7900x',
  'ryzen-7950x': 'ryzen-9-7950x',
  'ryzen-7950x3d': 'ryzen-9-7950x3d',
  'ryzen-9600x': 'ryzen-5-9600x',
  'ryzen-9700x': 'ryzen-7-9700x',
  'ryzen-9900x': 'ryzen-9-9900x',
  'ryzen-9950x': 'ryzen-9-9950x',
  'tr-1950x': 'ryzen-threadripper-1950x',
  'tr-2990wx': 'ryzen-threadripper-2990wx',
  'tr-3970x': 'ryzen-threadripper-3970x',
  'tr-3990x': 'ryzen-threadripper-3990x',
  'tr-pro-5995wx': 'ryzen-threadripper-pro-5995wx',
  'tr-pro-7995wx': 'ryzen-threadripper-pro-7995wx',
};

// ── Dynamic preset loading ──
// CPU/GPU presets are loaded from per-vendor JSON files at runtime.
// Call mergeCpuPresets()/mergeGpuPresets() after loading each file.

let _cpuPresets = [];
let _gpuPresets = [];

export const UNIFIED_MEMORY_CPU_PRESET = { id: 'unified-memory', name: 'Unified Memory' };

export function mergeCpuPresets(presets) {
  _cpuPresets.push(...(presets || []));
}

export function getCpuPresets() {
  return _cpuPresets;
}

export function mergeGpuPresets(presets) {
  _gpuPresets.push(...(presets || []));
}

export function getGpuPresets() {
  return _gpuPresets;
}

export function findCpuPreset(query) {
  if (!query) return null;
  const q = query.toLowerCase();
  const alias = CPU_ID_ALIASES[q];
  return _cpuPresets.find(p => p.id === q || p.id === alias)
      || _cpuPresets.find(p => p.name.toLowerCase() === q)
      || _cpuPresets.find(p => p.name.toLowerCase().includes(q))
      || null;
}

export function getSlowestCpuPreset() {
  let best = null;
  for (const p of _cpuPresets) {
    if (p.fp16Tflops == null || p.defaultRamBwGBps == null) continue;
    if (!best || p.fp16Tflops < best.fp16Tflops
        || (p.fp16Tflops === best.fp16Tflops && p.defaultRamBwGBps < best.defaultRamBwGBps)) {
      best = p;
    }
  }
  return best;
}
