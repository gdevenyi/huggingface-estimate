// Hand-curated RAM presets for the tokens/sec estimator.
//
// CPU and GPU presets live in per-vendor JSON files:
//   apple-cpu-presets.json, intel-cpu-presets.json, amd-cpu-presets.json
//   nvidia-gpu-presets.json, intel-gpu-presets.json, amd-gpu-presets.json
//
// CPU FP16 FLOPS approximation: cores x boostGHz x FLOPsPerCycle, where
// FLOPsPerCycle is the effective FP16 throughput per core per cycle using
// the widest available SIMD (AVX2 = 16, AVX-512 = 32, NEON = 8, Zen 1/+
// AVX2 on 128-bit FPU = 8). Same formula used by gguf-parser-go's docs
// (README.md section CPU FLOPS), pessimistic vs. real-world tensor-core-
// equivalent CPU kernels but good enough as a decode-time lower bound
// (decode is bandwidth-bound anyway).
//
// RAM bandwidth presets are theoretical peaks per channel-configuration.

// ── RAM presets (kept here, not vendor-specific) ──
export const RAM_PRESETS = [
  { id: 'ddr5-8000-dc',  name: 'DDR5-8000 dual-channel',    bandwidthGBps: 128 },
  { id: 'ddr5-7200-dc',  name: 'DDR5-7200 dual-channel',    bandwidthGBps: 115 },
  { id: 'ddr5-6400-dc',  name: 'DDR5-6400 dual-channel',    bandwidthGBps: 102 },
  { id: 'ddr5-6000-dc',  name: 'DDR5-6000 dual-channel',    bandwidthGBps: 96 },
  { id: 'ddr5-5600-dc',  name: 'DDR5-5600 dual-channel',    bandwidthGBps: 90 },
  { id: 'ddr5-4800-dc',  name: 'DDR5-4800 dual-channel',    bandwidthGBps: 77 },
  { id: 'ddr4-3600-dc',  name: 'DDR4-3600 dual-channel',    bandwidthGBps: 58 },
  { id: 'ddr4-3200-dc',  name: 'DDR4-3200 dual-channel',    bandwidthGBps: 51 },
  { id: 'ddr4-2933-dc',  name: 'DDR4-2933 dual-channel',    bandwidthGBps: 47 },
  { id: 'ddr4-2666-dc',  name: 'DDR4-2666 dual-channel',    bandwidthGBps: 42 },
  { id: 'ddr4-2400-dc',  name: 'DDR4-2400 dual-channel',    bandwidthGBps: 38 },
  { id: 'ddr4-2133-dc',  name: 'DDR4-2133 dual-channel',    bandwidthGBps: 34 },

  { id: 'ddr4-3200-4ch', name: 'DDR4-3200 quad-channel (sTRX4)',    bandwidthGBps: 102 },
  { id: 'ddr4-2666-4ch', name: 'DDR4-2666 quad-channel (X299/X399)',bandwidthGBps: 85 },

  { id: 'ddr5-4800-8ch', name: 'DDR5-4800 8-channel server',bandwidthGBps: 307 },
  { id: 'ddr5-5600-8ch', name: 'DDR5-5600 8-channel server',bandwidthGBps: 358 },
  { id: 'ddr5-6400-12ch',name: 'DDR5-6400 12-channel EPYC', bandwidthGBps: 614 },
  { id: 'ddr4-3200-8ch', name: 'DDR4-3200 8-channel server',bandwidthGBps: 205 },
  { id: 'ddr4-2666-8ch', name: 'DDR4-2666 8-channel (EPYC Naples)', bandwidthGBps: 170 },
  { id: 'ddr4-2666-6ch', name: 'DDR4-2666 6-channel (Skylake-SP)',  bandwidthGBps: 128 },

  { id: 'lpddr5x-8533',  name: 'LPDDR5X-8533 quad-channel', bandwidthGBps: 273 },
  { id: 'lpddr5-6400-qc',name: 'LPDDR5-6400 quad-channel',  bandwidthGBps: 205 },

  { id: 'apple-um-68',   name: 'Apple unified 68 GB/s (M1 base)',     bandwidthGBps: 68 },
  { id: 'apple-um-120',  name: 'Apple unified 120 GB/s (M4 base)',    bandwidthGBps: 120 },
  { id: 'apple-um-200',  name: 'Apple unified 200 GB/s (M2 Pro)',     bandwidthGBps: 200 },
  { id: 'apple-um-273',  name: 'Apple unified 273 GB/s (M4 Pro)',     bandwidthGBps: 273 },
  { id: 'apple-um-400',  name: 'Apple unified 400 GB/s (M1/2/3 Max)', bandwidthGBps: 400 },
  { id: 'apple-um-546',  name: 'Apple unified 546 GB/s (M4 Max)',     bandwidthGBps: 546 },
  { id: 'apple-um-819',  name: 'Apple unified 819 GB/s (M1/2/3 Ultra)', bandwidthGBps: 819 },
];

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

export function findRamPreset(query) {
  if (!query) return null;
  const q = query.toLowerCase();
  return RAM_PRESETS.find(p => p.id === q)
      || RAM_PRESETS.find(p => p.name.toLowerCase() === q)
      || RAM_PRESETS.find(p => p.name.toLowerCase().includes(q))
      || null;
}
