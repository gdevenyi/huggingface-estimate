#!/usr/bin/env node
// Reads specs/apple_silicon.tsv and emits apple-gpu-presets.json.
// One GPU preset per TSV row (per RAM SKU).
//
// Run: `node scripts/build-apple-presets.js`

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { round, stripUnderscored } from './lib/format.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const TSV_PATH = join(ROOT, 'specs', 'apple_silicon.tsv');
const GPU_OUT = join(ROOT, 'apple-gpu-presets.json');

// Vendor-specific (Apple TSV uses a different slug convention than the CSV
// scripts — drops hyphens, not preserved — kept local on purpose).
function slug(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .replace(/[\s_]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

// Empirical decode bandwidth-utilization scale per (generation, tier),
// derived from the tps calculator's Apple Silicon calibration (see AGENTS.md,
// resources/tps/). Consumed by estimatePerformance as gpu.preset.decodeBwScale.
// Previously these values were hand-patched into apple-gpu-presets.json after
// generation, which made the JSON unreproducible from this script.
const DECODE_BW_SCALE = {
  'M1 Base': 0.5,  'M1 Pro': 0.5,  'M1 Max': 0.49, 'M1 Ultra': 0.55,
  'M2 Base': 0.55, 'M2 Pro': 0.55, 'M2 Max': 0.58, 'M2 Ultra': 0.65,
  'M3 Base': 0.6,  'M3 Pro': 0.65, 'M3 Max': 0.76, 'M3 Ultra': 0.72,
  'M4 Base': 0.65, 'M4 Pro': 0.75, 'M4 Max': 0.85,
  'M5 Base': 0.65, 'M5 Pro': 0.75, 'M5 Max': 0.85,
};

const TIER_ORDER = { ultra: 0, max: 1, pro: 2, base: 3 };

function tierRank(tier) {
  return TIER_ORDER[tier.toLowerCase()] ?? 4;
}

function chipName(generation, tier) {
  return tier === 'Base' ? generation : `${generation} ${tier}`;
}

function parseRam(s) {
  const m = s.match(/(\d+)/);
  return m ? parseInt(m[1], 10) : null;
}

function isMobile(macModel) {
  return /^MacBook/.test(macModel);
}

function isDesktop(macModel) {
  return /^(iMac|Mac mini|Mac Studio|Mac Pro)/.test(macModel);
}

// ── Parse TSV ──
const text = readFileSync(TSV_PATH, 'utf8');
const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
const header = lines[0].split('\t');
const COL = Object.fromEntries(header.map((h, i) => [h.trim(), i]));

const gpuPresets = [];

for (let i = 1; i < lines.length; i++) {
  const line = lines[i].trim();
  if (!line) continue;

  const cols = line.split('\t');
  const generation = (cols[COL['Generation']] || '').trim();
  const tier = (cols[COL['Tier']] || '').trim();
  const cpuCores = parseInt(cols[COL['CPU Cores']] || '0', 10) || null;
  const gpuCores = parseInt(cols[COL['GPU Cores']] || '0', 10) || null;
  const fp32Tflops = parseFloat(cols[COL['FP32 (TFLOPS)']] || '') || null;
  const fp16Tflops = parseFloat(cols[COL['FP16 (TFLOPS)']] || '') || null;
  const memBwGBps = parseFloat(cols[COL['Memory Bandwidth (GB/s)']] || '') || null;
  const macModel = (cols[COL['Mac Model']] || '').trim();
  const ramStr = (cols[COL['RAM']] || '').trim();
  const ramGB = parseRam(ramStr);
  const year = parseInt(cols[COL['Model Year']] || '0', 10) || null;

  if (!macModel || !memBwGBps || !ramGB) continue;

  const chip = chipName(generation, tier);
  const name = `${macModel} (${chip} ${year}), ${cpuCores}c CPU / ${gpuCores}c GPU, ${ramStr}`;
  const id = slug(name);

  const flags = {};
  if (isMobile(macModel)) flags.mobile = true;
  if (isDesktop(macModel)) flags.desktop = true;

  gpuPresets.push({
    id,
    name,
    vendor: 'Apple',
    year,
    vramGB: ramGB,
    memBwGBps: round(memBwGBps, 1),
    fp16Tflops: fp16Tflops != null ? round(fp16Tflops, 2) : null,
    fp32Tflops: fp32Tflops != null ? round(fp32Tflops, 2) : null,
    memType: 'Unified',
    unifiedMemory: true,
    ...flags,
    // undefined for unknown (generation, tier) combos → key omitted by
    // JSON.stringify; the perf estimator then defaults to 1.0.
    decodeBwScale: DECODE_BW_SCALE[`${generation} ${tier}`],
    _year: year,
    _tier: tier,
    _gpuCores: gpuCores,
  });
}

gpuPresets.sort((a, b) => {
  if ((b._year ?? 0) !== (a._year ?? 0)) return (b._year ?? 0) - (a._year ?? 0);
  const ta = tierRank(a._tier), tb = tierRank(b._tier);
  if (ta !== tb) return ta - tb;
  if ((b._gpuCores ?? 0) !== (a._gpuCores ?? 0)) return (b._gpuCores ?? 0) - (a._gpuCores ?? 0);
  return a.name.localeCompare(b.name);
});

writeFileSync(GPU_OUT, JSON.stringify(gpuPresets.map(stripUnderscored), null, 2) + '\n');
console.error(`Wrote ${gpuPresets.length} Apple GPU presets to ${GPU_OUT}`);

const byYear = {};
for (const p of gpuPresets) {
  byYear[p._year] = (byYear[p._year] || 0) + 1;
}
console.error('By year:', byYear);
