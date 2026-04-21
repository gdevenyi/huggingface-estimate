// Reads apple_silicon_specs.csv and emits apple-cpu-presets.json and
// apple-gpu-presets.json. Apple Silicon is unified (GPU+CPU on-chip), so
// each chip variant produces both a CPU preset and a GPU preset.
//
// Run: `node scripts/build-apple-presets.js`

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const CSV_PATH = join(ROOT, 'resources', 'apple_silicon_specs.csv');
const CPU_OUT = join(ROOT, 'apple-cpu-presets.json');
const GPU_OUT = join(ROOT, 'apple-gpu-presets.json');

function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') { inQuotes = false; }
      else { field += c; }
    } else {
      if (c === '"') { inQuotes = true; }
      else if (c === ',') { row.push(field); field = ''; }
      else if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
      else if (c === '\r') { /* skip */ }
      else { field += c; }
    }
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows;
}

function parseFloat_(s) {
  if (!s) return null;
  const m = s.match(/([\d.]+)/);
  return m ? parseFloat(m[1]) : null;
}

function parseInt_(s) {
  if (!s) return null;
  const m = s.match(/(\d+)/);
  return m ? parseInt(m[1], 10) : null;
}

function round(n, d) {
  const m = Math.pow(10, d);
  return Math.round(n * m) / m;
}

function slug(chip) {
  return `apple-${chip}`
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .replace(/[\s_]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

const GEN_ORDER = { M5: 0, M4: 1, M3: 2, M2: 3, M1: 4, 'A-series in Mac': 5 };
const TIER_ORDER = { ultra: 0, max: 1, pro: 2, base: 3 };

function tierRank(name) {
  const n = name.toLowerCase();
  if (n.includes('ultra')) return 0;
  if (n.includes('max')) return 1;
  if (n.includes('pro')) return 2;
  if (n.includes('a18')) return 4;
  return 3;
}

function generationKey(name) {
  for (const [k, v] of Object.entries(GEN_ORDER)) {
    if (name.includes(k)) return v;
  }
  return 99;
}

function sortKey(rec) {
  const name = rec._chip;
  return [
    generationKey(name),
    tierRank(name),
    -(rec._gpuCores || 0),
    name,
  ];
}

// ── Parse CSV ──
const text = readFileSync(CSV_PATH, 'utf8');
const rows = parseCSV(text);
const header = rows[0].map(h => h.replace(/^\uFEFF/, ''));
const COL = Object.fromEntries(header.map((h, i) => [h, i]));

const cpuPresets = [];
const gpuPresets = [];

for (let r = 1; r < rows.length; r++) {
  const row = rows[r];
  if (!row || row.length < 10) continue;

  const chip = (row[COL.Chip] || '').trim();
  const generation = (row[COL.Generation] || '').trim();
  const perfCores = parseInt_(row[COL.Performance_Cores]);
  const effCores = parseInt_(row[COL.Efficiency_Cores]);
  const cpuFp32Tflops = parseFloat_(row[COL.CPU_FP32_TFLOPS_NEON]);
  const gpuCores = parseInt_(row[COL.GPU_Cores]);
  const gpuFp32Tflops = parseFloat_(row[COL.GPU_FP32_TFLOPS]);
  const memBwGBps = parseFloat_(row[COL.Memory_Bandwidth_GBps]);
  const maxMemGB = parseInt_(row[COL.Max_Unified_Memory_GB]);
  const year = parseInt_(row[COL.Release_Year]);
  const formFactor = (row[COL.Form_Factor] || '').trim();

  if (!chip || !memBwGBps) continue;

  const isMobile = formFactor === 'Mobile' || formFactor === 'Both';
  const isDesktop = formFactor === 'Desktop' || formFactor === 'Both';

  const id = slug(chip);
  const name = `Apple ${chip}`;

  // CPU FP16 = FP32 × 2 (NEON doubles throughput for FP16 vs FP32)
  const cpuFp16 = cpuFp32Tflops != null ? round(cpuFp32Tflops * 2, 2) : null;
  // GPU FP16 = FP32 × 2 (Apple GPU ALUs double throughput for FP16)
  const gpuFp16 = gpuFp32Tflops != null ? round(gpuFp32Tflops * 2, 2) : null;

  const cpuFlags = {};
  if (isMobile) cpuFlags.mobile = true;
  if (isDesktop) cpuFlags.desktop = true;

  const gpuFlags = {};
  if (isMobile) gpuFlags.mobile = true;
  if (isDesktop) gpuFlags.desktop = true;

  const sortMeta = { _chip: chip, _gpuCores: gpuCores };

  cpuPresets.push({
    id, name, vendor: 'Apple',
    fp16Tflops: cpuFp16,
    defaultRamBwGBps: round(memBwGBps, 1),
    ...cpuFlags,
    ...sortMeta,
  });

  gpuPresets.push({
    id, name, vendor: 'Apple',
    year: year ?? null,
    vramGB: maxMemGB,
    memBwGBps: round(memBwGBps, 1),
    fp16Tflops: gpuFp16,
    fp32Tflops: gpuFp32Tflops != null ? round(gpuFp32Tflops, 2) : null,
    memType: 'Unified',
    ...gpuFlags,
    ...sortMeta,
  });
}

// Sort: newer gen first, then higher tier, then more GPU cores
function cmp(a, b) {
  const ka = sortKey(a), kb = sortKey(b);
  for (let i = 0; i < Math.max(ka.length, kb.length); i++) {
    const x = ka[i], y = kb[i];
    if (x === undefined) return -1;
    if (y === undefined) return 1;
    if (typeof x === 'number' && typeof y === 'number') {
      if (x !== y) return x - y;
    } else {
      const c = String(x).localeCompare(String(y));
      if (c !== 0) return c;
    }
  }
  return 0;
}

cpuPresets.sort(cmp);
gpuPresets.sort(cmp);

// Strip internal sort metadata before writing
function stripMeta(arr) {
  return arr.map(({ _chip, _gpuCores, ...rest }) => rest);
}

writeFileSync(CPU_OUT, JSON.stringify(stripMeta(cpuPresets), null, 2) + '\n');
console.error(`Wrote ${cpuPresets.length} Apple CPU presets to ${CPU_OUT}`);

writeFileSync(GPU_OUT, JSON.stringify(stripMeta(gpuPresets), null, 2) + '\n');
console.error(`Wrote ${gpuPresets.length} Apple GPU presets to ${GPU_OUT}`);

const byGen = {};
for (const p of cpuPresets) {
  const gen = p._chip.split(' ')[0];
  byGen[gen] = (byGen[gen] || 0) + 1;
}
console.error('By generation:', byGen);
