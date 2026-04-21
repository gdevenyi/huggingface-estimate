// Reads apple_silicon_macs.csv and emits apple-cpu-presets.json and
// apple-gpu-presets.json. One GPU preset per Mac model/variant row.
// CPU preset is a single "Apple Unified Memory" entry (no CPU offloading on Apple).
//
// Run: `node scripts/build-apple-presets.js`

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const CSV_PATH = join(ROOT, 'resources', 'apple_silicon_macs.csv');
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
      if (c === '"' && field.length === 0) { inQuotes = true; }
      else if (c === '"') { field += c; }
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

function slug(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .replace(/[\s_]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

const TIER_ORDER = { ultra: 0, max: 1, pro: 2 };

function tierRank(chip) {
  const n = chip.toLowerCase();
  for (const [k, v] of Object.entries(TIER_ORDER)) {
    if (n.includes(k)) return v;
  }
  return 3;
}

// ── Parse CSV ──
const text = readFileSync(CSV_PATH, 'utf8');
const rows = parseCSV(text);
const header = rows[0].map(h => h.replace(/^\uFEFF/, ''));
const COL = Object.fromEntries(header.map((h, i) => [h, i]));

const gpuPresets = [];

for (let r = 1; r < rows.length; r++) {
  const row = rows[r];
  if (!row || row.length < 10) continue;

  const machine = (row[COL.Machine] || '').trim();
  const formFactor = (row[COL['Form Factor']] || '').trim();
  const year = parseInt_(row[COL.Year]);
  const chip = (row[COL.Chip] || '').trim();
  const chipVariant = (row[COL['Chip Variant']] || '').trim();
  const COL_MEM_BW = COL['Memory Bandwidth (GB/s)'];
  const COL_MEM_MAX = COL['Unified Memory Max (GB)'];
  const gpuCores = parseInt_(row[COL['GPU Cores']]);
  const memBwGBps = parseFloat_(row[COL_MEM_BW]);
  const maxMemGB = parseInt_(row[COL_MEM_MAX]);
  const fp32Tflops = parseFloat_(row[COL['FP32 TFLOPS']]);
  const fp16Tflops = parseFloat_(row[COL['FP16 TFLOPS']]);

  if (!machine || !memBwGBps) continue;

  const isMobile = formFactor === 'Laptop';
  const isDesktop = formFactor === 'Desktop' || formFactor === 'AIO Desktop';

  const fullName = chipVariant ? `${machine}, ${chipVariant}` : machine;
  const id = slug(fullName);

  const flags = {};
  if (isMobile) flags.mobile = true;
  if (isDesktop) flags.desktop = true;

  gpuPresets.push({
    id,
    name: fullName,
    vendor: 'Apple',
    year: year ?? null,
    vramGB: maxMemGB,
    memBwGBps: round(memBwGBps, 1),
    fp16Tflops: fp16Tflops != null ? round(fp16Tflops, 2) : null,
    fp32Tflops: fp32Tflops != null ? round(fp32Tflops, 2) : null,
    memType: 'Unified',
    ...flags,
    _year: year,
    _chip: chip,
    _gpuCores: gpuCores,
  });
}

gpuPresets.sort((a, b) => {
  if ((b._year ?? 0) !== (a._year ?? 0)) return (b._year ?? 0) - (a._year ?? 0);
  const ta = tierRank(a._chip), tb = tierRank(b._chip);
  if (ta !== tb) return ta - tb;
  if ((b._gpuCores ?? 0) !== (a._gpuCores ?? 0)) return (b._gpuCores ?? 0) - (a._gpuCores ?? 0);
  return a.name.localeCompare(b.name);
});

function stripMeta(arr) {
  return arr.map(({ _year, _chip, _gpuCores, ...rest }) => rest);
}

const cpuPresets = [
  { id: 'apple-unified-memory', name: 'Apple Unified Memory', vendor: 'Apple' },
];

writeFileSync(CPU_OUT, JSON.stringify(cpuPresets, null, 2) + '\n');
console.error(`Wrote ${cpuPresets.length} Apple CPU presets to ${CPU_OUT}`);

writeFileSync(GPU_OUT, JSON.stringify(stripMeta(gpuPresets), null, 2) + '\n');
console.error(`Wrote ${gpuPresets.length} Apple GPU presets to ${GPU_OUT}`);

const byGen = {};
for (const p of gpuPresets) {
  const gen = p._chip.split(' ')[0];
  byGen[gen] = (byGen[gen] || 0) + 1;
}
console.error('By generation:', byGen);
