#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const GPU_CSV = join(ROOT, 'resources', 'intel', 'intel_gpu_specs.csv');
const CPU_CSV = join(ROOT, 'resources', 'intel', 'intel_cpu_specs.csv');
const OUT_PATH = join(ROOT, 'intel-gpu-presets.json');

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

function parseRows(csvPath) {
  const text = readFileSync(csvPath, 'utf8');
  const rows = parseCSV(text);
  const header = rows[0].map(h => h.replace(/^\uFEFF/, ''));
  const COL = {};
  header.forEach((h, i) => { COL[h] = i; });
  const data = [];
  for (let r = 1; r < rows.length; r++) {
    const row = rows[r];
    if (!row || row.length < 5) continue;
    const rec = {};
    for (const [k, i] of Object.entries(COL)) {
      rec[k] = (row[i] || '').trim();
    }
    data.push(rec);
  }
  return data;
}

function parseGHz(s) {
  if (!s) return null;
  const m = s.match(/([\d.]+)\s*GHz/i);
  return m ? parseFloat(m[1]) : null;
}

function parseMHz(s) {
  if (!s) return null;
  const m = s.match(/([\d.]+)\s*MHz/i);
  return m ? parseFloat(m[1]) : null;
}

function parseInt_(s) {
  if (!s) return null;
  const m = s.match(/(\d+)/);
  return m ? parseInt(m[1], 10) : null;
}

function parseGB(s) {
  if (!s) return null;
  const m = s.match(/([\d.]+)\s*GB/i);
  return m ? parseFloat(m[1]) : null;
}

function parseBandwidth(s) {
  if (!s) return null;
  const m = s.match(/([\d.]+)\s*GB\/s/i);
  return m ? parseFloat(m[1]) : null;
}

function parseWatts(s) {
  if (!s) return null;
  const m = s.match(/([\d.]+)\s*W/i);
  return m ? parseFloat(m[1]) : null;
}

function parseYear(s) {
  if (!s) return null;
  const m = s.match(/(?:Q\d')?(\d{2})(?!\d)/);
  if (m) {
    const y = parseInt(m[1], 10);
    return y >= 90 ? 1900 + y : 2000 + y;
  }
  const m2 = s.match(/(\d{4})/);
  return m2 ? parseInt(m2[1], 10) : null;
}

function round(n, d) {
  const m = Math.pow(10, d);
  return Math.round(n * m) / m;
}

function slug(name) {
  return 'intel-' + name
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

function cleanGpuName(raw) {
  return raw
    .replace(/^Intel®\s*/, '')
    .replace(/\u2122/g, '')
    .replace(/\u00AE/g, '')
    .replace(/\s+Graphics\s*$/i, '')
    .replace(/\s+/g, ' ')
    .trim();
}

const out = [];
const seen = new Set();

const ARC_A_FALLBACKS = {
  'A770':  { vramGB: 16, memType: 'GDDR6', tdpW: 225 },
  'A770M': { vramGB: 16, memType: 'GDDR6', tdpW: 120 },
  'A750':  { vramGB: 8,  memType: 'GDDR6', tdpW: 225 },
  'A750E': { vramGB: 8,  memType: 'GDDR6', tdpW: 225 },
  'A730M': { vramGB: 12, memType: 'GDDR6', tdpW: 80 },
  'A580':  { vramGB: 8,  memType: 'GDDR6', tdpW: 185 },
  'A580E': { vramGB: 8,  memType: 'GDDR6', tdpW: 185 },
  'A570M': { vramGB: 8,  memType: 'GDDR6', tdpW: 75 },
  'A550M': { vramGB: 8,  memType: 'GDDR6', tdpW: 60 },
  'A530M': { vramGB: 8,  memType: 'GDDR6', tdpW: 65 },
  'A380':  { vramGB: 6,  memType: 'GDDR6', tdpW: 75 },
  'A380E': { vramGB: 6,  memType: 'GDDR6', tdpW: 75 },
  'A370E': { vramGB: 4,  memType: 'GDDR6', tdpW: 35 },
  'A370M': { vramGB: 4,  memType: 'GDDR6', tdpW: 35 },
  'A350E': { vramGB: 4,  memType: 'GDDR6', tdpW: 25 },
  'A350M': { vramGB: 4,  memType: 'GDDR6', tdpW: 25 },
  'A310':  { vramGB: 4,  memType: 'GDDR6', tdpW: 75 },
  'A310E': { vramGB: 4,  memType: 'GDDR6', tdpW: 75 },
  'Pro A60':  { vramGB: 12, memType: 'GDDR6', tdpW: 130 },
  'Pro A60M': { vramGB: 12, memType: 'GDDR6', tdpW: 35 },
  'Pro A50':  { vramGB: 6,  memType: 'GDDR6', tdpW: 75 },
  'Pro A40':  { vramGB: 6,  memType: 'GDDR6', tdpW: 50 },
  'Pro A30M': { vramGB: 4,  memType: 'GDDR6', tdpW: 25 },
};

// ── 1. Discrete GPUs from intel_gpu_specs.csv ──
{
  const rows = parseRows(GPU_CSV);
  for (const r of rows) {
    const rawName = r['GPU Name'];
    if (!rawName) continue;
    const name = cleanGpuName(rawName);
    const arch = r['Microarchitecture'] || '';
    const xveStr = r['Intel® Xe Matrix Extensions (Intel® XMX) Engines']
                || r['Xe Vector Engines'] || '';
    const xve = parseInt_(xveStr);
    const clockMHz = parseMHz(r['Graphics Clock']);
    let vramGB = parseGB(r['Memory']);
    const memBw = parseBandwidth(r['Graphics Memory Bandwidth']);
    let tdp = parseWatts(r['TBP']);
    const year = parseYear(r['Launch Date']);
    const segment = r['Vertical Segment'] || '';

    if (arch === 'Xe-HPC') continue;
    if (!clockMHz || !xve || !memBw) continue;

    const clockGHz = clockMHz / 1000;
    const fp32PerXvePerClock = arch === 'Xe2' ? 32 : 16;
    const fp32 = xve * fp32PerXvePerClock * clockGHz / 1000;
    const fp16 = fp32 * 2;

    const memTypeMatch = (r['Memory'] || '').match(/(GDDR\d[X]?)/i);
    let memType = memTypeMatch ? memTypeMatch[1] : null;

    const modelNum = r['Model Number'] || '';
    const fb = ARC_A_FALLBACKS[modelNum];
    if (fb) {
      if (vramGB == null) vramGB = fb.vramGB;
      if (!memType) memType = fb.memType;
      if (tdp == null) tdp = fb.tdpW;
    }

    const id = slug(name);
    if (seen.has(id)) continue;
    seen.add(id);

    const isMobile = segment === 'Mobile';
    const isServer = /Data Center/i.test(rawName);
    const isPro = /Pro/i.test(name);

    const flags = {};
    if (isMobile) flags.mobile = true;
    if (isServer) flags.server = true;

    out.push({
      id,
      vendor: 'Intel',
      name,
      year: year ?? null,
      vramGB: vramGB != null ? round(vramGB, 0) : null,
      memBwGBps: round(memBw, 1),
      fp16Tflops: round(fp16, 2),
      fp32Tflops: round(fp32, 2),
      memType,
      tdpW: tdp,
      tensorCore: false,
      ...flags,
    });
  }
}

// ── 2. Integrated GPUs from intel_cpu_specs.csv ──
{
  const rows = parseRows(CPU_CSV);
  for (const r of rows) {
    const xeCores = parseInt_(r['Xe-cores']);
    if (!xeCores || xeCores <= 0) continue;

    const rawName = r['CPU Name'] || '';
    const procNum = r['Processor Number'] || '';
    if (!rawName || !procNum) continue;

    const gpuClockGHz = parseGHz(r['Graphics Max Dynamic Frequency']);
    if (!gpuClockGHz) continue;

    const fp16 = xeCores * 128 * gpuClockGHz / 1000;

    const memSpec = r['Memory Types'] || '';
    const memChannels = parseInt_(r['Max # of Memory Channels']) || 2;
    const ramBW = computeRamBW(memSpec, memChannels);

    const segment = r['Vertical Segment'] || '';
    const isMobile = /Mobile/i.test(segment);
    const isDesktop = /Desktop/i.test(segment);
    const isEmbedded = /Embedded/i.test(segment);

    const cleanProc = procNum.replace(/[\u00AE\u2122]/g, '').trim();
    const name = `${cleanProc} iGPU (${xeCores} Xe)`;
    const id = slug(name);

    if (seen.has(id)) continue;
    seen.add(id);

    const flags = {};
    if (isMobile || isEmbedded) flags.mobile = true;
    if (isDesktop) flags.desktop = true;

    const iGpuYear = parseYear(r['Launch Date']);

    out.push({
      id,
      vendor: 'Intel',
      name,
      year: iGpuYear,
      vramGB: 32,
      memBwGBps: ramBW ? round(ramBW, 1) : null,
      fp16Tflops: round(fp16, 2),
      fp32Tflops: round(fp16 / 2, 2),
      memType: 'Unified',
      unifiedMemory: true,
      ...flags,
    });
  }
}

function computeRamBW(memSpec, channels) {
  if (!memSpec) return null;
  const mts = parseMTs(memSpec);
  if (mts && channels) return mts * channels * 8 / 1000;

  const lower = memSpec.toLowerCase();
  if (lower.includes('lpddr5x')) return 7500 * channels * 8 / 1000;
  if (lower.includes('lpddr5')) return 6400 * channels * 8 / 1000;
  if (lower.includes('ddr5')) return 4800 * channels * 8 / 1000;
  if (lower.includes('ddr4')) return 3200 * channels * 8 / 1000;
  return null;
}

function parseMTs(s) {
  if (!s) return null;
  const m = s.match(/([\d.]+)\s*MT\/s/i);
  if (m) return parseFloat(m[1]);
  const m2 = s.match(/DDR5[ -](\d+)/i);
  if (m2) return parseInt(m2[1], 10);
  const m3 = s.match(/DDR4[ -](\d+)/i);
  if (m3) return parseInt(m3[1], 10);
  return null;
}

// ── Sort ──
function gpuSortKey(g) {
  if (/^Arc B\d/i.test(g.name)) {
    const m = g.name.match(/B(\d{3})/i);
    return [0, -(m ? parseInt(m[1], 10) : 0), g.mobile ? 1 : 0, g.name];
  }
  if (/^Arc A\d/i.test(g.name)) {
    const m = g.name.match(/A(\d{3})/i);
    return [1, -(m ? parseInt(m[1], 10) : 0), g.mobile ? 1 : 0, g.name];
  }
  if (/Arc Pro B/i.test(g.name)) {
    const m = g.name.match(/B(\d{2})/i);
    return [2, -(m ? parseInt(m[1], 10) : 0), g.name];
  }
  if (/Arc Pro A/i.test(g.name)) {
    const m = g.name.match(/A(\d{2})/i);
    return [3, -(m ? parseInt(m[1], 10) : 0), g.mobile ? 1 : 0, g.name];
  }
  if (/iGPU/i.test(g.name)) {
    return [5, -(g.fp16Tflops || 0), g.name];
  }
  return [9, g.name];
}

function cmp(a, b) {
  const ka = gpuSortKey(a), kb = gpuSortKey(b);
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
out.sort(cmp);

writeFileSync(OUT_PATH, JSON.stringify(out, null, 2) + '\n');
console.error(`Wrote ${out.length} Intel GPUs to ${OUT_PATH}`);
const byFamily = {};
for (const g of out) {
  let fam = 'other';
  if (/^Arc B/i.test(g.name)) fam = 'Arc B';
  else if (/^Arc A/i.test(g.name)) fam = 'Arc A';
  else if (/Arc Pro B/i.test(g.name)) fam = 'Arc Pro B';
  else if (/Arc Pro A/i.test(g.name)) fam = 'Arc Pro A';
  else if (/iGPU/i.test(g.name)) fam = 'iGPU';
  byFamily[fam] = (byFamily[fam] || 0) + 1;
}
console.error('By family:', byFamily);
