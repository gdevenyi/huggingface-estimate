import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const OUT_PATH = join(ROOT, 'amd-gpu-presets.json');

const GRAPHICS_CSV = join(ROOT, 'resources', 'amd', 'Graphics Specifications.csv');
const ACCEL_CSV = join(ROOT, 'resources', 'amd', 'Accelerator Specifications.csv');
const PRO_CSV = join(ROOT, 'resources', 'amd', 'Compare AMD Radeon\u2122 PRO GPUs  Specifications  Features.csv');

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
    if (!row || row.length < 3) continue;
    const rec = {};
    for (const [k, i] of Object.entries(COL)) {
      rec[k] = (row[i] || '').trim();
    }
    data.push(rec);
  }
  return data;
}

function parseTFLOPS(s) {
  if (!s) return null;
  s = s.replace(/,/g, '').replace(/\u200B/g, '');
  const m = s.match(/([\d.]+)\s*(P|T|G|M|K)?FLOPS/i);
  if (!m) return null;
  const n = parseFloat(m[1]);
  const unit = (m[2] || 'T').toUpperCase();
  const mult = { P: 1e3, T: 1, G: 1e-3, M: 1e-6, K: 1e-9 }[unit] ?? 1;
  return n * mult;
}

function parseBandwidth(s) {
  if (!s) return null;
  s = s.replace(/,/g, '');
  const m = s.match(/([\d.]+)\s*(T|G|M|K)?B\/s/i);
  if (!m) return null;
  const n = parseFloat(m[1]);
  const unit = (m[2] || 'G').toUpperCase();
  const mult = { T: 1000, G: 1, M: 1e-3, K: 1e-6 }[unit] ?? 1;
  return n * mult;
}

function parseGB(s) {
  if (!s) return null;
  s = s.replace(/,/g, '');
  const m = s.match(/([\d.]+)\s*(T|G|M|K)?B/i);
  if (!m) return null;
  const n = parseFloat(m[1]);
  const unit = (m[2] || 'G').toUpperCase();
  const mult = { T: 1024, G: 1, M: 1 / 1024, K: 1 / (1024 * 1024) }[unit] ?? 1;
  return n * mult;
}

function parseWatts(s) {
  if (!s) return null;
  const m = s.match(/([\d.]+)\s*W/i);
  return m ? parseFloat(m[1]) : null;
}

function parseYear(s) {
  if (!s) return null;
  const m = s.match(/(\d{4})/);
  return m ? parseInt(m[1], 10) : null;
}

function slug(name) {
  return name
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

function round(n, d) {
  const m = Math.pow(10, d);
  return Math.round(n * m) / m;
}

function cleanName(raw) {
  return raw
    .replace(/^AMD\s+/, '')
    .replace(/\u2122/g, '')
    .replace(/\u00AE/g, '')
    .replace(/\u2019/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

const out = [];
const seen = new Set();

// ── 1. Consumer Radeon RX GPUs (Graphics Specifications.csv) ──
{
  const rows = parseRows(GRAPHICS_CSV);
  for (const r of rows) {
    const rawName = r['Name'];
    if (!rawName) continue;
    const name = cleanName(rawName);
    const series = r['Series'] || '';

    const isRadeonRX = /Radeon RX [5-9]\d{3}/i.test(name) || /Radeon RX 9\d{3}/i.test(name);
    const isRadeonVII = /Radeon VII/i.test(name);
    const isVegaDesktop = /Radeon.*RX Vega (64|56)/i.test(name) && /Desktop/i.test(r['Board Type'] || '');
    if (!isRadeonRX && !isRadeonVII && !isVegaDesktop) continue;

    const vramGB = parseGB(r['Max Memory Size']);
    const memBw = parseBandwidth(r['Memory Bandwidth']);
    const fp32 = parseTFLOPS(r['Peak Single Precision (FP32 Vector) Performance']);
    let fp16 = parseTFLOPS(r['Peak Half Precision (FP16 Vector) Performance']);

    // For RDNA3+ (RX 7000/9000), prefer FP16 matrix (WMMA) performance when available.
    // llama.cpp uses WMMA dot2 on RDNA3+ for FP16/BF16.
    const isRDNA3Plus = /RX [789]\d{3}/i.test(name);
    if (isRDNA3Plus) {
      const fp16Matrix = parseTFLOPS(r['Peak Half Precision (FP16 Matrix) Performance']);
      if (fp16Matrix && fp16Matrix > 0) fp16 = fp16Matrix;
    }

    // AMD consumer GPUs: FP16 vector == FP32 for RDNA, and FP16 matrix = 2x FP32
    // If no FP16 found, use 2x FP32 as approximation
    if ((!fp16 || fp16 === 0) && fp32) fp16 = fp32 * 2;

    const tdp = parseWatts(r['Typical Board Power (Desktop)']) || parseWatts(r['GPU Power']);
    const memType = r['Memory Type'] || null;
    const year = parseYear(r['Launch Date']);

    if (!memBw || !fp16 || !vramGB) continue;

    const id = 'amd-' + slug(name);
    if (seen.has(id)) continue;
    seen.add(id);

    const isMobile = /\d[MS]\b/i.test(name) || /\(Mobile\)/i.test(name);

    out.push({
      id, vendor: 'AMD', name,
      year,
      vramGB: round(vramGB, 0),
      memBwGBps: round(memBw, 1),
      fp16Tflops: round(fp16, 2),
      fp32Tflops: fp32 != null ? round(fp32, 2) : null,
      memType,
      tdpW: tdp,
      tensorCore: false,
      ...(isMobile ? { mobile: true } : {}),
    });
  }
}

// ── 2. AMD Instinct Accelerators (Accelerator Specifications.csv) ──
{
  const rows = parseRows(ACCEL_CSV);
  for (const r of rows) {
    const rawName = r['Name'];
    if (!rawName) continue;
    const name = cleanName(rawName);

    const vramGB = parseGB(r['Dedicated Memory Size']);
    const memBw = parseBandwidth(r['Peak Memory Bandwidth']);
    const fp32 = parseTFLOPS(r['Peak Single Precision (FP32) Performance']);
    // For Instinct, FP16 matrix is the relevant metric for LLM inference
    let fp16 = parseTFLOPS(r['Peak Half Precision Matrix (FP16) Performance']);
    // Fallback chain: FP16 matrix > FP16 vector > BF16 matrix > 2x FP32
    if (!fp16 || fp16 === 0) fp16 = parseTFLOPS(r['Peak Half Precision (FP16) Performance']);
    if (!fp16 || fp16 === 0) fp16 = parseTFLOPS(r['Peak bfloat16 Matrix performance']);
    if (!fp16 || fp16 === 0) fp16 = parseTFLOPS(r['Peak bfloat16']);
    if ((!fp16 || fp16 === 0) && fp32) fp16 = fp32 * 2;

    const tdpStr = r['Thermal Design Power (TDP)'] || r['Typical Board Power (TBP)'] || '';
    const tdp = parseWatts(tdpStr);
    const memType = r['Dedicated Memory Type'] || null;
    const year = parseYear(r['Launch Date']);

    // Determine tensor core: CDNA3+ has matrix cores
    const arch = r['GPU Architecture'] || '';
    const hasTensorCore = /^CDNA[345]/.test(arch);

    if (!memBw || !fp16 || !vramGB) continue;

    const id = 'amd-' + slug(name);
    if (seen.has(id)) continue;
    seen.add(id);

    out.push({
      id, vendor: 'AMD', name,
      year,
      vramGB: round(vramGB, 0),
      memBwGBps: round(memBw, 1),
      fp16Tflops: round(fp16, 2),
      fp32Tflops: fp32 != null ? round(fp32, 2) : null,
      memType,
      tdpW: tdp,
      tensorCore: hasTensorCore,
      server: true,
    });
  }
}

// ── 3. Radeon PRO / AI PRO / FirePro / Pro V (Radeon PRO CSV) ──
{
  const rows = parseRows(PRO_CSV);
  for (const r of rows) {
    const rawName = r['Name'];
    if (!rawName) continue;
    const name = cleanName(rawName);

    const vramGB = parseGB(r['Dedicated Memory Size']);
    const memBw = parseBandwidth(r['Peak Memory Bandwidth']);
    const fp32 = parseTFLOPS(r['Peak Single Precision (FP32 Vector) Performance']);
    let fp16 = parseTFLOPS(r['Peak Half Precision (FP16 Vector) Performance']);

    // RDNA3+ PRO cards: use FP16 matrix
    const isRDNA3Plus = /PRO W[789]|AI PRO R9|PRO V[67]/i.test(name);
    if (isRDNA3Plus) {
      const fp16Matrix = parseTFLOPS(r['Peak Half Precision (FP16 Matrix) Performance']);
      if (fp16Matrix && fp16Matrix > 0) fp16 = fp16Matrix;
    }

    if ((!fp16 || fp16 === 0) && fp32) fp16 = fp32 * 2;

    const tdp = parseWatts(r['Total Board Power (TBP)']) || parseWatts(r['TGP']);
    const memType = r['Dedicated Memory Type'] || null;
    const year = parseYear(r['Launch Date']);

    if (!memBw || !fp16 || !vramGB) continue;

    const id = 'amd-' + slug(name);
    if (seen.has(id)) continue;
    seen.add(id);

    const gpuFF = (r['GPU Form Factor'] || '').toLowerCase();
    const isMobile = /\d[MS]\b/i.test(name) || /\(Mobile\)/i.test(name) || /mxm/i.test(gpuFF);

    out.push({
      id, vendor: 'AMD', name,
      year,
      vramGB: round(vramGB, 0),
      memBwGBps: round(memBw, 1),
      fp16Tflops: round(fp16, 2),
      fp32Tflops: fp32 != null ? round(fp32, 2) : null,
      memType,
      tdpW: tdp,
      tensorCore: false,
      ...(isMobile ? { mobile: true } : {}),
    });
  }
}

// ── Sort AMD entries ──
function variantRank(name) {
  const n = name.toUpperCase();
  const mobile = /\b\d+[MS]\b|MAX-Q|MOBILE/i.test(name);
  const hasTi = /\bTI\b/.test(n);
  const hasSuper = /\bSUPER\b/.test(n);
  const hasXTX = /\bXTX\b/.test(n);
  const hasXT = /\bXT\b/.test(n);
  const hasGRE = /\bGRE\b/.test(n);
  const hasD = /\bD\b(?!DR)/.test(n);
  const memQual = /\b\d+\s*GB\b/i.test(name);
  let primary;
  if (hasTi && hasSuper) primary = 0;
  else if (hasTi) primary = 1;
  else if (hasXTX) primary = 0;
  else if (hasXT) primary = 1;
  else if (hasGRE) primary = 2;
  else if (hasSuper) primary = 2;
  else primary = 3;
  if (hasD) primary = Math.max(primary, 3) + 1;
  const secondary = (mobile ? 100 : 0) + (memQual ? 1 : 0);
  return [primary, secondary];
}

function amdRadeonRxKey(name) {
  const m = name.match(/Radeon RX (\d)(\d{3})/i);
  if (!m) return null;
  const series = parseInt(m[1], 10);
  const tier = parseInt(m[2], 10);
  return [-series, -tier, ...variantRank(name)];
}

function amdSortKey(g) {
  const k = amdRadeonRxKey(g.name);
  if (k) return [0, ...k, g.name];

  if (/Instinct MI/i.test(g.name)) {
    const m = g.name.match(/MI(\d{3})/i);
    if (m) return [1, -parseInt(m[1], 10), g.name];
  }

  // AI PRO R9xxx
  if (/AI PRO R9/i.test(g.name)) {
    const m = g.name.match(/R9(\d{3})/i);
    if (m) return [2, 0, -parseInt(m[1], 10), ...variantRank(g.name), g.name];
  }

  // Radeon PRO Wxxxx
  if (/PRO W/i.test(g.name)) {
    const m = g.name.match(/W(\d{4})/i);
    if (m) return [3, -parseInt(m[1], 10), ...variantRank(g.name), g.name];
  }

  // Pro V series
  if (/PRO V/i.test(g.name)) {
    const m = g.name.match(/V(\d{3,4})/i);
    if (m) return [4, -parseInt(m[1], 10), g.name];
  }

  // Pro VII
  if (/Pro VII/i.test(g.name)) return [4, -700, g.name];

  // FirePro
  if (/FirePro/i.test(g.name)) return [5, g.name];

  // Radeon VII, Vega
  if (/Radeon VII/i.test(g.name)) return [6, 0, g.name];
  if (/Vega/i.test(g.name)) return [6, 1, g.name];

  return [9, -(g.year ?? 0), g.name];
}

function cmp(a, b) {
  const ka = amdSortKey(a), kb = amdSortKey(b);
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

writeFileSync(OUT_PATH, JSON.stringify(out, null, 0) + '\n');
console.error(`Wrote ${out.length} AMD GPUs to ${OUT_PATH}`);
const byFamily = {};
for (const g of out) {
  let fam = 'other';
  if (/Radeon RX/i.test(g.name)) fam = 'Radeon RX';
  else if (/Instinct/i.test(g.name)) fam = 'Instinct';
  else if (/PRO W/i.test(g.name)) fam = 'PRO W';
  else if (/AI PRO/i.test(g.name)) fam = 'AI PRO';
  else if (/PRO V/i.test(g.name)) fam = 'PRO V';
  else if (/Pro VII/i.test(g.name)) fam = 'Pro VII';
  else if (/FirePro/i.test(g.name)) fam = 'FirePro';
  else if (/Radeon VII/i.test(g.name)) fam = 'Radeon VII';
  else if (/Vega/i.test(g.name)) fam = 'Vega';
  byFamily[fam] = (byFamily[fam] || 0) + 1;
}
console.error('By family:', byFamily);
