// Reads gpu_1986-2026.csv and emits per-vendor GPU preset JSON files used by
// the performance estimator. Also produces a merged gpu-data.json for
// backward compat. Filters to NVIDIA 1000-series+, AMD RX 5000+/MI,
// Intel Arc. AMD entries are built separately from first-party AMD CSVs
// via build-amd-gpu-list.js. Apple M-series entries are built from
// apple_silicon_specs.csv via build-apple-presets.js and merged below.
//
// Run once after updating the CSV: `node scripts/build-gpu-list.js`

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const CSV_PATH = join(ROOT, 'resources', 'gpu_1986-2026.csv');
const OUT_PATH = join(ROOT, 'gpu-data.json');

// ── CSV parser: handles quoted fields with embedded commas/newlines ──
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

// ── Field extractors ──
// Pull the primary numeric value (before a space or "(" annotation).
function parseTFLOPS(s) {
  if (!s) return null;
  const m = s.replace(/,/g, '').match(/([\d.]+)\s*(P|T|G|M|K)?FLOPS/i);
  if (!m) return null;
  const n = parseFloat(m[1]);
  const unit = (m[2] || 'T').toUpperCase();
  const mult = { P: 1e3, T: 1, G: 1e-3, M: 1e-6, K: 1e-9 }[unit] ?? 1;
  return n * mult;  // TFLOPS
}

function parseBandwidth(s) {
  if (!s) return null;
  const m = s.replace(/,/g, '').match(/([\d.]+)\s*(T|G|M|K)?B\/s/);
  if (!m) return null;
  const n = parseFloat(m[1]);
  const unit = (m[2] || 'G').toUpperCase();
  const mult = { T: 1000, G: 1, M: 1e-3, K: 1e-6 }[unit] ?? 1;
  return n * mult;  // GB/s
}

function parseMemSize(s) {
  if (!s) return null;
  const m = s.replace(/,/g, '').match(/([\d.]+)\s*(T|G|M|K)?B/);
  if (!m) return null;
  const n = parseFloat(m[1]);
  const unit = (m[2] || 'G').toUpperCase();
  const mult = { T: 1024, G: 1, M: 1 / 1024, K: 1 / (1024 * 1024) }[unit] ?? 1;
  return n * mult;  // GB
}

function parseTDP(s) {
  if (!s) return null;
  const m = s.match(/([\d.]+)\s*W/);
  return m ? parseFloat(m[1]) : null;
}

function parseYear(s) {
  if (!s) return null;
  const m = s.match(/(\d{4})/);
  return m ? parseInt(m[1], 10) : null;
}

function slug(vendor, name) {
  return `${vendor}-${name}`
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

// ── Vendor-specific eligibility ──
// NVIDIA: GeForce GTX 10-series+ (Pascal, 2016+), RTX 20-50, data-center
// (A/H/B/L/T-series), Quadro/Tesla where the model number indicates Pascal+.
// Skip mobile Max-Q/Mobile variants (different perf envelope).
function acceptNvidia(name, year) {
  if (!name || !year || year < 2016) return false;
  if (/\b(Max-Q|Mobile|Mining)\b/i.test(name)) return false;
  if (/\b(CMP|DRIVE|GRID|PG\d+)\b/i.test(name)) return false;
  if (/GeForce (GTX|RTX) [1-5]\d{3}/i.test(name)) return true;
  if (/\b(A|H|B|L|T)\d{2,4}([A-Z]| |$)/.test(name)) return true;          // A100, H100, L40, T4, B200
  if (/\b(RTX|Quadro) (A|RTX |PRO )/i.test(name) && year >= 2018) return true;
  if (/\bTitan (V|RTX|Xp)/i.test(name)) return true;
  return false;
}

// AMD entries are now built from first-party AMD CSVs via build-amd-gpu-list.js.
// Skip all AMD rows from the generic CSV.
function acceptAmd(name, year) {
  return false;
}

// Intel: Arc A-series / B-series discrete.
function acceptIntel(name) {
  if (!name) return false;
  if (/\bMobile\b/i.test(name)) return false;
  return /Arc [AB]\d{3,4}/i.test(name);
}

// ── Main ──
const text = readFileSync(CSV_PATH, 'utf8');
const rows = parseCSV(text);
const header = rows[0];
const COL = Object.fromEntries(header.map((h, i) => [h, i]));

const need = [
  'Brand', 'Name',
  'Graphics Card__Release Date',
  'Memory__Memory Size', 'Memory__Bandwidth', 'Memory__Memory Type',
  'Board Design__TDP',
  'Theoretical Performance__FP32 (float)',
  'Theoretical Performance__FP16 (half)',
  'Theoretical Performance__BF16',
  'Render Config__Tensor Cores',
];
for (const k of need) {
  if (COL[k] === undefined) throw new Error(`CSV missing column: ${k}`);
}

const out = [];
const seen = new Set();
for (let r = 1; r < rows.length; r++) {
  const row = rows[r];
  if (!row || row.length < 10) continue;
  const brand = row[COL.Brand];
  const name = row[COL.Name];
  if (!brand || !name) continue;
  const year = parseYear(row[COL['Graphics Card__Release Date']]);

  let vendor;
  if (brand === 'NVIDIA' && acceptNvidia(name, year ?? 0)) vendor = 'NVIDIA';
  else if (brand === 'AMD' && acceptAmd(name, year ?? 0)) vendor = 'AMD';
  else if (brand === 'Intel' && acceptIntel(name)) vendor = 'Intel';
  else continue;

  const vramGB = parseMemSize(row[COL['Memory__Memory Size']]);
  const memBwGBps = parseBandwidth(row[COL['Memory__Bandwidth']]);
  const fp32 = parseTFLOPS(row[COL['Theoretical Performance__FP32 (float)']]);
  const fp16Raw = parseTFLOPS(row[COL['Theoretical Performance__FP16 (half)']]);
  const bf16 = parseTFLOPS(row[COL['Theoretical Performance__BF16']]);
  const tdp = parseTDP(row[COL['Board Design__TDP']]);
  const memType = row[COL['Memory__Memory Type']] || null;
  const hasTensor = !!row[COL['Render Config__Tensor Cores']];

  // Effective FP16 rate for LLM inference (tensor-core path when available).
  // On consumer NVIDIA the CSV's FP16 column often reports the 1:64 "shader"
  // rate, which is meaningless for llama.cpp. Prefer BF16 (populated for
  // data-center cards), else FP16 when plausibly tensor-core (≥ 2× FP32),
  // else fall back to FP32 × 2 as an approximation of tensor-core FP16.
  let fp16;
  if (bf16 && bf16 > 0) fp16 = bf16;
  else if (fp16Raw && fp32 && fp16Raw >= fp32 * 1.5) fp16 = fp16Raw;
  else if (fp32) fp16 = fp32 * 2;
  else fp16 = fp16Raw;

  if (!memBwGBps || !fp16) continue;  // unusable for throughput estimate

  const id = slug(vendor, name);
  if (seen.has(id)) continue;
  seen.add(id);

  const flags = {};
  if (vendor === 'NVIDIA') {
    if (/^(A|H|B|L|T|P|V)\d{1,4}/i.test(name) && !/^(RTX|GeForce|Quadro|Titan)/i.test(name) && !/\dM$/i.test(name)) flags.server = true;
    if (/^Tesla\b/i.test(name)) flags.server = true;
    if (/\bServer\b/i.test(name)) flags.server = true;
    if (memType === 'LPDDR5X') flags.mobile = true;
  } else if (vendor === 'Intel') {
    if (/\dM\b/.test(name)) flags.mobile = true;
  }

  out.push({
    id, vendor, name,
    year: year ?? null,
    vramGB: vramGB != null ? round(vramGB, 0) : null,
    memBwGBps: round(memBwGBps, 1),
    fp16Tflops: round(fp16, 2),
    fp32Tflops: fp32 != null ? round(fp32, 2) : null,
    memType,
    tdpW: tdp,
    tensorCore: hasTensor,
    ...flags,
  });
}

function round(n, d) {
  const m = Math.pow(10, d);
  return Math.round(n * m) / m;
}

// Sort so the dropdown reads in the order users expect:
// NVIDIA GeForce: newer series first (50 → 10), within a series top-tier
// first (xx90 → xx50), within a tier Ti SUPER → Ti → SUPER → base → D →
// memory/chip-qualified variants. AMD Radeon RX: same idea with
// XTX → XT → GRE → base. Mobile (M / S suffix) sinks to the bottom of
// its tier. Other product lines fall back to year- or number-based order.
function nvidiaGeForceKey(name) {
  const m = name.match(/GeForce (?:RTX|GTX) (\d)(\d)(\d\d)/i);
  if (!m) return null;
  const series = parseInt(m[1] + m[2], 10);
  const tier = parseInt(m[3], 10);
  return [-series, -tier, ...variantRank(name)];
}
// Returns [primary, secondary] so the sort key keeps memory-qualified and
// mobile variants within their primary group (Ti / SUPER / XT etc.) instead
// of sinking below the base tier.
function variantRank(name) {
  const n = name.toUpperCase();
  const mobile = /\b\d+[MS]\b|MAX-Q|MOBILE/i.test(name);
  const hasTi = /\bTI\b/.test(n);
  const hasSuper = /\bSUPER\b/.test(n);
  const hasXTX = /\bXTX\b/.test(n);
  const hasXT = /\bXT\b/.test(n);
  const hasGRE = /\bGRE\b/.test(n);
  const hasD = /\bD\b(?!DR)/.test(n);
  const memQual = /\b\d+\s*GB\b|\bGDDR\d\b|\b(GA|AD|TU|GP)\d{3}\b|\bOEM\b|\bV2\b|\bTIM\b|\b50TH\b/i.test(name);
  let primary;
  if (hasTi && hasSuper) primary = 0;
  else if (hasTi) primary = 1;
  else if (hasXTX) primary = 0;
  else if (hasXT) primary = 1;
  else if (hasGRE) primary = 2;
  else if (hasSuper) primary = 2;
  else primary = 3;
  if (hasD) primary = Math.max(primary, 3) + 1;  // D variants after their base
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
function sortKey(g) {
  if (g.vendor === 'NVIDIA') {
    const k = nvidiaGeForceKey(g.name);
    if (k) return [0, ...k, g.name];
    if (/Titan/i.test(g.name)) return [1, -(g.year ?? 0), g.name];
    const dc = g.name.match(/^([A-Z])(\d{2,4})/);
    if (dc) {
      const letterRank = { B: 0, H: 1, L: 2, A: 3, T: 4, P: 5, V: 6 }[dc[1]] ?? 9;
      return [2, letterRank, -parseInt(dc[2], 10), g.name];
    }
    return [3, -(g.year ?? 0), g.name];
  }
  if (g.vendor === 'AMD') {
    const k = amdRadeonRxKey(g.name);
    if (k) return [0, ...k, g.name];
    if (/Instinct MI(\d{3})/i.test(g.name)) {
      const n = parseInt(g.name.match(/MI(\d{3})/i)[1], 10);
      return [1, -n, g.name];
    }
    if (/Radeon (PRO|Pro) W(\d{4})/i.test(g.name)) {
      const n = parseInt(g.name.match(/W(\d{4})/i)[1], 10);
      return [2, -n, ...variantRank(g.name), g.name];
    }
    return [3, -(g.year ?? 0), g.name];
  }
  if (g.vendor === 'Intel') {
    const m = g.name.match(/Arc ([AB])(\d{3,4})/i);
    if (m) {
      const famRank = { B: 0, A: 1 }[m[1].toUpperCase()];
      return [0, famRank, -parseInt(m[2], 10), ...variantRank(g.name), g.name];
    }
    return [1, g.name];
  }
  if (g.vendor === 'Apple') {
    const genOrder = { M5: 0, M4: 1, M3: 2, M2: 3, M1: 4 };
    const tierOrder = { Ultra: 0, Max: 1, Pro: 2, base: 3 };
    let gen = 99, tier = 3;
    for (const [k, v] of Object.entries(genOrder)) {
      if (g.name.includes(k)) { gen = v; break; }
    }
    if (/Ultra/i.test(g.name)) tier = 0;
    else if (/Max/i.test(g.name)) tier = 1;
    else if (/Pro/i.test(g.name)) tier = 2;
    else if (/A18/i.test(g.name)) { gen = 5; tier = 4; }
    return [gen, tier, -(g.fp16Tflops || 0), g.name];
  }
  return [9, g.name];
}
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
// ── Merge AMD entries from first-party CSV build ──
const AMD_DATA_PATH = join(ROOT, 'amd-gpu-presets.json');
try {
  const amdData = JSON.parse(readFileSync(AMD_DATA_PATH, 'utf8'));
  for (const g of amdData) out.push(g);
  console.error(`Merged ${amdData.length} AMD GPUs from ${AMD_DATA_PATH}`);
} catch (e) {
  if (e.code !== 'ENOENT') throw e;
  console.error(`Warning: ${AMD_DATA_PATH} not found, skipping AMD merge. Run scripts/build-amd-gpu-list.js first.`);
}

// ── Merge Apple entries from apple_silicon_specs.csv build ──
const APPLE_DATA_PATH = join(ROOT, 'apple-gpu-presets.json');
try {
  const appleData = JSON.parse(readFileSync(APPLE_DATA_PATH, 'utf8'));
  for (const g of appleData) out.push(g);
  console.error(`Merged ${appleData.length} Apple GPUs from ${APPLE_DATA_PATH}`);
} catch (e) {
  if (e.code !== 'ENOENT') throw e;
  console.error(`Warning: ${APPLE_DATA_PATH} not found, skipping Apple merge. Run scripts/build-apple-presets.js first.`);
}

out.sort(cmp);

const byVendor = out.reduce((a, g) => (a[g.vendor] = (a[g.vendor] || 0) + 1, a), {});

const vendorFiles = {
  NVIDIA: join(ROOT, 'nvidia-gpu-presets.json'),
  Intel: join(ROOT, 'intel-gpu-presets.json'),
};
for (const [vendor, path] of Object.entries(vendorFiles)) {
  const entries = out.filter(g => g.vendor === vendor);
  writeFileSync(path, JSON.stringify(entries, null, 2) + '\n');
  console.error(`Wrote ${entries.length} ${vendor} GPUs to ${path}`);
}

writeFileSync(OUT_PATH, JSON.stringify(out, null, 0) + '\n');
console.error(`Wrote ${out.length} total GPUs to ${OUT_PATH}`);
console.error('By vendor:', byVendor);
