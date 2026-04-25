#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const CPU_CSV = join(ROOT, 'resources', 'intel', 'intel_cpu_specs.csv');
const OUT_PATH = join(ROOT, 'intel-cpu-presets.json');

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

function parseInt_(s) {
  if (!s) return null;
  const m = s.match(/(\d+)/);
  return m ? parseInt(m[1], 10) : null;
}

function round(n, d) {
  const m = Math.pow(10, d);
  return Math.round(n * m) / m;
}

function slug(name) {
  return name
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

function cleanName(raw) {
  return raw
    .replace(/\u2122/g, '')
    .replace(/\u00AE/g, '')
    .replace(/\u2019/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

function detectIsa(rec) {
  const collection = rec['Product Collection'] || '';
  const codename = rec['Code Name'] || '';
  const litho = (rec['CPU Lithography'] || '').toLowerCase();
  const name = (rec['CPU Name'] || '').toLowerCase();

  if (/Xeon.*6 processors/i.test(collection)) return { avx512: true, fp16PerCycle: 32 };
  if (/Xeon.*Scalable/i.test(collection)) return { avx512: true, fp16PerCycle: 32 };
  if (/Xeon.*CPU Max/i.test(collection)) return { avx512: true, fp16PerCycle: 32 };
  if (/Xeon.*W Processor/i.test(collection)) {
    if (/Sapphire Rapids|Ice Lake|Cascade Lake|Skylake/i.test(codename))
      return { avx512: true, fp16PerCycle: 32 };
    return { avx512: false, fp16PerCycle: 16 };
  }
  if (/Xeon.*E Processor/i.test(collection)) {
    if (/Rocket Lake/i.test(codename))
      return { avx512: true, fp16PerCycle: 32 };
    return { avx512: false, fp16PerCycle: 16 };
  }
  if (/Xeon.*D Processor/i.test(collection)) {
    if (/Ice Lake/i.test(codename))
      return { avx512: true, fp16PerCycle: 32 };
    return { avx512: false, fp16PerCycle: 16 };
  }
  if (/Xeon.*E[57]/i.test(collection)) return { avx512: false, fp16PerCycle: 16 };

  return { avx512: false, fp16PerCycle: 16 };
}

function parseMTs(memSpec) {
  if (!memSpec) return null;
  const m = memSpec.match(/([\d.]+)\s*MT\/s/i);
  if (m) return parseFloat(m[1]);
  const m2 = memSpec.match(/DDR5[ -](\d+)/i);
  if (m2) return parseInt(m2[1], 10);
  const m3 = memSpec.match(/DDR4[ -](\d+)/i);
  if (m3) return parseInt(m3[1], 10);
  return null;
}

function computeRamBW(memSpec, channels) {
  if (!memSpec) return null;

  const mts = parseMTs(memSpec);
  if (mts && channels) return round(mts * channels * 8 / 1000, 1);

  const lower = memSpec.toLowerCase();
  if (lower.includes('lpddr5x')) return round(7500 * channels * 8 / 1000, 1);
  if (lower.includes('lpddr5')) return round(6400 * channels * 8 / 1000, 1);
  if (lower.includes('ddr5')) return round(4800 * channels * 8 / 1000, 1);
  if (lower.includes('ddr4')) return round(3200 * channels * 8 / 1000, 1);
  if (lower.includes('ddr3')) return round(1600 * channels * 8 / 1000, 1);
  return null;
}

function isPreAvx2(rec) {
  const collection = rec['Product Collection'] || '';
  const litho = (rec['CPU Lithography'] || '').toLowerCase();
  const codename = (rec['Code Name'] || '').toLowerCase();
  const name = (rec['CPU Name'] || '').toLowerCase();

  if (/32\s*nm/i.test(litho)) return true;
  if (/westmere/i.test(codename)) return true;
  if (/sandy bridge/i.test(codename)) return true;
  if (/Xeon.*E7 Family/i.test(collection) && /ivy bridge/i.test(codename)) return false;
  if (/Xeon.*E[57].*v[12]/i.test(collection) && /sandy bridge/i.test(codename)) return true;

  if (/Xeon.*E7 Family/i.test(collection)) {
    if (/westmere/i.test(codename)) return true;
    return false;
  }
  if (/Xeon.*E[57].*v1/i.test(collection)) return true;
  if (/Xeon.*E[57].*v2/i.test(collection)) return false;

  return false;
}

function archLabel(rec, isa) {
  const collection = rec['Product Collection'] || '';
  const codename = rec['Code Name'] || '';

  if (/Panther Lake/i.test(codename)) return 'Panther Lake';
  if (/Arrow Lake/i.test(codename)) return 'Arrow Lake';
  if (/Lunar Lake/i.test(codename)) return 'Lunar Lake';
  if (/Meteor Lake/i.test(codename)) return 'Meteor Lake';
  if (/Raptor Lake/i.test(codename)) return 'Raptor Lake';
  if (/Alder Lake/i.test(codename)) return 'Alder Lake';
  if (/Rocket Lake/i.test(codename)) return 'Rocket Lake';
  if (/Comet Lake/i.test(codename)) return 'Comet Lake';
  if (/Coffee Lake/i.test(codename)) return 'Coffee Lake';
  if (/Kaby Lake/i.test(codename)) return 'Kaby Lake';
  if (/Skylake/i.test(codename)) return 'Skylake';
  if (/Granite Rapids/i.test(codename)) return 'Granite Rapids';
  if (/Sierra Forest/i.test(codename)) return 'Sierra Forest';
  if (/Emerald Rapids/i.test(codename)) return 'Emerald Rapids';
  if (/Sapphire Rapids/i.test(codename)) return 'Sapphire Rapids';
  if (/Ice Lake/i.test(codename)) return 'Ice Lake';
  if (/Cooper Lake/i.test(codename)) return 'Cooper Lake';
  if (/Cascade Lake/i.test(codename)) return 'Cascade Lake';
  if (/Broadwell/i.test(codename)) return 'Broadwell';
  if (/Haswell/i.test(codename)) return 'Haswell';
  if (/Ivy Bridge/i.test(codename)) return 'Ivy Bridge';

  return isa.avx512 ? 'AVX-512' : 'AVX2';
}

function coreLabel(rec) {
  const totalCores = parseInt_(rec['Total Cores']);
  const pCores = parseInt_(rec['# of Performance-cores']);
  const eCores = parseInt_(rec['# of Efficient-cores']);
  const lpCores = parseInt_(rec['# of Low Power Efficient-cores']);

  const parts = [];
  if (pCores != null && pCores > 0) parts.push(`${pCores}P`);
  if (eCores != null && eCores > 0) parts.push(`${eCores}E`);
  if (lpCores != null && lpCores > 0) parts.push(`${lpCores}LPE`);

  if (parts.length > 0 && parts.join('+') !== `${totalCores}`) {
    return `${totalCores}C (${parts.join('+')})`;
  }
  return `${totalCores}C`;
}

const out = [];
const seen = new Set();

{
  const rows = parseRows(CPU_CSV);
  for (const r of rows) {
    const rawName = r['CPU Name'] || '';
    if (!rawName) continue;

    if (isPreAvx2(r)) continue;

    const totalCores = parseInt_(r['Total Cores']);
    const maxTurbo = parseGHz(r['Max Turbo Frequency']);
    if (!totalCores || !maxTurbo) continue;

    const isa = detectIsa(r);
    const fp16Tflops = round(totalCores * maxTurbo * isa.fp16PerCycle / 1000, 2);

    const memSpec = r['Memory Types'] || '';
    const memChannels = parseInt_(r['Max # of Memory Channels']) || 2;
    const ramBW = computeRamBW(memSpec, memChannels);

    const clean = cleanName(rawName);
    const label = `${clean} (${archLabel(r, isa)}, ${coreLabel(r)})`;
    const procNum = (r['Processor Number'] || '').replace(/[\u00AE\u2122]/g, '').trim();
    const id = slug(procNum || clean);

    if (seen.has(id)) continue;
    seen.add(id);

    const segment = r['Vertical Segment'] || '';
    const flags = {};
    if (/Server/i.test(segment) || /Xeon/i.test(clean)) flags.server = true;
    else if (/Mobile/i.test(segment)) flags.mobile = true;
    if (/Desktop/i.test(segment) && !flags.server) flags.desktop = true;
    if (/Embedded/i.test(segment)) flags.mobile = true;

    out.push({
      id,
      name: label,
      vendor: 'Intel',
      fp16Tflops,
      defaultRamBwGBps: ramBW,
      ...flags,
    });
  }
}

// ── Sort ──
function cpuSortKey(p) {
  const n = p.name.toLowerCase();
  if (/xeon 6/i.test(n)) return [0, 0, -(parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0') || 0), p.name];
  if (/xeon.*scalable/i.test(n)) {
    const genMatch = p.name.match(/(\d)(?:st|nd|rd|th)\s*Gen/i);
    const gen = genMatch ? parseInt(genMatch[1], 10) : 1;
    return [1, -gen, -(parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0') || 0), p.name];
  }
  if (/xeon.*cpu max/i.test(n)) return [2, 0, -(parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0') || 0), p.name];
  if (/xeon w/i.test(n)) return [3, 0, -(parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0') || 0), p.name];
  if (/xeon e\b/i.test(n)) return [4, 0, -(parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0') || 0), p.name];
  if (/xeon d/i.test(n)) return [5, 0, -(parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0') || 0), p.name];
  if (/xeon.*e[57]/i.test(n)) return [6, 0, -(parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0') || 0), p.name];
  if (/core ultra.*series 3/i.test(n) && !/mobile/i.test(n))
    return [10, -(parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0') || 0), p.name];
  if (/core ultra/i.test(n) && !/mobile/i.test(n))
    return [11, -(parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0') || 0), p.name];
  if (/core i/i.test(n) && !/mobile/i.test(n))
    return [12, -(parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0') || 0), p.name];
  if (/core ultra/i.test(n))
    return [20, -(parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0') || 0), p.name];
  if (/core i/i.test(n))
    return [21, -(parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0') || 0), p.name];
  return [99, p.name];
}

function cmp(a, b) {
  const ka = cpuSortKey(a), kb = cpuSortKey(b);
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

writeFileSync(OUT_PATH, JSON.stringify([
  { id: 'intel-unified-memory', name: 'Intel Unified Memory', vendor: 'Intel' },
  ...out,
], null, 2) + '\n');
console.error(`Wrote ${out.length + 1} Intel CPU presets to ${OUT_PATH}`);
const byGroup = {};
for (const p of out) {
  let g = 'other';
  if (/xeon 6/i.test(p.name)) g = 'Xeon 6';
  else if (/xeon.*scalable/i.test(p.name)) g = 'Xeon Scalable';
  else if (/xeon.*cpu max/i.test(p.name)) g = 'Xeon Max';
  else if (/xeon w/i.test(p.name)) g = 'Xeon W';
  else if (/xeon e\b/i.test(p.name)) g = 'Xeon E';
  else if (/xeon d/i.test(p.name)) g = 'Xeon D';
  else if (/xeon.*e[57]/i.test(p.name)) g = 'Xeon E5/E7';
  else if (/core ultra/i.test(p.name)) g = 'Core Ultra';
  else if (/core i/i.test(p.name)) g = 'Core i';
  byGroup[g] = (byGroup[g] || 0) + 1;
}
console.error('By group:', byGroup);
