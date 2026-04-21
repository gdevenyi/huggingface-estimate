import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const DESKTOP_CSV = join(ROOT, 'resources', 'amd', 'Processor Specifications.csv');
const SERVER_CSV = join(ROOT, 'resources', 'amd', 'Server Processor Specifications.csv');
const OUT_PATH = join(ROOT, 'amd-cpu-presets.json');

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

function parseMTs(s) {
  if (!s) return null;
  const m = s.match(/([\d.]+)\s*MT\/s/i);
  return m ? parseFloat(m[1]) : null;
}

function parseGBps(s) {
  if (!s) return null;
  s = s.replace(/,/g, '');
  const m = s.match(/([\d.]+)\s*GB\/s/i);
  return m ? parseFloat(m[1]) : null;
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
    .replace(/^AMD\s+/, '')
    .replace(/\u2122/g, '')
    .replace(/\u00AE/g, '')
    .replace(/\u2019/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

function round(n, d) {
  const m = Math.pow(10, d);
  return Math.round(n * m) / m;
}

// ── Zen detection ──
// FP16 FLOPs per cycle per core:
//   Zen 5: AVX-512 = 32
//   Zen 4: AVX-512 = 32
//   Zen 3: AVX2    = 16
//   Zen 2: AVX2    = 16
//   Zen 1/+: AVX2 half-rate (128-bit FPU, 256-bit cracked) = 8
function detectZen(rec, isServer) {
  const name = rec['Name'] || '';
  const series = rec['Series'] || '';
  const tech = (rec['Processor Technology for CPU Cores'] || '').toLowerCase();
  const cpuType = (rec['CPU Type'] || '').toLowerCase();

  // Explicit CPU type field (Embedded CSV)
  if (cpuType === 'zen 5' || cpuType === 'zen 5c') return { zen: 5, flopsPerCycle: 32 };
  if (cpuType === 'zen 4' || cpuType === 'zen 4c') return { zen: 4, flopsPerCycle: 32 };
  if (cpuType === 'zen 3') return { zen: 3, flopsPerCycle: 16 };
  if (cpuType === 'zen 2') return { zen: 2, flopsPerCycle: 16 };
  if (cpuType === 'zen+' || cpuType === 'zen 1+') return { zen: 1, flopsPerCycle: 8 };
  if (cpuType === 'zen' || cpuType === 'zen 1') return { zen: 1, flopsPerCycle: 8 };

  // Server EPYC: series-based detection
  if (isServer) {
    if (/9005/i.test(series)) {
      return { zen: 5, flopsPerCycle: 32 };
    }
    if (/9004/i.test(series)) return { zen: 4, flopsPerCycle: 32 };
    if (/8004/i.test(series)) return { zen: 4, flopsPerCycle: 32 };  // Zen 4c but same ISA
    if (/7003/i.test(series)) return { zen: 3, flopsPerCycle: 16 };
    if (/7002/i.test(series)) return { zen: 2, flopsPerCycle: 16 };
    if (/7001/i.test(series)) return { zen: 1, flopsPerCycle: 8 };
    if (/4005/i.test(series)) return { zen: 5, flopsPerCycle: 32 };
    if (/4004/i.test(series)) return { zen: 4, flopsPerCycle: 32 };
  }

  // Desktop/mobile Ryzen: name-pattern detection
  const n = name.toLowerCase();
  // Ryzen 9000 series = Zen 5
  if (/ryzen[\s"]*9 \d{4}x?3?d?2?\b/i.test(name) && /99\d\d|98\d\d|97\d\d|96\d\d|95\d\d/.test(name)) return { zen: 5, flopsPerCycle: 32 };
  if (/ryzen.*9 \d{4}x/i.test(name) && name.match(/\d{4}/)) {
    const num = parseInt(name.match(/\d{4}/)[0]);
    if (num >= 9000) return { zen: 5, flopsPerCycle: 32 };
    if (num >= 7000) return { zen: 4, flopsPerCycle: 32 };
    if (num >= 5000) return { zen: 3, flopsPerCycle: 16 };
    if (num >= 3000) return { zen: 2, flopsPerCycle: 16 };
  }

  // Series-based
  if (/ryzen 9000/i.test(series) || /ryzen 9000/i.test(name)) return { zen: 5, flopsPerCycle: 32 };
  if (/ryzen.*9000/i.test(name)) return { zen: 5, flopsPerCycle: 32 };
  if (/ryzen embedded 9000/i.test(name) || /ryzen embedded r7000/i.test(name)) return { zen: 4, flopsPerCycle: 32 };

  // Ryzen AI 400 = Zen 5
  if (/ryzen ai.*400/i.test(series) || /ryzen ai.*4\d{2}/i.test(name)) return { zen: 5, flopsPerCycle: 32 };
  // Ryzen AI 300 = Zen 5 (actually Zen 5 for AI 300 series)
  if (/ryzen ai.*300/i.test(series) || /ryzen ai.*3\d{2}/i.test(name)) return { zen: 5, flopsPerCycle: 32 };
  // Ryzen AI Max = Zen 5
  if (/ryzen ai max/i.test(series) || /ryzen ai max/i.test(name)) return { zen: 5, flopsPerCycle: 32 };

  // Ryzen 200 (mobile) = Zen 4 (actually Zen 4 for FP7/FP7r2)
  if (/ryzen.*200 series/i.test(series)) return { zen: 4, flopsPerCycle: 32 };
  // Ryzen PRO 8000 = Zen 4
  if (/ryzen pro.*8000/i.test(series) || /pro.*8\d{3}/i.test(name)) return { zen: 4, flopsPerCycle: 32 };
  // Ryzen Embedded 8000 = Zen 4
  if (/ryzen embedded.*8000/i.test(series)) return { zen: 4, flopsPerCycle: 32 };
  // Ryzen Z1/Z2 (check Series too since name may be "Ryzen AI Z2 Extreme")
  if (/ryzen (ai )?z/i.test(name) || /ryzen z[12]/i.test(series)) return { zen: 4, flopsPerCycle: 32 };
  // Ryzen Embedded V3000 = Zen 3
  if (/ryzen embedded v3/i.test(name)) return { zen: 3, flopsPerCycle: 16 };
  // Ryzen PRO 9000 desktop = Zen 5
  if (/ryzen pro.*9000/i.test(series)) return { zen: 5, flopsPerCycle: 32 };
  // Ryzen PRO 200 mobile = Zen 4
  if (/ryzen pro.*200/i.test(series)) return { zen: 4, flopsPerCycle: 32 };
  // Ryzen 6000 Series (Rembrandt) = Zen 3+
  if (/ryzen.*6000/i.test(series)) return { zen: 3, flopsPerCycle: 16 };
  // Ryzen 7035 Series (Rembrandt-R) = Zen 3+
  if (/ryzen.*7035/i.test(series)) return { zen: 3, flopsPerCycle: 16 };
  // Ryzen 7020 Series (Mendocino) = Zen 2
  if (/ryzen.*7020/i.test(series)) return { zen: 2, flopsPerCycle: 16 };
  // Ryzen 100 Series / Athlon 10/100 (Mendocino/Rembrandt derivative) = Zen 2
  if (/ryzen.*100 series/i.test(series) || /athlon.*(10|100)/i.test(series)) return { zen: 2, flopsPerCycle: 16 };

  // Technology-based fallback
  if (/(?:^|\s)4nm/i.test(tech)) return { zen: 5, flopsPerCycle: 32 };
  if (/(?:^|\s)5nm/i.test(tech)) return { zen: 4, flopsPerCycle: 32 };
  if (tech.includes('6nm') && !isServer) {
    // 6nm desktop/laptop parts are Zen 3+ (Barcelo-R) or Zen 4 (Phoenix/Hawk Point).
    // Phoenix/Hawk Point have model numbers 7x4x or 8x4x.
    if (/\d{4}/.test(name)) {
      const num = parseInt(name.match(/\d{4}/)[0]);
      if (num >= 8500 || (num >= 7400 && num < 7500)) return { zen: 4, flopsPerCycle: 32 };
    }
    return { zen: 3, flopsPerCycle: 16 };
  }
  if (tech.includes('7nm')) {
    if (/ryzen.*[579].*5\d{3}/i.test(name) || /ryzen.*5\d{3}/i.test(name)) return { zen: 3, flopsPerCycle: 16 };
    return { zen: 2, flopsPerCycle: 16 };
  }
  if (tech.includes('14nm') || tech.includes('12nm') || tech.includes('28nm')) return { zen: 1, flopsPerCycle: 8 };

  // Final fallback by name pattern
  if (/\d{4}/.test(name)) {
    const num = parseInt(name.match(/\d{4}/)[0]);
    if (num >= 9000) return { zen: 5, flopsPerCycle: 32 };
    if (num >= 7000) return { zen: 4, flopsPerCycle: 32 };
    if (num >= 5000) return { zen: 3, flopsPerCycle: 16 };
    if (num >= 3000) return { zen: 2, flopsPerCycle: 16 };
  }

  return { zen: 1, flopsPerCycle: 8 };
}

function parseTDP(rec) {
  const s = rec['Default TDP'] || '';
  const m = s.match(/([\d.]+)\s*W/i);
  return m ? parseFloat(m[1]) : null;
}

function computeRamBW(rec, isServer) {
  if (isServer) {
    const directBW = parseGBps(rec['Per Socket Mem BW']);
    if (directBW) return directBW;
  }

  const memSpec = rec['System Memory Specification'] || '';
  let memChannels = parseInt_(rec['Memory Channels']);
  const memType = (rec['System Memory Type'] || '').toLowerCase();

  // Extract bus width from memType (e.g., "256-bit LPDDR5x" → compute BW directly)
  const busMatch = memType.match(/(\d+)-bit/i);
  if (busMatch) {
    const busBytes = parseInt(busMatch[1], 10) / 8;
    const mts = parseMTs(memSpec);
    if (mts) return round(mts * busBytes / 1000, 1);
    if (memType.includes('lpddr5x')) return round(7500 * busBytes / 1000, 1);
    if (memType.includes('lpddr5')) return round(6400 * busBytes / 1000, 1);
    if (memType.includes('ddr5')) return round(4800 * busBytes / 1000, 1);
    if (memType.includes('ddr4')) return round(3200 * busBytes / 1000, 1);
  }

  if (!memChannels && !isServer) {
    const name = (rec['Name'] || '').toLowerCase();
    if (/threadripper pro/i.test(name)) memChannels = 8;
    else if (/threadripper/i.test(name)) memChannels = 4;
    else memChannels = 2;
  }

  if (!memChannels) return null;

  const mts = parseMTs(memSpec);
  if (mts) {
    return round(mts * memChannels * 8 / 1000, 1);
  }

  // Check LPDDR before DDR (LPDDR strings contain "ddr5"/"ddr4")
  if (memType.includes('lpddr5x')) return round(7500 * memChannels * 8 / 1000, 1);
  if (memType.includes('lpddr5')) return round(6400 * memChannels * 8 / 1000, 1);
  if (memType.includes('lpddr4')) return round(4266 * memChannels * 8 / 1000, 1);
  if (memType.includes('ddr5')) {
    if (memChannels <= 2) return round(5200 * memChannels * 8 / 1000, 1);
    return round(4800 * memChannels * 8 / 1000, 1);
  }
  if (memType.includes('ddr4')) {
    if (memChannels <= 2) return round(3200 * memChannels * 8 / 1000, 1);
    return round(2666 * memChannels * 8 / 1000, 1);
  }

  // Final fallback by product category
  const name = (rec['Name'] || '').replace(/[\u2122\u00AE]/g, '').toLowerCase();
  if (/3\d{3}c|5\d{3}c|7\d{3}c/i.test(name)) {
    return 30;
  }
  if (/ryzen (ai )?z/i.test(name)) {
    return 51;
  }

  return null;
}

function zenLabel(zen) {
  const labels = { 1: 'Zen 1', 2: 'Zen 2', 3: 'Zen 3', 4: 'Zen 4', 5: 'Zen 5' };
  return labels[zen] || 'Zen ?';
}

const out = [];
const seen = new Set();

function addPreset(rec, isServer) {
  const rawName = rec['Name'] || '';
  if (!rawName) return;

  const name = cleanName(rawName);
  const nameLower = name.toLowerCase();

  if (!/^(ryzen|epyc|threadripper|athlon)/i.test(name)) return;
  if (/opteron/i.test(name)) return;
  // Pre-Zen Athlons: Athlon X4 740/750K/760K etc (FM2 socket), no Gold/Silver suffix
  if (/athlon/i.test(name) && !/gold|silver|3000g/i.test(name)) {
    const m = name.match(/\d{3,4}/);
    if (m && parseInt(m[0]) < 3000) return;
  }

  const cores = parseInt_(rec['# of CPU Cores']);
  const maxBoost = parseGHz(rec['Max. Boost Clock']);
  if (!cores || !maxBoost) return;

  const zen = detectZen(rec, isServer);
  const fp16Tflops = round(cores * maxBoost * zen.flopsPerCycle / 1000, 2);
  const ramBW = computeRamBW(rec, isServer);

  const id = slug(name);
  if (seen.has(id)) return;
  // Skip duplicate IDs that differ only by PRO/non-PRO suffix with same core count
  seen.add(id);

  const label = `${name} (${zenLabel(zen.zen)}, ${cores}C)`;

  const formFactor = (rec['Form Factor'] || '').toLowerCase();
  const hasLaptop = /laptop/i.test(formFactor);
  const hasHandheld = /handheld/i.test(formFactor);
  const hasDesktop = /desktop|boxed/i.test(formFactor);

  const flags = {};
  if (isServer || /epyc/i.test(nameLower)) flags.server = true;
  else if (hasLaptop || hasHandheld) flags.mobile = true;
  if (hasDesktop && (hasLaptop || hasHandheld)) flags.desktop = true;

  out.push({
    id, name: label, vendor: 'AMD',
    fp16Tflops,
    defaultRamBwGBps: ramBW,
    ...flags,
  });
}

// ── Parse Desktop/Mobile Processors ──
{
  const rows = parseRows(DESKTOP_CSV);
  for (const r of rows) {
    const formFactor = (r['Form Factor'] || '').toLowerCase();
    const series = (r['Series'] || '').toLowerCase();
    const name = (r['Name'] || '').toLowerCase();

    // Skip embedded-only variants that are better covered by the Embedded CSV
    // But keep all laptop, desktop, and handheld
    // Skip if "Discrete Graphics Card Required" and it's an integrated-only context
    // Actually, include everything

    addPreset(r, false);
  }
}

// ── Parse Server Processors (EPYC) ──
{
  const rows = parseRows(SERVER_CSV);
  for (const r of rows) {
    addPreset(r, true);
  }
}

// ── Sort: Desktop > HEDT/TR > Server, by core count desc within group ──
function sortGroup(p) {
  const n = p.name.toLowerCase();
  if (/epyc/i.test(n)) return [3, -parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0'), p.name];
  if (/threadripper/i.test(n)) return [2, -parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0'), p.name];
  return [1, -parseInt(n.match(/(\d+)\s*c/i)?.[1] || '0'), p.name];
}
out.sort((a, b) => {
  const ga = sortGroup(a), gb = sortGroup(b);
  for (let i = 0; i < Math.max(ga.length, gb.length); i++) {
    const x = ga[i], y = gb[i];
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
});

writeFileSync(OUT_PATH, JSON.stringify(out, null, 2) + '\n');
console.error(`Wrote ${out.length} AMD CPU presets to ${OUT_PATH}`);
const byGroup = {};
for (const p of out) {
  let g = 'other';
  if (/epyc/i.test(p.name)) g = 'EPYC';
  else if (/threadripper/i.test(p.name)) g = 'Threadripper';
  else if (/ryzen ai max/i.test(p.name)) g = 'Ryzen AI Max';
  else if (/ryzen ai/i.test(p.name)) g = 'Ryzen AI';
  else if (/ryzen z/i.test(p.name)) g = 'Ryzen Z';
  else if (/ryzen/i.test(p.name)) g = 'Ryzen';
  else if (/athlon/i.test(p.name)) g = 'Athlon';
  byGroup[g] = (byGroup[g] || 0) + 1;
}
console.error('By group:', byGroup);
