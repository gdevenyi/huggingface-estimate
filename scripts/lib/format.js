// Shared helpers for the build-* preset-generation scripts.
//
// The 6 build-* scripts in scripts/ share a surprising amount of duplicated
// scaffolding (~400 lines total): the `round` function appears 6×, `parseCSV`
// 5×, `parseGHz` 3×, `parseInt_` 3×, and so on. This module centralizes the
// truly-identical helpers so each script can shrink to its unique logic.
//
// Per-vendor helpers (slug, cleanName, computeRamBW, detectZen, etc.) stay in
// their respective scripts because they vary by vendor — see AGENTS.md for the
// rationale and the Phase 10 plan for the broader consolidation.

import { readFileSync } from 'node:fs';

// Round to `d` decimal places. Identical body across all 6 build scripts.
export function round(n, d) {
  const m = Math.pow(10, d);
  return Math.round(n * m) / m;
}

// Build a comparator from a key function returning an array of comparables.
// Numbers compare numerically, everything else via localeCompare; shorter
// keys sort first. Byte-identical to the cmp() bodies previously copy-pasted
// in build-gpu-list, build-amd-gpu-list, build-intel-cpu-presets, and
// build-intel-gpu-presets.
export function makeCmp(keyFn) {
  return (a, b) => {
    const ka = keyFn(a), kb = keyFn(b);
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
  };
}

// Kebab-case ID from a display name. Identical body previously copy-pasted in
// the AMD GPU/CPU and Intel CPU builders; the Intel GPU builder prefixes the
// result with 'intel-'.
export function slug(name) {
  return name
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

// Strip sort-metadata keys (those prefixed with `_`) from output objects.
// Replaces the per-script `stripMeta` functions (e.g. build-apple-presets.js).
export function stripUnderscored(obj) {
  const out = {};
  for (const [k, v] of Object.entries(obj)) {
    if (!k.startsWith('_')) out[k] = v;
  }
  return out;
}

// Generic state-machine CSV parser (handles quoted fields with embedded
// commas/newlines). Byte-identical to the 5 copies across the build scripts.
export function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQuotes = false;
      } else field += c;
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ',') { row.push(field); field = ''; }
      else if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
      else if (c === '\r') { /* skip — handled by \n */ }
      else field += c;
    }
  }
  if (field.length > 0 || row.length > 0) { row.push(field); rows.push(row); }
  return rows;
}

// Build a {colName → index} map from a header row, then return parsed rows as
// keyed objects (filtered to those with at least `minCols` columns).
// Returns just the rows array (the COL map is internal — no external consumer
// in any build script uses it).
export function parseRows(text, { minCols = 5 } = {}) {
  const stripped = text.replace(/^\uFEFF/, '');
  const rows = parseCSV(stripped);
  if (rows.length === 0) return [];
  const header = rows[0].map(h => h.replace(/^\uFEFF/, ''));
  const COL = {};
  header.forEach((h, i) => { COL[h] = i; });
  const out = [];
  for (let r = 1; r < rows.length; r++) {
    const row = rows[r];
    if (!row || row.length < minCols) continue;
    const rec = {};
    for (const [k, i] of Object.entries(COL)) {
      rec[k] = (row[i] || '').trim();
    }
    out.push(rec);
  }
  return out;
}

// Convenience: read CSV from disk and parse. Matches the `parseRows(csvPath)`
// signature used by 4 build scripts (AMD CPU/GPU, Intel CPU/GPU).
export function parseRowsFromPath(csvPath, { minCols = 5 } = {}) {
  const text = readFileSync(csvPath, 'utf8');
  return parseRows(text, { minCols });
}

// Integer extraction: matches the first run of digits in `s`. Identical across
// the 3 Intel/AMD CPU+GPU build scripts.
export function parseInt_(s) {
  if (!s) return null;
  const m = s.match(/(\d+)/);
  return m ? parseInt(m[1], 10) : null;
}

// GHz extraction (e.g. "Up to 4.6 GHz" → 4.6). Identical across 3 scripts.
export function parseGHz(s) {
  if (!s) return null;
  const m = s.match(/([\d.]+)\s*GHz/i);
  return m ? parseFloat(m[1]) : null;
}

// MHz extraction (e.g. "1500 MHz" → 1500). Identical in amd-gpu + intel-gpu.
export function parseMHz(s) {
  if (!s) return null;
  const m = s.match(/([\d.]+)\s*MHz/i);
  return m ? parseFloat(m[1]) : null;
}
