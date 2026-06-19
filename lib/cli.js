// Shared CLI helpers used by run-calc.js and scan-metadata.js.
// Extracted to eliminate ~150 lines of duplicated argv parsing, batch-file
// reading, and HF repo+parse dance across the two CLIs.

import { readFileSync } from 'node:fs';
import { resolveHFModel, parseGGUF, buildResolveUrl } from '../parsing.js';

// ── Batch file (.list) reader ──
// One repo per line, '#' comments and blank lines stripped. Used by both CLIs.
export function readRepoList(path) {
  return readFileSync(path, 'utf-8')
    .split('\n')
    .map(l => l.trim())
    .filter(l => l && !l.startsWith('#'));
}

// ── Uniform success/failure envelope ──
// Both CLIs produce arrays of this shape from batch runs.
export function ok(data)   { return { success: true,  data }; }
export function fail(repo, err) {
  return { success: false, repo, error: err.message ?? String(err) };
}

// ── Concurrency pool ──
// Spawns `concurrency` workers, each pulling from a shared counter.
// Each item's result is wrapped in the uniform envelope above.
export async function parallelMap(items, fn, concurrency = 1) {
  const results = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (next < items.length) {
      const idx = next++;
      try {
        results[idx] = ok(await fn(items[idx], idx));
      } catch (err) {
        results[idx] = fail(items[idx], err);
      }
    }
  });
  await Promise.all(workers);
  return results;
}

// ── Resolve + parse dance ──
// Wraps the "if resolveHFModel returns no .url (multi-file repo), pick the
// first ggufFile and build a resolve URL" recipe. Returns
// { url, metadata, tensorInfos, fork, ggufFiles }.
//
// `firstFilePickMessage` is optional; if supplied, a warning is written to
// stderr when auto-picking the first file (matches run-calc.js's behavior).
export async function resolveAndParse(repo, { firstFilePickMessage } = {}) {
  const result = await resolveHFModel(repo);
  let url = result.url;
  if (!url) {
    const first = result.ggufFiles[0];
    if (firstFilePickMessage) {
      console.error(firstFilePickMessage(repo, first));
    }
    url = buildResolveUrl(repo, first);
  }
  const { metadata, tensorInfos, fork } = await parseGGUF(url);
  return { url, metadata, tensorInfos, fork, ggufFiles: result.ggufFiles };
}
