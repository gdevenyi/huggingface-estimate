#!/usr/bin/env node
// Snapshot test: re-runs run-calc.js + scan-metadata.js on test/baseline.list
// and diffs JSON output against committed golden files.
//
// Network-required. Skip with SKIP_SNAPSHOTS=1.
//
//   node --test test/snapshot.mjs                 # run via node:test
//   node test/snapshot.mjs                        # run standalone
//   SKIP_SNAPSHOTS=1 node --test test/snapshot.mjs # skip
//
// Snapshot semantics:
//   - run-calc batch JSON must match test/golden/run-calc-batch.json exactly
//     (modulo per-repo success/failure envelope on transient HF failures —
//      we only compare repos that succeeded in BOTH golden and current run).
//   - scan-metadata --json aggregate output is compared structurally
//     (key sets and arch name lists), since per-run flakiness on sharded
//     repos makes byte-equivalence impractical.

import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = dirname(__dirname);
const SKIP = process.env.SKIP_SNAPSHOTS === '1';

function runCli(script, args) {
  return spawnSync('node', [join(ROOT, script), ...args], {
    cwd: ROOT,
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
    timeout: 600000,
  });
}

function loadGolden(name) {
  return JSON.parse(readFileSync(join(__dirname, 'golden', name), 'utf8'));
}

describe('snapshot: run-calc.js --batch baseline.list', { skip: SKIP && 'SKIP_SNAPSHOTS=1' }, () => {
  let result;
  let current;
  let golden;

  test('setup', () => {
    const proc = runCli('run-calc.js', ['--batch', 'test/baseline.list']);
    assert.equal(proc.status, 0, `run-calc exited ${proc.status}\n${proc.stderr}`);
    current = JSON.parse(proc.stdout);
    golden = loadGolden('run-calc-batch.json');
    assert.ok(Array.isArray(current) && Array.isArray(golden), 'batch output should be an array');
    result = { current, golden };
  });

  test('same repos succeed in both runs', () => {
    const currOk = new Set(result.current.filter(r => r.success).map(r => r.data.repo));
    const goldOk = new Set(result.golden.filter(r => r.success).map(r => r.data.repo));
    const intersection = [...currOk].filter(r => goldOk.has(r));
    assert.ok(intersection.length >= 40, `only ${intersection.length} repos succeeded in both runs (network flakiness?)`);
    result.intersection = intersection;
  });

  test('per-repo output byte-equivalent for shared successes', () => {
    const currByRepo = new Map(result.current.filter(r => r.success).map(r => [r.data.repo, r.data]));
    const goldByRepo = new Map(result.golden.filter(r => r.success).map(r => [r.data.repo, r.data]));
    const diffs = [];
    for (const repo of result.intersection) {
      const c = currByRepo.get(repo);
      const g = goldByRepo.get(repo);
      // Normalize volatile fields before comparison.
      const cNorm = JSON.parse(JSON.stringify(c));
      const gNorm = JSON.parse(JSON.stringify(g));
      // url can change if HF reorders files; tolerate but warn.
      try {
        assert.deepEqual(cNorm, gNorm);
      } catch (e) {
        diffs.push({ repo, msg: e.message.slice(0, 200) });
      }
    }
    assert.equal(diffs.length, 0, `${diffs.length} repo(s) changed output:\n${diffs.map(d => `  - ${d.repo}: ${d.msg}`).join('\n')}`);
  });
});

describe('snapshot: scan-metadata.js --json --batch baseline.list', { skip: SKIP && 'SKIP_SNAPSHOTS=1' }, () => {
  test('architecture registry coverage unchanged', () => {
    const proc = runCli('scan-metadata.js', ['--batch', 'test/baseline.list', '--json']);
    assert.equal(proc.status, 1, `scan-metadata exited ${proc.status} (1 = gaps-found is expected)\n${proc.stderr}`);
    const current = JSON.parse(proc.stdout);
    const golden = loadGolden('scan-metadata-batch.json');

    const currArchs = Object.keys(current.architectures || {}).sort();
    const goldArchs = Object.keys(golden.architectures || {}).sort();
    // Allow transient fetch failures (missing archs are tolerated if at least 30 match).
    const shared = currArchs.filter(a => goldArchs.includes(a));
    assert.ok(shared.length >= 30, `only ${shared.length} archs in common (network flakiness?)`);

    // For archs in both, known-flag and repo set must match.
    for (const arch of shared) {
      assert.equal(current.architectures[arch].known, golden.architectures[arch].known,
        `${arch}: known flag changed`);
      const currRepos = current.architectures[arch].repos.slice().sort();
      const goldRepos = golden.architectures[arch].repos.slice().sort();
      assert.deepEqual(currRepos, goldRepos, `${arch}: repo set changed`);
    }
  });
});
