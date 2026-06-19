#!/usr/bin/env node
// Unit tests for pure functions in calculations.js / parsing.js.
// No network. Run with: node --test test/unit.mjs

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  BPE,
  QUANT_NAMES,
  TQ3_FORK_BPE,
  BUUN_FORK_BPE,
  PRISM_ML_FORK_BPE,
  IK_LLAMA_FORK_BPE,
} from '../quant-types.js';
import {
  globMatch,
  getMeta,
  getModelArch,
  getArchHandler,
  formatBytes,
  formatElements,
  ARCHITECTURES,
  ARCH_ALIASES,
  calcRecurrentState,
  classifyBottleneck,
} from '../calculations.js';
import {
  KV_VALID_QUANTS,
  KV_FORK_GROUPS,
  detectFork,
  applyForkOverrides,
} from '../parsing.js';

describe('BPE registry', () => {
  test('every standard BPE entry is a finite positive number', () => {
    for (const [k, v] of Object.entries(BPE)) {
      assert.ok(typeof v === 'number' && Number.isFinite(v) && v > 0,
        `BPE[${k}] = ${v} is not a positive finite number`);
    }
  });

  test('fork BPE maps only contain keys also present in base BPE or documented as fork-unique', () => {
    // Fork-unique IDs (200, 201, 202) are intentionally not in base BPE if
    // they collide; this test just guards against typos.
    for (const [fork, map] of [['tq3', TQ3_FORK_BPE], ['buun', BUUN_FORK_BPE], ['prism-ml', PRISM_ML_FORK_BPE]]) {
      for (const [k, v] of Object.entries(map)) {
        assert.ok(typeof v === 'number' && Number.isFinite(v) && v > 0,
          `${fork} BPE[${k}] = ${v} invalid`);
      }
    }
  });

  test('PHASE-1-BUG-FIXED: BPE[202] collision resolved via ik_llama fork override', () => {
    // Historically BPE[202] was declared twice (ik_llama Q4_0_R8 line 84 and
    // tq3 TURBO4_0 line 140), and last-wins silently lost the ik_llama value.
    // After the fix: base BPE[202] holds tq3's value (for KV cache use via
    // --kvTypeK=202), and ik_llama tensor sizing uses IK_LLAMA_FORK_BPE[202].
    assert.equal(BPE[202], 68 / 128, 'BPE[202] should hold tq3 value (68/128)');
    assert.equal(IK_LLAMA_FORK_BPE[202], 18 / 32,
      'IK_LLAMA_FORK_BPE[202] must hold ik_llama Q4_0_R8 value (18/32)');
    // Display names should be present for both.
    assert.equal(QUANT_NAMES[202], 'TURBO4_0 (tq3 KV)',
      'base QUANT_NAMES[202] should hold tq3 name');
  });
});

describe('QUANT_NAMES', () => {
  test('every numeric BPE key has a display name', () => {
    // String keys (e.g. 'TURBO3_0') are fork KV cache types whose names live
    // in the fork-specific *_QUANT_NAMES maps, not the global QUANT_NAMES.
    for (const k of Object.keys(BPE)) {
      if (typeof BPE[k] !== 'number') continue;
      const numK = Number(k);
      if (!Number.isFinite(numK)) continue;
      const hasName = QUANT_NAMES[k] !== undefined;
      assert.ok(hasName, `numeric BPE[${k}] has no QUANT_NAMES entry`);
    }
  });
});

describe('KV_VALID_QUANTS / KV_FORK_GROUPS', () => {
  test('every fork-group quant appears in KV_VALID_QUANTS', () => {
    for (const group of KV_FORK_GROUPS) {
      for (const q of group.quants) {
        assert.ok(KV_VALID_QUANTS.includes(q),
          `${group.label}: ${q} missing from KV_VALID_QUANTS`);
      }
    }
  });
});

describe('detectFork', () => {
  // Helper: build minimal {metadata, tensorInfos} shape.
  const model = (ftype, dtypes) => ({
    metadata: { 'general.file_type': ftype },
    tensorInfos: dtypes.map(d => ({ dtype: d })),
  });

  test('PHASE-1: ik_llama detected via dtype 202 (collides with tq3 KV)', () => {
    // A model with dtype-202 weight tensors is ik_llama (Q4_0_R8), not tq3.
    assert.equal(detectFork(model(undefined, [202]).metadata, model(undefined, [202]).tensorInfos), 'ik_llama');
  });

  test('tq3 still detected via its unique signals (dtype 200, ftype 200/45/40)', () => {
    assert.equal(detectFork(model(200, [42]).metadata, model(200, [42]).tensorInfos), 'tq3');
    assert.equal(detectFork(model(45, [42]).metadata, model(45, [42]).tensorInfos), 'tq3');
    assert.equal(detectFork(model(40, [42]).metadata, model(40, [42]).tensorInfos), 'tq3');
  });

  test('tq3 with dtype 202 in tensors still resolves to tq3 (ftype signal fires first)', () => {
    // Edge case: if a model had both tq3 ftype AND dtype 202, tq3 wins.
    assert.equal(detectFork(model(200, [202]).metadata, model(200, [202]).tensorInfos), 'tq3');
  });

  test('prism-ml detected via ftype 28 or ftype 41 + dtype 42', () => {
    assert.equal(detectFork(model(28, [42]).metadata, model(28, [42]).tensorInfos), 'prism-ml');
    assert.equal(detectFork(model(41, [42]).metadata, model(41, [42]).tensorInfos), 'prism-ml');
  });

  test('buun detected via dtype 47 (TURBO8_0)', () => {
    assert.equal(detectFork(model(undefined, [47]).metadata, model(undefined, [47]).tensorInfos), 'buun');
  });

  test('turboquant detected via dtype 45/46', () => {
    assert.equal(detectFork(model(undefined, [45]).metadata, model(undefined, [45]).tensorInfos), 'turboquant');
    assert.equal(detectFork(model(undefined, [46]).metadata, model(undefined, [46]).tensorInfos), 'turboquant');
  });

  test('no fork signal returns null', () => {
    assert.equal(detectFork(model(undefined, [0, 1, 8]).metadata, model(undefined, [0, 1, 8]).tensorInfos), null);
  });
});

describe('applyForkOverrides (Phase 3 FORK_OVERRIDES table)', () => {
  test('tq3 fork: dtype-42 tensor gets Q1_0 BPE and name override', () => {
    const tensors = [{ dtype: 42, shape: [1] }];
    applyForkOverrides(tensors, 'tq3');
    assert.equal(tensors[0]._bpeOverride, 18 / 128);
    assert.equal(tensors[0]._nameOverride, 'Q1_0 (tq3)');
  });

  test('ik_llama fork: dtype-202 tensor gets Q4_0_R8 BPE override', () => {
    const tensors = [{ dtype: 202, shape: [1] }];
    applyForkOverrides(tensors, 'ik_llama');
    assert.equal(tensors[0]._bpeOverride, 18 / 32);
    assert.equal(tensors[0]._nameOverride, 'Q4_0_R8 (ik_llama)');
  });

  test('non-colliding dtypes are left untouched (no override stamping)', () => {
    const tensors = [{ dtype: 0, shape: [1] }, { dtype: 8, shape: [1] }];
    applyForkOverrides(tensors, 'tq3');
    assert.equal(tensors[0]._bpeOverride, undefined);
    assert.equal(tensors[1]._bpeOverride, undefined);
  });

  test('unknown fork is a no-op (returns silently)', () => {
    const tensors = [{ dtype: 42, shape: [1] }];
    applyForkOverrides(tensors, 'nonexistent-fork');
    assert.equal(tensors[0]._bpeOverride, undefined);
  });
});

describe('Architecture registry', () => {
  test('every registry entry has required handler fields', () => {
    for (const [key, entry] of Object.entries(ARCHITECTURES)) {
      assert.ok(Array.isArray(entry.categories), `${key}: missing categories`);
      assert.equal(typeof entry.kvCache, 'function', `${key}: missing kvCache`);
      assert.equal(typeof entry.activations, 'function', `${key}: missing activations`);
    }
  });

  test('PHASE-7: redundant name fields removed; display uses raw arch string', () => {
    // The 143 `name:` fields duplicating registry keys were removed in Phase 7.
    // The 3 entries whose key differs from the canonical llama.cpp arch name
    // (ernie4_5_moe/ernie4_5-moe, hunyuan_moe/hunyuan-moe, lfm2_moe/lfm2moe)
    // display correctly because the UI reads `metadata['general.architecture']`
    // directly (which contains the canonical form), not handler.name.
    // The registry keys are internal-only; ARCH_ALIASES maps the canonical
    // form to the registry key for lookup.
    for (const [key, entry] of Object.entries(ARCHITECTURES)) {
      assert.equal(entry.name, undefined, `${key}: should have no name: field (Phase 7 cleanup)`);
    }
  });

  test('every ARCH_ALIASES target exists in registry', () => {
    for (const [alias, target] of Object.entries(ARCH_ALIASES)) {
      assert.ok(ARCHITECTURES[target], `ARCH_ALIASES[${alias}] -> missing target ${target}`);
    }
  });

  test('getArchHandler resolves aliases and falls back to llama', () => {
    // Phase 7: handler.name removed. Verify alias resolution + fallback by
    // checking categories (a stable field) instead of name.
    assert.equal(getArchHandler('ernie4_5-moe').categories, ARCHITECTURES.ernie4_5_moe.categories);
    assert.equal(getArchHandler('hunyuan-moe').categories, ARCHITECTURES.hunyuan_moe.categories);
    assert.equal(getArchHandler('lfm2moe').categories, ARCHITECTURES.lfm2_moe.categories);
    assert.equal(getArchHandler('totally-unknown-arch').categories, ARCHITECTURES.llama.categories,
      'unknown falls back to llama');
  });
});

describe('Utility exports', () => {
  test('globMatch: literal patterns', () => {
    assert.equal(globMatch('foo', 'foo'), true);
    assert.equal(globMatch('foo', 'bar'), false);
  });

  test('globMatch: wildcard patterns', () => {
    assert.equal(globMatch('*ffn_gate_exps*', 'blk.0.ffn_gate_exps.weight'), true);
    assert.equal(globMatch('*ffn_gate_inp*', 'blk.0.ffn_gate_inp.weight'), true);
    assert.equal(globMatch('*ffn_*_shexp*', 'blk.0.ffn_up_shexp.weight'), true);
    assert.equal(globMatch('*ffn_gate_exps*', 'blk.0.ffn_gate_inp.weight'), false);
  });

  test('getMeta: numeric coercion and fallback', () => {
    const meta = { 'llama.embedding_length': 4096, 'llama.attention.head_count': '8' };
    assert.equal(getMeta(meta, 'llama.embedding_length'), 4096);
    assert.equal(getMeta(meta, 'llama.attention.head_count'), 8);
    assert.equal(getMeta(meta, 'llama.missing_key'), 0);
    assert.equal(getMeta(meta, 'llama.missing_key', 42), 42);
  });

  test('getModelArch: extracts general.architecture', () => {
    assert.equal(getModelArch({ 'general.architecture': 'qwen2' }), 'qwen2');
    assert.equal(getModelArch({}), 'unknown', 'defaults to unknown when missing');
  });

  test('formatBytes: human-readable scaling (1 dp < GiB, 2 dp >= GiB)', () => {
    assert.equal(formatBytes(0), '0.0 KiB');
    assert.equal(formatBytes(1024), '1.0 KiB');
    assert.equal(formatBytes(1024 ** 3), '1.00 GiB');
    assert.equal(formatBytes(1024 ** 4), '1.00 TiB');
  });

  test('formatElements: human-readable scaling', () => {
    assert.equal(formatElements(0), '0');
    assert.equal(formatElements(1000), '1.0K');
    assert.equal(formatElements(1_000_000), '1.00M');
    assert.equal(formatElements(1_000_000_000), '1.00B');
    assert.equal(formatElements(1_000_000_000_000), '1.00T');
  });
});

describe('calcRecurrentState', () => {
  test('returns null for non-recurrent architecture', () => {
    const meta = { 'general.architecture': 'llama', 'llama.block_count': 32 };
    assert.equal(calcRecurrentState(meta), null);
  });

  test('computes SSM state for mamba2', () => {
    // Minimal mamba2 metadata shape.
    const meta = {
      'general.architecture': 'mamba2',
      'mamba2.block_count': 24,
      'mamba2.embedding_length': 768,
      'mamba2.ssm.conv_kernel': 4,
      'mamba2.ssm.inner_size': 1536,
      'mamba2.ssm.state_size': 16,
      'mamba2.ssm.time_step_rank': 48,
    };
    const r = calcRecurrentState(meta);
    assert.notEqual(r, null);
    assert.ok(r.totalBytes > 0);
    assert.ok(r.ssmStateBytes > 0);
    assert.ok(r.convStateBytes > 0);
    assert.equal(r.recurrentLayers, 24);
  });
});

describe('classifyBottleneck (Phase 7 extraction)', () => {
  const base = {
    nGpuLayers: 0, nHybridLayers: 0, nCpuLayers: 0,
    cpuAvailable: false, tDecodeCpu: 0, tDecodeHybridCpu: 0, tDecode: 1,
    gpuBottleneck: null,
  };

  test('all-GPU model returns gpuBottleneck', () => {
    assert.equal(classifyBottleneck({ ...base, nGpuLayers: 32, gpuBottleneck: 'bandwidth' }), 'bandwidth');
  });

  test('all-GPU model without gpuBottleneck returns n/a', () => {
    assert.equal(classifyBottleneck({ ...base, nGpuLayers: 32, gpuBottleneck: null }), 'n/a');
  });

  test('CPU layers present but no CPU available -> cpu-layers-unrun', () => {
    assert.equal(classifyBottleneck({ ...base, nCpuLayers: 5, cpuAvailable: false }), 'cpu-layers-unrun');
  });

  test('CPU decode time > 50% of total -> cpu-dram-spill', () => {
    assert.equal(classifyBottleneck({ ...base, nCpuLayers: 5, cpuAvailable: true, tDecodeCpu: 0.6, tDecode: 1 }), 'cpu-dram-spill');
  });

  test('hybrid CPU experts > 50% of total -> cpu-experts', () => {
    assert.equal(classifyBottleneck({ ...base, nHybridLayers: 5, cpuAvailable: true, tDecodeHybridCpu: 0.7, tDecode: 1 }), 'cpu-experts');
  });

  test('CPU present but not dominant falls back to gpuBottleneck', () => {
    assert.equal(classifyBottleneck({ ...base, nCpuLayers: 2, cpuAvailable: true, tDecodeCpu: 0.1, tDecode: 1, gpuBottleneck: 'compute' }), 'compute');
  });
});
