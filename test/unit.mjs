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
  BEELLAMA_FORK_BPE,
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
  computeOffloadSplit,
  calcActualMemory,
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
    for (const [fork, map] of [['tq3', TQ3_FORK_BPE], ['buun', BUUN_FORK_BPE], ['prism-ml', PRISM_ML_FORK_BPE], ['beellama', BEELLAMA_FORK_BPE]]) {
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
    // Value corrected from 68/128 to 66/128: tq3's TURBO4_USE_4BIT=1 default
    // drops rnorm (66 bytes per QK=128 block, not 68).
    assert.equal(BPE[202], 66 / 128, 'BPE[202] should hold tq3 value (66/128)');
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

  test('tq3 still detected via its unique signals (dtype 200, ftype 200/45, ftype 40+dtype42)', () => {
    assert.equal(detectFork(model(200, [42]).metadata, model(200, [42]).tensorInfos), 'tq3');
    assert.equal(detectFork(model(45, [42]).metadata, model(45, [42]).tensorInfos), 'tq3');
    assert.equal(detectFork(model(40, [42]).metadata, model(40, [42]).tensorInfos), 'tq3');
  });

  test('ftype 40 without dtype 42 does NOT trigger tq3 (upstream MOSTLY_Q1_0 uses dtype 41)', () => {
    assert.equal(detectFork(model(40, [41]).metadata, model(40, [41]).tensorInfos), null);
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

  test('beellama detected via ftype 43+dtype47 (MOSTLY_TQ3_1S) or ftype 44+dtype48 (MOSTLY_TQ4_1S)', () => {
    assert.equal(detectFork(model(43, [47]).metadata, model(43, [47]).tensorInfos), 'beellama');
    assert.equal(detectFork(model(44, [48]).metadata, model(44, [48]).tensorInfos), 'beellama');
    // beellama takes priority over buun when ftype signals are present
    assert.equal(detectFork(model(43, [47, 49]).metadata, model(43, [47, 49]).tensorInfos), 'beellama');
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

  test('beellama fork: dtype-47 tensor gets TQ3_1S BPE override (not buun TURBO8_0)', () => {
    const tensors = [{ dtype: 47, shape: [1] }, { dtype: 48, shape: [1] }, { dtype: 49, shape: [1] }];
    applyForkOverrides(tensors, 'beellama');
    assert.equal(tensors[0]._bpeOverride, 16 / 32);
    assert.equal(tensors[0]._nameOverride, 'TQ3_1S (beellama)');
    assert.equal(tensors[1]._bpeOverride, 20 / 32);
    assert.equal(tensors[1]._nameOverride, 'TQ4_1S (beellama)');
    assert.equal(tensors[2]._bpeOverride, 26 / 32);
    assert.equal(tensors[2]._nameOverride, 'Q6_0 (beellama)');
  });

  test('tq3 BPE values match current ggml-common.h (QK=128, not legacy QK=32)', () => {
    // tq3 updated TURBO3_0/TURBO4_0 to QK=128 (matching turboquant). The old
    // values (14/32, 68/128) are stale; current values are 50/128 and 66/128.
    assert.equal(BPE[201], 50 / 128, 'tq3 TURBO3_0 BPE should be 50/128 (QK=128)');
    assert.equal(BPE[202], 66 / 128, 'tq3 TURBO4_0 BPE should be 66/128 (TURBO4_USE_4BIT=1)');
    // String-keyed TURBO4_0 matches turboquant's default (66, drops rnorm)
    assert.equal(BPE['TURBO4_0'], 66 / 128, 'string-keyed TURBO4_0 should be 66/128');
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

// Build a synthetic per-layer footprint for computeOffloadSplit / calcActualMemory
// tests. nonExpert is split into equal attn/up/gate/down quarters unless given.
function makeFootprint({
  nLayers, nonExpert = 100, expert = 0, kv = 10, inputEmb = 0, output = 50, recurrent = 0,
  attn, up, gate, down,
}) {
  const layers = [];
  for (let i = 0; i < nLayers; i++) {
    const a = attn ?? nonExpert / 4;
    const u = up ?? nonExpert / 4;
    const g = gate ?? nonExpert / 4;
    const d = down ?? nonExpert / 4;
    layers.push({
      bytes: nonExpert + expert, elems: nonExpert + expert,
      nonExpertBytes: nonExpert, nonExpertElems: nonExpert,
      attnBytes: a, attnElems: a,
      upBytes: u, upElems: u,
      gateBytes: g, gateElems: g,
      downBytes: d, downElems: d,
      expertBytesFull: expert, expertElemsFull: expert,
      expertBytesActive: expert, expertElemsActive: expert,
      activeBytes: nonExpert + expert, activeElems: nonExpert + expert,
    });
  }
  return {
    nLayers, layers,
    kvBytesPerLayer: kv, recurrentBytesPerLayer: recurrent,
    outputBytes: output, outputElems: output,
    inputEmbBytes: inputEmb, inputEmbElems: inputEmb,
    mtpBytes: 0, mtpElems: 0,
    hasExperts: expert > 0,
  };
}

describe('computeOffloadSplit (llama.cpp --fit algorithm)', () => {
  test('dense model: full back-to-back fill when VRAM is sufficient', () => {
    const fp = makeFootprint({ nLayers: 4, nonExpert: 100, kv: 10, output: 50 });
    // reserved = output(50); 4 layers * (100+10) = 440; 50+440 = 490 fits.
    const s = computeOffloadSplit({ vramBytes: 500, footprint: fp });
    assert.equal(s.nGpuLayers, 4);
    assert.equal(s.nCpuLayers, 0);
    assert.equal(s.nPartialLayers, 0);
  });

  test('dense model: partial fill leaves low-index layers on CPU', () => {
    const fp = makeFootprint({ nLayers: 4, nonExpert: 100, kv: 10, output: 50 });
    const s = computeOffloadSplit({ vramBytes: 300, footprint: fp });
    assert.equal(s.nGpuLayers, 2); // layers 2,3
    assert.equal(s.nCpuLayers, 2); // layers 0,1
  });

  test('MoE: two-pass fill (hybrid back-to-front, no upgrade when no budget)', () => {
    const fp = makeFootprint({ nLayers: 4, nonExpert: 100, expert: 200, kv: 10, output: 50 });
    // reserved=50; pass1 hybridNeed=110 each. vram=400 -> remaining 350 -> 3 layers fit (330), 4th no.
    const s = computeOffloadSplit({ vramBytes: 400, footprint: fp });
    assert.equal(s.nHybridLayers, 3);
    assert.equal(s.nCpuLayers, 1);
    assert.equal(s.nPartialLayers, 0);
  });

  test('MoE: pass 2 upgrades hybrid->gpu contiguously when budget allows', () => {
    const fp = makeFootprint({ nLayers: 4, nonExpert: 100, expert: 200, kv: 10, output: 50 });
    // enough for all dense-only (440) + 1 full expert (200) = 690 -> use 700
    const s = computeOffloadSplit({ vramBytes: 700, footprint: fp });
    // pass1: all 4 hybrid (440), remaining 210; pass2: layer0 expert 200 fits -> gpu, layer1 200 > 10 break
    assert.equal(s.nGpuLayers, 1);
    assert.equal(s.nHybridLayers, 3);
  });

  test('MoE: pass 3 fits one boundary layer at ATTN fraction', () => {
    const fp = makeFootprint({ nLayers: 4, nonExpert: 100, expert: 200, kv: 10, output: 50 });
    // pass1 3 hybrid (330), remaining 40; attn fraction = 25+10 = 35 fits on layer 0.
    const s = computeOffloadSplit({ vramBytes: 420, footprint: fp });
    assert.equal(s.nPartialLayers, 1);
    assert.equal(s.nCpuLayers, 0);
    assert.ok(s.modes[0].startsWith('partial-'));
  });

  test('cpuMoe forces ngl=-1: all expert layers hybrid, ignores VRAM budget', () => {
    const fp = makeFootprint({ nLayers: 4, nonExpert: 100, expert: 200, kv: 10, output: 50 });
    // tiny VRAM but cpu-moe must still assign all layers (ngl=-1), no partial fill.
    const s = computeOffloadSplit({ vramBytes: 10, footprint: fp, cpuMoe: true });
    assert.equal(s.nHybridLayers, 4);
    assert.equal(s.nCpuLayers, 0);
    assert.equal(s.nPartialLayers, 0);
  });

  test('nCpuMoe forces ngl=-1: first N layers hybrid, rest full gpu', () => {
    const fp = makeFootprint({ nLayers: 4, nonExpert: 100, expert: 200, kv: 10, output: 50 });
    const s = computeOffloadSplit({ vramBytes: 10, footprint: fp, nCpuMoe: 2 });
    assert.equal(s.nHybridLayers, 2); // layers 0,1
    assert.equal(s.nGpuLayers, 2);    // layers 2,3
    assert.equal(s.nCpuLayers, 0);
  });

  test('manual ngl: last N layers offloaded back-to-front', () => {
    const fp = makeFootprint({ nLayers: 4, nonExpert: 100, expert: 200, kv: 10, output: 50 });
    const s = computeOffloadSplit({ vramBytes: 1000, footprint: fp, nLayerOverride: 2 });
    assert.equal(s.nGpuLayers, 2);
    assert.equal(s.nCpuLayers, 2);
    assert.equal(s.auto, false);
  });

  test('token_embd (inputEmb) is excluded from VRAM reserved budget', () => {
    const fp = makeFootprint({ nLayers: 4, nonExpert: 100, kv: 10, output: 50, inputEmb: 20 });
    // If inputEmb were reserved, 50+20+440 = 510 > 500 wouldn't fit all.
    // Excluded: 50 + 440 = 490 <= 500 -> all GPU.
    const s = computeOffloadSplit({ vramBytes: 500, footprint: fp });
    assert.equal(s.nGpuLayers, 4);
  });
});

describe('calcActualMemory (VRAM/RAM accounting)', () => {
  test('token_embd counted as RAM, not VRAM', () => {
    const fp = makeFootprint({ nLayers: 4, nonExpert: 100, kv: 10, output: 50, inputEmb: 20 });
    const m = calcActualMemory({ vramBytes: 1000, footprint: fp });
    // 4 gpu layers = 4*(100+10)=440 VRAM + output 50 = 490; inputEmb 20 in RAM.
    assert.equal(m.actualVram, 490);
    assert.equal(m.actualRam, 20);
  });

  test('cpuMoe ngl=-1: non-expert VRAM, experts RAM, OOM flaggable by caller', () => {
    const fp = makeFootprint({ nLayers: 4, nonExpert: 100, expert: 200, kv: 10, output: 50, inputEmb: 20 });
    const m = calcActualMemory({ vramBytes: 10, footprint: fp, cpuMoe: true });
    // all 4 hybrid: VRAM = 4*(100+10) + 50 = 490; RAM = 4*200 + 20 = 820.
    assert.equal(m.actualVram, 490);
    assert.equal(m.actualRam, 820);
    assert.ok(m.actualVram > 10, 'caller can detect OOM via actualVram > budget');
  });

  test('partial layer: attn fraction in VRAM, down+experts in RAM', () => {
    const fp = makeFootprint({ nLayers: 4, nonExpert: 100, expert: 200, kv: 10, output: 50 });
    const m = calcActualMemory({ vramBytes: 420, footprint: fp });
    // layers 1,2,3 hybrid; layer 0 partial-attn (attn=25, kv=10 -> 35 VRAM).
    // VRAM = 3*(100+10) hybrid + 25 attn + 10 kv + 50 output = 330+35+50 = 415
    assert.equal(m.actualVram, 415);
    // RAM = 3*200 (experts) + (up+gate+down=75) layer0 + 200 expert layer0 = 600+75+200 = 875
    assert.equal(m.actualRam, 875);
    assert.equal(m.nPartialLayers, 1);
  });
});
