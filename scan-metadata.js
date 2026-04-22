import { resolveHFModel, parseGGUF, buildResolveUrl } from './parsing.js';
import {
  getArchHandler,
  getModelArch,
  ARCHITECTURES,
  ARCH_ALIASES,
  BPE,
  QUANT_NAMES,
  globMatch,
} from './calculations.js';
import { readFileSync } from 'node:fs';

// ── Consumed metadata key suffixes (after stripping arch prefix) ──
// Sourced from calculations.js + ui.js. Split into:
//   CALC_KEYS  — affect memory/performance estimation (exit 1 if unhandled)
//   DISPLAY_KEYS — ui.js only, no calculation impact (informational)

const CALC_KEYS = new Set([
  // Core dimensions (feed into KV cache, activations, MoE, performance)
  'embedding_length',
  'block_count',
  'feed_forward_length',
  // Attention — KV cache sizing
  'attention.head_count',
  'attention.head_count_kv',
  'attention.key_length',
  'attention.value_length',
  'attention.key_length_swa',
  'attention.value_length_swa',
  'attention.sliding_window',
  'attention.sliding_window_pattern',
  // MLA
  'attention.kv_lora_rank',
  'attention.q_lora_rank',
  // RoPE — MLA cache dimension
  'rope.dimension_count',
  // MoE — weight classification, activation sizing
  'expert_count',
  'expert_used_count',
  'expert_feed_forward_length',
  'expert_shared_feed_forward_length',
  'leading_dense_block_count',
  // ISWA / hybrid patterns
  'attention.full_attention_interval',
  'attention.shared_kv_layers',
  'attention.layer_kv_from_start',
  // Next-token prediction layers (reduce KV layers)
  'nextn_predict_layers',
  // Architecture-specific
  'attention.indexer.top_k',
  'altup_num_inputs',
]);

const DISPLAY_KEYS = new Set([
  'attention.key_length_mla',
  'attention.value_length_mla',
  'context_length',
  'vocab_size',
]);

// Standard GGUF metadata keys that are informational only — they describe
// configuration, training details, or tokenizer params but do NOT affect
// memory/performance estimation. Presence of these should NOT trigger exit 1.
const NON_SIZING_KEYS = new Set([
  // RoPE configuration (freq_base/scale don't change tensor/memory sizes)
  'rope.freq_base',
  'rope.freq_scale',
  'rope.dimension_sections',
  // RoPE scaling — changes effective context positioning but not memory sizing;
  // the context length is user-specified (--ctx), not derived from scaling metadata
  'rope.scaling.type',
  'rope.scaling.factor',
  'rope.scaling.original_context_length',
  'rope.scaling.yarn_beta_fast',
  'rope.scaling.yarn_beta_slow',
  'rope.scaling.yarn_log_multiplier',
  'rope.scaling.mscale',
  'rope.freq_base_swa',
  'rope.dimension_count_swa',
  // Normalization epsilon (math constant, not a dimension)
  'attention.layer_norm_rms_epsilon',
  'attention.layer_norm_epsilon',
  // Attention configuration flags
  'attention.causal_mask',
  'attention.diagonal_mask',
  'attention.use_alibi',
  'attention.use_qk_norm',
  // Pooling / head type
  'pooling_type',
  'logit_scale',
  'final_logit_softcapping',
  'attention.logit_softcapping',
  'attention.attention_softmax_floor',
  // Position embedding type
  'position_embedding_type',
  // MoE routing metadata — affect expert selection but not memory sizing.
  // Memory is determined by actual tensor sizes (already captured), not counts.
  'expert_group_count',
  'expert_group_used_count',
  'expert_gating_func',
  'expert_shared_count',
  'expert_weights_scale',
  'expert_weights_norm',
  // Gemma4 per-layer input embedding — value is 0 (unused) in practice
  'embedding_length_per_layer_input',
  // DSA indexer attention configuration — used for attention computation,
  // not for activation sizing (which uses only indexer.top_k)
  'attention.indexer.head_count',
  'attention.indexer.key_length',
  // T5 relative attention bias — bias tensors are already counted as weights
  'attention.relative_buckets_count',
  // SSM/Mamba configuration — these are structural but the current codebase
  // doesn't estimate SSM layer memory. Flagged as informational until support
  // is added. TODO: promote to CALC_KEYS when SSM estimation is implemented.
  'ssm.conv_kernel',
  'ssm.state_size',
  'ssm.group_count',
  'ssm.time_step_rank',
  'ssm.inner_size',
  'ssm.dt_rank',
  // Repeat-alternate for Qwen3.5 MoE-style DeltaNet layers
  'full_attention_interval',
]);

const GENERAL_KEYS = new Set([
  'general.architecture',
  'general.name',
  'general.basename',
]);

const CLIP_KEYS = new Set([
  'clip.has_vision_encoder',
  'clip.has_audio_encoder',
  'clip.projector_type',
  'clip.vision.projector_type',
  'clip.audio.projector_type',
  'clip.vision.image_size',
  'clip.vision.patch_size',
  'clip.vision.embedding_length',
  'clip.vision.block_count',
  'clip.vision.projection_dim',
  'clip.vision.spatial_merge_size',
  'clip.vision.projector.scale_factor',
  'clip.minicpmv_query_num',
  'clip.minicpmv_version',
]);

// Known projector types (from calculations.js KNOWN_PROJ_TYPES)
const KNOWN_PROJ_TYPES = new Set([
  'mlp', 'mlp_norm', 'phi4', 'pixtral', 'lightonocr', 'janus_pro',
  'dots_ocr', 'paddleocr', 'ldp', 'ldpv2', 'adapter', 'resampler',
  'qwen2vl_merger', 'qwen2.5vl_merger', 'qwen3vl_merger', 'glm4v', 'youtuvl',
  'step3vl', 'gemma3', 'gemma4v', 'idefics3', 'internvl', 'nemotron_v2_vl',
  'llama4', 'gemma3nv', 'lfm2', 'kimivl', 'kimik25', 'cogvlm', 'deepseekocr',
  'hunyuanocr', 'yasa2', 'qwen2.5o',
  'ultravox', 'voxtral', 'qwen2a', 'qwen3a', 'glma', 'lfm2a',
  'gemma4a', 'gemma3na', 'meralion', 'musicflamingo',
]);

// ── CLI args ──

function parseArgs(argv) {
  const args = {
    repo: null,
    batch: null,
    json: false,
    summary: false,
    unknownOnly: false,
    concurrency: 3,
  };
  let i = 2;
  const needValue = (flag) => {
    const val = argv[++i];
    if (val === undefined || val.startsWith('-')) {
      console.error(`Error: ${flag} requires a value`);
      process.exit(1);
    }
    return val;
  };
  while (i < argv.length) {
    const arg = argv[i];
    if (arg === '--batch') args.batch = needValue(arg);
    else if (arg === '--json') args.json = true;
    else if (arg === '--summary') args.summary = true;
    else if (arg === '--unknown-only') args.unknownOnly = true;
    else if (arg === '--concurrency') args.concurrency = parseInt(needValue(arg), 10);
    else if (!arg.startsWith('-')) args.repo = arg;
    else { console.error(`Unknown flag: ${arg}`); process.exit(1); }
    i++;
  }
  if (!args.repo && !args.batch) {
    console.error(`Usage: node scan-metadata.js <repo> [options]
       node scan-metadata.js --batch <listfile> [options]

Arguments:
  <repo>                  HuggingFace repo (e.g. bartowski/Llama-3.1-8B-GGUF)
  --batch <file>          Process repos from a file (one per line, # comments)
  --json                  JSON output to stdout
  --summary               Deduplicate by architecture, skip per-repo detail
  --unknown-only          Only report unknown architectures, quants, projector types
  --concurrency N         Parallel fetches (default: 3)

Exit codes:
  0  All clear or informational findings only
  1  Functional gaps found (unknown arch, unknown quant, or unhandled calc key)`);
    process.exit(1);
  }
  return args;
}

// ── Repo scanner ──

async function scanRepo(repo) {
  const resolved = await resolveHFModel(repo);
  let url = resolved.url;
  if (!url) {
    url = buildResolveUrl(repo, resolved.ggufFiles[0]);
  }
  const { metadata, tensorInfos } = await parseGGUF(url);
  const arch = metadata['general.architecture'] || 'unknown';

  const allMetaKeys = Object.keys(metadata);

  const dtypes = new Set();
  for (const t of tensorInfos) dtypes.add(t.dtype);

  const tensorNames = tensorInfos.map(t => t.name);

  const projType = metadata['clip.projector_type']
    || metadata['clip.vision.projector_type']
    || metadata['clip.audio.projector_type']
    || null;

  return { repo, arch, allMetaKeys, dtypes, tensorNames, projType, url };
}

// ── Concurrency limiter ──

async function parallelMap(items, fn, concurrency) {
  const results = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (next < items.length) {
      const idx = next++;
      try {
        results[idx] = { success: true, data: await fn(items[idx], idx) };
      } catch (err) {
        results[idx] = { success: false, repo: items[idx], error: err.message };
      }
    }
  });
  await Promise.all(workers);
  return results;
}

// ── Analysis ──

function classifyKey(key, arch) {
  if (GENERAL_KEYS.has(key)) return 'general';
  if (CLIP_KEYS.has(key)) return 'clip';
  const archPrefix = arch + '.';
  if (key.startsWith(archPrefix)) {
    const suffix = key.slice(archPrefix.length);
    if (CALC_KEYS.has(suffix)) return 'calc';
    if (DISPLAY_KEYS.has(suffix)) return 'display';
    if (NON_SIZING_KEYS.has(suffix)) return 'non-sizing';
    return 'unhandled';
  }
  if (key.startsWith('general.') || key.startsWith('clip.')) return 'ignored';
  if (key.includes('.')) {
    const otherArch = key.split('.')[0];
    if (otherArch !== arch) return 'other-arch';
  }
  return 'ignored';
}

function analyzeResults(results) {
  const archGroups = new Map();
  const unknownQuants = new Map();
  const unknownProjTypes = new Map();

  for (const r of results) {
    if (!r.success) continue;
    const { arch, allMetaKeys, dtypes, tensorNames, projType, repo, url } = r.data;

    if (!archGroups.has(arch)) {
      archGroups.set(arch, {
        repos: [],
        metaKeyPresence: new Map(),
        unhandledCalcKeys: new Map(),
        unhandledDisplayKeys: new Map(),
        unknownDtypes: new Map(),
        unmatchedTensors: new Set(),
      });
    }
    const group = archGroups.get(arch);
    group.repos.push(repo);

    for (const key of allMetaKeys) {
      const cat = classifyKey(key, arch);
      const count = (group.metaKeyPresence.get(key) || 0) + 1;
      group.metaKeyPresence.set(key, count);

      if (cat === 'unhandled') {
        const m = (group.unhandledCalcKeys.get(key) || 0) + 1;
        group.unhandledCalcKeys.set(key, m);
      }
    }

    for (const dt of dtypes) {
      if (BPE[dt] === undefined) {
        const repos = unknownQuants.get(dt) || [];
        if (!repos.includes(repo)) repos.push(repo);
        unknownQuants.set(dt, repos);
        const m = (group.unknownDtypes.get(dt) || 0) + 1;
        group.unknownDtypes.set(dt, m);
      }
    }

    if (projType && !KNOWN_PROJ_TYPES.has(projType.toLowerCase())) {
      const repos = unknownProjTypes.get(projType) || [];
      if (!repos.includes(repo)) repos.push(repo);
      unknownProjTypes.set(projType, repos);
    }
  }

  for (const [arch, group] of archGroups) {
    const handler = getArchHandler(arch);
    const patterns = handler.tensorGroups
      ? [...(handler.tensorGroups.expert || []), ...(handler.tensorGroups.router || []), ...(handler.tensorGroups.shared || [])]
      : [];
    const allTensors = new Map();
    for (const r of results) {
      if (!r.success || r.data.arch !== arch) continue;
      for (const name of r.data.tensorNames) {
        const base = name.replace(/^blk\.\d+\./, 'blk.N.');
        allTensors.set(base, (allTensors.get(base) || 0) + 1);
      }
    }
    const matched = new Set();
    for (const pat of patterns) {
      for (const name of allTensors.keys()) {
        if (globMatch(pat.replace(/blk\.\*\./, 'blk.N.'), name)) matched.add(name);
      }
    }
    for (const name of allTensors.keys()) {
      const norm = name.replace(/^blk\.N\./, '');
      const isGeneric = /^(token_embd|output_norm|output|ffn_|attn_|blk\.N\.attn_|blk\.N\.ffn_)/.test(name);
      if (!matched.has(name) && !isGeneric) {
        group.unmatchedTensors.add(name);
      }
    }
  }

  return { archGroups, unknownQuants, unknownProjTypes };
}

// ── Reporting ──

function fmtCount(n, total) {
  return `${n}/${total}`;
}

function reportHuman(analysis) {
  const { archGroups, unknownQuants, unknownProjTypes } = analysis;
  let hasGaps = false;

  const sortedArchs = [...archGroups.keys()].sort();

  for (const arch of sortedArchs) {
    const group = archGroups.get(arch);
    const handler = getArchHandler(arch);
    const isKnown = !!(ARCHITECTURES[arch] || ARCHITECTURES[ARCH_ALIASES[arch]]);
    const totalRepos = group.repos.length;

    console.log(`\n${'='.repeat(60)}`);
    console.log(`Architecture: ${arch} ${isKnown ? '(known)' : '(UNKNOWN — falls back to llama)'}`);
    console.log(`  Repos: ${group.repos.length === 1 ? group.repos[0] : group.repos.slice(0, 3).join(', ') + (totalRepos > 3 ? ` (+${totalRepos - 3} more)` : '')}`);
    if (!isKnown) hasGaps = true;

    const unhandledCalc = [...group.unhandledCalcKeys.entries()]
      .sort((a, b) => b[1] - a[1]);
    if (unhandledCalc.length > 0) {
      console.log('\n  Unhandled CALCULATION keys (affect memory/performance):');
      for (const [key, count] of unhandledCalc) {
        console.log(`    ${key}  (${fmtCount(count, totalRepos)} repos)`);
      }
      hasGaps = true;
    }

    const nonSizingKeys = [...group.metaKeyPresence.entries()]
      .filter(([key]) => classifyKey(key, arch) === 'non-sizing')
      .sort((a, b) => b[1] - a[1]);
    if (nonSizingKeys.length > 0) {
      console.log('\n  Non-sizing keys (informational, no estimation impact):');
      for (const [key, count] of nonSizingKeys) {
        console.log(`    ${key}  (${fmtCount(count, totalRepos)} repos)`);
      }
    }

    const unhandledDisplay = [...group.metaKeyPresence.entries()]
      .filter(([key]) => classifyKey(key, arch) === 'display')
      .sort((a, b) => b[1] - a[1]);
    if (unhandledDisplay.length > 0) {
      console.log('\n  Display-only keys (informational):');
      for (const [key, count] of unhandledDisplay) {
        console.log(`    ${key}  (${fmtCount(count, totalRepos)} repos)`);
      }
    }

    const ignoredKeys = [...group.metaKeyPresence.entries()]
      .filter(([key]) => classifyKey(key, arch) === 'ignored')
      .sort((a, b) => b[1] - a[1]);
    if (ignoredKeys.length > 0) {
      console.log(`\n  Ignored keys (${ignoredKeys.length} total, e.g. general.*, rope.*):`);
      const sample = ignoredKeys.slice(0, 5);
      for (const [key, count] of sample) {
        console.log(`    ${key}  (${fmtCount(count, totalRepos)} repos)`);
      }
      if (ignoredKeys.length > 5) {
        console.log(`    ... and ${ignoredKeys.length - 5} more`);
      }
    }

    if (group.unknownDtypes.size > 0) {
      console.log('\n  Unknown quantization types (missing BPE):');
      for (const [dt, count] of group.unknownDtypes) {
        const name = QUANT_NAMES[dt] || `type_${dt}`;
        console.log(`    ${name} (dtype=${dt})  (${count} occurrences)`);
      }
      hasGaps = true;
    }

    if (group.unmatchedTensors.size > 0) {
      console.log('\n  Unmatched tensor name patterns:');
      for (const name of [...group.unmatchedTensors].sort()) {
        console.log(`    ${name}`);
      }
    }
  }

  if (unknownQuants.size > 0) {
    console.log(`\n${'='.repeat(60)}`);
    console.log('Unknown quantization types across all repos:');
    for (const [dt, repos] of unknownQuants) {
      const name = QUANT_NAMES[dt] || `type_${dt}`;
      console.log(`  ${name} (dtype=${dt}): ${repos.length} repo(s)`);
    }
    hasGaps = true;
  }

  if (unknownProjTypes.size > 0) {
    console.log(`\n${'='.repeat(60)}`);
    console.log('Unknown projector types:');
    for (const [pt, repos] of unknownProjTypes) {
      console.log(`  "${pt}": ${repos.length} repo(s)`);
    }
  }

  console.log(`\n${'='.repeat(60)}`);
  if (hasGaps) {
    console.log('RESULT: Functional gaps found (exit 1)');
  } else {
    console.log('RESULT: No functional gaps found (exit 0)');
  }

  return hasGaps;
}

function reportJson(analysis) {
  const { archGroups, unknownQuants, unknownProjTypes } = analysis;
  let hasGaps = false;

  const architectures = {};
  for (const [arch, group] of archGroups) {
    const handler = getArchHandler(arch);
    const isKnown = !!(ARCHITECTURES[arch] || ARCHITECTURES[ARCH_ALIASES[arch]]);
    if (!isKnown) hasGaps = true;

    const unhandledCalc = [...group.unhandledCalcKeys.entries()]
      .map(([key, count]) => ({ key, repoCount: count, totalRepos: group.repos.length }));

    const nonSizingKeys = [...group.metaKeyPresence.entries()]
      .filter(([key]) => classifyKey(key, arch) === 'non-sizing')
      .map(([key, count]) => ({ key, repoCount: count, totalRepos: group.repos.length }));

    const unknownDtypes = [...group.unknownDtypes.entries()]
      .map(([dt, count]) => ({ dtype: dt, name: QUANT_NAMES[dt] || `type_${dt}`, occurrences: count }));

    if (unhandledCalc.length > 0 || unknownDtypes.length > 0) hasGaps = true;

    architectures[arch] = {
      known: isKnown,
      repos: group.repos,
      unhandledCalcKeys: unhandledCalc,
      nonSizingKeys,
      unknownDtypes,
      unmatchedTensors: [...group.unmatchedTensors].sort(),
    };
  }

  const quants = {};
  for (const [dt, repos] of unknownQuants) {
    quants[dt] = { name: QUANT_NAMES[dt] || `type_${dt}`, repos };
  }
  if (Object.keys(quants).length > 0) hasGaps = true;

  const projs = {};
  for (const [pt, repos] of unknownProjTypes) {
    projs[pt] = { repos };
  }

  console.log(JSON.stringify({ hasGaps, architectures, unknownQuants: quants, unknownProjTypes: projs }, null, 2));
  return hasGaps;
}

// ── Main ──

async function main() {
  const args = parseArgs(process.argv);

  let repos;
  if (args.batch) {
    repos = readFileSync(args.batch, 'utf-8')
      .split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('#'));
  } else {
    repos = [args.repo];
  }

  process.stderr.write(`Scanning ${repos.length} repo(s) with concurrency ${args.concurrency}...\n`);

  let progress = 0;
  const results = await parallelMap(repos, async (repo) => {
    const result = await scanRepo(repo);
    progress++;
    process.stderr.write(`[${progress}/${repos.length}] ${repo} → ${result.arch}\n`);
    return result;
  }, args.concurrency);

  const errors = results.filter(r => !r.success);
  for (const e of errors) {
    process.stderr.write(`  ERROR: ${e.repo}: ${e.error}\n`);
  }

  const analysis = analyzeResults(results);
  const hasGaps = args.json ? reportJson(analysis) : reportHuman(analysis);
  process.exit(hasGaps ? 1 : 0);
}

main().catch(err => {
  console.error(`Fatal: ${err.message}`);
  process.exit(2);
});
