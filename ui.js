import { GGMLQuantizationType, KV_VALID_QUANTS, KV_FORK_GROUPS, parseGGUF, resolveHFModel, buildResolveUrl, extractHfSlug } from './parsing.js';
import { QUANT_NAMES, getArchHandler, getModelArch, getMeta, calcWeightSize, calcKVCache, calcActivations, calcMoEInfo, calcMmProj, calcPerLayerFootprint, calcMemoryBreakdown, calcActualMemory, estimatePerformance, formatBytes, formatElements } from './calculations.js';
import { mergeCpuPresets, mergeGpuPresets, getCpuPresets, getGpuPresets, findCpuPreset, getSlowestCpuPreset, UNIFIED_MEMORY_CPU_PRESET } from './hardware-presets.js';

const CPU_JSON_FILES = ['intel-cpu-presets.json', 'amd-cpu-presets.json'];
const GPU_JSON_FILES = ['nvidia-gpu-presets.json', 'intel-gpu-presets.json', 'amd-gpu-presets.json', 'apple-gpu-presets.json'];

const _cpuCounter = { n: 0 };
const _gpuCounter = { n: 0 };
let _configLoaded = false;

function tryLoadConfig() {
  if (_configLoaded) return;
  if (_cpuCounter.n >= CPU_JSON_FILES.length && _gpuCounter.n >= GPU_JSON_FILES.length) {
    _configLoaded = true;
    loadConfig();
  }
}

// Load all vendor JSON files for a hardware class (CPU or GPU), then invoke
// the populate callback once everything has settled (success or failure).
// Replaces two near-identical 6-line startup loops that differed only in
// the merge function, the counter, and the populate callback.
function loadPresetFiles(files, { merge, counter, populate }) {
  for (const f of files) {
    fetch('./' + f)
      .then(r => r.ok ? r.json() : [])
      .then(d => { merge(d); })
      .catch(() => { /* tolerate missing vendor file */ })
      .finally(() => {
        counter.n++;
        if (counter.n === files.length) {
          populate();
          tryLoadConfig();
        }
      });
  }
}

const $ = (s) => document.querySelector(s);

// Unit constants (Phase 8 magic-number sweep). Previously 1024 ** 3 was
// inlined 4× and the RAM thresholds 80/100 appeared as bare literals.
const GIB = 1024 ** 3;
const RAM_GREEN_PCT = 80;
const RAM_YELLOW_PCT = 100;

// Sum of mmproj weights + per-image activation bytes. Single source of truth
// used by renderMemoryPanel, renderFitCheck, and the resolveMmProj path.
// Previously this was duplicated in three places (ui.js:726, ui.js:940, plus
// run-calc.js) and could drift.
function totalMmProjBytes(info) {
  if (!info) return 0;
  return info.weightBytes + (info.perImageActBytes || 0);
}

// Tiny DOM helpers used in many places. Extracted to dedupe ~12 sites.
function fillSelectWithOptions(select, values) {
  for (const v of values) {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    select.appendChild(opt);
  }
}

// Apply a saved config value to a <select>, going through SlimSelect when present.
function applySelectValue(el, ss, value) {
  if (value == null) return;
  if (!el.querySelector(`option[value="${value}"]`)) return;
  if (ss) ss.setSelected(value);
  else el.value = value;
}

// Tiny visibility helpers (Phase 11 consolidation: replaces ~16 inline
// `style.display = 'none'|''` assignments). `.hidden` is defined as
// `display: none !important`, so adding it always hides; removing it restores
// the element's CSS default display (block / flex / etc.).
function show(el) { el?.classList.remove('hidden'); }
function hide(el) { el?.classList.add('hidden'); }

if (location.protocol === 'file:') {
  showError('\u26A0 Open via a local server for best results. Run: python3 -m http.server 8000 then visit http://localhost:8000');
}
const hfPathEl = $('#hfPath');
const resolveBtn = $('#resolveBtn');
const modelSelectWrap = $('#modelSelectWrap');
const modelSelect = $('#modelSelect');
const mmProjSelectWrap = $('#mmProjSelectWrap');
const mmProjSelect = $('#mmProjSelect');
const mmProjDeviceWrap = $('#mmProjDeviceWrap');
const mmProjDeviceEl = $('#mmProjDevice');
const contextLenEl = $('#contextLen');
const batchSizeEl = $('#batchSize');
const vramEl = $('#vram');
const ramEl = $('#ram');
const kvTypeKEl = $('#kvTypeK');
const kvTypeVEl = $('#kvTypeV');
const swaFullEl = $('#swaFull');
const gpuPresetEl = $('#gpuPreset');
const gpuFlopsEl = $('#gpuFlops');
const gpuBwEl = $('#gpuBw');
const cpuPresetEl = $('#cpuPreset');
const cpuFlopsEl = $('#cpuFlops');
const ramBwEl = $('#ramBw');
const nglOverrideEl = $('#nglOverride');
const moeOffloadGroup = $('#moeOffloadGroup');
const cpuMoeEl = $('#cpuMoe');
const nCpuMoeEl = $('#nCpuMoe');
const bwUtilEl = $('#bwUtil');
const computeUtilEl = $('#computeUtil');
const prefillMfuEl = $('#prefillMfu');
const weightReadRatioEl = $('#weightReadRatio');
const gpuCountEl = $('#gpuCount');
const interconnectBwEl = $('#interconnectBw');
const flashAttentionEl = $('#flashAttention');
const specDecodeEl = $('#specDecode');
const specDecodeOpts = $('#specDecodeOpts');
const acceptanceRateEl = $('#acceptanceRate');
const draftLenEl = $('#draftLen');
const perfPanel = $('#perfPanel');

const calcBtn = $('#calcBtn');
const loadingEl = $('#loading');
const loadingText = $('#loadingText');
const emptyState = $('#emptyState');
const readyState = $('#readyState');
const resultsEl = $('#results');
const errorMsg = $('#errorMsg');
const modelInfoGrid = $('#modelInfoGrid');
const archBadge = $('#archBadge');
const moeSection = $('#moeSection');
const quantTableBody = $('#quantTableBody');
const fitCheckPanel = $('#fitCheckPanel');

let ssGpu = null, ssCpu = null;

const allWordsSearchFilter = (opt, q) => {
  const words = q.toLowerCase().trim().split(/\s+/);
  if (words.length === 1 && words[0] === '') return true;
  const t = opt.text.toLowerCase();
  return words.every(w => t.includes(w));
};

const allowSpaceInSearch = (ss) => {
  const input = ss?.render?.content?.search?.input;
  if (!input) return;
  const orig = input.onkeydown;
  input.onkeydown = function (e) {
    if (e.key === ' ') return true;
    return orig ? orig.call(this, e) : true;
  };
};

function populateQuantSelect(sel, defaultType) {
  const forkQuantSet = new Set(KV_FORK_GROUPS.flatMap(g => g.quants));
  for (const q of KV_VALID_QUANTS) {
    if (forkQuantSet.has(q)) continue;
    const opt = document.createElement('option');
    opt.value = q;
    opt.textContent = QUANT_NAMES[q] || q;
    if (q === defaultType) opt.selected = true;
    sel.appendChild(opt);
  }
  for (const group of KV_FORK_GROUPS) {
    const og = document.createElement('optgroup');
    og.label = group.label;
    for (const q of group.quants) {
      const opt = document.createElement('option');
      opt.value = q;
      opt.textContent = QUANT_NAMES[q] || q;
      if (q === defaultType) opt.selected = true;
      og.appendChild(opt);
    }
    sel.appendChild(og);
  }
}

populateQuantSelect(kvTypeKEl, GGMLQuantizationType.F16);
populateQuantSelect(kvTypeVEl, GGMLQuantizationType.F16);

function addOptgroup(selectEl, label, entries, fmt) {
  if (!entries.length) return;
  const og = document.createElement('optgroup');
  og.label = label;
  for (const e of entries) {
    const opt = document.createElement('option');
    opt.value = e.id;
    opt.textContent = fmt(e);
    og.appendChild(opt);
  }
  selectEl.appendChild(og);
}

function partitionByGroup(entries) {
  const byVendor = {};
  for (const e of entries) {
    (byVendor[e.vendor] = byVendor[e.vendor] || []).push(e);
  }
  return byVendor;
}

// Shared hardware-select populator. populateGpuSelect and populateCpuSelect
// are 95% structurally identical; only the formatter, the placeholder text,
// and (for CPU) the unified-memory option differ. The caller supplies
// getSS/destroySS/setSS callbacks because ssGpu/ssCpu are module-level lets
// (can't be passed by reference).
function populateHardwareSelect({ el, selectId, presets, fmt, extraOptions, placeholder, getSS, destroySS, setSS }) {
  if (getSS()) { destroySS(); }
  el.innerHTML = '';
  el.appendChild(new Option('Custom', 'custom'));
  for (const opt of (extraOptions || [])) {
    el.appendChild(new Option(opt.name, opt.id));
  }
  const byVendor = partitionByGroup(presets);
  for (const [vendor, all] of Object.entries(byVendor)) {
    const desktop = all.filter(p => (!p.mobile && !p.server) || p.desktop);
    const mobile  = all.filter(p => p.mobile);
    const server  = all.filter(p => p.server);
    addOptgroup(el, vendor, desktop, fmt);
    addOptgroup(el, `${vendor} (mobile)`, mobile, fmt);
    addOptgroup(el, `${vendor} (server)`, server, fmt);
  }
  const ss = new SlimSelect({
    select: el,
    settings: { showSearch: true, searchPlaceholder: placeholder },
    events: { searchFilter: allWordsSearchFilter },
  });
  allowSpaceInSearch(ss);
  setSS(ss);
}

function populateGpuSelect() {
  populateHardwareSelect({
    el: gpuPresetEl,
    presets: getGpuPresets(),
    fmt: g => `${g.name} \u2014 ${g.fp16Tflops} TF, ${g.memBwGBps} GB/s${g.vramGB ? `, ${g.vramGB} GiB` : ''}`,
    placeholder: 'Search GPU...',
    getSS: () => ssGpu,
    destroySS: () => { ssGpu.destroy(); ssGpu = null; },
    setSS: (ss) => { ssGpu = ss; },
  });
}

function populateCpuSelect() {
  populateHardwareSelect({
    el: cpuPresetEl,
    presets: getCpuPresets(),
    fmt: c => (c.fp16Tflops != null && c.defaultRamBwGBps != null)
      ? `${c.name} \u2014 ${c.fp16Tflops} TF, ${c.defaultRamBwGBps} GB/s RAM`
      : c.name,
    extraOptions: [UNIFIED_MEMORY_CPU_PRESET],
    placeholder: 'Search CPU...',
    getSS: () => ssCpu,
    destroySS: () => { ssCpu.destroy(); ssCpu = null; },
    setSS: (ss) => { ssCpu = ss; },
  });
}

// Kick off vendor-preset loading. The populate callbacks fire once each
// class (CPU/GPU) has fully settled, and tryLoadConfig() runs once both
// classes are done.
loadPresetFiles(CPU_JSON_FILES, { merge: mergeCpuPresets, counter: _cpuCounter, populate: populateCpuSelect });
loadPresetFiles(GPU_JSON_FILES, { merge: mergeGpuPresets, counter: _gpuCounter, populate: populateGpuSelect });

const CONFIG_KEY = 'gguf-estimator-config';
const CONFIG_DEFAULTS = {
  contextLen: '4096',
  batchSize: '2048',
  kvTypeK: String(GGMLQuantizationType.F16),
  kvTypeV: String(GGMLQuantizationType.F16),
  gpuPreset: 'custom',
  cpuPreset: 'custom',
  mmProjDevice: 'vram',
  vram: '',
  gpuFlops: '',
  gpuBw: '',
  ram: '',
  cpuFlops: '',
  ramBw: '',
  nglOverride: '',
  cpuMoe: false,
  nCpuMoe: '',
  hfPath: '',
  bwUtil: '0.85',
  computeUtil: '0.50',
  prefillMfu: '0.15',
  weightReadRatio: '1.0',
  gpuCount: '1',
  interconnectBw: '',
  flashAttention: false,
  specDecode: false,
  acceptanceRate: '0.5',
  draftLen: '4',
};

function saveConfig() {
  try {
    const cfg = {
      contextLen: contextLenEl.value,
      batchSize: batchSizeEl.value,
      kvTypeK: kvTypeKEl.value,
      kvTypeV: kvTypeVEl.value,
      gpuPreset: gpuPresetEl.value,
      cpuPreset: cpuPresetEl.value,
      mmProjDevice: mmProjDeviceEl.value,
      vram: vramEl.value,
      gpuFlops: gpuFlopsEl.value,
      gpuBw: gpuBwEl.value,
      ram: ramEl.value,
      cpuFlops: cpuFlopsEl.value,
      ramBw: ramBwEl.value,
      nglOverride: nglOverrideEl.value,
      cpuMoe: cpuMoeEl.checked,
      nCpuMoe: nCpuMoeEl.value,
      swaFull: swaFullEl.checked,
      hfPath: hfPathEl.value,
      bwUtil: bwUtilEl.value,
      computeUtil: computeUtilEl.value,
      prefillMfu: prefillMfuEl.value,
      weightReadRatio: weightReadRatioEl.value,
      gpuCount: gpuCountEl.value,
      interconnectBw: interconnectBwEl.value,
      flashAttention: flashAttentionEl.checked,
      specDecode: specDecodeEl.checked,
      acceptanceRate: acceptanceRateEl.value,
      draftLen: draftLenEl.value,
    };
    localStorage.setItem(CONFIG_KEY, JSON.stringify(cfg));
  } catch {}
}

function applyConfigValues(cfg) {
  if (cfg.hfPath != null) hfPathEl.value = cfg.hfPath;
  if (cfg.contextLen != null) contextLenEl.value = cfg.contextLen;
  if (cfg.batchSize != null) batchSizeEl.value = cfg.batchSize;
  if (cfg.vram != null) vramEl.value = cfg.vram;
  if (cfg.gpuFlops != null) gpuFlopsEl.value = cfg.gpuFlops;
  if (cfg.gpuBw != null) gpuBwEl.value = cfg.gpuBw;
  if (cfg.ram != null) ramEl.value = cfg.ram;
  if (cfg.cpuFlops != null) cpuFlopsEl.value = cfg.cpuFlops;
  if (cfg.ramBw != null) ramBwEl.value = cfg.ramBw;
  if (cfg.nglOverride != null) nglOverrideEl.value = cfg.nglOverride;
  if (cfg.nCpuMoe != null) nCpuMoeEl.value = cfg.nCpuMoe;
  if (cfg.cpuMoe != null) cpuMoeEl.checked = cfg.cpuMoe;
  if (cfg.swaFull != null) swaFullEl.checked = cfg.swaFull;
  if (cfg.mmProjDevice != null) mmProjDeviceEl.value = cfg.mmProjDevice;
  if (cfg.kvTypeK != null) applySelectValue(kvTypeKEl, null, cfg.kvTypeK);
  if (cfg.kvTypeV != null) applySelectValue(kvTypeVEl, null, cfg.kvTypeV);
  if (cfg.gpuPreset != null) applySelectValue(gpuPresetEl, ssGpu, cfg.gpuPreset);
  if (cfg.cpuPreset != null) applySelectValue(cpuPresetEl, ssCpu, cfg.cpuPreset);
  if (cfg.bwUtil != null) bwUtilEl.value = cfg.bwUtil;
  if (cfg.computeUtil != null) computeUtilEl.value = cfg.computeUtil;
  if (cfg.prefillMfu != null) prefillMfuEl.value = cfg.prefillMfu;
  if (cfg.weightReadRatio != null) weightReadRatioEl.value = cfg.weightReadRatio;
  if (cfg.gpuCount != null) gpuCountEl.value = cfg.gpuCount;
  if (cfg.interconnectBw != null) interconnectBwEl.value = cfg.interconnectBw;
  if (cfg.flashAttention != null) flashAttentionEl.checked = cfg.flashAttention;
  if (cfg.specDecode != null) {
    specDecodeEl.checked = cfg.specDecode;
    if (cfg.specDecode) specDecodeOpts.classList.remove('hidden');
  }
  if (cfg.acceptanceRate != null) acceptanceRateEl.value = cfg.acceptanceRate;
  if (cfg.draftLen != null) draftLenEl.value = cfg.draftLen;
}

function loadConfig() {
  try {
    const raw = localStorage.getItem(CONFIG_KEY);
    if (!raw) return;
    const cfg = JSON.parse(raw);
    applyConfigValues(cfg);
    saveConfig();
  } catch {}
}

function resetConfig() {
  try { localStorage.removeItem(CONFIG_KEY); } catch {}
  applyConfigValues(CONFIG_DEFAULTS);
  if (ssGpu) ssGpu.setSelected('custom');
  if (ssCpu) ssCpu.setSelected('custom');
  modelSelectWrap.classList.remove('visible');
  mmProjSelectWrap.classList.remove('visible');
  hide(mmProjDeviceWrap);
  hide(specDecodeOpts);
  moeOffloadGroup.classList.add('hidden');
  errorMsg.classList.remove('visible');
  errorMsg.textContent = '';
  resultsEl.classList.remove('visible');
  emptyState.classList.remove('hidden');
  readyState.classList.add('hidden');
  perfPanel.classList.add('hidden');
  fitCheckPanel.classList.add('hidden');
  currentGGUFUrl = null;
  currentMetadata = null;
  currentTensorInfos = null;
  currentFork = null;
  resetMmProjState();
}

function isUnifiedMemory() {
  const g = getGpuPresets().find(x => x.id === gpuPresetEl.value);
  return !!g?.unifiedMemory;
}

gpuPresetEl.addEventListener('change', () => {
  const g = getGpuPresets().find(x => x.id === gpuPresetEl.value);
  if (g) {
    gpuFlopsEl.value = g.fp16Tflops;
    gpuBwEl.value = g.memBwGBps;
    if (g.vramGB) vramEl.value = g.vramGB;
    if (g.unifiedMemory && ssCpu) ssCpu.setSelected('unified-memory');
    if (g.unifiedMemory) {
      cpuFlopsEl.value = '';
      ramBwEl.value = '';
    }
  }
  saveConfig();
  if (currentMetadata) renderResults();
});
cpuPresetEl.addEventListener('change', () => {
  const c = findCpuPreset(cpuPresetEl.value);
  if (c) {
    cpuFlopsEl.value = c.fp16Tflops ?? '';
    ramBwEl.value = c.defaultRamBwGBps ?? '';
  } else {
    cpuFlopsEl.value = '';
    ramBwEl.value = '';
  }
  saveConfig();
  if (currentMetadata) renderResults();
});
for (const el of [gpuFlopsEl, gpuBwEl, vramEl]) {
  el.addEventListener('input', () => { if (ssGpu) ssGpu.setSelected('custom'); saveConfig(); });
}
for (const el of [cpuFlopsEl, ramBwEl]) {
  el.addEventListener('input', () => { if (ssCpu) ssCpu.setSelected('custom'); saveConfig(); });
}
for (const el of [bwUtilEl, computeUtilEl, prefillMfuEl, weightReadRatioEl, gpuCountEl, interconnectBwEl]) {
  el.addEventListener('input', () => { saveConfig(); if (currentMetadata) renderResults(); });
  el.addEventListener('change', () => { saveConfig(); if (currentMetadata) renderResults(); });
}
flashAttentionEl.addEventListener('change', () => { saveConfig(); if (currentMetadata) renderResults(); });
specDecodeEl.addEventListener('change', () => {
  if (specDecodeEl.checked) specDecodeOpts.classList.remove('hidden');
  else specDecodeOpts.classList.add('hidden');
  saveConfig();
  if (currentMetadata) renderResults();
});
for (const el of [acceptanceRateEl, draftLenEl]) {
  el.addEventListener('input', () => { saveConfig(); if (currentMetadata) renderResults(); });
}

let currentGGUFUrl = null;
let currentMetadata = null;
let currentTensorInfos = null;
let currentFork = null;
let currentMmProjInfo = null;

function resetMmProjState() {
  currentMmProjInfo = null;
  hide(mmProjDeviceWrap);
}

async function doParseGGUF(url) {
  loadingEl.classList.add('visible');
  loadingText.textContent = 'Parsing GGUF metadata...';
  readyState.classList.add('hidden');

  try {
    const result = await parseGGUF(url);
    currentMetadata = result.metadata;
    currentTensorInfos = result.tensorInfos;
    currentFork = result.fork || null;

    loadingEl.classList.remove('visible');
    resultsEl.classList.add('visible');
    renderResults();
  } catch (err) {
    loadingEl.classList.remove('visible');
    readyState.classList.remove('hidden');
    resultsEl.classList.remove('visible');
    currentFork = null;
    showError(err.message);
  }
}

async function doResolveHFModel(path) {
  loadingEl.classList.add('visible');
  emptyState.classList.add('hidden');
  readyState.classList.add('hidden');
  resultsEl.classList.remove('visible');
  errorMsg.classList.remove('visible');
  errorMsg.textContent = '';
  modelSelectWrap.classList.remove('visible');
  mmProjSelectWrap.classList.remove('visible');
  mmProjSelect.value = '';
  resetMmProjState();
  currentMetadata = null;
  currentTensorInfos = null;
  currentFork = null;
  resolveBtn.disabled = true;

  try {
    const result = await resolveHFModel(path);

    loadingEl.classList.remove('visible');

    if (!result.url) {
      modelSelect.innerHTML = '';
      fillSelectWithOptions(modelSelect, result.ggufFiles);
      modelSelectWrap.classList.add('visible');
      currentGGUFUrl = buildResolveUrl(path, modelSelect.value);
    } else {
      currentGGUFUrl = result.url;
    }

    mmProjSelect.innerHTML = '<option value="">None</option>';
    if (result.mmProjFiles && result.mmProjFiles.length) {
      fillSelectWithOptions(mmProjSelect, result.mmProjFiles);
      mmProjSelectWrap.classList.add('visible');
    }

    readyState.classList.remove('hidden');

  } catch (err) {
    loadingEl.classList.remove('visible');
    emptyState.classList.remove('hidden');
    showError(err.message);
  } finally {
    resolveBtn.disabled = false;
  }
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.add('visible');
}

function deriveGgufId(repoPath, ggufUrl, basename) {
  if (!repoPath || !ggufUrl) return null;
  const repo = extractHfSlug(repoPath) ?? repoPath.trim();
  const filename = decodeURIComponent(ggufUrl.split('/').pop().replace(/#.*$/, ''));
  const stem = filename.replace(/\.gguf$/i, '');
  if (!stem) return null;
  if (basename && stem.startsWith(basename + '-')) {
    return `${repo}:${stem.slice(basename.length + 1)}`;
  }
  return `${repo}:${stem}`;
}

function renderModelInfo(arch, handler, isMoe, isMla, moe, ctx_len, vocab) {
  const n_embd = getMeta(currentMetadata, `${arch}.embedding_length`);
  const n_head = getMeta(currentMetadata, `${arch}.attention.head_count`);
  const n_head_kv = getMeta(currentMetadata, `${arch}.attention.head_count_kv`);
  const n_layer = getMeta(currentMetadata, `${arch}.block_count`);
  const n_ff = getMeta(currentMetadata, `${arch}.feed_forward_length`);
  const nextn = getMeta(currentMetadata, `${arch}.nextn_predict_layers`);
  const n_main = Math.max(0, n_layer - nextn);
  const modelName = currentMetadata['general.name'] || currentMetadata['general.basename'] || arch;
  const archCategories = handler.categories.join(', ');

  if (ctx_len && ctx_len > 0) {
    contextLenEl.max = ctx_len;
    // Clamp existing input value to the new model max. Previously this ran
    // inside renderResults(), which fires on every input change and could
    // fight the user's typing as well as leave saveConfig() persisting the
    // un-clamped value. Doing it here runs once per model load.
    const cur = parseInt(contextLenEl.value, 10);
    if (Number.isFinite(cur) && cur > ctx_len) {
      contextLenEl.value = ctx_len;
      saveConfig();
    }
    $('#ctxMaxLabel').textContent = `(max ${formatElements(BigInt(ctx_len))})`;
  } else {
    $('#ctxMaxLabel').textContent = '';
  }

  const denseOrMoeBadge = isMoe
    ? '<span class="status-badge moe">MoE</span>'
    : '<span class="status-badge dense">Dense</span>';
  const mtpBadge = nextn > 0
    ? `<span class="status-badge mtp" title="Multi-Token Prediction: ${nextn} trailing layer${nextn > 1 ? 's' : ''} loaded for speculative decoding, skipped by main decoder">MTP</span>`
    : '';
  const forkBadge = currentFork
    ? `<span class="status-badge fork fork-${currentFork}" title="Quantization fork detected: ${currentFork}. BPE overrides applied to match fork-specific type assignments.">${currentFork}</span>`
    : '';
  archBadge.innerHTML = denseOrMoeBadge + mtpBadge + forkBadge;

  modelInfoGrid.textContent = '';

  const formatInfoValue = (value) => {
    if (Array.isArray(value)) {
      if (value.length === 0) return '-';
      const nums = value.map(Number);
      if (nums.every(Number.isFinite)) {
        const min = Math.min(...nums);
        const max = Math.max(...nums);
        return min === max ? String(min) : `${min}\u2013${max}`;
      }
      return String(value[0]);
    }
    return value;
  };

  const addInfo = (label, value, smallValue = false, fullWidth = false) => {
    const item = document.createElement('div');
    item.className = 'info-item';
    if (fullWidth) item.classList.add('full-width');
    const labelEl = document.createElement('div');
    labelEl.className = 'label';
    labelEl.textContent = label;
    const valueEl = document.createElement('div');
    valueEl.className = 'value';
    if (smallValue) valueEl.style.fontSize = smallValue === true ? '0.85rem' : smallValue;
    valueEl.textContent = formatInfoValue(value);
    item.appendChild(labelEl);
    item.appendChild(valueEl);
    modelInfoGrid.appendChild(item);
  };

  addInfo('Model', modelName, true);
  addInfo('Architecture', arch);
  addInfo('Categories', archCategories, '0.75rem');
  addInfo('Layers', nextn > 0 ? `${n_layer} (${n_main} + ${nextn} MTP)` : n_layer, nextn > 0 ? '0.85rem' : false);
  addInfo('Heads', n_head);
  addInfo('KV Heads', n_head_kv);
  addInfo('Hidden', n_embd);
  addInfo('FFN', n_ff);
  addInfo('Context', ctx_len ? formatElements(ctx_len) : '-');
  addInfo('Vocab', vocab ? formatElements(BigInt(vocab)) : '-');
  if (isMla) {
    addInfo('KV LoRA Rank', getMeta(currentMetadata, `${arch}.attention.kv_lora_rank`));
    addInfo('Q LoRA Rank', getMeta(currentMetadata, `${arch}.attention.q_lora_rank`));
    addInfo('Key MLA', getMeta(currentMetadata, `${arch}.attention.key_length_mla`));
    addInfo('Value MLA', getMeta(currentMetadata, `${arch}.attention.value_length_mla`));
  }
  if (handler.categories.includes('iswa')) {
    addInfo('Sliding Window', getMeta(currentMetadata, `${arch}.attention.sliding_window`) || 'off');
  }
  if (isMoe) {
    addInfo('Experts', `${moe.expertCount} (\u00D7${moe.expertUsedCount})`);
  }

  const ggufId = deriveGgufId(hfPathEl.value, currentGGUFUrl, currentMetadata['general.basename']);
  if (ggufId) addInfo('GGUF', ggufId, true, true);
}

function renderMoeSection(moe, cpuMoe, nCpuMoe) {
  if (moe) {
    moeSection.classList.remove('hidden');
    moeOffloadGroup.classList.remove('hidden');
    $('#moeTotalExperts').textContent = moe.expertCount;
    $('#moeActiveExperts').textContent = `${moe.expertUsedCount} per token`;
    $('#moeTotalParams').textContent = `${formatElements(moe.totalModelParams)} params (${formatBytes(moe.totalWeightBytes)} weights)`;
    $('#moeActiveParams').textContent = `${formatElements(moe.expertParams)} params (${formatBytes(moe.activeExpertWeightBytes)} weights)`;
    $('#moeRouterSize').textContent = formatBytes(moe.routerBytes);
  } else {
    moeSection.classList.add('hidden');
    moeOffloadGroup.classList.add('hidden');
  }
}

function renderWeightsTable(weights) {
  $('#weightTotal').textContent = formatBytes(weights.total);

  const sortedQuants = Object.entries(weights.byQuant)
    .sort((a, b) => b[1].bytes - a[1].bytes);

  quantTableBody.textContent = '';
  for (const [name, info] of sortedQuants) {
    const tr = document.createElement('tr');
    const cells = [
      { text: name },
      { text: String(info.count), cls: 'right' },
      { text: formatElements(info.elements), cls: 'right' },
      { text: formatBytes(info.bytes), cls: 'right' },
    ];
    for (const c of cells) {
      const td = document.createElement('td');
      if (c.cls) td.className = c.cls;
      td.textContent = c.text;
      tr.appendChild(td);
    }
    quantTableBody.appendChild(tr);
  }
}

function renderKvCache(kv, kvTypeK, kvTypeV, isMla, arch) {
  $('#kvKLabel').textContent = QUANT_NAMES[kvTypeK] || kvTypeK;
  $('#kvVLabel').textContent = QUANT_NAMES[kvTypeV] || kvTypeV;
  $('#kvKSize').textContent = formatBytes(kv.bytesK);
  $('#kvVSize').textContent = formatBytes(kv.bytesV);
  $('#kvLayers').textContent = kv.layers;
  if (isMla) {
    const archMeta = getModelArch(currentMetadata);
    const kvLora = getMeta(currentMetadata, `${archMeta}.attention.kv_lora_rank`);
    const nRot = getMeta(currentMetadata, `${archMeta}.rope.dimension_count`);
    $('#kvHeads').textContent = `K:${kvLora}+${nRot} V:none (MLA)`;
  } else {
    $('#kvHeads').textContent = kv.avgHeadsKV.toFixed(1);
  }
}

function renderMemoryPanel({ weights, moe, kv, acts, memBreakdown, footprint, mmProjBytes, mmProjDevice, vramBytes, ramBytes }) {
  const totalBytes = vramBytes + ramBytes;
  const pctOf = (b, total) => total > 0 ? `${(b / total * 100).toFixed(1)}%` : '0%';
  // Write "<bytes> (<pct of device total>)" into the element at `sel`.
  const sized = (sel, bytes, total) => { $(sel).textContent = `${formatBytes(bytes)} (${pctOf(bytes, total)})`; };
  const mtpBytes = (footprint && footprint.mtpBytes) || 0;
  // token_embd is always on the CPU in llama.cpp; keep it out of the VRAM
  // weights figure so the bar sums correctly, and show it in the RAM panel.
  const inputEmb = (footprint && footprint.inputEmbBytes) || 0;

  let nonMoEWeightBytes, vramExpertBytes = 0, vramRouterSharedBytes = 0;
  if (moe) {
    nonMoEWeightBytes = weights.total - moe.expertWeightBytes - moe.routerBytes - moe.sharedBytes - mtpBytes - inputEmb;
    vramExpertBytes = memBreakdown.vramWeightBytes - (nonMoEWeightBytes + mtpBytes) - moe.routerBytes - moe.sharedBytes;
    vramRouterSharedBytes = moe.routerBytes + moe.sharedBytes;
  } else {
    nonMoEWeightBytes = weights.total - mtpBytes - inputEmb;
  }

  $('#vramSize').textContent = formatBytes(vramBytes);

  if (moe) {
    $('#vramWeightsRow .label').textContent = 'Attention + embedding weights';
    show($('#vramActiveExpertRow'));
    $('#vramActiveExpertLabel').textContent = `All experts (${moe.expertCount})`;
    sized('#vramActiveExpertSize', vramExpertBytes, vramBytes);
    if (vramRouterSharedBytes > 0) {
      show($('#vramRouterRow'));
      sized('#vramRouterSize', vramRouterSharedBytes, vramBytes);
    } else {
      hide($('#vramRouterRow'));
    }
  } else {
    $('#vramWeightsRow .label').textContent = 'Weights';
    hide($('#vramActiveExpertRow'));
    hide($('#vramRouterRow'));
  }
  sized('#vramWeightsSize', nonMoEWeightBytes, vramBytes);
  if (mtpBytes > 0) {
    show($('#vramMtpRow'));
    sized('#vramMtpSize', mtpBytes, vramBytes);
  } else {
    hide($('#vramMtpRow'));
  }
  sized('#vramKVSize', kv.bytesK + kv.bytesV, vramBytes);
  if (kv.bytesRecurrent > 0) {
    show($('#vramRecurrentRow'));
    sized('#vramRecurrentSize', kv.bytesRecurrent, vramBytes);
  } else {
    hide($('#vramRecurrentRow'));
  }
  sized('#vramActSize', acts.totalBytes, vramBytes);
  if (currentMmProjInfo && mmProjDevice === 'vram') {
    show($('#vramMmProjRow'));
    sized('#vramMmProjSize', mmProjBytes, vramBytes);
  } else {
    hide($('#vramMmProjRow'));
  }

  $('#ramSize').textContent = ramBytes > 0 ? formatBytes(ramBytes) : 'None';
  if (moe && memBreakdown.ramExpertBytes > 0) {
    show($('#ramInactiveRow'));
    $('#ramInactiveLabel').textContent = 'Inactive experts';
    sized('#ramInactiveSize', memBreakdown.ramExpertBytes, ramBytes);
  } else {
    hide($('#ramInactiveRow'));
  }
  if (inputEmb > 0) {
    show($('#ramInputEmbRow'));
    sized('#ramInputEmbSize', inputEmb, ramBytes);
  } else {
    hide($('#ramInputEmbRow'));
  }
  if (currentMmProjInfo && mmProjDevice === 'ram') {
    show($('#ramMmProjRow'));
    sized('#ramMmProjSize', mmProjBytes, ramBytes);
  } else {
    hide($('#ramMmProjRow'));
  }

  $('#totalSize').textContent = formatBytes(totalBytes);
}

function renderMmProjPanel(mmProjDevice) {
  const mmProjPanel = $('#mmProjPanel');
  if (currentMmProjInfo) {
    mmProjPanel.classList.remove('hidden');
    const mp = currentMmProjInfo;
    const modalityParts = [];
    if (mp.hasVision) modalityParts.push('vision');
    if (mp.hasAudio) modalityParts.push('audio');
    $('#mmProjFile').textContent = mmProjSelect.value || '-';
    $('#mmProjType').textContent = mp.projType
      ? (mp.projTypeKnown ? mp.projType : `${mp.projType} (unknown formula \u2014 generic fallback)`)
      : '(unspecified)';
    $('#mmProjModality').textContent = modalityParts.length ? modalityParts.join(' + ') : 'unknown';
    $('#mmProjImage').textContent = (mp.imageSize && mp.patchSize)
      ? `${mp.imageSize}\u00D7${mp.imageSize} / patch ${mp.patchSize}${mp.nMerge > 1 ? ` (merge ${mp.nMerge})` : ''}`
      : '-';
    $('#mmProjDims').textContent = (mp.nLayerV || mp.nEmbdV)
      ? `${mp.nLayerV || '-'} layers / ${mp.nEmbdV || '-'} hidden \u2192 ${mp.projDim || '-'} out`
      : '-';
    if (mp.isAudioProj) {
      $('#mmProjTokens').textContent = 'n/a (audio, runtime-dependent)';
      $('#mmProjAct').textContent = 'n/a (audio, runtime-dependent)';
    } else {
      $('#mmProjTokens').textContent = mp.nOutputTokens ? mp.nOutputTokens.toString() : '-';
      $('#mmProjAct').textContent = mp.perImageActBytes ? formatBytes(mp.perImageActBytes) : '-';
    }
    $('#mmProjWeights').textContent = formatBytes(mp.weightBytes);
    $('#mmProjPlacement').textContent = mmProjDevice === 'ram' ? 'RAM (--no-mmproj-offload)' : 'VRAM';
  } else {
    mmProjPanel.classList.add('hidden');
  }
}

// Shared usage-bar scaffolding for the VRAM/RAM fit sections: bar width +
// color class, status line, percentage text, "<used> / <total> GiB" label,
// and the "A · B · C" breakdown line. Callers supply the color and status
// text since the wording differs per device.
function renderUsageBar(prefix, { usagePct, color, statusText, usedBytes, totalGB, breakdownParts }) {
  const bar = $(`#${prefix}Bar`);
  bar.style.width = `${Math.min(usagePct, 100)}%`;
  bar.className = `usage-bar ${color}`;
  const status = $(`#${prefix}Status`);
  status.className = `usage-status ${color}`;
  status.textContent = statusText;
  $(`#${prefix}BarText`).textContent = `${usagePct.toFixed(1)}%`;
  $(`#${prefix}FitLabel`).textContent = `${formatBytes(usedBytes)} / ${totalGB} GiB`;
  $(`#${prefix}Breakdown`).textContent = breakdownParts.length > 0 ? breakdownParts.join(' · ') : '';
}

function renderFitCheck({ vramGB, ramGB, acts, layerFootprint, mmProjDevice, cpuMoe, nCpuMoe, nglOverride, unifiedMemory }) {
  const fitPanel = $('#fitCheckPanel');
  const showVramBar = vramGB > 0;

  if (!showVramBar) {
    fitPanel.classList.add('hidden');
    return;
  }

  fitPanel.classList.remove('hidden');

  const mmProjActBytes = totalMmProjBytes(currentMmProjInfo);
  const reservedBytes = acts.totalBytes + (mmProjDevice !== 'ram' ? mmProjActBytes : 0);
  const actual = calcActualMemory({
    vramBytes: vramGB * GIB,
    footprint: layerFootprint,
    activationBytes: reservedBytes,
    nLayerOverride: nglOverride,
    cpuMoe,
    nCpuMoe,
    unifiedMemory: !!unifiedMemory,
  });

  show($('#vramFitSection'));
  const vramUsagePct = (actual.actualVram / (vramGB * GIB) * 100);

  const np = actual.nPartialLayers || 0;
  const parts = [`${actual.nGpuLayers} GPU`];
  if (np > 0) parts.push(`${np} partial`);
  if (actual.nHybridLayers > 0) parts.push(`${actual.nHybridLayers} hybrid`);
  if (actual.nCpuLayers > 0) parts.push(`${actual.nCpuLayers} CPU`);
  const fullOffload = actual.nGpuLayers > 0 && np === 0 && actual.nHybridLayers === 0 && actual.nCpuLayers === 0;
  const layerSplitStr = fullOffload ? `${actual.nGpuLayers} GPU (full offload)` : parts.join(' / ');

  let vramColor, vramStatusText;
  if (vramUsagePct > 100) {
    vramColor = 'red';
    vramStatusText = `\u2717 Overflow \u2014 ${formatBytes(actual.actualVram)} needed, ${vramGB} GiB available \u2014 ${layerSplitStr}`;
  } else if (actual.nCpuLayers > 0) {
    vramColor = 'yellow';
    vramStatusText = `\u26A0 Partial offload \u2014 ${formatBytes(actual.actualVram)} of ${vramGB} GiB (${vramUsagePct.toFixed(0)}%) \u2014 ${layerSplitStr}`;
  } else {
    vramColor = 'green';
    vramStatusText = `\u2713 Fits \u2014 ${formatBytes(actual.actualVram)} of ${vramGB} GiB (${vramUsagePct.toFixed(0)}%) \u2014 ${layerSplitStr}`;
  }

  const vramParts = [];
  if (actual.vramWeightsBytes > 0) vramParts.push(`Weights ${formatBytes(actual.vramWeightsBytes)}`);
  if (actual.vramExpertBytes > 0) vramParts.push(`Experts ${formatBytes(actual.vramExpertBytes)}`);
  if (actual.vramKvBytes > 0) vramParts.push(`KV ${formatBytes(actual.vramKvBytes)}`);
  if (actual.vramActivationBytes > 0) vramParts.push(`Activations ${formatBytes(actual.vramActivationBytes)}`);
  if (actual.vramOutputBytes > 0) vramParts.push(`Output ${formatBytes(actual.vramOutputBytes)}`);
  renderUsageBar('vram', {
    usagePct: vramUsagePct, color: vramColor, statusText: vramStatusText,
    usedBytes: actual.actualVram, totalGB: vramGB, breakdownParts: vramParts,
  });

  const actualRamBytes = actual.actualRam + (mmProjDevice === 'ram' ? mmProjActBytes : 0);
  const showRamBar = ramGB > 0 && actualRamBytes > 0;
  if (showRamBar) {
    show($('#ramFitSection'));
    const ramUsagePct = (actualRamBytes / (ramGB * GIB) * 100);

    let ramColor, ramStatusText;
    if (ramUsagePct <= RAM_GREEN_PCT) {
      ramColor = 'green';
      ramStatusText = `\u2713 RAM usage \u2014 ${formatBytes(actualRamBytes)} of ${ramGB} GiB (${ramUsagePct.toFixed(0)}% used)`;
    } else if (ramUsagePct <= RAM_YELLOW_PCT) {
      ramColor = 'yellow';
      ramStatusText = `\u26A0 RAM usage \u2014 ${formatBytes(actualRamBytes)} of ${ramGB} GiB (${ramUsagePct.toFixed(0)}% used)`;
    } else {
      ramColor = 'red';
      ramStatusText = `\u2717 RAM overflow \u2014 ${formatBytes(actualRamBytes)} needed, ${ramGB} GiB available`;
    }

    const ramParts = [];
    if (actual.ramExpertBytes > 0) ramParts.push(`Expert weights ${formatBytes(actual.ramExpertBytes)}`);
    if (actual.ramCpuLayerBytes > 0) ramParts.push(`CPU layer weights ${formatBytes(actual.ramCpuLayerBytes)}`);
    if (actual.ramCpuKvBytes > 0) ramParts.push(`CPU layer KV ${formatBytes(actual.ramCpuKvBytes)}`);
    if (actual.ramInputEmbBytes > 0) ramParts.push(`Input embedding ${formatBytes(actual.ramInputEmbBytes)}`);
    renderUsageBar('ram', {
      usagePct: ramUsagePct, color: ramColor, statusText: ramStatusText,
      usedBytes: actualRamBytes, totalGB: ramGB, breakdownParts: ramParts,
    });
  } else {
    hide($('#ramFitSection'));
  }
}

function renderPerformance({ ctxSize, batchSize, kv, moe, acts, vramGB, mmProjDevice, cpuMoe, nCpuMoe, nglOverride, unifiedMemory }) {
  const gpuFlopsV = parseFloat(gpuFlopsEl.value);
  const gpuBwV = parseFloat(gpuBwEl.value);
  let cpuFlopsV = parseFloat(cpuFlopsEl.value);
  let ramBwV = parseFloat(ramBwEl.value);
  const hasGpuPerf = Number.isFinite(gpuFlopsV) && Number.isFinite(gpuBwV) && gpuFlopsV > 0 && gpuBwV > 0;
  let hasCpuPerf = Number.isFinite(cpuFlopsV) && Number.isFinite(ramBwV) && cpuFlopsV > 0 && ramBwV > 0;
  let cpuFallback = null;
  if (!unifiedMemory && !hasCpuPerf && hasGpuPerf) {
    const slow = getSlowestCpuPreset();
    if (slow) {
      cpuFlopsV = slow.fp16Tflops;
      ramBwV = slow.defaultRamBwGBps;
      cpuFallback = slow;
      hasCpuPerf = true;
    }
  }
  if (unifiedMemory) hasCpuPerf = false;

  if (!hasGpuPerf) {
    perfPanel.classList.add('hidden');
    return;
  }

  perfPanel.classList.remove('hidden');
  const gpuCount = parseInt(gpuCountEl.value, 10) || 1;
  const perf = estimatePerformance({
    metadata: currentMetadata, tensorInfos: currentTensorInfos,
    ctx: ctxSize, batchSize,
    kv, moe, activations: acts, mmproj: currentMmProjInfo,
    device: {
      gpu: {
        flopsFp16Tflops: gpuFlopsV,
        bwGBps: gpuBwV,
        vramBytes: vramGB > 0 ? vramGB * GIB : 0,
        preset: (() => {
          const id = gpuPresetEl.value;
          if (!id || id === 'custom') return null;
          const p = getGpuPresets().find(g => g.id === id);
          return p || null;
        })(),
      },
      cpu: hasCpuPerf ? { flopsFp16Tflops: cpuFlopsV, bwGBps: ramBwV } : null,
      nGpuLayers: nglOverride,
      mmprojOnGpu: mmProjDevice !== 'ram',
      cpuMoe,
      nCpuMoe,
      unifiedMemory: !!unifiedMemory,
      efficiency: {
        bwUtilization: parseFloat(bwUtilEl.value) || 0.85,
        computeUtilization: parseFloat(computeUtilEl.value) || 0.50,
        prefillMfu: parseFloat(prefillMfuEl.value) || 0.15,
        weightReadRatio: parseFloat(weightReadRatioEl.value) || 1.0,
      },
      gpuCount,
      interconnectBwGBps: parseFloat(interconnectBwEl.value) || 0,
      flashAttention: flashAttentionEl.checked,
      speculativeDecoding: specDecodeEl.checked ? {
        acceptanceRate: parseFloat(acceptanceRateEl.value) || 0.5,
        draftLen: parseInt(draftLenEl.value, 10) || 4,
      } : null,
    },
  });
  $('#perfDecode').textContent = `${perf.decodeTPS.toFixed(1)} tok/s`;
  $('#perfPrefill').textContent = `${perf.prefillTPS.toFixed(0)} tok/s`;
  $('#perfTtft').textContent = perf.ttftSec < 1
    ? `${(perf.ttftSec * 1000).toFixed(0)} ms`
    : `${perf.ttftSec.toFixed(2)} s`;
  const nHybrid = perf.nHybridLayers || 0;
  const nPartial = perf.nPartialLayers || 0;
  const nSplit = nHybrid + nPartial;
  $('#perfSplit').textContent = nSplit > 0
    ? `${perf.nGpuLayers} / ${nSplit} / ${perf.nCpuLayers}${perf.autoSplit ? ' (auto)' : ''}`
    : `${perf.nGpuLayers} / ${perf.nCpuLayers}${perf.autoSplit ? ' (auto)' : ''}`;
  $('#perfGpuLayer').textContent = perf.nGpuLayers > 0
    ? `${perf.perLayerMs.gpu.toFixed(3)} ms (${perf.bottleneck.gpu || '-'})`
    : '\u2014';
  const hybridRow = $('#perfHybridRow');
  if (nSplit > 0) {
    hybridRow.classList.remove('hidden');
    // When both hybrid and partial layers exist, show their weighted-average
    // per-layer ms so the displayed value reflects all split layers.
    const hMs = perf.perLayerMs.hybrid || 0;
    const pMs = perf.perLayerMs.partial || 0;
    const splitMs = (nHybrid > 0 && nPartial > 0)
      ? (hMs * nHybrid + pMs * nPartial) / nSplit
      : (nHybrid > 0 ? hMs : pMs);
    const splitLabel = nHybrid > 0 && nPartial > 0 ? 'hybrid+partial' : (nPartial > 0 ? 'partial' : 'hybrid');
    $('#perfHybridLayer').textContent = hasCpuPerf && !cpuFallback
      ? `${splitMs.toFixed(2)} ms (${perf.bottleneck.cpu || '-'}) \u00B7 ${splitLabel}`
      : cpuFallback
        ? `${splitMs.toFixed(2)} ms (${perf.bottleneck.cpu || '-'}, est. from ${cpuFallback.name}) \u00B7 ${splitLabel}`
        : '\u2014 (CPU not set \u2014 expert cost unaccounted)';
  } else {
    hybridRow.classList.add('hidden');
  }
  $('#perfCpuLayer').textContent = perf.nCpuLayers > 0 && hasCpuPerf && !cpuFallback
    ? `${perf.perLayerMs.cpu.toFixed(2)} ms (${perf.bottleneck.cpu || '-'})`
    : perf.nCpuLayers > 0 && cpuFallback
      ? `${perf.perLayerMs.cpu.toFixed(2)} ms (${perf.bottleneck.cpu || '-'}, est. from ${cpuFallback.name})`
      : (perf.nCpuLayers > 0 ? '\u2014 (CPU not set \u2014 spill unaccounted)' : '\u2014 (no spill)');
  $('#perfBottleneck').textContent = perf.bottleneck.overall;
  const gpuLabel = gpuPresetEl.options[gpuPresetEl.selectedIndex]?.textContent || 'Custom';
  const cpuLabel = cpuFallback
    ? cpuFallback.name + ' (fallback)'
    : hasCpuPerf
      ? (cpuPresetEl.options[cpuPresetEl.selectedIndex]?.textContent || 'Custom')
      : 'none';
  $('#perfHardware').textContent = `GPU: ${gpuLabel.replace(/ \u2014.*$/, '')}${gpuCount > 1 ? ` ×${gpuCount}` : ''} \u00B7 CPU: ${cpuLabel.replace(/ \u2014.*$/, '')}`;
  const cal = perf.calibration;
  if (cal) {
    const parts = [
      `bw ${cal.bwUtilization.toFixed(2)}`,
      `cmp ${cal.computeUtilization.toFixed(2)}`,
      `mfu ${cal.prefillMfu.toFixed(2)}`,
      `wrr ${cal.weightReadRatio.toFixed(2)}`,
    ];
    if (cal.gpuCount > 1) parts.push(`TP×${cal.gpuCount}`);
    if (cal.flashAttention) parts.push(`FA ${cal.flashAttentionBoost.toFixed(2)}×`);
    if (cal.speculativeSpeedup > 1) parts.push(`spec ${cal.speculativeSpeedup.toFixed(2)}×`);
    if (cal.moeDispatchLatencyMs > 0) parts.push(`MoE ${cal.moeDispatchLatencyMs.toFixed(3)}ms`);
    if (cal.appleBwScale !== 1) parts.push(`apple ${cal.appleBwScale.toFixed(2)}×`);
    $('#perfCalibration').textContent = parts.join(' \u00B7 ');
  }
}

function renderResults() {
  if (!currentMetadata || !currentTensorInfos) return;

  const forkHintEl = $('#forkHint');
  if (currentFork && forkHintEl) {
    const forkKvTypes = KV_FORK_GROUPS.find(g => g.label === currentFork);
    const hasKvTypes = forkKvTypes && forkKvTypes.quants.length > 0;
    forkHintEl.innerHTML = `<strong>${currentFork}</strong> fork detected &mdash; weight BPE overrides applied.${hasKvTypes ? ` Use the <strong>${currentFork}</strong> KV cache types below for this fork.` : ''}`;
    forkHintEl.className = `fork-hint fork-${currentFork}`;
    show(forkHintEl);
  } else if (forkHintEl) {
    hide(forkHintEl);
  }

  const arch = getModelArch(currentMetadata);
  const handler = getArchHandler(arch);

  const modelCtxLen = getMeta(currentMetadata, `${arch}.context_length`);
  // ctxSize clamping happens in renderModelInfo() on model load; here we
  // just read the (already-valid) input value.
  const ctxSize = parseInt(contextLenEl.value, 10) || 4096;
  // Use CONFIG_DEFAULTS.batchSize as the fallback so that HTML default,
  // reset-to-default, and cleared-field all agree. Previously fell back to
  // `1` (the CLI default), disagreeing with both HTML value="2048" and
  // CONFIG_DEFAULTS.
  const batchSize = parseInt(batchSizeEl.value, 10) || parseInt(CONFIG_DEFAULTS.batchSize, 10);
  // Parse kvTypeK/V: numeric IDs become numbers, named types (F16, TURBO3_0, etc.) stay strings.
  // Use a strict integer regex instead of isNaN() because isNaN('') === false, which would
  // propagate NaN through parseInt('', 10) when the select is momentarily empty.
  const parseKv = (v) => /^\d+$/.test(v) ? parseInt(v, 10) : v;
  const kvTypeK = parseKv(kvTypeKEl.value);
  const kvTypeV = parseKv(kvTypeVEl.value);
  const vramGB = parseFloat(vramEl.value) || 0;
  const ramGB = parseFloat(ramEl.value) || 0;
  const nglRaw = nglOverrideEl.value.trim();
  const nglOverride = (nglRaw === '' || nglRaw.toLowerCase() === 'auto')
    ? 'auto'
    : (Number.isFinite(parseInt(nglRaw, 10)) ? parseInt(nglRaw, 10) : 'auto');

  const weights = calcWeightSize(currentTensorInfos);
  const kv = calcKVCache(currentMetadata, ctxSize, kvTypeK, kvTypeV, 1, swaFullEl.checked);
  const acts = calcActivations(currentMetadata, batchSize);
  const moe = calcMoEInfo(currentMetadata, currentTensorInfos);
  const cpuMoe = cpuMoeEl.checked;
  const nCpuMoe = parseInt(nCpuMoeEl.value, 10) || 0;
  const layerFootprint = calcPerLayerFootprint(currentMetadata, currentTensorInfos, kv, moe);
  const memBreakdown = calcMemoryBreakdown({
    weights, kv, activations: acts,
    footprint: layerFootprint,
  });

  const mmProjDevice = mmProjDeviceEl.value;
  let vramBytes = memBreakdown.vramBytes;
  let ramBytes = memBreakdown.ramBytes;
  const mmProjBytes = totalMmProjBytes(currentMmProjInfo);
  if (mmProjBytes > 0) {
    if (mmProjDevice === 'ram') ramBytes += mmProjBytes;
    else vramBytes += mmProjBytes;
  }

  const isMoe = (moe !== null);
  const isMla = handler.categories.includes('mla');
  const isIswa = handler.categories.includes('iswa');
  const vocab = getMeta(currentMetadata, `${arch}.vocab_size`);

  swaFullEl.disabled = !isIswa;

  renderModelInfo(arch, handler, isMoe, isMla, moe, modelCtxLen, vocab);
  renderMoeSection(moe, cpuMoe, nCpuMoe);
  renderWeightsTable(weights);
  renderKvCache(kv, kvTypeK, kvTypeV, isMla, arch);
  $('#actSize').textContent = formatBytes(acts.totalBytes);
  renderMemoryPanel({ weights, moe, kv, acts, memBreakdown, footprint: layerFootprint, mmProjBytes, mmProjDevice, vramBytes, ramBytes });
  renderMmProjPanel(mmProjDevice);
  const um = isUnifiedMemory();
  renderFitCheck({ vramGB, ramGB, acts, layerFootprint, mmProjDevice, cpuMoe, nCpuMoe, nglOverride, unifiedMemory: um });
  renderPerformance({ ctxSize, batchSize, kv, moe, acts, vramGB, mmProjDevice, cpuMoe, nCpuMoe, nglOverride, unifiedMemory: um });
}

resolveBtn.addEventListener('click', () => {
  const path = hfPathEl.value.trim();
  if (!path) { showError('Please enter a HuggingFace model path or URL.'); return; }
  doResolveHFModel(path);
});

calcBtn.addEventListener('click', async () => {
  if (!currentGGUFUrl) {
    showError('No model loaded. Resolve a model first.');
    return;
  }
  if (!currentMetadata || !currentTensorInfos) {
    await doParseGGUF(currentGGUFUrl);
  } else {
    renderResults();
  }
});

['contextLen', 'batchSize', 'vram', 'ram', 'kvTypeK', 'kvTypeV',
 'gpuFlops', 'gpuBw', 'cpuFlops', 'ramBw', 'nglOverride', 'nCpuMoe'].forEach(id => {
  document.getElementById(id).addEventListener('change', () => {
    saveConfig();
    if (currentMetadata && currentTensorInfos) renderResults();
  });
});
cpuMoeEl.addEventListener('change', () => {
  saveConfig();
  if (currentMetadata && currentTensorInfos) renderResults();
});
swaFullEl.addEventListener('change', () => {
  saveConfig();
  if (currentMetadata && currentTensorInfos) renderResults();
});
mmProjDeviceEl.addEventListener('change', () => {
  saveConfig();
  if (currentMetadata && currentMmProjInfo) renderResults();
});
hfPathEl.addEventListener('input', () => { saveConfig(); });

$('#resetBtn').addEventListener('click', resetConfig);

modelSelect.addEventListener('change', () => {
  const path = hfPathEl.value.trim();
  const url = buildResolveUrl(path, modelSelect.value);
  currentGGUFUrl = url;
  doParseGGUF(url);
});

async function doParseMmProj(url) {
  try {
    const { metadata, tensorInfos } = await parseGGUF(url);
    currentMmProjInfo = calcMmProj(metadata, tensorInfos);
    show(mmProjDeviceWrap);
  } catch (err) {
    resetMmProjState();
    showError(`mmproj parse failed: ${err.message}`);
  }
}

mmProjSelect.addEventListener('change', async () => {
  const filename = mmProjSelect.value;
  if (!filename) {
    resetMmProjState();
    if (currentMetadata) renderResults();
    return;
  }
  const path = hfPathEl.value.trim();
  const mmProjUrl = buildResolveUrl(path, filename);
  loadingEl.classList.add('visible');
  loadingText.textContent = 'Parsing mmproj metadata...';
  await doParseMmProj(mmProjUrl);
  loadingEl.classList.remove('visible');
  if (currentMetadata) renderResults();
});

hfPathEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') resolveBtn.click();
});
