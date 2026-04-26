import { GGMLQuantizationType, KV_VALID_QUANTS, KV_FORK_GROUPS, parseGGUF, resolveHFModel, buildResolveUrl } from './parsing.js';
import { QUANT_NAMES, getArchHandler, getModelArch, getMeta, calcWeightSize, calcKVCache, calcActivations, calcMoEInfo, calcMmProj, calcPerLayerFootprint, calcMemoryBreakdown, calcActualMemory, estimatePerformance, formatBytes, formatElements } from './calculations.js';
import { mergeCpuPresets, mergeGpuPresets, getCpuPresets, getGpuPresets, findCpuPreset, getSlowestCpuPreset, UNIFIED_MEMORY_CPU_PRESET } from './hardware-presets.js';

const CPU_JSON_FILES = ['intel-cpu-presets.json', 'amd-cpu-presets.json'];
const GPU_JSON_FILES = ['nvidia-gpu-presets.json', 'intel-gpu-presets.json', 'amd-gpu-presets.json', 'apple-gpu-presets.json'];

let _cpuLoaded = 0, _gpuLoaded = 0;
let _configLoaded = false;

function tryLoadConfig() {
  if (_configLoaded) return;
  if (_cpuLoaded >= CPU_JSON_FILES.length && _gpuLoaded >= GPU_JSON_FILES.length) {
    _configLoaded = true;
    loadConfig();
  }
}

for (const f of CPU_JSON_FILES) {
  fetch('./' + f)
    .then(r => r.ok ? r.json() : [])
    .then(d => { mergeCpuPresets(d); _cpuLoaded++; if (_cpuLoaded === CPU_JSON_FILES.length) { populateCpuSelect(); tryLoadConfig(); } })
    .catch(() => { _cpuLoaded++; if (_cpuLoaded === CPU_JSON_FILES.length) { populateCpuSelect(); tryLoadConfig(); } });
}

for (const f of GPU_JSON_FILES) {
  fetch('./' + f)
    .then(r => r.ok ? r.json() : [])
    .then(d => { mergeGpuPresets(d); _gpuLoaded++; if (_gpuLoaded === GPU_JSON_FILES.length) { populateGpuSelect(); tryLoadConfig(); } })
    .catch(() => { _gpuLoaded++; if (_gpuLoaded === GPU_JSON_FILES.length) { populateGpuSelect(); tryLoadConfig(); } });
}

const $ = (s) => document.querySelector(s);

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

function populateGpuSelect() {
  if (ssGpu) { ssGpu.destroy(); ssGpu = null; }
  gpuPresetEl.innerHTML = '';
  gpuPresetEl.appendChild(new Option('Custom', 'custom'));
  const byVendor = partitionByGroup(getGpuPresets());
  const fmt = g => `${g.name} \u2014 ${g.fp16Tflops} TF, ${g.memBwGBps} GB/s${g.vramGB ? `, ${g.vramGB} GiB` : ''}`;
  for (const [vendor, all] of Object.entries(byVendor)) {
    const desktop = all.filter(g => (!g.mobile && !g.server) || g.desktop);
    const mobile = all.filter(g => g.mobile);
    const server = all.filter(g => g.server);
    addOptgroup(gpuPresetEl, vendor, desktop, fmt);
    addOptgroup(gpuPresetEl, `${vendor} (mobile)`, mobile, fmt);
    addOptgroup(gpuPresetEl, `${vendor} (server)`, server, fmt);
  }
  ssGpu = new SlimSelect({ select: '#gpuPreset', settings: { showSearch: true, searchPlaceholder: 'Search GPU...' }, events: { searchFilter: allWordsSearchFilter } });
  allowSpaceInSearch(ssGpu);
}

function populateCpuSelect() {
  if (ssCpu) { ssCpu.destroy(); ssCpu = null; }
  cpuPresetEl.innerHTML = '';
  cpuPresetEl.appendChild(new Option('Custom', 'custom'));
  cpuPresetEl.appendChild(new Option(UNIFIED_MEMORY_CPU_PRESET.name, UNIFIED_MEMORY_CPU_PRESET.id));
  const byVendor = partitionByGroup(getCpuPresets());
  const fmt = c => (c.fp16Tflops != null && c.defaultRamBwGBps != null)
    ? `${c.name} \u2014 ${c.fp16Tflops} TF, ${c.defaultRamBwGBps} GB/s RAM`
    : c.name;
  for (const [vendor, all] of Object.entries(byVendor)) {
    const desktop = all.filter(c => (!c.mobile && !c.server) || c.desktop);
    const mobile = all.filter(c => c.mobile);
    const server = all.filter(c => c.server);
    addOptgroup(cpuPresetEl, vendor, desktop, fmt);
    addOptgroup(cpuPresetEl, `${vendor} (mobile)`, mobile, fmt);
    addOptgroup(cpuPresetEl, `${vendor} (server)`, server, fmt);
  }
  ssCpu = new SlimSelect({ select: '#cpuPreset', settings: { showSearch: true, searchPlaceholder: 'Search CPU...' }, events: { searchFilter: allWordsSearchFilter } });
  allowSpaceInSearch(ssCpu);
}

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
      hfPath: hfPathEl.value,
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
  if (cfg.mmProjDevice != null) mmProjDeviceEl.value = cfg.mmProjDevice;
  if (cfg.kvTypeK != null && kvTypeKEl.querySelector(`option[value="${cfg.kvTypeK}"]`)) {
    kvTypeKEl.value = cfg.kvTypeK;
  }
  if (cfg.kvTypeV != null && kvTypeVEl.querySelector(`option[value="${cfg.kvTypeV}"]`)) {
    kvTypeVEl.value = cfg.kvTypeV;
  }
  if (cfg.gpuPreset != null && gpuPresetEl.querySelector(`option[value="${cfg.gpuPreset}"]`)) {
    if (ssGpu) ssGpu.setSelected(cfg.gpuPreset);
    else gpuPresetEl.value = cfg.gpuPreset;
  }
  if (cfg.cpuPreset != null && cpuPresetEl.querySelector(`option[value="${cfg.cpuPreset}"]`)) {
    if (ssCpu) ssCpu.setSelected(cfg.cpuPreset);
    else cpuPresetEl.value = cfg.cpuPreset;
  }
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
  mmProjDeviceWrap.style.display = 'none';
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

let currentGGUFUrl = null;
let currentMetadata = null;
let currentTensorInfos = null;
let currentMmProjUrl = null;
let currentMmProjMetadata = null;
let currentMmProjTensorInfos = null;
let currentMmProjInfo = null;

function resetMmProjState() {
  currentMmProjUrl = null;
  currentMmProjMetadata = null;
  currentMmProjTensorInfos = null;
  currentMmProjInfo = null;
  mmProjDeviceWrap.style.display = 'none';
}

async function doParseGGUF(url) {
  loadingEl.classList.add('visible');
  loadingText.textContent = 'Parsing GGUF metadata...';
  readyState.classList.add('hidden');

  try {
    const result = await parseGGUF(url);
    currentMetadata = result.metadata;
    currentTensorInfos = result.tensorInfos;

    loadingEl.classList.remove('visible');
    resultsEl.classList.add('visible');
    renderResults();
  } catch (err) {
    loadingEl.classList.remove('visible');
    readyState.classList.remove('hidden');
    resultsEl.classList.remove('visible');
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
  resolveBtn.disabled = true;

  try {
    const result = await resolveHFModel(path);

    loadingEl.classList.remove('visible');

    if (!result.url) {
      modelSelect.innerHTML = '';
      for (const f of result.ggufFiles) {
        const opt = document.createElement('option');
        opt.value = f;
        opt.textContent = f;
        modelSelect.appendChild(opt);
      }
      modelSelectWrap.classList.add('visible');
      currentGGUFUrl = buildResolveUrl(path, modelSelect.value);
    } else {
      currentGGUFUrl = result.url;
    }

    mmProjSelect.innerHTML = '<option value="">None</option>';
    if (result.mmProjFiles && result.mmProjFiles.length) {
      for (const f of result.mmProjFiles) {
        const opt = document.createElement('option');
        opt.value = f;
        opt.textContent = f;
        mmProjSelect.appendChild(opt);
      }
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
  let repo = repoPath.trim();
  const urlMatch = repo.match(/^https?:\/\/huggingface\.co\/([^/?#]+\/[^/?#]+)/i);
  if (urlMatch) repo = urlMatch[1];
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
  const modelName = currentMetadata['general.name'] || currentMetadata['general.basename'] || arch;
  const archCategories = handler.categories.join(', ');

  if (ctx_len && ctx_len > 0) {
    contextLenEl.max = ctx_len;
    $('#ctxMaxLabel').textContent = `(max ${formatElements(BigInt(ctx_len))})`;
  } else {
    $('#ctxMaxLabel').textContent = '';
  }

  archBadge.innerHTML = isMoe
    ? '<span class="status-badge moe">MoE</span>'
    : '<span class="status-badge dense">Dense</span>';

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
  addInfo('Layers', n_layer);
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

function renderMemoryPanel({ weights, moe, kv, acts, memBreakdown, mmProjBytes, mmProjDevice, cpuMoe, nCpuMoe, vramBytes, ramBytes }) {
  const totalBytes = vramBytes + ramBytes;
  const vramPct = (b) => vramBytes > 0 ? `${(b / vramBytes * 100).toFixed(1)}%` : '0%';

  let nonMoEWeightBytes, vramExpertBytes = 0, vramRouterSharedBytes = 0;
  if (moe) {
    nonMoEWeightBytes = weights.total - moe.expertWeightBytes - moe.routerBytes - moe.sharedBytes;
    vramExpertBytes = memBreakdown.vramWeightBytes - nonMoEWeightBytes - moe.routerBytes - moe.sharedBytes;
    vramRouterSharedBytes = moe.routerBytes + moe.sharedBytes;
  } else {
    nonMoEWeightBytes = weights.total;
  }

  $('#vramSize').textContent = formatBytes(vramBytes);

  if (moe) {
    $('#vramWeightsRow .label').textContent = 'Attention + embedding weights';
    $('#vramActiveExpertRow').style.display = '';
    const expertLabel = cpuMoe ? 'Expert weights (in VRAM)' : (nCpuMoe > 0 ? `Experts in VRAM (layers ${nCpuMoe}+)` : `All experts (${moe.expertCount})`);
    $('#vramActiveExpertLabel').textContent = expertLabel;
    $('#vramActiveExpertSize').textContent = `${formatBytes(vramExpertBytes)} (${vramPct(vramExpertBytes)})`;
    if (vramRouterSharedBytes > 0) {
      $('#vramRouterRow').style.display = '';
      $('#vramRouterSize').textContent = `${formatBytes(vramRouterSharedBytes)} (${vramPct(vramRouterSharedBytes)})`;
    } else {
      $('#vramRouterRow').style.display = 'none';
    }
  } else {
    $('#vramWeightsRow .label').textContent = 'Weights';
    $('#vramActiveExpertRow').style.display = 'none';
    $('#vramRouterRow').style.display = 'none';
  }
  $('#vramWeightsSize').textContent = `${formatBytes(nonMoEWeightBytes)} (${vramPct(nonMoEWeightBytes)})`;
  const kvOnlyBytes = kv.bytesK + kv.bytesV;
  $('#vramKVSize').textContent = `${formatBytes(kvOnlyBytes)} (${vramPct(kvOnlyBytes)})`;
  if (kv.bytesRecurrent > 0) {
    $('#vramRecurrentRow').style.display = '';
    $('#vramRecurrentSize').textContent = `${formatBytes(kv.bytesRecurrent)} (${vramPct(kv.bytesRecurrent)})`;
  } else {
    $('#vramRecurrentRow').style.display = 'none';
  }
  $('#vramActSize').textContent = `${formatBytes(acts.totalBytes)} (${vramPct(acts.totalBytes)})`;
  if (currentMmProjInfo && mmProjDevice === 'vram') {
    $('#vramMmProjRow').style.display = '';
    $('#vramMmProjSize').textContent = `${formatBytes(mmProjBytes)} (${vramPct(mmProjBytes)})`;
  } else {
    $('#vramMmProjRow').style.display = 'none';
  }

  $('#ramSize').textContent = ramBytes > 0 ? formatBytes(ramBytes) : 'None';
  if (moe && ramBytes > 0) {
    $('#ramInactiveRow').style.display = '';
    const expertInRam = memBreakdown.ramExpertBytes;
    if (cpuMoe) {
      $('#ramInactiveLabel').textContent = `All experts (${moe.expertCount})`;
    } else if (nCpuMoe > 0) {
      $('#ramInactiveLabel').textContent = `Experts in RAM (layers 0\u2013${nCpuMoe - 1})`;
    } else {
      $('#ramInactiveLabel').textContent = 'Inactive experts';
    }
    $('#ramInactiveSize').textContent = `${formatBytes(expertInRam)} (${ramBytes > 0 ? (expertInRam / ramBytes * 100).toFixed(1) : '0'}%)`;
  } else {
    $('#ramInactiveRow').style.display = 'none';
  }
  if (currentMmProjInfo && mmProjDevice === 'ram') {
    $('#ramMmProjRow').style.display = '';
    const ramPct = ramBytes > 0 ? (mmProjBytes / ramBytes * 100).toFixed(1) : '0';
    $('#ramMmProjSize').textContent = `${formatBytes(mmProjBytes)} (${ramPct}%)`;
  } else {
    $('#ramMmProjRow').style.display = 'none';
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

function renderFitCheck({ vramGB, ramGB, acts, layerFootprint, mmProjDevice, cpuMoe, nCpuMoe, nglOverride, unifiedMemory }) {
  const fitPanel = $('#fitCheckPanel');
  const showVramBar = vramGB > 0;

  if (!showVramBar) {
    fitPanel.classList.add('hidden');
    return;
  }

  fitPanel.classList.remove('hidden');

  const mmProjActBytes = currentMmProjInfo ? (currentMmProjInfo.weightBytes + (currentMmProjInfo.perImageActBytes || 0)) : 0;
  const reservedBytes = acts.totalBytes + (mmProjDevice !== 'ram' ? mmProjActBytes : 0);
  const actual = calcActualMemory({
    vramBytes: vramGB * (1024 ** 3),
    footprint: layerFootprint,
    activationBytes: reservedBytes,
    nLayerOverride: nglOverride,
    cpuMoe,
    nCpuMoe,
    unifiedMemory: !!unifiedMemory,
  });

  $('#vramFitSection').style.display = '';
  const vramAvailBytes = vramGB * (1024 ** 3);
  const vramUsagePct = (actual.actualVram / vramAvailBytes * 100);
  const clampedVramPct = Math.min(vramUsagePct, 100);

  const vramBar = $('#vramBar');
  const vramBarText = $('#vramBarText');
  const vramStatus = $('#vramStatus');

  vramBar.style.width = `${clampedVramPct}%`;

  const layerSplitStr = actual.nHybridLayers > 0
    ? `${actual.nGpuLayers} GPU / ${actual.nHybridLayers} hybrid / ${actual.nCpuLayers} CPU`
    : actual.nCpuLayers > 0
      ? `${actual.nGpuLayers} GPU / ${actual.nCpuLayers} CPU`
      : `${actual.nGpuLayers} GPU (full offload)`;

  if (vramUsagePct > 100) {
    vramBar.className = 'vram-bar red';
    vramStatus.className = 'vram-status red';
    vramStatus.textContent = `\u2717 Overflow \u2014 ${formatBytes(actual.actualVram)} needed, ${vramGB} GiB available \u2014 ${layerSplitStr}`;
  } else if (actual.nCpuLayers === 0 && (actual.nHybridLayers === 0 || cpuMoe || nCpuMoe > 0)) {
    vramBar.className = 'vram-bar green';
    vramStatus.className = 'vram-status green';
    vramStatus.textContent = `\u2713 Fits \u2014 ${formatBytes(actual.actualVram)} of ${vramGB} GiB (${vramUsagePct.toFixed(0)}%) \u2014 ${layerSplitStr}`;
  } else {
    vramBar.className = 'vram-bar yellow';
    vramStatus.className = 'vram-status yellow';
    vramStatus.textContent = `\u26A0 Partial offload \u2014 ${formatBytes(actual.actualVram)} of ${vramGB} GiB (${vramUsagePct.toFixed(0)}%) \u2014 ${layerSplitStr}`;
  }

  vramBarText.textContent = `${vramUsagePct.toFixed(1)}%`;
  $('#vramFitLabel').textContent = `${formatBytes(actual.actualVram)} / ${vramGB} GiB`;

  const actualRamBytes = actual.actualRam + (mmProjDevice === 'ram' ? mmProjActBytes : 0);
  const showRamBar = ramGB > 0 && actualRamBytes > 0;
  if (showRamBar) {
    $('#ramFitSection').style.display = '';
    const ramUsagePct = (actualRamBytes / (ramGB * (1024 ** 3)) * 100);
    const clampedRamPct = Math.min(ramUsagePct, 100);

    const ramBar = $('#ramBar');
    const ramBarText = $('#ramBarText');
    const ramStatus = $('#ramStatus');

    ramBar.style.width = `${clampedRamPct}%`;

    if (ramUsagePct <= 80) {
      ramBar.className = 'vram-bar green';
      ramStatus.className = 'vram-status green';
      ramStatus.textContent = `\u2713 RAM usage \u2014 ${formatBytes(actualRamBytes)} of ${ramGB} GiB (${ramUsagePct.toFixed(0)}% used)`;
    } else if (ramUsagePct <= 100) {
      ramBar.className = 'vram-bar yellow';
      ramStatus.className = 'vram-status yellow';
      ramStatus.textContent = `\u26A0 RAM usage \u2014 ${formatBytes(actualRamBytes)} of ${ramGB} GiB (${ramUsagePct.toFixed(0)}% used)`;
    } else {
      ramBar.className = 'vram-bar red';
      ramStatus.className = 'vram-status red';
      ramStatus.textContent = `\u2717 RAM overflow \u2014 ${formatBytes(actualRamBytes)} needed, ${ramGB} GiB available`;
    }

    ramBarText.textContent = `${ramUsagePct.toFixed(1)}%`;
    $('#ramFitLabel').textContent = `${formatBytes(actualRamBytes)} / ${ramGB} GiB`;
  } else {
    $('#ramFitSection').style.display = 'none';
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
  const perf = estimatePerformance({
    metadata: currentMetadata, tensorInfos: currentTensorInfos,
    ctx: ctxSize, batchSize,
    kv, moe, activations: acts, mmproj: currentMmProjInfo,
    device: {
      gpu: {
        flopsFp16Tflops: gpuFlopsV,
        bwGBps: gpuBwV,
        vramBytes: vramGB > 0 ? vramGB * (1024 ** 3) : 0,
      },
      cpu: hasCpuPerf ? { flopsFp16Tflops: cpuFlopsV, bwGBps: ramBwV } : null,
      nGpuLayers: nglOverride,
      mmprojOnGpu: mmProjDevice !== 'ram',
      cpuMoe,
      nCpuMoe,
      unifiedMemory: !!unifiedMemory,
    },
  });
  $('#perfDecode').textContent = `${perf.decodeTPS.toFixed(1)} tok/s`;
  $('#perfPrefill').textContent = `${perf.prefillTPS.toFixed(0)} tok/s`;
  $('#perfTtft').textContent = perf.ttftSec < 1
    ? `${(perf.ttftSec * 1000).toFixed(0)} ms`
    : `${perf.ttftSec.toFixed(2)} s`;
  const nHybrid = perf.nHybridLayers || 0;
  $('#perfSplit').textContent = nHybrid > 0
    ? `${perf.nGpuLayers} / ${nHybrid} / ${perf.nCpuLayers}${perf.autoSplit ? ' (auto)' : ''}`
    : `${perf.nGpuLayers} / ${perf.nCpuLayers}${perf.autoSplit ? ' (auto)' : ''}`;
  $('#perfGpuLayer').textContent = perf.nGpuLayers > 0
    ? `${perf.perLayerMs.gpu.toFixed(3)} ms (${perf.bottleneck.gpu || '-'})`
    : '\u2014';
  const hybridRow = $('#perfHybridRow');
  if (nHybrid > 0) {
    hybridRow.classList.remove('hidden');
    $('#perfHybridLayer').textContent = hasCpuPerf && !cpuFallback
      ? `${perf.perLayerMs.hybrid.toFixed(2)} ms (${perf.bottleneck.cpu || '-'})`
      : cpuFallback
        ? `${perf.perLayerMs.hybrid.toFixed(2)} ms (${perf.bottleneck.cpu || '-'}, est. from ${cpuFallback.name})`
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
  $('#perfHardware').textContent = `GPU: ${gpuLabel.replace(/ \u2014.*$/, '')} \u00B7 CPU: ${cpuLabel.replace(/ \u2014.*$/, '')}`;
}

function renderResults() {
  if (!currentMetadata || !currentTensorInfos) return;

  const arch = getModelArch(currentMetadata);
  const handler = getArchHandler(arch);

  const modelCtxLen = getMeta(currentMetadata, `${arch}.context_length`);
  let ctxSize = parseInt(contextLenEl.value, 10) || 4096;
  if (modelCtxLen > 0 && ctxSize > modelCtxLen) {
    ctxSize = modelCtxLen;
    contextLenEl.value = modelCtxLen;
  }
  const batchSize = parseInt(batchSizeEl.value, 10) || 1;
  const kvTypeK = isNaN(kvTypeKEl.value) ? kvTypeKEl.value : parseInt(kvTypeKEl.value, 10);
  const kvTypeV = isNaN(kvTypeVEl.value) ? kvTypeVEl.value : parseInt(kvTypeVEl.value, 10);
  const vramGB = parseFloat(vramEl.value) || 0;
  const ramGB = parseFloat(ramEl.value) || 0;
  const nglRaw = nglOverrideEl.value.trim();
  const nglOverride = (nglRaw === '' || nglRaw.toLowerCase() === 'auto')
    ? 'auto'
    : (Number.isFinite(parseInt(nglRaw, 10)) ? parseInt(nglRaw, 10) : 'auto');

  const weights = calcWeightSize(currentTensorInfos);
  const kv = calcKVCache(currentMetadata, ctxSize, kvTypeK, kvTypeV);
  const acts = calcActivations(currentMetadata, batchSize);
  const moe = calcMoEInfo(currentMetadata, currentTensorInfos);
  const cpuMoe = cpuMoeEl.checked;
  const nCpuMoe = parseInt(nCpuMoeEl.value, 10) || 0;
  const layerFootprint = calcPerLayerFootprint(currentMetadata, currentTensorInfos, kv, moe);
  const memBreakdown = calcMemoryBreakdown({
    weights, moe, kv, activations: acts,
    footprint: layerFootprint,
    cpuMoe, nCpuMoe,
  });

  const mmProjDevice = mmProjDeviceEl.value;
  let vramBytes = memBreakdown.vramBytes;
  let ramBytes = memBreakdown.ramBytes;
  let mmProjBytes = 0;
  if (currentMmProjInfo) {
    mmProjBytes = currentMmProjInfo.weightBytes + (currentMmProjInfo.perImageActBytes || 0);
    if (mmProjDevice === 'ram') ramBytes += mmProjBytes;
    else vramBytes += mmProjBytes;
  }

  const isMoe = (moe !== null);
  const isMla = handler.categories.includes('mla');
  const vocab = getMeta(currentMetadata, `${arch}.vocab_size`);

  renderModelInfo(arch, handler, isMoe, isMla, moe, modelCtxLen, vocab);
  renderMoeSection(moe, cpuMoe, nCpuMoe);
  renderWeightsTable(weights);
  renderKvCache(kv, kvTypeK, kvTypeV, isMla, arch);
  $('#actSize').textContent = formatBytes(acts.totalBytes);
  renderMemoryPanel({ weights, moe, kv, acts, memBreakdown, mmProjBytes, mmProjDevice, cpuMoe, nCpuMoe, vramBytes, ramBytes });
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
    currentMmProjMetadata = metadata;
    currentMmProjTensorInfos = tensorInfos;
    currentMmProjInfo = calcMmProj(metadata, tensorInfos);
    mmProjDeviceWrap.style.display = '';
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
  currentMmProjUrl = buildResolveUrl(path, filename);
  loadingEl.classList.add('visible');
  loadingText.textContent = 'Parsing mmproj metadata...';
  await doParseMmProj(currentMmProjUrl);
  loadingEl.classList.remove('visible');
  if (currentMetadata) renderResults();
});

hfPathEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') resolveBtn.click();
});
