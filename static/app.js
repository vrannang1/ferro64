'use strict';

const BOOT = JSON.parse(document.getElementById('boot').textContent || '{}');
const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const icon = (name, cls = '') => `<svg aria-hidden="true"${cls ? ` class="${cls}"` : ''}><use href="#i-${name}"/></svg>`;
const store = {
  get(key) { try { return localStorage.getItem(`ferro64.${key}`); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(`ferro64.${key}`, value); } catch { /* storage is optional */ } },
};
const VIEWS = ['chat', 'documents', 'images', 'video', 'agents', 'code', 'settings', 'api'];
const EFFORTS = (BOOT.efforts || []).map((e) => e.value);
const TEMPERATURES = (BOOT.temperatures || []).map((t) => t.value);
const IMAGE_NOTES = {
  'flux2-klein': 'flux.2 klein 4b · 1024 px · about a minute · also edits',
  'z-image-turbo': 'z-image-turbo · 1024 px · about 2.5 minutes',
};

const state = {
  view: 'chat',
  models: [],
  conversations: [],
  conversation: null,
  busy: false,
  controller: null,
  attachment: null,
  model: store.get('model'),
  thinking: store.get('thinking') === '1',
  web: store.get('web') === '1',
  effort: EFFORTS.includes(store.get('effort')) ? store.get('effort') : (BOOT.default_effort || 'medium'),
  temperature: TEMPERATURES.includes(store.get('temperature')) ? store.get('temperature') : (BOOT.default_temperature || 'balanced'),
  status: null,
  store: null,
  loading: null,
  ocr: {data: null, busy: false},
  imageBusy: false,
  aspect: '1:1',
  imageModel: store.get('imageModel') || 'z-image-turbo',
  // The image being edited: {id, url, name} for the gallery or {data, url, name} for a photo.
  edit: null,
  gallery: [],
  tasks: [],
  agentBusy: false,
  agentEffort: 'medium',
  agentThinking: false,
};

/* ---------- utilities ---------- */

async function api(path, options = {}) {
  const {timeout = 20000, ...request} = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, {...request, headers: {'Content-Type': 'application/json', ...request.headers}, signal: controller.signal});
    const text = await response.text();
    let data = null;
    if (text) { try { data = JSON.parse(text); } catch { data = text; } }
    if (!response.ok) throw new Error((data && typeof data === 'object' ? data.error : data) || `Request failed (${response.status})`);
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The request timed out. The model may still be working; check history before retrying.');
    if (error instanceof TypeError) throw new Error('Cannot reach Ferro64. Check that the server is running.');
    throw error;
  } finally { clearTimeout(timer); }
}
const post = (path, body, timeout = 900000) => api(path, {method: 'POST', body: JSON.stringify(body), timeout});

function toast(message, error = false) {
  const item = document.createElement('div');
  item.className = `toast${error ? ' is-error' : ''}`;
  item.innerHTML = `<span>${esc(message)}</span><button class="icon-btn" aria-label="Dismiss">${icon('x')}</button>`;
  item.querySelector('button').onclick = () => item.remove();
  $('toasts').append(item);
  setTimeout(() => item.remove(), error ? 12000 : 4000);
}
const fail = (error) => toast(error?.message || String(error), true);

function duration(ms) {
  if (ms == null) return '';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`;
  return `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`;
}
function bytes(n) {
  if (!n) return '';
  return n >= 1e9 ? `${(n / 1e9).toFixed(1)} gb` : `${Math.round(n / 1e6)} mb`;
}
function when(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  const today = new Date();
  return date.toDateString() === today.toDateString()
    ? date.toLocaleTimeString(undefined, {hour: 'numeric', minute: '2-digit'})
    : date.toLocaleDateString(undefined, {month: 'short', day: 'numeric'});
}
function shortModel(id) { return String(id || 'model').split('/').pop().replace(':latest', ''); }
function localURL(url, allowData = false) {
  if (typeof url !== 'string') return '';
  if (/^\/static\/[a-zA-Z0-9_./-]+$/.test(url) && !url.includes('..')) return url;
  if (allowData && /^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(url)) return url;
  return '';
}
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast('Copied'); }
  catch {
    const field = Object.assign(document.createElement('textarea'), {value: text});
    field.style.cssText = 'position:fixed;opacity:0';
    document.body.append(field); field.select();
    const ok = document.execCommand('copy'); field.remove();
    toast(ok ? 'Copied' : 'Clipboard is unavailable in this browser.', !ok);
  }
}
function confirmDialog(title, text) {
  return new Promise((resolve) => {
    const dialog = $('confirm');
    if (dialog.open) return resolve(false);
    $('confirm-title').textContent = title;
    $('confirm-text').textContent = text || 'This removes it from this machine.';
    dialog.returnValue = 'cancel';
    dialog.addEventListener('close', () => resolve(dialog.returnValue === 'confirm'), {once: true});
    dialog.showModal();
  });
}
function fileToData(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('Could not read this file.'));
    reader.readAsDataURL(file);
  });
}
function checkFile(file) {
  if (!file) return false;
  if (file.size > 18 * 1024 * 1024) throw new Error('Choose a file smaller than 18 MB.');
  if (!['image/png', 'image/jpeg', 'image/webp', 'application/pdf'].includes(file.type)) throw new Error('Choose a PNG, JPEG or WebP image, or a PDF.');
  return true;
}
const SHEET_TYPES = {csv: 'text/csv', tsv: 'text/tab-separated-values', xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', xlsm: 'application/vnd.ms-excel.sheet.macroenabled.12', xls: 'application/vnd.ms-excel', ods: 'application/vnd.oasis.opendocument.spreadsheet'};
/** A spreadsheet's type, going by its name: browsers report CSV differently. */
function sheetType(name) { return SHEET_TYPES[String(name || '').split('.').pop().toLowerCase()] || ''; }
function isSheetURL(url) { return typeof url === 'string' && /^\/static\/uploads\/[^/]+\.(csv|tsv|xlsx|xlsm|xls|ods)$/i.test(url); }
function setRadio(group, value, attr) {
  for (const button of group.querySelectorAll('[role="radio"]')) button.setAttribute('aria-checked', String(button.dataset[attr] === value));
}

/* ---------- markdown ----------
   Only escaped text and this fixed set of generated tags reach innerHTML.
   Links are limited to http(s) and open in a new tab without referrer. */

function inline(text) {
  const codes = [];
  let out = esc(text).replace(/`([^`]+)`/g, (_, code) => { codes.push(code); return `\u0000${codes.length - 1}\u0000`; });
  out = out
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/(^|[\s(])(https?:\/\/[^\s<)]+[^\s<).,;:!?])/g, '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_]+)__/g, '<strong>$1</strong>')
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\*)/g, '$1<em>$2</em>')
    .replace(/~~([^~]+)~~/g, '<del>$1</del>');
  return out.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[Number(i)]}</code>`);
}
function markdown(source) {
  const lines = String(source || '').replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let i = 0;
  const isRule = (line) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
  const cells = (line) => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
  const listRe = /^(\s*)([-*+]|\d+[.)])\s+/;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    const fence = /^\s*(`{3,}|~{3,})\s*([\w+#.-]*)/.exec(line);
    if (fence) {
      // Closes on a bare fence at least as long. A fence with a language inside
      // opens a nested sample, so a markdown block can hold code blocks.
      const marker = fence[1][0];
      const closes = new RegExp(`^\\s*\\${marker}{${fence[1].length},}\\s*$`);
      const opens = new RegExp(`^\\s*\\${marker}{${fence[1].length},}\\s*[\\w+#.-]+\\s*$`);
      const code = []; i++;
      let depth = 0;
      while (i < lines.length) {
        if (opens.test(lines[i])) depth++;
        else if (closes.test(lines[i])) { if (depth === 0) break; depth--; }
        code.push(lines[i++]);
      }
      i++;
      out.push(`<div class="code"><div class="code__bar"><span class="readout">${esc(fence[2] || 'text')}</span><button class="icon-btn" data-copy-code aria-label="Copy code" title="Copy code">${icon('copy')}</button></div><pre><code>${esc(code.join('\n'))}</code></pre></div>`);
      continue;
    }
    if (i + 1 < lines.length && line.includes('|') && isRule(lines[i + 1])) {
      const head = cells(line); i += 2; const rows = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) rows.push(cells(lines[i++]));
      out.push(`<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${head.map((_, k) => `<td>${inline(r[k] || '')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
      continue;
    }
    const heading = /^(#{1,6})\s+(.+?)\s*#*$/.exec(line);
    if (heading) { const level = Math.min(heading[1].length, 4); out.push(`<h${level}>${inline(heading[2])}</h${level}>`); i++; continue; }
    if (listRe.test(line)) {
      const ordered = /\d/.test(listRe.exec(line)[2]);
      const items = [];
      while (i < lines.length && (listRe.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length))) {
        if (listRe.test(lines[i])) items.push(lines[i].replace(listRe, ''));
        else items[items.length - 1] += `\n${lines[i].trim()}`;
        i++;
      }
      const tag = ordered ? 'ol' : 'ul';
      out.push(`<${tag}>${items.map((item) => `<li>${inline(item).replace(/\n/g, '<br>')}</li>`).join('')}</${tag}>`);
      continue;
    }
    if (/^>\s?/.test(line)) {
      const quote = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) quote.push(lines[i++].replace(/^>\s?/, ''));
      out.push(`<blockquote>${quote.map(inline).join('<br>')}</blockquote>`);
      continue;
    }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { out.push('<hr>'); i++; continue; }
    const para = [inline(line)]; i++;
    while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|\s*```|\s*~~~|>)/.test(lines[i]) && !listRe.test(lines[i]) && !(i + 1 < lines.length && isRule(lines[i + 1]))) para.push(inline(lines[i++]));
    out.push(`<p>${para.join('<br>')}</p>`);
  }
  return out.join('');
}
function bindCodeCopy(root) {
  for (const button of root.querySelectorAll('[data-copy-code]')) {
    button.onclick = () => copyText(button.closest('.code').querySelector('code').textContent);
  }
}

/* ---------- views & rail ---------- */

function setView(view, push = true) {
  view = String(view || '').split('/')[0];
  if (view === 'models') view = 'settings';
  if (!VIEWS.includes(view) || (view === 'api' && !$('view-api'))) view = 'chat';
  state.view = view;
  for (const section of document.querySelectorAll('.view')) section.hidden = section.id !== `view-${view}`;
  for (const button of document.querySelectorAll('.nav__item')) {
    if (button.dataset.view === view) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
  }
  document.title = view === 'chat' ? 'Ferro64' : `${view} · Ferro64`;
  closeRail();
  const hash = view === 'chat' && state.conversation ? `#chat/${state.conversation}` : `#${view}`;
  if (push && location.hash !== hash) history.replaceState(null, '', hash);
  if (view === 'images' && !state.gallery.length) loadGallery().catch(fail);
  if (view === 'agents') loadTasks().catch(fail);
  if (view === 'settings') { loadModelStore(); loadSettings(); renderModelsView(); }
  if (view === 'api' && typeof openHostedView === 'function') openHostedView();
  if (view === 'code' && typeof openCodeView === 'function') openCodeView();
  if (view === 'video' && typeof openVideoView === 'function') openVideoView();
  if (view === 'chat') setTimeout(() => $('prompt').focus(), 0);
}
function openRail() { $('rail').classList.add('is-open'); $('scrim').hidden = false; }
function closeRail() { $('rail').classList.remove('is-open'); $('scrim').hidden = true; }

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  if (typeof refreshTerminalTheme === 'function') refreshTerminalTheme();
  $('theme-toggle').innerHTML = icon(theme === 'day' ? 'moon' : 'sun');
}
function initialTheme() {
  const saved = store.get('theme');
  if (saved === 'day' || saved === 'night') return saved;
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'day' : 'night';
}

/* ---------- status ---------- */

function showBanner(text, action, handler) {
  $('banner').hidden = false;
  $('banner-text').innerHTML = text;
  $('banner-action').textContent = action;
  $('banner-action').onclick = handler;
}
async function refreshStatus() {
  try {
    const status = await api('/api/status', {timeout: 8000});
    state.status = status;
    const resident = status.active_models || [];
    const online = status.runtime_online;
    $('runtime-dot').className = `dot ${state.loading ? 'is-busy' : online ? (resident.length ? 'is-live' : '') : 'is-off'}`;
    $('runtime-engine').textContent = status.engine || BOOT.engine;
    const engines = {sdcpp: 'stable-diffusion.cpp', mlx: 'mflux', ollama: 'ollama'};
    $('image-backend').textContent = `${engines[status.image_backend] || status.image_backend} · text models unload first`;
    renderImageModels();
    $('runtime-model').textContent = state.loading ? `loading ${shortModel(state.loading)}…` : resident.length ? `${resident.map(shortModel).join(', ')} · resident` : 'nothing resident';
    for (const model of state.models) model.is_resident = resident.includes(model.id);
    if (!online && status.runtime_error) {
      showBanner(esc(status.runtime_error), state.view === 'settings' ? 'refresh' : 'open settings', () => {
        if (state.view !== 'settings') setView('settings'); else refreshAll();
      });
    } else {
      $('banner').hidden = true;
    }
    renderPicker();
    if (state.view === 'settings') renderModelsView();
  } catch (error) {
    $('runtime-dot').className = 'dot is-off';
    $('runtime-model').textContent = 'server unreachable';
    showBanner('Cannot reach the Ferro64 server. Start it with <code>cargo run --release</code>.', 'retry', refreshAll);
  }
}

/* ---------- models & picker ---------- */

const isChat = (m) => m.id !== 'qwen3:4b' && (m.capabilities || []).includes('chat') && !m.is_image_generator && !(m.capabilities || []).includes('embedding');
const chatModels = () => state.models.filter((m) => isChat(m) && m.installed !== false);
const current = () => state.models.find((m) => m.id === state.model);

function thinkingNote(model) {
  return {
    none: 'no reasoning mode',
    always: 'always reasons',
    toggle: 'reasoning on demand',
    effort: 'reasoning with effort levels',
  }[model?.thinking || 'none'];
}
function modelSpecs(model) {
  return [model.parameter_size && model.parameter_size !== 'GGUF' && model.parameter_size !== 'Unknown' ? model.parameter_size.toLowerCase() : '', model.quantization?.toLowerCase(), bytes(model.size_bytes), model.context_length ? `${Math.round(model.context_length / 1024)}k ctx` : ''].filter(Boolean);
}
function modelTags(model) {
  const tags = [];
  if ((model.capabilities || []).includes('vision')) tags.push('vision');
  if (model.thinking === 'always') tags.push('thinks');
  if (model.thinking === 'toggle' || model.thinking === 'effort') tags.push('thinking');
  return tags;
}

async function loadModels() {
  const models = await api('/api/models', {timeout: 60000});
  state.models = Array.isArray(models) ? models : [];
  const chats = chatModels();
  if (!chats.some((m) => m.id === state.model)) {
    // The strongest general model that is installed, in order of preference.
    const preferred = ['qwen3.5:35b-a3b', 'glm-4.7-flash', 'qwen3.5:9b'];
    state.model = (preferred.map((id) => chats.find((m) => m.id === id)).find(Boolean) || chats.find((m) => m.is_resident) || chats[0])?.id || null;
  }
  renderPicker();
  fillAgentSelects();
  if (typeof renderCodeModels === 'function') renderCodeModels();
  if (state.view === 'settings') renderModelsView();
}

function renderPicker() {
  const model = current();
  $('picker-name').textContent = model ? (model.name || shortModel(model.id)) : chatModels().length ? 'choose a model' : 'no models yet';
  $('picker-dot').className = `dot ${state.loading && state.loading === state.model ? 'is-busy' : model?.is_resident ? 'is-live' : ''}`;
  $('picker-button').title = model ? `${model.id}${model.is_resident ? ' · resident' : ''}` : 'Choose a chat model';
  const toggle = $('thinking');
  const mode = model?.thinking || 'none';
  const forced = mode === 'always';
  const unsupported = mode === 'none';
  toggle.disabled = state.busy || forced || unsupported;
  toggle.setAttribute('aria-checked', String(forced || (!unsupported && state.thinking)));
  toggle.title = forced ? `${model.name} always reasons before answering` : unsupported ? `${model?.name || 'This model'} has no reasoning mode; effort still sets length and care` : 'Let the model reason before answering';
  setRadio($('effort'), state.effort, 'effort');
  for (const button of $('effort').querySelectorAll('button')) button.disabled = state.busy;
  setRadio($('temperature'), state.temperature, 'temperature');
  $('web').setAttribute('aria-checked', String(state.web));
  for (const button of $('temperature').querySelectorAll('button')) button.disabled = state.busy;
  if (!$('picker-menu').hidden) renderPickerMenu();
}

function renderPickerMenu() {
  const menu = $('picker-menu');
  const models = chatModels().sort((a, b) => (b.is_resident - a.is_resident) || (a.name || a.id).localeCompare(b.name || b.id));
  if (!models.length) {
    menu.innerHTML = `<div class="option__empty">No chat models are installed. <a href="#settings" data-goto-models>Download one in settings</a>.</div>`;
    menu.querySelector('[data-goto-models]').onclick = (event) => { event.preventDefault(); closePicker(); setView('settings'); };
    return;
  }
  menu.innerHTML = `<div class="picker__group label">${models.length} on this machine · one in memory at a time</div>` + models.map((m) => `
    <button type="button" class="option" role="option" data-model="${esc(m.id)}" aria-selected="${m.id === state.model}">
      <span class="dot ${m.is_resident ? 'is-live' : ''}" title="${m.is_resident ? 'resident' : 'not loaded'}"></span>
      <span class="option__name">${esc(m.name || shortModel(m.id))}</span>
      <span class="option__tags">${modelTags(m).map((t) => `<span class="tag">${t}</span>`).join('')}</span>
      <span class="option__meta">${esc([m.id, ...modelSpecs(m)].join(' · '))}</span>
    </button>`).join('');
  for (const option of menu.querySelectorAll('[data-model]')) option.onclick = () => { closePicker(); selectModel(option.dataset.model); };
}
function openPicker() {
  if (state.busy) return;
  $('picker-menu').hidden = false;
  $('picker-button').setAttribute('aria-expanded', 'true');
  renderPickerMenu();
  ($('picker-menu').querySelector('[aria-selected="true"]') || $('picker-menu').querySelector('.option'))?.focus();
}
function closePicker() {
  $('picker-menu').hidden = true;
  $('picker-button').setAttribute('aria-expanded', 'false');
}

async function selectModel(id, {preload = true} = {}) {
  if (state.busy) return toast('Wait for the current reply first.', true);
  state.model = id;
  store.set('model', id);
  renderPicker();
  const model = current();
  if (!preload || !model || model.is_resident || state.status?.keep_alive === '0s') return;
  state.loading = id;
  setChatState(`loading ${shortModel(id)}`);
  renderPicker();
  try {
    const result = await post('/api/models/select', {model: id}, 600000);
    if (result.load_ms) toast(`${model.name || shortModel(id)} loaded in ${duration(result.load_ms)}`);
  } catch (error) { fail(error); }
  finally {
    state.loading = null;
    setChatState(null);
    await refreshStatus();
  }
}

/* ---------- conversations ---------- */

async function loadConversations() {
  state.conversations = await api('/api/chat/conversations');
  renderThreads();
}
function renderThreads() {
  const query = $('thread-search').value.trim().toLowerCase();
  const list = state.conversations.filter((c) => !query || c.title.toLowerCase().includes(query));
  const container = $('thread-list');
  if (!state.conversations.length) { container.innerHTML = '<p class="threads__empty">Conversations are saved here, on this machine.</p>'; return; }
  if (!list.length) { container.innerHTML = '<p class="threads__empty">No match.</p>'; return; }
  container.innerHTML = list.map((c) => `
    <div class="thread${c.id === state.conversation ? ' is-on' : ''}" role="listitem" data-id="${esc(c.id)}">
      <button class="thread__open" data-open ${state.busy ? 'disabled' : ''}><span class="thread__title">${esc(c.title)}</span><span class="thread__meta">${esc(when(c.updated_at))} · ${esc(shortModel(c.model))}</span></button>
      <span class="thread__tools"><button class="icon-btn" data-rename aria-label="Rename ${esc(c.title)}" title="Rename">${icon('pen')}</button><button class="icon-btn" data-delete aria-label="Delete ${esc(c.title)}" title="Delete" ${state.busy ? 'disabled' : ''}>${icon('trash')}</button></span>
    </div>`).join('');
  for (const row of container.querySelectorAll('.thread')) {
    const conversation = state.conversations.find((c) => c.id === row.dataset.id);
    row.querySelector('[data-open]').onclick = () => openConversation(conversation).catch(fail);
    row.querySelector('[data-rename]').onclick = () => startRename(row, conversation);
    row.querySelector('[data-delete]').onclick = async () => {
      if (!await confirmDialog('delete this conversation?', `“${conversation.title}” and its messages will be removed from this machine.`)) return;
      try {
        await api(`/api/chat/conversations/${encodeURIComponent(conversation.id)}`, {method: 'DELETE'});
        if (state.conversation === conversation.id) newChat();
        await loadConversations();
      } catch (error) { fail(error); }
    };
  }
}
function startRename(row, conversation) {
  const input = Object.assign(document.createElement('input'), {className: 'thread__rename', value: conversation.title, maxLength: 120});
  input.setAttribute('aria-label', 'Conversation title');
  row.replaceChildren(input);
  input.focus(); input.select();
  let done = false;
  const finish = async (save) => {
    if (done) return; done = true;
    const title = input.value.trim();
    if (save && title && title !== conversation.title) {
      try {
        await api(`/api/chat/conversations/${encodeURIComponent(conversation.id)}`, {method: 'PATCH', body: JSON.stringify({title})});
        conversation.title = title;
        if (state.conversation === conversation.id) $('chat-title').textContent = title;
      } catch (error) { fail(error); }
    }
    renderThreads();
  };
  input.onkeydown = (event) => { if (event.key === 'Enter') finish(true); if (event.key === 'Escape') finish(false); };
  input.onblur = () => finish(true);
}
async function openConversation(conversation) {
  if (state.busy) return;
  const messages = await api(`/api/chat/conversations/${encodeURIComponent(conversation.id)}`);
  state.conversation = conversation.id;
  $('chat-title').textContent = conversation.title;
  if (chatModels().some((m) => m.id === conversation.model)) { state.model = conversation.model; store.set('model', conversation.model); renderPicker(); }
  clearAttachment();
  const log = $('chat-log');
  log.replaceChildren();
  for (const message of messages) log.append(message.role === 'user' ? userTurn(message) : assistantTurn(message).element);
  if (!messages.length) showHero();
  bindCodeCopy(log);
  renderThreads();
  setView('chat');
  log.scrollTop = log.scrollHeight;
}
function openFromHash() {
  const id = /^#chat\/([\w-]+)$/.exec(location.hash)?.[1];
  const conversation = id && id !== state.conversation && state.conversations.find((c) => c.id === id);
  if (conversation) openConversation(conversation).catch(fail);
}
function newChat() {
  if (state.busy) return;
  state.conversation = null;
  $('chat-title').textContent = 'new conversation';
  $('prompt').value = '';
  autosize();
  clearAttachment();
  showHero();
  renderThreads();
  setView('chat');
}

/* ---------- chat rendering ---------- */

function showHero() {
  const model = current();
  $('chat-log').innerHTML = `
    <div class="hero">
      <h2>what shall we <span class="verb">work</span> on</h2>
      <p class="lead">A private assistant on your own hardware. Nothing you type leaves this machine.</p>
      <div class="hero__grid">
        <button class="suggest" data-prompt="Explain how this works, step by step, with a small example: "><strong>explain something</strong><span>a concept, a paper, a tool</span></button>
        <button class="suggest" data-prompt="Review this code for bugs and readability, then suggest a cleaner version:\n\n"><strong>review code</strong><span>bugs, clarity, a better version</span></button>
        <button class="suggest" data-prompt="Draft a short, friendly message that says: "><strong>draft a message</strong><span>email, note, announcement</span></button>
        <button class="suggest" data-prompt="Help me plan this. Ask me two questions first, then give a step-by-step plan: "><strong>plan a project</strong><span>steps, risks, a first move</span></button>
      </div>
      <p class="readout hero__foot">${model ? `${esc(shortModel(model.id))} on ${esc(BOOT.engine)} · effort ${esc(state.effort)}` : `${esc(BOOT.engine)} · no model selected`}</p>
    </div>`;
  for (const button of $('chat-log').querySelectorAll('[data-prompt]')) button.onclick = () => {
    $('prompt').value = button.dataset.prompt.replace(/\\n/g, '\n');
    autosize(); $('prompt').focus();
  };
}
function splitDocument(content) {
  const match = /^Attached document text:\n([\s\S]*)\n\nRequest: ([\s\S]*)$/.exec(content || '');
  return match ? {doc: match[1], request: match[2]} : {doc: null, request: content};
}
function userTurn(message) {
  const element = document.createElement('article');
  element.className = 'turn turn--user';
  const sheet = isSheetURL(message.image_url);
  const image = sheet ? '' : localURL(message.image_url, true);
  const {doc, request} = splitDocument(message.content);
  element.innerHTML = `${sheet ? `<span class="tag bubble__file">spreadsheet</span>` : ''}${image ? `<img class="bubble__image" src="${esc(image)}" alt="Attachment">` : ''}${doc ? `<details class="reasoning bubble__doc"><summary>${icon('doc')}<span class="grow">document text · ${doc.length.toLocaleString()} chars</span>${icon('chev', 'chev')}</summary><div class="reasoning__body">${esc(doc)}</div></details>` : ''}${request ? `<div class="bubble">${esc(request)}</div>` : ''}`;
  return element;
}
function metaLine(model, stats = {}, content = '') {
  const parts = [`<span class="readout">${esc(shortModel(model))}</span>`];
  if (stats.tokens_per_second) parts.push(`<span class="readout">${stats.tokens_per_second.toFixed(1)} tok/s</span>`);
  if (stats.completion_tokens) parts.push(`<span class="readout">${stats.completion_tokens} tok</span>`);
  if (stats.duration_ms) parts.push(`<span class="readout">${duration(stats.duration_ms)}</span>`);
  if (stats.load_ms) parts.push(`<span class="readout" title="Model load time">load ${duration(stats.load_ms)}</span>`);
  if (stats.effort) parts.push(`<span class="readout">effort ${esc(stats.effort)}</span>`);
  if (stats.temperature && stats.temperature !== 'balanced') parts.push(`<span class="readout">${esc(stats.temperature)}</span>`);
  if (stats.cancelled) parts.push('<span class="readout is-brass">stopped</span>');
  else if (stats.finish_reason === 'length') parts.push('<span class="readout is-brass" title="Raise the effort for a longer budget">cut off at budget</span>');
  return `${parts.join('')}<button class="icon-btn" data-copy aria-label="Copy answer" title="Copy answer">${icon('copy')}</button>`;
}
function assistantTurn(message) {
  const element = document.createElement('article');
  element.className = 'turn turn--assistant';
  element.innerHTML = `
    <details class="reasoning" hidden><summary>${icon('brain')}<span class="grow">reasoning</span>${icon('chev', 'chev')}</summary><div class="reasoning__body"></div></details>
    <div class="prose"></div>
    <div class="sources" hidden></div>
    <div class="meta"></div>`;
  const parts = {
    element,
    reasoning: element.querySelector('.reasoning'),
    reasoningBody: element.querySelector('.reasoning__body'),
    prose: element.querySelector('.prose'),
    meta: element.querySelector('.meta'),
    sources: element.querySelector('.sources'),
    sourceList: message.stats?.sources || [],
    text: message.content || '',
    thought: message.reasoning || '',
  };
  if (parts.thought) { parts.reasoning.hidden = false; parts.reasoningBody.textContent = parts.thought; }
  if (parts.text) parts.prose.innerHTML = markdown(parts.text);
  renderSources(parts);
  if (message.model) { parts.meta.innerHTML = metaLine(message.model, message.stats || {}); bindMetaCopy(parts); }
  return parts;
}
function webURL(url) {
  try { const u = new URL(url); return /^https?:$/.test(u.protocol) ? u : null; } catch { return null; }
}
/* Web sources under an answer, and [1]-style citations in it turned into links. */
function renderSources(parts) {
  const list = parts.sourceList || [];
  if (!list.length) return;
  parts.sources.hidden = false;
  parts.sources.innerHTML = `<p class="sources__label">${icon('search')}<span>${list.length} web sources</span></p><ol>${list.map((s, i) => {
    const u = webURL(s.url);
    return `<li><a ${u ? `href="${esc(u.href)}"` : ''} target="_blank" rel="noopener noreferrer" title="${esc((s.snippet || '').slice(0, 300))}"><span class="sources__n">${i + 1}</span><span class="sources__title">${esc(s.title || s.url)}</span><span class="sources__host">${esc(u ? u.hostname.replace(/^www\./, '') : '')}</span></a></li>`;
  }).join('')}</ol>`;
  linkCitations(parts.prose, list);
}
function linkCitations(root, list) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => node.parentElement.closest('pre, code, a') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
  });
  const nodes = [];
  while (walker.nextNode()) if (/\[\d/.test(walker.currentNode.nodeValue)) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const html = esc(node.nodeValue).replace(/\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g, (match, nums) => {
      const links = nums.split(',').map((n) => {
        const u = webURL(list[Number(n) - 1]?.url);
        return u ? `<a class="cite" href="${esc(u.href)}" target="_blank" rel="noopener noreferrer" title="${esc(list[Number(n) - 1].title)}">${Number(n)}</a>` : null;
      });
      return links.every(Boolean) ? links.join('') : match;
    });
    if (html !== esc(node.nodeValue)) {
      const span = document.createElement('span');
      span.innerHTML = html;
      node.replaceWith(...span.childNodes);
    }
  }
}
function bindMetaCopy(parts) {
  const button = parts.meta.querySelector('[data-copy]');
  if (button) button.onclick = () => copyText(parts.text);
}

/* ---------- chat sending (SSE) ---------- */

function setChatState(text) {
  $('chat-state').hidden = !text;
  $('chat-state-text').textContent = text || '';
}
function setBusy(busy) {
  state.busy = busy;
  $('chat-log').setAttribute('aria-busy', String(busy));
  const send = $('send');
  send.classList.toggle('is-stop', busy);
  send.innerHTML = icon(busy ? 'stop' : 'up');
  send.setAttribute('aria-label', busy ? 'Stop' : 'Send');
  send.title = busy ? 'Stop (Esc)' : 'Send (Enter)';
  $('attach').disabled = busy;
  $('new-chat').disabled = busy;
  $('picker-button').disabled = busy;
  renderPicker();
  renderThreads();
}
function nearBottom(log) { return log.scrollHeight - log.scrollTop - log.clientHeight < 120; }

async function* readSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const {value, done} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    let cut;
    while ((cut = buffer.indexOf('\n\n')) >= 0) {
      const frame = buffer.slice(0, cut); buffer = buffer.slice(cut + 2);
      let event = 'message'; const data = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
      }
      if (data.length) yield {event, data: JSON.parse(data.join('\n'))};
    }
  }
}

async function send(event) {
  event?.preventDefault();
  if (state.busy) { stop(); return; }
  const text = $('prompt').value.trim();
  const attachment = state.attachment;
  if (!text && !attachment) return;
  const model = current();
  if (!model) { toast('Choose or download a chat model first.', true); openPicker(); return; }
  const log = $('chat-log');
  log.querySelector('.hero')?.remove();
  const userEl = userTurn({content: text, image_url: attachment?.data});
  if (attachment && !localURL(attachment.data, true)) userEl.insertAdjacentHTML('afterbegin', `<span class="tag bubble__file">${esc(attachment.name)}</span>`);
  log.append(userEl);
  const turn = assistantTurn({});
  turn.prose.innerHTML = '<p class="cursor"></p>';
  log.append(turn.element);
  log.scrollTop = log.scrollHeight;
  $('prompt').value = ''; autosize(); clearAttachment();
  const thinking = model.thinking === 'always' || (model.thinking !== 'none' && state.thinking);
  const controller = new AbortController();
  state.controller = controller;
  setBusy(true);
  setChatState(model.is_resident ? 'thinking' : `loading ${shortModel(model.id)}`);
  let meta = null; let done = null; let frame = 0; let reasoningOpened = false;
  const started = performance.now();
  const paint = () => {
    frame = 0;
    const stick = nearBottom(log);
    if (turn.thought) {
      turn.reasoning.hidden = false;
      turn.reasoningBody.textContent = turn.thought;
      turn.reasoningBody.scrollTop = turn.reasoningBody.scrollHeight;
      if (!reasoningOpened && !turn.text) { turn.reasoning.open = true; reasoningOpened = true; }
    }
    if (turn.text) {
      turn.prose.innerHTML = markdown(turn.text);
      turn.prose.lastElementChild?.classList.add('cursor');
    }
    if (stick) log.scrollTop = log.scrollHeight;
  };
  const schedule = () => { if (!frame) frame = requestAnimationFrame(paint); };
  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, signal: controller.signal,
      body: JSON.stringify({conversation_id: state.conversation, model: model.id, message: text, attachment_name: attachment?.name || null, image_data: attachment?.data || null, thinking, effort: state.effort, temperature: state.temperature, web: state.web}),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error || `Request failed (${response.status})`);
    }
    for await (const {event: name, data} of readSSE(response)) {
      if (name === 'meta') {
        meta = data;
        if (data.routed) toast(`Image routed to ${shortModel(data.model)}, a vision model.`);
        const preview = isSheetURL(data.image_url) ? '' : localURL(data.image_url);
        if (preview && !userEl.querySelector('img')) userEl.insertAdjacentHTML('afterbegin', `<img class="bubble__image" src="${esc(preview)}" alt="Document preview">`);
      } else if (name === 'status') {
        if (data.phase === 'searching') setChatState('searching the web');
        else if (data.phase === 'working out the figures') setChatState('working out the figures');
        else if (data.phase === 'loading') setChatState(`loading ${shortModel(meta?.model || model.id)}`);
        else if (data.phase === 'generating') setChatState(thinking ? 'reasoning' : 'writing');
        else if (data.phase.startsWith('reading ')) setChatState(data.phase.replace('reading', 'reading the conversation,'));
      } else if (name === 'sources') {
        turn.sourceList = data.sources || []; renderSources(turn);
      } else if (name === 'reasoning') {
        turn.thought += data.text; turn.reasoning.classList.add('is-live'); schedule();
      } else if (name === 'content') {
        if (!turn.text && turn.thought) { turn.reasoning.classList.remove('is-live'); turn.reasoning.open = false; setChatState('writing'); }
        turn.text += data.text; schedule();
      } else if (name === 'done') {
        done = data;
      } else if (name === 'error') {
        throw new Error(data.error);
      }
    }
    if (!done) throw new Error('The stream ended early. The server may have restarted.');
  } catch (error) {
    if (error.name !== 'AbortError') {
      if (frame) cancelAnimationFrame(frame);
      if (!turn.text) {
        turn.element.remove();
        $('prompt').value = text; autosize();
        if (attachment) { state.attachment = attachment; renderAttachment(); }
      }
      userEl.insertAdjacentHTML('beforeend', `<p class="turn__error">${esc(error.message)}</p>`);
      fail(error);
    } else {
      done = {cancelled: true};
    }
  } finally {
    if (frame) { cancelAnimationFrame(frame); paint(); }
    turn.prose.querySelector('.cursor')?.classList.remove('cursor');
    turn.reasoning.classList.remove('is-live');
    if (turn.thought) turn.reasoning.querySelector('.grow').textContent = `reasoning · ${turn.thought.length.toLocaleString()} chars`;
    if (turn.text) {
      const stats = {...(done?.stats || {}), duration_ms: done?.duration_ms ?? Math.round(performance.now() - started), effort: state.effort, temperature: state.temperature, cancelled: !!done?.cancelled};
      turn.meta.innerHTML = metaLine(meta?.model || model.id, stats);
      bindMetaCopy(turn);
      bindCodeCopy(turn.prose);
      linkCitations(turn.prose, turn.sourceList || []);
    } else if (done?.cancelled) {
      turn.element.remove();
    }
    state.controller = null;
    setBusy(false);
    setChatState(null);
    if (meta?.conversation_id && (done?.saved || done?.cancelled)) {
      // A stopped reply is saved a moment after the stream closes.
      for (let attempt = 0; attempt < 6; attempt++) {
        await loadConversations().catch(() => {});
        if (state.conversations.some((c) => c.id === meta.conversation_id) || !done?.cancelled) break;
        await new Promise((resolve) => setTimeout(resolve, 400));
      }
      const saved = state.conversations.find((c) => c.id === meta.conversation_id);
      if (saved) {
        state.conversation = saved.id;
        $('chat-title').textContent = saved.title;
        history.replaceState(null, '', `#chat/${saved.id}`);
        renderThreads();
      }
    }
    refreshStatus();
    $('prompt').focus();
  }
}
function stop() { state.controller?.abort(); }

function autosize() {
  const prompt = $('prompt');
  prompt.style.height = 'auto';
  prompt.style.height = `${Math.min(prompt.scrollHeight, window.innerHeight * 0.4)}px`;
}
async function attach(file) {
  try {
    const sheet = file && sheetType(file.name);
    if (sheet) {
      if (file.size > 18 * 1024 * 1024) throw new Error('Choose a file smaller than 18 MB.');
      const data = await fileToData(file);
      state.attachment = {name: file.name, data: `data:${sheet};base64,${data.split(';base64,')[1] || ''}`};
    } else {
      if (!checkFile(file)) return;
      state.attachment = {name: file.name, data: await fileToData(file)};
    }
    renderAttachment();
  } catch (error) { fail(error); }
}
function renderAttachment() {
  const box = $('attachment');
  box.hidden = !state.attachment;
  if (!state.attachment) return;
  box.innerHTML = `${icon(state.attachment.data.startsWith('data:image/') ? 'image' : 'doc')}<span class="readout">${esc(state.attachment.name)}</span><button type="button" class="icon-btn" aria-label="Remove attachment">${icon('x')}</button>`;
  box.querySelector('svg').style.cssText = 'width:15px;height:15px;fill:none;stroke:currentColor;stroke-width:1.5';
  box.querySelector('button').onclick = clearAttachment;
}
function clearAttachment() { state.attachment = null; $('file').value = ''; renderAttachment(); }

/* ---------- documents ---------- */

async function setOcrFile(file) {
  if (state.ocr.busy) return;
  try {
    if (!checkFile(file)) return;
    state.ocr.data = await fileToData(file);
    state.ocr.name = file.name;
    $('ocr-name').textContent = file.name;
    $('ocr-run').disabled = false;
    const image = localURL(state.ocr.data, true);
    $('drop').hidden = true;
    $('ocr-preview').hidden = false;
    $('ocr-preview').innerHTML = `${image ? `<img src="${esc(image)}" alt="Selected document">` : `<span class="tag">${icon('doc')}${esc(file.name)}</span>`}<button class="btn btn--ghost btn--xs" id="ocr-change">choose another</button>`;
    $('ocr-change').onclick = () => $('ocr-file').click();
    $('ocr-meta').textContent = 'ready';
  } catch (error) { fail(error); }
}
async function extractText() {
  if (!state.ocr.data || state.ocr.busy) return;
  state.ocr.busy = true;
  const button = $('ocr-run');
  button.disabled = true; button.setAttribute('aria-busy', 'true'); button.innerHTML = '<span class="spinner"></span>reading';
  try {
    const result = await post('/api/ocr', {image_data: state.ocr.data}, 180000);
    $('ocr-text').value = result.text || '';
    const has = !!result.text;
    $('ocr-copy').disabled = !has; $('ocr-chat').disabled = !has;
    $('ocr-meta').textContent = has ? `${result.text.length.toLocaleString()} chars · ${result.engine || 'local ocr'}` : 'no text found';
    if (!has) toast('No text was recognized. Try a sharper image.', true);
  } catch (error) { $('ocr-meta').textContent = 'failed'; fail(error); }
  finally { state.ocr.busy = false; button.disabled = false; button.removeAttribute('aria-busy'); button.textContent = 'extract text'; }
}

/* ---------- images ---------- */

function showImage(item) {
  const url = localURL(item.image_url);
  if (!url) return;
  $('canvas').classList.remove('blueprint');
  $('canvas').innerHTML = `<div class="canvas__result"><img src="${esc(url)}" alt="${esc(item.prompt)}"><div class="row"><span class="readout grow">${item.width}×${item.height} · ${esc(duration(item.duration_ms))} · ${esc(item.backend || 'local')}</span><button class="btn btn--ghost btn--xs" id="canvas-edit">edit</button><a class="btn btn--ghost btn--xs" href="${esc(url)}" download="ferro64-${esc(item.id)}.png">save</a></div></div>`;
  $('canvas-edit').onclick = () => startEdit({id: item.id, url, name: item.prompt});
}
function startEdit(edit) {
  if (state.imageBusy) return;
  state.edit = edit;
  renderEdit();
  $('image-prompt').value = '';
  $('image-prompt').focus();
  $('image-form').scrollIntoView({behavior: 'smooth', block: 'start'});
}
function stopEdit() { state.edit = null; renderEdit(); }
function renderEdit() {
  const editing = !!state.edit;
  $('editing').hidden = !editing;
  $('image-options').hidden = editing;
  $('image-prompt-label').textContent = editing ? 'what should change' : 'prompt';
  $('image-prompt').placeholder = editing ? 'Remove the ceiling fan. Make the cushions sage green.' : 'A ceramic cup on a sunlit oak table, soft morning light';
  $('generate').textContent = editing ? 'apply edit' : 'generate';
  if (!editing) return;
  $('editing-img').src = localURL(state.edit.url, true);
  $('editing-name').textContent = state.edit.name || 'image';
}
async function usePhoto(file) {
  try {
    if (!file) return;
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) throw new Error('Choose a PNG, JPEG or WebP photo.');
    if (file.size > 18 * 1024 * 1024) throw new Error('Choose a photo smaller than 18 MB.');
    const data = await fileToData(file);
    startEdit({data, url: data, name: file.name});
  } catch (error) { fail(error); }
}
function renderImageModels() {
  const models = state.status?.image_models || [];
  const sdcpp = state.status?.image_backend === 'sdcpp';
  $('image-model').hidden = !sdcpp;
  // Offer only downloaded models; fall back to the first one that is ready.
  const ready = models.filter((m) => m.ready).map((m) => m.id);
  if (ready.length && !ready.includes(state.imageModel)) state.imageModel = ready[0];
  for (const button of $('image-model').querySelectorAll('[data-image-model]')) button.hidden = ready.length > 0 && !ready.includes(button.dataset.imageModel);
  setRadio($('image-model'), state.imageModel, 'imageModel');
  const current = models.find((m) => m.id === state.imageModel);
  const note = $('image-model-note');
  note.hidden = !sdcpp;
  if (!sdcpp) return;
  note.innerHTML = !current ? '' : current.ready
    ? esc(IMAGE_NOTES[state.imageModel] || '')
    : `${esc(current.name.toLowerCase())} is not downloaded · <a href="#models">get it in models</a>`;
  const flux = models.find((m) => m.id === 'flux2-klein');
  $('generate').disabled = state.imageBusy || (state.edit ? flux && !flux.ready : current && !current.ready);
  $('photo-button').disabled = state.imageBusy || (flux && !flux.ready);
  $('photo-button').title = flux && !flux.ready ? 'Editing needs FLUX.2 Klein; download it in models' : 'Start from your own photo and describe a change';
}

async function generateImage(event) {
  event.preventDefault();
  if (state.imageBusy) return;
  const prompt = $('image-prompt').value.trim();
  if (!prompt) return;
  state.imageBusy = true;
  const button = $('generate');
  const edit = state.edit;
  button.disabled = true; button.innerHTML = `<span class="spinner"></span>${edit ? 'editing' : 'generating'}`;
  const previous = $('canvas').innerHTML;
  $('canvas').innerHTML = `<p class="readout"><span class="spinner" style="display:inline-block;vertical-align:-2px;margin-right:8px"></span>${edit ? 'editing with flux.2 klein · about a minute' : `generating with ${IMAGE_NOTES[state.imageModel] || state.imageModel}`}</p>`;
  try {
    const body = {prompt, aspect_ratio: state.aspect, model: edit ? 'flux2-klein' : state.imageModel};
    if (edit?.id) body.edit_of = edit.id;
    if (edit?.data) body.image_data = edit.data;
    const response = await post('/api/generate', body, 1800000);
    if (!response.success || !response.item) throw new Error(response.error || 'Generation returned no image.');
    showImage(response.item);
    // Further edits continue from the new image.
    if (edit) { state.edit = {id: response.item.id, url: response.item.image_url, name: prompt}; $('image-prompt').value = ''; }
    await loadGallery();
  } catch (error) { $('canvas').innerHTML = previous; fail(error); }
  finally { state.imageBusy = false; button.disabled = false; renderEdit(); renderImageModels(); refreshStatus(); }
}
async function loadGallery() {
  state.gallery = await api('/api/history');
  $('gallery-count').textContent = state.gallery.length ? String(state.gallery.length) : '';
  if (!state.gallery.length) { $('gallery').innerHTML = '<p class="empty">Generated images are kept here.</p>'; return; }
  $('gallery').innerHTML = state.gallery.map((item, i) => {
    const url = localURL(item.image_url);
    return `<figure class="shot" style="margin:0"><button class="shot__open" data-open="${i}" aria-label="Show ${esc(item.prompt)}">${url ? `<img src="${esc(url)}" loading="lazy" alt="${esc(item.prompt)}">` : ''}</button><figcaption class="shot__bar"><span class="readout" title="${esc(item.prompt)}">${esc(item.prompt)}</span><button class="icon-btn" data-edit="${i}" aria-label="Edit image" title="Edit">${icon('pen')}</button><button class="icon-btn" data-animate="${i}" aria-label="Make a video from this image" title="Animate">${icon('film')}</button><button class="icon-btn" data-delete="${i}" aria-label="Delete image">${icon('trash')}</button></figcaption></figure>`;
  }).join('');
  for (const button of $('gallery').querySelectorAll('[data-open]')) button.onclick = () => { const item = state.gallery[Number(button.dataset.open)]; showImage(item); $('image-prompt').value = item.prompt; $('canvas').scrollIntoView({behavior: 'smooth', block: 'center'}); };
  for (const button of $('gallery').querySelectorAll('[data-edit]')) button.onclick = () => { const item = state.gallery[Number(button.dataset.edit)]; startEdit({id: item.id, url: item.image_url, name: item.prompt}); };
  for (const button of $('gallery').querySelectorAll('[data-animate]')) button.onclick = () => animateImage(state.gallery[Number(button.dataset.animate)]);
  for (const button of $('gallery').querySelectorAll('[data-delete]')) button.onclick = async () => {
    const item = state.gallery[Number(button.dataset.delete)];
    if (!await confirmDialog('delete this image?')) return;
    try { await api(`/api/history/${encodeURIComponent(item.id)}`, {method: 'DELETE'}); await loadGallery(); } catch (error) { fail(error); }
  };
}

/* ---------- agents ---------- */

function fillAgentSelects() {
  const models = chatModels();
  // A strong Qwen drafts and writes the result; another maker reviews.
  const defaults = {'agent-planner': ['qwen3.5:35b-a3b', 'qwen3.5:9b', 'qwen3:8b'], 'agent-reviewer': ['nemotron-3-nano:30b-a3b', 'glm-4.7-flash', 'qwen3.5:9b'], 'agent-synth': ['qwen3.5:35b-a3b', 'qwen3.5:9b', 'qwen3:8b']};
  for (const [id, preferred] of Object.entries(defaults)) {
    const select = $(id); const old = select.value;
    select.innerHTML = models.length ? models.map((m) => `<option value="${esc(m.id)}">${esc(m.name || shortModel(m.id))}</option>`).join('') : '<option value="">no chat models</option>';
    select.disabled = !models.length || state.agentBusy;
    const pick = [old, ...preferred].find((value) => models.some((m) => m.id === value));
    if (pick) select.value = pick;
  }
}
function renderTrack(task) {
  const roles = ['Planner', 'Reviewer', 'Synthesizer'];
  const labels = ['draft', 'review', 'synthesis'];
  const steps = task?.steps || [];
  return `<div class="track">${roles.map((role, i) => {
    const step = steps.find((s) => s.role === role);
    const running = !step && steps.length === i && task?.status === 'running';
    return `<div class="track__step"><span class="readout"><span class="dot ${step ? 'is-live' : running ? 'is-busy' : ''}"></span>0${i + 1} · ${labels[i]}</span><small>${step ? `${esc(shortModel(step.model))} · ${duration(step.duration_ms)}` : running ? 'working…' : 'waiting'}</small></div>`;
  }).join('')}</div>`;
}
function renderAgentResult(task, live = false) {
  const finalText = task.final_result || (task.status === 'failed' ? 'The workflow stopped before a final answer. Finished stages are below.' : '');
  $('agent-live').innerHTML = `
    ${renderTrack(task)}
    ${live ? '' : `<article class="well result"><div class="result__head"><h3>${task.status === 'completed' ? 'result' : esc(task.status)}</h3><span class="row"><span class="readout">${esc(duration(task.total_duration_ms))}</span><button class="icon-btn" id="agent-copy" aria-label="Copy result">${icon('copy')}</button></span></div>
      <div class="prose">${markdown(finalText)}</div>
      ${(task.steps || []).map((s) => `<details class="step"><summary><span class="readout">0${s.step_index} · ${esc(s.role.toLowerCase())}</span><span class="readout">${esc(shortModel(s.model))} · ${esc(duration(s.duration_ms))}</span></summary><div class="prose">${markdown(s.output)}</div></details>`).join('')}
    </article>`}`;
  bindCodeCopy($('agent-live'));
  if (!live) $('agent-copy').onclick = () => copyText(task.final_result || '');
}
async function loadTasks() {
  state.tasks = await api('/api/agent/tasks');
  if (!state.tasks.length) { $('agent-history').innerHTML = '<p class="empty">Workflow runs and every stage are saved here.</p>'; return; }
  $('agent-history').innerHTML = state.tasks.map((t, i) => `<button class="item item--run" data-task="${i}"><span class="item__main"><span class="item__title"><strong>${esc(t.goal.length > 120 ? `${t.goal.slice(0, 117)}…` : t.goal)}</strong></span><span class="item__specs"><span class="readout">${esc(when(t.created_at))}</span><span class="readout">${(t.steps || []).length}/3 stages</span><span class="readout">${esc(duration(t.total_duration_ms))}</span></span></span><span class="tag status-tag ${t.status === 'failed' ? 'is-failed' : t.status === 'running' ? 'is-running' : ''}">${esc(t.status)}</span></button>`).join('');
  for (const button of $('agent-history').querySelectorAll('[data-task]')) button.onclick = () => {
    if (state.agentBusy) return toast('Wait for the current run to finish.', true);
    renderAgentResult(state.tasks[Number(button.dataset.task)]);
    $('agent-live').scrollIntoView({behavior: 'smooth', block: 'start'});
  };
}
async function runAgent(event) {
  event.preventDefault();
  if (state.agentBusy) return;
  const goal = $('agent-goal').value.trim();
  const models = ['agent-planner', 'agent-reviewer', 'agent-synth'].map((id) => $(id).value);
  if (!goal) return;
  if (models.some((m) => !m)) return toast('Download a chat model before running a workflow.', true);
  state.agentBusy = true;
  const button = $('agent-run');
  button.disabled = true; button.innerHTML = '<span class="spinner"></span>running';
  for (const id of ['agent-goal', 'agent-planner', 'agent-reviewer', 'agent-synth']) $(id).disabled = true;
  const startedAt = Date.now();
  renderAgentResult({status: 'running', steps: []}, true);
  // The run is saved after each stage; poll to show progress while it works.
  const poll = setInterval(async () => {
    try {
      const tasks = await api('/api/agent/tasks');
      const live = tasks.find((t) => t.status === 'running' && t.goal === goal && new Date(t.created_at).getTime() >= startedAt - 10000);
      if (live && state.agentBusy) renderAgentResult(live, true);
    } catch { /* progress is best effort */ }
  }, 2000);
  try {
    const task = await post('/api/agent/run', {goal, planner_model: models[0], reviewer_model: models[1], synthesizer_model: models[2], effort: state.agentEffort, thinking: state.agentThinking}, 3600000);
    clearInterval(poll);
    renderAgentResult(task);
  } catch (error) {
    clearInterval(poll);
    fail(error);
    const tasks = await api('/api/agent/tasks').catch(() => []);
    const saved = tasks.find((t) => t.goal === goal && new Date(t.created_at).getTime() >= startedAt - 10000);
    if (saved) renderAgentResult(saved); else $('agent-live').innerHTML = '';
  } finally {
    clearInterval(poll);
    state.agentBusy = false;
    button.disabled = false; button.textContent = 'run workflow';
    for (const id of ['agent-goal', 'agent-planner', 'agent-reviewer', 'agent-synth']) $(id).disabled = false;
    loadTasks().catch(() => {});
    refreshStatus();
  }
}

/* ---------- models view ---------- */

function renderModelsView() {
  const status = state.status || {};
  const resident = status.active_models || [];
  $('resident-dot').className = `dot ${state.loading ? 'is-busy' : resident.length ? 'is-live' : status.runtime_online === false ? 'is-off' : ''}`;
  $('resident-state').textContent = state.loading ? `loading ${shortModel(state.loading)}` : resident.length ? `${resident.map(shortModel).join(', ')} resident` : 'idle · nothing in memory';
  const installed = state.models.filter((m) => m.installed !== false);
  const db = status.database || {};
  const rows = [
    ['engine', status.engine || BOOT.engine],
    ['runtime', status.runtime_online ? 'ready' : status.runtime_online === false ? 'unavailable' : '…'],
    ['keep warm', status.keep_alive || BOOT.keep_alive],
    ['context', `${(status.context || BOOT.context || 0).toLocaleString()} tok`],
    ['models on disk', String(installed.length)],
    ['conversations', `${db.conversations ?? '…'} · ${db.messages ?? '…'} msgs`],
    ['ocr', status.ocr_available ? 'vision + pdfkit' : 'helper not built'],
    ['images', status.image_backend || 'ollama'],
  ];
  $('readouts').innerHTML = rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd title="${esc(v)}">${esc(v)}</dd></div>`).join('');
  $('installed-count').textContent = installed.length ? String(installed.length) : '';
  if (!state.models.length) {
    $('installed').innerHTML = '<p class="empty">Nothing yet. Download a model below, or put a .gguf file in data/models and refresh.</p>';
  } else {
    $('installed').innerHTML = state.models.map((m) => {
      const chat = isChat(m);
      const loading = state.loading === m.id;
      return `<div class="item">
        <div class="item__main">
          <div class="item__title"><span class="dot ${m.is_resident ? 'is-live' : loading ? 'is-busy' : ''}"></span><strong>${esc(m.name || shortModel(m.id))}</strong>${modelTags(m).map((t) => `<span class="tag">${t}</span>`).join('')}${m.installed === false ? '<span class="tag">unverified</span>' : ''}</div>
          <p class="item__desc">${esc(m.role_description || '')}</p>
          <div class="item__specs"><span class="readout">${esc(m.id)}</span>${modelSpecs(m).map((s) => `<span class="readout">${esc(s)}</span>`).join('')}${chat ? `<span class="readout">${esc(thinkingNote(m))}</span>` : ''}</div>
        </div>
        <div class="item__actions">${chat && m.installed !== false ? `<button class="btn btn--ghost btn--xs" data-load="${esc(m.id)}" ${m.is_resident || loading || state.busy ? 'disabled' : ''}>${m.is_resident ? 'resident' : loading ? 'loading…' : 'load'}</button><button class="btn btn--primary btn--xs" data-use="${esc(m.id)}">use in chat</button>` : ''}${catalogIds().has(m.id) ? `<button class="btn btn--ghost btn--xs" data-remove="${esc(m.id)}">remove</button>` : ''}</div>
      </div>`;
    }).join('');
    for (const button of $('installed').querySelectorAll('[data-use]')) button.onclick = () => { selectModel(button.dataset.use, {preload: false}); setView('chat'); };
    for (const button of $('installed').querySelectorAll('[data-load]')) button.onclick = () => selectModel(button.dataset.load);
    for (const button of $('installed').querySelectorAll('[data-remove]')) button.onclick = () => removeModel(button.dataset.remove);
  }
  renderCatalog();
}

let storeTimer = null;
async function loadModelStore() {
  try {
    state.store = await api('/api/model-store');
  } catch { return; }
  renderCatalog();
  const active = (state.store.downloads || []).some((d) => ['queued', 'downloading'].includes(d.status));
  if (active && !storeTimer) storeTimer = setInterval(loadModelStore, 1500);
  if (!active && storeTimer) { clearInterval(storeTimer); storeTimer = null; await loadModels().catch(() => {}); refreshStatus(); }
}
function renderCatalog() {
  const catalog = state.store?.catalog || [];
  if (BOOT.backend !== 'llamacpp') {
    $('catalog').innerHTML = '<p class="empty">Downloads here are GGUF files for the llama.cpp engine. With Ollama, use <span class="readout">ollama pull &lt;model&gt;</span>, then refresh.</p>';
    return;
  }
  if (!catalog.length) { $('catalog').innerHTML = '<p class="empty">Catalog unavailable.</p>'; return; }
  const jobs = state.store.downloads || [];
  // What runs here first; what won't run last, greyed out.
  const order = {good: 0, tight: 1, no: 2};
  const sorted = [...catalog].sort((a, b) => (order[a.fit] ?? 1) - (order[b.fit] ?? 1));
  const rec = state.store.recommended || {};
  const jobsFor = (id) => Object.entries({chat: 'chat', code: 'code', agents: 'agents', reviewer: 'agent review', images: 'images', edits: 'image edits', search: 'search'}).filter(([key]) => rec[key] === id).map(([, label]) => label);
  // Z-Image and FLUX.2 read prompts with Qwen3 4B, so it comes with them.
  const needsReader = (id) => id === 'qwen3:4b' && (rec.images || rec.edits);
  const isPick = (entry) => jobsFor(entry.id).length > 0 || needsReader(entry.id);
  const row = (entry) => {
    const job = jobs.find((j) => j.model === entry.id);
    const busy = job && ['queued', 'downloading'].includes(job.status);
    const installed = entry.installed || job?.status === 'completed';
    const pct = job?.progress != null ? Math.round(job.progress) : 0;
    const tags = [...new Set([entry.kind, entry.vision ? 'vision' : ''].filter(Boolean))];
    const best = jobsFor(entry.id);
    const fitTag = entry.fit === 'good' && entry.dense ? 'fits, but slow' : {good: 'fits', tight: 'slow here', no: 'too big'}[entry.fit] || '';
    const fitClass = entry.fit === 'good' && entry.dense ? 'tight' : entry.fit;
    return `<div class="item${entry.fit === 'no' && !installed ? ' item--nofit' : ''}">
      <div class="item__main">
        <div class="item__title"><strong>${esc(entry.name)}</strong>${best.length ? `<span class="tag tag--best">best here for ${esc(best.join(', '))}</span>` : ''}${needsReader(entry.id) ? '<span class="tag tag--best">needed for images</span>' : ''}${fitTag ? `<span class="tag tag--fit-${esc(fitClass)}">${esc(fitTag)}</span>` : ''}${tags.map((t) => `<span class="tag">${esc(t)}</span>`).join('')}</div>
        <p class="item__desc">${esc(entry.role)}</p>
        ${entry.fit_note ? `<p class="item__fit">${esc(entry.fit_note)}${entry.dense ? '. A dense model: about 11–15 tok/s on an M1 Max, against about 50 for the A3B models' : ''}</p>` : ''}
        <div class="item__specs"><span class="readout">${esc(entry.id)}</span><span class="readout">${esc(bytes(entry.size_bytes))}</span><a class="readout" href="${esc(entry.source)}" target="_blank" rel="noopener noreferrer">${esc(entry.source.replace('https://huggingface.co/', 'hf/'))}</a></div>
        ${busy ? `<div class="progress" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"><span style="width:${pct}%"></span></div><span class="readout">${job.status === 'queued' ? 'queued' : `${pct}% · ${bytes(job.downloaded_bytes)} of ${bytes(job.total_bytes)}`}</span>` : ''}
        ${job?.status === 'failed' ? `<p class="item__error">${esc(job.error || 'Download failed')}</p>` : ''}
      </div>
      <div class="item__actions">${installed ? `<span class="tag">on disk</span><button class="btn btn--ghost btn--xs" data-remove="${esc(entry.id)}">remove</button>` : entry.fit === 'no' && !busy ? `<button class="btn btn--ghost btn--xs" data-download="${esc(entry.id)}" data-anyway="1" title="${esc(entry.fit_note)}">download anyway</button>` : `<button class="btn ${busy ? 'btn--ghost' : 'btn--primary'} btn--xs" data-download="${esc(entry.id)}" ${busy ? 'disabled' : ''}>${busy ? 'downloading' : job?.status === 'failed' ? 'retry' : 'download'}</button>`}</div>
    </div>`;
  };
  const picks = sorted.filter(isPick);
  const more = sorted.filter((entry) => !isPick(entry));
  const total = picks.filter((e) => !e.installed).reduce((sum, e) => sum + e.size_bytes, 0);
  $('catalog').innerHTML = machineSummary()
    + `<div class="section-sub"><span class="label">recommended for this machine</span>${total ? `<span class="readout">${esc(bytes(total))} left to download</span>` : ''}</div>`
    + picks.map(row).join('')
    + (more.length ? `<details class="more-models" id="more-models"${state.moreModelsOpen ? ' open' : ''}><summary>more models <span class="readout">${more.length}</span></summary>${more.map(row).join('')}</details>` : '');
  const moreEl = $('more-models');
  if (moreEl) moreEl.ontoggle = () => { state.moreModelsOpen = moreEl.open; };
  for (const button of $('catalog').querySelectorAll('[data-remove]')) button.onclick = () => removeModel(button.dataset.remove);
  for (const button of $('catalog').querySelectorAll('[data-download]')) button.onclick = async () => {
    const entry = catalog.find((e) => e.id === button.dataset.download);
    if (button.dataset.anyway && !await confirmDialog(`download ${entry.name.toLowerCase()} anyway?`, `${entry.fit_note}. It is ${bytes(entry.size_bytes)} on disk.`)) return;
    button.disabled = true;
    try { await post('/api/models/download', {model: button.dataset.download}, 30000); await loadModelStore(); }
    catch (error) { button.disabled = false; fail(error); }
  };
}
/* What this machine is, and the best model per job that fits it. */
function machineSummary() {
  const hw = state.store?.hardware;
  if (!hw) return '';
  const rec = state.store.recommended || {};
  const name = (id) => (state.store.catalog || []).find((e) => e.id === id)?.name;
  const where = hw.kind === 'mac' ? `${hw.gpu_name || 'This Mac'} · ${Math.round(hw.ram_gb)} GB, of which the GPU can use about ${Math.round(hw.gpu_gb)} GB`
    : hw.kind === 'cpu' ? `${Math.round(hw.ram_gb)} GB memory, no supported GPU found: models run on the CPU, slowly`
    : `${hw.gpu_name || 'GPU'} with ${Math.round(hw.gpu_gb)} GB · ${Math.round(hw.ram_gb)} GB system memory`;
  const picks = [['chat', rec.chat], ['code', rec.code], ['agents', rec.agents && rec.reviewer ? `${name(rec.agents)} with ${name(rec.reviewer)} reviewing` : rec.agents], ['images', rec.images], ['image edits', rec.edits], ['search by meaning', rec.search]]
    .filter(([, id]) => id).map(([job, id]) => {
      const slow = (state.store.catalog || []).find((e) => e.id === (rec[job] || id))?.fit === 'tight';
      return `<li><span class="readout">${esc(job)}</span> ${esc(name(id) || id)}${slow ? ' <span class="tag tag--fit-tight">slow here</span>' : ''}</li>`;
    }).join('');
  return `<div class="well well--pad machine"><span class="label">for this machine</span><p>${esc(where)}</p>${picks ? `<ul class="machine__picks">${picks}</ul>` : '<p class="readout">Nothing in the catalog fits comfortably; the smallest models below may still run slowly.</p>'}<p class="readout">Estimates. Models marked too big are greyed out; you can still download them.</p></div>`;
}
function catalogIds() { return new Set((state.store?.catalog || []).map((e) => e.id)); }
async function removeModel(id) {
  const entry = (state.store?.catalog || []).find((e) => e.id === id);
  const name = entry?.name || shortModel(id);
  const imageModels = (state.store?.catalog || []).filter((e) => e.kind === 'image' && e.installed).map((e) => e.name);
  const note = id === 'qwen3:4b' && imageModels.length
    ? `${imageModels.join(' and ')} use Qwen3 4B to read prompts, so image generation will stop working. `
    : '';
  if (!await confirmDialog(`remove ${name.toLowerCase()}?`, `${note}Its files (${bytes(entry?.size_bytes) || 'on disk'}) are deleted from data/models. You can download it again later.`)) return;
  try {
    const result = await post('/api/models/remove', {model: id}, 120000);
    toast(`Removed ${name} (${result.files.length} file${result.files.length === 1 ? '' : 's'})`);
    if (state.model === id) { state.model = null; store.set('model', ''); }
    await Promise.all([loadModels(), loadModelStore(), refreshStatus()]);
    renderModelsView();
  } catch (error) { fail(error); }
}
async function freeMemory() {
  const button = $('unload');
  button.disabled = true;
  try {
    const result = await post('/api/models/unload', {}, 120000);
    toast(result.unloaded_models?.length ? `Unloaded ${result.unloaded_models.map(shortModel).join(', ')}` : 'Nothing was loaded');
    await refreshStatus();
  } catch (error) { fail(error); }
  finally { button.disabled = false; }
}
async function refreshAll() {
  await Promise.allSettled([refreshStatus(), loadModels(), loadModelStore(), loadConversations()]);
}

/* ---------- settings ---------- */

async function loadSettings() {
  try { state.settings = await api('/api/settings', {timeout: 15000}); } catch (error) { fail(error); return; }
  if (!state.promptDirty) $('system-prompt').value = state.settings.system_prompt || '';
  $('system-prompt').maxLength = state.settings.system_prompt_limit;
  $('recall').setAttribute('aria-checked', String(state.settings.recall));
  renderSearchPanel();
  $('recall-note').textContent = state.settings.embedding_installed
    ? 'related questions from your other conversations are added to the model\'s context'
    : 'needs Qwen3 Embedding 0.6B: download it in models below';
  renderStorage();
}
function renderStorage() {
  const settings = state.settings;
  if (!settings) return;
  $('storage-note').textContent = `data/models · ${bytes(settings.storage.models_bytes) || 'empty'}`;
}
/* Web search: which engine answers, and the free Keenable key. */
function renderSearchPanel() {
  const settings = state.settings;
  if (!settings) return;
  const key = settings.keenable || {};
  $('search-engine').textContent = `engine · ${settings.web_search.toLowerCase()}`;
  const panel = $('search-panel');
  if (key.source === 'env') {
    panel.innerHTML = `<p>Using the Keenable key from <span class="readout">.env</span> (<span class="readout">${esc(key.hint || '')}</span>): 100,000 searches a month.</p>`;
    return;
  }
  if (key.source === 'settings') {
    panel.innerHTML = `<div class="row row--wrap"><p class="grow">Using your Keenable key <span class="readout">${esc(key.hint || '')}</span>: 100,000 searches a month.</p><button class="btn btn--ghost btn--xs" id="keenable-remove" type="button">remove</button></div>`;
    $('keenable-remove').onclick = () => saveKeenableKey('');
    return;
  }
  panel.innerHTML = `<p>Web search works without an account, on Keenable's shared free tier (1,000 searches an hour per network). A free Keenable account gives you <strong>100,000 searches a month</strong> of your own, which is steadier:</p>
    <ol class="steps-list">
      <li>Go to <a href="https://keenable.ai" target="_blank" rel="noopener noreferrer">keenable.ai</a> and sign up for the free plan.</li>
      <li>In your Keenable dashboard, create an API key and copy it.</li>
      <li>Paste it here. It is stored only on this computer and sent only to Keenable with your searches.</li>
    </ol>
    <form class="row row--wrap" id="keenable-form" autocomplete="off">
      <label class="visually-hidden" for="keenable-key">Keenable API key</label>
      <input class="field grow" id="keenable-key" type="password" placeholder="paste your Keenable API key" spellcheck="false" autocomplete="off">
      <button class="btn btn--primary btn--xs">save key</button>
    </form>`;
  $('keenable-form').onsubmit = (event) => { event.preventDefault(); saveKeenableKey($('keenable-key').value.trim()); };
}
async function saveKeenableKey(key) {
  try {
    await api('/api/settings', {method: 'PUT', body: JSON.stringify({keenable_api_key: key})});
    toast(key ? 'Keenable key saved; web search now uses your account' : 'Keenable key removed');
    await loadSettings();
  } catch (error) { fail(error); }
}

async function saveSystemPrompt(event) {
  event.preventDefault();
  $('system-prompt-save').disabled = true;
  try {
    await api('/api/settings', {method: 'PUT', body: JSON.stringify({system_prompt: $('system-prompt').value})});
    state.promptDirty = false;
    $('system-prompt-note').textContent = $('system-prompt').value.trim() ? 'saved · added to every chat and every agent stage' : 'saved · no system prompt';
  } catch (error) { fail(error); }
  finally { $('system-prompt-save').disabled = false; }
}

/* ---------- wiring ---------- */

// In the desktop app, links that would open a new tab go to the browser.
document.addEventListener('click', (event) => {
  const link = event.target.closest?.('a[target="_blank"]');
  const opener = window.__TAURI__?.opener;
  if (!link || !opener || !/^https?:/.test(link.href)) return;
  event.preventDefault();
  opener.openUrl(link.href).catch(fail);
});

function bind() {
  for (const button of document.querySelectorAll('.nav__item')) button.onclick = () => setView(button.dataset.view);
  for (const button of document.querySelectorAll('[data-open-rail]')) button.onclick = openRail;
  $('rail-close').onclick = closeRail;
  $('scrim').onclick = closeRail;
  window.addEventListener('hashchange', () => { setView(location.hash.slice(1), false); openFromHash(); });
  $('theme-toggle').onclick = () => { const next = document.documentElement.dataset.theme === 'day' ? 'night' : 'day'; store.set('theme', next); applyTheme(next); };
  window.matchMedia?.('(prefers-color-scheme: light)').addEventListener?.('change', (event) => { if (!store.get('theme')) applyTheme(event.matches ? 'day' : 'night'); });

  $('composer').onsubmit = send;
  $('prompt').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!state.busy) send(); }
  });
  $('prompt').addEventListener('input', autosize);
  $('prompt').addEventListener('paste', (event) => {
    const file = [...(event.clipboardData?.files || [])][0];
    if (file) { event.preventDefault(); attach(file); }
  });
  $('view-chat').addEventListener('dragover', (event) => { if (event.dataTransfer?.types.includes('Files')) event.preventDefault(); });
  $('view-chat').addEventListener('drop', (event) => { const file = event.dataTransfer?.files?.[0]; if (file) { event.preventDefault(); attach(file); } });
  $('attach').onclick = () => $('file').click();
  $('file').onchange = () => attach($('file').files[0]);
  $('new-chat').onclick = newChat;
  $('thread-search').oninput = renderThreads;

  $('picker-button').onclick = () => ($('picker-menu').hidden ? openPicker() : closePicker());
  $('picker-menu').addEventListener('keydown', (event) => {
    const options = [...$('picker-menu').querySelectorAll('.option')];
    const index = options.indexOf(document.activeElement);
    if (event.key === 'ArrowDown') { event.preventDefault(); options[Math.min(index + 1, options.length - 1)]?.focus(); }
    if (event.key === 'ArrowUp') { event.preventDefault(); options[Math.max(index - 1, 0)]?.focus(); }
    if (event.key === 'Escape') { closePicker(); $('picker-button').focus(); }
  });
  document.addEventListener('click', (event) => { if (!$('picker').contains(event.target)) closePicker(); });

  $('web').onclick = () => { state.web = !state.web; store.set('web', state.web ? '1' : '0'); renderPicker(); };
  $('thinking').onclick = () => { state.thinking = !state.thinking; store.set('thinking', state.thinking ? '1' : '0'); renderPicker(); };
  for (const button of $('effort').querySelectorAll('[data-effort]')) button.onclick = () => { state.effort = button.dataset.effort; store.set('effort', state.effort); renderPicker(); if ($('chat-log').querySelector('.hero')) showHero(); };
  for (const button of $('temperature').querySelectorAll('[data-temperature]')) button.onclick = () => { state.temperature = button.dataset.temperature; store.set('temperature', state.temperature); renderPicker(); };
  for (const button of $('agent-effort').querySelectorAll('[data-effort]')) button.onclick = () => { state.agentEffort = button.dataset.effort; setRadio($('agent-effort'), state.agentEffort, 'effort'); };
  $('agent-thinking').onclick = () => { state.agentThinking = !state.agentThinking; $('agent-thinking').setAttribute('aria-checked', String(state.agentThinking)); };

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      if (state.busy) { stop(); return; }
      closeRail(); closePicker();
    }
    if (event.altKey && !event.metaKey && !event.ctrlKey && /^Digit[1-8]$/.test(event.code)) { event.preventDefault(); setView(VIEWS[Number(event.code.slice(5)) - 1]); }
    if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === 'o') { event.preventDefault(); newChat(); }
    if (event.key === '/' && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName) && state.view === 'chat') { event.preventDefault(); $('prompt').focus(); }
  });

  $('ocr-file').onchange = () => setOcrFile($('ocr-file').files[0]);
  $('drop').addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('ocr-file').click(); } });
  for (const name of ['dragenter', 'dragover']) $('drop').addEventListener(name, (event) => { event.preventDefault(); $('drop').classList.add('is-over'); });
  for (const name of ['dragleave', 'drop']) $('drop').addEventListener(name, (event) => { event.preventDefault(); $('drop').classList.remove('is-over'); if (name === 'drop') setOcrFile(event.dataTransfer.files[0]); });
  $('ocr-run').onclick = extractText;
  $('ocr-copy').onclick = () => copyText($('ocr-text').value);
  $('ocr-text').oninput = () => { const empty = !$('ocr-text').value.trim(); $('ocr-copy').disabled = empty; $('ocr-chat').disabled = empty; };
  $('ocr-chat').onclick = () => {
    if (state.busy) return toast('Wait for the current reply first.', true);
    newChat();
    $('prompt').value = `Here is a document:\n\n${$('ocr-text').value}\n\nSummarize it and list anything that needs action.`;
    autosize();
  };

  $('image-form').onsubmit = generateImage;
  $('enhance').onclick = async () => {
    const prompt = $('image-prompt').value.trim();
    if (!prompt) return toast('Write a prompt first.', true);
    const button = $('enhance');
    button.disabled = true; button.innerHTML = '<span class="spinner"></span>refining';
    try {
      const result = await post('/api/enhance', {prompt, model: state.edit ? 'flux2-klein' : state.imageModel}, 120000);
      $('image-prompt').value = result.enhanced;
      if (result.by && result.by !== 'rules') toast(`Rewritten by ${shortModel(result.by)}`);
    } catch (error) { fail(error); }
    finally { button.disabled = false; button.textContent = 'refine prompt'; }
  };
  for (const button of $('aspect').querySelectorAll('[data-aspect]')) button.onclick = () => { if (state.imageBusy) return; state.aspect = button.dataset.aspect; setRadio($('aspect'), state.aspect, 'aspect'); };
  for (const button of $('image-model').querySelectorAll('[data-image-model]')) button.onclick = () => { if (state.imageBusy) return; state.imageModel = button.dataset.imageModel; store.set('imageModel', state.imageModel); renderImageModels(); };
  $('editing-clear').onclick = stopEdit;
  $('photo-button').onclick = () => $('photo-file').click();
  $('photo-file').onchange = () => { usePhoto($('photo-file').files[0]); $('photo-file').value = ''; };
  $('gallery-refresh').onclick = () => loadGallery().catch(fail);

  $('agent-form').onsubmit = runAgent;
  $('agent-refresh').onclick = () => loadTasks().catch(fail);
  $('unload').onclick = freeMemory;
  $('instructions-form').onsubmit = saveSystemPrompt;
  $('recall').onclick = async () => {
    const on = $('recall').getAttribute('aria-checked') !== 'true';
    try {
      await api('/api/settings', {method: 'PUT', body: JSON.stringify({recall: on})});
      $('recall').setAttribute('aria-checked', String(on));
      if (state.settings) state.settings.recall = on;
    } catch (error) { fail(error); }
  };
  $('system-prompt').oninput = () => { state.promptDirty = true; $('system-prompt-note').textContent = `${$('system-prompt').value.length.toLocaleString()} of ${(state.settings?.system_prompt_limit || 8000).toLocaleString()} characters · not saved`; };
  $('models-refresh').onclick = async () => { await refreshAll(); renderModelsView(); };
}

async function init() {
  applyTheme(initialTheme());
  bind();
  setRadio($('agent-effort'), state.agentEffort, 'effort');
  showHero();
  setView(location.hash.slice(1) || 'chat', false);
  const results = await Promise.allSettled([loadModels(), loadConversations(), refreshStatus(), loadModelStore()]);
  if (results[0].status === 'rejected') fail(results[0].reason);
  if (results[1].status === 'rejected') $('thread-list').innerHTML = '<p class="threads__empty">Could not load conversations.</p>';
  if (state.view === 'chat' && !state.conversation) showHero();
  openFromHash();
  setInterval(() => { if (!document.hidden && !state.busy) refreshStatus(); }, 8000);
}
init().catch(fail);
