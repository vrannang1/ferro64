'use strict';
/* Code view: a project's files, a plain editor, and the coding agent.
   Uses app.js helpers: $, esc, icon, api, post, toast, fail, markdown,
   readSSE, confirmDialog, shortModel, store, state. */

const code = {
  projects: [],
  project: null,
  expanded: new Set(['.']),
  tree: new Map(),
  changed: new Set(),
  tabs: [], // {path, saved, text, top, left, start, end, stale}; text is current only while the tab is in the background
  file: null, // the open tab
  runId: null,
  busy: false,
  controller: null,
  thinking: store.get('codeThinking') !== '0',
  model: store.get('codeModel') || '',
};

const codeProject = () => code.projects.find((p) => p.id === code.project);
const projectApi = (suffix = '') => `/api/code/projects/${encodeURIComponent(code.project)}${suffix}`;

async function openCodeView() {
  try {
    code.projects = await api('/api/code/projects');
  } catch (error) { fail(error); return; }
  if (!code.projects.some((p) => p.id === code.project)) code.project = store.get('codeProject');
  if (!code.projects.some((p) => p.id === code.project)) code.project = code.projects[0]?.id || null;
  renderCodeProjects();
  renderCodeModels();
  if (code.project && !code.tree.size) await loadTree('.');
  if (!code.watch) findRunning();
}

function renderCodeProjects() {
  const select = $('code-project');
  select.innerHTML = code.projects.length
    ? code.projects.map((p) => `<option value="${esc(p.id)}"${p.id === code.project ? ' selected' : ''}>${esc(p.name)}</option>`).join('') + '<option value="" disabled>──</option><option value="__remove">remove this project…</option>'
    : '<option value="">no projects yet</option>';
  select.disabled = !code.projects.length;
  const has = Boolean(code.project);
  $('code-body').hidden = !has;
  $('code-empty').hidden = has;
  for (const button of $('code-git').querySelectorAll('button')) button.disabled = !has;
  if (!has && !code.projects.length) $('code-add-form').hidden = false;
  const project = codeProject();
  $('code-project').title = project ? project.root : '';
}

function renderCodeModels() {
  const models = [...chatModels(), ...(typeof hostedModels === 'function' ? hostedModels() : [])];
  const preferred = ['qwen3.6:35b-a3b', 'qwen3.5:35b-a3b', 'glm-4.7-flash', 'qwen3.5:9b'];
  // In order of preference, not the list's alphabetical order.
  if (!models.some((m) => m.id === code.model)) code.model = (preferred.map((id) => models.find((m) => m.id === id)).find(Boolean) || models[0])?.id || '';
  models.sort((a, b) => (preferred.indexOf(b.id) >= 0) - (preferred.indexOf(a.id) >= 0) || (a.name || a.id).localeCompare(b.name || b.id));
  $('code-model').innerHTML = models.length
    ? models.map((m) => `<option value="${esc(m.id)}"${m.id === code.model ? ' selected' : ''}>${esc(m.name || shortModel(m.id))}${m.id === 'qwen3.6:35b-a3b' ? ' · coding' : ''}</option>`).join('')
    : '<option value="">no chat models</option>';
  $('code-thinking').setAttribute('aria-checked', String(code.thinking));
}

async function switchProject(id) {
  if (id === '__remove') { $('code-project').value = code.project; removeProject(); return; }
  if (code.busy) { toast('Wait for the run to finish, or stop it, before switching projects.', true); $('code-project').value = code.project; return; }
  if (!await leaveFile()) { $('code-project').value = code.project; return; }
  code.project = id; store.set('codeProject', id);
  endTerminal();
  if (!$('code-term-panel').hidden) startTerminal();
  code.tree.clear(); code.expanded = new Set(['.']); code.changed.clear(); code.runId = null;
  closeFile();
  renderPlan([]);
  $('code-log').innerHTML = '';
  renderCodeProjects();
  await loadTree('.');
  stopWatching();
  findRunning();
}

async function addProject(event) {
  event.preventDefault();
  const path = $('code-add-path').value.trim();
  if (!path) return;
  try {
    const project = await post('/api/code/projects', {path}, 20000);
    $('code-add-path').value = '';
    $('code-add-form').hidden = true;
    code.projects = await api('/api/code/projects');
    await switchProject(project.id);
    toast(`Added ${project.name}`);
  } catch (error) { fail(error); }
}

// The desktop app has a native folder picker; a browser types the path.
async function pickProject() {
  const dialog = window.__TAURI__?.dialog;
  if (!dialog) {
    $('code-add-form').hidden = !$('code-add-form').hidden;
    if (!$('code-add-form').hidden) $('code-add-path').focus();
    return;
  }
  try {
    const path = await dialog.open({directory: true, multiple: false, title: 'Choose a project folder'});
    if (!path) return;
    $('code-add-path').value = path;
    await addProject(new Event('submit'));
  } catch (error) { fail(error); }
}

async function removeProject() {
  const project = codeProject();
  if (!project) return;
  if (!await confirmDialog(`remove ${project.name.toLowerCase()}?`, `Ferro64 forgets the project and its run history. The folder ${project.root} and its files are not touched.`)) return;
  try {
    await api(projectApi(), {method: 'DELETE'});
    code.project = null; store.set('codeProject', '');
    endTerminal(); toggleTerminal(false);
    code.tree.clear(); closeFile(); $('code-log').innerHTML = '';
    await openCodeView();
  } catch (error) { fail(error); }
}

/* ---------- file tree ---------- */

async function loadTree(path) {
  try {
    code.tree.set(path, await api(`${projectApi('/tree')}?path=${encodeURIComponent(path)}`));
  } catch (error) { code.tree.set(path, []); fail(error); }
  renderTree();
}

function renderTree() {
  const rows = [];
  const walk = (path, depth) => {
    for (const entry of code.tree.get(path) || []) {
      const open = entry.dir && code.expanded.has(entry.path);
      const current = !entry.dir && code.file?.path === entry.path;
      rows.push(`<button class="code__node${code.changed.has(entry.path) ? ' is-changed' : ''}" style="--depth:${depth}" data-path="${esc(entry.path)}" data-dir="${entry.dir}"${current ? ' aria-current="true"' : ''}${entry.dir ? ` aria-expanded="${open}"` : ''} title="${esc(entry.path)}">${icon(entry.dir ? 'folder' : 'file')}<span>${esc(entry.name)}${entry.dir ? '/' : ''}</span></button>`);
      if (open) walk(entry.path, depth + 1);
    }
  };
  walk('.', 0);
  $('code-tree').innerHTML = rows.join('') || '<p class="readout" style="padding: 0 12px">empty folder</p>';
  for (const node of $('code-tree').querySelectorAll('.code__node')) {
    node.onclick = () => node.dataset.dir === 'true' ? toggleDir(node.dataset.path) : openFile(node.dataset.path);
  }
}

async function toggleDir(path) {
  if (code.expanded.has(path)) { code.expanded.delete(path); renderTree(); return; }
  code.expanded.add(path);
  if (!code.tree.has(path)) await loadTree(path); else renderTree();
}

async function refreshTree() {
  const open = [...code.expanded].filter((p) => p === '.' || code.tree.has(p));
  code.tree.clear();
  await Promise.all(open.map(async (path) => {
    try { code.tree.set(path, await api(`${projectApi('/tree')}?path=${encodeURIComponent(path)}`)); }
    catch { code.expanded.delete(path); }
  }));
  renderTree();
}

/* ---------- editor ---------- */

const tabText = (tab) => (tab === code.file ? $('code-editor').value : tab.text);
const tabDirty = (tab) => tabText(tab) !== tab.saved;
const dirtyTabs = () => code.tabs.filter(tabDirty);
const dirty = () => Boolean(code.file) && tabDirty(code.file);
const findTab = (path) => code.tabs.find((t) => t.path === path);

function renderDirty() {
  const isDirty = dirty();
  $('code-dirty').hidden = !isDirty;
  $('code-dirty').textContent = code.file?.stale ? 'unsaved · changed on disk' : 'unsaved';
  $('code-save').disabled = !isDirty;
  for (const el of $('code-tabs').children) {
    const tab = findTab(el.dataset.path);
    if (tab) { el.classList.toggle('is-dirty', tabDirty(tab)); el.classList.toggle('is-stale', Boolean(tab.stale)); }
  }
}

function tabLabel(tab) {
  const parts = tab.path.split('/');
  const name = parts.at(-1);
  // Two open files can share a name, such as index.ex in different folders.
  const shared = code.tabs.some((t) => t !== tab && t.path.split('/').at(-1) === name);
  return shared && parts.length > 1 ? parts.slice(-2).join('/') : name;
}

function renderTabs() {
  const bar = $('code-tabs');
  bar.hidden = !code.tabs.length;
  bar.innerHTML = code.tabs.map((tab) => `<div class="code__tab${tab === code.file ? ' is-active' : ''}" data-path="${esc(tab.path)}"><button type="button" class="code__tabname" role="tab" aria-selected="${tab === code.file}" title="${esc(tab.path)}">${esc(tabLabel(tab))}</button><button type="button" class="code__tabclose" aria-label="Close ${esc(tab.path)}" title="Close">${icon('x')}</button></div>`).join('');
  for (const el of bar.children) {
    const tab = findTab(el.dataset.path);
    el.querySelector('.code__tabname').onclick = () => showTab(tab);
    el.querySelector('.code__tabclose').onclick = () => closeTab(tab);
    el.onauxclick = (event) => { if (event.button === 1) { event.preventDefault(); closeTab(tab); } };
  }
  bar.querySelector('.is-active')?.scrollIntoView({block: 'nearest', inline: 'nearest'});
  renderDirty();
}

/* Keeps the open tab's text, scroll and cursor while another tab is shown. */
function stashTab() {
  const tab = code.file;
  if (!tab) return;
  const editor = $('code-editor');
  Object.assign(tab, {text: editor.value, top: editor.scrollTop, left: editor.scrollLeft, start: editor.selectionStart, end: editor.selectionEnd});
}

function showTab(tab) {
  if (tab !== code.file) stashTab();
  code.file = tab;
  const editor = $('code-editor');
  editor.value = tab.text;
  editor.disabled = false;
  editor.setSelectionRange(tab.start, tab.end);
  editor.scrollTop = tab.top;
  editor.scrollLeft = tab.left;
  $('code-path').textContent = tab.path;
  code.changed.delete(tab.path);
  renderTabs(); renderTree();
}

async function leaveFile() {
  const edited = dirtyTabs();
  if (!edited.length) return true;
  return confirmDialog('discard your edits?', edited.length === 1 ? `${edited[0].path} has unsaved changes.` : `${edited.length} open files have unsaved changes.`);
}

function closeFile() {
  code.tabs = [];
  code.file = null;
  $('code-editor').value = '';
  $('code-editor').disabled = true;
  $('code-path').textContent = 'no file open';
  renderTabs(); renderTree();
}

async function closeTab(tab) {
  if (tabDirty(tab) && !await confirmDialog('discard your edits?', `${tab.path} has unsaved changes.`)) return;
  if (!code.tabs.includes(tab)) return;
  const index = code.tabs.indexOf(tab);
  const next = tab === code.file ? code.tabs[index + 1] || code.tabs[index - 1] : code.file;
  if (tab === code.file) code.file = null; // nothing to keep from a closed tab
  code.tabs.splice(index, 1);
  if (next) showTab(next); else closeFile();
}

async function openFile(path) {
  const open = findTab(path);
  if (open) { showTab(open); return; }
  try {
    const file = await api(`${projectApi('/file')}?path=${encodeURIComponent(path)}`);
    // A second click can arrive while the first is still loading.
    let tab = findTab(path);
    if (!tab) {
      tab = {path, saved: file.content, text: file.content, top: 0, left: 0, start: 0, end: 0, stale: false};
      const at = code.tabs.indexOf(code.file);
      code.tabs.splice(at < 0 ? code.tabs.length : at + 1, 0, tab);
    }
    showTab(tab);
  } catch (error) { fail(error); }
}

/* The agent or an undo changed a file on disk. A clean tab follows it; a tab
   with unsaved edits keeps them and is marked, so saving can warn first. */
async function fileChanged(path) {
  const tab = findTab(path);
  if (!tab) return;
  try {
    const file = await api(`${projectApi('/file')}?path=${encodeURIComponent(path)}`);
    if (!code.tabs.includes(tab) || file.content === tab.saved) return;
    if (tabDirty(tab)) {
      if (!tab.stale) toast(`${path} changed on disk. Your unsaved edits are kept.`, true);
      tab.stale = true;
      renderDirty();
      return;
    }
    tab.saved = file.content;
    tab.stale = false;
    if (tab === code.file) {
      const editor = $('code-editor');
      const {scrollTop, scrollLeft, selectionStart, selectionEnd} = editor;
      editor.value = file.content;
      editor.setSelectionRange(selectionStart, selectionEnd);
      editor.scrollTop = scrollTop;
      editor.scrollLeft = scrollLeft;
      code.changed.delete(path);
    } else tab.text = file.content;
    renderDirty();
  } catch (error) { fail(error); }
}

async function saveTab(tab) {
  if (!tabDirty(tab)) return true;
  if (tab.stale && !await confirmDialog('overwrite the newer file?', `${tab.path} changed on disk after you started editing. Saving replaces that version with yours.`)) return false;
  const content = tabText(tab);
  try {
    await api(projectApi('/file'), {method: 'PUT', body: JSON.stringify({path: tab.path, content})});
    tab.saved = content;
    tab.stale = false;
    renderDirty();
    toast(`Saved ${tab.path}`);
    return true;
  } catch (error) { fail(error); return false; }
}

function saveFile() { return code.file ? saveTab(code.file) : false; }

function editorKeys(event) {
  const editor = $('code-editor');
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') { event.preventDefault(); saveFile(); return; }
  if (event.key === 'Tab' && !event.shiftKey) {
    event.preventDefault();
    // Indent with what the file already uses.
    const unit = /\n\t/.test(code.file?.saved || '') ? '\t' : /\n {2}\S/.test(code.file?.saved || '') && !/\n {4}/.test(code.file?.saved || '') ? '  ' : '    ';
    editor.setRangeText(unit, editor.selectionStart, editor.selectionEnd, 'end');
    renderDirty();
  }
}

/* ---------- agent ---------- */

/* The state line counts up, so a long wait never looks like a hang. */
function codeState(text) {
  code.state = text || '';
  code.stateAt = Date.now();
  paintCodeState();
}
function paintCodeState() {
  const seconds = Math.round((Date.now() - code.stateAt) / 1000);
  $('code-state').textContent = code.busy && code.state && seconds >= 5 ? `${code.state} · ${seconds}s` : code.state;
}

function setCodeBusy(busy) {
  code.busy = busy;
  $('code-run').textContent = busy ? 'stop' : 'run';
  $('code-run').classList.toggle('btn--primary', !busy);
  $('code-run').classList.toggle('btn--ghost', busy);
  $('code-task').disabled = busy;
  if (!busy) codeState('');
}

function logAppend(html) {
  const log = $('code-log');
  const stick = log.scrollHeight - log.scrollTop - log.clientHeight < 120;
  log.insertAdjacentHTML('beforeend', html);
  if (stick) log.scrollTop = log.scrollHeight;
  return log.lastElementChild;
}

/* The plan stays above the log, so it cannot scroll away. */
function renderPlan(items) {
  const panel = $('code-plan');
  code.plan = items || [];
  panel.hidden = !code.plan.length;
  if (!code.plan.length) { panel.innerHTML = ''; return; }
  const done = code.plan.filter((i) => i.status === 'done').length;
  const open = panel.querySelector('details')?.open ?? true;
  const mark = {done: '✓', doing: '▸', dropped: '–'};
  panel.innerHTML = `<details${open ? ' open' : ''}><summary><b>plan</b><span class="readout">${done} of ${code.plan.length} done</span></summary><ol>${code.plan.map((item) => `<li class="is-${esc(item.status)}"><span class="code__planmark" aria-hidden="true">${mark[item.status] || '○'}</span><span class="grow"><span class="code__plantext">${esc(item.text)}</span>${item.proof ? `<span class="readout">${esc(item.proof)}</span>` : ''}</span></li>`).join('')}</ol></details>`;
}

function toolBrief(name, args = {}) {
  if (name === 'update_plan') return `${(args.items || []).length} items`;
  if (name === 'save_note') return args.title || '';
  if (name === 'recall_notes') return args.query || '';
  if (name === 'run_command') return args.command || '';
  if (name === 'open_page') return args.url || '';
  if (name === 'search') return `${args.query || ''}${args.path && args.path !== '.' ? ` in ${args.path}` : ''}`;
  if (name === 'read_file' && args.start_line) return `${args.path} ${args.start_line}–${args.end_line || ''}`;
  return args.path || '';
}

function renderDiff(text) {
  return `<pre class="diff">${String(text).split('\n').map((line) => {
    const cls = line.startsWith('@@') ? 'hunk' : line.startsWith('+') ? 'add' : line.startsWith('-') ? 'del' : '';
    return `<span${cls ? ` class="${cls}"` : ''}>${esc(line) || ' '}</span>`;
  }).join('')}</pre>`;
}

function approvalCard(data) {
  const edit = data.kind === 'edit';
  const always = edit ? 'allow all edits this run' : data.always ? `always allow “${data.always}”` : '';
  const body = edit
    ? `<div class="row"><b>${data.new_file ? 'create' : 'edit'}</b><span class="readout grow">${esc(data.path)}</span></div>${renderDiff(data.diff)}`
    : `<div class="row"><b>run a command</b>${data.risky ? '<span class="tag">can delete, change the system or use the network</span>' : ''}</div><pre class="code__cmd">${esc(data.command)}</pre>`;
  const card = logAppend(`<div class="code__ask${data.risky ? ' is-risky' : ''}" data-call="${esc(data.id)}">${body}
    <input class="field" placeholder="note for the model if you deny (optional)" aria-label="Note for the model">
    <div class="row row--wrap"><button class="btn btn--primary btn--xs" data-answer="once">allow</button>${always ? `<button class="btn btn--ghost btn--xs" data-answer="always">${esc(always)}</button>` : ''}<button class="btn btn--ghost btn--xs" data-answer="deny">deny</button></div></div>`);
  codeState('waiting for you');
  card.querySelector('[data-answer="once"]').focus({preventScroll: true});
  for (const button of card.querySelectorAll('[data-answer]')) {
    button.onclick = async () => {
      for (const b of card.querySelectorAll('button')) b.disabled = true;
      const decision = button.dataset.answer;
      try {
        await post('/api/code/approve', {run_id: data.run_id, call_id: data.id, decision, note: card.querySelector('input').value.trim() || null}, 15000);
        card.classList.add('is-answered');
        card.querySelector('.row:last-child').innerHTML = `<span class="readout">${decision === 'deny' ? 'denied' : decision === 'always' ? esc(always) : 'allowed'}</span>`;
        card.querySelector('input').remove();
        codeState('working');
      } catch (error) {
        if (String(error.message).includes('no longer waiting')) closeApproval(card, 'expired: no longer waiting for an answer');
        else { fail(error); for (const b of card.querySelectorAll('button')) b.disabled = false; }
      }
    };
  }
}

/* A card that can no longer be answered stops looking as if it can. */
function closeApproval(card, text) {
  card.classList.add('is-answered');
  card.querySelector('.row:last-child').innerHTML = `<span class="readout">${esc(text)}</span>`;
  card.querySelector('input')?.remove();
}

async function runTask(event) {
  event?.preventDefault();
  if (code.busy) {
    // Closing the connection would not stop an overnight run, so ask it to.
    if (code.running) stopRun(code.running); else code.controller?.abort();
    return;
  }
  const task = $('code-task').value.trim();
  if (!task || !code.project) return;
  if (!code.model) { toast('Download a chat model first (Qwen3.6 is the coding pick).', true); return; }
  const edited = dirtyTabs();
  if (edited.length && !await confirmDialog('save first?', `${edited.length === 1 ? `${edited[0].path} has` : `${edited.length} open files have`} unsaved edits the agent would not see. Save them, then run.`)) return;
  for (const tab of edited) if (!await saveTab(tab)) return;
  $('code-task').value = '';
  logAppend(`<div class="code__task">${esc(task)}</div>`);
  stopWatching();
  const overnight = code.overnight;
  setOvernight(false);
  const controller = new AbortController();
  code.controller = controller;
  setCodeBusy(true);
  codeState('starting');
  const tools = new Map();
  let say = null; let think = null;
  const newSay = () => { say = {text: '', el: logAppend('<div class="code__say"><div class="prose"></div></div>')}; };
  try {
    const response = await fetch('/api/code/run', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, signal: controller.signal,
      body: JSON.stringify({project_id: code.project, task, model: code.model, run_id: code.runId, thinking: code.thinking, effort: 'medium', overnight: overnight}),
    });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || `Request failed (${response.status})`);
    for await (const {event: name, data} of readSSE(response)) {
      if (name === 'run') {
        code.runId = data.run_id;
        code.running = data.overnight ? data.run_id : null;
        if (data.overnight) logAppend('<p class="code__note">running by itself: you can close this page and come back to it from here</p>');
        codeState(`${shortModel(data.model)}`);
      }
      else if (name === 'status') codeState(data.phase === 'running' ? `running ${data.command}` : data.phase === 'generating' ? 'thinking' : data.phase === 'loading' ? 'loading the model' : data.phase.startsWith('reading ') ? data.phase.replace('reading', 'reading history,') : data.phase);
      else if (name === 'reasoning') {
        if (!think) { think = {text: '', el: logAppend('<details class="code__think"><summary>reasoning</summary><pre></pre></details>')}; }
        think.text += data.text; think.el.querySelector('pre').textContent = think.text;
      } else if (name === 'content') {
        think = null;
        if (!say) newSay();
        say.text += data.text; say.el.querySelector('.prose').innerHTML = markdown(say.text);
        codeState('writing');
      } else if (name === 'tool') {
        say = null; think = null;
        const el = logAppend(`<details class="code__tool is-running"><summary><b>${esc(data.name.replace('_', ' '))}</b><span class="grow">${esc(toolBrief(data.name, data.args))}</span></summary><pre class="code__output"></pre></details>`);
        tools.set(data.id, el);
      } else if (name === 'plan') renderPlan(data.items);
      else if (name === 'notes') {
        logAppend(`<details class="code__tool"><summary><b>notes found</b><span class="grow">${data.items.length}</span></summary><pre class="code__output">${esc(data.items.map((n) => `${n.level}${n.stale ? ', may be out of date' : ''}: ${n.title}`).join('\n'))}</pre></details>`);
      }
      else if (name === 'approval') approvalCard(data);
      else if (name === 'approval_expired') {
        const card = [...$('code-log').querySelectorAll('.code__ask')].find((c) => c.dataset.call === data.id);
        if (card) closeApproval(card, 'no answer in 30 minutes, so the agent was told no');
        codeState('working');
      }
      else if (name === 'change') {
        code.changed.add(data.path);
        fileChanged(data.path);
        refreshTree();
      } else if (name === 'tool_result') {
        const el = tools.get(data.id);
        if (el) { el.classList.remove('is-running'); el.classList.toggle('is-failed', !data.ok); el.querySelector('pre').textContent = data.output; }
      } else if (name === 'done') {
        logAppend(`<div class="code__done"><span>${data.status === 'completed' ? 'done' : esc(data.status)}${data.tokens && data.tokens.input + data.tokens.output ? ` · ${(data.tokens.input + data.tokens.output).toLocaleString()} tokens` : ''}${data.plan_total ? ` · ${data.plan_done} of ${data.plan_total} plan items` : ''}${data.changes ? ` · ${data.changes} change${data.changes === 1 ? '' : 's'}` : ''}</span><span class="grow"></span>${data.changes ? `<button class="btn btn--ghost btn--xs" data-undo="${esc(data.run_id)}">undo this run</button>` : ''}</div>`);
        bindUndo();
      } else if (name === 'error') logAppend(`<p class="code__error">${esc(data.error)}</p>`);
    }
  } catch (error) {
    if (error.name === 'AbortError') logAppend('<p class="code__error">stopped</p>');
    else logAppend(`<p class="code__error">${esc(error.message || error)}</p>`);
  } finally {
    for (const card of $('code-log').querySelectorAll('.code__ask:not(.is-answered)')) closeApproval(card, 'expired: the run ended');
    setCodeBusy(false);
    code.controller = null;
    code.running = null;
    refreshTree();
  }
}

function bindUndo() {
  for (const button of $('code-log').querySelectorAll('[data-undo]')) {
    button.onclick = () => undoRun(button.dataset.undo, button);
  }
}

async function undoRun(runId, button) {
  if (!await confirmDialog('undo this run?', 'The files it changed go back to how they were before it. A file you have edited since is left alone.')) return;
  button.disabled = true;
  try {
    const result = await post(`/api/code/runs/${encodeURIComponent(runId)}/undo`, {}, 30000);
    toast(result.skipped.length ? `Restored ${result.restored.length}; left ${result.skipped.join(', ')} (edited since)` : `Restored ${result.restored.length} file${result.restored.length === 1 ? '' : 's'}`);
    button.textContent = 'undone';
    for (const path of result.restored) code.changed.delete(path);
    for (const path of result.restored) fileChanged(path);
    refreshTree();
  } catch (error) { fail(error); button.disabled = false; }
}

/* Overnight runs. They work with the page closed, so a page that comes back
   finds them in the saved history and follows along from there. */
function setOvernight(on) {
  code.overnight = on;
  $('code-overnight').setAttribute('aria-checked', String(on));
}
async function toggleOvernight() {
  if (code.overnight) { setOvernight(false); return; }
  const ok = await confirmDialog('let this run work by itself?', 'The agent will approve its own edits and commands in this project and keep going until the plan is finished, even with this page closed. It cannot install software, push, or go outside the project; on a Mac the system enforces the file limits too. It stops after 12 hours, or if it keeps getting nowhere. Read and test what it did before you use it. Undo still works.');
  if (ok) setOvernight(true);
}
async function stopRun(id) {
  try {
    await post(`/api/code/runs/${encodeURIComponent(id)}/stop`, {}, 15000);
    codeState('stopping after this step');
    toast('It will stop after the current step.');
  } catch (error) { fail(error); }
}
function stopWatching() {
  clearInterval(code.watch);
  code.watch = null;
}
function watchRun(id) {
  stopWatching();
  code.watch = setInterval(async () => {
    if (code.busy || code.runId !== id) { stopWatching(); return; }
    const log = $('code-log');
    const atEnd = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
    await openRun(id);
    if (atEnd) log.scrollTop = log.scrollHeight;
  }, 10000);
}
/** A run still going when the page opens is the first thing to show. */
async function findRunning() {
  if (code.busy || !code.project) return;
  try {
    const found = await api(projectApi('/runs'));
    const running = (Array.isArray(found) ? found : found.runs || found.items || []).find((run) => run.status === 'running');
    if (running) await openRun(running.id);
  } catch (error) { console.warn(error); }
}

function newTask() {
  if (code.busy) return;
  stopWatching();
  code.runId = null;
  $('code-log').innerHTML = '';
  renderPlan([]);
  $('code-task').focus();
}

async function showHistory() {
  if (code.busy || !code.project) return;
  try {
    const runs = await api(projectApi('/runs'));
    $('code-log').innerHTML = runs.length
      ? `<div class="code__runs"><span class="label">past runs</span>${runs.map((r) => `<button class="item item--run" data-run="${esc(r.id)}"><span class="item__main"><span>${esc(r.task.slice(0, 120))}</span><span class="readout">${esc(r.status)} · ${esc(shortModel(r.model))} · ${esc(when(r.created_at))}</span></span></button>`).join('')}</div>`
      : '<p class="readout">No runs in this project yet.</p>';
    for (const row of $('code-log').querySelectorAll('[data-run]')) row.onclick = () => openRun(row.dataset.run);
  } catch (error) { fail(error); }
}

async function openRun(id) {
  try {
    const run = await api(`/api/code/runs/${encodeURIComponent(id)}`);
    code.runId = run.id;
    renderPlan(run.plan);
    const parts = [];
    for (const m of run.transcript) {
      if (m.ferro64_checkpoint) parts.push(`<details class="code__tool"><summary><b>summary of the work so far</b></summary><pre class="code__output">${esc(m.content)}</pre></details>`);
      else if (m.role === 'user') parts.push(`<div class="code__task">${esc(m.content)}</div>`);
      else if (m.role === 'assistant') {
        if (m.content) parts.push(`<div class="code__say"><div class="prose">${markdown(m.content)}</div></div>`);
        for (const call of m.tool_calls || []) {
          let args = {}; try { args = JSON.parse(call.function.arguments || '{}'); } catch { /* shown raw */ }
          parts.push(`<details class="code__tool" data-call="${esc(call.id)}"><summary><b>${esc(call.function.name.replace('_', ' '))}</b><span class="grow">${esc(toolBrief(call.function.name, args))}</span></summary><pre class="code__output"></pre></details>`);
        }
      } else if (m.role === 'tool') {
        parts.push({call: m.tool_call_id, output: m.content});
      }
    }
    $('code-log').innerHTML = parts.filter((p) => typeof p === 'string').join('');
    for (const p of parts.filter((p) => typeof p !== 'string')) {
      const pre = $('code-log').querySelector(`[data-call="${CSS.escape(p.call)}"] pre`);
      if (pre) pre.textContent = p.output;
    }
    if (run.status === 'running') {
      logAppend('<div class="code__done"><span>still working by itself · this page updates every 10 seconds</span><span class="grow"></span><button class="btn btn--ghost btn--xs" type="button" id="code-stop-run">stop it</button></div>');
      $('code-stop-run').onclick = () => stopRun(run.id);
      if (!code.watch) watchRun(run.id);
      return;
    }
    stopWatching();
    logAppend(`<div class="code__done"><span>${esc(run.status)} · the next task continues this run</span><span class="grow"></span>${run.status !== 'undone' ? `<button class="btn btn--ghost btn--xs" data-undo="${esc(run.id)}">undo this run</button>` : ''}</div>`);
    bindUndo();
  } catch (error) { fail(error); }
}

/* What Ferro64 keeps about the project. The user can correct it. */
async function showBrief() {
  if (!code.project || code.busy) return;
  try {
    const brief = await api(projectApi('/brief'));
    stopWatching();
    if (!brief.text) {
      $('code-log').innerHTML = '<div class="code__runs"><span class="label">project summary</span><p class="muted">None yet. Ferro64 writes one the first time it works on this project.</p></div>';
      return;
    }
    const state = brief.stale ? 'the project has changed a lot since this was written' : brief.edited ? 'your version, kept until you ask for a new one' : 'written by Ferro64 and given to each new session';
    $('code-log').innerHTML = `<div class="code__runs"><div class="row"><span class="label grow">project summary</span><button class="btn btn--ghost btn--xs" type="button" id="code-brief-new">write a new one next run</button><button class="btn btn--primary btn--xs" type="button" id="code-brief-save">save</button></div><p class="muted">${esc(state)}</p><textarea class="field" id="code-brief-text" rows="14">${esc(brief.text)}</textarea></div>`;
    $('code-brief-save').onclick = async () => {
      try {
        await api(projectApi('/brief'), {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text: $('code-brief-text').value})});
        toast('Summary saved.');
      } catch (error) { fail(error); }
    };
    $('code-brief-new').onclick = async () => {
      try {
        await api(projectApi('/brief'), {method: 'DELETE'});
        toast('The next run will write a new summary.');
        showBrief();
      } catch (error) { fail(error); }
    };
  } catch (error) { fail(error); }
}

/* What the agent has learned. It is the user's to read and remove. */
async function showNotes() {
  if (!code.project) return;
  try {
    const {enabled, items} = await api(projectApi('/notes'));
    const scope = (n) => (n.project_id ? 'this project' : 'every project');
    const rows = items.map((n) => `<details class="code__note${n.state === 'active' ? '' : ' is-old'}"><summary><span class="grow">${esc(n.title)}</span><span class="readout">${n.state === 'active' ? esc(scope(n)) : 'replaced'}</span></summary><p>${esc(n.body)}</p>${n.source_path ? `<p class="readout">from ${esc(n.source_path)}</p>` : ''}<div class="row">${n.state === 'active' ? '' : `<button class="btn btn--ghost btn--xs" data-note-restore="${esc(n.id)}">restore</button>`}<button class="btn btn--ghost btn--xs" data-note-delete="${esc(n.id)}">delete</button></div></details>`).join('');
    $('code-log').innerHTML = `<div class="code__runs"><div class="row"><span class="label grow">notes from earlier work</span><button class="btn btn--ghost btn--xs" id="code-notes-toggle">${enabled ? 'turn off' : 'turn on'}</button></div><p class="readout">${enabled ? 'The agent is shown matching notes and can save new ones.' : 'Off: the agent cannot see or save notes. Use this for clean model tests.'}</p>${rows || '<p class="readout">No notes yet.</p>'}</div>`;
    $('code-notes-toggle').onclick = async () => {
      try { await post('/api/code/notes/settings', {enabled: !enabled}, 15000); showNotes(); } catch (error) { fail(error); }
    };
    for (const button of $('code-log').querySelectorAll('[data-note-delete]')) {
      button.onclick = async () => {
        try { await api(`/api/code/notes/${encodeURIComponent(button.dataset.noteDelete)}`, {method: 'DELETE'}); showNotes(); } catch (error) { fail(error); }
      };
    }
    for (const button of $('code-log').querySelectorAll('[data-note-restore]')) {
      button.onclick = async () => {
        try { await post(`/api/code/notes/${encodeURIComponent(button.dataset.noteRestore)}/restore`, {}, 15000); showNotes(); } catch (error) { fail(error); }
      };
    }
  } catch (error) { fail(error); }
}

async function showPermissions() {
  if (!code.project) return;
  try {
    const allowed = await api(projectApi('/permissions'));
    $('code-log').innerHTML = `<div class="code__runs"><span class="label">always allowed in this project</span>${allowed.length ? allowed.map((p) => `<div class="row"><code class="grow">${esc(p)}</code><button class="btn btn--ghost btn--xs" data-revoke="${esc(p)}">remove</button></div>`).join('') : '<p class="readout">Nothing yet. Commands ask each time until you choose “always allow”. Commands that delete, change the system, push or use the network always ask.</p>'}</div>`;
    for (const button of $('code-log').querySelectorAll('[data-revoke]')) {
      button.onclick = async () => {
        try { await api(projectApi('/permissions'), {method: 'DELETE', body: JSON.stringify({prefix: button.dataset.revoke})}); showPermissions(); }
        catch (error) { fail(error); }
      };
    }
  } catch (error) { fail(error); }
}

/* ---------- git ---------- */

async function gitAction(action) {
  if (!code.project) return;
  const dialog = $('code-dialog');
  const input = $('code-dialog-input');
  const ok = $('code-dialog-ok');
  const output = $('code-dialog-output');
  $('code-dialog-title').textContent = `git ${action}`;
  input.hidden = action !== 'commit';
  ok.hidden = !['commit', 'push'].includes(action);
  ok.textContent = action;
  output.textContent = action === 'commit' || action === 'push' ? 'loading status…' : `running git ${action}…`;
  input.value = '';
  dialog.showModal();
  const run = async (name, body = {}) => {
    try { return await post(projectApi('/git'), {action: name, ...body}, 180000); }
    catch (error) { return {ok: false, output: error.message}; }
  };
  if (action === 'commit' || action === 'push') {
    const status = await run(action === 'push' ? 'log' : 'status');
    output.textContent = action === 'push' ? `Push these commits to the remote?\n\n${status.output}` : status.output || 'nothing to commit';
    if (action === 'commit') input.focus();
    const answer = await new Promise((resolve) => dialog.addEventListener('close', () => resolve(dialog.returnValue), {once: true}));
    if (answer !== 'confirm') return;
    if (action === 'commit' && !input.value.trim()) { toast('Write a commit message.', true); return; }
    const result = await run(action, {message: input.value.trim()});
    toast(result.ok ? `git ${action} done` : `git ${action} failed: ${result.output.split('\n').slice(-1)[0]}`, !result.ok);
    if (!result.ok) { $('code-dialog-title').textContent = `git ${action} failed`; ok.hidden = true; input.hidden = true; output.textContent = result.output; dialog.showModal(); }
    return;
  }
  const result = await run(action);
  output.textContent = result.output || '(nothing)';
}

/* ---------- terminal ---------- */

const term = {xterm: null, fit: null, socket: null, project: null, observer: null};

function loadAsset(tag, attrs) {
  return new Promise((resolve, reject) => {
    const el = Object.assign(document.createElement(tag), attrs);
    el.onload = resolve;
    el.onerror = () => reject(new Error(`Could not load ${attrs.src || attrs.href}`));
    document.head.append(el);
  });
}

// xterm.js (vendored, about 0.5 MB) loads the first time the terminal opens.
async function loadXterm() {
  if (window.Terminal && window.FitAddon) return;
  await Promise.all([
    loadAsset('script', {src: '/static/vendor/xterm/xterm.js'}),
    loadAsset('link', {rel: 'stylesheet', href: '/static/vendor/xterm/xterm.css'}),
  ]);
  await loadAsset('script', {src: '/static/vendor/xterm/addon-fit.js'});
}

// xterm wants plain colors; the theme defines oklch, so let the browser convert.
function rgb(variable) {
  const probe = document.createElement('canvas').getContext('2d');
  probe.fillStyle = getComputedStyle(document.documentElement).getPropertyValue(variable).trim() || '#000';
  probe.fillRect(0, 0, 1, 1);
  const [r, g, b] = probe.getImageData(0, 0, 1, 1).data;
  return `rgb(${r}, ${g}, ${b})`;
}

function termTheme() {
  return {background: rgb('--color-code'), foreground: rgb('--color-ink'), cursor: rgb('--color-accent'), selectionBackground: 'rgba(128, 128, 160, 0.35)'};
}

/* app.js calls this after switching day and night. */
function refreshTerminalTheme() {
  if (term.xterm) term.xterm.options.theme = termTheme();
}

function sendResize() {
  if (term.socket?.readyState === WebSocket.OPEN && term.xterm) term.socket.send(JSON.stringify({resize: [term.xterm.cols, term.xterm.rows]}));
}

function endTerminal() {
  term.socket?.close();
  term.socket = null;
  term.project = null;
}

async function startTerminal() {
  try { await loadXterm(); } catch (error) { fail(error); return; }
  if (!term.xterm) {
    term.xterm = new window.Terminal({fontFamily: '"Geist Mono", ui-monospace, "SF Mono", monospace', fontSize: 12.5, cursorBlink: true, scrollback: 5000, theme: termTheme(), allowProposedApi: false});
    term.fit = new window.FitAddon.FitAddon();
    term.xterm.loadAddon(term.fit);
    term.xterm.open($('code-term'));
    const encoder = new TextEncoder();
    term.xterm.onData((data) => { if (term.socket?.readyState === WebSocket.OPEN) term.socket.send(encoder.encode(data)); });
    term.observer = new ResizeObserver(() => { if (!$('code-term-panel').hidden) { term.fit.fit(); sendResize(); } });
    term.observer.observe($('code-term'));
  }
  endTerminal();
  term.xterm.reset();
  term.fit.fit();
  const project = codeProject();
  if (!project) return;
  term.project = project.id;
  $('code-term-title').textContent = `terminal · ${project.name}`;
  $('code-term-title').title = project.root;
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${scheme}://${location.host}${projectApi('/terminal')}`);
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => { sendResize(); term.xterm.focus(); };
  socket.onmessage = (event) => term.xterm.write(typeof event.data === 'string' ? event.data : new Uint8Array(event.data));
  socket.onclose = () => { if (term.socket === socket) { term.xterm.write('\r\n\x1b[2m[shell ended · restart for a new one]\x1b[0m\r\n'); term.socket = null; } };
  term.socket = socket;
}

async function toggleTerminal(show = $('code-term-panel').hidden) {
  $('code-term-panel').hidden = !show;
  $('code-term-toggle').setAttribute('aria-expanded', String(show));
  if (!show) return;
  // A new shell when there is none, or the project changed.
  if (!term.socket || term.project !== code.project) await startTerminal();
  else { term.fit.fit(); sendResize(); term.xterm.focus(); }
}

/* ---------- wiring ---------- */

function bindCode() {
  $('code-project').onchange = () => switchProject($('code-project').value);
  $('code-add').onclick = pickProject;
  $('code-add-cancel').onclick = () => { $('code-add-form').hidden = true; };
  $('code-add-form').onsubmit = addProject;
  $('code-log').insertAdjacentHTML('beforebegin', '<div class="code__plan" id="code-plan" hidden></div>');
  document.querySelector('.code__file').insertAdjacentHTML('beforebegin', '<div class="code__tabs" id="code-tabs" role="tablist" aria-label="Open files" hidden></div>');
  $('code-editor').addEventListener('input', renderDirty);
  $('code-editor').addEventListener('keydown', editorKeys);
  $('code-save').onclick = saveFile;
  $('code-form').onsubmit = runTask;
  $('code-task').addEventListener('keydown', (event) => { if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) runTask(event); });
  $('code-model').onchange = () => { code.model = $('code-model').value; store.set('codeModel', code.model); };
  $('code-overnight').onclick = toggleOvernight;
  $('code-thinking').onclick = () => { code.thinking = !code.thinking; store.set('codeThinking', code.thinking ? '1' : '0'); renderCodeModels(); };
  $('code-new').onclick = newTask;
  $('code-history').onclick = showHistory;
  $('code-perms').insertAdjacentHTML('beforebegin', '<button class="btn btn--ghost btn--xs" type="button" id="code-notes">notes</button>');
  $('code-notes').onclick = showNotes;
  $('code-perms').insertAdjacentHTML('beforebegin', '<button class="btn btn--ghost btn--xs" type="button" id="code-brief">summary</button>');
  $('code-brief').onclick = showBrief;
  setInterval(paintCodeState, 1000);
  $('code-perms').onclick = showPermissions;
  for (const button of $('code-git').querySelectorAll('[data-git]')) button.onclick = () => gitAction(button.dataset.git);
  $('code-term-toggle').onclick = () => toggleTerminal();
  $('code-term-close').onclick = () => toggleTerminal(false);
  $('code-term-restart').onclick = () => startTerminal();
  document.addEventListener('keydown', (event) => {
    if (state.view === 'code' && event.ctrlKey && event.key === '`' && code.project) { event.preventDefault(); toggleTerminal(); }
  });
  window.addEventListener('beforeunload', (event) => { if (dirtyTabs().length) { event.preventDefault(); event.returnValue = ''; } });
}
bindCode();
if (state.view === 'code') openCodeView();
