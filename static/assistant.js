'use strict';
/* Assistant view: standing jobs that research the web on a schedule and
   leave a report. It suggests; the user decides, and marks each pick as
   bought, passed or watching. Runs happen on the server, so this page only
   polls. Uses app.js helpers: $, esc, icon, api, post, toast, fail, duration,
   when, confirmDialog, markdown, bindCodeCopy, shortModel, chatModels, state. */

const asst = {
  jobs: [],
  runs: [],
  editing: null, // job id, or '' for a new one
  enabled: true,
  table: false,
  open: null, // the run shown in the report panel
  timer: null,
};

const STOCK_TEMPLATE = `My watchlist: AAPL, MSFT, NVDA, VOO
How I invest: long-term, holding 1 to 3 years. Moderate risk is fine. I don't trade options.

Each run:
1. Get quotes for the whole watchlist, and price history for anything that moved more than 3% or has news.
2. Search for news from the last few days on each ticker: earnings, guidance, analyst rating changes, lawsuits, regulation, management changes.
3. Check the broad market: the S&P 500 (use SPY), interest-rate news and big economic events this week.
4. Suggest at most two new ideas outside my watchlist, and only with a clear, well-sourced reason. Say why each fits how I invest.

Write the report as:
## Market today
Two or three sentences.
## Watchlist
For each ticker: the price and move, what happened, and what it means for a long-term holder.
## New ideas
Only if there is a real reason; otherwise say there are none today.
## What I would consider
For each ticker: buy, hold, sell, watch or avoid, with the reason and the main risk.
## Check before acting
What I should verify myself before I decide.

Then add the picks block.`;

const LIST_TEMPLATE = `Each day, find 10 medical doctors practising in Kolkata and Hyderabad (5 in each city).

Look on hospital websites (doctor profile pages), clinic sites and medical directories. Prefer a doctor's profile page on their hospital's own site.
For each doctor, give their full name, specialty, the hospital or clinic, the area and city, a public phone number for the clinic or hospital if one is listed, and the page you found them on.
Only list doctors you found on a page in this run. If you can't find 10 solid new ones, list fewer and say so.

In the report, say briefly where you looked and anything that looked unreliable.`;
const LIST_COLUMNS = 'name, hospital or clinic, specialty, area, city, phone, source url';

const ACTIONS = {buy: 'Consider buying', hold: 'Hold', sell: 'Consider selling', watch: 'Watch', avoid: 'Avoid'};
const DECISIONS = ['bought', 'sold', 'passed', 'watching'];

async function openAssistantView() {
  fillAssistantModels();
  await Promise.allSettled([loadAssistantJobs(), loadAssistantRuns(), loadDecisions(false), loadTelegram(), loadEmail()]).then((results) => {
    const failed = results.find((r) => r.status === 'rejected');
    if (failed) fail(failed.reason);
  });
  pollAssistant();
}

function fillAssistantModels() {
  const select = $('asst-model');
  const old = select.value;
  const models = typeof chatModels === 'function' ? chatModels() : [];
  select.innerHTML = '<option value="">best installed</option>' + models.map((m) => `<option value="${esc(m.id)}">${esc(m.name || shortModel(m.id))}</option>`).join('');
  select.value = models.some((m) => m.id === old) ? old : '';
}

function scheduleText(job) {
  if (!job.time) return 'Only when you press run';
  return `${job.weekdays_only ? 'Weekdays' : 'Every day'} at ${job.time}`;
}

const statusTag = (status) => `<span class="tag status-tag ${status === 'failed' ? 'is-failed' : status === 'running' || status === 'queued' ? 'is-running' : ''}">${esc(String(status || '').replace(/^./, (c) => c.toUpperCase()))}</span>`;

async function loadAssistantJobs() {
  asst.jobs = await api('/api/assistant/jobs');
  const list = $('asst-jobs');
  if (!asst.jobs.length) {
    list.innerHTML = `<p class="empty">No jobs yet. Press <strong>New job</strong> and start from the stock research template, or write your own: "every Monday, summarize what changed in local AI models".</p>`;
    return;
  }
  list.innerHTML = asst.jobs.map(({job, last_run}, i) => `
    <div class="item item--job${job.enabled ? '' : ' is-off'}">
      <span class="item__main">
        <span class="item__title"><strong>${esc(job.name)}</strong>${job.enabled ? '' : '<span class="tag">Off</span>'}</span>
        <span class="item__specs"><span class="readout">${esc(scheduleText(job))}</span><span class="readout">${esc(job.model ? shortModel(job.model) : 'Best installed model')}</span>${last_run ? `<span class="readout">last ${esc(when(last_run.started_at))}</span>${statusTag(last_run.status)}` : ''}</span>
      </span>
      <span class="row">
        ${job.table_columns ? `<a class="btn btn--ghost btn--xs" href="/api/assistant/jobs/${encodeURIComponent(job.id)}/csv" download>all rows · ${asst.jobs[i].rows}</a>` : ''}
        <button class="btn btn--primary btn--xs" data-run="${i}">Run now</button>
        <button class="icon-btn" data-edit="${i}" aria-label="Edit ${esc(job.name)}">${icon('pen')}</button>
        <button class="icon-btn" data-delete="${i}" aria-label="Delete ${esc(job.name)}">${icon('trash')}</button>
      </span>
    </div>`).join('');
  for (const b of list.querySelectorAll('[data-run]')) b.onclick = () => runAssistantJob(asst.jobs[Number(b.dataset.run)].job, b);
  for (const b of list.querySelectorAll('[data-edit]')) b.onclick = () => editAssistantJob(asst.jobs[Number(b.dataset.edit)].job);
  for (const b of list.querySelectorAll('[data-delete]')) b.onclick = () => deleteAssistantJob(asst.jobs[Number(b.dataset.delete)].job);
}

function editAssistantJob(job) {
  asst.editing = job ? job.id : '';
  $('asst-form').hidden = false;
  $('asst-name').value = job?.name || '';
  $('asst-instructions').value = job?.instructions || '';
  $('asst-schedule').value = !job ? 'weekdays' : !job.time ? 'manual' : job.weekdays_only ? 'weekdays' : 'daily';
  $('asst-time').value = job?.time || '07:30';
  fillAssistantModels();
  $('asst-model').value = job?.model || '';
  setTable(Boolean(job?.table_columns), job?.table_columns || '');
  asst.enabled = job ? job.enabled : true;
  $('asst-enabled').setAttribute('aria-checked', String(asst.enabled));
  syncSchedule();
  loadNotes(job).catch(fail);
  $('asst-form').scrollIntoView({behavior: 'smooth', block: 'start'});
  $('asst-name').focus();
}

async function loadNotes(job) {
  const box = $('asst-notes');
  if (!job) { box.hidden = true; box.innerHTML = ''; return; }
  const notes = await api(`/api/assistant/jobs/${encodeURIComponent(job.id)}/notes`);
  box.hidden = !notes.length;
  box.innerHTML = notes.length ? `<span class="label">notes it keeps for itself · it reads these at the start of each run</span><ul class="asst-list asst-notes">${notes.map((n) => `<li><span>${esc(n.text)}</span> <span class="readout">${esc(when(n.created_at))}</span> <button type="button" class="icon-btn" data-note="${esc(n.id)}" aria-label="Delete note">${icon('x')}</button></li>`).join('')}</ul>` : '';
  for (const b of box.querySelectorAll('[data-note]')) b.onclick = async () => {
    try { await api(`/api/assistant/notes/${encodeURIComponent(b.dataset.note)}`, {method: 'DELETE'}); await loadNotes(job); } catch (error) { fail(error); }
  };
}

function setTable(on, columns) {
  asst.table = on;
  $('asst-table').setAttribute('aria-checked', String(on));
  $('asst-columns').hidden = !on;
  if (columns !== undefined) $('asst-columns').value = columns;
}

function syncSchedule() { $('asst-time').disabled = $('asst-schedule').value === 'manual'; }

async function saveAssistantJob(event) {
  event.preventDefault();
  const schedule = $('asst-schedule').value;
  const body = {
    id: asst.editing || undefined,
    name: $('asst-name').value.trim(),
    instructions: $('asst-instructions').value.trim(),
    time: schedule === 'manual' ? null : $('asst-time').value,
    weekdays_only: schedule === 'weekdays',
    enabled: asst.enabled,
    model: $('asst-model').value || null,
    table_columns: asst.table ? $('asst-columns').value.trim() : null,
  };
  if (asst.table && body.table_columns.split(',').filter((c) => c.trim()).length < 2) return toast('Name at least two columns, separated by commas.', true);
  if (!body.name || !body.instructions) return toast('Give the job a name and instructions.', true);
  if (schedule !== 'manual' && !body.time) return toast('Pick a time, or choose "only when I press run".', true);
  try {
    const saved = await post('/api/assistant/jobs', body, 20000);
    $('asst-form').hidden = true;
    asst.editing = null;
    const now = new Date();
    const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    toast(saved.time ? `Saved. ${saved.last_run_on === today ? 'First run is on the next scheduled day; press run now to try it today.' : `It runs ${scheduleText(saved)}.`}` : 'Saved. Press run now whenever you want a report.');
    await loadAssistantJobs();
  } catch (error) { fail(error); }
}

async function deleteAssistantJob(job) {
  if (!await confirmDialog(`delete “${job.name}”?`, 'Its reports and your decisions on them are removed from this machine.')) return;
  try {
    await api(`/api/assistant/jobs/${encodeURIComponent(job.id)}`, {method: 'DELETE'});
    if (asst.open?.job_id === job.id) { asst.open = null; $('asst-report').innerHTML = ''; }
    await Promise.all([loadAssistantJobs(), loadAssistantRuns(), loadDecisions(false)]);
  } catch (error) { fail(error); }
}

async function runAssistantJob(job, button) {
  button.disabled = true;
  try {
    const run = await post(`/api/assistant/jobs/${encodeURIComponent(job.id)}/run`, {}, 30000);
    toast('Started. A run usually takes a few minutes; you can leave this page.');
    await Promise.all([loadAssistantJobs(), loadAssistantRuns()]);
    showAssistantRun(run.id);
  } catch (error) { fail(error); } finally { button.disabled = false; }
}

async function loadAssistantRuns() {
  asst.runs = await api('/api/assistant/runs');
  const list = $('asst-runs');
  if (!asst.runs.length) { list.innerHTML = '<p class="empty">Reports appear here, newest first, with every search and source the assistant used.</p>'; return; }
  list.innerHTML = asst.runs.map((r, i) => {
    const picks = (r.picks || []).map((p) => `<span class="tag pick-tag is-${esc(p.action)}">${esc(p.symbol)} · ${esc(ACTIONS[p.action] || p.action)}</span>`).join('');
    const length = r.finished_at ? duration(new Date(r.finished_at) - new Date(r.started_at)) : '';
    return `<button class="item item--run" data-open="${i}"><span class="item__main"><span class="item__title"><strong>${esc(r.job_name)}</strong></span>
      <span class="item__specs"><span class="readout">${esc(new Date(r.started_at).toLocaleString(undefined, {weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'}))}</span><span class="readout">${r.trigger === 'schedule' ? 'Scheduled' : 'Manual'}</span><span class="readout">${r.sources} sources</span>${r.rows != null ? `<span class="readout">${r.rows} new rows</span>` : ''}${length ? `<span class="readout">${esc(length)}</span>` : ''}</span>
      ${picks ? `<span class="pick-tags">${picks}</span>` : ''}</span>${statusTag(r.status)}</button>`;
  }).join('');
  for (const b of list.querySelectorAll('[data-open]')) b.onclick = () => showAssistantRun(asst.runs[Number(b.dataset.open)].id, true);
}

function citeSources(html, sources) {
  const byN = new Map((sources || []).map((s) => [String(s.n), s]));
  const link = (n) => {
    const s = byN.get(n);
    if (!s || !/^https?:\/\//i.test(s.url)) return `S${esc(n)}`;
    return `<a class="cite" href="${esc(s.url)}" target="_blank" rel="noopener noreferrer" title="${esc(s.title)}">${esc(n)}</a>`;
  };
  // [S1], and lists the model writes as [S3, S4] or [S3][S4].
  return html.replace(/\[(S\d+(?:\s*,\s*S?\d+)*)\]/g, (match, list) => list.split(',').map((n) => link(n.trim().replace(/^S/, ''))).join(''));
}

async function showAssistantRun(id, scroll = false) {
  let data;
  try { data = await api(`/api/assistant/runs/${encodeURIComponent(id)}`); } catch (error) { return fail(error); }
  const {run, decisions} = data;
  asst.open = run;
  const decided = new Map((decisions || []).map((d) => [d.symbol, d]));
  const live = run.status === 'queued' || run.status === 'running';
  const head = `<div class="result__head"><h3>${esc(run.job_name)}</h3><span class="row"><span class="readout">${esc(new Date(run.started_at).toLocaleString())} · ${esc(shortModel(run.model))}</span>
    ${!live && run.report ? `<button class="icon-btn" id="asst-copy" aria-label="Copy report">${icon('copy')}</button>` : ''}
    ${!live ? `<button class="icon-btn" id="asst-delete-run" aria-label="Delete report">${icon('trash')}</button>` : ''}</span></div>`;
  let body;
  if (live) {
    const last = run.steps[run.steps.length - 1];
    body = `<p class="readout"><span class="spinner"></span> ${run.status === 'queued' ? 'waiting for the model' : `researching · ${run.steps.length} steps · ${run.sources.length} sources${last ? ` · last: ${esc(last.tool)} ${esc(last.input)}` : ''}`}</p>`;
  } else if (run.status === 'failed') {
    body = `<p class="empty">${esc(run.error || 'The run failed.')}</p>`;
  } else {
    body = `<div class="prose">${citeSources(markdown(run.report || ''), run.sources)}</div>`;
  }
  const picks = (run.picks || []).map((p) => {
    const d = decided.get(p.symbol);
    return `<div class="pick is-${esc(p.action)}" data-symbol="${esc(p.symbol)}">
      <div class="pick__head"><strong>${esc(p.symbol)}</strong><span class="tag pick-tag is-${esc(p.action)}">${esc(ACTIONS[p.action] || p.action)}</span>${p.conviction ? `<span class="readout">${esc(p.conviction)} conviction</span>` : ''}<span class="grow"></span>${p.price ? `<span class="readout">$${p.price.toFixed(2)} at the report</span>` : ''}</div>
      ${p.thesis ? `<p>${citeSources(esc(p.thesis), run.sources)}</p>` : ''}
      ${p.risks ? `<p class="pick__risk"><span class="label">Risk</span> ${citeSources(esc(p.risks), run.sources)}</p>` : ''}
      <div class="row row--wrap pick__decide"><span class="label">Your call</span>${DECISIONS.map((x) => `<button type="button" class="btn btn--xs ${d?.decision === x ? 'btn--primary' : 'btn--ghost'}" data-decide="${x}">${x}</button>`).join('')}
        <input class="field pick__note" placeholder="why (optional; the assistant reads this next time)" value="${esc(d?.note || '')}" maxlength="2000"></div>
    </div>`;
  }).join('');
  const t = run.table;
  const tableBlock = !t ? '' : `<div class="section-head section-head--tight"><h3>Table</h3><span class="row"><span class="readout">${t.rows.length} new${t.repeats.length ? ` · ${t.repeats.length} already saved, left out` : ''}</span>${t.rows.length ? `<a class="btn btn--ghost btn--xs" href="/api/assistant/runs/${encodeURIComponent(run.id)}/csv" download>Download CSV</a>` : ''}<a class="btn btn--ghost btn--xs" href="/api/assistant/jobs/${encodeURIComponent(run.job_id)}/csv" download>All rows</a></span></div>
    ${t.rows.length ? `<div class="asst-table"><table><thead><tr>${t.columns.map((c) => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${t.rows.map((r) => `<tr>${t.columns.map((_, i) => {
      const v = r[i] || '';
      return `<td>${/^https?:\/\/\S+$/i.test(v) ? `<a href="${esc(v)}" target="_blank" rel="noopener noreferrer">${esc(v.replace(/^https?:\/\/(www\.)?/i, '').slice(0, 40))}</a>` : esc(v)}</td>`;
    }).join('')}</tr>`).join('')}</tbody></table></div>` : '<p class="empty">No new rows this time. The report says why.</p>'}`;
  const reminders = (run.reminders || []).map((r, i) => `<div class="pick"><div class="pick__head"><strong class="pick__title">${esc(r.title)}</strong><span class="readout">${esc(r.date)}${r.time ? ` · ${esc(r.time)}` : ' · all day'}</span><span class="grow"></span><a class="btn btn--ghost btn--xs" href="/api/assistant/runs/${encodeURIComponent(run.id)}/reminders/${i}" download>Add to calendar</a></div>${r.note ? `<p>${esc(r.note)}</p>` : ''}</div>`).join('');
  const sources = (run.sources || []).map((s) => `<li><span class="readout">S${s.n}</span> <a href="${/^https?:\/\//i.test(s.url) ? esc(s.url) : '#'}" target="_blank" rel="noopener noreferrer">${esc(s.title || s.url)}</a>${s.published ? ` <span class="readout">${esc(s.published)}</span>` : ''}</li>`).join('');
  const steps = (run.steps || []).map((s) => `<li><span class="readout">${esc(s.tool)}</span> ${esc(s.input)} <span class="readout">${s.ok ? '' : 'failed · '}${esc(duration(s.ms))}</span><br><small>${esc(s.summary)}</small></li>`).join('');
  $('asst-report').innerHTML = `<article class="well result asst-result">${head}${body}
    ${tableBlock}
    ${reminders ? `<div class="section-head section-head--tight"><h3>Reminders</h3><span class="readout">suggested · opening one asks your calendar to add it</span></div><div class="picks">${reminders}</div>` : ''}
    ${picks ? `<div class="section-head section-head--tight"><h3>Picks</h3><span class="readout">Suggestions, not advice · you decide</span></div><div class="picks">${picks}</div>` : ''}
    ${sources ? `<details class="step"><summary><span class="readout">sources · ${run.sources.length}</span></summary><ol class="asst-list">${sources}</ol></details>` : ''}
    ${steps ? `<details class="step"><summary><span class="readout">what it did · ${run.steps.length} steps</span></summary><ol class="asst-list">${steps}</ol></details>` : ''}
  </article>`;
  bindCodeCopy($('asst-report'));
  if ($('asst-copy')) $('asst-copy').onclick = () => copyText(run.report || '');
  if ($('asst-delete-run')) $('asst-delete-run').onclick = async () => {
    if (!await confirmDialog('delete this report?', 'Your decisions on its picks are removed too.')) return;
    try {
      await api(`/api/assistant/runs/${encodeURIComponent(run.id)}`, {method: 'DELETE'});
      asst.open = null; $('asst-report').innerHTML = '';
      await Promise.all([loadAssistantRuns(), loadAssistantJobs(), loadDecisions(false)]);
    } catch (error) { fail(error); }
  };
  for (const card of $('asst-report').querySelectorAll('.pick')) {
    for (const b of card.querySelectorAll('[data-decide]')) b.onclick = () => decide(run.id, card.dataset.symbol, b.dataset.decide, card.querySelector('.pick__note').value);
  }
  if (scroll) $('asst-report').scrollIntoView({behavior: 'smooth', block: 'start'});
}

async function decide(runId, symbol, decision, note) {
  try {
    await post('/api/assistant/decisions', {run_id: runId, symbol, decision, note}, 20000);
    toast(`${symbol}: ${decision}`);
    await Promise.all([showAssistantRun(runId), loadDecisions(false)]);
  } catch (error) { fail(error); }
}

async function loadDecisions(live) {
  const button = $('asst-prices');
  if (live) { button.disabled = true; button.innerHTML = '<span class="spinner"></span>checking'; }
  try {
    const list = await api(`/api/assistant/decisions${live ? '?live=true' : ''}`, {timeout: 60000});
    const box = $('asst-decisions');
    if (!list.length) { box.innerHTML = '<p class="empty">When you mark a pick as bought, sold, passed or watching, it is kept here with the price at the time, so you can see how the assistant\'s ideas and your calls worked out.</p>'; return; }
    const change = (d) => {
      if (d.current == null || !d.price) return '<span class="readout">–</span>';
      const pct = (d.current / d.price - 1) * 100;
      return `<span class="readout ${pct >= 0 ? 'is-up' : 'is-down'}">$${d.current.toFixed(2)} · ${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%</span>`;
    };
    box.innerHTML = `<div class="well asst-table"><table><thead><tr><th>Date</th><th>Ticker</th><th>It said</th><th>You</th><th>Price then</th><th>Since</th><th>Note</th><th></th></tr></thead><tbody>
      ${list.map((d) => `<tr><td class="readout">${esc(new Date(d.decided_at).toLocaleDateString(undefined, {month: 'short', day: 'numeric'}))}</td><td><strong>${esc(d.symbol)}</strong></td><td>${d.action ? `<span class="tag pick-tag is-${esc(d.action)}">${esc(ACTIONS[d.action] || d.action)}</span>` : ''}</td><td>${esc(d.decision)}</td><td class="readout">${d.price ? `$${d.price.toFixed(2)}` : '–'}</td><td>${change(d)}</td><td><small>${esc(d.note)}</small></td><td><button class="icon-btn" data-undo="${esc(d.id)}" aria-label="Remove decision">${icon('x')}</button></td></tr>`).join('')}
    </tbody></table></div>`;
    for (const b of box.querySelectorAll('[data-undo]')) b.onclick = async () => {
      try { await api(`/api/assistant/decisions/${encodeURIComponent(b.dataset.undo)}`, {method: 'DELETE'}); await loadDecisions(false); if (asst.open) showAssistantRun(asst.open.id); } catch (error) { fail(error); }
    };
  } finally {
    if (live) { button.disabled = false; button.textContent = 'Check prices now'; }
  }
}

/* Telegram: reports to the user's own chat. */
async function loadTelegram() {
  const t = await api('/api/assistant/telegram');
  $('tg-token').value = '';
  $('tg-token').placeholder = t.token ? `saved (${t.token}); paste a new one to replace it` : 'bot token, like 123456789:AA…';
  $('tg-remove').hidden = !t.token;
  const select = $('tg-chat');
  if (t.chat && ![...select.options].some((o) => o.value === t.chat)) select.add(new Option(`chat ${t.chat}`, t.chat));
  select.value = t.chat || '';
  $('tg-state').textContent = t.token && t.chat ? 'Telegram · on' : t.token ? 'Telegram · pick your chat' : 'Telegram · off';
}
async function saveTelegram(body) {
  try { await api('/api/assistant/telegram', {method: 'PUT', body: JSON.stringify(body)}); await loadTelegram(); return true; } catch (error) { fail(error); return false; }
}
$('tg-save').onclick = async () => { const token = $('tg-token').value.trim(); if (!token) return toast('Paste the token from BotFather first.', true); if (await saveTelegram({token})) toast('Token saved. Send your bot "hi", then press find my chat.'); };
$('tg-remove').onclick = async () => { if (await confirmDialog('stop sending to telegram?', 'The bot token and chat are removed from this machine.')) saveTelegram({token: ''}); };
$('tg-find').onclick = async () => {
  try {
    const chats = await post('/api/assistant/telegram/chats', {}, 20000);
    const select = $('tg-chat');
    select.innerHTML = '<option value="">no chat picked</option>' + chats.map((c) => `<option value="${esc(c.id)}">${esc(c.name)} (${esc(c.id)})</option>`).join('');
    if (chats.length === 1) { select.value = chats[0].id; await saveTelegram({chat: chats[0].id}); toast(`Picked ${chats[0].name}. Press send a test.`); }
    else toast('Pick your chat from the list.');
  } catch (error) { fail(error); }
};
$('tg-chat').onchange = () => saveTelegram({chat: $('tg-chat').value});
$('tg-test').onclick = async () => { try { await post('/api/assistant/telegram/test', {}, 20000); toast('Sent. Check Telegram.'); } catch (error) { fail(error); } };

/* Email: reports from the user's own mail account. */
const em = {presets: [], saved: null};
async function loadEmail() {
  const data = await api('/api/assistant/email');
  em.presets = data.presets || [];
  em.saved = data.settings;
  $('em-provider').innerHTML = em.presets.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join('');
  const s = em.saved;
  if (s) {
    $('em-provider').value = s.provider || 'other';
    $('em-host').value = s.host; $('em-port').value = s.port; $('em-security').value = s.security;
    $('em-user').value = s.username; $('em-to').value = s.to;
    $('em-pass').value = ''; $('em-pass').placeholder = s.has_password ? 'saved; type a new one to replace it' : 'abcd efgh ijkl mnop';
  } else {
    $('em-provider').value = 'gmail';
    applyPreset();
  }
  $('em-remove').hidden = !s;
  $('em-state').textContent = s ? `Email · on · to ${s.to}` : 'Email · off';
  showPresetHelp();
}
function showPresetHelp() {
  const p = em.presets.find((x) => x.id === $('em-provider').value);
  $('em-help').textContent = p ? p.help : '';
  const fixed = p && p.id !== 'other';
  for (const id of ['em-host', 'em-port', 'em-security']) $(id).readOnly = fixed;
}
function applyPreset() {
  const p = em.presets.find((x) => x.id === $('em-provider').value);
  if (!p) return;
  if (p.id !== 'other') { $('em-host').value = p.host; $('em-port').value = p.port; $('em-security').value = p.security; }
  else if (em.saved?.provider !== 'other') { $('em-host').value = ''; $('em-port').value = 587; $('em-security').value = 'starttls'; }
  showPresetHelp();
}
$('em-provider').onchange = applyPreset;
$('em-form').onsubmit = async (event) => {
  event.preventDefault();
  const user = $('em-user').value.trim();
  const body = {provider: $('em-provider').value, host: $('em-host').value.trim(), port: Number($('em-port').value), security: $('em-security').value,
    username: user, password: $('em-pass').value || null, from: user || $('em-to').value.split(',')[0].trim(), to: $('em-to').value.trim()};
  try { await api('/api/assistant/email', {method: 'PUT', body: JSON.stringify(body)}); await loadEmail(); toast('Saved. Press send a test to check it.'); }
  catch (error) { fail(error); }
};
$('em-test').onclick = async () => {
  const button = $('em-test');
  button.disabled = true; button.innerHTML = '<span class="spinner"></span>sending';
  try { const r = await post('/api/assistant/email/test', {}, 60000); toast(`Sent to ${r.to}. Check your inbox (and spam, the first time).`); }
  catch (error) { fail(error); }
  finally { button.disabled = false; button.textContent = 'Send a test'; }
};
$('em-remove').onclick = async () => {
  if (!await confirmDialog('stop emailing reports?', 'Your mail settings and app password are removed from this machine.')) return;
  try { await api('/api/assistant/email', {method: 'DELETE'}); em.saved = null; await loadEmail(); } catch (error) { fail(error); }
};

/* While a run is going, refresh it every few seconds. */
function pollAssistant() {
  clearInterval(asst.timer);
  asst.timer = setInterval(async () => {
    if (state.view !== 'assistant' || document.hidden) return;
    const wasLive = asst.runs.some((r) => r.status === 'queued' || r.status === 'running');
    try {
      await loadAssistantRuns();
      const live = asst.runs.some((r) => r.status === 'queued' || r.status === 'running');
      if (wasLive || live) await loadAssistantJobs();
      if (asst.open && (asst.open.status === 'queued' || asst.open.status === 'running')) await showAssistantRun(asst.open.id);
    } catch { /* best effort */ }
  }, 4000);
}

$('asst-new').onclick = () => editAssistantJob(null);
$('asst-cancel').onclick = () => { $('asst-form').hidden = true; asst.editing = null; };
$('asst-form').onsubmit = saveAssistantJob;
$('asst-schedule').onchange = syncSchedule;
$('asst-enabled').onclick = () => { asst.enabled = !asst.enabled; $('asst-enabled').setAttribute('aria-checked', String(asst.enabled)); };
$('asst-template').onclick = () => {
  if ($('asst-instructions').value.trim() && $('asst-instructions').value.trim() !== STOCK_TEMPLATE && !confirm('Replace what you wrote with the template?')) return;
  if (!$('asst-name').value.trim()) $('asst-name').value = 'Morning stock research';
  $('asst-instructions').value = STOCK_TEMPLATE;
  $('asst-instructions').focus();
};
$('asst-table').onclick = () => setTable(!asst.table);
$('asst-template-list').onclick = () => {
  if ($('asst-instructions').value.trim() && $('asst-instructions').value.trim() !== LIST_TEMPLATE && !confirm('Replace what you wrote with the template?')) return;
  if (!$('asst-name').value.trim()) $('asst-name').value = 'Doctors in Kolkata and Hyderabad';
  $('asst-instructions').value = LIST_TEMPLATE;
  setTable(true, LIST_COLUMNS);
  $('asst-instructions').focus();
};
$('asst-refresh').onclick = () => Promise.all([loadAssistantRuns(), loadAssistantJobs()]).catch(fail);
$('asst-prices').onclick = () => loadDecisions(true).catch(fail);
if (state.view === 'assistant') openAssistantView();
