'use strict';
/* API models: vendor keys, which of their models to offer, and the one address
   that reaches them all. Keys go in and never come back out. */

const hosted = {vendors: [], token_limit: 0, proxy_token: '', found: {}};

function hostedModels() {
  return hosted.vendors.filter((v) => v.key_source).flatMap((v) => v.models.map((m) => ({id: `api:${v.id}:${m}`, name: `${v.name} · ${m} · hosted`})));
}

async function loadHosted() {
  Object.assign(hosted, await api('/api/hosted'));
  if (typeof renderCodeModels === 'function') renderCodeModels();
}

async function saveVendor(id, change) {
  Object.assign(hosted, await api(`/api/hosted/${id}`, {method: 'PUT', body: JSON.stringify(change)}));
  if (typeof renderCodeModels === 'function') renderCodeModels();
  renderHosted();
}

function vendorRow(v) {
  const source = v.key_source === 'environment' ? `key from ${esc(v.env)}` : 'key saved';
  const found = hosted.found[v.id] || [];
  return `<div class="well well--pad" data-vendor="${esc(v.id)}">
    <div class="row row--wrap"><strong class="grow">${esc(v.name)}</strong>
      <span class="readout">${v.key_source ? `${source} ${esc(v.key_hint || '')}` : 'no key'}</span>
      ${v.keys_url ? `<a class="readout" href="${esc(v.keys_url)}" target="_blank" rel="noopener noreferrer">get a key</a>` : ''}</div>
    ${v.id === 'custom' ? `<div class="row row--wrap"><input class="field grow" data-url placeholder="https://… ending before /chat/completions" value="${esc(v.url || '')}" spellcheck="false" aria-label="Custom address"><button class="btn btn--ghost btn--xs" data-save-url>save address</button></div>` : ''}
    <form class="row row--wrap" data-key-form autocomplete="off">
      <input class="field grow" type="password" data-key placeholder="paste the ${esc(v.name)} API key" spellcheck="false" autocomplete="off" aria-label="${esc(v.name)} API key">
      <button class="btn btn--ghost btn--xs">save key</button>
      ${v.key_source === 'saved' ? '<button type="button" class="btn btn--ghost btn--xs" data-remove-key>remove key</button>' : ''}
      ${v.key_source ? '<button type="button" class="btn btn--ghost btn--xs" data-find>check key · list models</button>' : ''}
    </form>
    ${v.key_source ? `<form class="row row--wrap" data-add-form>
      <input class="field grow" data-add list="models-${esc(v.id)}" placeholder="model name, as the vendor writes it" spellcheck="false" aria-label="Model to add">
      <datalist id="models-${esc(v.id)}">${found.map((m) => `<option value="${esc(m)}"></option>`).join('')}</datalist>
      <button class="btn btn--ghost btn--xs">add model</button>
      ${found.length ? `<span class="readout">${found.length} available</span>` : ''}
    </form>
    <div class="row row--wrap">${v.models.map((m) => `<button class="btn btn--ghost btn--xs" data-drop="${esc(m)}" title="Remove">${esc(m)} ×</button>`).join('') || '<span class="readout">no models added yet</span>'}</div>` : ''}
  </div>`;
}

function renderHosted() {
  const url = `${location.origin}/v1`;
  $('hosted-limit').value = hosted.token_limit;
  $('hosted-vendors').innerHTML = hosted.vendors.map(vendorRow).join('');
  $('hosted-proxy').innerHTML = `
    <div class="row row--wrap"><span class="label">base url</span><code class="grow">${esc(url)}</code></div>
    <div class="row row--wrap"><span class="label">api key</span><code class="grow">${esc(hosted.proxy_token || 'only shown on this machine')}</code></div>
    <p class="readout">Point any OpenAI-style tool here. The model name picks the vendor, such as <code>api:novita:some/model</code>, or a local one such as <code>qwen3.6:35b-a3b</code>. It answers this machine only. The run limit below does not cover it, so check the vendor's own usage page.</p>`;
  for (const row of document.querySelectorAll('[data-vendor]')) {
    const id = row.dataset.vendor;
    const vendor = hosted.vendors.find((v) => v.id === id);
    const report = (work) => work.catch((error) => toast(error.message || String(error), true));
    row.querySelector('[data-key-form]').onsubmit = (event) => {
      event.preventDefault();
      const key = row.querySelector('[data-key]').value.trim();
      if (key) report(saveVendor(id, {key}).then(() => toast('Key saved')));
    };
    const remove = row.querySelector('[data-remove-key]');
    if (remove) remove.onclick = () => report(saveVendor(id, {key: ''}));
    const saveUrl = row.querySelector('[data-save-url]');
    if (saveUrl) saveUrl.onclick = () => report(saveVendor(id, {url: row.querySelector('[data-url]').value}));
    const find = row.querySelector('[data-find]');
    if (find) find.onclick = () => report((async () => {
      find.disabled = true;
      try {
        hosted.found[id] = (await api(`/api/hosted/${id}/models`)).models;
        toast(`The key works: ${hosted.found[id].length} models`);
        renderHosted();
      } finally { find.disabled = false; }
    })());
    const add = row.querySelector('[data-add-form]');
    if (add) add.onsubmit = (event) => {
      event.preventDefault();
      const name = row.querySelector('[data-add]').value.trim();
      if (name) report(saveVendor(id, {models: [...vendor.models, name]}));
    };
    for (const drop of row.querySelectorAll('[data-drop]')) {
      drop.onclick = () => report(saveVendor(id, {models: vendor.models.filter((m) => m !== drop.dataset.drop)}));
    }
  }
}

function openHostedView() {
  loadHosted().then(renderHosted).catch(fail);
}

document.addEventListener('DOMContentLoaded', () => {
  $('hosted-limit-form').onsubmit = (event) => {
    event.preventDefault();
    api('/api/hosted', {method: 'PUT', body: JSON.stringify({token_limit: Number($('hosted-limit').value)})})
      .then((reply) => { Object.assign(hosted, reply); toast('Limit saved'); })
      .catch((error) => toast(error.message || String(error), true));
  };
  // The Code view's list needs these even if this tab is never opened.
  // The page can open directly on this tab, before this file has loaded.
  loadHosted().then(() => { if (state.view === 'api') renderHosted(); }).catch(fail);
});
