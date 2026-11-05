'use strict';
/* Video view: LTX-2.3 clips from text or a starting picture. A video is a
   job on the server; this page polls it, so leaving and coming back is fine.
   Uses app.js helpers: $, esc, icon, api, post, toast, fail, duration,
   when, confirmDialog, setRadio, fileToData, checkFile, localURL, state. */

const vid = {
  aspect: '16:9',
  seconds: 4,
  audio: false,
  enhance: false,
  start: null, // {id, url, name} for a gallery image or {data, url, name} for a photo
  job: null,
  timer: null,
  videos: [],
};

async function openVideoView() {
  try {
    const active = await api('/api/video/active');
    $('video-engine').hidden = active.engine_ready && active.fits;
    if (!active.fits) $('video-engine').textContent = `Video needs a Mac with 32 GB of memory or more; this one has ${active.ram_gb} GB. LTX-2.3 would not load here.`;
    $('video-generate').disabled = !active.engine_ready || !active.fits;
    if (active.job) followJob(active.job.id);
  } catch (error) { fail(error); }
  loadVideos().catch(fail);
}

function renderVideoStart() {
  const start = vid.start;
  $('video-start').hidden = !start;
  if (start) {
    $('video-start-img').src = start.url;
    $('video-start-name').textContent = start.name;
  }
}

/* Called from the image gallery's "animate" button. */
function animateImage(item) {
  vid.start = {id: item.id, url: localURL(item.image_url), name: item.prompt};
  setView('video');
  renderVideoStart();
  if (!$('video-prompt').value.trim()) $('video-prompt').value = item.prompt;
  $('video-prompt').focus();
}

async function useVideoPhoto(file) {
  try {
    if (!file || !checkFile(file)) return;
    if (file.type === 'application/pdf') throw new Error('Start a video from an image, not a PDF.');
    const data = await fileToData(file);
    vid.start = {data, url: data, name: file.name};
    renderVideoStart();
  } catch (error) { fail(error); }
}

function videoProgress(job) {
  const canvas = $('video-canvas');
  canvas.classList.add('blueprint');
  const pct = job.percent ?? null;
  const elapsed = job.started_at ? duration(Date.now() - job.started_at) : '';
  canvas.innerHTML = `<div class="video__progress">
    <span class="label">${job.status === 'queued' ? 'waiting' : 'making the video'}${elapsed ? ` · ${esc(elapsed)}` : ''}</span>
    <p class="readout video__phase">${esc(job.phase || '…')}</p>
    ${pct !== null ? `<div class="progress" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"><span style="width:${pct}%"></span></div>` : '<div class="progress progress--busy"><span></span></div>'}
    <p class="readout">${/downloading|resolving checkpoint|\d+(\.\d+)? GB \//i.test(job.phase || '')
      ? 'Downloading the model into data/models/video. This happens once.'
      : 'Loading the model takes about half a minute, then a 4-second clip takes 3 to 4 minutes; sound adds a little.'}</p>
  </div>`;
}

function showVideo(item) {
  const url = localURL(item.video_url);
  if (!url) return;
  const canvas = $('video-canvas');
  canvas.classList.remove('blueprint');
  const seconds = (item.frames - 1) / 24;
  canvas.innerHTML = `<div class="canvas__result"><video src="${esc(url)}" controls autoplay loop playsinline></video><div class="row row--wrap"><span class="readout grow">${item.width}×${item.height} · ${seconds.toFixed(0)} s${item.audio ? ' · sound' : ''} · made in ${esc(duration(item.duration_ms))}</span><button type="button" class="btn btn--ghost btn--xs" id="video-continue" title="Start the next shot from this video's last frame">continue from last frame</button><a class="btn btn--ghost btn--xs" href="${esc(url)}" download="ferro64-${esc(item.id)}.mp4">save</a></div>${item.enhanced_prompt ? `<details class="video__refined"><summary class="readout">refined prompt</summary><p>${esc(item.enhanced_prompt)}</p><button type="button" class="btn btn--ghost btn--xs" id="video-use-refined">use this prompt</button></details>` : ''}</div>`;
  $('video-continue').onclick = () => continueFrom(item);
  if (item.enhanced_prompt) $('video-use-refined').onclick = () => { $('video-prompt').value = item.enhanced_prompt; $('video-prompt').focus(); };
}

/* The next shot starts where this one ends: the last frame becomes the
   starting picture, taken in the browser, so the person and place carry on. */
async function continueFrom(item) {
  const button = $('video-continue');
  if (button) button.disabled = true;
  try {
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';
    video.src = localURL(item.video_url);
    await new Promise((resolve, reject) => { video.onloadedmetadata = resolve; video.onerror = () => reject(new Error('Could not read the video')); });
    await new Promise((resolve, reject) => { video.onseeked = resolve; video.onerror = () => reject(new Error('Could not read the last frame')); video.currentTime = Math.max(0, video.duration - 0.05); });
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const data = canvas.toDataURL('image/png');
    vid.start = {data, url: data, name: `last frame of: ${item.prompt.slice(0, 60)}`};
    renderVideoStart();
    $('video-prompt').value = '';
    $('video-prompt').placeholder = 'What happens next, continuing from this frame?';
    $('video-prompt').focus();
    $('video-form').scrollIntoView({behavior: 'smooth', block: 'center'});
    toast('The next shot starts from this video\'s last frame');
  } catch (error) { fail(error); }
  finally { if (button) button.disabled = false; }
}

function followJob(id) {
  clearInterval(vid.timer);
  vid.job = {id, started_at: Date.now()};
  setVideoBusy(true);
  const poll = async () => {
    let job;
    try { job = await api(`/api/video/jobs/${encodeURIComponent(id)}`); }
    catch (error) { clearInterval(vid.timer); setVideoBusy(false); fail(error); return; }
    job.started_at = vid.job.started_at;
    if (job.status === 'completed') {
      clearInterval(vid.timer); setVideoBusy(false);
      await loadVideos();
      const item = vid.videos.find((v) => v.id === id);
      if (item) showVideo(item);
      toast(`Video ready in ${duration(job.duration_ms)}`);
    } else if (job.status === 'failed') {
      clearInterval(vid.timer); setVideoBusy(false);
      $('video-canvas').innerHTML = `<div class="video__progress"><span class="label">the video failed</span><pre class="code__output">${esc(job.error || 'unknown error')}</pre></div>`;
    } else {
      videoProgress(job);
    }
  };
  poll();
  vid.timer = setInterval(poll, 1500);
}

function setVideoBusy(busy) {
  $('video-generate').disabled = busy;
  $('video-generate').textContent = busy ? 'making…' : 'make video';
}

async function makeVideo(event) {
  event.preventDefault();
  const prompt = $('video-prompt').value.trim();
  if (!prompt) return;
  const body = {prompt, aspect: vid.aspect, seconds: vid.seconds, audio: vid.audio, enhance: vid.enhance};
  if (vid.start?.id) body.image_of = vid.start.id;
  else if (vid.start?.data) body.image_data = vid.start.data;
  try {
    const started = await post('/api/video', body, 60000);
    followJob(started.job_id);
  } catch (error) { fail(error); }
}

async function loadVideos() {
  vid.videos = await api('/api/videos');
  $('video-count').textContent = vid.videos.length ? String(vid.videos.length) : '';
  const gallery = $('video-gallery');
  if (!vid.videos.length) { gallery.innerHTML = '<p class="empty">Finished videos are kept here.</p>'; return; }
  gallery.innerHTML = vid.videos.map((item, i) => {
    const url = localURL(item.video_url);
    return `<figure class="shot" style="margin:0"><button class="shot__open" data-play="${i}" aria-label="Play ${esc(item.prompt)}">${url ? `<video src="${esc(url)}#t=0.5" preload="metadata" muted playsinline></video>` : ''}</button><figcaption class="shot__bar"><span class="readout" title="${esc(item.prompt)}">${esc(item.prompt)}</span><button class="icon-btn" data-delete="${i}" aria-label="Delete video">${icon('trash')}</button></figcaption></figure>`;
  }).join('');
  for (const button of gallery.querySelectorAll('[data-play]')) button.onclick = () => {
    const item = vid.videos[Number(button.dataset.play)];
    showVideo(item);
    $('video-prompt').value = item.prompt;
    $('video-canvas').scrollIntoView({behavior: 'smooth', block: 'center'});
  };
  for (const button of gallery.querySelectorAll('[data-delete]')) button.onclick = async () => {
    const item = vid.videos[Number(button.dataset.delete)];
    if (!await confirmDialog('delete this video?')) return;
    try { await api(`/api/videos/${encodeURIComponent(item.id)}`, {method: 'DELETE'}); await loadVideos(); } catch (error) { fail(error); }
  };
}

function bindVideo() {
  $('video-form').onsubmit = makeVideo;
  $('video-start-clear').onclick = () => { vid.start = null; renderVideoStart(); };
  $('video-photo').onclick = () => $('video-photo-file').click();
  $('video-photo-file').onchange = () => { useVideoPhoto($('video-photo-file').files[0]); $('video-photo-file').value = ''; };
  for (const button of $('video-aspect').querySelectorAll('[data-aspect]')) button.onclick = () => { vid.aspect = button.dataset.aspect; setRadio($('video-aspect'), vid.aspect, 'aspect'); };
  for (const button of $('video-seconds').querySelectorAll('[data-seconds]')) button.onclick = () => { vid.seconds = Number(button.dataset.seconds); setRadio($('video-seconds'), button.dataset.seconds, 'seconds'); };
  $('video-audio').onclick = () => { vid.audio = !vid.audio; $('video-audio').setAttribute('aria-checked', String(vid.audio)); };
  $('video-enhance').onclick = () => { vid.enhance = !vid.enhance; $('video-enhance').setAttribute('aria-checked', String(vid.enhance)); };
  $('video-refresh').onclick = () => loadVideos().catch(fail);
}
bindVideo();
if (state.view === 'video') openVideoView();
