# Ferro64

**Your own AI studio, on your own laptop.** Chat, a coding agent you can watch, images and video, with open models running on Apple Silicon. One Rust service, nothing leaves your machine unless you turn on web search, and every change the coding agent makes is shown to you first.

[ferro64.com](https://ferro64.com) · built and tested on a MacBook Pro (M1 Max, 64 GB)

| | |
| --- | --- |
| **Chat** | streaming answers, thinking and effort controls, photos and PDFs read on-device, web search with cited sources, recall of related earlier questions |
| **Code** | an agent for a project folder: reads, searches, edits and runs your build, with every edit as a diff you approve, commands that ask first, undo per run, a real terminal, semantic code search and read-only access to library source |
| **Agents** | draft → review → final answer, each stage on its own model |
| **Images** | Z-Image-Turbo and FLUX.2 Klein through stable-diffusion.cpp; edit by describing the change |
| **Video** | LTX-2.3 clips from text or a picture, with sound, and continue-from-last-frame for longer scenes |
| **Documents** | text from photos, scans and PDFs with Apple Vision |

Under the hood: an **Actix Web** server manages `llama-server` (llama.cpp), `sd-cli` (stable-diffusion.cpp) and the LTX video build for you, renders one **Tera** page, and keeps history in SQLite. Fonts and assets are served locally: no Node build, no CDN, no network calls at runtime. A **Tauri** app wraps it as a native desktop app (macOS; it also builds for Windows and Linux). The interface uses the Lumen theme from Hallmark.

**Runs on:** Apple Silicon Macs (macOS 15+) from 16 GB, best at 32–64 GB; Linux and Windows PCs with an NVIDIA, AMD or Intel GPU, or CPU only (new, untested on real hardware). The setup script installs Rust and everything else; the app suggests models that fit your machine. Video needs a Mac with 32 GB+ and Xcode.

## Install

One script per system, after `git clone`. It installs what is missing (Rust, the inference engines, the build tools), builds the app, and tells you which models suit your machine. **It downloads no models.**

```sh
git clone https://github.com/vrannang1/ferro64.git && cd ferro64

./scripts/setup.sh                 # macOS or Linux
./scripts/setup.sh --desktop       # also the desktop app
./scripts/setup.sh --video         # also video (Mac with 32 GB+, needs Xcode)
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1            # add -Desktop for the desktop app
```

Then start it and open [localhost:3042](http://localhost:3042):

```sh
cargo run --release
```

That is the only service. The app starts and stops the native engines it needs, with no Python and no Docker:

| Engine | Used for | macOS | Linux | Windows |
| --- | --- | --- | --- | --- |
| `llama-server` (llama.cpp) | chat, code, agents, vision | Metal | CUDA, Vulkan or CPU | CUDA, Vulkan or CPU |
| `sd-cli` (stable-diffusion.cpp) | images | Metal | Vulkan or CPU | CUDA, Vulkan or CPU |
| `ltx-video` (Swift MLX) | video | Metal, 32 GB+ | – | – |
| Apple Vision | documents (OCR) | ✓ | – | – |

The scripts pin the tested engine versions (override with `LLAMA_TAG` and `SD_TAG`) and put them in `data/bin/`. **Status by system:** macOS is where Ferro64 is built and tested every day. The Linux and Windows builds compile cleanly (checked by cross-compiling the server and the Windows desktop app), but they have not been run on real Linux or Windows machines yet; reports and fixes are welcome.

### Models that fit your machine

Go to **Settings → Models**. Ferro64 reads your memory and GPU and marks every model: **fits**, **slow here** (it runs partly in system memory, or needs the Mac GPU limit raised), or **too big** (greyed out; you can still download it). A panel at the top names the best fit per job:

**Ferro64 is meant to be useful on an ordinary 16 GB laptop, and quality comes before speed,** so **Qwen3.5 9B** is the default for chat, web search, agents and small coding tasks on every machine. Our tests ran on an M1 Max, so expect a laptop without a strong GPU to be much slower. With more memory you can download bigger models from the same list: Qwen3.5 or Qwen3.6 35B-A3B on 32–64 GB gives more knowledge at good speed, and if one is installed, Ferro64 prefers it. Machines with under 16 GB can try it, but it may not fit.

`ferro64 --hardware` prints the same report in a terminal (the setup scripts show it). To see what another machine would get, set `FERRO64_HARDWARE`, for example `FERRO64_HARDWARE=mac:16` or `FERRO64_HARDWARE=nvidia:4:16` (GPU GB, then system GB; a name can follow, as in `mac:16:Apple M2`). On GPUs with less than 12 GB the context window defaults to 8k so small models' caches fit; set `LLAMACPP_CTX_SIZE` to change it. The Qwen3.5 9B and 4B picks are from the same family as the tested 35B, but have not been benchmarked here yet.

Models download from Hugging Face into `data/models/` and are checked against the publisher's SHA-256 before they appear. Vision models also fetch their projector (`mmproj`). You can also drop any `.gguf` into `data/models/`, point `LLAMACPP_MODEL_DIRS` at other folders, or reuse an Ollama cache. To use a running Ollama instead: `LLM_BACKEND=ollama cargo run --release`.

### Free web search key

Web search works without an account, on Keenable's shared free tier (1,000 searches an hour per network). For **100,000 searches a month** of your own: sign up for the free plan at [keenable.ai](https://keenable.ai), create an API key in your Keenable dashboard, and paste it into **Settings → web search**. It is stored only on your computer and sent only to Keenable with your searches. (Or set `KEENABLE_API_KEY` in `.env`, which takes precedence.)

## Chat controls

- **Model picker** lists every chat model on disk with its size, quantization, context length and whether it reasons. Picking one loads it so the first reply starts quickly.
- **Thinking** asks the model to reason before answering. The reasoning streams into a collapsible block above the answer and is saved with it. The control follows the model's own chat template, which is read from the GGUF file:
  - **toggle** (Qwen3, Nemotron): on/off is passed as `enable_thinking`.
  - **effort** (gpt-oss style templates): the effort level is passed as `reasoning_effort`.
  - **always** (DeepSeek R1 distills): it always reasons; the switch shows on and is locked.
  - **none**: the switch is disabled.
- **Effort** sets the output budget and tells the model how careful and long to be, so it works with every model. Without reasoning the budget is 1k / 4k / 8k tokens; with reasoning (thinking on, or a model that always reasons) it is 4k / 12k / 24k, so the model has room to think and still answer.
- **Temperature** (precise / balanced / creative = 0.2 / 0.6 / 1.0) sets how varied the wording is. It does not change what the model knows: a wrong fact stays wrong at any setting.
- **Web** searches before answering. The app reads the top pages and the model answers from them, citing sources as numbered links, with the source list under the answer. A follow-up question is first rewritten into a standalone search. Searches go to Keenable (with your free key from Settings or `KEENABLE_API_KEY`, or keyless at 1,000 an hour), Brave Search or a SearXNG server, falling back to DuckDuckGo; Settings shows which one is in use. This is the only feature that sends anything off your machine: the search query and page requests. Only public http(s) addresses are fetched.
- **Stop** (or Esc) ends generation at once. The partial answer is saved and marked as stopped.
- Replies show tokens per second, token count, time, and model load time when a load happened.

Images and PDFs can be attached, pasted or dropped. PDFs and text-heavy images go through on-device OCR. Photos go to a vision model, and if the selected model can't see, the app routes them to an installed one.

Spreadsheets (CSV, TSV, XLSX, XLS, ODS) can be attached the same way, up to 20 MB. Ask for an analysis, or ask follow-up questions in the same chat. The figures don't come from the model's head: the sheet goes into a throwaway in-memory database, the model writes read-only queries, Ferro64 runs them, and the model explains the results. The queries are listed under each answer so you can check them. Nothing leaves your machine.

## Desktop app

```sh
scripts/build-desktop.sh    # builds dist/Ferro64.app (Tauri 2)
open dist/Ferro64.app
```

The app is the same page in a native window, with a Dock icon, a native folder picker for Code projects, and links opening in your browser. It starts the Ferro64 server and stops it on quit (⌘Q), which also unloads the model. If a Ferro64 server is already running on the port (3042, or `PORT`), the app uses it instead of starting a second one, so two models never compete for memory. A server the app started also shuts down if the app is force-quit. The server inside the app still uses this folder's `data/`; rebuild the app after updating the code. `cargo run --release` and a browser keep working as before.

## Code

The code view is a coding agent for a project folder: a file tree, a plain editor (⌘S saves; Tab indents the way the file does), and the agent beside them. It uses the best coding model you have installed, Qwen3.5 9B by default; it reads, searches, edits and runs commands through tools.

**What to expect.** The local agent is here to help you fix errors, write functions and do small tasks on an app you already have. On small tasks with hidden tests, the 9B default did well in Rust, Go, Elixir and Python, did well on Haskell in one of two runs, and failed in Swift, Erlang and C#, where it needs more checking from you. For big tasks, such as designing a system or building a whole app, use a bigger model, API keys (set `FERRO64_API_MODELS=on`, and expect to pay per token) or other tools. Read and run what it gives you: passing tests only show the code does what the tests say, and tests it wrote can lock in a bug.

**On the first run in an existing project,** Ferro64 indexes the code and writes a short summary: what the project is, how to run and test it, and where the main parts live. It is kept with the project, so a new session starts with that overview, plus the code that looks related to its task, instead of exploring from nothing. The model is told to check both against the code. The summary is rewritten when a build file or much of the project changes. **summary** shows it, and an edit of yours is kept until you ask for a new one.

**Overnight** lets a run work by itself. Turn it on before a run, and the agent approves its own edits and commands inside the project, keeps going with the page closed, and continues until its plan is finished. It stops after 12 hours or after six tries in a row with no plan item finished. Rules refuse installs, pushes, other folders and the internet, apart from package registry lookups. On macOS its commands also run under the system sandbox, so nothing can write outside the project and its package caches, or read keys. On Linux and Windows only the rules apply, and rules can be got around, so use it on projects you could afford to lose. Reopen the Code view to follow a run or stop it. Stopping takes effect after the current step.

The agent has to follow a plan. It writes one before changing anything, the plan stays on screen and goes with every request, and an item only counts as done with a command that passed since the last file change. A run with open items ends as "unfinished", whatever the model says. It keeps going while items get finished and stops after 30 steps without one. Commands that only look around, like `ls`, never count as proof.

A new run also knows the last three runs on the project (the task, how it ended, files changed, what was left open) and gets a short map of the code, with the files closest to its task first. When a long run nears the context limit, it writes a handoff: what was asked, in your words, what is done, what failed, and what comes next, and carries on from that with its most recent steps intact.

Scope is enforced where it can be. A question, or a request that says "do not change anything", can read and run tests but cannot edit, so it gets an answer rather than a piece of work. Before an existing project's run finishes, the agent checks its changes against the request. One project runs one job at a time.

With the embedding model installed, the agent also keeps notes across runs: a decision and why, a gotcha, a fix that took real effort. Matching notes are shown when a run starts, marked as something to check. A note can only be saved after a command has passed, and it is flagged when its source file changes. Read, delete or turn them off under **notes**; turn them off for a clean test of a model.

| Tool | What it does | Approval |
| --- | --- | --- |
| list folder, read file, search | look at the project | none; only inside the project |
| edit file, write file | change code | you see the diff and allow or deny; "allow all edits" covers the rest of the run |
| run command | build, test, git… in the project root (zsh, with your PATH) | **allow**, **always allow** that command (e.g. `cargo test`) in this project, or **deny** with an optional note for the model |

Safety is enforced by the app, not left to the model:

- Paths resolve inside the project after `..` and symlinks; anything else is refused. Build output and dependency folders (`target`, `node_modules`, folders the root `.gitignore` names…) are hidden.
- Commands run in the project root with a 2-minute default limit (up to 15), output capped, and the whole process group stopped on timeout. "Always allow" only matches a plain command: anything with `;`, `&&`, `|`, `>` or `$(` asks again. Commands that delete (`rm`), change the system (`sudo`, `chmod`), reach the network (`curl`, `ssh`), install packages, publish, or rewrite git history (`push`, `reset`, `clean`, `checkout`…) ask every time.
- Every file the agent writes is recorded with its previous content. **Undo this run** restores them, skipping any file you have edited since.
- Runs, tool calls, approvals and permissions are kept in SQLite; **history** reopens a run, and a new task continues it.
- Git buttons (status, diff, commit, push) are yours; the model never pushes on its own.

**Terminal** (the button in the file bar, or ⌃\`) opens your login shell in the project folder, in a real terminal ([xterm.js](https://xtermjs.org), saved in `static/vendor`), so interactive programs, colours and REPLs work. It is yours, not the agent's: nothing typed there goes through approvals. It keeps running while hidden; **restart** gives a fresh shell, and switching projects starts one in the new folder. Because a shell is full access and the server has no login, the terminal only connects from Ferro64's own page: a loopback host with a matching origin, so other websites cannot open one through `127.0.0.1`.

Projects can't be the home folder, a system folder or `~/Library`. The agent sees your Settings system prompt too.

With **Qwen3 Embedding 0.6B** downloaded (Settings → Models, 639 MB), the agent also gets **semantic search**: "where are discounts applied" finds the code by what it does, not by name. The project is cut into overlapping 40-line ranges and embedded on first use; later searches only re-embed files whose size or modified time changed. The index lives in SQLite and goes with the project when you remove it.

## Recall and embeddings

The same embedding model runs in its own small `llama-server` (about 1 GB) beside the chat model, starting on first use and stopping after 10 idle minutes. Besides code search, it powers **recall** (Settings, on by default): when you ask something, related questions from your *other* conversations are added to the model's context as background. Only close matches are used (cosine similarity above 0.55); unrelated questions bring nothing. Every new message is embedded as it is saved, and past messages are indexed once in the background after the model is downloaded. The embedding server's log is `data/embed-server.log`.

## One model at a time, kept warm

All inference shares one queue. With llama.cpp, the app runs a single `llama-server` for the selected model and reuses it between messages. Switching models stops it and starts the next. It is also stopped after `LLAMACPP_KEEP_ALIVE` idle seconds (default 600; `0` restores unload-after-every-reply), before image generation, and on **free memory**. The server's log is written to `data/llama-server.log`; if a model fails to load, the error quotes it.

With Ollama, the same policy uses `keep_alive` (`OLLAMA_KEEP_ALIVE`, default `10m`), and other residents are unloaded first.

The agent workflow (draft → review → synthesis) runs its stages in sequence, takes the same effort and thinking options, and shows each stage as it completes.

## Settings page

- **System prompt**: standing instructions (up to 8,000 characters) added to every chat turn and every agent stage, for example your location, units, or preferred answer length.
- **Models**: what is on disk and in memory, how much space `data/models` uses, and the download catalog.

Settings are stored in SQLite.

## Environment

See [.env.example](.env.example). The app reads `.env` at startup; variables exported in your shell win.

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_BACKEND` | `llamacpp` | `llamacpp` or `ollama` |
| `LLAMACPP_BIN` | `data/bin/llamacpp`, then PATH | `llama-server` binary; Homebrew locations are also searched |
| `SDCPP_BIN` | `data/bin/sdcpp/sd-cli`, then PATH | stable-diffusion.cpp binary for images |
| `IMAGE_BACKEND` | auto | `sdcpp`, `mlx` or `ollama`; auto picks the first one installed |
| `LLAMACPP_KEEP_ALIVE` | `600` | Idle seconds before the model unloads |
| `LLAMACPP_CTX_SIZE` | `65536` with 40 GB of GPU memory or more, `8192` on machines under 16 GB, otherwise `32768` | Context window; older history is trimmed to fit |
| `LLAMACPP_GPU_LAYERS` | `99` | Layers offloaded to Metal |
| `LLAMACPP_PORT` | free port | Fixed loopback port for `llama-server` |
| `LLAMACPP_MODEL_DIRS` | — | Extra `:`-separated folders to scan |
| `LLAMACPP_MODELS_FILE` | `data/llamacpp-models.json` | Explicit registry ([example](data/llamacpp-models.example.json)) |
| `OLLAMA_KEEP_ALIVE` / `OLLAMA_NUM_CTX` | `10m` / `32768` | Ollama equivalents |
| `KEENABLE_API_KEY` / `BRAVE_API_KEY` / `SEARXNG_URL` | — | Web search engine, first match wins; a Keenable key pasted in Settings comes next, then keyless Keenable |
| `AUTO_DOWNLOAD_MODEL` | `0` | `1` downloads Qwen3.5 9B on first start; otherwise nothing is downloaded until you choose |
| `FERRO64_HARDWARE` | detected | Pretend to be another machine, e.g. `mac:16` or `nvidia:4:16` |
| `HOST` / `PORT` / `FERRO64_DIR` | `127.0.0.1` / `3042` / repo | Bind address and data root |

## Documents and images

**OCR** uses the native macOS Vision/PDFKit helper (`tools/ocr.swift`, rebuilt with `scripts/build-ocr.sh`). It accepts PNG, JPEG, WebP and PDF up to 18 MB.

**Image generation** runs stable-diffusion.cpp with one of two models, picked in the images view and downloaded from Models:

| Choice | Model | Size | Time on an M1 Max |
| --- | --- | --- | --- |
| quality | FLUX.2 Klein 4B (Q8) | 4.6 GB, plus Qwen3 4B to read prompts | about 65 s at 1024 px |
| photo | Z-Image-Turbo (Q8) | 6.9 GB, plus Qwen3 4B | about 160 s at 1024 px |
| fast | SDXL-Turbo | 6.9 GB | about 10 s at 512 px |

FLUX.2 Klein and Z-Image-Turbo read prompts with Qwen3 4B. It is a part of them, downloaded with them and not offered for chat, so keep it. Only models you have downloaded appear in the images view.

**Edit** (on any gallery image, or **use a photo** for your own) keeps the picture and changes what you describe, for example "remove the ceiling fan". Edits use FLUX.2 Klein and keep the original's shape at up to about one megapixel; each result becomes the starting point for the next edit. SDXL-Turbo's license is non-commercial. Without `sd-cli`, the app falls back to mflux (`IMAGE_BACKEND=mlx`, macOS only) and then Ollama, whose recent versions no longer generate images. Failed generation returns an error and never substitutes a placeholder.

OCR uses Apple's Vision framework, so the documents view works on macOS only for now.

**Video** uses LTX-2.3 through [ltx-video-swift-mlx](https://github.com/VincentGourbin/ltx-video-swift-mlx), a Swift build that runs on the Mac's GPU with MLX. Build it once with `scripts/setup-video.sh` (Xcode with Swift 6.3 or newer; the script pins v0.4.2). The first video downloads the model (about 46 GB) into `data/models/video`. Describe a shot, pick a shape (16:9, 9:16, square) and a length (2–10 s), optionally with sound, or start from a picture: **animate** on any gallery image, or **start from a photo**. A video is a job: it holds the GPU queue with text models unloaded, the page shows its progress, and you can leave and come back. Clips are saved in `static/generated/videos/`. The transformer runs at 8-bit (`qint8`) to keep memory down. macOS only.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /` | The app, rendered by Tera from `templates/index.html` |
| `GET /api/status` | Runtime, resident model, keep-alive, context, OCR, SQLite counts |
| `GET, POST /api/code/projects`, `DELETE /api/code/projects/{id}` | Code projects (POST `{"path": "~/Sandbox/app"}`) |
| `GET /api/code/projects/{id}/tree?path=`, `GET, PUT …/file` | Browse and edit a project's files |
| `POST /api/code/run` | Run the coding agent as server-sent events: `run`, `status`, `reasoning`, `content`, `tool`, `approval`, `change`, `tool_result`, `done` or `error` |
| `POST /api/code/approve` | Answer an approval (`run_id`, `call_id`, `decision`: `once`, `always` or `deny`, optional `note`) |
| `GET /api/code/projects/{id}/runs`, `GET /api/code/runs/{id}`, `POST …/undo` | Run history and undo |
| `GET, DELETE /api/code/projects/{id}/permissions` | Commands allowed in a project |
| `GET /api/code/projects/{id}/terminal` | WebSocket shell in the project folder (same-origin only); binary frames carry keystrokes and output, text `{"resize": [cols, rows]}` |
| `POST /api/code/projects/{id}/git` | `status`, `diff`, `log`, `commit` (with `message`) or `push` |
| `GET, PUT /api/settings` | System prompt, recall, model storage use and web search engine; PUT takes `system_prompt` and/or `recall` (bool) |
| `GET /api/models` | Models on disk with capabilities, thinking mode, quantization, context |
| `POST /api/models/select` | Load a model now (`{"model":"qwen3.5:9b"}`) |
| `POST /api/models/unload` | Free model memory |
| `GET /api/model-store` | Download catalog and progress |
| `POST /api/models/download` | Download a catalog model |
| `POST /api/models/remove` | Delete a catalog model's files (`{"model":"gemma2:2b"}`); unloads it first |
| `GET, POST /api/chat/conversations` | List / create conversations |
| `GET, PATCH, DELETE /api/chat/conversations/{id}` | Messages / rename (`{"title":…}`) / delete |
| `POST /api/chat/stream` | Chat as server-sent events: `meta`, `status`, `sources` (web), `reasoning`, `content`, `done` or `error` |
| `POST /api/chat/send` | The same turn, as one JSON response |
| `POST /api/ocr` | Extract text from an image or PDF |
| `POST /api/agent/run`, `GET /api/agent/tasks` | Run / list workflows |
| `POST /api/generate` | Generate an image (`prompt`, `aspect_ratio`, `model`: `flux2-klein`, `z-image-turbo` or `sdxl-turbo`), or edit one with `edit_of` (gallery ID) or `image_data` (a photo) |
| `POST /api/enhance` | Refine an image prompt |
| `POST /api/video` | Start a video (`prompt`, `aspect`, `seconds`, `audio`, optional `image_of` or `image_data`); returns `job_id` |
| `GET /api/video/jobs/{id}`, `GET /api/video/active` | Video progress: status, phase, percent, last log lines |
| `GET /api/videos`, `DELETE /api/videos/{id}` | Video gallery |
| `GET /api/history`, `DELETE /api/history/{id}` | Image history |

Chat requests take `model`, `message`, optional `conversation_id`, `image_data` (a data URL), `thinking` (bool), `effort` (`low`, `medium`, `high`), `temperature` (`precise`, `balanced`, `creative`) and `web` (bool).

This is a local desktop service with no authentication. Keep it on loopback.

## Verification

```sh
cargo test
node --check static/app.js && node --check static/code.js
```

The tests cover: GGUF metadata and template classification; streaming and inline `<think>` splitting; warm reuse, swapping and idle reaping of a fake `llama-server`; strict unload mode; forwarding of thinking and effort; Ollama streaming and the shared queue; download hash checks and catalog coverage of `data/list.txt`; HTTP validation, rename and page rendering; and for the coding agent, paths that try to leave the project (`..`, absolute paths, symlinks), diffs, command gating, timeouts that stop the whole process group, context trimming, and search that skips build folders. Real output quality and speed depend on your weights and hardware; see [docs/RUNTIMES.md](docs/RUNTIMES.md).

## License

Ferro64 is licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. See [NOTICE.md](NOTICE.md) for the theme, fonts and engines it builds on. Models are downloaded by you, under their own licenses.
