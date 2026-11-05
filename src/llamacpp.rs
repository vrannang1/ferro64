//! Managed llama.cpp binary inference. The shared LlmEngine GPU queue is held
//! throughout startup, generation and process termination. One llama-server
//! process stays warm for the selected model and is replaced on a model switch
//! or stopped after LLAMACPP_KEEP_ALIVE idle seconds.
use crate::gguf::{self, Thinking};
use crate::llm::{finalize, ChatMessagePayload, ChatOptions, ChatOutcome, ChatStats, Effort, Emitter, ModelSpec, ToolCall, ToolTurn};
#[cfg(test)]
use crate::llm::Temperature;
use reqwest::Client;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

#[derive(Clone, Deserialize)]
struct LocalModel {
    id: String,
    name: Option<String>,
    path: PathBuf,
    mmproj: Option<PathBuf>,
}

struct ManagedChild {
    model: String,
    alias: String,
    base: String,
    child: Child,
    last_used: Instant,
}

pub struct LlamaCpp {
    binary: PathBuf,
    registry: PathBuf,
    base_dir: PathBuf,
    extra_dirs: Vec<PathBuf>,
    keep_alive: Duration,
    context: u32,
    gpu_layers: u32,
    port: Option<u16>,
    process: Mutex<Option<ManagedChild>>,
}

/// Cancellation of an HTTP handler during model load must also release model
/// memory. A dropped future kills immediately and schedules reaping.
struct ProcessCleanup<'a> {
    runtime: &'a LlamaCpp,
    armed: bool,
}

impl Drop for ProcessCleanup<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Ok(mut process) = self.runtime.process.try_lock() {
            if let Some(mut managed) = process.take() {
                let _ = managed.child.start_kill();
                if let Ok(handle) = tokio::runtime::Handle::try_current() {
                    handle.spawn(async move {
                        let _ = managed.child.wait().await;
                    });
                }
            }
        }
    }
}

fn env_u32(key: &str, default: u32, min: u32, max: u32) -> u32 {
    match std::env::var(key).ok().and_then(|v| v.trim().parse::<u32>().ok()) {
        Some(value) if (min..=max).contains(&value) => value,
        Some(value) => {
            tracing::warn!("{key}={value} is outside {min}..={max}; using {default}");
            default
        }
        None => default,
    }
}

impl LlamaCpp {
    pub fn new() -> Self {
        let base = std::env::var("FERRO64_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
        Self {
            // LLAMACPP_BIN, the copy fetched by scripts/setup.sh, or PATH.
            binary: std::env::var("LLAMACPP_BIN")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    let local = base.join(if cfg!(windows) { "data/bin/llamacpp/llama-server.exe" } else { "data/bin/llamacpp/llama-server" });
                    if local.is_file() {
                        local
                    } else {
                        PathBuf::from(if cfg!(windows) { "llama-server.exe" } else { "llama-server" })
                    }
                }),
            registry: std::env::var("LLAMACPP_MODELS_FILE")
                .map(PathBuf::from)
                .unwrap_or_else(|_| base.join("data/llamacpp-models.json")),
            extra_dirs: std::env::var_os("LLAMACPP_MODEL_DIRS")
                .map(|dirs| std::env::split_paths(&dirs).collect())
                .unwrap_or_default(),
            base_dir: base,
            keep_alive: Duration::from_secs(env_u32("LLAMACPP_KEEP_ALIVE", 600, 0, 86_400) as u64),
            context: env_u32("LLAMACPP_CTX_SIZE", crate::hardware::default_context(crate::hardware::this_machine()), 512, 1_048_576),
            gpu_layers: env_u32("LLAMACPP_GPU_LAYERS", 99, 0, 1_000),
            port: std::env::var("LLAMACPP_PORT")
                .ok()
                .and_then(|v| v.parse::<u16>().ok())
                .filter(|p| *p > 0),
            process: Mutex::new(None),
        }
    }

    pub fn keep_alive(&self) -> Duration {
        self.keep_alive
    }

    pub fn binary_path(&self) -> PathBuf {
        find_binary(&self.binary).unwrap_or_else(|| self.binary.clone())
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    pub fn context_size(&self) -> u32 {
        self.context
    }

    /// The most recent llama-server process log (overwritten on each start).
    pub fn log_path(&self) -> PathBuf {
        self.base_dir.join("data/llama-server.log")
    }

    fn registry(&self) -> Result<Vec<LocalModel>, String> {
        let data = match std::fs::read_to_string(&self.registry) {
            Ok(data) => data,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => "[]".to_string(),
            Err(error) => {
                return Err(format!(
                    "Cannot read llama.cpp registry {}: {error}",
                    self.registry.display()
                ))
            }
        };
        let mut models: Vec<LocalModel> = serde_json::from_str(&data).map_err(|error| {
            format!(
                "Invalid llama.cpp registry {}: {error}",
                self.registry.display()
            )
        })?;
        let parent = self.registry.parent().unwrap_or(Path::new("."));
        let mut ids = std::collections::HashSet::new();
        for model in &mut models {
            if model.id.trim().is_empty() || !ids.insert(model.id.clone()) {
                return Err("llama.cpp registry IDs must be unique and nonempty".to_string());
            }
            if model.path.is_relative() {
                model.path = parent.join(&model.path);
            }
            if let Some(path) = &mut model.mmproj {
                if path.is_relative() {
                    *path = parent.join(&*path);
                }
            }
        }
        if models.is_empty() {
            let mut folders = vec![self.base_dir.join("data/models"), self.base_dir.join("data")];
            folders.extend(self.extra_dirs.iter().cloned());
            for folder in folders {
                for model in discover_folder(&folder) {
                    if !models.iter().any(|existing| existing.id == model.id) {
                        models.push(model);
                    }
                }
            }
            let cache = std::env::var_os("OLLAMA_MODELS")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".ollama/models"))
                });
            if let Some(cache) = cache {
                for model in discover_ollama(&cache) {
                    if !models.iter().any(|existing| existing.id == model.id) {
                        models.push(model);
                    }
                }
            }
        }
        Ok(models)
    }

    pub fn available(&self) -> Result<(), String> {
        let models = self.registry()?;
        if find_binary(&self.binary).is_none() {
            return Err(format!(
                "llama-server binary not found: {}. Install llama.cpp (brew install llama.cpp) or set LLAMACPP_BIN.",
                self.binary.display()
            ));
        }
        if models.is_empty() {
            return Err("No chat model yet. Download one in Settings → Models; the best fits for this machine are marked.".to_string());
        }
        Ok(())
    }

    pub async fn models(&self) -> Result<Vec<ModelSpec>, String> {
        let loaded = self.loaded().await?;
        Ok(self
            .registry()?
            .into_iter()
            .map(|model| {
                let valid = valid_gguf(&model.path)
                    && model.mmproj.as_ref().is_none_or(|path| valid_gguf(path));
                let info = valid.then(|| gguf::read_cached(&model.path)).flatten();
                let thinking = info.as_ref().map_or(Thinking::None, |info| info.thinking());
                let mut capabilities = vec!["chat".to_string()];
                if model.mmproj.is_some() {
                    capabilities.push("vision".to_string());
                }
                if thinking != Thinking::None {
                    capabilities.push("thinking".to_string());
                }
                let catalog = crate::model_store::catalog_entry(&model.id);
                ModelSpec {
                    id: model.id.clone(),
                    name: catalog
                        .map(|entry| entry.name.to_string())
                        .or(model.name)
                        .unwrap_or_else(|| model.id.clone()),
                    parameter_size: info
                        .as_ref()
                        .and_then(|info| info.size_label.clone())
                        .unwrap_or_else(|| "GGUF".to_string()),
                    role_description: if !valid {
                        "GGUF or projector file is missing or invalid; update the local registry"
                            .to_string()
                    } else if let Some(entry) = catalog {
                        entry.role.to_string()
                    } else {
                        format!("Local GGUF · {}", model.path.display())
                    },
                    provider: "llama.cpp".to_string(),
                    is_image_generator: false,
                    is_resident: loaded.iter().any(|id| id == &model.id),
                    capabilities,
                    installed: valid,
                    source: "llamacpp".to_string(),
                    size_bytes: std::fs::metadata(&model.path).ok().map(|meta| meta.len()),
                    thinking: thinking.as_str().to_string(),
                    context_length: info.as_ref().and_then(|info| info.context_length),
                    quantization: info
                        .as_ref()
                        .and_then(|info| info.quantization())
                        .map(str::to_string),
                    architecture: info.and_then(|info| info.architecture),
                }
            })
            .collect())
    }

    pub async fn loaded(&self) -> Result<Vec<String>, String> {
        let mut process = self.process.lock().await;
        if let Some(managed) = process.as_mut() {
            match managed
                .child
                .try_wait()
                .map_err(|error| format!("Cannot inspect llama-server process: {error}"))?
            {
                None => return Ok(vec![managed.model.clone()]),
                Some(_) => {
                    *process = None;
                }
            }
        }
        Ok(Vec::new())
    }

    pub async fn unload(&self) -> Result<Vec<String>, String> {
        let mut process = self.process.lock().await;
        let Some(managed) = process.as_mut() else {
            return Ok(Vec::new());
        };
        let model = managed.model.clone();
        if managed
            .child
            .try_wait()
            .map_err(|error| format!("Cannot inspect llama-server: {error}"))?
            .is_none()
        {
            managed
                .child
                .kill()
                .await
                .map_err(|error| format!("Cannot terminate llama-server: {error}"))?;
        }
        managed
            .child
            .wait()
            .await
            .map_err(|error| format!("Cannot reap llama-server: {error}"))?;
        *process = None;
        Ok(vec![model])
    }

    /// Keeps the server when it already holds `model`; otherwise stops it.
    pub async fn unload_unless(&self, model: &str) -> Result<Vec<String>, String> {
        let keep = self
            .process
            .lock()
            .await
            .as_mut()
            .is_some_and(|m| m.model == model && matches!(m.child.try_wait(), Ok(None)));
        if keep {
            Ok(Vec::new())
        } else {
            self.unload().await
        }
    }

    /// Called by the idle reaper while it holds the inference queue.
    pub async fn reap_idle(&self) -> Option<String> {
        if self.keep_alive.is_zero() {
            return None;
        }
        let idle = self
            .process
            .lock()
            .await
            .as_ref()
            .is_some_and(|m| m.last_used.elapsed() >= self.keep_alive);
        if idle {
            self.unload().await.ok()?.into_iter().next()
        } else {
            None
        }
    }

    fn find(&self, model_id: &str) -> Result<LocalModel, String> {
        let model = self
            .registry()?
            .into_iter()
            .find(|model| model.id == model_id)
            .ok_or_else(|| format!("No llama.cpp GGUF configured for '{model_id}'"))?;
        if !valid_gguf(&model.path) {
            return Err(format!(
                "Not a readable GGUF file: {}",
                model.path.display()
            ));
        }
        if let Some(projector) = &model.mmproj {
            if !valid_gguf(projector) {
                return Err(format!(
                    "Not a readable GGUF projector: {}",
                    projector.display()
                ));
            }
        }
        Ok(model)
    }

    pub async fn preload(&self, client: &Client, model_id: &str) -> Result<(), String> {
        self.available()?;
        let model = self.find(model_id)?;
        let mut emitter = Emitter::new(None);
        self.ensure_running(client, &model, &mut emitter).await?;
        Ok(())
    }

    /// Returns (base URL, alias, load time) for a ready server holding `model`.
    async fn ensure_running(
        &self,
        client: &Client,
        model: &LocalModel,
        emitter: &mut Emitter,
    ) -> Result<(String, String, Option<u64>), String> {
        {
            let mut process = self.process.lock().await;
            if let Some(managed) = process.as_mut() {
                if managed.model == model.id && matches!(managed.child.try_wait(), Ok(None)) {
                    managed.last_used = Instant::now();
                    return Ok((managed.base.clone(), managed.alias.clone(), None));
                }
            }
        }
        self.unload().await?;
        emitter.status("loading");
        // An automatically chosen port can be taken between probing and the
        // server binding it; a server that exits at once is retried on a new one.
        let attempts = if self.port.is_some() { 1 } else { 3 };
        let mut attempt = 1;
        loop {
            let started = Instant::now();
            match self.launch(client, model).await {
                Err(error)
                    if attempt < attempts
                        && error.contains("exited during model load")
                        && started.elapsed() < Duration::from_secs(3) =>
                {
                    attempt += 1;
                }
                result => return result,
            }
        }
    }

    async fn launch(
        &self,
        client: &Client,
        model: &LocalModel,
    ) -> Result<(String, String, Option<u64>), String> {
        let port = match self.port {
            Some(port) => {
                // Refuse an occupied port rather than talk to an unrelated server.
                let probe = std::net::TcpListener::bind(("127.0.0.1", port))
                    .map_err(|error| format!("llama-server port {port} is unavailable: {error}. Set LLAMACPP_PORT to a free port or unset it."))?;
                drop(probe);
                port
            }
            None => std::net::TcpListener::bind(("127.0.0.1", 0))
                .and_then(|listener| listener.local_addr())
                .map(|address| address.port())
                .map_err(|error| format!("Cannot find a free port for llama-server: {error}"))?,
        };
        // The per-process alias ensures readiness checks cannot accidentally
        // connect to an unrelated server.
        let alias = format!("ferro64-{}", uuid::Uuid::new_v4());
        let mut command = Command::new(find_binary(&self.binary).unwrap_or_else(|| self.binary.clone()));
        command
            .arg("--model")
            .arg(&model.path)
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--parallel",
                "1",
                "-c",
                &self.context.to_string(),
                "-ngl",
                &self.gpu_layers.to_string(),
                "--alias",
                &alias,
                "--jinja",
                "--reasoning-format",
                "deepseek",
                "--no-webui",
                "--offline",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(
                std::fs::File::create(self.log_path())
                    .map(Stdio::from)
                    .unwrap_or_else(|_| Stdio::null()),
            )
            .kill_on_drop(true);
        // llama.cpp's bfloat16 Metal kernels need macOS 26; on older systems
        // the shader library fails to compile and nothing can load.
        if std::env::var_os("GGML_METAL_BF16_DISABLE").is_none() && macos_major().is_some_and(|v| v < 26) {
            command.env("GGML_METAL_BF16_DISABLE", "1");
        }
        if let Some(projector) = &model.mmproj {
            command.arg("--mmproj").arg(projector);
        }
        let start = Instant::now();
        let child = command
            .spawn()
            .map_err(|error| format!("Could not start {}: {error}", self.binary.display()))?;
        let base = format!("http://127.0.0.1:{port}");
        *self.process.lock().await = Some(ManagedChild {
            model: model.id.clone(),
            alias: alias.clone(),
            base: base.clone(),
            child,
            last_used: Instant::now(),
        });
        let mut cleanup = ProcessCleanup {
            runtime: self,
            armed: true,
        };
        match self.wait_ready(client, &base, &alias).await {
            Ok(()) => {
                cleanup.armed = false;
                Ok((base, alias, Some(start.elapsed().as_millis() as u64)))
            }
            Err(error) => {
                cleanup.armed = false;
                let _ = self.unload().await;
                Err(error)
            }
        }
    }

    pub async fn chat(
        &self,
        client: &Client,
        model_id: &str,
        messages: &[ChatMessagePayload],
        options: ChatOptions,
        mut emitter: Emitter,
    ) -> Result<ChatOutcome, String> {
        self.available()?;
        let model = self.find(model_id)?;
        if messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|images| !images.is_empty()))
            && model.mmproj.is_none()
        {
            return Err(format!("Vision model '{model_id}' requires a compatible mmproj file in the llama.cpp registry"));
        }
        let mode = gguf::read_cached(&model.path).map_or(Thinking::None, |info| info.thinking());
        let reasons = options.thinking || matches!(mode, Thinking::Always | Thinking::Effort);
        let start = Instant::now();
        let result = async {
            let (base, alias, load_ms) = self.ensure_running(client, &model, &mut emitter).await?;
            let mut outcome = self
                .stream(client, &base, &alias, messages, options, reasons, &mut emitter)
                .await?;
            outcome.stats.load_ms = load_ms;
            outcome.duration_ms = start.elapsed().as_millis() as u64;
            Ok(outcome)
        }
        .await;
        // A failed request may leave the server in a bad state; restart next time.
        if result.is_err() || self.keep_alive.is_zero() {
            let cleanup = self.unload().await;
            return crate::llm::finish_with_cleanup(result, cleanup);
        }
        if let Some(managed) = self.process.lock().await.as_mut() {
            managed.last_used = Instant::now();
        }
        result
    }

    /// One turn of a tool-using conversation. `messages` are OpenAI-format
    /// values (they carry assistant tool_calls and tool results, which
    /// ChatMessagePayload cannot). Text streams through `emitter`; the
    /// requested calls come back whole.
    pub async fn chat_tools(
        &self,
        client: &Client,
        model_id: &str,
        messages: &[serde_json::Value],
        tools: &serde_json::Value,
        options: ChatOptions,
        emitter: &mut Emitter,
    ) -> Result<ToolTurn, String> {
        self.available()?;
        let model = self.find(model_id)?;
        let mode = gguf::read_cached(&model.path).map_or(Thinking::None, |info| info.thinking());
        if !gguf::read_cached(&model.path).and_then(|info| info.chat_template).is_some_and(|t| t.contains("tool")) {
            return Err(format!("'{model_id}' has no tool-calling chat template; choose Qwen3.6 or another tool-capable model"));
        }
        let reasons = options.thinking || matches!(mode, Thinking::Always | Thinking::Effort);
        let start = Instant::now();
        let result = async {
            let (base, alias, load_ms) = self.ensure_running(client, &model, emitter).await?;
            let mut body = chat_payload(&alias, &[], options, reasons);
            body["messages"] = serde_json::Value::from(messages.to_vec());
            body["tools"] = tools.clone();
            let mut turn = self.stream_tools(client, &base, &body, emitter).await?;
            turn.stats.load_ms = load_ms;
            turn.duration_ms = start.elapsed().as_millis() as u64;
            Ok(turn)
        }
        .await;
        if result.is_err() || self.keep_alive.is_zero() {
            let cleanup = self.unload().await;
            return crate::llm::finish_with_cleanup(result, cleanup);
        }
        if let Some(managed) = self.process.lock().await.as_mut() {
            managed.last_used = Instant::now();
        }
        result
    }

    async fn stream_tools(
        &self,
        client: &Client,
        base: &str,
        body: &serde_json::Value,
        emitter: &mut Emitter,
    ) -> Result<ToolTurn, String> {
        let mut response = client
            .post(format!("{base}/v1/chat/completions"))
            .json(body)
            .send()
            .await
            .map_err(|error| format!("llama-server inference failed: {error}"))?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            let detail = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| v.pointer("/error/message").and_then(|m| m.as_str()).map(str::to_string))
                .unwrap_or(text);
            return Err(format!("llama-server {status}: {detail}"));
        }
        emitter.status("generating");
        let mut stats = ChatStats::default();
        let mut calls: Vec<ToolCall> = Vec::new();
        let mut buffer = Vec::new();
        'read: while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| format!("llama-server stream interrupted: {error}"))?
        {
            buffer.extend_from_slice(&chunk);
            while let Some(newline) = buffer.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = buffer.drain(..=newline).collect();
                collect_tool_calls(&line, &mut calls);
                if apply_sse_line(&line, emitter, &mut stats)? || emitter.closed {
                    break 'read;
                }
            }
        }
        emitter.finish();
        Ok(ToolTurn {
            content: emitter.content.clone(),
            reasoning: emitter.reasoning.trim().to_string(),
            tool_calls: calls.into_iter().filter(|c| !c.name.is_empty()).collect(),
            duration_ms: 0,
            stats,
            cancelled: emitter.closed,
        })
    }

    async fn stream(
        &self,
        client: &Client,
        base: &str,
        alias: &str,
        messages: &[ChatMessagePayload],
        options: ChatOptions,
        reasons: bool,
        emitter: &mut Emitter,
    ) -> Result<ChatOutcome, String> {
        let mut response = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&chat_payload(alias, messages, options, reasons))
            .send()
            .await
            .map_err(|error| format!("llama-server inference failed: {error}"))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let detail = serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v.pointer("/error/message").and_then(|m| m.as_str()).map(str::to_string))
                .unwrap_or(body);
            return Err(format!("llama-server {status}: {detail}"));
        }
        emitter.status("generating");
        let mut stats = ChatStats::default();
        let mut buffer = Vec::new();
        'read: while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| format!("llama-server stream interrupted: {error}"))?
        {
            buffer.extend_from_slice(&chunk);
            while let Some(newline) = buffer.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = buffer.drain(..=newline).collect();
                if apply_sse_line(&line, emitter, &mut stats)? {
                    break 'read;
                }
                if emitter.closed {
                    // Dropping the response closes the connection, which makes
                    // llama-server abandon the task; the process stays warm.
                    break 'read;
                }
            }
        }
        emitter.finish();
        finalize(emitter, stats)
    }

    async fn wait_ready(&self, client: &Client, base: &str, alias: &str) -> Result<(), String> {
        let deadline = Instant::now() + Duration::from_secs(300);
        while Instant::now() < deadline {
            {
                let mut process = self.process.lock().await;
                let managed = process
                    .as_mut()
                    .ok_or_else(|| "llama-server process disappeared".to_string())?;
                if let Some(status) = managed
                    .child
                    .try_wait()
                    .map_err(|error| error.to_string())?
                {
                    let log = std::fs::read_to_string(self.log_path()).unwrap_or_default();
                    return Err(format!(
                        "llama-server exited during model load ({status}). {} Full log: {}",
                        load_failure_reason(&log),
                        self.log_path().display()
                    ));
                }
            }
            if let Ok(response) = client
                .get(format!("{base}/v1/models"))
                .timeout(Duration::from_secs(2))
                .send()
                .await
            {
                if response.status().is_success() {
                    if let Ok(data) = response.json::<serde_json::Value>().await {
                        if data["data"]
                            .as_array()
                            .is_some_and(|models| models.iter().any(|model| model["id"] == alias))
                        {
                            return Ok(());
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
        Err("llama-server model startup timed out after 300 seconds".to_string())
    }
}

/// Merges the tool-call fragments in one SSE line: the first fragment of a
/// call carries its id and name, later ones append to its arguments.
fn collect_tool_calls(line: &[u8], calls: &mut Vec<ToolCall>) {
    let text = String::from_utf8_lossy(line);
    let Some(data) = text.trim().strip_prefix("data:") else { return };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(data.trim()) else { return };
    let Some(parts) = value.pointer("/choices/0/delta/tool_calls").and_then(|v| v.as_array()) else { return };
    for part in parts {
        let index = part["index"].as_u64().unwrap_or(calls.len() as u64) as usize;
        if calls.len() <= index {
            calls.resize(index + 1, ToolCall::default());
        }
        let call = &mut calls[index];
        if let Some(id) = part["id"].as_str() {
            call.id = id.to_string();
        }
        if let Some(name) = part.pointer("/function/name").and_then(|v| v.as_str()) {
            call.name.push_str(name);
        }
        if let Some(arguments) = part.pointer("/function/arguments").and_then(|v| v.as_str()) {
            call.arguments.push_str(arguments);
        }
    }
}

/// \"reading 82%\" while a long prompt is read in. What is already cached does
/// not count, so the number is how far through the real work it is.
fn prompt_progress(value: &serde_json::Value) -> Option<String> {
    let progress = value.get("prompt_progress")?;
    let total = progress["total"].as_u64()?;
    let cached = progress["cache"].as_u64().unwrap_or(0);
    let processed = progress["processed"].as_u64()?;
    let (work, done) = (total.saturating_sub(cached), processed.saturating_sub(cached));
    (work > 0 && done < work).then(|| format!("reading {}%", done * 100 / work))
}

/// Applies one SSE line. Returns true at the end-of-stream marker.
fn apply_sse_line(line: &[u8], emitter: &mut Emitter, stats: &mut ChatStats) -> Result<bool, String> {
    let text = String::from_utf8_lossy(line);
    let Some(data) = text.trim().strip_prefix("data:") else {
        return Ok(false);
    };
    let data = data.trim();
    if data == "[DONE]" {
        return Ok(true);
    }
    let value: serde_json::Value =
        serde_json::from_str(data).map_err(|error| format!("Invalid llama-server stream: {error}"))?;
    if let Some(error) = value.get("error") {
        return Err(format!(
            "llama-server: {}",
            error.get("message").and_then(|m| m.as_str()).unwrap_or("request failed")
        ));
    }
    if let Some(status) = prompt_progress(&value) {
        emitter.status(status);
    }
    if let Some(choice) = value.pointer("/choices/0") {
        if let Some(reasoning) = choice.pointer("/delta/reasoning_content").and_then(|v| v.as_str()) {
            emitter.reasoning(reasoning);
        }
        if let Some(content) = choice.pointer("/delta/content").and_then(|v| v.as_str()) {
            emitter.content(content);
        }
        if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            stats.finish_reason = Some(reason.to_string());
        }
    }
    if let Some(timings) = value.get("timings") {
        stats.prompt_tokens = timings.get("prompt_n").and_then(|v| v.as_u64());
        stats.completion_tokens = timings.get("predicted_n").and_then(|v| v.as_u64());
        stats.tokens_per_second = timings.get("predicted_per_second").and_then(|v| v.as_f64());
    }
    if let Some(usage) = value.get("usage").filter(|u| u.is_object()) {
        stats.prompt_tokens = stats
            .prompt_tokens
            .or_else(|| usage.get("prompt_tokens").and_then(|v| v.as_u64()));
        stats.completion_tokens = stats
            .completion_tokens
            .or_else(|| usage.get("completion_tokens").and_then(|v| v.as_u64()));
    }
    Ok(false)
}

/// The cause of a failed load, read from the server log, with what to do.
fn load_failure_reason(log: &str) -> String {
    if let Some(at) = log.find("unknown model architecture: '") {
        let rest = &log[at + "unknown model architecture: '".len()..];
        let arch = rest.split('\'').next().unwrap_or("?");
        return format!("This llama.cpp is too old for the '{arch}' architecture; update it (brew upgrade llama.cpp).");
    }
    if log.contains("failed to initialize the Metal library") {
        return "llama.cpp's GPU code failed to compile on this macOS; set GGML_METAL_BF16_DISABLE=1 or update macOS.".to_string();
    }
    if log.contains("failed to allocate") || log.contains("out of memory") {
        return "Not enough memory for this model; free memory or lower LLAMACPP_CTX_SIZE.".to_string();
    }
    // Otherwise the first errors carry the cause; later ones only repeat it.
    let errors: Vec<_> = log
        .lines()
        // llama.cpp marks error lines with an "E" level token.
        .filter(|line| line.split_whitespace().take(2).any(|token| token == "E"))
        .take(2)
        .map(str::trim)
        .collect();
    if errors.is_empty() {
        "The model architecture or projector may be unsupported by this llama.cpp.".to_string()
    } else {
        format!("Log: {}", errors.join(" | "))
    }
}

/// Major macOS version, e.g. 15 for 15.7.3; None elsewhere.
fn macos_major() -> Option<u32> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let output = std::process::Command::new("sw_vers").arg("-productVersion").output().ok()?;
    String::from_utf8_lossy(&output.stdout).trim().split('.').next()?.parse().ok()
}

fn discovered_id(path: &Path) -> String {
    let file = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if let Some(id) = crate::model_store::catalog_id_for_file(file) {
        return id.to_string();
    }
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("local-model");
    let lower = stem.to_lowercase();
    if lower.contains("qwen2.5") && lower.contains("coder") && lower.contains("3b") {
        "qwen2.5-coder:3b".to_string()
    } else if lower.contains("qwen2.5") && lower.contains("3b") && !lower.contains("vl") {
        "qwen2.5:3b".to_string()
    } else {
        stem.to_string()
    }
}

fn is_auxiliary(lower: &str) -> bool {
    [
        "mmproj",
        "projector",
        "embed",
        "all-minilm",
        "sentence-transformer",
        "flux",
        "diffusion",
        "imatrix",
        "dflash",
    ]
    .iter()
    .any(|word| lower.contains(word))
}

fn discover_folder(folder: &Path) -> Vec<LocalModel> {
    let Ok(files) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    let mut paths: Vec<_> = files
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
                && valid_gguf(path)
        })
        .collect();
    paths.sort();
    let mut models = Vec::new();
    for path in &paths {
        let stem = path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        let lower = stem.to_lowercase();
        if is_auxiliary(&lower) {
            continue;
        }
        let projector_name = format!("{lower}.mmproj.gguf");
        let mmproj = paths
            .iter()
            .find(|candidate| {
                candidate
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.to_lowercase() == projector_name)
            })
            .cloned();
        let id = discovered_id(path);
        // Quantization alternatives are kept separately rather than silently
        // aliasing different weights under the same selected model ID.
        let id = if models.iter().any(|model: &LocalModel| model.id == id) {
            stem.to_string()
        } else {
            id
        };
        models.push(LocalModel {
            id,
            name: Some(stem.to_string()),
            path: path.clone(),
            mmproj,
        });
    }
    models
}

fn discover_ollama(cache: &Path) -> Vec<LocalModel> {
    fn walk(folder: &Path, depth: usize, output: &mut Vec<PathBuf>) {
        if depth == 0 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(folder) else {
            return;
        };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            // Never follow symlinks while walking manifests.
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() {
                walk(&path, depth - 1, output);
            } else if kind.is_file() {
                output.push(path);
            }
        }
    }
    let manifest_root = cache.join("manifests");
    let mut manifests = Vec::new();
    walk(&manifest_root, 8, &mut manifests);
    manifests.sort();
    let mut models = Vec::new();
    for path in manifests {
        let Some(parts) = path.strip_prefix(&manifest_root).ok().map(|path| {
            path.components()
                .map(|part| part.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
        }) else {
            continue;
        };
        if parts.len() < 4 {
            continue;
        }
        let tag = parts.last().unwrap();
        let name_parts = &parts[..parts.len() - 1];
        let name = if name_parts[0] == "registry.ollama.ai"
            && name_parts.get(1).is_some_and(|part| part == "library")
        {
            name_parts[2..].join("/")
        } else if name_parts[0] == "registry.ollama.ai" {
            name_parts[1..].join("/")
        } else {
            name_parts.join("/")
        };
        let id = format!("{name}:{tag}");
        let lower = id.to_lowercase();
        if ["embed", "all-minilm", "flux", "diffusion"]
            .iter()
            .any(|word| lower.contains(word))
        {
            continue;
        }
        let Some(data) = std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        else {
            continue;
        };
        let Some(layers) = data.get("layers").and_then(|layers| layers.as_array()) else {
            continue;
        };
        let layer_path = |media_type: &str| {
            layers
                .iter()
                .find(|layer| layer["mediaType"] == media_type)
                .and_then(|layer| layer["digest"].as_str())
                .filter(|digest| {
                    digest.starts_with("sha256:")
                        && digest[7..].len() == 64
                        && digest[7..].chars().all(|c| c.is_ascii_hexdigit())
                })
                .map(|digest| cache.join("blobs").join(digest.replace(':', "-")))
                .filter(|path| valid_gguf(path))
        };
        if let Some(model_path) = layer_path("application/vnd.ollama.image.model") {
            let mmproj = layer_path("application/vnd.ollama.image.projector");
            models.push(LocalModel {
                id: id.clone(),
                name: Some(format!("{id} (Ollama cache)")),
                path: model_path,
                mmproj,
            });
        }
    }
    models
}

/// `reasons` is true when the model will reason: thinking was asked for, or
/// its template always reasons (gpt-oss, DeepSeek R1).
fn chat_payload(
    alias: &str,
    messages: &[ChatMessagePayload],
    options: ChatOptions,
    reasons: bool,
) -> serde_json::Value {
    let messages: Vec<_> = messages
        .iter()
        .map(|message| {
            if let Some(images) = message.images.as_ref().filter(|images| !images.is_empty()) {
                let mut content =
                    vec![serde_json::json!({"type": "text", "text": message.content})];
                for image in images {
                    let uri = if image.starts_with("data:") {
                        image.clone()
                    } else if image.starts_with("/9j/") {
                        format!("data:image/jpeg;base64,{image}")
                    } else if image.starts_with("UklGR") {
                        format!("data:image/webp;base64,{image}")
                    } else {
                        format!("data:image/png;base64,{image}")
                    };
                    content
                        .push(serde_json::json!({"type": "image_url", "image_url": {"url": uri}}));
                }
                serde_json::json!({"role": message.role, "content": content})
            } else {
                serde_json::json!({"role": message.role, "content": message.content})
            }
        })
        .collect();
    // Templates ignore kwargs they do not use: Qwen3/Nemotron read
    // enable_thinking, gpt-oss style templates read reasoning_effort.
    let mut kwargs = serde_json::json!({"enable_thinking": options.thinking});
    // Medium is every template's default, and some templates accept only
    // "low" and "high", so medium leaves the template to pick.
    if options.effort != Effort::Medium {
        kwargs["reasoning_effort"] = serde_json::Value::from(options.effort.as_str());
    }
    serde_json::json!({
        "model": alias, "messages": messages, "stream": true,
        "max_tokens": options.max_tokens(reasons), "temperature": options.temperature.value(),
        "reasoning_format": "deepseek", "cache_prompt": true, "return_progress": true,
        "chat_template_kwargs": kwargs,
    })
}

fn valid_gguf(path: &Path) -> bool {
    use std::io::Read;
    let mut header = [0u8; 24];
    std::fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut header))
        .is_ok()
        && &header[..4] == b"GGUF"
        && matches!(u32::from_le_bytes(header[4..8].try_into().unwrap()), 2 | 3)
}

fn find_binary(binary: &Path) -> Option<PathBuf> {
    let candidates = if binary.components().count() > 1 || binary.is_absolute() {
        vec![binary.to_path_buf()]
    } else {
        std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
            .chain([PathBuf::from("/opt/homebrew/bin"), PathBuf::from("/usr/local/bin")])
            .map(|path| path.join(binary))
            .collect()
    };
    candidates.into_iter().find(|path| {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            path.is_file()
                && std::fs::metadata(path).is_ok_and(|meta| meta.permissions().mode() & 0o111 != 0)
        }
        #[cfg(not(unix))]
        {
            path.is_file()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Effort, StreamEvent};

    #[test]
    fn vision_payload_uses_openai_content_parts_and_forwards_reasoning_controls() {
        let payload = chat_payload(
            "managed-alias",
            &[ChatMessagePayload {
                role: "user".to_string(),
                content: "Describe this".to_string(),
                images: Some(vec![
                    "/9j/example".to_string(),
                    "data:image/png;base64,abc".to_string(),
                ]),
            }],
            ChatOptions {
                thinking: false,
                effort: Effort::Low,
                temperature: Temperature::Precise,
            },
            false,
        );
        assert_eq!(
            payload["messages"][0]["content"][0]["text"],
            "Describe this"
        );
        assert_eq!(
            payload["messages"][0]["content"][1]["image_url"]["url"],
            "data:image/jpeg;base64,/9j/example"
        );
        assert_eq!(
            payload["messages"][0]["content"][2]["image_url"]["url"],
            "data:image/png;base64,abc"
        );
        assert_eq!(payload["stream"], true);
        assert_eq!(payload["max_tokens"], Effort::Low.max_tokens());
        assert_eq!(payload["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(payload["chat_template_kwargs"]["reasoning_effort"], "low");
        assert_eq!(payload["temperature"], 0.2_f32);
        let medium = chat_payload("a", &[], ChatOptions { thinking: true, effort: Effort::Medium, ..Default::default() }, true);
        assert_eq!(medium["max_tokens"], 12_288);
        assert!(medium["chat_template_kwargs"].get("reasoning_effort").is_none());
    }

    #[test]
    fn sse_lines_route_reasoning_content_and_timings() {
        let mut emitter = Emitter::new(None);
        let mut stats = ChatStats::default();
        for line in [
            r#"data: {"choices":[{"delta":{"reasoning_content":"think "}}]}"#,
            r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#,
            "",
            r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"prompt_n":12,"predicted_n":34,"predicted_per_second":56.5}}"#,
        ] {
            assert!(!apply_sse_line(line.as_bytes(), &mut emitter, &mut stats).unwrap());
        }
        assert!(apply_sse_line(b"data: [DONE]", &mut emitter, &mut stats).unwrap());
        emitter.finish();
        assert_eq!(emitter.reasoning, "think ");
        assert_eq!(emitter.content, "Hello");
        assert_eq!(stats.finish_reason.as_deref(), Some("stop"));
        assert_eq!(stats.completion_tokens, Some(34));
        assert_eq!(stats.tokens_per_second, Some(56.5));
        let error = apply_sse_line(
            br#"data: {"error":{"message":"context exceeded"}}"#,
            &mut emitter,
            &mut stats,
        )
        .unwrap_err();
        assert!(error.contains("context exceeded"));
    }

    #[test]
    fn load_failures_name_the_cause_and_the_fix() {
        let unknown = "0.1 E llama_model_load: error loading model: unknown model architecture: 'lfm3'\n0.2 E cmn common_init_: failed to load model 'x.gguf'";
        assert!(load_failure_reason(unknown).contains("too old for the 'lfm3' architecture"));
        let metal = "E ggml_metal_init: error: failed to initialize the Metal library";
        assert!(load_failure_reason(metal).contains("GGML_METAL_BF16_DISABLE"));
        let other = "I loading\nE first: bad tensor shape\nE second: failed to load\nE third: exiting";
        assert_eq!(load_failure_reason(other), "Log: E first: bad tensor shape | E second: failed to load");
    }

    #[test]
    fn gguf_validation_rejects_other_files() {
        assert!(!valid_gguf(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("Cargo.toml")
                .as_path()
        ));
        assert!(!valid_gguf(Path::new(
            "/definitely-missing-ferro64-model.gguf"
        )));
    }

    const FAKE_SERVER: &str = r#"#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
root = Path(__file__).parent
with open(root / 'starts', 'a') as f: f.write(str(os.getpid()) + '\n')
args = sys.argv
alias = args[args.index('--alias') + 1]
port = int(args[args.index('--port') + 1])
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({'data': [{'id': alias}]}).encode())
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
        (root / 'last-request').write_text(json.dumps(body))
        if (root / 'fail').exists():
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({'error': {'message': 'fixture inference failure'}}).encode())
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        events = [{'choices': [{'delta': {'reasoning_content': 'mull'}}]},
                  {'choices': [{'delta': {'content': 'fixture '}}]},
                  {'choices': [{'delta': {'content': 'answer'}, 'finish_reason': 'stop'}], 'timings': {'predicted_n': 2, 'predicted_per_second': 40.0}}]
        for event in events:
            self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
        self.wfile.write(b'data: [DONE]\n\n')
HTTPServer(('127.0.0.1', port), Handler).serve_forever()
"#;

    #[cfg(unix)]
    fn fixture_runtime(keep_alive: u64) -> (LlamaCpp, PathBuf) {
        use std::os::unix::fs::PermissionsExt;
        let folder =
            std::env::temp_dir().join(format!("ferro64-llama-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&folder).unwrap();
        let binary = folder.join("fake-llama-server");
        std::fs::write(&binary, FAKE_SERVER).unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        for name in ["model.gguf", "other.gguf"] {
            std::fs::write(
                folder.join(name),
                gguf::fixture(&[("tokenizer.chat_template", "{% if enable_thinking %}{% endif %}")], &[]),
            )
            .unwrap();
        }
        let registry = folder.join("models.json");
        std::fs::write(
            &registry,
            r#"[{"id":"fixture","path":"model.gguf"},{"id":"other","path":"other.gguf"}]"#,
        )
        .unwrap();
        let runtime = LlamaCpp {
            binary,
            registry,
            base_dir: folder.clone(),
            extra_dirs: Vec::new(),
            keep_alive: Duration::from_secs(keep_alive),
            context: 4096,
            gpu_layers: 0,
            port: None,
            process: Mutex::new(None),
        };
        (runtime, folder)
    }

    #[cfg(unix)]
    fn alive(pid: &str) -> bool {
        std::process::Command::new("/bin/kill")
            .args(["-0", pid.trim()])
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success()
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn strict_mode_terminates_after_success_and_inference_failure() {
        let (runtime, folder) = fixture_runtime(0);
        for fail in [false, true] {
            if fail {
                std::fs::write(folder.join("fail"), b"yes").unwrap();
            }
            let result = runtime
                .chat(&Client::new(), "fixture", &[], ChatOptions::default(), Emitter::new(None))
                .await;
            if fail {
                assert!(result.unwrap_err().contains("fixture inference failure"));
            } else {
                let outcome = result.unwrap();
                assert_eq!(outcome.content, "fixture answer");
                assert_eq!(outcome.reasoning, "mull");
                assert_eq!(outcome.stats.tokens_per_second, Some(40.0));
                assert!(outcome.stats.load_ms.is_some());
            }
            assert!(runtime.loaded().await.unwrap().is_empty());
            let starts = std::fs::read_to_string(folder.join("starts")).unwrap();
            let pid = starts.lines().last().unwrap();
            assert!(!alive(pid), "managed llama-server process remained alive");
        }
        std::fs::remove_dir_all(folder).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn warm_server_is_reused_swapped_on_model_change_and_reaped_when_idle() {
        let (runtime, folder) = fixture_runtime(3600);
        let client = Client::new();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let options = ChatOptions {
            thinking: true,
            effort: Effort::High,
            ..Default::default()
        };
        let first = runtime
            .chat(&client, "fixture", &[], options, Emitter::new(Some(tx)))
            .await
            .unwrap();
        assert!(first.stats.load_ms.is_some());
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        assert!(events.contains(&StreamEvent::Status("loading".into())));
        assert!(events.contains(&StreamEvent::Content("fixture ".into())));
        let sent: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(folder.join("last-request")).unwrap()).unwrap();
        assert_eq!(sent["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(sent["chat_template_kwargs"]["reasoning_effort"], "high");

        let second = runtime
            .chat(&client, "fixture", &[], options, Emitter::new(None))
            .await
            .unwrap();
        assert_eq!(second.stats.load_ms, None, "warm server must be reused");
        let starts = std::fs::read_to_string(folder.join("starts")).unwrap();
        assert_eq!(starts.lines().count(), 1);
        let first_pid = starts.lines().next().unwrap().to_string();
        assert_eq!(runtime.loaded().await.unwrap(), ["fixture"]);
        assert!(runtime.unload_unless("fixture").await.unwrap().is_empty());

        runtime
            .chat(&client, "other", &[], options, Emitter::new(None))
            .await
            .unwrap();
        assert!(!alive(&first_pid), "switching models must stop the old server");
        assert_eq!(runtime.loaded().await.unwrap(), ["other"]);

        let models = runtime.models().await.unwrap();
        let other = models.iter().find(|m| m.id == "other").unwrap();
        assert!(other.is_resident);
        assert_eq!(other.thinking, "toggle");

        assert_eq!(runtime.reap_idle().await, None);
        runtime.process.lock().await.as_mut().unwrap().last_used =
            Instant::now() - Duration::from_secs(7200);
        assert_eq!(runtime.reap_idle().await.as_deref(), Some("other"));
        assert!(runtime.loaded().await.unwrap().is_empty());
        std::fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn folder_discovery_maps_catalog_files_and_pairs_only_matching_projector() {
        let folder =
            std::env::temp_dir().join(format!("ferro64-discovery-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&folder).unwrap();
        for name in [
            "qwen2.5-3b-instruct-q4_k_m.gguf",
            "qwen2.5-coder-3b-q4_k_m.gguf",
            "GLM-4.7-Flash-Q4_K_M.gguf",
            "photo.gguf",
            "photo.mmproj.gguf",
            "other-projector.gguf",
            "all-minilm.gguf",
            "flux.gguf",
        ] {
            std::fs::write(folder.join(name), b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
        }
        std::fs::write(folder.join("not-valid.gguf"), b"wrong format").unwrap();
        let models = discover_folder(&folder);
        assert_eq!(models.len(), 4);
        assert!(models.iter().any(|model| model.id == "qwen2.5:3b"));
        assert!(models.iter().any(|model| model.id == "qwen2.5-coder:3b"));
        assert!(models.iter().any(|model| model.id == "glm-4.7-flash"));
        assert_eq!(
            models
                .iter()
                .find(|model| model.id == "photo")
                .unwrap()
                .mmproj
                .as_ref()
                .unwrap(),
            &folder.join("photo.mmproj.gguf")
        );
        std::fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn ollama_discovery_reuses_only_valid_model_layers_and_preserves_exact_tags() {
        let folder =
            std::env::temp_dir().join(format!("ferro64-cache-test-{}", uuid::Uuid::new_v4()));
        let digest = "a".repeat(64);
        let manifests = folder.join("manifests/registry.ollama.ai/library/qwen2.5");
        let namespaced = folder.join("manifests/registry.ollama.ai/SparkLLM/Spark-X2.6-4B");
        std::fs::create_dir_all(&manifests).unwrap();
        std::fs::create_dir_all(&namespaced).unwrap();
        std::fs::create_dir_all(folder.join("blobs")).unwrap();
        std::fs::write(
            folder.join("blobs").join(format!("sha256-{digest}")),
            b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )
        .unwrap();
        let manifest = serde_json::json!({"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": format!("sha256:{digest}")}]}).to_string();
        std::fs::write(manifests.join("3b"), &manifest).unwrap();
        std::fs::write(namespaced.join("q4_k_m"), &manifest).unwrap();
        let models = discover_ollama(&folder);
        assert_eq!(models.len(), 2);
        assert!(models.iter().any(|model| model.id == "qwen2.5:3b"));
        assert!(models
            .iter()
            .any(|model| model.id == "SparkLLM/Spark-X2.6-4B:q4_k_m"));
        assert!(models.iter().all(|model| valid_gguf(&model.path)));
        std::fs::remove_dir_all(folder).unwrap();
    }
}
