use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex, MutexGuard};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub id: String,
    pub name: String,
    pub parameter_size: String,
    pub role_description: String,
    pub provider: String,
    pub is_image_generator: bool,
    pub is_resident: bool,
    pub capabilities: Vec<String>,
    pub installed: bool,
    pub source: String,
    pub size_bytes: Option<u64>,
    /// "none", "always", "toggle" or "effort"; see gguf::Thinking.
    #[serde(default = "thinking_none")]
    pub thinking: String,
    #[serde(default)]
    pub context_length: Option<u64>,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub architecture: Option<String>,
}

fn thinking_none() -> String {
    "none".to_string()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Low,
    #[default]
    Medium,
    High,
}

impl Effort {
    pub fn as_str(self) -> &'static str {
        match self {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
        }
    }

    /// Output budget, including any reasoning tokens.
    pub fn max_tokens(self) -> u32 {
        match self {
            Effort::Low => 1024,
            Effort::Medium => 4096,
            Effort::High => 8192,
        }
    }

    /// Prompt guidance so that effort also changes non-reasoning models.
    pub fn guidance(self) -> &'static str {
        match self {
            Effort::Low => "Effort: low. Answer directly and briefly; skip preamble and optional detail.",
            Effort::Medium => "Effort: medium. Give a complete answer at a sensible length.",
            Effort::High => "Effort: high. Work carefully: consider edge cases, verify each step, and give a thorough answer.",
        }
    }
}

/// Sampling temperature. It changes how varied the wording is, not what the
/// model knows: a wrong fact stays wrong at any setting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Temperature {
    Precise,
    #[default]
    Balanced,
    Creative,
}

impl Temperature {
    pub fn as_str(self) -> &'static str {
        match self {
            Temperature::Precise => "precise",
            Temperature::Balanced => "balanced",
            Temperature::Creative => "creative",
        }
    }

    pub fn value(self) -> f32 {
        match self {
            Temperature::Precise => 0.2,
            Temperature::Balanced => 0.6,
            Temperature::Creative => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ChatOptions {
    pub thinking: bool,
    pub effort: Effort,
    pub temperature: Temperature,
}

impl ChatOptions {
    /// Output budget. Reasoning shares it with the answer, so a reasoning
    /// model gets room to think as well as answer.
    pub fn max_tokens(&self, reasons: bool) -> u32 {
        if !reasons {
            return self.effort.max_tokens();
        }
        match self.effort {
            Effort::Low => 4096,
            Effort::Medium => 12_288,
            Effort::High => 24_576,
        }
    }
}

pub const REASONING_EXHAUSTED: &str =
    "The model used its whole output budget on reasoning. Raise the effort or turn thinking off.";

#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    Status(String),
    Reasoning(String),
    Content(String),
}

pub type StreamSink = mpsc::UnboundedSender<StreamEvent>;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ChatStats {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    /// Of the prompt, how much a hosted vendor served from its cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    pub tokens_per_second: Option<f64>,
    pub load_ms: Option<u64>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ChatOutcome {
    pub content: String,
    pub reasoning: String,
    pub duration_ms: u64,
    pub stats: ChatStats,
    /// The receiver went away (user pressed stop or disconnected).
    pub cancelled: bool,
}

/// A tool the model asked to run: `arguments` is its JSON text, as sent.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// One assistant turn of a tool-using conversation.
#[derive(Debug, Clone, Default)]
pub struct ToolTurn {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub duration_ms: u64,
    pub stats: ChatStats,
    pub cancelled: bool,
}

/// Collects a streamed answer, forwards deltas to an optional listener and
/// separates inline `<think>…</think>` markup when a runtime does not.
pub struct Emitter {
    sink: Option<StreamSink>,
    pub content: String,
    pub reasoning: String,
    pending: String,
    in_think: bool,
    pub closed: bool,
}

const OPEN_THINK: &str = "<think>";
const CLOSE_THINK: &str = "</think>";

impl Emitter {
    pub fn new(sink: Option<StreamSink>) -> Self {
        Self {
            sink,
            content: String::new(),
            reasoning: String::new(),
            pending: String::new(),
            in_think: false,
            closed: false,
        }
    }

    fn send(&mut self, event: StreamEvent) {
        if let Some(sink) = &self.sink {
            if sink.send(event).is_err() {
                self.closed = true;
            }
        }
    }

    pub fn status(&mut self, text: impl Into<String>) {
        self.send(StreamEvent::Status(text.into()));
    }

    pub fn reasoning(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        self.reasoning.push_str(delta);
        self.send(StreamEvent::Reasoning(delta.to_string()));
    }

    fn content_raw(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        self.content.push_str(delta);
        self.send(StreamEvent::Content(delta.to_string()));
    }

    /// Content deltas may contain think tags split across chunk boundaries.
    pub fn content(&mut self, delta: &str) {
        self.pending.push_str(delta);
        loop {
            let tag = if self.in_think { CLOSE_THINK } else { OPEN_THINK };
            if let Some(at) = self.pending.find(tag) {
                let before: String = self.pending.drain(..at).collect();
                self.pending.drain(..tag.len());
                if self.in_think {
                    self.reasoning(&before);
                } else {
                    self.content_raw(&before);
                }
                self.in_think = !self.in_think;
                continue;
            }
            // Hold back a suffix that could be the start of the tag.
            let keep = (1..tag.len())
                .rev()
                .find(|n| self.pending.ends_with(&tag[..*n]))
                .unwrap_or(0);
            let flush: String = self.pending.drain(..self.pending.len() - keep).collect();
            if self.in_think {
                self.reasoning(&flush);
            } else {
                self.content_raw(&flush);
            }
            break;
        }
    }

    pub fn finish(&mut self) {
        let rest = std::mem::take(&mut self.pending);
        if self.in_think {
            self.reasoning(&rest);
        } else {
            self.content_raw(&rest);
        }
        // Leading whitespace after a closed think block is noise.
        let trimmed = self.content.trim_start().len();
        if trimmed != self.content.len() {
            self.content = self.content.trim_start().to_string();
        }
    }
}

#[derive(Deserialize)]
struct ModelList {
    models: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    name: String,
    size: Option<u64>,
    details: Option<ModelDetails>,
}

#[derive(Deserialize)]
struct ModelDetails {
    parameter_size: Option<String>,
    quantization_level: Option<String>,
    family: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessagePayload {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

/// Every GPU operation initiated by this app shares this queue, including image
/// generation in main.rs. Run one app process per Ollama/MLX worker instance.
/// One model stays resident between requests; switching models unloads it first.
#[derive(Clone)]
pub struct LlmEngine {
    client: Client,
    base_url: String,
    worker_url: Option<String>,
    inference: std::sync::Arc<Mutex<()>>,
    backend: String,
    llamacpp: Option<std::sync::Arc<crate::llamacpp::LlamaCpp>>,
    /// Semantic search, beside the llama.cpp chat model.
    embedder: Option<std::sync::Arc<crate::embed::Embedder>>,
    ollama_keep_alive: serde_json::Value,
    num_ctx: u32,
}

fn parse_keep_alive(raw: Option<String>) -> serde_json::Value {
    match raw {
        Some(value) => value
            .trim()
            .parse::<i64>()
            .map(serde_json::Value::from)
            .unwrap_or_else(|_| serde_json::Value::from(value.trim())),
        None => serde_json::Value::from("10m"),
    }
}

impl LlmEngine {
    pub fn new() -> Self {
        let backend = std::env::var("LLM_BACKEND")
            .unwrap_or_else(|_| "llamacpp".to_string())
            .to_lowercase();
        let llamacpp =
            (backend == "llamacpp").then(|| std::sync::Arc::new(crate::llamacpp::LlamaCpp::new()));
        let embedder = llamacpp
            .as_ref()
            .map(|runtime| std::sync::Arc::new(crate::embed::Embedder::new(runtime.binary_path(), runtime.base_dir())));
        Self {
            client: Client::builder()
                .connect_timeout(Duration::from_secs(3))
                // An idle limit, not a total one: a slow dense model can
                // stream for well over 15 minutes, but a stalled one is cut
                // off. Five minutes covers reading a long prompt.
                .read_timeout(Duration::from_secs(300))
                .build()
                .expect("HTTP client initialization failed"),
            base_url: std::env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string())
                .trim_end_matches('/')
                .to_string(),
            worker_url: Some(
                std::env::var("IMAGE_WORKER_URL")
                    .unwrap_or_else(|_| "http://127.0.0.1:8001".to_string())
                    .trim_end_matches('/')
                    .to_string(),
            ),
            inference: std::sync::Arc::new(Mutex::new(())),
            backend,
            llamacpp,
            embedder,
            ollama_keep_alive: parse_keep_alive(std::env::var("OLLAMA_KEEP_ALIVE").ok()),
            num_ctx: std::env::var("OLLAMA_NUM_CTX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32_768),
        }
    }

    /// The embedding model, when the llama.cpp backend is in use.
    pub fn embedder(&self) -> Option<&std::sync::Arc<crate::embed::Embedder>> {
        self.embedder.as_ref()
    }

    pub fn engine_label(&self) -> &'static str {
        if self.backend == "ollama" { "Ollama" } else { "llama.cpp" }
    }

    pub fn backend_name(&self) -> &'static str {
        match self.backend.as_str() {
            "ollama" => "ollama",
            "llamacpp" => "llamacpp",
            _ => "unknown",
        }
    }

    /// Seconds a model stays resident after its last request (0 = unload at once).
    pub fn keep_alive_label(&self) -> String {
        match &self.llamacpp {
            Some(runtime) => format!("{}s", runtime.keep_alive().as_secs()),
            None => match &self.ollama_keep_alive {
                serde_json::Value::String(s) => s.clone(),
                other => format!("{other}s"),
            },
        }
    }

    pub fn context_size(&self) -> u32 {
        match &self.llamacpp {
            Some(runtime) => runtime.context_size(),
            None => self.num_ctx,
        }
    }

    fn keeps_models_warm(&self) -> bool {
        match &self.llamacpp {
            Some(runtime) => !runtime.keep_alive().is_zero(),
            None => !matches!(&self.ollama_keep_alive, serde_json::Value::Number(n) if n.as_i64() == Some(0))
                && self.ollama_keep_alive != serde_json::Value::from("0"),
        }
    }

    pub async fn runtime_available(&self) -> Result<(), String> {
        if let Some(runtime) = &self.llamacpp {
            return runtime.available();
        }
        if self.backend != "ollama" {
            return Err(format!(
                "Unsupported LLM_BACKEND '{}'; use ollama or llamacpp",
                self.backend
            ));
        }
        self.entries("/api/ps").await.map(|_| ())
    }

    pub async fn acquire(&self) -> MutexGuard<'_, ()> {
        self.inference.lock().await
    }

    /// Unloads an idle llama-server once its keep-alive has elapsed. Never
    /// waits for the queue: a busy queue means the model is in use.
    pub fn spawn_idle_reaper(self: &std::sync::Arc<Self>) {
        let Some(runtime) = self.llamacpp.clone() else {
            return;
        };
        let engine = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(5));
            loop {
                tick.tick().await;
                if let Some(embedder) = &engine.embedder {
                    if embedder.reap_idle().await {
                        tracing::info!("Stopped the idle embedding server");
                    }
                }
                if let Ok(_guard) = engine.inference.try_lock() {
                    if let Some(model) = runtime.reap_idle().await {
                        tracing::info!("Unloaded idle model {model}");
                    }
                }
            }
        });
    }

    async fn entries(&self, endpoint: &str) -> Result<Vec<ModelEntry>, String> {
        let response = self
            .client
            .get(format!("{}{endpoint}", self.base_url))
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| format!("Ollama is unavailable at {}: {e}", self.base_url))?;
        let response = response
            .error_for_status()
            .map_err(|e| format!("Ollama {endpoint}: {e}"))?;
        response
            .json::<ModelList>()
            .await
            .map(|data| data.models)
            .map_err(|e| format!("Invalid Ollama {endpoint} response: {e}"))
    }

    pub async fn get_loaded_models_checked(&self) -> Result<Vec<String>, String> {
        if let Some(runtime) = &self.llamacpp {
            return runtime.loaded().await;
        }
        Ok(self
            .entries("/api/ps")
            .await?
            .into_iter()
            .map(|model| model.name)
            .collect())
    }

    async fn show_capabilities(&self, model: &str) -> Result<Vec<String>, String> {
        let response = self
            .client
            .post(format!("{}/api/show", self.base_url))
            .json(&serde_json::json!({"model": model}))
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| format!("Cannot inspect model '{model}': {e}"))?;
        let response = response
            .error_for_status()
            .map_err(|e| format!("Cannot inspect model '{model}': {e}"))?;
        let data: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
        Ok(data
            .get("capabilities")
            .and_then(|v| v.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Only /api/tags verifies installation. The saved list is an offline inventory,
    /// never proof that Ollama is reachable or that a tag is still installed.
    pub async fn models(&self) -> Result<Vec<ModelSpec>, String> {
        let mut models = self.all_models().await?;
        // Image models need this exact file as their text reader. It is a part
        // of them now, not a model to choose.
        for model in models.iter_mut().filter(|model| model.id == "qwen3:4b") {
            model.capabilities.retain(|capability| capability != "chat");
        }
        Ok(models)
    }

    async fn all_models(&self) -> Result<Vec<ModelSpec>, String> {
        if let Some(runtime) = &self.llamacpp {
            return runtime.models().await;
        }
        if self.backend != "ollama" {
            return Err(format!("Unsupported LLM_BACKEND: {}", self.backend));
        }
        let entries = match self.entries("/api/tags").await {
            Ok(entries) => entries,
            Err(error) => {
                tracing::warn!("{error}; displaying unverified data/list.txt inventory");
                return Ok(self.get_curated_models());
            }
        };
        let loaded = self.get_loaded_models_checked().await.unwrap_or_default();
        let mut models = Vec::with_capacity(entries.len());
        for entry in entries {
            let mut spec = model_spec(&entry.name);
            spec.installed = true;
            spec.source = "ollama".to_string();
            spec.size_bytes = entry.size;
            if let Some(details) = entry.details {
                spec.parameter_size = details
                    .parameter_size
                    .unwrap_or_else(|| "Unknown".to_string());
                spec.quantization = details.quantization_level;
                spec.architecture = details.family;
            }
            spec.is_resident = loaded.iter().any(|name| same_model(name, &spec.id));
            if let Ok(caps) = self.show_capabilities(&entry.name).await {
                // Some older Ollama versions omit capabilities. In that case use
                // conservative name hints, without invented model specifications.
                if !caps.is_empty() {
                    spec.capabilities.clear();
                    if caps.iter().any(|c| c == "completion") {
                        spec.capabilities.push("chat".to_string());
                    }
                    if caps.iter().any(|c| c == "vision") {
                        spec.capabilities.push("vision".to_string());
                    }
                    if caps.iter().any(|c| c == "embedding") {
                        spec.capabilities.push("embedding".to_string());
                    }
                    if caps.iter().any(|c| c == "thinking") {
                        spec.capabilities.push("thinking".to_string());
                        spec.thinking = if entry.name.contains("gpt-oss") {
                            "effort"
                        } else {
                            "toggle"
                        }
                        .to_string();
                    }
                    if spec.is_image_generator {
                        spec.capabilities = vec!["image_generation".to_string()];
                    }
                }
            }
            models.push(spec);
        }
        models.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(models)
    }

    pub fn get_curated_models(&self) -> Vec<ModelSpec> {
        let base = std::env::var("FERRO64_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
        let inventory = std::fs::read_to_string(base.join("data/list.txt")).unwrap_or_default();
        parse_inventory(&inventory)
    }

    /// Caller MUST hold acquire() across this call and the external GPU operation.
    pub async fn unload_all_except_locked(
        &self,
        keep_model: Option<&str>,
    ) -> Result<Vec<String>, String> {
        if let Some(runtime) = &self.llamacpp {
            let mut unloaded = match keep_model {
                Some(keep) => runtime.unload_unless(keep).await?,
                None => runtime.unload().await?,
            };
            unloaded.extend(self.purge_ollama_if_running().await?);
            return Ok(unloaded);
        }
        let loaded = self.get_loaded_models_checked().await?;
        let mut unloaded = Vec::new();
        for name in loaded {
            if keep_model.is_some_and(|keep| same_model(keep, &name)) {
                continue;
            }
            self.unload_model(&name).await?;
            unloaded.push(name);
        }
        // Reject a transition if Ollama still reports an old resident. Prefix
        // matching here would incorrectly equate qwen2.5:3b and qwen2.5:32b.
        let remaining = self.get_loaded_models_checked().await?;
        let unexpected: Vec<_> = remaining
            .into_iter()
            .filter(|name| !keep_model.is_some_and(|keep| same_model(keep, name)))
            .collect();
        if !unexpected.is_empty() {
            return Err(format!(
                "Ollama still reports resident models after unload: {}",
                unexpected.join(", ")
            ));
        }
        Ok(unloaded)
    }

    async fn purge_ollama_if_running(&self) -> Result<Vec<String>, String> {
        let response = match self
            .client
            .get(format!("{}/api/ps", self.base_url))
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) if error.is_connect() => return Ok(Vec::new()),
            Err(error) => return Err(format!("Cannot verify previous Ollama residency: {error}")),
        };
        let data = response
            .error_for_status()
            .map_err(|error| error.to_string())?
            .json::<ModelList>()
            .await
            .map_err(|error| error.to_string())?;
        let mut unloaded = Vec::new();
        for model in data.models {
            self.unload_model(&model.name).await?;
            unloaded.push(model.name);
        }
        if !self.entries("/api/ps").await?.is_empty() {
            return Err("Ollama still has resident models; stop those models before direct llama.cpp inference".to_string());
        }
        Ok(unloaded)
    }

    async fn unload_model(&self, model: &str) -> Result<(), String> {
        let response = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&serde_json::json!({"model": model, "keep_alive": 0, "stream": false}))
            .send()
            .await
            .map_err(|e| format!("Failed to unload '{model}': {e}"))?;
        if !response.status().is_success() {
            return Err(format!(
                "Failed to unload '{model}': {}",
                response.text().await.unwrap_or_default()
            ));
        }
        Ok(())
    }

    pub async fn unload_all_except(&self, keep_model: Option<&str>) -> Result<Vec<String>, String> {
        let _guard = self.acquire().await;
        self.unload_worker_locked().await?;
        self.unload_all_except_locked(keep_model).await
    }

    /// Loads a model ahead of the first message so the reply starts sooner.
    pub async fn preload(&self, model: &str) -> Result<u64, String> {
        let engine = self.clone();
        let model = model.to_string();
        tokio::spawn(async move {
            let _guard = engine.acquire().await;
            engine.unload_worker_locked().await?;
            engine.unload_all_except_locked(Some(&model)).await?;
            let start = Instant::now();
            if let Some(runtime) = &engine.llamacpp {
                runtime.preload(&engine.client, &model).await?;
            } else {
                let response = engine
                    .client
                    .post(format!("{}/api/generate", engine.base_url))
                    .json(&serde_json::json!({"model": model, "keep_alive": engine.ollama_keep_alive}))
                    .send()
                    .await
                    .map_err(|e| format!("Could not load '{model}': {e}"))?;
                if !response.status().is_success() {
                    return Err(format!(
                        "Could not load '{model}': {}",
                        response.text().await.unwrap_or_default()
                    ));
                }
            }
            Ok(start.elapsed().as_millis() as u64)
        })
        .await
        .map_err(|error| format!("Model loader failed: {error}"))?
    }

    /// A reachable legacy MLX worker without /unload is unsafe to coexist with a
    /// second GPU model. An absent worker is fine; its CLI fallback exits itself.
    pub async fn unload_worker_locked(&self) -> Result<(), String> {
        let Some(url) = &self.worker_url else {
            return Ok(());
        };
        // The port may belong to something else, such as an app the coding agent
        // is testing. Only the worker itself is told to unload.
        let identity = self.client.get(format!("{url}/health")).timeout(Duration::from_secs(3)).send().await;
        let is_worker = match identity {
            Ok(response) => is_image_worker(&response.text().await.unwrap_or_default()),
            Err(_) => false,
        };
        if !is_worker {
            return Ok(());
        }
        match self.client.post(format!("{url}/unload")).timeout(Duration::from_secs(60)).send().await {
            Ok(response) if response.status().is_success() => Ok(()),
            Ok(response) => Err(format!("Image worker could not unload (HTTP {}). Restart the updated worker.py before using another model.", response.status())),
            Err(error) if error.is_connect() => Ok(()),
            Err(error) => Err(format!("Could not confirm image worker unload: {error}")),
        }
    }

    pub async fn generate_embedding(&self, text: &str) -> Option<Vec<f32>> {
        match self.generate_embedding_checked(text).await {
            Ok(vector) => Some(vector),
            Err(error) => {
                tracing::warn!("Semantic recall skipped: {error}");
                None
            }
        }
    }

    pub async fn generate_embedding_checked(&self, text: &str) -> Result<Vec<f32>, String> {
        let engine = self.clone();
        let text = text.to_string();
        // Owned work keeps the GPU lease through cleanup even when the HTTP
        // client disconnects and its request future is dropped.
        tokio::spawn(async move { engine.generate_embedding_inner(&text).await })
            .await
            .map_err(|error| format!("Embedding worker failed: {error}"))?
    }

    async fn generate_embedding_inner(&self, text: &str) -> Result<Vec<f32>, String> {
        if self.llamacpp.is_some() {
            return Err("Semantic recall is disabled for the direct llama.cpp backend; no embedding model is configured".to_string());
        }
        let _guard = self.acquire().await;
        self.unload_worker_locked().await?;
        self.unload_all_except_locked(None).await?;
        let result = async {
            let response = self.client.post(format!("{}/api/embed", self.base_url))
                .json(&serde_json::json!({"model": "all-minilm:latest", "input": text, "keep_alive": 0}))
                .send().await.map_err(|e| format!("Embedding request failed: {e}"))?;
            if !response.status().is_success() { return Err(format!("Embedding error: {}", response.text().await.unwrap_or_default())); }
            let data: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
            serde_json::from_value::<Vec<Vec<f32>>>(data.get("embeddings").cloned().unwrap_or_default())
                .map_err(|e| format!("Invalid embedding response: {e}"))?
                .into_iter().next().filter(|v| !v.is_empty()).ok_or_else(|| "Empty embedding response".to_string())
        }.await;
        let cleanup = self.unload_all_except_locked(None).await;
        finish_with_cleanup(result, cleanup)
    }

    /// Non-streaming convenience used by the agent workflow.
    pub async fn chat_completion(
        &self,
        model: &str,
        messages: &[ChatMessagePayload],
        options: ChatOptions,
    ) -> Result<(String, u64), String> {
        let outcome = self.chat_stream(model, messages, options, None).await?;
        Ok((outcome.content, outcome.duration_ms))
    }

    /// Runs one chat turn. Deltas go to `sink` as they arrive; dropping the
    /// receiver stops generation and returns the partial answer as cancelled.
    pub async fn chat_stream(
        &self,
        model: &str,
        messages: &[ChatMessagePayload],
        options: ChatOptions,
        sink: Option<StreamSink>,
    ) -> Result<ChatOutcome, String> {
        let engine = self.clone();
        let model = model.to_string();
        let messages = messages.to_vec();
        tokio::spawn(async move {
            engine
                .chat_inner(&model, &messages, options, Emitter::new(sink))
                .await
        })
        .await
        .map_err(|error| format!("Inference worker failed: {error}"))?
    }

    /// One turn of a tool-using conversation (llama.cpp only). Text deltas go
    /// to `sink`; the GPU queue is held for the turn, not while tools run.
    pub async fn chat_tools(
        &self,
        model: &str,
        messages: Vec<serde_json::Value>,
        tools: serde_json::Value,
        options: ChatOptions,
        sink: Option<StreamSink>,
    ) -> Result<ToolTurn, String> {
        if crate::hosted::is_hosted(model) {
            let model = model.to_string();
            return tokio::spawn(async move {
                let mut emitter = Emitter::new(sink);
                crate::hosted::chat_tools(&model, &messages, &tools, options, &mut emitter).await
            })
            .await
            .map_err(|error| format!("Inference worker failed: {error}"))?;
        }
        let Some(runtime) = self.llamacpp.clone() else {
            return Err("Coding tools need the llama.cpp backend".to_string());
        };
        let engine = self.clone();
        let model = model.to_string();
        tokio::spawn(async move {
            let mut emitter = Emitter::new(sink);
            emitter.status("queued");
            let _guard = engine.acquire().await;
            engine.unload_worker_locked().await?;
            engine.unload_all_except_locked(Some(&model)).await?;
            runtime
                .chat_tools(&engine.client, &model, &messages, &tools, options, &mut emitter)
                .await
        })
        .await
        .map_err(|error| format!("Inference worker failed: {error}"))?
    }

    async fn chat_inner(
        &self,
        model: &str,
        messages: &[ChatMessagePayload],
        options: ChatOptions,
        mut emitter: Emitter,
    ) -> Result<ChatOutcome, String> {
        if crate::hosted::is_hosted(model) {
            return crate::hosted::chat(model, messages, options, emitter).await;
        }
        emitter.status("queued");
        let _guard = self.acquire().await;
        self.unload_worker_locked().await?;
        self.unload_all_except_locked(Some(model)).await?;
        if let Some(runtime) = &self.llamacpp {
            return runtime
                .chat(&self.client, model, messages, options, emitter)
                .await;
        }
        if self.backend != "ollama" {
            return Err(format!("Unsupported LLM_BACKEND: {}", self.backend));
        }
        let t0 = Instant::now();
        let result = self.ollama_chat(model, messages, options, &mut emitter).await;
        let result = result.map(|mut outcome| {
            outcome.duration_ms = t0.elapsed().as_millis() as u64;
            outcome
        });
        // A failed request may leave a half-loaded model; always clear it then.
        if result.is_err() || !self.keeps_models_warm() {
            let cleanup = self.unload_all_except_locked(None).await;
            return finish_with_cleanup(result, cleanup);
        }
        result
    }

    async fn ollama_chat(
        &self,
        model: &str,
        messages: &[ChatMessagePayload],
        options: ChatOptions,
        emitter: &mut Emitter,
    ) -> Result<ChatOutcome, String> {
        let capabilities = self.show_capabilities(model).await?;
        if capabilities.iter().any(|c| c == "embedding")
            && !capabilities.iter().any(|c| c == "completion")
        {
            return Err(format!(
                "'{model}' is an embedding model; choose a chat model"
            ));
        }
        let mut body = serde_json::json!({
            "model": model, "messages": messages, "stream": true, "keep_alive": self.ollama_keep_alive,
            "options": {"num_predict": options.max_tokens(options.thinking), "num_ctx": self.num_ctx, "temperature": options.temperature.value()}
        });
        if capabilities.iter().any(|c| c == "thinking") {
            // gpt-oss takes a level and cannot turn reasoning off.
            body["think"] = if model.contains("gpt-oss") {
                serde_json::Value::from(options.effort.as_str())
            } else {
                serde_json::Value::Bool(options.thinking)
            };
        }
        emitter.status("loading");
        let mut response = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Ollama request failed: {e}"))?;
        if !response.status().is_success() {
            let status = response.status();
            return Err(format!(
                "Ollama {status}: {}",
                response.text().await.unwrap_or_default()
            ));
        }
        let mut stats = ChatStats::default();
        let mut buffer = Vec::new();
        let mut generating = false;
        'read: while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| format!("Ollama stream interrupted: {e}"))?
        {
            buffer.extend_from_slice(&chunk);
            while let Some(newline) = buffer.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = buffer.drain(..=newline).collect();
                apply_ollama_line(&line, emitter, &mut stats, &mut generating)?;
                if emitter.closed {
                    break 'read;
                }
            }
        }
        if !emitter.closed && !buffer.is_empty() {
            apply_ollama_line(&buffer, emitter, &mut stats, &mut generating)?;
        }
        emitter.finish();
        finalize(emitter, stats)
    }
}

fn apply_ollama_line(
    line: &[u8],
    emitter: &mut Emitter,
    stats: &mut ChatStats,
    generating: &mut bool,
) -> Result<(), String> {
    let text = String::from_utf8_lossy(line);
    let text = text.trim();
    if text.is_empty() {
        return Ok(());
    }
    let data: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("Invalid Ollama stream: {e}"))?;
    if let Some(error) = data.get("error").and_then(|e| e.as_str()) {
        return Err(format!("Ollama: {error}"));
    }
    if !*generating {
        *generating = true;
        emitter.status("generating");
    }
    if let Some(thinking) = data.pointer("/message/thinking").and_then(|v| v.as_str()) {
        emitter.reasoning(thinking);
    }
    if let Some(content) = data.pointer("/message/content").and_then(|v| v.as_str()) {
        emitter.content(content);
    }
    if data.get("done").and_then(|v| v.as_bool()) == Some(true) {
        let eval = data.get("eval_count").and_then(|v| v.as_u64());
        let eval_ns = data.get("eval_duration").and_then(|v| v.as_u64());
        stats.completion_tokens = eval;
        stats.prompt_tokens = data.get("prompt_eval_count").and_then(|v| v.as_u64());
        stats.load_ms = data
            .get("load_duration")
            .and_then(|v| v.as_u64())
            .map(|ns| ns / 1_000_000);
        stats.tokens_per_second = match (eval, eval_ns) {
            (Some(count), Some(ns)) if ns > 0 => Some(count as f64 * 1e9 / ns as f64),
            _ => None,
        };
        stats.finish_reason = data
            .get("done_reason")
            .and_then(|v| v.as_str())
            .map(str::to_string);
    }
    Ok(())
}

/// A reasoning-only reply is incomplete and must not be presented as an answer.
pub fn finalize(emitter: &Emitter, stats: ChatStats) -> Result<ChatOutcome, String> {
    if !emitter.closed && emitter.content.trim().is_empty() {
        return Err(if emitter.reasoning.is_empty() {
            "Model returned no final answer. Try another model or a shorter prompt.".to_string()
        } else {
            REASONING_EXHAUSTED.to_string()
        });
    }
    Ok(ChatOutcome {
        content: emitter.content.clone(),
        reasoning: emitter.reasoning.trim().to_string(),
        duration_ms: 0,
        stats,
        cancelled: emitter.closed,
    })
}

pub fn finish_with_cleanup<T>(
    result: Result<T, String>,
    cleanup: Result<Vec<String>, String>,
) -> Result<T, String> {
    match (result, cleanup) {
        (Ok(value), Ok(_)) => Ok(value),
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(error)) => Err(format!(
            "Inference completed but model cleanup failed: {error}"
        )),
        (Err(error), Err(cleanup)) => Err(format!("{error}; cleanup also failed: {cleanup}")),
    }
}

fn same_model(a: &str, b: &str) -> bool {
    fn normalized(value: &str) -> String {
        if value.rsplit('/').next().unwrap_or(value).contains(':') {
            value.to_string()
        } else {
            format!("{value}:latest")
        }
    }
    normalized(a) == normalized(b)
}

fn model_spec(id: &str) -> ModelSpec {
    let lower = id.to_lowercase();
    let image = lower.contains("flux");
    let embedding = lower.starts_with("all-minilm:") || lower.contains("embed");
    let vision = lower.contains("moondream") || lower.contains("wevisone-vl");
    let capabilities = if image {
        vec!["image_generation".to_string()]
    } else if embedding {
        vec!["embedding".to_string()]
    } else if vision {
        vec!["chat".to_string(), "vision".to_string()]
    } else {
        vec!["chat".to_string()]
    };
    ModelSpec {
        id: id.to_string(),
        name: id.rsplit('/').next().unwrap_or(id).replace(":latest", ""),
        parameter_size: "Unknown".to_string(),
        role_description: if image {
            "Local image generation; MLX worker requires separate mflux weights"
        } else if embedding {
            "Local semantic search embeddings"
        } else if vision {
            "Image understanding; runtime support verified by Ollama"
        } else {
            "Local text model; capabilities inspected through Ollama"
        }
        .to_string(),
        provider: "Local".to_string(),
        is_image_generator: image,
        is_resident: false,
        capabilities,
        installed: false,
        source: "list.txt".to_string(),
        size_bytes: None,
        thinking: if lower.contains("deepseek-r1") {
            "always"
        } else {
            "none"
        }
        .to_string(),
        context_length: None,
        quantization: None,
        architecture: None,
    }
}

fn parse_inventory(inventory: &str) -> Vec<ModelSpec> {
    let mut models = Vec::new();
    for line in inventory.lines() {
        let columns: Vec<_> = line.split_whitespace().collect();
        if columns.len() < 4 || columns[0] == "NAME" || columns[0] == "ollama" {
            continue;
        }
        if !columns[1].chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        let mut spec = model_spec(columns[0]);
        if let Ok(number) = columns[2].parse::<f64>() {
            let factor = match columns[3] {
                "GB" => 1_000_000_000.0,
                "MB" => 1_000_000.0,
                "KB" => 1_000.0,
                _ => 1.0,
            };
            spec.size_bytes = Some((number * factor) as u64);
        }
        models.push(spec);
    }
    models
}

/// How worker.py answers /health.
fn is_image_worker(health: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(health).is_ok_and(|body| {
        body.get("resident").is_some_and(serde_json::Value::is_boolean) && body["model"].as_str().is_some_and(|model| model.starts_with("flux"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_image_worker_counts_as_the_image_worker() {
        assert!(is_image_worker(r#"{"status":"ready","model":"flux2-klein-4b","resident":false}"#));
        for other in ["", "ok", "<html></html>", r#"{"status":"ok"}"#, r#"{"status":"ok","service":"story-maker"}"#, r#"{"model":"llama3","resident":true}"#] {
            assert!(!is_image_worker(other), "{other}");
        }
    }

    #[test]
    fn tags_are_exact_and_default_latest_is_equivalent() {
        assert!(same_model("all-minilm", "all-minilm:latest"));
        assert!(!same_model("qwen2.5:3b", "qwen2.5:32b"));
        assert!(!same_model("qwen2.5", "qwen2.5:3b"));
    }

    #[test]
    fn inventory_is_unverified_and_routes_specialists() {
        let models = parse_inventory(include_str!("../data/list.txt"));
        assert_eq!(models.len(), 13);
        assert!(models
            .iter()
            .all(|model| !model.installed && !model.is_resident));
        let embeddings = models.iter().find(|m| m.id == "all-minilm:latest").unwrap();
        assert_eq!(embeddings.capabilities, ["embedding"]);
        assert!(models
            .iter()
            .find(|m| m.id.contains("WeVisOne"))
            .unwrap()
            .capabilities
            .iter()
            .any(|c| c == "vision"));
        assert!(
            models
                .iter()
                .find(|m| m.id.contains("flux"))
                .unwrap()
                .is_image_generator
        );
        assert_eq!(
            models.iter().find(|m| m.id == "deepseek-r1:1.5b").unwrap().thinking,
            "always"
        );
    }

    fn drain(mut rx: mpsc::UnboundedReceiver<StreamEvent>) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        events
    }

    #[test]
    fn emitter_splits_think_tags_across_chunk_boundaries() {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut emitter = Emitter::new(Some(tx));
        for chunk in ["<th", "ink>plan ", "it</thi", "nk>\n\nThe ", "answer <b>", "</b>"] {
            emitter.content(chunk);
        }
        emitter.finish();
        assert_eq!(emitter.reasoning, "plan it");
        assert_eq!(emitter.content, "The answer <b></b>");
        let events = drain(rx);
        assert!(events.contains(&StreamEvent::Reasoning("plan ".into())));
        assert!(!events
            .iter()
            .any(|e| matches!(e, StreamEvent::Content(c) if c.contains("think"))));
    }

    #[test]
    fn emitter_reports_a_closed_listener_and_reasoning_only_replies_fail() {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        let mut emitter = Emitter::new(Some(tx));
        emitter.content("partial");
        assert!(emitter.closed);
        assert!(finalize(&emitter, ChatStats::default()).unwrap().cancelled);

        let mut thinking_only = Emitter::new(None);
        thinking_only.reasoning("still thinking");
        let error = finalize(&thinking_only, ChatStats::default()).unwrap_err();
        assert!(error.contains("budget"), "{error}");
    }

    #[test]
    fn effort_changes_budget_and_keep_alive_parses_numbers_and_durations() {
        assert!(Effort::Low.max_tokens() < Effort::Medium.max_tokens());
        assert!(Effort::Medium.max_tokens() < Effort::High.max_tokens());
        assert_eq!(serde_json::from_str::<Effort>("\"high\"").unwrap(), Effort::High);
        assert_eq!(parse_keep_alive(Some("0".into())), serde_json::json!(0));
        assert_eq!(parse_keep_alive(Some("30m".into())), serde_json::json!("30m"));
        assert_eq!(parse_keep_alive(None), serde_json::json!("10m"));
    }

    #[derive(Default)]
    struct MockState {
        resident: Vec<String>,
        calls: Vec<(String, serde_json::Value)>,
        active: usize,
        peak: usize,
        reject_unload: bool,
        reject_chat: bool,
        thinking: bool,
    }

    async fn mock_engine(
        state: std::sync::Arc<std::sync::Mutex<MockState>>,
        keep_alive: serde_json::Value,
    ) -> (std::sync::Arc<LlmEngine>, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let state = state.clone();
                tokio::spawn(async move {
                    let mut bytes = Vec::new();
                    let header_end;
                    loop {
                        let mut chunk = [0u8; 2048];
                        let count = stream.read(&mut chunk).await.unwrap();
                        if count == 0 {
                            return;
                        }
                        bytes.extend_from_slice(&chunk[..count]);
                        if let Some(position) =
                            bytes.windows(4).position(|window| window == b"\r\n\r\n")
                        {
                            header_end = position + 4;
                            break;
                        }
                    }
                    let headers = String::from_utf8_lossy(&bytes[..header_end]).to_string();
                    let path = headers
                        .lines()
                        .next()
                        .unwrap()
                        .split_whitespace()
                        .nth(1)
                        .unwrap()
                        .to_string();
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            line.to_lowercase()
                                .strip_prefix("content-length:")
                                .map(|value| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    while bytes.len() < header_end + length {
                        let mut chunk = [0u8; 2048];
                        let count = stream.read(&mut chunk).await.unwrap();
                        if count == 0 {
                            break;
                        }
                        bytes.extend_from_slice(&chunk[..count]);
                    }
                    let body: serde_json::Value =
                        serde_json::from_slice(&bytes[header_end..]).unwrap_or_default();
                    let (status, response, delay) = {
                        let mut state = state.lock().unwrap();
                        state.calls.push((path.clone(), body.clone()));
                        match path.as_str() {
                            "/api/ps" => (
                                200,
                                serde_json::json!({"models": state.resident.iter().map(|name| serde_json::json!({"name": name})).collect::<Vec<_>>()}).to_string(),
                                false,
                            ),
                            "/api/show" => (
                                200,
                                if state.thinking {
                                    serde_json::json!({"capabilities": ["completion", "thinking"]})
                                } else {
                                    serde_json::json!({"capabilities": ["completion"]})
                                }
                                .to_string(),
                                false,
                            ),
                            "/api/generate" if state.reject_unload => {
                                (500, serde_json::json!({"error": "cannot unload"}).to_string(), false)
                            }
                            "/api/generate" => {
                                let name = body["model"].as_str().unwrap();
                                state.resident.retain(|resident| resident != name);
                                (200, serde_json::json!({"done": true}).to_string(), false)
                            }
                            "/api/chat" if state.reject_chat => {
                                state
                                    .resident
                                    .push(body["model"].as_str().unwrap().to_string());
                                (500, serde_json::json!({"error": "inference failed"}).to_string(), false)
                            }
                            "/api/chat" | "/api/embed" => {
                                state.active += 1;
                                state.peak = state.peak.max(state.active);
                                let response = if path == "/api/chat" {
                                    let name = body["model"].as_str().unwrap().to_string();
                                    if !state.resident.contains(&name) {
                                        state.resident.push(name);
                                    }
                                    [
                                        serde_json::json!({"message": {"role": "assistant", "content": "", "thinking": "hmm"}}),
                                        serde_json::json!({"message": {"role": "assistant", "content": "ans"}}),
                                        serde_json::json!({"message": {"role": "assistant", "content": "wer"}, "done": true, "eval_count": 10, "eval_duration": 500_000_000u64}),
                                    ]
                                    .iter()
                                    .map(|line| format!("{line}\n"))
                                    .collect::<String>()
                                } else {
                                    serde_json::json!({"embeddings": [[0.1, 0.2]]}).to_string()
                                };
                                (200, response, true)
                            }
                            _ => (404, serde_json::json!({"error": "missing route"}).to_string(), false),
                        }
                    };
                    if delay {
                        tokio::time::sleep(Duration::from_millis(25)).await;
                        state.lock().unwrap().active -= 1;
                    }
                    let response = format!("HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len());
                    stream.write_all(response.as_bytes()).await.unwrap();
                });
            }
        });
        (
            std::sync::Arc::new(LlmEngine {
                client: Client::new(),
                base_url: format!("http://{address}"),
                worker_url: None,
                inference: std::sync::Arc::new(Mutex::new(())),
                backend: "ollama".to_string(),
                llamacpp: None,
                embedder: None,
                ollama_keep_alive: keep_alive,
                num_ctx: 8192,
            }),
            server,
        )
    }

    fn hello() -> [ChatMessagePayload; 1] {
        [ChatMessagePayload {
            role: "user".to_string(),
            content: "hello".to_string(),
            images: None,
        }]
    }

    #[tokio::test]
    async fn concurrent_chat_and_embedding_share_one_gpu_queue() {
        let state = std::sync::Arc::new(std::sync::Mutex::new(MockState::default()));
        let (engine, server) = mock_engine(state.clone(), serde_json::json!(0)).await;
        let messages = hello();
        let (one, two, three) = tokio::join!(
            engine.chat_completion("a:latest", &messages, ChatOptions::default()),
            engine.chat_completion("b:latest", &messages, ChatOptions::default()),
            engine.generate_embedding_checked("memory"),
        );
        assert_eq!(one.unwrap().0, "answer");
        assert!(two.is_ok(), "{two:?}");
        assert!(three.is_ok(), "{three:?}");
        let state = state.lock().unwrap();
        assert_eq!(state.peak, 1);
        assert!(state.resident.is_empty(), "keep_alive 0 unloads after each turn");
        let inference_calls: Vec<_> = state
            .calls
            .iter()
            .filter(|(path, _)| path == "/api/chat" || path == "/api/embed")
            .collect();
        assert_eq!(inference_calls.len(), 3);
        assert!(inference_calls
            .iter()
            .all(|(_, body)| body["keep_alive"] == 0));
        assert!(inference_calls
            .iter()
            .all(|(_, body)| body.get("think").is_none()));
        server.abort();
    }

    #[tokio::test]
    async fn warm_model_stays_resident_streams_and_forwards_thinking_and_effort() {
        let state = std::sync::Arc::new(std::sync::Mutex::new(MockState {
            thinking: true,
            ..MockState::default()
        }));
        let (engine, server) = mock_engine(state.clone(), serde_json::json!("10m")).await;
        let (tx, rx) = mpsc::unbounded_channel();
        let outcome = engine
            .chat_stream(
                "think:latest",
                &hello(),
                ChatOptions {
                    thinking: true,
                    effort: Effort::High,
                    ..Default::default()
                },
                Some(tx),
            )
            .await
            .unwrap();
        assert_eq!(outcome.content, "answer");
        assert_eq!(outcome.reasoning, "hmm");
        assert_eq!(outcome.stats.tokens_per_second, Some(20.0));
        let events = drain(rx);
        assert!(events.contains(&StreamEvent::Content("ans".into())));
        assert!(events.contains(&StreamEvent::Reasoning("hmm".into())));
        let state = state.lock().unwrap();
        assert_eq!(state.resident, ["think:latest"]);
        let (_, body) = state.calls.iter().find(|(p, _)| p == "/api/chat").unwrap();
        assert_eq!(body["think"], true);
        assert_eq!(body["stream"], true);
        assert_eq!(body["options"]["num_predict"], 24_576, "thinking gets the larger budget");
        assert!(ChatOptions { thinking: false, effort: Effort::High, ..Default::default() }.max_tokens(false) < 24_576);
        server.abort();
    }

    #[tokio::test]
    async fn unload_failure_prevents_loading_the_next_model() {
        let state = std::sync::Arc::new(std::sync::Mutex::new(MockState {
            resident: vec!["old:latest".to_string()],
            reject_unload: true,
            ..MockState::default()
        }));
        let (engine, server) = mock_engine(state.clone(), serde_json::json!("10m")).await;
        let error = engine
            .chat_completion("next:latest", &[], ChatOptions::default())
            .await
            .unwrap_err();
        assert!(error.contains("Failed to unload"), "{error}");
        assert!(!state
            .lock()
            .unwrap()
            .calls
            .iter()
            .any(|(path, _)| path == "/api/chat"));
        server.abort();
    }

    #[tokio::test]
    async fn failed_inference_unloads_any_model_left_resident() {
        let state = std::sync::Arc::new(std::sync::Mutex::new(MockState {
            reject_chat: true,
            ..MockState::default()
        }));
        let (engine, server) = mock_engine(state.clone(), serde_json::json!("10m")).await;
        let error = engine
            .chat_completion("failed:latest", &[], ChatOptions::default())
            .await
            .unwrap_err();
        assert!(error.contains("inference failed"), "{error}");
        let state = state.lock().unwrap();
        assert!(state.resident.is_empty());
        assert!(state
            .calls
            .iter()
            .any(|(path, body)| path == "/api/generate" && body["model"] == "failed:latest"));
        server.abort();
    }
}
