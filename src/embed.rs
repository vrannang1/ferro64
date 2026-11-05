//! Text embeddings with Qwen3-Embedding-0.6B for semantic search: related
//! code in Code projects, and earlier questions for chat recall. A small
//! llama-server in embedding mode starts on first use and stops when idle; it
//! runs beside the chat model (about 1 GB) rather than taking turns with it.
use reqwest::Client;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

pub const MODEL_FILE: &str = "Qwen3-Embedding-0.6B-Q8_0.gguf";
pub const DIMS: usize = 1024;
/// Queries carry a task description; stored documents do not.
pub const CODE_TASK: &str = "Given a question or description about a codebase, retrieve the code that answers it";
pub const RECALL_TASK: &str = "Given a user's new message, retrieve their earlier messages about the same topic";
/// Characters of one input; longer text is cut (the window is 8k tokens).
const INPUT_CHARS: usize = 6_000;
const BATCH: usize = 8;
const IDLE: Duration = Duration::from_secs(600);

struct Running {
    child: Child,
    base: String,
    last_used: Instant,
}

pub struct Embedder {
    binary: PathBuf,
    model: PathBuf,
    log: PathBuf,
    client: Client,
    process: Mutex<Option<Running>>,
}

pub fn query(task: &str, text: &str) -> String {
    format!("Instruct: {task}\nQuery: {text}")
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

pub fn to_blob(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn from_blob(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

impl Embedder {
    pub fn new(binary: PathBuf, base_dir: &Path) -> Self {
        Self {
            binary,
            model: base_dir.join("data/models").join(MODEL_FILE),
            log: base_dir.join("data/embed-server.log"),
            client: Client::builder().connect_timeout(Duration::from_secs(3)).timeout(Duration::from_secs(300)).build().expect("HTTP client"),
            process: Mutex::new(None),
        }
    }

    /// The model file is on disk (downloaded in Settings → Models).
    pub fn installed(&self) -> bool {
        self.model.is_file()
    }

    async fn base(&self) -> Result<String, String> {
        let mut process = self.process.lock().await;
        if let Some(running) = process.as_mut() {
            if matches!(running.child.try_wait(), Ok(None)) {
                running.last_used = Instant::now();
                return Ok(running.base.clone());
            }
        }
        *process = None;
        if !self.installed() {
            return Err("Download Qwen3 Embedding 0.6B in Settings → Models to search by meaning".to_string());
        }
        let port = std::net::TcpListener::bind(("127.0.0.1", 0))
            .and_then(|l| l.local_addr())
            .map(|a| a.port())
            .map_err(|e| format!("No free port for the embedding server: {e}"))?;
        let log = std::fs::File::create(&self.log).map(Stdio::from).unwrap_or_else(|_| Stdio::null());
        let child = Command::new(&self.binary)
            .arg("--model")
            .arg(&self.model)
            .args(["--embedding", "--pooling", "last", "-c", "8192", "-b", "8192", "-ub", "8192", "-ngl", "99"])
            .args(["--host", "127.0.0.1", "--port", &port.to_string(), "--parallel", "1", "--no-webui", "--offline"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(log)
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("Could not start the embedding server ({}): {e}", self.binary.display()))?;
        let base = format!("http://127.0.0.1:{port}");
        let mut running = Running { child, base: base.clone(), last_used: Instant::now() };
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if let Ok(Some(status)) = running.child.try_wait() {
                return Err(format!("The embedding server exited ({status}); see {}", self.log.display()));
            }
            if let Ok(response) = self.client.get(format!("{base}/health")).timeout(Duration::from_secs(2)).send().await {
                if response.status().is_success() {
                    break;
                }
            }
            if Instant::now() > deadline {
                return Err("The embedding server did not start within a minute".to_string());
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
        *process = Some(running);
        Ok(base)
    }

    /// One vector per input, in order.
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let base = self.base().await?;
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(BATCH) {
            let input: Vec<String> = batch.iter().map(|t| t.chars().take(INPUT_CHARS).collect()).collect();
            let response = self
                .client
                .post(format!("{base}/v1/embeddings"))
                .json(&serde_json::json!({"input": input}))
                .send()
                .await
                .map_err(|e| format!("Embedding request failed: {e}"))?;
            if !response.status().is_success() {
                let status = response.status();
                return Err(format!("Embedding server {status}: {}", response.text().await.unwrap_or_default()));
            }
            let data: serde_json::Value = response.json().await.map_err(|e| format!("Invalid embedding response: {e}"))?;
            let mut items: Vec<(usize, Vec<f32>)> = data["data"]
                .as_array()
                .ok_or("Invalid embedding response")?
                .iter()
                .map(|item| {
                    let index = item["index"].as_u64().unwrap_or(0) as usize;
                    let vector = item["embedding"].as_array().map(|v| v.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect()).unwrap_or_default();
                    (index, vector)
                })
                .collect();
            items.sort_by_key(|(i, _)| *i);
            if items.len() != batch.len() || items.iter().any(|(_, v)| v.len() != DIMS) {
                return Err("The embedding server returned an unexpected result".to_string());
            }
            vectors.extend(items.into_iter().map(|(_, v)| v));
        }
        if let Some(running) = self.process.lock().await.as_mut() {
            running.last_used = Instant::now();
        }
        Ok(vectors)
    }

    pub async fn embed_one(&self, text: String) -> Result<Vec<f32>, String> {
        self.embed(&[text]).await?.pop().ok_or_else(|| "No embedding returned".to_string())
    }

    /// Stops the server after it has been idle; never while it is starting.
    pub async fn reap_idle(&self) -> bool {
        let Ok(mut process) = self.process.try_lock() else { return false };
        if process.as_ref().is_some_and(|r| r.last_used.elapsed() > IDLE) {
            *process = None; // kill_on_drop
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_round_trip_and_compare() {
        let v = vec![0.25f32, -1.5, 3.0];
        assert_eq!(from_blob(&to_blob(&v)), v);
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert!(query(CODE_TASK, "where is the cart total").starts_with("Instruct: "));
    }
}
