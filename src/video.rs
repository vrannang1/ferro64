//! Video generation with LTX-2.3 through ltx-video (a Swift MLX build in
//! data/bin/ltx). A video takes minutes, so it runs as a job: one at a time,
//! holding the shared GPU queue with text models unloaded, reporting
//! progress from the program's output.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Instant;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

pub const FPS: u32 = 24;
const LOG_LINES: usize = 20;

pub fn binary() -> Option<PathBuf> {
    let path = crate::sdcpp::base_dir().join("data/bin/ltx/ltx-video");
    path.is_file().then_some(path)
}

pub fn models_dir() -> PathBuf {
    crate::sdcpp::base_dir().join("data/models/video")
}

/// Width and height for a shape; both divisible by 64 for the two-stage
/// pipeline.
pub fn size(aspect: &str) -> Option<(u32, u32)> {
    match aspect {
        "16:9" => Some((768, 448)),
        "9:16" => Some((448, 768)),
        "1:1" => Some((512, 512)),
        _ => None,
    }
}

/// Frames for a length in seconds: LTX takes 8n+1 frames.
pub fn frames(seconds: f32) -> u32 {
    let n = ((seconds.clamp(1.0, 10.0) * FPS as f32) / 8.0).round().max(1.0) as u32;
    8 * n + 1
}

#[derive(Deserialize)]
pub struct VideoRequest {
    pub prompt: String,
    #[serde(default = "default_aspect")]
    pub aspect: String,
    #[serde(default = "default_seconds")]
    pub seconds: f32,
    /// Generate sound with the picture.
    #[serde(default)]
    pub audio: bool,
    /// Let Gemma rewrite the prompt in LTX's detailed style first (a few
    /// minutes more; it also looks at the starting picture).
    #[serde(default)]
    pub enhance: bool,
    /// Start from a gallery image (its history ID)…
    pub image_of: Option<String>,
    /// …or an uploaded photo (a data URL).
    pub image_data: Option<String>,
    pub seed: Option<u64>,
}

fn default_aspect() -> String {
    "16:9".to_string()
}
fn default_seconds() -> f32 {
    4.0
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Job {
    pub id: String,
    /// queued, running, completed or failed
    pub status: String,
    pub phase: String,
    pub percent: Option<u32>,
    pub log: Vec<String>,
    pub error: Option<String>,
    pub video_url: Option<String>,
    pub enhanced_prompt: Option<String>,
    pub started_ms: u64,
    pub duration_ms: Option<u64>,
}

#[derive(Default)]
pub struct Jobs(Mutex<HashMap<String, Job>>);

impl Jobs {
    pub fn get(&self, id: &str) -> Option<Job> {
        self.0.lock().unwrap().get(id).cloned()
    }
    pub fn active(&self) -> Option<Job> {
        self.0.lock().unwrap().values().find(|j| j.status == "queued" || j.status == "running").cloned()
    }
    pub fn update(&self, id: &str, change: impl FnOnce(&mut Job)) {
        if let Some(job) = self.0.lock().unwrap().get_mut(id) {
            change(job);
        }
    }
    pub fn insert(&self, job: Job) {
        let mut jobs = self.0.lock().unwrap();
        // Keep recent finished jobs only.
        if jobs.len() > 20 {
            jobs.retain(|_, j| j.status == "queued" || j.status == "running");
        }
        jobs.insert(job.id.clone(), job);
    }
}

/// Overall progress for a line, on one bar: loading reports its own 0–100%
/// (shown as 0–10%), the denoising steps fill 10–95%, then decoding and
/// export. Lines without progress return None.
pub fn overall(line: &str) -> Option<u32> {
    let lower = line.to_lowercase();
    if lower.contains("[decoding]") {
        return Some(96);
    }
    if lower.contains("[exporting]") {
        return Some(99);
    }
    let value = percent(line)?;
    Some(if lower.contains("step ") { 10 + value * 85 / 100 } else { value / 10 })
}

/// Collects the rewritten prompt, which ltx-video prints after an
/// "Enhanced prompt" line and before it loads the models.
#[derive(Default)]
pub struct EnhancedPrompt {
    capturing: bool,
    pub text: String,
}

impl EnhancedPrompt {
    /// Returns true when the line belonged to the rewritten prompt.
    pub fn feed(&mut self, line: &str) -> bool {
        if line.starts_with("Enhanced prompt") {
            self.capturing = true;
            return true;
        }
        if self.capturing {
            if line.starts_with("Loading models") || line.starts_with("Unloading") || line.starts_with("Models loaded") {
                self.capturing = false;
                return false;
            }
            if !self.text.is_empty() {
                self.text.push('\n');
            }
            self.text.push_str(line);
            return true;
        }
        false
    }
}

/// A percentage in a progress line: "(42%)", "42%", or "step 3/8".
pub fn percent(line: &str) -> Option<u32> {
    if let Some(at) = line.find('%') {
        let digits: String = line[..at].chars().rev().take_while(|c| c.is_ascii_digit() || *c == '.').collect::<Vec<_>>().into_iter().rev().collect();
        if let Ok(value) = digits.parse::<f32>() {
            return Some(value.clamp(0.0, 100.0) as u32);
        }
    }
    let lower = line.to_lowercase();
    let at = lower.find("step ")?;
    let rest = &lower[at + 5..];
    let (done, total) = rest.split_once('/')?;
    let done: u32 = done.trim().parse().ok()?;
    let total: u32 = total.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()?;
    (total > 0).then(|| (done * 100 / total).min(100))
}

/// Runs ltx-video and streams its output lines to `progress`.
pub async fn render(
    prompt: &str,
    width: u32,
    height: u32,
    frames: u32,
    seed: u64,
    audio: bool,
    enhance: bool,
    image: Option<&Path>,
    output: &Path,
    mut progress: impl FnMut(&str),
) -> Result<u64, String> {
    let binary = binary().ok_or("The video engine is not built. Run scripts/setup-video.sh (it needs Xcode).")?;
    std::fs::create_dir_all(models_dir()).map_err(|e| e.to_string())?;
    let mut command = Command::new(&binary);
    command
        .arg("generate")
        .arg(prompt)
        .args(["-w", &width.to_string(), "-h", &height.to_string(), "-f", &frames.to_string()])
        .args(["--seed", &seed.to_string(), "--transformer-quant", "qint8"])
        .arg("--models-dir")
        .arg(models_dir())
        .arg("-o")
        .arg(output)
        // The metallib bundle sits next to the binary; run from there.
        .current_dir(binary.parent().unwrap_or(Path::new(".")))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(image) = image {
        command.arg("--image").arg(image);
    }
    if audio {
        command.arg("--audio");
    }
    if enhance {
        command.arg("--enhance-prompt");
    }
    let started = Instant::now();
    let mut child = command.spawn().map_err(|e| format!("Could not start {}: {e}", binary.display()))?;
    let mut stdout = child.stdout.take().expect("piped");
    let mut stderr = child.stderr.take().expect("piped");
    let mut tail: Vec<String> = Vec::new();
    let (mut out_buf, mut err_buf) = ([0u8; 4096], [0u8; 4096]);
    let mut pending = String::new();
    let (mut out_open, mut err_open) = (true, true);
    while out_open || err_open {
        let read = tokio::select! {
            n = stdout.read(&mut out_buf), if out_open => match n { Ok(0) | Err(_) => { out_open = false; None } Ok(n) => Some(String::from_utf8_lossy(&out_buf[..n]).to_string()) },
            n = stderr.read(&mut err_buf), if err_open => match n { Ok(0) | Err(_) => { err_open = false; None } Ok(n) => Some(String::from_utf8_lossy(&err_buf[..n]).to_string()) },
        };
        let Some(text) = read else { continue };
        pending.push_str(&text);
        // Progress bars redraw with \r; treat it as a line end.
        while let Some(at) = pending.find(['\n', '\r']) {
            let line: String = pending.drain(..=at).collect();
            let line = line.trim();
            if !line.is_empty() {
                progress(line);
                tail.push(line.to_string());
                if tail.len() > LOG_LINES {
                    tail.remove(0);
                }
            }
        }
    }
    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() || !output.is_file() {
        return Err(format!("Video generation failed ({status}). Last output:\n{}", tail.join("\n")));
    }
    Ok(started.elapsed().as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_become_8n_plus_1_frames() {
        assert_eq!(frames(4.0), 97);
        assert_eq!(frames(2.0), 49);
        assert_eq!(frames(10.0), 241);
        assert_eq!(frames(60.0), 241, "capped at 10 seconds");
        assert!(frames(0.1) >= 9);
        for (w, h) in ["16:9", "9:16", "1:1"].map(|a| size(a).unwrap()) {
            assert!(w % 64 == 0 && h % 64 == 0);
        }
        assert!(size("4:3").is_none());
    }

    #[test]
    fn the_rewritten_prompt_is_captured() {
        let mut enhanced = EnhancedPrompt::default();
        for line in ["Enhancing prompt...", "Enhanced prompt (Gemma 3 12B):", "A close-up of a hand holding a phone.", "Golden particles drift across it.", "Loading models (this may take a while)...", "Step 1/11"] {
            enhanced.feed(line);
        }
        assert_eq!(enhanced.text, "A close-up of a hand holding a phone.\nGolden particles drift across it.");
    }

    #[test]
    fn progress_lines_give_a_percentage() {
        assert_eq!(percent("  Resolving checkpoint... (10%)"), Some(10));
        assert_eq!(percent("Downloading 42.5%"), Some(42));
        assert_eq!(percent("Denoising step 3/8"), Some(37));
        assert_eq!(percent("Loading models"), None);
        assert_eq!(overall("Quantizing transformer to 8-bit (qint8)... (60%)"), Some(6));
        assert_eq!(overall("Step 1/11 [denoising] (σ=1.0000)"), Some(17));
        assert_eq!(overall("Step 11/11 [refinement] (σ=0.4219)"), Some(95));
        assert_eq!(overall("[decoding]"), Some(96));
    }
}
