use base64::Engine;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::process::Command;

pub struct GenerationResult {
    pub width: u32,
    pub height: u32,
    pub backend: String,
    pub duration_ms: u64,
    pub seed: u64,
}

/// The caller holds LlmEngine::acquire() and unloads models before and after this
/// operation. Image generation must use the same queue as text and embeddings.
pub async fn generate(
    prompt: &str,
    aspect_ratio: &str,
    image_model: Option<&str>,
    reference: Option<&Path>,
    output_path: &Path,
) -> Result<GenerationResult, String> {
    if prompt.trim().is_empty() {
        return Err("Enter a prompt to generate an image".to_string());
    }
    if !["1:1", "16:9", "9:16"].contains(&aspect_ratio) {
        return Err("Choose an aspect ratio of 1:1, 16:9, or 9:16".to_string());
    }
    let start = Instant::now();
    // Preserve the requested aspect ratio with dimensions divisible by 16.
    let (width, height) = match aspect_ratio {
        "16:9" => (768, 432),
        "9:16" => (432, 768),
        _ => (512, 512),
    };
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
        % 1_000_000_000;
    let backend = backend_name();
    if reference.is_some() && backend != "sdcpp" {
        return Err("Editing images needs stable-diffusion.cpp; run scripts/setup.sh.".to_string());
    }
    let (actual_width, actual_height, actual_seed, label) = match backend.as_str() {
        "sdcpp" => {
            let model = match image_model {
                Some(id) => crate::sdcpp::ImageModel::parse(id)
                    .ok_or_else(|| format!("Unknown image model '{id}'"))?,
                None => crate::sdcpp::ImageModel::ALL
                    .into_iter()
                    .find(|m| crate::sdcpp::ready(*m))
                    .unwrap_or(crate::sdcpp::ImageModel::FluxKlein),
            };
            // Only FLUX.2 Klein follows edit instructions.
            let model = if reference.is_some() { crate::sdcpp::ImageModel::FluxKlein } else { model };
            let (w, h) = match reference {
                Some(path) => crate::sdcpp::edit_size(path)?,
                None => model.size(aspect_ratio).unwrap_or((width, height)),
            };
            crate::sdcpp::generate(model, prompt, w, h, seed, reference, output_path).await?;
            let bytes = tokio::fs::read(output_path)
                .await
                .map_err(|e| format!("stable-diffusion.cpp did not create the image: {e}"))?;
            let dimensions = save_validated_image(&bytes, output_path)?;
            (
                dimensions.0,
                dimensions.1,
                seed,
                format!(
                    "stable-diffusion.cpp · {}{}",
                    model.label(),
                    if reference.is_some() { " · edit" } else { "" }
                ),
            )
        }
        "ollama" => {
            let model =
                std::env::var("IMAGE_MODEL").unwrap_or_else(|_| "x/flux2-klein:4b".to_string());
            let dimensions =
                generate_with_ollama(&model, prompt, width, height, seed, output_path).await?;
            (
                dimensions.0,
                dimensions.1,
                seed,
                format!("Ollama · {model}"),
            )
        }
        "mlx" => {
            let result = generate_with_mlx(prompt, width, height, seed, output_path).await?;
            (result.0, result.1, result.2, result.3)
        }
        _ => {
            return Err(format!(
                "Unknown IMAGE_BACKEND '{backend}'; choose sdcpp, mlx or ollama"
            ))
        }
    };
    Ok(GenerationResult {
        width: actual_width,
        height: actual_height,
        backend: label,
        duration_ms: start.elapsed().as_millis() as u64,
        seed: actual_seed,
    })
}

/// IMAGE_BACKEND, or by default stable-diffusion.cpp when sd-cli is present,
/// then mflux, then Ollama (recent Ollama releases removed image generation).
pub fn backend_name() -> String {
    match std::env::var("IMAGE_BACKEND") {
        Ok(value) if !value.trim().is_empty() && value != "auto" => value.trim().to_lowercase(),
        _ if crate::sdcpp::binary().is_some() => "sdcpp".to_string(),
        _ if mflux_binary().is_some() => "mlx".to_string(),
        _ => "ollama".to_string(),
    }
}

/// MFLUX_BIN, or mflux-generate-flux2 on PATH or in ~/.local/bin (uv tools).
fn mflux_binary() -> Option<std::path::PathBuf> {
    if let Some(path) = std::env::var_os("MFLUX_BIN") {
        return Some(path.into());
    }
    let name = "mflux-generate-flux2";
    std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .chain(std::env::var_os("HOME").map(|home| Path::new(&home).join(".local/bin")))
        .map(|dir| dir.join(name))
        .find(|path| path.is_file())
}

fn http_client() -> Result<Client, String> {
    Client::builder()
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(|e| format!("Could not initialize image client: {e}"))
}

async fn generate_with_ollama(
    model: &str,
    prompt: &str,
    width: u32,
    height: u32,
    seed: u64,
    output_path: &Path,
) -> Result<(u32, u32), String> {
    let base_url =
        std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
    let response = http_client()?
        .post(format!("{}/api/generate", base_url.trim_end_matches('/')))
        .json(&json!({
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": 4,
            "stream": true,
            "keep_alive": 0,
            "options": { "seed": seed }
        }))
        .send()
        .await
        .map_err(|e| format!("Ollama image generation is unavailable: {e}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| format!("Could not read Ollama image response: {e}"))?;
    if !status.is_success() {
        let message = error_message(&body);
        if message.contains("not currently supported") {
            return Err("This Ollama version no longer generates images. Install mflux (uv tool install --python 3.12 mflux) and restart Ferro64; it is used automatically.".to_string());
        }
        return Err(format!("Ollama image generation ({status}): {message}"));
    }
    let bytes = parse_image_response(&body)?;
    save_validated_image(&bytes, output_path)
}

fn error_message(body: &str) -> String {
    let message = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("error").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_else(|| body.to_string());
    message.chars().take(500).collect()
}

/// Ollama's experimental image API sends NDJSON progress followed by a base64
/// `image` field. Also accept a single JSON response for compatible versions.
fn parse_image_response(body: &str) -> Result<Vec<u8>, String> {
    let mut image = None;
    let mut done = false;
    let parts: Vec<Value> = if let Ok(single) = serde_json::from_str::<Value>(body) {
        vec![single]
    } else {
        body.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                serde_json::from_str::<Value>(line)
                    .map_err(|_| "Ollama returned invalid image-generation JSON".to_string())
            })
            .collect::<Result<_, _>>()?
    };
    for part in parts {
        if let Some(error) = part.get("error").and_then(Value::as_str) {
            return Err(format!(
                "Ollama image generation: {}",
                error.chars().take(500).collect::<String>()
            ));
        }
        if let Some(encoded) = part.get("image").and_then(Value::as_str) {
            if !encoded.trim().is_empty() {
                image = Some(
                    base64::engine::general_purpose::STANDARD
                        .decode(encoded)
                        .map_err(|_| "Ollama returned invalid base64 image data".to_string())?,
                );
            }
        }
        done |= part.get("done").and_then(Value::as_bool).unwrap_or(false);
    }
    // Some experimental versions emit the image as the last event without a
    // separate `done` event. A valid image is the completion artifact.
    match image {
        Some(bytes) if !bytes.is_empty() => Ok(bytes),
        _ => Err(format!(
            "Ollama returned no generated image{}. Image generation is experimental and requires a compatible macOS Ollama runtime and an installed image model.",
            if done { " (request completed)" } else { "" }
        )),
    }
}

fn save_validated_image(bytes: &[u8], output_path: &Path) -> Result<(u32, u32), String> {
    let image = image::load_from_memory(bytes)
        .map_err(|e| format!("The backend returned invalid image bytes: {e}"))?;
    let dimensions = (image.width(), image.height());
    image
        .save_with_format(output_path, image::ImageFormat::Png)
        .map_err(|e| format!("Could not save the generated PNG: {e}"))?;
    Ok(dimensions)
}

#[derive(Deserialize)]
struct WorkerResponse {
    seed: Option<u64>,
    error: Option<String>,
}

async fn generate_with_mlx(
    prompt: &str,
    width: u32,
    height: u32,
    seed: u64,
    output_path: &Path,
) -> Result<(u32, u32, u64, String), String> {
    let worker_url =
        std::env::var("IMAGE_WORKER_URL").unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());
    let worker = http_client()?
        .post(format!("{}/generate", worker_url.trim_end_matches('/')))
        .json(&json!({
            "prompt": prompt,
            "output": output_path.to_string_lossy(),
            "width": width,
            "height": height,
            "steps": 4,
            "seed": seed
        }))
        .send()
        .await;
    match worker {
        Ok(response) => {
            let status = response.status();
            let body = response.text().await.map_err(|e| e.to_string())?;
            if !status.is_success() {
                return Err(format!("MLX worker ({status}): {}", error_message(&body)));
            }
            let result: WorkerResponse = serde_json::from_str(&body)
                .map_err(|e| format!("Invalid MLX worker response: {e}"))?;
            if let Some(error) = result.error {
                return Err(format!("MLX worker: {error}"));
            }
            let bytes = tokio::fs::read(output_path)
                .await
                .map_err(|e| format!("MLX worker did not create the output image: {e}"))?;
            let dimensions = save_validated_image(&bytes, output_path)?;
            Ok((
                dimensions.0,
                dimensions.1,
                result.seed.unwrap_or(seed),
                "MLX · FLUX.2 Klein worker".to_string(),
            ))
        }
        Err(error) if error.is_connect() => {
            generate_with_mflux_cli(prompt, width, height, seed, output_path).await
        }
        Err(error) => Err(format!("MLX worker generation failed: {error}")),
    }
}

async fn generate_with_mflux_cli(
    prompt: &str,
    width: u32,
    height: u32,
    seed: u64,
    output_path: &Path,
) -> Result<(u32, u32, u64, String), String> {
    // Only run an already installed executable. No automatic package/model
    // manager invocation, and no user-specific executable path.
    let executable = mflux_binary()
        .ok_or_else(|| "mflux is not installed. Run: uv tool install --python 3.12 mflux".to_string())?;
    let executable = executable.display().to_string();
    let mut command = Command::new(&executable);
    command
        .kill_on_drop(true)
        .arg("--prompt")
        .arg(prompt)
        .arg("--model")
        .arg("flux2-klein-4b")
        .arg("-q")
        .arg("4")
        .arg("--steps")
        .arg("4")
        .arg("--width")
        .arg(width.to_string())
        .arg("--height")
        .arg(height.to_string())
        .arg("--seed")
        .arg(seed.to_string())
        .arg("--output")
        .arg(output_path);
    let output = tokio::time::timeout(Duration::from_secs(600), command.output())
        .await
        .map_err(|_| "MLX CLI generation timed out after 10 minutes".to_string())?
        .map_err(|e| format!("MLX worker is offline and '{executable}' could not start: {e}. Configure an installed MFLUX_BIN or use IMAGE_BACKEND=ollama."))?;
    if !output.status.success() {
        return Err(format!(
            "mflux generation failed: {}",
            String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(500)
                .collect::<String>()
        ));
    }
    let bytes = tokio::fs::read(output_path)
        .await
        .map_err(|e| format!("mflux did not create the output image: {e}"))?;
    let dimensions = save_validated_image(&bytes, output_path)?;
    Ok((
        dimensions.0,
        dimensions.1,
        seed,
        "MLX · FLUX.2 Klein CLI".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_stream_ignores_progress_and_decodes_final_image() {
        let response = "{\"completed\":1,\"total\":4}\n{\"image\":\"AQID\"}\n{\"done\":true}\n";
        assert_eq!(parse_image_response(response).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn image_stream_accepts_single_json_and_rejects_errors() {
        assert_eq!(
            parse_image_response("{\"image\":\"AQID\",\"done\":true}").unwrap(),
            vec![1, 2, 3]
        );
        assert!(parse_image_response("{\"error\":\"unsupported model\"}")
            .unwrap_err()
            .contains("unsupported model"));
        assert!(parse_image_response("{\"image\":\"?\"}").is_err());
        assert!(parse_image_response("{\"done\":true}")
            .unwrap_err()
            .contains("no generated image"));
        assert!(parse_image_response("not JSON").is_err());
    }

    #[test]
    fn image_stream_does_not_hide_an_error_after_image() {
        let response = "{\"image\":\"AQID\"}\n{\"error\":\"runner failed\"}";
        assert!(parse_image_response(response)
            .unwrap_err()
            .contains("runner failed"));
    }

    #[test]
    fn rejects_non_image_bytes_instead_of_creating_a_placeholder() {
        let output =
            std::env::temp_dir().join(format!("ferro64-invalid-{}.png", uuid::Uuid::new_v4()));
        assert!(save_validated_image(b"not an image", &output).is_err());
        assert!(!output.exists());
    }
}
