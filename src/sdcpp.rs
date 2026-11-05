//! Image generation through stable-diffusion.cpp's `sd-cli`, a native binary
//! for macOS (Metal) and Linux (CPU, Vulkan, ROCm, CUDA). No Python involved.
//! Weights live in data/models/image/ and are downloaded from the Models view.
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageModel {
    /// FLUX.2 Klein 4B: best quality and prompt following; slower.
    FluxKlein,
    /// Z-Image-Turbo: very realistic photos in 8 steps.
    ZImageTurbo,
}

impl ImageModel {
    pub const ALL: [ImageModel; 2] = [ImageModel::ZImageTurbo, ImageModel::FluxKlein];

    pub fn id(self) -> &'static str {
        match self {
            ImageModel::FluxKlein => "flux2-klein",
            ImageModel::ZImageTurbo => "z-image-turbo",
        }
    }

    pub fn parse(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.id() == id)
    }

    pub fn label(self) -> &'static str {
        match self {
            ImageModel::FluxKlein => "FLUX.2 Klein 4B",
            ImageModel::ZImageTurbo => "Z-Image-Turbo",
        }
    }

    /// The catalog entry that downloads this model's weights.
    pub fn catalog_id(self) -> &'static str {
        match self {
            ImageModel::FluxKlein => "image:flux2-klein-4b",
            ImageModel::ZImageTurbo => "image:z-image-turbo",
        }
    }

    /// Both models need sides divisible by 16 and are trained near 1 megapixel.
    pub fn size(self, aspect: &str) -> Option<(u32, u32)> {
        Some(match (self, aspect) {
            (ImageModel::FluxKlein | ImageModel::ZImageTurbo, "1:1") => (1024, 1024),
            (ImageModel::FluxKlein | ImageModel::ZImageTurbo, "16:9") => (1344, 768),
            (ImageModel::FluxKlein | ImageModel::ZImageTurbo, "9:16") => (768, 1344),
            _ => return None,
        })
    }
}

pub fn base_dir() -> PathBuf {
    std::env::var("FERRO64_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

pub fn image_dir() -> PathBuf {
    base_dir().join("data/models/image")
}

/// SDCPP_BIN, the copy fetched by scripts/setup.sh, or sd-cli on PATH.
pub fn binary() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("SDCPP_BIN") {
        return Some(path.into());
    }
    std::iter::once(base_dir().join("data/bin/sdcpp"))
        .chain(std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()))
        .map(|dir| dir.join(if cfg!(windows) { "sd-cli.exe" } else { "sd-cli" }))
        .find(|path| path.is_file())
}

/// FLUX.2 Klein 4B and Z-Image-Turbo read prompts with Qwen3 4B, the same
/// model offered for chat.
fn text_encoder() -> Option<PathBuf> {
    let entry = crate::model_store::catalog_entry("qwen3:4b")?;
    let path = base_dir().join("data/models").join(entry.local_model());
    path.is_file().then_some(path)
}

struct Files {
    args: Vec<String>,
}

fn files(model: ImageModel) -> Result<Files, String> {
    let dir = image_dir();
    let need = |name: &str| -> Result<String, String> {
        let path = dir.join(name);
        if path.is_file() {
            Ok(path.display().to_string())
        } else {
            Err(format!(
                "{} is not downloaded yet. Get it in Models → download.",
                model.label()
            ))
        }
    };
    Ok(Files {
        args: match model {
            ImageModel::FluxKlein => vec![
                "--diffusion-model".into(),
                need("flux-2-klein-4b-Q8_0.gguf")?,
                "--vae".into(),
                need("flux2-vae.safetensors")?,
                "--llm".into(),
                text_encoder()
                    .ok_or("FLUX.2 Klein reads prompts with Qwen3 4B. Download Qwen3 4B in Models first.")?
                    .display()
                    .to_string(),
                "--cfg-scale".into(),
                "1.0".into(),
                "--steps".into(),
                "4".into(),
                "--sampling-method".into(),
                "euler".into(),
                "--diffusion-fa".into(),
            ],
            ImageModel::ZImageTurbo => vec![
                "--diffusion-model".into(),
                need("z_image_turbo-Q8_0.gguf")?,
                "--vae".into(),
                need("ae.safetensors")?,
                "--llm".into(),
                text_encoder()
                    .ok_or("Z-Image-Turbo reads prompts with Qwen3 4B. Download Qwen3 4B in Models first.")?
                    .display()
                    .to_string(),
                "--cfg-scale".into(),
                "1.0".into(),
                "--steps".into(),
                "8".into(),
                "--diffusion-fa".into(),
            ],
        },
    })
}

/// Output size for an edit: the reference's shape at about one megapixel at
/// most, with sides divisible by 16.
pub fn edit_size(reference: &Path) -> Result<(u32, u32), String> {
    let (w, h) = image::image_dimensions(reference)
        .map_err(|e| format!("Cannot read the image to edit: {e}"))?;
    Ok(fit_megapixel(w, h))
}

fn fit_megapixel(w: u32, h: u32) -> (u32, u32) {
    let scale = ((1024.0 * 1024.0) / (w as f64 * h as f64)).sqrt().min(1.0);
    let side = |v: u32| (((v as f64 * scale) / 16.0).round() as u32 * 16).max(256);
    (side(w), side(h))
}

/// Which image models have their weights (and, for FLUX, the text encoder).
pub fn ready(model: ImageModel) -> bool {
    files(model).is_ok()
}

pub async fn generate(
    model: ImageModel,
    prompt: &str,
    width: u32,
    height: u32,
    seed: u64,
    reference: Option<&Path>,
    output: &Path,
) -> Result<(), String> {
    let binary = binary().ok_or(
        "stable-diffusion.cpp is not installed. Run scripts/setup.sh, or set SDCPP_BIN to an sd-cli binary.",
    )?;
    let files = files(model)?;
    let mut command = Command::new(&binary);
    command
        .args(&files.args)
        .arg("-p")
        .arg(prompt)
        .args(["-W", &width.to_string(), "-H", &height.to_string()])
        .args(["-s", &seed.to_string()])
        .arg("-o")
        .arg(output);
    // FLUX.2 Klein edits: the reference image is kept and the prompt says what to change.
    if let Some(reference) = reference {
        command.arg("-r").arg(reference);
    }
    command
        .kill_on_drop(true);
    let result = tokio::time::timeout(Duration::from_secs(20 * 60), command.output())
        .await
        .map_err(|_| "Image generation timed out after 20 minutes".to_string())?
        .map_err(|e| format!("Could not start {}: {e}", binary.display()))?;
    if !result.status.success() || !output.is_file() {
        let log = String::from_utf8_lossy(&result.stderr).to_string()
            + &String::from_utf8_lossy(&result.stdout);
        return Err(format!("stable-diffusion.cpp failed: {}", failure_reason(&log)));
    }
    Ok(())
}

/// sd-cli marks errors with "[E]"; the first ones carry the cause.
fn failure_reason(log: &str) -> String {
    let errors: Vec<_> = log
        .lines()
        .filter(|line| line.contains("[E]") || line.to_lowercase().contains("error"))
        .take(2)
        .map(str::trim)
        .collect();
    if errors.is_empty() {
        "it exited without an image".to_string()
    } else {
        errors.join(" | ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_follow_each_model_and_reject_unknown_aspects() {
        for model in ImageModel::ALL {
            assert_eq!(ImageModel::parse(model.id()), Some(model));
            assert!(crate::model_store::catalog_entry(model.catalog_id()).is_some());
            for aspect in ["1:1", "16:9", "9:16"] {
                let (w, h) = model.size(aspect).unwrap();
                assert_eq!((w % 16, h % 16), (0, 0), "{model:?} {aspect}");
            }
            assert_eq!(model.size("4:3"), None);
        }
        assert_eq!(ImageModel::FluxKlein.size("1:1"), Some((1024, 1024)));
        assert_eq!(ImageModel::parse("nope"), None);
    }

    #[test]
    fn edits_keep_the_reference_shape_within_a_megapixel() {
        assert_eq!(fit_megapixel(1344, 768), (1344, 768));
        assert_eq!(fit_megapixel(4032, 3024), (1184, 880));
        assert_eq!(fit_megapixel(300, 200), (304, 256));
        let (w, h) = fit_megapixel(3000, 4000);
        assert!(w * h <= 1024 * 1024 + 16 * 1024 && w % 16 == 0 && h % 16 == 0);
    }

    #[test]
    fn failures_report_the_first_errors() {
        let log = "[I] loading\n[E] model_loader: tensor 'x' has wrong shape\n[E] failed to load\n[E] exit";
        assert_eq!(
            failure_reason(log),
            "[E] model_loader: tensor 'x' has wrong shape | [E] failed to load"
        );
        assert_eq!(failure_reason("[I] ok"), "it exited without an image");
    }
}
