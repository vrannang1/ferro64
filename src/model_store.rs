//! Allowlisted GGUF downloads from Hugging Face. Files become discoverable only
//! after size, SHA-256, and header validation. Vision models also fetch their
//! projector, saved beside the weights as `<model stem>.mmproj.gguf`.
use reqwest::Client;
use ring::digest::{Context, SHA256};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

const MAX_DOWNLOAD_BYTES: u64 = 32 * 1024 * 1024 * 1024;
const DEFAULT_MODEL: &str = "qwen3.5:9b";

#[derive(Clone, Copy, Debug)]
pub struct FileSpec {
    /// Path inside the Hugging Face repository.
    pub remote: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct CatalogEntry {
    /// Matches the Ollama tag in data/list.txt so history and defaults line up.
    pub id: &'static str,
    pub name: &'static str,
    pub role: &'static str,
    pub kind: &'static str,
    pub repo: &'static str,
    pub model: FileSpec,
    pub projector: Option<FileSpec>,
    /// Listed in data/list.txt (as opposed to a suggested addition).
    pub listed: bool,
    /// Subfolder of data/models; image weights live in "image".
    pub folder: &'static str,
    /// Local name for the second file when it is not a vision projector.
    pub extra_name: Option<&'static str>,
}

impl CatalogEntry {
    fn url(&self, file: &FileSpec) -> String {
        if file.remote.starts_with("https://huggingface.co/") {
            return file.remote.to_string();
        }
        format!("https://huggingface.co/{}/resolve/main/{}", self.repo, file.remote)
    }
    fn source(&self) -> String {
        format!("https://huggingface.co/{}", self.repo)
    }
    pub fn local_model(&self) -> &'static str {
        self.model.remote.rsplit('/').next().unwrap_or(self.model.remote)
    }
    pub fn local_projector(&self) -> Option<String> {
        if let Some(name) = self.extra_name {
            return self.projector.map(|_| name.to_string());
        }
        self.projector.map(|_| {
            let stem = self.local_model().trim_end_matches(".gguf");
            format!("{stem}.mmproj.gguf")
        })
    }
    pub fn is_image(&self) -> bool {
        self.kind == "image"
    }
    fn total_bytes(&self) -> u64 {
        self.model.bytes + self.projector.map_or(0, |p| p.bytes)
    }
}

/// Memory while running: an image model also loads its prompt readers and
/// works on the image itself.
pub fn memory_needed(entry: &CatalogEntry) -> f64 {
    let readers: u64 = prompt_readers(entry.id).iter().filter_map(|id| catalog_entry(id)).map(|e| e.total_bytes()).sum();
    let extra = if entry.is_image() { readers + 400_000_000 } else { 0 };
    crate::hardware::needed_gb(entry.total_bytes() + extra)
}

/// Dense models read every weight for every word, so they run far slower
/// than their size suggests: 11–15 tok/s on an M1 Max, against about 50 for
/// the A3B models.
const DENSE_MODELS: &[&str] = &["gemma4:31b"];

/// The best model per job that fits this machine well, by preference.
pub fn recommended(hw: &crate::hardware::Hardware) -> Value {
    use crate::hardware::{fit, Fit};
    let fits = |id: &&str, want: Fit| catalog_entry(id).is_some_and(|e| fit(hw, memory_needed(e)) == want);
    // The best that fits well; failing that, the smallest that runs at all.
    let pick = |ids: &[&'static str]| -> Option<&'static str> {
        ids.iter().find(|id| fits(id, Fit::Good)).or_else(|| ids.iter().rev().find(|id| fits(id, Fit::Tight))).copied()
    };
    json!({
        // The 9B model fits an ordinary 16 GB laptop. Offer the coding
        // specialists as the recommendation when the hardware fits them.
        "chat": pick(&["qwen3.5:9b"]),
        "code": pick(&["inkling-2-mini:26b", "devstral-small-2:24b", "qwen3.5:9b"]),
        "agents": pick(&["qwen3.5:9b"]),
        "reviewer": pick(&["qwen3.5:9b"]),
        "images": pick(&["image:z-image-turbo", "image:flux2-klein-4b"]),
        "edits": pick(&["image:flux2-klein-4b"]),
        "search": pick(&["qwen3-embedding:0.6b"]),
    })
}

// Sizes and SHA-256 values are the publishers' LFS metadata from the Hugging
// Face API, verified November 2026. No URL or destination path is accepted
// from an HTTP caller.
pub const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        id: "qwen3:4b",
        name: "Qwen3 4B",
        role: "A part of image generation: FLUX.2 Klein and Z-Image need this exact file to read prompts. It is not offered for chat",
        kind: "reasoning",
        repo: "Qwen/Qwen3-4B-GGUF",
        model: FileSpec {
            remote: "Qwen3-4B-Q4_K_M.gguf",
            sha256: "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
            bytes: 2_497_280_256,
        },
        projector: None,
        listed: false,
        folder: "",
        extra_name: None,
    },
    CatalogEntry {
        id: "qwen3.5:9b",
        name: "Qwen3.5 9B",
        role: "The default: chat, agents, web search and small coding tasks, reads images. Made for an ordinary 16 GB laptop",
        kind: "vision",
        repo: "unsloth/Qwen3.5-9B-GGUF",
        model: FileSpec {
            remote: "Qwen3.5-9B-Q4_K_M.gguf",
            sha256: "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
            bytes: 5_680_522_464,
        },
        projector: Some(FileSpec {
            remote: "mmproj-F16.gguf",
            sha256: "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f",
            bytes: 918_166_080,
        }),
        listed: false,
        folder: "",
        extra_name: None,
    },
    CatalogEntry {
        id: "gemma4:e4b",
        name: "Gemma 4 E4B",
        role: "Google's small model: fast for chat and agents, and reads images. Weak at coding in our tests",
        kind: "vision",
        repo: "unsloth/gemma-4-E4B-it-GGUF",
        model: FileSpec {
            remote: "gemma-4-E4B-it-Q4_K_M.gguf",
            sha256: "85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87",
            bytes: 4_977_171_584,
        },
        projector: Some(FileSpec {
            remote: "mmproj-F16.gguf",
            sha256: "ddf46c21d7078e95338cfc22306b19b276a29a5ad089023449dd54d4b6170a51",
            bytes: 990_372_672,
        }),
        listed: false,
        folder: "",
        extra_name: None,
    },
    CatalogEntry {
        id: "inkling-2-mini:26b",
        name: "Inkling 2 Mini",
        role: "Our strongest local coding pick across Python, Swift, Erlang, C# and Haskell. Requires llama.cpp b12623 or newer",
        kind: "coding",
        repo: "ggml-org/Inkling-2-Mini-GGUF",
        model: FileSpec {
            remote: "https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/e8a2cecf0e78914f07cc27b9be711a9b2c7561d9/Inkling-2-Mini-IQ4_NL.gguf",
            sha256: "cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4",
            bytes: 17_627_940_768,
        },
        projector: None,
        listed: false,
        folder: "",
        extra_name: None,
    },
    CatalogEntry {
        id: "devstral-small-2:24b",
        name: "Devstral Small 2 24B",
        role: "Faster on our small coding tasks, especially Swift and Haskell. Check its C# edits carefully",
        kind: "coding",
        repo: "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF",
        model: FileSpec {
            remote: "https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/resolve/6e458b8add42681bfd023de5eab93637694aaf82/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf",
            sha256: "d14ba9edee1bb4c4996a726deb81e49ae81800a3216f0774634238c380aee496",
            bytes: 14_334_446_752,
        },
        projector: None,
        listed: false,
        folder: "",
        extra_name: None,
    },
    CatalogEntry {
        id: "qwen3.6:35b-a3b",
        name: "Qwen3.6 35B-A3B",
        role: "Qwen3.5's successor, tuned for agentic coding and whole-repository reasoning; same size and speed, reads images",
        kind: "vision",
        repo: "unsloth/Qwen3.6-35B-A3B-GGUF",
        model: FileSpec {
            remote: "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            sha256: "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
            bytes: 22_134_528_992,
        },
        projector: Some(FileSpec {
            remote: "mmproj-F16.gguf",
            sha256: "8971ee4f331ff0a4c609374f32984b3d4e6dc086c0aa35f1d637fad1829e887f",
            bytes: 899_283_680,
        }),
        listed: false,
        folder: "",
        extra_name: None,
    },
    CatalogEntry {
        id: "gemma4:31b",
        name: "Gemma 4 31B",
        role: "Google's largest open model, dense 31B trained for 4-bit (QAT): strong writing and reasoning, reads images; slower than the A3B models",
        kind: "vision",
        repo: "unsloth/gemma-4-31B-it-qat-GGUF",
        model: FileSpec {
            remote: "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
            sha256: "00b5a7c497f0c8934033088c10a7fa9a4c015e46ee6d89e9c6890650ba5d0e71",
            bytes: 17_287_670_048,
        },
        projector: Some(FileSpec {
            remote: "mmproj-F16.gguf",
            sha256: "4775e1bf6ef6f8df94caed2d672c74432f0722565e5225a170c72949ebe4cf23",
            bytes: 1_198_957_024,
        }),
        listed: false,
        folder: "",
        extra_name: None,
    },
    CatalogEntry {
        id: "qwen3-embedding:0.6b",
        name: "Qwen3 Embedding 0.6B",
        role: "Search by meaning: finds related code in Code projects and earlier conversations for recall; runs beside the chat model",
        kind: "embedding",
        repo: "Qwen/Qwen3-Embedding-0.6B-GGUF",
        model: FileSpec {
            remote: "Qwen3-Embedding-0.6B-Q8_0.gguf",
            sha256: "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
            bytes: 639_150_592,
        },
        projector: None,
        listed: false,
        folder: "",
        extra_name: None,
    },
    CatalogEntry {
        id: "nemotron-3-nano:30b-a3b",
        name: "Nemotron 3 Nano 30B-A3B",
        role: "Agent reviewer: catches mistakes in a draft without adding new ones; strong reasoning",
        kind: "reasoning",
        repo: "unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        model: FileSpec {
            remote: "Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf",
            sha256: "0e7f6e51fdd9039928749d07eed9e846dbfd97681646544c5406bcdd788e5940",
            bytes: 24_574_373_664,
        },
        projector: None,
        listed: false,
        folder: "",
        extra_name: None,
    },
    CatalogEntry {
        id: "image:flux2-klein-4b",
        name: "FLUX.2 Klein 4B",
        role: "Optional: enables image edits (remove or change parts of a picture). Uses Qwen3 4B",
        kind: "image",
        repo: "leejet/FLUX.2-klein-4B-GGUF",
        model: FileSpec {
            remote: "flux-2-klein-4b-Q8_0.gguf",
            sha256: "0bba6951258ec8f92d51114a8fa13e66828297bfff58a738f52729b3ef66fa28",
            bytes: 4_300_629_440,
        },
        // The image decoder, from Comfy-Org's mirror of FLUX.2.
        projector: Some(FileSpec {
            remote: "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
            sha256: "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5",
            bytes: 336_213_556,
        }),
        listed: false,
        folder: "image",
        extra_name: Some("flux2-vae.safetensors"),
    },
    CatalogEntry {
        id: "image:z-image-turbo",
        name: "Z-Image-Turbo",
        role: "Image generation with very realistic photos in 8 steps. Uses Qwen3 4B to read prompts",
        kind: "image",
        repo: "leejet/Z-Image-Turbo-GGUF",
        model: FileSpec {
            remote: "z_image_turbo-Q8_0.gguf",
            sha256: "df1c5baa86d1398c979495a6072dbcee79444fdb884a2445582ba0769c44e9a1",
            bytes: 6_577_440_704,
        },
        // The FLUX.1 image decoder, from Comfy-Org's open mirror.
        projector: Some(FileSpec {
            remote: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
            sha256: "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
            bytes: 335_304_388,
        }),
        listed: false,
        folder: "image",
        extra_name: Some("ae.safetensors"),
    },
    // Image models with no Qwen inside: their prompts are read by Google's
    // T5 and OpenAI's CLIP, for teams that cannot use Chinese-made models.
    CatalogEntry {
        id: "image:flux1-schnell",
        name: "FLUX.1 schnell",
        role: "Image generation in 4 steps from Black Forest Labs (Germany), Apache-2.0. Reads prompts with T5-XXL and CLIP-L, no Qwen",
        kind: "image",
        repo: "leejet/FLUX.1-schnell-gguf",
        model: FileSpec {
            remote: "flux1-schnell-q8_0.gguf",
            sha256: "098c424cfa97526d9227da984563ad1509298f7573458ed210cd5ba0220ab635",
            bytes: 12_801_721_248,
        },
        // The same FLUX.1 decoder Z-Image uses; shared on disk.
        projector: Some(FileSpec {
            remote: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
            sha256: "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
            bytes: 335_304_388,
        }),
        listed: false,
        folder: "image",
        extra_name: Some("ae.safetensors"),
    },
    CatalogEntry {
        id: "image:sd3.5-large-turbo",
        name: "SD 3.5 Large Turbo",
        role: "Image generation in 4 steps from Stability AI (UK); its own license is free under $1M revenue. Reads prompts with T5-XXL, CLIP-L and CLIP-G, no Qwen",
        kind: "image",
        repo: "stduhpf/SD3.5-Large-Turbo-GGUF-mixed-sdcpp",
        // Includes its image decoder.
        model: FileSpec {
            remote: "legacy/sd3.5_large_turbo-q8_0.gguf",
            sha256: "c4d96c365b2c6952f4fd303d6242e2be1de6e68a37a4275d550aec0cc8e87f2f",
            bytes: 9_114_470_368,
        },
        projector: None,
        listed: false,
        folder: "image",
        extra_name: None,
    },
    CatalogEntry {
        id: "image:sdxl",
        name: "SDXL 1.0",
        role: "Older, lighter image model from Stability AI (UK), CreativeML OpenRAIL++. Reads prompts with its own CLIP, no extra downloads; best with keyword-style prompts",
        kind: "image",
        repo: "stabilityai/stable-diffusion-xl-base-1.0",
        model: FileSpec {
            remote: "sd_xl_base_1.0.safetensors",
            sha256: "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b",
            bytes: 6_938_078_334,
        },
        // A decoder fixed to run at half precision, as Metal does.
        projector: Some(FileSpec {
            remote: "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors",
            sha256: "235745af8d86bf4a4c1b5b4f529868b37019a10f7c0b2e79ad0abca3a22bc6e1",
            bytes: 334_641_162,
        }),
        listed: false,
        folder: "image",
        extra_name: Some("sdxl_vae.safetensors"),
    },
    // Prompt readers for the models above.
    CatalogEntry {
        id: "image:clip-l",
        name: "CLIP-L",
        role: "A part of image generation: FLUX.1 schnell and SD 3.5 read prompts with it (OpenAI, MIT)",
        kind: "encoder",
        repo: "comfyanonymous/flux_text_encoders",
        model: FileSpec {
            remote: "clip_l.safetensors",
            sha256: "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
            bytes: 246_144_152,
        },
        projector: None,
        listed: false,
        folder: "image",
        extra_name: None,
    },
    CatalogEntry {
        id: "image:clip-g",
        name: "CLIP-G",
        role: "A part of image generation: SD 3.5 reads prompts with it (OpenCLIP ViT-bigG, LAION)",
        kind: "encoder",
        repo: "Comfy-Org/stable-diffusion-3.5",
        model: FileSpec {
            remote: "text_encoders/clip_g.safetensors",
            sha256: "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4",
            bytes: 1_389_382_176,
        },
        projector: None,
        listed: false,
        folder: "image",
        extra_name: None,
    },
    CatalogEntry {
        id: "image:t5xxl",
        name: "T5-XXL",
        role: "A part of image generation: FLUX.1 schnell and SD 3.5 read prompts with it (Google, Apache-2.0)",
        kind: "encoder",
        repo: "city96/t5-v1_1-xxl-encoder-gguf",
        model: FileSpec {
            remote: "t5-v1_1-xxl-encoder-Q8_0.gguf",
            sha256: "9ec60f6028534b7fe5af439fcb535d75a68592a9ca3fcdeb175ef89e3ee99825",
            bytes: 5_061_584_064,
        },
        projector: None,
        listed: false,
        folder: "image",
        extra_name: None,
    },
];

/// The prompt readers (text encoders) an image model loads beside its own
/// weights. Downloading the model downloads these too.
pub fn prompt_readers(id: &str) -> &'static [&'static str] {
    match id {
        "image:z-image-turbo" | "image:flux2-klein-4b" => &["qwen3:4b"],
        "image:flux1-schnell" => &["image:clip-l", "image:t5xxl"],
        "image:sd3.5-large-turbo" => &["image:clip-l", "image:clip-g", "image:t5xxl"],
        _ => &[],
    }
}

pub fn catalog_entry(id: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|entry| entry.id == id)
}

/// Maps a downloaded weights filename back to its catalog ID.
pub fn catalog_id_for_file(filename: &str) -> Option<&'static str> {
    CATALOG
        .iter()
        .find(|entry| !entry.is_image() && entry.local_model().eq_ignore_ascii_case(filename))
        .map(|entry| entry.id)
}

#[derive(Clone, Serialize)]
struct DownloadStatus {
    model: String,
    status: String,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
    progress: Option<f64>,
    error: Option<String>,
}

pub struct ModelStore {
    data_dir: PathBuf,
    directory: PathBuf,
    downloads: Mutex<BTreeMap<String, DownloadStatus>>,
    queue: Mutex<()>,
}

impl ModelStore {
    pub fn new(base_dir: &Path) -> Arc<Self> {
        let data_dir = base_dir.join("data");
        Arc::new(Self {
            directory: data_dir.join("models"),
            data_dir,
            downloads: Mutex::new(BTreeMap::new()),
            queue: Mutex::new(()),
        })
    }

    fn installed(&self, entry: &CatalogEntry) -> bool {
        let dir = self.directory.join(entry.folder);
        valid_file(&dir.join(entry.local_model()), entry.model.bytes)
            && match (entry.projector, entry.local_projector()) {
                (Some(file), Some(name)) => valid_file(&dir.join(name), file.bytes),
                _ => true,
            }
    }

    pub fn catalog(&self) -> Vec<Value> {
        let hw = crate::hardware::this_machine();
        CATALOG
            .iter()
            .map(|entry| {
                let need = memory_needed(entry);
                let fit = crate::hardware::fit(hw, need);
                json!({
                    "need_gb": (need * 10.0).round() / 10.0,
                    "fit": fit,
                    "fit_note": crate::hardware::explain(hw, need, fit),
                    "dense": DENSE_MODELS.contains(&entry.id),
                    "id": entry.id,
                    "name": entry.name,
                    "role": entry.role,
                    "kind": entry.kind,
                    "filename": entry.local_model(),
                    "source": entry.source(),
                    "sha256": entry.model.sha256,
                    "size_bytes": entry.total_bytes(),
                    "vision": entry.projector.is_some() && !entry.is_image(),
                    "listed": entry.listed,
                    "installed": self.installed(entry),
                })
            })
            .collect()
    }

    /// Deletes a catalog model's own files from data/models. The caller makes
    /// sure the model is not loaded. Returns the removed file names.
    pub async fn remove(&self, id: &str) -> Result<Vec<String>, String> {
        let entry = catalog_entry(id).ok_or("Only catalog models can be removed here")?;
        if self
            .downloads
            .lock()
            .await
            .get(id)
            .is_some_and(|d| matches!(d.status.as_str(), "queued" | "downloading"))
        {
            return Err("Wait for the download to finish before removing it".to_string());
        }
        let dir = self.directory.join(entry.folder);
        let mut names = vec![entry.local_model().to_string()];
        names.extend(entry.local_projector());
        let mut removed = Vec::new();
        for name in names {
            let path = dir.join(&name);
            match tokio::fs::remove_file(&path).await {
                Ok(()) => removed.push(name),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(format!("Could not remove {name}: {error}")),
            }
        }
        if removed.is_empty() {
            return Err(format!("{} is not on disk", entry.name));
        }
        self.downloads.lock().await.remove(id);
        Ok(removed)
    }

    pub async fn status(&self) -> Value {
        let downloads: Vec<_> = self.downloads.lock().await.values().cloned().collect();
        json!({
            "directory": self.directory,
            "hardware": crate::hardware::this_machine(),
            "recommended": recommended(crate::hardware::this_machine()),
            "catalog": self.catalog(),
            "downloads": downloads,
            "local_models": self.local_models(),
        })
    }

    /// Scan only data/ and data/models/. Partial downloads and projectors do
    /// not count as ready language models. Called on startup by main.rs.
    pub async fn ensure_default(self: &Arc<Self>) -> Result<bool, String> {
        tokio::fs::create_dir_all(&self.directory)
            .await
            .map_err(|e| e.to_string())?;
        if !self.local_models().is_empty() {
            return Ok(false);
        }
        self.start_download(DEFAULT_MODEL).await
    }

    fn local_models(&self) -> Vec<Value> {
        let mut files = Vec::new();
        for directory in [&self.data_dir, &self.directory] {
            let Ok(entries) = std::fs::read_dir(directory) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();
                if path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
                    && !name.to_ascii_lowercase().contains("mmproj")
                    && valid_gguf(&path)
                {
                    files.push(json!({"filename": name, "path": path, "size_bytes": entry.metadata().ok().map(|metadata| metadata.len())}));
                }
            }
        }
        files
    }

    /// Returns true when queued. Duplicate requests are idempotent; a failed
    /// transfer can be retried explicitly. Downloads run one at a time.
    pub async fn start_download(self: &Arc<Self>, id: &str) -> Result<bool, String> {
        let entry = *catalog_entry(id)
            .ok_or_else(|| "Choose a model from the download catalog".to_string())?;
        if self.installed(&entry) {
            return Ok(false);
        }
        let mut downloads = self.downloads.lock().await;
        if downloads
            .get(id)
            .is_some_and(|download| matches!(download.status.as_str(), "queued" | "downloading"))
        {
            return Ok(false);
        }
        downloads.insert(
            id.to_string(),
            status(id, "queued", 0, entry.total_bytes(), None),
        );
        drop(downloads);
        let store = Arc::clone(self);
        tokio::spawn(async move {
            let _serial = store.queue.lock().await;
            store
                .update(entry.id, "downloading", 0, entry.total_bytes(), None)
                .await;
            match store.download_entry(&entry).await {
                Ok(bytes) => store.update(entry.id, "completed", bytes, bytes, None).await,
                Err(error) => {
                    let bytes = store
                        .downloads
                        .lock()
                        .await
                        .get(entry.id)
                        .map_or(0, |state| state.downloaded_bytes);
                    store
                        .update(entry.id, "failed", bytes, entry.total_bytes(), Some(error))
                        .await;
                }
            }
        });
        Ok(true)
    }

    async fn download_entry(&self, entry: &CatalogEntry) -> Result<u64, String> {
        let total = entry.total_bytes();
        let mut files = vec![(entry.model, entry.local_model().to_string())];
        if let (Some(projector), Some(name)) = (entry.projector, entry.local_projector()) {
            files.push((projector, name));
        }
        let mut done = 0;
        let dir = self.directory.join(entry.folder);
        for (file, local) in files {
            let target = dir.join(&local);
            if !valid_file(&target, file.bytes) {
                let partial = dir.join(format!("{local}.{}.part", uuid::Uuid::new_v4()));
                let result = self
                    .download_file(entry.id, &entry.url(&file), file, &target, &partial, done, total)
                    .await;
                if result.is_err() {
                    let _ = tokio::fs::remove_file(&partial).await;
                }
                result?;
            }
            done += file.bytes;
        }
        Ok(done)
    }

    async fn update(&self, id: &str, state: &str, bytes: u64, total: u64, error: Option<String>) {
        self.downloads
            .lock()
            .await
            .insert(id.to_string(), status(id, state, bytes, total, error));
    }

    #[allow(clippy::too_many_arguments)]
    async fn download_file(
        &self,
        id: &str,
        url: &str,
        file: FileSpec,
        target: &Path,
        partial: &Path,
        offset: u64,
        total: u64,
    ) -> Result<u64, String> {
        tokio::fs::create_dir_all(target.parent().unwrap_or(&self.directory))
            .await
            .map_err(|e| e.to_string())?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(4 * 60 * 60))
            .build()
            .map_err(|e| e.to_string())?;
        let mut response = client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("Could not start download: {e}"))?
            .error_for_status()
            .map_err(|e| format!("Model download failed: {e}"))?;
        if file.bytes > MAX_DOWNLOAD_BYTES {
            return Err("Model exceeds the 32 GiB download limit".to_string());
        }
        if let Some(actual) = response.content_length() {
            if actual != file.bytes {
                return Err(format!(
                    "The publisher's file size changed: expected {}, received {actual}",
                    file.bytes
                ));
            }
        }
        let mut output = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(partial)
            .await
            .map_err(|e| format!("Could not create partial model: {e}"))?;
        let mut digest = Context::new(&SHA256);
        let mut bytes = 0_u64;
        let mut updated = Instant::now();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| format!("Model transfer interrupted: {e}"))?
        {
            bytes += chunk.len() as u64;
            if bytes > file.bytes {
                return Err("Downloaded model exceeds the expected size".to_string());
            }
            digest.update(&chunk);
            output
                .write_all(&chunk)
                .await
                .map_err(|e| format!("Could not write model (check disk space): {e}"))?;
            if updated.elapsed() >= Duration::from_millis(250) {
                self.update(id, "downloading", offset + bytes, total, None)
                    .await;
                updated = Instant::now();
            }
        }
        output.flush().await.map_err(|e| e.to_string())?;
        output.sync_all().await.map_err(|e| e.to_string())?;
        drop(output);
        if bytes != file.bytes {
            return Err(format!(
                "Incomplete model: received {bytes} bytes, expected {}",
                file.bytes
            ));
        }
        let actual = digest
            .finish()
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        if actual != file.sha256 {
            return Err(
                "Model SHA-256 verification failed; the partial file was discarded".to_string(),
            );
        }
        if is_gguf(target) && !valid_gguf(partial) {
            return Err("The downloaded file is not a supported GGUF".to_string());
        }
        // A manually supplied ready model takes precedence over this transfer.
        if valid_file(target, file.bytes) {
            tokio::fs::remove_file(partial)
                .await
                .map_err(|e| e.to_string())?;
        } else {
            tokio::fs::rename(partial, target)
                .await
                .map_err(|e| format!("Could not finalize model: {e}"))?;
        }
        Ok(bytes)
    }
}

fn status(id: &str, state: &str, bytes: u64, total: u64, error: Option<String>) -> DownloadStatus {
    DownloadStatus {
        model: id.to_string(),
        status: state.to_string(),
        downloaded_bytes: bytes,
        total_bytes: Some(total),
        progress: (total > 0).then(|| (bytes as f64 / total as f64 * 100.0).min(100.0)),
        error,
    }
}

fn is_gguf(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
}

/// GGUF files are checked by header; other weights (safetensors) by exact size.
fn valid_file(path: &Path, bytes: u64) -> bool {
    if is_gguf(path) {
        valid_gguf(path)
    } else {
        std::fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() == bytes)
    }
}

fn valid_gguf(path: &Path) -> bool {
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut header = [0_u8; 24];
    file.read_exact(&mut header).is_ok()
        && &header[..4] == b"GGUF"
        && matches!(u32::from_le_bytes(header[4..8].try_into().unwrap()), 2 | 3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;

    async fn serve_model(bytes: Vec<u8>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/model.gguf", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request).await.unwrap();
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                bytes.len()
            );
            socket.write_all(header.as_bytes()).await.unwrap();
            socket.write_all(&bytes).await.unwrap();
        });
        url
    }

    #[test]
    fn coding_recommendations_follow_available_memory() {
        let mac = |ram_gb, gpu_gb| crate::hardware::Hardware {
            kind: "mac".into(),
            ram_gb,
            gpu_gb,
            gpu_name: None,
        };
        assert_eq!(recommended(&mac(16.0, 10.67))["code"], "qwen3.5:9b");
        assert_eq!(recommended(&mac(24.0, 16.0))["code"], "devstral-small-2:24b");
        assert_eq!(recommended(&mac(32.0, 24.0))["code"], "inkling-2-mini:26b");
    }

    #[test]
    fn catalog_paths_hashes_and_ids_are_fixed_and_cover_the_inventory() {
        let mut ids = std::collections::HashSet::new();
        for entry in CATALOG {
            assert!(ids.insert(entry.id), "duplicate id {}", entry.id);
            assert_eq!(Path::new(entry.local_model()).components().count(), 1);
            assert!(entry.url(&entry.model).starts_with("https://huggingface.co/"));
            for file in std::iter::once(entry.model).chain(entry.projector) {
                assert_eq!(file.sha256.len(), 64);
                assert!(file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()));
                assert!(file.bytes > 0 && file.bytes <= MAX_DOWNLOAD_BYTES);
            }
            if !entry.is_image() && entry.kind != "encoder" {
                assert_eq!(catalog_id_for_file(entry.local_model()), Some(entry.id));
            }
        }
        // The curated set: fast default, strong model, reviewer, coders,
        // vision, images and their prompt readers.
        for id in ["qwen3.5:9b", "qwen3.6:35b-a3b", "nemotron-3-nano:30b-a3b", "inkling-2-mini:26b", "devstral-small-2:24b", "qwen3:4b", "image:z-image-turbo", "image:flux1-schnell", "image:t5xxl"] {
            assert!(catalog_entry(id).is_some(), "{id} missing from catalog");
        }
        for id in ["qwen3.5:35b-a3b", "qwen3.8:27b", "glm-4.7-flash", "granite4.1:8b"] {
            assert!(catalog_entry(id).is_none(), "{id} was trimmed");
        }
        // Every model's prompt readers are in the catalog.
        for entry in CATALOG.iter().filter(|e| e.is_image()) {
            for reader in prompt_readers(entry.id) {
                assert!(catalog_entry(reader).is_some(), "{} needs {reader}", entry.id);
            }
        }
        let vision = catalog_entry("qwen3.6:35b-a3b").unwrap();
        assert_eq!(
            vision.local_projector().as_deref(),
            Some("Qwen3.6-35B-A3B-UD-Q4_K_M.mmproj.gguf")
        );
    }

    #[tokio::test]
    async fn scan_finds_gguf_and_does_not_download_when_present() {
        let root = std::env::temp_dir().join(format!("ferro64-store-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(root.join("data/models"))
            .await
            .unwrap();
        let mut header = [0_u8; 24];
        header[..4].copy_from_slice(b"GGUF");
        header[4..8].copy_from_slice(&3_u32.to_le_bytes());
        tokio::fs::write(root.join("data/models/custom.gguf"), header)
            .await
            .unwrap();
        let store = ModelStore::new(&root);
        assert!(!store.ensure_default().await.unwrap());
        assert_eq!(store.local_models().len(), 1);
        assert!(store.start_download("../../outside").await.is_err());
        tokio::fs::remove_dir_all(root).await.unwrap();
    }

    #[tokio::test]
    async fn remove_deletes_only_the_entrys_files() {
        let root = std::env::temp_dir().join(format!("ferro64-remove-{}", uuid::Uuid::new_v4()));
        let store = ModelStore::new(&root);
        let entry = catalog_entry("image:flux2-klein-4b").unwrap();
        let dir = store.directory.join("image");
        tokio::fs::create_dir_all(&dir).await.unwrap();
        for name in [entry.local_model(), "flux2-vae.safetensors", "keep-me.gguf"] {
            tokio::fs::write(dir.join(name), b"x").await.unwrap();
        }
        let removed = store.remove("image:flux2-klein-4b").await.unwrap();
        assert_eq!(removed.len(), 2);
        assert!(dir.join("keep-me.gguf").exists());
        assert!(store.remove("image:flux2-klein-4b").await.unwrap_err().contains("not on disk"));
        assert!(store.remove("../etc").await.is_err());
        tokio::fs::remove_dir_all(root).await.unwrap();
    }

    #[test]
    fn truncated_and_non_model_files_are_rejected() {
        let path = std::env::temp_dir().join(format!("ferro64-header-{}", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"GGUF").unwrap();
        assert!(!valid_gguf(&path));
        std::fs::write(&path, b"<html>not a model file</html>").unwrap();
        assert!(!valid_gguf(&path));
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn download_validates_digest_before_exposing_the_model() {
        let root = std::env::temp_dir().join(format!("ferro64-transfer-{}", uuid::Uuid::new_v4()));
        let store = ModelStore::new(&root);
        let mut bytes = vec![0_u8; 24];
        bytes[..4].copy_from_slice(b"GGUF");
        bytes[4..8].copy_from_slice(&3_u32.to_le_bytes());
        let digest = ring::digest::digest(&SHA256, &bytes)
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let file = FileSpec {
            remote: "fixture.gguf",
            sha256: Box::leak(digest.into_boxed_str()),
            bytes: 24,
        };
        let url = serve_model(bytes.clone()).await;
        let target = store.directory.join("fixture.gguf");
        let partial = store.directory.join("fixture.part");
        let received = store
            .download_file("fixture", &url, file, &target, &partial, 0, 24)
            .await
            .unwrap();
        assert_eq!(received, 24);
        assert!(valid_gguf(&target));
        assert!(!partial.exists());

        let wrong_url = serve_model(bytes).await;
        let bad = FileSpec {
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
            ..file
        };
        let corrupt = store.directory.join("corrupt.gguf");
        let error = store
            .download_file("fixture", &wrong_url, bad, &corrupt, &store.directory.join("corrupt.part"), 0, 24)
            .await
            .unwrap_err();
        assert!(error.contains("SHA-256"));
        assert!(!corrupt.exists());
        tokio::fs::remove_dir_all(root).await.unwrap();
    }
}
