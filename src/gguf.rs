//! Minimal GGUF metadata reader. Only the header key/value section is parsed;
//! tensor data is never read. Large arrays (tokenizer vocabularies) are skipped
//! by seeking, so a multi-GB model is inspected in a few milliseconds.
use std::collections::HashMap;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgufInfo {
    pub architecture: Option<String>,
    pub name: Option<String>,
    pub size_label: Option<String>,
    pub file_type: Option<u32>,
    pub context_length: Option<u64>,
    pub chat_template: Option<String>,
}

/// How a model's chat template handles reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Thinking {
    /// No reasoning markup in the template.
    None,
    /// The template reasons unconditionally (e.g. DeepSeek-R1 distills).
    Always,
    /// The template honours `enable_thinking` (Qwen3, Nemotron, …).
    Toggle,
    /// The template honours `reasoning_effort` (gpt-oss style levels).
    Effort,
}

impl Thinking {
    pub fn as_str(self) -> &'static str {
        match self {
            Thinking::None => "none",
            Thinking::Always => "always",
            Thinking::Toggle => "toggle",
            Thinking::Effort => "effort",
        }
    }
}

impl GgufInfo {
    pub fn thinking(&self) -> Thinking {
        let Some(template) = self.chat_template.as_deref() else {
            return Thinking::None;
        };
        if template.contains("reasoning_effort") {
            Thinking::Effort
        } else if template.contains("enable_thinking") {
            Thinking::Toggle
        } else if template.contains("<think>") || template.contains("<|thinking|>") {
            Thinking::Always
        } else {
            Thinking::None
        }
    }

    /// Human label for `general.file_type` (llama_ftype).
    pub fn quantization(&self) -> Option<&'static str> {
        Some(match self.file_type? {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            7 => "Q8_0",
            8 => "Q5_0",
            9 => "Q5_1",
            10 => "Q2_K",
            11 => "Q3_K_S",
            12 => "Q3_K_M",
            13 => "Q3_K_L",
            14 => "Q4_K_S",
            15 => "Q4_K_M",
            16 => "Q5_K_S",
            17 => "Q5_K_M",
            18 => "Q6_K",
            19 => "IQ2_XXS",
            20 => "IQ2_XS",
            21 => "Q2_K_S",
            22 => "IQ3_XS",
            23 => "IQ3_XXS",
            24 => "IQ1_S",
            25 => "IQ4_NL",
            26 => "IQ3_S",
            27 => "IQ3_M",
            28 => "IQ2_S",
            29 => "IQ2_M",
            30 => "IQ4_XS",
            31 => "IQ1_M",
            32 => "BF16",
            36 => "TQ1_0",
            37 => "TQ2_0",
            38 => "MXFP4",
            _ => return None,
        })
    }
}

const MAX_STRING: u64 = 4 * 1024 * 1024;

struct Reader<R: Read + Seek> {
    inner: BufReader<R>,
}

impl<R: Read + Seek> Reader<R> {
    fn u32(&mut self) -> std::io::Result<u32> {
        let mut b = [0u8; 4];
        self.inner.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> std::io::Result<u64> {
        let mut b = [0u8; 8];
        self.inner.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn skip(&mut self, n: u64) -> std::io::Result<()> {
        self.inner.seek_relative(n as i64)
    }
    fn string(&mut self) -> std::io::Result<String> {
        let len = self.u64()?;
        if len > MAX_STRING {
            return Err(invalid("GGUF string too long"));
        }
        let mut buf = vec![0u8; len as usize];
        self.inner.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }
    fn skip_string(&mut self) -> std::io::Result<()> {
        let len = self.u64()?;
        self.skip(len)
    }
}

fn invalid(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

fn scalar_size(kind: u32) -> Option<u64> {
    Some(match kind {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}

enum Value {
    Uint(u64),
    Str(String),
    Other,
}

fn read_value<R: Read + Seek>(r: &mut Reader<R>, kind: u32, keep_string: bool) -> std::io::Result<Value> {
    match kind {
        8 if keep_string => Ok(Value::Str(r.string()?)),
        8 => r.skip_string().map(|_| Value::Other),
        9 => {
            let item = r.u32()?;
            let count = r.u64()?;
            if let Some(size) = scalar_size(item) {
                r.skip(size.checked_mul(count).ok_or_else(|| invalid("GGUF array overflow"))?)?;
            } else if item == 8 {
                for _ in 0..count {
                    r.skip_string()?;
                }
            } else {
                return Err(invalid("Nested GGUF arrays are not supported"));
            }
            Ok(Value::Other)
        }
        0 | 1 | 7 => {
            let mut b = [0u8; 1];
            r.inner.read_exact(&mut b)?;
            Ok(Value::Uint(b[0] as u64))
        }
        2 | 3 => {
            let mut b = [0u8; 2];
            r.inner.read_exact(&mut b)?;
            Ok(Value::Uint(u16::from_le_bytes(b) as u64))
        }
        4 | 5 => Ok(Value::Uint(r.u32()? as u64)),
        6 => r.u32().map(|_| Value::Other),
        10 | 11 => Ok(Value::Uint(r.u64()?)),
        12 => r.u64().map(|_| Value::Other),
        _ => Err(invalid("Unknown GGUF value type")),
    }
}

pub fn read(path: &Path) -> std::io::Result<GgufInfo> {
    let file = std::fs::File::open(path)?;
    let mut r = Reader {
        inner: BufReader::with_capacity(64 * 1024, file),
    };
    let mut magic = [0u8; 4];
    r.inner.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(invalid("Not a GGUF file"));
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(invalid("Unsupported GGUF version"));
    }
    let _tensors = r.u64()?;
    let kv_count = r.u64()?;
    let mut info = GgufInfo::default();
    let mut uints: HashMap<String, u64> = HashMap::new();
    for _ in 0..kv_count {
        let key = r.string()?;
        let kind = r.u32()?;
        let wanted_string = matches!(
            key.as_str(),
            "general.architecture" | "general.name" | "general.size_label" | "tokenizer.chat_template"
        );
        match read_value(&mut r, kind, wanted_string)? {
            Value::Str(s) => match key.as_str() {
                "general.architecture" => info.architecture = Some(s),
                "general.name" => info.name = Some(s),
                "general.size_label" => info.size_label = Some(s),
                _ => info.chat_template = Some(s),
            },
            Value::Uint(v) => {
                if key == "general.file_type" || key.ends_with(".context_length") {
                    uints.insert(key, v);
                }
            }
            Value::Other => {}
        }
    }
    info.file_type = uints.get("general.file_type").map(|v| *v as u32);
    if let Some(arch) = &info.architecture {
        info.context_length = uints.get(&format!("{arch}.context_length")).copied();
    }
    Ok(info)
}

/// Metadata keyed by path and modification time, so a replaced file is re-read.
static CACHE: Mutex<Option<HashMap<PathBuf, (SystemTime, GgufInfo)>>> = Mutex::new(None);

pub fn read_cached(path: &Path) -> Option<GgufInfo> {
    let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok()?;
    if let Some((stamp, info)) = CACHE
        .lock()
        .ok()?
        .get_or_insert_with(HashMap::new)
        .get(path)
    {
        if *stamp == modified {
            return Some(info.clone());
        }
    }
    let info = read(path).ok()?;
    if let Ok(mut cache) = CACHE.lock() {
        cache
            .get_or_insert_with(HashMap::new)
            .insert(path.to_path_buf(), (modified, info.clone()));
    }
    Some(info)
}

#[cfg(test)]
pub(crate) fn fixture(kvs: &[(&str, &str)], ints: &[(&str, u32)]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend(((kvs.len() + ints.len() + 1) as u64).to_le_bytes());
    let push_str = |out: &mut Vec<u8>, s: &str| {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    };
    // A string array, as found in tokenizer vocabularies, must be skipped.
    push_str(&mut out, "tokenizer.ggml.tokens");
    out.extend(9u32.to_le_bytes());
    out.extend(8u32.to_le_bytes());
    out.extend(3u64.to_le_bytes());
    for token in ["a", "bb", "<think>"] {
        push_str(&mut out, token);
    }
    for (key, value) in kvs {
        push_str(&mut out, key);
        out.extend(8u32.to_le_bytes());
        push_str(&mut out, value);
    }
    for (key, value) in ints {
        push_str(&mut out, key);
        out.extend(4u32.to_le_bytes());
        out.extend(value.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_metadata_and_classifies_reasoning_templates() {
        let path = std::env::temp_dir().join(format!("gguf-meta-{}.gguf", uuid::Uuid::new_v4()));
        std::fs::write(
            &path,
            fixture(
                &[
                    ("general.architecture", "qwen3"),
                    ("general.name", "Qwen3 4B"),
                    ("tokenizer.chat_template", "{% if enable_thinking %}<think>{% endif %}"),
                ],
                &[("general.file_type", 15), ("qwen3.context_length", 40960)],
            ),
        )
        .unwrap();
        let info = read(&path).unwrap();
        assert_eq!(info.architecture.as_deref(), Some("qwen3"));
        assert_eq!(info.name.as_deref(), Some("Qwen3 4B"));
        assert_eq!(info.context_length, Some(40960));
        assert_eq!(info.quantization(), Some("Q4_K_M"));
        assert_eq!(info.thinking(), Thinking::Toggle);
        assert_eq!(read_cached(&path), Some(info));
        std::fs::remove_file(path).unwrap();

        let always = GgufInfo {
            chat_template: Some("<｜Assistant｜><think>\n".into()),
            ..Default::default()
        };
        assert_eq!(always.thinking(), Thinking::Always);
        let effort = GgufInfo {
            chat_template: Some("Reasoning: {{ reasoning_effort }}".into()),
            ..Default::default()
        };
        assert_eq!(effort.thinking(), Thinking::Effort);
        assert_eq!(GgufInfo::default().thinking(), Thinking::None);
    }

    #[test]
    fn rejects_non_gguf() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        assert!(read(&path).is_err());
    }
}
