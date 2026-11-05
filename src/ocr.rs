//! Validated local uploads and native macOS Vision / PDFKit OCR.
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::Serialize;
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::process::Command;
use uuid::Uuid;

pub const MAX_FILE_BYTES: usize = 20 * 1024 * 1024;
pub const MAX_JSON_BYTES: usize = 29 * 1024 * 1024;
pub struct Upload {
    pub path: PathBuf,
    pub url: String,
    pub base64: String,
    pub is_pdf: bool,
}
#[derive(Serialize)]
pub struct OcrResult {
    pub text: String,
    pub document_url: String,
    pub preview_url: Option<String>,
    pub engine: String,
}

pub fn binary_path(static_dir: &Path) -> PathBuf {
    std::env::var("OCR_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            static_dir
                .parent()
                .unwrap_or(Path::new("."))
                .join("bin/ocr")
        })
}

fn decode_upload(raw: &str) -> Result<(Vec<u8>, &'static str), String> {
    let raw = raw.trim();
    let (encoded, mime) = if raw.starts_with("data:") {
        let (header, encoded) = raw.split_once(";base64,").ok_or("Use a base64 data URL")?;
        let mime = header.strip_prefix("data:").unwrap_or("");
        if !["application/pdf", "image/png", "image/jpeg", "image/webp"].contains(&mime) {
            return Err("Supported file types: PDF, PNG, JPEG, WebP".into());
        }
        (encoded, Some(mime))
    } else {
        (raw, None)
    };
    if encoded.len() > MAX_FILE_BYTES.div_ceil(3) * 4 {
        return Err("File exceeds the 20 MB limit".into());
    }
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|_| "Invalid base64 attachment")?;
    if bytes.is_empty() || bytes.len() > MAX_FILE_BYTES {
        return Err("Attachment is empty or exceeds 20 MB".into());
    }
    let (ext, actual_mime) = if bytes.starts_with(b"%PDF-") {
        ("pdf", "application/pdf")
    } else {
        let format =
            image::guess_format(&bytes).map_err(|_| "Attachment is not a valid image or PDF")?;
        let pair = match format {
            image::ImageFormat::Png => ("png", "image/png"),
            image::ImageFormat::Jpeg => ("jpg", "image/jpeg"),
            image::ImageFormat::WebP => ("webp", "image/webp"),
            _ => return Err("Supported images: PNG, JPEG, WebP".into()),
        };
        let reader = image::ImageReader::with_format(std::io::Cursor::new(&bytes), format);
        let (w, h) = reader
            .into_dimensions()
            .map_err(|_| "Image header is corrupt")?;
        if w == 0 || h == 0 || u64::from(w) * u64::from(h) > 40_000_000 {
            return Err("Image exceeds 40 megapixels".into());
        }
        pair
    };
    if mime.is_some_and(|m| m != actual_mime) {
        return Err("File content does not match its declared type".into());
    }
    Ok((bytes, ext))
}

pub async fn save_upload(static_dir: &Path, raw: &str) -> Result<Upload, String> {
    let (bytes, ext) = decode_upload(raw)?;
    let filename = format!("{}.{}", Uuid::new_v4(), ext);
    let path = static_dir.join("uploads").join(&filename);
    tokio::fs::write(&path, &bytes)
        .await
        .map_err(|e| format!("Could not save attachment: {e}"))?;
    Ok(Upload {
        path,
        url: format!("/static/uploads/{filename}"),
        base64: STANDARD.encode(&bytes),
        is_pdf: ext == "pdf",
    })
}

pub async fn extract(static_dir: &Path, doc: &Upload) -> Result<OcrResult, String> {
    let binary = binary_path(static_dir);
    if !binary.is_file() {
        return Err("Native OCR helper is missing. On macOS run scripts/build-ocr.sh, or set OCR_BIN to a compatible helper.".into());
    }
    let preview = doc.path.with_extension("preview.png");
    let mut command = Command::new(binary);
    command.arg(&doc.path).kill_on_drop(true);
    if doc.is_pdf {
        command.arg(&preview);
    }
    let output = tokio::time::timeout(Duration::from_secs(120), command.output())
        .await
        .map_err(|_| "OCR timed out after 120 seconds")?
        .map_err(|e| format!("Cannot run native OCR helper: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "OCR failed: {}",
            String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(500)
                .collect::<String>()
        ));
    }
    let text = String::from_utf8(output.stdout)
        .map_err(|_| "OCR returned invalid UTF-8")?
        .trim()
        .to_string();
    if text.len() > 500_000 {
        return Err(
            "Extracted document text exceeds 500,000 bytes. Split the document into smaller files."
                .into(),
        );
    }
    let preview_url = if doc.is_pdf && preview.is_file() {
        Some(format!(
            "/static/uploads/{}",
            preview.file_name().unwrap().to_string_lossy()
        ))
    } else if !doc.is_pdf {
        Some(doc.url.clone())
    } else {
        None
    };
    Ok(OcrResult {
        text,
        document_url: doc.url.clone(),
        preview_url,
        engine: "Apple Vision + PDFKit".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_or_mislabeled_attachments() {
        assert!(decode_upload("data:image/png;base64,bm90IGEgcG5n").is_err());
        assert!(decode_upload("data:image/svg+xml;base64,PHN2Zz4=").is_err());
        assert!(decode_upload("data:image/png;base64,JVBERi0xLjc=").is_err());
        assert!(decode_upload("data:application/pdf;base64,JVBERi0xLjc=").is_ok());
        assert!(decode_upload("%%%invalid%%%").is_err());
    }
}
