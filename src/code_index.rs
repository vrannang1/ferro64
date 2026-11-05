//! Search a project by meaning. Files are cut into overlapping line ranges,
//! embedded once, and re-embedded only when their size or modified time
//! changes; a query is compared against every stored range.
use crate::coder;
use crate::db::Database;
use crate::embed::{self, Embedder};
use std::path::Path;

const CHUNK_LINES: usize = 40;
const OVERLAP: usize = 8;
const MAX_FILES: usize = 5_000;
/// Larger files are usually generated or data, not code worth searching.
const MAX_FILE_BYTES: u64 = 400_000;
const RESULTS: usize = 8;
/// Handed over at the start of a task, so fewer and shorter than a search.
const RELATED: usize = 5;
const RELATED_LINES: usize = 12;
/// Measured on one project: code that mattered scored 0.56 to 0.69, and code that
/// merely shared a word 0.50 or less. A weak match sends the model the wrong way.
const RELATED_MIN: f32 = 0.55;

/// Line ranges of a file: (first line, last line, text with its path on top
/// so the vector knows where the code lives).
pub fn chunks(path: &str, text: &str) -> Vec<(usize, usize, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut start = 0;
    while start < lines.len() {
        let end = (start + CHUNK_LINES).min(lines.len());
        let body = lines[start..end].join("\n");
        if !body.trim().is_empty() {
            out.push((start + 1, end, format!("{path}\n{body}")));
        }
        if end == lines.len() {
            break;
        }
        start = end - OVERLAP;
    }
    out
}

fn stamp(path: &Path) -> Option<(i64, i64)> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs() as i64;
    Some((modified, meta.len() as i64))
}

/// Brings the index up to date. Returns (files embedded, files in index).
pub async fn update(db: &Database, embedder: &Embedder, project_id: &str, root: &Path) -> Result<(usize, usize), String> {
    let mut known = db.indexed_files(project_id).map_err(|e| e.to_string())?;
    let mut embedded = 0;
    let mut total = 0;
    for file in coder::walk_files(root, root, MAX_FILES) {
        let rel = coder::relative_path(root, &file);
        let Some(current) = stamp(&file) else { continue };
        if current.1 as u64 > MAX_FILE_BYTES {
            continue;
        }
        let previous = known.remove(&rel);
        if previous == Some(current) {
            total += 1;
            continue;
        }
        // Binary or unreadable files are remembered with no chunks, so they
        // are not read again until they change.
        let pieces = coder::read_text_file(&file).map(|text| chunks(&rel, &text)).unwrap_or_default();
        let texts: Vec<String> = pieces.iter().map(|(_, _, t)| t.clone()).collect();
        let vectors = if texts.is_empty() { Vec::new() } else { embedder.embed(&texts).await? };
        let rows: Vec<(i64, i64, String, Vec<u8>)> = pieces
            .into_iter()
            .zip(vectors)
            .map(|((start, end, text), vector)| (start as i64, end as i64, text, embed::to_blob(&vector)))
            .collect();
        db.replace_file_chunks(project_id, &rel, Some(current), &rows).map_err(|e| e.to_string())?;
        embedded += 1;
        total += 1;
    }
    // Files that are gone, or now skipped.
    for (path, _) in known {
        db.replace_file_chunks(project_id, &path, None, &[]).map_err(|e| e.to_string())?;
    }
    Ok((embedded, total))
}

type Hit = (f32, String, i64, i64, String);

/// Line ranges by how well they match, best first, one per place.
async fn ranked(db: &Database, embedder: &Embedder, project_id: &str, root: &Path, query: &str, limit: usize) -> Result<(usize, usize, Vec<Hit>), String> {
    let (embedded, total) = update(db, embedder, project_id, root).await?;
    let wanted = embedder.embed_one(embed::query(embed::CODE_TASK, query)).await?;
    let mut scored: Vec<Hit> = db
        .code_chunks(project_id)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|(path, start, end, text, blob)| (embed::cosine(&wanted, &embed::from_blob(&blob)), path, start, end, text))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    // One range per place: overlapping neighbours of a better hit are skipped.
    let mut picked: Vec<Hit> = Vec::new();
    for hit in scored {
        if picked.iter().any(|p| p.1 == hit.1 && hit.2 <= p.3 && hit.3 >= p.2) {
            continue;
        }
        picked.push(hit);
        if picked.len() == limit {
            break;
        }
    }
    Ok((embedded, total, picked))
}

/// Code that looks related to a task, found before the model starts so that
/// using the index does not depend on it remembering to search.
pub async fn related(db: &Database, embedder: &Embedder, project_id: &str, root: &Path, task: &str) -> Result<String, String> {
    let (_, _, picked) = ranked(db, embedder, project_id, root, task, RELATED).await?;
    let found: Vec<String> = picked
        .into_iter()
        .filter(|hit| hit.0 >= RELATED_MIN)
        .map(|(score, path, start, end, text)| {
            let body: String = text.lines().skip(1).take(RELATED_LINES).collect::<Vec<_>>().join("\n");
            format!("{path}:{start}-{end} (match {score:.2})\n{body}")
        })
        .collect();
    if found.is_empty() {
        return Ok(String::new());
    }
    Ok(format!("\n\nCode in this project that may relate to the task, found by meaning. It can be wrong, and it is only part of each file, so read a file before relying on it.\n\n{}", found.join("\n\n")))
}

/// The most relevant line ranges for a question, best first, as text for
/// the model: path:lines, score, then the code.
pub async fn search(db: &Database, embedder: &Embedder, project_id: &str, root: &Path, query: &str) -> Result<String, String> {
    if query.trim().is_empty() {
        return Err("Describe what to look for".to_string());
    }
    let (embedded, total, picked) = ranked(db, embedder, project_id, root, query, RESULTS).await?;
    let mut out = vec![format!("Most relevant code for {query:?} ({total} files indexed{}):", if embedded > 0 { format!(", {embedded} updated now") } else { String::new() })];
    for (score, path, start, end, text) in picked {
        let body: String = text.lines().skip(1).take(14).collect::<Vec<_>>().join("\n");
        out.push(format!("\n{path}:{start}-{end} (match {score:.2})\n{body}"));
    }
    out.push("\nRead a file for the full context before changing it.".to_string());
    Ok(out.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_overlap_and_cover_every_line() {
        let text: String = (1..=100).map(|n| format!("line {n}\n")).collect();
        let pieces = chunks("src/a.rs", &text);
        assert_eq!(pieces[0].0, 1);
        assert_eq!(pieces[0].1, 40);
        assert_eq!(pieces[1].0, 33, "8 lines of overlap");
        assert_eq!(pieces.last().unwrap().1, 100);
        assert!(pieces[0].2.starts_with("src/a.rs\nline 1\n"));
        assert!(chunks("empty.rs", "\n\n").is_empty());
        assert_eq!(chunks("short.rs", "fn main() {}").len(), 1);
    }
}
