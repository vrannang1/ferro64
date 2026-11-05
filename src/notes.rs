//! What the coding agent learned, kept across runs: a fix, a gotcha, a decision
//! and why. Only the title is embedded, so it should read as the answer someone
//! would look for. A note is a hint to check, never a fact: it can go out of date.
//!
//! A lookup needs both meaning and shared wording. Meaning alone is not enough:
//! with Qwen3-Embedding, a note about the same library but another problem can
//! score higher than the right note in different words.

use crate::db::{Database, Note};
use crate::embed::{self, Embedder};
use serde::Serialize;
use std::path::Path;

const TASK: &str = "Given a coding task or problem, retrieve notes about the same problem or decision";
const KEY: &str = "code_notes";
/// Measured with Qwen3-Embedding-0.6B on 18 worded examples; see `levels_...` below.
const STRONG_MEANING: f32 = 0.65;
const STRONG_WORDS: f32 = 0.20;
const WEAK_MEANING: f32 = 0.50;
const WEAK_WORDS: f32 = 0.15;
/// A new note this close to an active one should replace it.
const SAME_NOTE: f32 = 0.92;
const LIMIT: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    /// The whole note is handed over.
    Strong,
    /// Only the title: probably related, not certainly.
    Weak,
    Miss,
}

#[derive(Debug, Clone, Serialize)]
pub struct Found {
    pub id: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    pub level: Level,
    /// The file it came from has changed since.
    pub stale: bool,
}

pub fn enabled(db: &Database) -> bool {
    db.setting(KEY).ok().flatten().as_deref() != Some("off")
}

pub fn set_enabled(db: &Database, on: bool) -> Result<(), String> {
    db.set_setting(KEY, if on { "on" } else { "off" }).map_err(|e| e.to_string())
}

/// Three-letter pieces of each word, so `migrations` and `migrate` overlap.
fn pieces(text: &str) -> std::collections::HashSet<String> {
    let lower = text.to_lowercase();
    let mut found = std::collections::HashSet::new();
    for word in lower.split(|c: char| !c.is_alphanumeric()).filter(|w| w.chars().count() > 2) {
        let padded: Vec<char> = format!(" {word} ").chars().collect();
        found.extend(padded.windows(3).map(|w| w.iter().collect::<String>()));
    }
    found
}

/// How much of the title appears in the query. Not two-way overlap, which would
/// punish a long task description for being long.
fn shared_words(query: &str, title: &str) -> f32 {
    let (query, title) = (pieces(query), pieces(title));
    if title.is_empty() {
        return 0.0;
    }
    title.iter().filter(|p| query.contains(*p)).count() as f32 / title.len() as f32
}

pub fn level(meaning: f32, words: f32) -> Level {
    if meaning >= STRONG_MEANING && words >= STRONG_WORDS {
        Level::Strong
    } else if meaning >= WEAK_MEANING && words >= WEAK_WORDS {
        Level::Weak
    } else {
        Level::Miss
    }
}

pub fn file_hash(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let digest = ring::digest::digest(&ring::digest::SHA256, &bytes);
    Some(digest.as_ref().iter().map(|b| format!("{b:02x}")).collect())
}

fn is_stale(note: &Note, root: &Path) -> bool {
    match (&note.source_path, &note.source_hash) {
        (Some(path), Some(hash)) => file_hash(&root.join(path)).as_deref() != Some(hash.as_str()),
        _ => false,
    }
}

/// The notes this query matches, best first.
pub async fn lookup(db: &Database, embedder: &Embedder, project_id: &str, root: &Path, query: &str) -> Result<Vec<Found>, String> {
    let notes = db.active_notes(project_id).map_err(|e| e.to_string())?;
    if notes.is_empty() {
        return Ok(Vec::new());
    }
    let vector = embedder.embed_one(embed::query(TASK, query)).await?;
    let mut scored: Vec<(f32, Level, &Note)> = notes
        .iter()
        .map(|note| {
            let meaning = embed::cosine(&vector, &note.embedding);
            (meaning, level(meaning, shared_words(query, &note.title)), note)
        })
        .filter(|(_, level, _)| *level != Level::Miss)
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(scored
        .into_iter()
        .take(LIMIT)
        .map(|(_, level, note)| Found {
            id: note.id.clone(),
            title: note.title.clone(),
            body: (level == Level::Strong).then(|| note.body.clone()),
            level,
            stale: is_stale(note, root),
        })
        .collect())
}

/// How the model sees what was found.
pub fn for_model(found: &[Found]) -> String {
    if found.is_empty() {
        return String::new();
    }
    let lines: Vec<String> = found
        .iter()
        .map(|note| {
            let stale = if note.stale { " (its source file has changed since; it may be out of date)" } else { "" };
            match &note.body {
                Some(body) => format!("- [{}] {}{stale}\n  {body}", note.id, note.title),
                None => format!("- [{}] {}{stale} (possibly related; search with recall_notes for more)", note.id, note.title),
            }
        })
        .collect();
    format!(
        "[Ferro64 notes] From earlier work. They may be wrong, out of date or about a different problem, and the user's request and the code come first. Check one before relying on it. If it is wrong, replace it with save_note and `replaces`.\n{}",
        lines.join("\n")
    )
}

pub struct NewNote<'a> {
    pub project_id: Option<&'a str>,
    pub run_id: &'a str,
    pub title: &'a str,
    pub body: &'a str,
    pub source: Option<(String, String)>,
    pub replaces: Option<&'a str>,
}

/// Saves a note and returns its id, or what the model has to correct.
pub async fn save(db: &Database, embedder: &Embedder, scope_project: &str, new: NewNote<'_>) -> Result<String, String> {
    let (title, body) = (new.title.trim(), new.body.trim());
    if title.chars().count() < 15 || title.chars().count() > 200 {
        return Err("The title must be a full sentence of 15 to 200 characters: the answer someone would look for.".to_string());
    }
    if body.is_empty() || body.chars().count() > 1_500 {
        return Err("The body must say why in at most 1500 characters.".to_string());
    }
    let existing = db.active_notes(scope_project).map_err(|e| e.to_string())?;
    if let Some(id) = new.replaces {
        if !existing.iter().any(|note| note.id == id) {
            let all = db.all_notes(scope_project).map_err(|e| e.to_string())?;
            return Err(match all.iter().find(|note| note.id == id).and_then(|note| note.superseded_by.as_deref()) {
                Some(by) => format!("Note {id} has already been replaced by note {by}, so nothing more is needed. Go back to your plan."),
                None => format!("There is no active note {id} to replace."),
            });
        }
    }
    let vector = embedder.embed_one(title.to_string()).await?;
    for note in &existing {
        if new.replaces == Some(note.id.as_str()) {
            continue;
        }
        if note.title == title && note.body == body {
            return Err(format!("This is already saved as note {}.", note.id));
        }
        if embed::cosine(&vector, &note.embedding) >= SAME_NOTE {
            return Err(format!("Note {} already says nearly this: \"{}\". To correct it, pass replaces with its id.", note.id, note.title));
        }
    }
    let id = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
    let note = Note {
        id: id.clone(),
        project_id: new.project_id.map(str::to_string),
        run_id: Some(new.run_id.to_string()),
        title: title.to_string(),
        body: body.to_string(),
        source_path: new.source.as_ref().map(|s| s.0.clone()),
        source_hash: new.source.as_ref().map(|s| s.1.clone()),
        state: "active".to_string(),
        superseded_by: None,
        embedding: vector,
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    db.save_note(&note).map_err(|e| e.to_string())?;
    if let Some(old) = new.replaces {
        db.supersede_note(old, &id).map_err(|e| e.to_string())?;
    }
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scores measured for real with Qwen3-Embedding-0.6B.
    #[test]
    fn levels_keep_the_measured_examples_apart() {
        // Right note, reworded.
        for (meaning, words) in [(0.823, 0.405), (0.891, 0.548), (0.707, 0.265), (0.737, 0.512), (0.729, 0.458), (0.838, 0.313)] {
            assert_eq!(level(meaning, words), Level::Strong, "{meaning} {words}");
        }
        for (meaning, words) in [(0.585, 0.281), (0.515, 0.156)] {
            assert_eq!(level(meaning, words), Level::Weak, "{meaning} {words}");
        }
        // Same technology, another problem: the first would pass on meaning alone.
        for (meaning, words) in [(0.626, 0.119), (0.511, 0.095), (0.478, 0.093), (0.442, 0.024), (0.422, 0.181), (0.316, 0.172)] {
            assert_eq!(level(meaning, words), Level::Miss, "{meaning} {words}");
        }
        for (meaning, words) in [(0.250, 0.0), (0.264, 0.102), (0.220, 0.047), (0.367, 0.090)] {
            assert_eq!(level(meaning, words), Level::Miss, "{meaning} {words}");
        }
    }

    #[test]
    fn shared_words_measures_the_title_and_ignores_length() {
        let title = "Ecto only finds migrations in priv/repo/migrations";
        let short = shared_words("ecto migrations not found", title);
        let long = shared_words(&format!("ecto migrations not found {}", "and many other unrelated words here ".repeat(20)), title);
        assert!(short > 0.3, "{short}");
        assert!(long >= short, "a longer query must not score lower");
        assert_eq!(shared_words("what is the capital of france", "Dapper returns dates as text"), 0.0);
    }

    #[test]
    fn the_model_is_told_a_note_may_be_wrong() {
        let found = [
            Found { id: "abc".into(), title: "Strong title".into(), body: Some("why it matters".into()), level: Level::Strong, stale: true },
            Found { id: "def".into(), title: "Weak title".into(), body: None, level: Level::Weak, stale: false },
        ];
        let text = for_model(&found);
        assert!(text.contains("may be wrong"));
        assert!(text.contains("why it matters"));
        assert!(text.contains("out of date"));
        assert!(text.contains("possibly related"));
        assert!(for_model(&[]).is_empty());
    }

    #[test]
    fn a_changed_source_file_makes_a_note_stale() {
        let dir = std::env::temp_dir().join(format!("ferro64-note-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.ex"), "one").unwrap();
        let note = Note {
            id: "n".into(), project_id: None, run_id: None, title: "t".into(), body: "b".into(),
            source_path: Some("a.ex".into()), source_hash: file_hash(&dir.join("a.ex")),
            state: "active".into(), superseded_by: None, embedding: vec![], created_at: String::new(),
        };
        assert!(!is_stale(&note, &dir));
        std::fs::write(dir.join("a.ex"), "two").unwrap();
        assert!(is_stale(&note, &dir));
        std::fs::remove_file(dir.join("a.ex")).unwrap();
        assert!(is_stale(&note, &dir));
    }
}
