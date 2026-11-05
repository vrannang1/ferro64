use rusqlite::{params, Connection, OptionalExtension, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub model: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    pub conversation_id: String,
    pub role: String,
    pub content: String,
    pub model: String,
    pub created_at: String,
    pub semantic_context: Option<String>,
    pub image_url: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub stats: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbHistoryItem {
    pub id: String,
    pub prompt: String,
    pub enhanced_prompt: Option<String>,
    pub aspect_ratio: String,
    pub width: u32,
    pub height: u32,
    pub image_url: String,
    pub created_at: String,
    pub duration_ms: u64,
    pub backend: String,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    pub step_index: usize,
    pub role: String,
    pub model: String,
    pub description: String,
    pub output: String,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub id: String,
    pub goal: String,
    pub status: String,
    pub steps: Vec<AgentStep>,
    pub final_result: Option<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
    pub total_duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Video {
    pub id: String,
    pub prompt: String,
    pub video_url: String,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub seed: u64,
    pub audio: bool,
    /// The picture it started from, if any.
    pub image_url: Option<String>,
    pub duration_ms: u64,
    pub created_at: String,
    /// The prompt as the model rewrote it, when refine was on.
    #[serde(default)]
    pub enhanced_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: String,
    pub name: String,
    /// Canonical absolute path of the project folder.
    pub root: String,
    pub created_at: String,
}

/// A project's summary; see brief.rs.
#[derive(Debug, Clone)]
pub struct Brief {
    pub project_id: String,
    pub text: String,
    pub files_json: String,
    pub key_hash: String,
    /// The user changed it, so it is not rewritten automatically.
    pub edited: bool,
    pub updated_at: String,
}

/// What the coding agent learned; see notes.rs. `project_id` empty means it
/// applies to every project.
#[derive(Debug, Clone, Serialize)]
pub struct Note {
    pub id: String,
    pub project_id: Option<String>,
    pub run_id: Option<String>,
    pub title: String,
    pub body: String,
    pub source_path: Option<String>,
    #[serde(skip)]
    pub source_hash: Option<String>,
    /// "active" or "superseded".
    pub state: String,
    pub superseded_by: Option<String>,
    #[serde(skip)]
    pub embedding: Vec<f32>,
    pub created_at: String,
}

fn note_row(row: &rusqlite::Row<'_>) -> Result<Note> {
    Ok(Note {
        id: row.get(0)?,
        project_id: row.get(1)?,
        run_id: row.get(2)?,
        title: row.get(3)?,
        body: row.get(4)?,
        source_path: row.get(5)?,
        source_hash: row.get(6)?,
        state: row.get(7)?,
        superseded_by: row.get(8)?,
        embedding: crate::embed::from_blob(&row.get::<_, Vec<u8>>(9)?),
        created_at: row.get(10)?,
    })
}

const NOTE_COLUMNS: &str = "id, project_id, run_id, title, body, source_path, source_hash, state, superseded_by, embedding, created_at";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunTokens {
    pub input: u64,
    pub output: u64,
    pub cached: u64,
}

impl RunTokens {
    pub fn add(&mut self, stats: &crate::llm::ChatStats) {
        self.input += stats.prompt_tokens.unwrap_or(0);
        self.output += stats.completion_tokens.unwrap_or(0);
        self.cached += stats.cached_tokens.unwrap_or(0);
    }

    pub fn total(&self) -> u64 {
        self.input + self.output
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeRun {
    pub id: String,
    pub project_id: String,
    pub task: String,
    pub model: String,
    /// "running", "completed", "unfinished", "failed", "stopped" or "undone".
    pub status: String,
    /// OpenAI-format messages, tool calls and results included.
    pub transcript: Vec<serde_json::Value>,
    /// The agent's plan; see plan.rs.
    #[serde(default)]
    pub plan: Vec<crate::plan::PlanItem>,
    /// What the run sent to and got from the model.
    #[serde(default)]
    pub tokens: RunTokens,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone)]
pub struct CodeChange {
    pub id: i64,
    pub path: String,
    pub before: Option<String>,
    pub after: String,
}

fn code_run_row(row: &rusqlite::Row<'_>) -> Result<CodeRun> {
    Ok(CodeRun {
        id: row.get(0)?,
        project_id: row.get(1)?,
        task: row.get(2)?,
        model: row.get(3)?,
        status: row.get(4)?,
        transcript: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
        plan: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
        tokens: serde_json::from_str(&row.get::<_, String>(9)?).unwrap_or_default(),
    })
}

#[derive(Clone)]
pub struct Database {
    db_path: PathBuf,
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db_path = path.as_ref().to_path_buf();
        if let Some(parent) = db_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let conn = Connection::open(&db_path)?;

        // Pragmas for performance and safety
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;",
        )?;

        let db = Self {
            db_path,
            conn: Arc::new(Mutex::new(conn)),
        };

        db.init_schema()?;
        db.migrate_legacy_history()?;

        Ok(db)
    }

    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL,
                semantic_context TEXT,
                embedding_json TEXT,
                image_url TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS image_history (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                enhanced_prompt TEXT,
                aspect_ratio TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                image_url TEXT NOT NULL,
                created_at TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                backend TEXT NOT NULL,
                seed INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prompt_archive (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                category TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding_json TEXT,
                created_at TEXT NOT NULL,
                use_count INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS agent_tasks (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                status TEXT NOT NULL,
                steps_json TEXT NOT NULL,
                final_result TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                total_duration_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                root TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS code_runs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                task TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                transcript_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- Every file the agent writes, with its previous content (NULL for
            -- a new file), so a run can be undone.
            CREATE TABLE IF NOT EXISTS code_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES code_runs(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                before TEXT,
                after TEXT NOT NULL,
                undone INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            -- What the agent learned, across runs.
            CREATE TABLE IF NOT EXISTS code_notes (
                id TEXT PRIMARY KEY,
                project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
                run_id TEXT,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                source_path TEXT,
                source_hash TEXT,
                state TEXT NOT NULL DEFAULT 'active',
                superseded_by TEXT,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL
            );

            -- Commands the user chose to always allow in a project.
            CREATE TABLE IF NOT EXISTS code_permissions (
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                prefix TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (project_id, prefix)
            );

            -- Semantic index of a project: files with their size and time
            -- when indexed, and line-range chunks with their vectors.
            CREATE TABLE IF NOT EXISTS code_briefs (
                project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                files_json TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                edited INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS code_files (
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                modified INTEGER NOT NULL,
                size INTEGER NOT NULL,
                PRIMARY KEY (project_id, path)
            );

            CREATE TABLE IF NOT EXISTS code_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                vector BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_code_chunks_file ON code_chunks(project_id, path);

            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                video_url TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                frames INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                audio INTEGER NOT NULL,
                image_url TEXT,
                duration_ms INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
            CREATE INDEX IF NOT EXISTS idx_image_history_created ON image_history(created_at);",
        )?;

        let has_plan = conn.prepare("SELECT 1 FROM pragma_table_info('code_runs') WHERE name = 'plan_json'")?.exists([])?;
        if !has_plan {
            conn.execute("ALTER TABLE code_runs ADD COLUMN plan_json TEXT NOT NULL DEFAULT '[]'", [])?;
        }
        let has_tokens = conn.prepare("SELECT 1 FROM pragma_table_info('code_runs') WHERE name = 'tokens_json'")?.exists([])?;
        if !has_tokens {
            conn.execute("ALTER TABLE code_runs ADD COLUMN tokens_json TEXT NOT NULL DEFAULT '{}'", [])?;
        }
        // Safe column migrations for existing messages tables
        for column in ["image_url", "reasoning", "stats_json"] {
            let exists = conn
                .prepare("SELECT 1 FROM pragma_table_info('messages') WHERE name = ?1")?
                .exists(params![column])?;
            if !exists {
                conn.execute(&format!("ALTER TABLE messages ADD COLUMN {column} TEXT"), [])?;
            }
        }
        // The prompt as Gemma rewrote it, when refine was on.
        let has_enhanced = conn
            .prepare("SELECT 1 FROM pragma_table_info('videos') WHERE name = 'enhanced_prompt'")?
            .exists([])?;
        if !has_enhanced {
            conn.execute("ALTER TABLE videos ADD COLUMN enhanced_prompt TEXT", [])?;
        }

        info!("✅ SQLite3 database initialized at {:?}", self.db_path);
        Ok(())
    }

    fn migrate_legacy_history(&self) -> Result<()> {
        // If data/history.json exists, copy over any items not yet in image_history
        let parent = self.db_path.parent().unwrap_or(Path::new("."));
        let legacy_file = parent.join("history.json");

        if legacy_file.exists() {
            if let Ok(content) = std::fs::read_to_string(&legacy_file) {
                if let Ok(items) = serde_json::from_str::<Vec<DbHistoryItem>>(&content) {
                    let conn = self.conn.lock().unwrap();
                    let count: i64 = conn
                        .query_row("SELECT count(*) FROM image_history", [], |r| r.get(0))
                        .unwrap_or(0);

                    if count == 0 && !items.is_empty() {
                        for item in &items {
                            let duration_i64 = item.duration_ms as i64;
                            let seed_i64 = item.seed as i64;
                            let _ = conn.execute(
                                "INSERT OR IGNORE INTO image_history (
                                    id, prompt, enhanced_prompt, aspect_ratio, width, height, image_url, created_at, duration_ms, backend, seed
                                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                                params![
                                    item.id,
                                    item.prompt,
                                    item.enhanced_prompt,
                                    item.aspect_ratio,
                                    item.width,
                                    item.height,
                                    item.image_url,
                                    item.created_at,
                                    duration_i64,
                                    item.backend,
                                    seed_i64,
                                ],
                            );
                        }
                        info!("Transferred {} legacy generations to SQLite", items.len());
                    }
                }
            }
        }
        Ok(())
    }

    // --- Conversations & Messages ---

    pub fn create_conversation(&self, id: &str, title: &str, model: &str) -> Result<Conversation> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO conversations (id, title, model, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, title, model, now, now],
        )?;

        Ok(Conversation {
            id: id.to_string(),
            title: title.to_string(),
            model: model.to_string(),
            created_at: now.clone(),
            updated_at: now,
        })
    }

    pub fn list_conversations(&self) -> Result<Vec<Conversation>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, title, model, created_at, updated_at FROM conversations ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Conversation {
                id: row.get(0)?,
                title: row.get(1)?,
                model: row.get(2)?,
                created_at: row.get(3)?,
                updated_at: row.get(4)?,
            })
        })?;

        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    pub fn get_conversation_messages(&self, conv_id: &str) -> Result<Vec<ChatMessage>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, role, content, model, created_at, semantic_context, image_url, reasoning, stats_json
             FROM messages WHERE conversation_id = ?1 ORDER BY created_at ASC, rowid ASC",
        )?;
        let rows = stmt.query_map(params![conv_id], |row| {
            Ok(ChatMessage {
                id: row.get(0)?,
                conversation_id: row.get(1)?,
                role: row.get(2)?,
                content: row.get(3)?,
                model: row.get(4)?,
                created_at: row.get(5)?,
                semantic_context: row.get(6)?,
                image_url: row.get(7)?,
                reasoning: row.get(8)?,
                stats: row
                    .get::<_, Option<String>>(9)?
                    .and_then(|json| serde_json::from_str(&json).ok()),
            })
        })?;

        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn save_message(
        &self,
        id: &str,
        conv_id: &str,
        role: &str,
        content: &str,
        model: &str,
        semantic_context: Option<&str>,
        embedding_json: Option<&str>,
        image_url: Option<&str>,
        reasoning: Option<&str>,
        stats_json: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, model, created_at, semantic_context, embedding_json, image_url, reasoning, stats_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![id, conv_id, role, content, model, now, semantic_context, embedding_json, image_url, reasoning, stats_json],
        )?;

        // The conversation remembers the model that answered last.
        conn.execute(
            "UPDATE conversations SET updated_at = ?1, model = ?2 WHERE id = ?3",
            params![now, model, conv_id],
        )?;

        // Also index in prompt archive if user message
        if role == "user" {
            let archive_id = uuid::Uuid::new_v4().to_string();
            let _ = conn.execute(
                "INSERT INTO prompt_archive (id, prompt, category, model, embedding_json, created_at)
                 VALUES (?1, ?2, 'chat', ?3, ?4, ?5)",
                params![archive_id, content, model, embedding_json, now],
            );
        }

        Ok(())
    }

    pub fn conversation_exists(&self, conv_id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let exists = conn
            .prepare("SELECT 1 FROM conversations WHERE id = ?1")?
            .exists(params![conv_id])?;
        Ok(exists)
    }

    pub fn rename_conversation(&self, conv_id: &str, title: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let updated = conn.execute(
            "UPDATE conversations SET title = ?1 WHERE id = ?2",
            params![title, conv_id],
        )?;
        Ok(updated > 0)
    }

    pub fn delete_conversation(&self, conv_id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let deleted = conn.execute("DELETE FROM conversations WHERE id = ?1", params![conv_id])?;
        Ok(deleted > 0)
    }

    // --- Semantic Context Search (Embedding Similarity in SQLite) ---

    /// Earlier user messages similar to the query, from other conversations
    /// than `exclude`, scored by cosine similarity above `threshold`.
    pub fn find_similar_contexts(&self, query_emb: &[f32], limit: usize, threshold: f32, exclude: Option<&str>) -> Result<Vec<(String, f32)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT content, embedding_json FROM messages
             WHERE embedding_json IS NOT NULL AND role = 'user' AND conversation_id != ?1
             ORDER BY created_at DESC LIMIT 3000",
        )?;

        let mut scored: Vec<(String, f32)> = Vec::new();

        let mut rows = stmt.query(params![exclude.unwrap_or("")])?;
        while let Some(row) = rows.next()? {
            let content: String = row.get(0)?;
            let emb_str: Option<String> = row.get(1)?;

            if let Some(json) = emb_str {
                if let Ok(vec) = serde_json::from_str::<Vec<f32>>(&json) {
                    if vec.len() == query_emb.len() {
                        let sim = cosine_similarity(query_emb, &vec);
                        if sim > threshold && !scored.iter().any(|(c, _)| *c == content) {
                            scored.push((content, sim));
                        }
                    }
                }
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored)
    }

    /// User messages with no embedding, or one from another model (a
    /// different length), oldest first.
    pub fn messages_to_embed(&self, dims: usize, limit: usize) -> Result<Vec<(String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, content FROM messages
             WHERE role = 'user' AND trim(content) != ''
               AND (embedding_json IS NULL OR json_array_length(embedding_json) != ?1)
             ORDER BY created_at LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![dims as i64, limit as i64], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect()
    }

    pub fn set_message_embedding(&self, id: &str, embedding_json: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("UPDATE messages SET embedding_json = ?1 WHERE id = ?2", params![embedding_json, id])?;
        Ok(())
    }

    // --- Code index ---

    /// Indexed files: path → (modified, size).
    pub fn indexed_files(&self, project_id: &str) -> Result<std::collections::HashMap<String, (i64, i64)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT path, modified, size FROM code_files WHERE project_id = ?1")?;
        let rows = stmt.query_map(params![project_id], |row| Ok((row.get::<_, String>(0)?, (row.get(1)?, row.get(2)?))))?;
        rows.collect()
    }

    /// Replaces a file's chunks; with `file` None the file is forgotten.
    pub fn replace_file_chunks(&self, project_id: &str, path: &str, file: Option<(i64, i64)>, chunks: &[(i64, i64, String, Vec<u8>)]) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM code_chunks WHERE project_id = ?1 AND path = ?2", params![project_id, path])?;
        match file {
            Some((modified, size)) => {
                tx.execute(
                    "INSERT INTO code_files (project_id, path, modified, size) VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(project_id, path) DO UPDATE SET modified = excluded.modified, size = excluded.size",
                    params![project_id, path, modified, size],
                )?;
                for (start, end, text, vector) in chunks {
                    tx.execute(
                        "INSERT INTO code_chunks (project_id, path, start_line, end_line, text, vector) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        params![project_id, path, start, end, text, vector],
                    )?;
                }
            }
            None => {
                tx.execute("DELETE FROM code_files WHERE project_id = ?1 AND path = ?2", params![project_id, path])?;
            }
        }
        tx.commit()
    }

    /// Every chunk of a project: (path, start, end, text, vector).
    pub fn code_chunks(&self, project_id: &str) -> Result<Vec<(String, i64, i64, String, Vec<u8>)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT path, start_line, end_line, text, vector FROM code_chunks WHERE project_id = ?1")?;
        let rows = stmt.query_map(params![project_id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)))?;
        rows.collect()
    }

    // --- Videos ---

    pub fn add_video(&self, video: &Video) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO videos (id, prompt, video_url, width, height, frames, seed, audio, image_url, duration_ms, created_at, enhanced_prompt)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![video.id, video.prompt, video.video_url, video.width, video.height, video.frames, video.seed as i64,
                    video.audio, video.image_url, video.duration_ms as i64, video.created_at, video.enhanced_prompt],
        )?;
        Ok(())
    }

    pub fn list_videos(&self) -> Result<Vec<Video>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, prompt, video_url, width, height, frames, seed, audio, image_url, duration_ms, created_at, enhanced_prompt
             FROM videos ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Video {
                id: row.get(0)?,
                prompt: row.get(1)?,
                video_url: row.get(2)?,
                width: row.get(3)?,
                height: row.get(4)?,
                frames: row.get(5)?,
                seed: row.get::<_, i64>(6)? as u64,
                audio: row.get(7)?,
                image_url: row.get(8)?,
                duration_ms: row.get::<_, i64>(9)? as u64,
                created_at: row.get(10)?,
                enhanced_prompt: row.get(11)?,
            })
        })?;
        rows.collect()
    }

    /// Removes the record; returns its file URL so the caller can delete it.
    pub fn delete_video(&self, id: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let url: Option<String> = conn.query_row("SELECT video_url FROM videos WHERE id = ?1", params![id], |row| row.get(0)).optional()?;
        conn.execute("DELETE FROM videos WHERE id = ?1", params![id])?;
        Ok(url)
    }

    // --- Image History ---

    pub fn add_image(&self, item: &DbHistoryItem) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let duration_i64 = item.duration_ms as i64;
        let seed_i64 = item.seed as i64;
        conn.execute(
            "INSERT INTO image_history (
                id, prompt, enhanced_prompt, aspect_ratio, width, height, image_url, created_at, duration_ms, backend, seed
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                item.id,
                item.prompt,
                item.enhanced_prompt,
                item.aspect_ratio,
                item.width,
                item.height,
                item.image_url,
                item.created_at,
                duration_i64,
                item.backend,
                seed_i64,
            ],
        )?;
        Ok(())
    }

    pub fn list_images(&self) -> Result<Vec<DbHistoryItem>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, prompt, enhanced_prompt, aspect_ratio, width, height, image_url, created_at, duration_ms, backend, seed
             FROM image_history ORDER BY created_at DESC",
        )?;

        let rows = stmt.query_map([], |row| {
            let duration_i64: i64 = row.get(8)?;
            let seed_i64: i64 = row.get(10)?;
            Ok(DbHistoryItem {
                id: row.get(0)?,
                prompt: row.get(1)?,
                enhanced_prompt: row.get(2)?,
                aspect_ratio: row.get(3)?,
                width: row.get(4)?,
                height: row.get(5)?,
                image_url: row.get(6)?,
                created_at: row.get(7)?,
                duration_ms: duration_i64 as u64,
                backend: row.get(9)?,
                seed: seed_i64 as u64,
            })
        })?;

        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    pub fn delete_image(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let deleted = conn.execute("DELETE FROM image_history WHERE id = ?1", params![id])?;
        Ok(deleted > 0)
    }

    // --- Coding projects ---

    pub fn add_project(&self, project: &Project) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO projects (id, name, root, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![project.id, project.name, project.root, project.created_at],
        )?;
        Ok(())
    }

    pub fn list_projects(&self) -> Result<Vec<Project>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, name, root, created_at FROM projects ORDER BY name")?;
        let rows = stmt.query_map([], |row| {
            Ok(Project { id: row.get(0)?, name: row.get(1)?, root: row.get(2)?, created_at: row.get(3)? })
        })?;
        rows.collect()
    }

    pub fn project(&self, id: &str) -> Result<Option<Project>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT id, name, root, created_at FROM projects WHERE id = ?1", params![id], |row| {
            Ok(Project { id: row.get(0)?, name: row.get(1)?, root: row.get(2)?, created_at: row.get(3)? })
        })
        .optional()
    }

    /// Forgets the project and its run history; its files are not touched.
    pub fn remove_project(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.execute("DELETE FROM projects WHERE id = ?1", params![id])? > 0)
    }

    pub fn brief(&self, project_id: &str) -> Result<Option<Brief>> {
        let conn = self.conn.lock().unwrap();
        let mut statement = conn.prepare("SELECT project_id, text, files_json, key_hash, edited, updated_at FROM code_briefs WHERE project_id = ?1")?;
        let mut rows = statement.query([project_id])?;
        Ok(match rows.next()? {
            Some(row) => Some(Brief { project_id: row.get(0)?, text: row.get(1)?, files_json: row.get(2)?, key_hash: row.get(3)?, edited: row.get::<_, i64>(4)? != 0, updated_at: row.get(5)? }),
            None => None,
        })
    }

    pub fn save_brief(&self, brief: &Brief) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO code_briefs (project_id, text, files_json, key_hash, edited, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![brief.project_id, brief.text, brief.files_json, brief.key_hash, i64::from(brief.edited), brief.updated_at],
        )?;
        Ok(())
    }

    pub fn delete_brief(&self, project_id: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM code_briefs WHERE project_id = ?1", [project_id])?;
        Ok(())
    }

    /// Runs marked as running when the server starts did not survive the last
    /// shutdown. Their work so far is saved, and a new task continues them.
    pub fn close_interrupted_runs(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.execute("UPDATE code_runs SET status = 'interrupted' WHERE status = 'running'", [])?)
    }

    pub fn save_code_run(&self, run: &CodeRun) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO code_runs (id, project_id, task, model, status, transcript_json, created_at, updated_at, plan_json, tokens_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(id) DO UPDATE SET task = excluded.task, model = excluded.model, status = excluded.status,
               transcript_json = excluded.transcript_json, updated_at = excluded.updated_at, plan_json = excluded.plan_json, tokens_json = excluded.tokens_json",
            params![run.id, run.project_id, run.task, run.model, run.status,
                    serde_json::to_string(&run.transcript).unwrap_or_else(|_| "[]".into()), run.created_at, run.updated_at,
                    serde_json::to_string(&run.plan).unwrap_or_else(|_| "[]".into()),
                    serde_json::to_string(&run.tokens).unwrap_or_else(|_| "{}".into())],
        )?;
        Ok(())
    }

    pub fn code_run(&self, id: &str) -> Result<Option<CodeRun>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, project_id, task, model, status, transcript_json, created_at, updated_at, plan_json, tokens_json FROM code_runs WHERE id = ?1",
            params![id],
            code_run_row,
        )
        .optional()
    }

    pub fn list_code_runs(&self, project_id: &str) -> Result<Vec<CodeRun>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, project_id, task, model, status, transcript_json, created_at, updated_at, plan_json, tokens_json
             FROM code_runs WHERE project_id = ?1 ORDER BY created_at DESC LIMIT 50",
        )?;
        let rows = stmt.query_map(params![project_id], code_run_row)?;
        rows.collect()
    }

    pub fn save_note(&self, note: &Note) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            &format!("INSERT INTO code_notes ({NOTE_COLUMNS}) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)"),
            params![note.id, note.project_id, note.run_id, note.title, note.body, note.source_path, note.source_hash, note.state, note.superseded_by, crate::embed::to_blob(&note.embedding), note.created_at],
        )?;
        Ok(())
    }

    /// Notes a run in this project can use: its own and the ones for every project.
    pub fn active_notes(&self, project_id: &str) -> Result<Vec<Note>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!("SELECT {NOTE_COLUMNS} FROM code_notes WHERE state = 'active' AND (project_id = ?1 OR project_id IS NULL)"))?;
        let rows = stmt.query_map(params![project_id], note_row)?;
        rows.collect()
    }

    /// Everything the user can see for this project, newest first.
    pub fn all_notes(&self, project_id: &str) -> Result<Vec<Note>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!("SELECT {NOTE_COLUMNS} FROM code_notes WHERE project_id = ?1 OR project_id IS NULL ORDER BY created_at DESC"))?;
        let rows = stmt.query_map(params![project_id], note_row)?;
        rows.collect()
    }

    pub fn supersede_note(&self, id: &str, by: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("UPDATE code_notes SET state = 'superseded', superseded_by = ?2 WHERE id = ?1", params![id, by])?;
        Ok(())
    }

    pub fn restore_note(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.execute("UPDATE code_notes SET state = 'active', superseded_by = NULL WHERE id = ?1", params![id])? > 0)
    }

    pub fn delete_note(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.execute("DELETE FROM code_notes WHERE id = ?1", params![id])? > 0)
    }

    pub fn record_change(&self, run_id: &str, path: &str, before: Option<&str>, after: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO code_changes (run_id, path, before, after, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![run_id, path, before, after, chrono::Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    /// The run's changes that are not undone yet, newest first.
    pub fn pending_changes(&self, run_id: &str) -> Result<Vec<CodeChange>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, path, before, after FROM code_changes WHERE run_id = ?1 AND undone = 0 ORDER BY id DESC",
        )?;
        let rows = stmt.query_map(params![run_id], |row| {
            Ok(CodeChange { id: row.get(0)?, path: row.get(1)?, before: row.get(2)?, after: row.get(3)? })
        })?;
        rows.collect()
    }

    pub fn mark_undone(&self, change_id: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("UPDATE code_changes SET undone = 1 WHERE id = ?1", params![change_id])?;
        Ok(())
    }

    pub fn allowed_commands(&self, project_id: &str) -> Result<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT prefix FROM code_permissions WHERE project_id = ?1 ORDER BY prefix")?;
        let rows = stmt.query_map(params![project_id], |row| row.get(0))?;
        rows.collect()
    }

    pub fn allow_command(&self, project_id: &str, prefix: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO code_permissions (project_id, prefix, created_at) VALUES (?1, ?2, ?3)",
            params![project_id, prefix, chrono::Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn revoke_command(&self, project_id: &str, prefix: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.execute("DELETE FROM code_permissions WHERE project_id = ?1 AND prefix = ?2", params![project_id, prefix])? > 0)
    }

    // --- Settings ---

    pub fn setting(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT value FROM settings WHERE key = ?1", params![key], |row| row.get(0))
            .optional()
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    // --- Agent Tasks ---

    pub fn save_agent_task(&self, task: &AgentTask) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let steps_json = serde_json::to_string(&task.steps).unwrap_or_default();
        let total_duration_i64 = task.total_duration_ms as i64;
        conn.execute(
            "INSERT OR REPLACE INTO agent_tasks (id, goal, status, steps_json, final_result, created_at, completed_at, total_duration_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                task.id,
                task.goal,
                task.status,
                steps_json,
                task.final_result,
                task.created_at,
                task.completed_at,
                total_duration_i64,
            ],
        )?;
        Ok(())
    }

    pub fn list_agent_tasks(&self) -> Result<Vec<AgentTask>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, goal, status, steps_json, final_result, created_at, completed_at, total_duration_ms
             FROM agent_tasks ORDER BY created_at DESC LIMIT 50",
        )?;

        let rows = stmt.query_map([], |row| {
            let steps_str: String = row.get(3)?;
            let steps: Vec<AgentStep> = serde_json::from_str(&steps_str).unwrap_or_default();
            let total_dur_i64: i64 = row.get(7)?;
            Ok(AgentTask {
                id: row.get(0)?,
                goal: row.get(1)?,
                status: row.get(2)?,
                steps,
                final_result: row.get(4)?,
                created_at: row.get(5)?,
                completed_at: row.get(6)?,
                total_duration_ms: total_dur_i64 as u64,
            })
        })?;

        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    // --- Stats ---
    pub fn get_stats(&self) -> Result<(i64, i64, i64, i64)> {
        let conn = self.conn.lock().unwrap();
        let conv_count: i64 = conn.query_row("SELECT count(*) FROM conversations", [], |r| r.get(0)).unwrap_or(0);
        let msg_count: i64 = conn.query_row("SELECT count(*) FROM messages", [], |r| r.get(0)).unwrap_or(0);
        let img_count: i64 = conn.query_row("SELECT count(*) FROM image_history", [], |r| r.get(0)).unwrap_or(0);
        let task_count: i64 = conn.query_row("SELECT count(*) FROM agent_tasks", [], |r| r.get(0)).unwrap_or(0);
        Ok((conv_count, msg_count, img_count, task_count))
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}
