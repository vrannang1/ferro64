//! The personal assistant: standing jobs that research the web on a schedule
//! and write a dated report, such as "every weekday at 7:30, look at my
//! watchlist and suggest what to buy, hold or avoid".
//!
//! The assistant only reads. It has web search, page reading and market
//! quotes, and nothing that can act: no brokerage, no email, no files. It
//! writes a report and the user decides. Picks in a report can be marked
//! bought, passed or watching, and that decision log goes back to the next
//! run so it knows what the user did with its earlier ideas.
//!
//! Prices and percentages come from tools (computed here, not by the model),
//! the same rule as spreadsheets in chat.

#[path = "assistant_table.rs"]
mod table;
#[path = "assistant_telegram.rs"]
mod telegram;
#[path = "assistant_email.rs"]
mod email;

use crate::db::Database;
use crate::llm::{ChatOptions, Effort, LlmEngine, Temperature, ToolCall, REASONING_EXHAUSTED};
use crate::websearch;
use actix_web::{web, HttpResponse};
use chrono::{Datelike, Local, NaiveTime};
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Model turns before the assistant is told to write up what it has.
const MAX_TURNS: usize = 14;
/// Tool calls in one run, across all turns.
const MAX_TOOL_CALLS: usize = 28;
const MAX_CALLS_PER_TURN: usize = 5;
const PAGE_CHARS: usize = 6_000;
const SEARCH_CHARS: usize = 7_000;
const PREVIOUS_REPORT_CHARS: usize = 2_500;
/// Notes the assistant keeps per job, across runs.
const MAX_NOTES: usize = 40;
const NOTE_CHARS: usize = 400;
const NOTES_PER_RUN: usize = 4;
const REMINDERS_PER_RUN: usize = 3;
/// Room in the prompt for rows saved before, so the model can skip them.
const SAVED_LIST_CHARS: usize = 4_000;

// ---------- records ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub name: String,
    pub instructions: String,
    /// "HH:MM" local time, or None for a job that only runs when asked.
    pub time: Option<String>,
    pub weekdays_only: bool,
    pub enabled: bool,
    pub model: Option<String>,
    pub created_at: String,
    /// The local date of the last scheduled run, so a day runs once.
    pub last_run_on: Option<String>,
    /// Column names for a table saved as CSV, e.g. "name, clinic, phone".
    #[serde(default)]
    pub table_columns: Option<String>,
}

impl Job {
    fn columns(&self) -> Vec<String> {
        self.table_columns.as_deref().map(table::parse_columns).unwrap_or_default()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceRef {
    pub n: usize,
    pub title: String,
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Step {
    pub tool: String,
    pub input: String,
    pub summary: String,
    pub ok: bool,
    pub ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Pick {
    #[serde(default)]
    pub symbol: String,
    /// buy, hold, sell, watch or avoid.
    #[serde(default)]
    pub action: String,
    #[serde(default)]
    pub conviction: String,
    #[serde(default)]
    pub thesis: String,
    #[serde(default)]
    pub risks: String,
    #[serde(default)]
    pub sources: Vec<String>,
    /// The quote when the report was written, filled in here.
    #[serde(default)]
    pub price: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub id: String,
    pub job_id: String,
    pub job_name: String,
    /// queued, running, completed or failed.
    pub status: String,
    /// schedule or manual.
    pub trigger: String,
    pub model: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub report: Option<String>,
    pub sources: Vec<SourceRef>,
    pub steps: Vec<Step>,
    pub picks: Vec<Pick>,
    pub error: Option<String>,
    #[serde(default)]
    pub table: Option<table::Table>,
    /// Reminders the assistant suggests. Nothing is added to a calendar
    /// until the user opens one.
    #[serde(default)]
    pub reminders: Vec<Reminder>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Reminder {
    pub title: String,
    /// YYYY-MM-DD, local.
    pub date: String,
    /// HH:MM, local; none for an all-day reminder.
    pub time: Option<String>,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    pub id: String,
    pub job_id: String,
    pub text: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub id: String,
    pub run_id: String,
    pub symbol: String,
    /// bought, sold, passed or watching.
    pub decision: String,
    pub note: String,
    /// The price at the pick, or what the user entered.
    pub price: Option<f64>,
    pub decided_at: String,
    #[serde(default)]
    pub action: String,
    #[serde(default)]
    pub job_name: String,
    /// Filled in when the list is asked for with live prices.
    #[serde(default, skip_deserializing)]
    pub current: Option<f64>,
}

// ---------- storage ----------

pub fn init(db: &Database) -> rusqlite::Result<()> {
    db.with_conn(|conn| {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS assistant_jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                instructions TEXT NOT NULL,
                time TEXT,
                weekdays_only INTEGER NOT NULL DEFAULT 0,
                enabled INTEGER NOT NULL DEFAULT 1,
                model TEXT,
                created_at TEXT NOT NULL,
                last_run_on TEXT
            );
            CREATE TABLE IF NOT EXISTS assistant_runs (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                job_name TEXT NOT NULL,
                status TEXT NOT NULL,
                trigger TEXT NOT NULL,
                model TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                report TEXT,
                sources_json TEXT NOT NULL DEFAULT '[]',
                steps_json TEXT NOT NULL DEFAULT '[]',
                picks_json TEXT NOT NULL DEFAULT '[]',
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS assistant_decisions (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                decision TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                price REAL,
                decided_at TEXT NOT NULL,
                UNIQUE(run_id, symbol)
            );
            CREATE INDEX IF NOT EXISTS idx_assistant_runs_job ON assistant_runs(job_id, started_at);
            CREATE TABLE IF NOT EXISTS assistant_notes (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS assistant_rows (
                job_id TEXT NOT NULL,
                key TEXT NOT NULL,
                row_json TEXT NOT NULL,
                run_id TEXT NOT NULL,
                added_at TEXT NOT NULL,
                PRIMARY KEY (job_id, key)
            );",
        )?;
        for (table, column) in [("assistant_jobs", "table_columns"), ("assistant_runs", "table_json"), ("assistant_runs", "reminders_json")] {
            let exists = conn.prepare(&format!("SELECT 1 FROM pragma_table_info('{table}') WHERE name = ?1"))?.exists(params![column])?;
            if !exists {
                conn.execute(&format!("ALTER TABLE {table} ADD COLUMN {column} TEXT"), [])?;
            }
        }
        let has_heartbeat = conn.prepare("SELECT 1 FROM pragma_table_info('assistant_runs') WHERE name = 'updated_at'")?.exists([])?;
        if !has_heartbeat {
            conn.execute("ALTER TABLE assistant_runs ADD COLUMN updated_at TEXT", [])?;
        }
        Ok(())
    })?;
    close_stale_runs(db)
}

/// A run saves itself after every step. One that has not for this long was
/// cut off by a shutdown. Going by time, not by "unfinished at startup",
/// leaves alone a run that a second Ferro64 on the same data is working on.
const STALE_AFTER_MINUTES: i64 = 30;

fn close_stale_runs(db: &Database) -> rusqlite::Result<()> {
    let cutoff = (chrono::Utc::now() - chrono::Duration::minutes(STALE_AFTER_MINUTES)).to_rfc3339();
    db.with_conn(|conn| {
        conn.execute(
            "UPDATE assistant_runs SET status = 'failed', error = 'Ferro64 stopped during this run', finished_at = COALESCE(updated_at, started_at)
             WHERE status IN ('queued', 'running') AND COALESCE(updated_at, started_at) < ?1",
            params![cutoff],
        )?;
        Ok(())
    })
}

/// Marks a scheduled job as run today, and says whether this call did it.
/// The update is one statement, so two Ferro64 processes on the same data
/// cannot both start the same job.
fn claim_today(db: &Database, job_id: &str, today: &str) -> rusqlite::Result<bool> {
    db.with_conn(|conn| {
        let changed = conn.execute(
            "UPDATE assistant_jobs SET last_run_on = ?2 WHERE id = ?1 AND (last_run_on IS NULL OR last_run_on <> ?2)",
            params![job_id, today],
        )?;
        Ok(changed == 1)
    })
}

fn job_row(row: &rusqlite::Row) -> rusqlite::Result<Job> {
    Ok(Job {
        id: row.get(0)?,
        name: row.get(1)?,
        instructions: row.get(2)?,
        time: row.get(3)?,
        weekdays_only: row.get::<_, i64>(4)? != 0,
        enabled: row.get::<_, i64>(5)? != 0,
        model: row.get(6)?,
        created_at: row.get(7)?,
        last_run_on: row.get(8)?,
        table_columns: row.get(9)?,
    })
}

const JOB_COLUMNS: &str = "id, name, instructions, time, weekdays_only, enabled, model, created_at, last_run_on, table_columns";
const RUN_COLUMNS: &str = "id, job_id, job_name, status, trigger, model, started_at, finished_at, report, sources_json, steps_json, picks_json, error, table_json, reminders_json";

fn run_row(row: &rusqlite::Row) -> rusqlite::Result<Run> {
    let json = |i: usize| -> rusqlite::Result<String> { row.get(i) };
    Ok(Run {
        id: row.get(0)?,
        job_id: row.get(1)?,
        job_name: row.get(2)?,
        status: row.get(3)?,
        trigger: row.get(4)?,
        model: row.get(5)?,
        started_at: row.get(6)?,
        finished_at: row.get(7)?,
        report: row.get(8)?,
        sources: serde_json::from_str(&json(9)?).unwrap_or_default(),
        steps: serde_json::from_str(&json(10)?).unwrap_or_default(),
        picks: serde_json::from_str(&json(11)?).unwrap_or_default(),
        error: row.get(12)?,
        table: row.get::<_, Option<String>>(13)?.and_then(|t| serde_json::from_str(&t).ok()),
        reminders: row.get::<_, Option<String>>(14)?.and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default(),
    })
}

fn list_jobs(db: &Database) -> rusqlite::Result<Vec<Job>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(&format!("SELECT {JOB_COLUMNS} FROM assistant_jobs ORDER BY created_at"))?;
        let rows = stmt.query_map([], job_row)?;
        rows.collect()
    })
}

fn get_job(db: &Database, id: &str) -> rusqlite::Result<Option<Job>> {
    db.with_conn(|conn| {
        conn.query_row(&format!("SELECT {JOB_COLUMNS} FROM assistant_jobs WHERE id = ?1"), params![id], job_row)
            .optional()
    })
}

fn save_job(db: &Database, job: &Job) -> rusqlite::Result<()> {
    db.with_conn(|conn| {
        conn.execute(
            &format!("INSERT OR REPLACE INTO assistant_jobs ({JOB_COLUMNS}) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)"),
            params![job.id, job.name, job.instructions, job.time, job.weekdays_only as i64, job.enabled as i64, job.model, job.created_at, job.last_run_on, job.table_columns],
        )?;
        Ok(())
    })
}

fn save_run(db: &Database, run: &Run) -> rusqlite::Result<()> {
    db.with_conn(|conn| {
        conn.execute(
            &format!("INSERT OR REPLACE INTO assistant_runs ({RUN_COLUMNS}, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)"),
            params![
                run.id, run.job_id, run.job_name, run.status, run.trigger, run.model, run.started_at, run.finished_at, run.report,
                to_json(&run.sources), to_json(&run.steps), to_json(&run.picks), run.error, run.table.as_ref().map(to_json), to_json(&run.reminders), chrono::Utc::now().to_rfc3339()
            ],
        )?;
        Ok(())
    })
}

fn to_json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "[]".into())
}

fn get_run(db: &Database, id: &str) -> rusqlite::Result<Option<Run>> {
    db.with_conn(|conn| {
        conn.query_row(&format!("SELECT {RUN_COLUMNS} FROM assistant_runs WHERE id = ?1"), params![id], run_row)
            .optional()
    })
}

fn list_runs(db: &Database, job_id: Option<&str>, limit: usize) -> rusqlite::Result<Vec<Run>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(&format!(
            "SELECT {RUN_COLUMNS} FROM assistant_runs WHERE (?1 IS NULL OR job_id = ?1) ORDER BY started_at DESC LIMIT ?2"
        ))?;
        let rows = stmt.query_map(params![job_id, limit as i64], run_row)?;
        rows.collect()
    })
}

fn last_report(db: &Database, job_id: &str) -> rusqlite::Result<Option<(String, String)>> {
    db.with_conn(|conn| {
        conn.query_row(
            "SELECT started_at, report FROM assistant_runs WHERE job_id = ?1 AND status = 'completed' AND report IS NOT NULL ORDER BY started_at DESC LIMIT 1",
            params![job_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
    })
}

fn list_decisions(db: &Database, job_id: Option<&str>, limit: usize) -> rusqlite::Result<Vec<Decision>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT d.id, d.run_id, d.symbol, d.decision, d.note, d.price, d.decided_at, COALESCE(r.picks_json, '[]'), COALESCE(r.job_name, '')
             FROM assistant_decisions d LEFT JOIN assistant_runs r ON r.id = d.run_id
             WHERE (?1 IS NULL OR r.job_id = ?1) ORDER BY d.decided_at DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![job_id, limit as i64], |row| {
            let symbol: String = row.get(2)?;
            let picks: Vec<Pick> = serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default();
            Ok(Decision {
                id: row.get(0)?,
                run_id: row.get(1)?,
                action: picks.iter().find(|p| p.symbol == symbol).map(|p| p.action.clone()).unwrap_or_default(),
                symbol,
                decision: row.get(3)?,
                note: row.get(4)?,
                price: row.get(5)?,
                decided_at: row.get(6)?,
                job_name: row.get(8)?,
                current: None,
            })
        })?;
        rows.collect()
    })
}

// ---------- saved table rows ----------

fn known_keys(db: &Database, job_id: &str) -> rusqlite::Result<HashSet<String>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare("SELECT key FROM assistant_rows WHERE job_id = ?1")?;
        let rows = stmt.query_map(params![job_id], |r| r.get(0))?;
        rows.collect()
    })
}

/// Every row a job has saved, oldest first.
fn saved_rows(db: &Database, job_id: &str) -> rusqlite::Result<Vec<Vec<String>>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare("SELECT row_json FROM assistant_rows WHERE job_id = ?1 ORDER BY added_at, rowid")?;
        let rows = stmt.query_map(params![job_id], |r| r.get::<_, String>(0))?;
        Ok(rows.filter_map(Result::ok).filter_map(|j| serde_json::from_str(&j).ok()).collect())
    })
}

fn save_rows(db: &Database, job_id: &str, run_id: &str, rows: &[Vec<String>]) -> rusqlite::Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    db.with_conn(|conn| {
        for row in rows {
            conn.execute(
                "INSERT OR IGNORE INTO assistant_rows (job_id, key, row_json, run_id, added_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![job_id, table::key(row), to_json(row), run_id, now],
            )?;
        }
        Ok(())
    })
}

// ---------- notes ----------

fn list_notes(db: &Database, job_id: &str) -> rusqlite::Result<Vec<Note>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare("SELECT id, job_id, text, created_at FROM assistant_notes WHERE job_id = ?1 ORDER BY created_at DESC, rowid DESC")?;
        let rows = stmt.query_map(params![job_id], |r| Ok(Note { id: r.get(0)?, job_id: r.get(1)?, text: r.get(2)?, created_at: r.get(3)? }))?;
        rows.collect()
    })
}

/// Adds a run's notes and keeps the newest MAX_NOTES.
fn add_notes(db: &Database, job_id: &str, run_id: &str, notes: &[String]) -> rusqlite::Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    db.with_conn(|conn| {
        for text in notes {
            conn.execute("INSERT INTO assistant_notes (id, job_id, run_id, text, created_at) VALUES (?1, ?2, ?3, ?4, ?5)", params![Uuid::new_v4().to_string(), job_id, run_id, text, now])?;
        }
        conn.execute(
            "DELETE FROM assistant_notes WHERE job_id = ?1 AND id NOT IN (SELECT id FROM assistant_notes WHERE job_id = ?1 ORDER BY created_at DESC, rowid DESC LIMIT ?2)",
            params![job_id, MAX_NOTES as i64],
        )?;
        Ok(())
    })
}

// ---------- scheduling ----------

fn parse_time(time: &str) -> Option<NaiveTime> {
    NaiveTime::parse_from_str(time.trim(), "%H:%M").ok()
}

/// Whether a scheduled job should start now: its time has passed today, on a
/// day it runs, and it has not run today. A laptop asleep at 7:30 runs the
/// job when it wakes, the same day.
fn is_due(job: &Job, now: chrono::DateTime<Local>) -> bool {
    let Some(at) = job.time.as_deref().and_then(parse_time) else { return false };
    if !job.enabled {
        return false;
    }
    if job.weekdays_only && matches!(now.weekday(), chrono::Weekday::Sat | chrono::Weekday::Sun) {
        return false;
    }
    let today = now.date_naive().to_string();
    now.time() >= at && job.last_run_on.as_deref() != Some(today.as_str())
}

// ---------- the assistant ----------

#[derive(Clone)]
pub struct Assistant {
    llm: Arc<LlmEngine>,
    db: Database,
    client: reqwest::Client,
    /// Where tables are written: data/assistant/<job>/<date>.csv.
    data_dir: PathBuf,
    /// One run at a time; the model queue would serialize them anyway, and
    /// this keeps a queued run's status honest.
    one_at_a_time: Arc<tokio::sync::Mutex<()>>,
}

impl Assistant {
    pub fn new(llm: Arc<LlmEngine>, db: Database, client: reqwest::Client, data_dir: PathBuf) -> Self {
        Self { llm, db, client, data_dir, one_at_a_time: Default::default() }
    }

    /// Checks the schedule every 30 seconds.
    pub fn spawn_scheduler(&self) {
        let assistant = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                let now = Local::now();
                let jobs = match list_jobs(&assistant.db) {
                    Ok(jobs) => jobs,
                    Err(error) => {
                        tracing::warn!("assistant: cannot read jobs: {error}");
                        continue;
                    }
                };
                if let Err(error) = close_stale_runs(&assistant.db) {
                    tracing::warn!("assistant: cannot close stale runs: {error}");
                }
                let today = now.date_naive().to_string();
                for mut job in jobs.into_iter().filter(|job| is_due(job, now)) {
                    match claim_today(&assistant.db, &job.id, &today) {
                        Ok(true) => job.last_run_on = Some(today.clone()),
                        Ok(false) => continue,
                        Err(error) => {
                            tracing::warn!("assistant: cannot mark '{}' as run: {error}", job.name);
                            continue;
                        }
                    }
                    tracing::info!("assistant: starting scheduled job '{}'", job.name);
                    if let Err(error) = assistant.start(job, "schedule").await {
                        tracing::warn!("assistant: {error}");
                    }
                }
            }
        });
    }

    /// Saves a queued run and works on it in the background.
    pub async fn start(&self, job: Job, trigger: &str) -> Result<Run, String> {
        let model = self.choose_model(job.model.as_deref()).await?;
        let run = Run {
            id: Uuid::new_v4().to_string(),
            job_id: job.id.clone(),
            job_name: job.name.clone(),
            status: "queued".into(),
            trigger: trigger.into(),
            model,
            started_at: chrono::Utc::now().to_rfc3339(),
            finished_at: None,
            report: None,
            sources: vec![],
            steps: vec![],
            picks: vec![],
            error: None,
            table: None,
            reminders: vec![],
        };
        save_run(&self.db, &run).map_err(|e| format!("Cannot save the run: {e}"))?;
        let assistant = self.clone();
        let queued = run.clone();
        tokio::spawn(async move {
            let _turn = assistant.one_at_a_time.lock().await;
            let mut run = queued;
            run.status = "running".into();
            run.started_at = chrono::Utc::now().to_rfc3339();
            let _ = save_run(&assistant.db, &run);
            let outcome = assistant.work(&job, &mut run).await;
            run.finished_at = Some(chrono::Utc::now().to_rfc3339());
            match outcome {
                Ok(()) => run.status = "completed".into(),
                Err(error) => {
                    run.status = "failed".into();
                    run.error = Some(error);
                }
            }
            if let Err(error) = save_run(&assistant.db, &run) {
                tracing::error!("assistant: cannot save run {}: {error}", run.id);
            }
            notify(&run);
            assistant.send_to_telegram(&run).await;
            assistant.send_by_email(&run).await;
        });
        Ok(run)
    }

    /// Sends a finished run to the user's Telegram chat, when one is set.
    async fn send_to_telegram(&self, run: &Run) {
        let (Some(token), Some(chat)) = telegram::settings(&self.db) else { return };
        if let Err(error) = telegram::send_text(&self.client, &token, &chat, &telegram::message(run)).await {
            tracing::warn!("assistant: Telegram message failed: {error}");
            return;
        }
        if let Some(t) = run.table.as_ref().filter(|t| !t.rows.is_empty()) {
            let name = format!("{}-{}.csv", table::slug(&run.job_name), run.started_at.get(..10).unwrap_or("run"));
            let caption = format!("{}: {} new rows", run.job_name, t.rows.len());
            if let Err(error) = telegram::send_file(&self.client, &token, &chat, &name, table::to_csv(&t.columns, &t.rows).into_bytes(), &caption).await {
                tracing::warn!("assistant: Telegram file failed: {error}");
            }
        }
    }

    /// Emails a finished run, when email is set up.
    async fn send_by_email(&self, run: &Run) {
        let Some(settings) = email::load(&self.db) else { return };
        let (subject, body) = email::report(run);
        let csv = run.table.as_ref().filter(|t| !t.rows.is_empty()).map(|t| {
            (format!("{}-{}.csv", table::slug(&run.job_name), run.started_at.get(..10).unwrap_or("run")), table::to_csv(&t.columns, &t.rows).into_bytes())
        });
        if let Err(error) = email::send(&settings, &subject, &body, csv).await {
            tracing::warn!("assistant: email failed: {error}");
        }
    }

    async fn choose_model(&self, requested: Option<&str>) -> Result<String, String> {
        if let Some(id) = requested.filter(|id| crate::hosted::is_hosted(id)) {
            return if crate::hosted::available(id) {
                Ok(id.to_string())
            } else {
                Err(format!("'{id}' is not set up. Check its key in the API models tab."))
            };
        }
        let models = self.llm.models().await?;
        let chat: Vec<_> = models
            .iter()
            .filter(|m| m.installed && m.capabilities.iter().any(|c| c == "chat") && !m.is_image_generator && !m.capabilities.iter().any(|c| c == "embedding"))
            .collect();
        if let Some(id) = requested.filter(|id| !id.is_empty()) {
            return chat
                .iter()
                .find(|m| m.id == id)
                .map(|m| m.id.clone())
                .ok_or_else(|| format!("'{id}' is not an installed chat model. Pick another in the job's settings."));
        }
        ["qwen3.6:35b-a3b", "inkling-2-mini:26b", "devstral-small-2:24b", "gemma4:31b", "qwen3.5:9b"]
            .iter()
            .find_map(|id| chat.iter().find(|m| m.id == *id))
            .or_else(|| chat.first())
            .map(|m| m.id.clone())
            .ok_or_else(|| "No chat model is installed. Download Qwen3.5 9B in Settings → Models.".to_string())
    }

    /// The research loop: the model calls tools until it writes the report,
    /// or runs out of turns and is asked to write up what it has.
    async fn work(&self, job: &Job, run: &mut Run) -> Result<(), String> {
        let ctx = self.llm.context_size() as usize;
        let options = ChatOptions { thinking: false, effort: Effort::Medium, temperature: Temperature::Precise };
        let final_options = ChatOptions { effort: if ctx >= 24_000 { Effort::High } else { Effort::Medium }, ..options };
        // About three characters a token, less room for the answer and the tools.
        let budget = ctx.saturating_sub(final_options.max_tokens(false) as usize + 1_500).max(2_000) * 3;
        let result_cap = (budget / 3).clamp(1_500, SEARCH_CHARS);

        let mut messages = vec![
            json!({"role": "system", "content": self.system_prompt()}),
            json!({"role": "user", "content": self.task_prompt(job)}),
        ];
        let mut tools = Tools { client: &self.client, sources: vec![], quotes: HashMap::new(), cap: result_cap, notes: vec![], reminders: vec![] };
        let columns = job.columns();
        let known = if columns.is_empty() { HashSet::new() } else { known_keys(&self.db, &job.id).map_err(|e| format!("Cannot read saved rows: {e}"))? };
        let mut calls_made = 0usize;
        let mut asked = Asked::default();
        let mut revisions = 0;
        let mut wrapped = false;
        let mut report = None;
        // Research turns, then up to three follow-ups (citations, the
        // table, repeats), each with a turn of its own.
        for turn_index in 0..MAX_TURNS + 6 {
            let wrap_up = turn_index >= MAX_TURNS || calls_made >= MAX_TOOL_CALLS;
            if wrap_up && !wrapped {
                wrapped = true;
                messages.push(json!({"role": "user", "content": "Research time is over. Write the final report now from what you found, in the format the job asks for. Do not call any more tools."}));
            }
            fit(&mut messages, budget);
            let opts = if wrap_up { final_options } else { options };
            let turn = self.turn(&run.model, &messages, opts).await?;
            let calls: Vec<ToolCall> = turn
                .tool_calls
                .into_iter()
                .enumerate()
                .map(|(i, mut c)| {
                    if c.id.is_empty() {
                        c.id = format!("call_{turn_index}_{i}");
                    }
                    c
                })
                .collect();
            if calls.is_empty() || wrap_up {
                if turn.content.trim().is_empty() {
                    if wrapped && revisions > 0 {
                        break;
                    }
                    messages.push(json!({"role": "user", "content": "That reply was empty. Continue: call a tool, or write the final report."}));
                    continue;
                }
                let more_research = calls_made < MAX_TOOL_CALLS;
                match (revisions < 3).then(|| follow_up(&turn.content, &tools.sources, &columns, &known, &mut asked, more_research)).flatten() {
                    Some(ask) => {
                        revisions += 1;
                        report = Some(turn.content.clone());
                        messages.push(json!({"role": "assistant", "content": turn.content}));
                        messages.push(json!({"role": "user", "content": ask}));
                        continue;
                    }
                    None => {
                        report = Some(turn.content);
                        break;
                    }
                }
            }
            let calls: Vec<ToolCall> = calls.into_iter().take(MAX_CALLS_PER_TURN).collect();
            messages.push(json!({
                "role": "assistant",
                "content": turn.content,
                "tool_calls": calls.iter().map(|c| json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}})).collect::<Vec<_>>()
            }));
            for call in &calls {
                calls_made += 1;
                let started = Instant::now();
                let (output, step) = tools.run(call).await;
                run.steps.push(Step { ms: started.elapsed().as_millis() as u64, ..step });
                messages.push(json!({"role": "tool", "tool_call_id": call.id, "content": output}));
            }
            run.sources = tools.sources.clone();
            let _ = save_run(&self.db, run);
        }
        // The last report written wins, even if a follow-up went unanswered.
        let report = report.filter(|r| !r.trim().is_empty()).ok_or("The model finished without writing a report. Try a larger model or simpler instructions.")?;
        let (text, mut picks) = split_picks(&report);
        let text = if columns.is_empty() {
            text
        } else {
            let (text, rows) = table::extract(&text, &columns);
            let (rows, repeats) = table::split_new(rows.unwrap_or_default(), &known);
            save_rows(&self.db, &job.id, &run.id, &rows).map_err(|e| format!("Cannot save the table: {e}"))?;
            if !rows.is_empty() {
                self.write_files(job, &columns, &rows);
            }
            run.table = Some(table::Table { columns: columns.clone(), rows, repeats });
            text
        };
        // Every pick carries the price when the report was written.
        let missing: Vec<String> = picks.iter().filter(|p| !tools.quotes.contains_key(&p.symbol)).map(|p| p.symbol.clone()).collect();
        for symbol in missing.into_iter().take(15) {
            if let Ok(quote) = quote(&self.client, &symbol).await {
                tools.quotes.insert(symbol, quote);
            }
        }
        for pick in &mut picks {
            pick.price = tools.quotes.get(&pick.symbol).map(|q| q.price);
        }
        if !tools.notes.is_empty() {
            add_notes(&self.db, &job.id, &run.id, &tools.notes).map_err(|e| format!("Cannot save notes: {e}"))?;
        }
        run.report = Some(text);
        run.picks = picks;
        run.sources = tools.sources;
        run.reminders = tools.reminders;
        Ok(())
    }

    /// data/assistant/<job>/<date>.csv with this run's new rows, and
    /// all.csv with every row the job has saved. A failure here is logged:
    /// the rows are in the database and can be downloaded from the page.
    fn write_files(&self, job: &Job, columns: &[String], rows: &[Vec<String>]) {
        let folder = self.data_dir.join("assistant").join(table::slug(&job.name));
        let write = || -> std::io::Result<PathBuf> {
            std::fs::create_dir_all(&folder)?;
            let now = Local::now();
            let mut path = folder.join(format!("{}.csv", now.format("%Y-%m-%d")));
            if path.exists() {
                path = folder.join(format!("{}.csv", now.format("%Y-%m-%d-%H%M%S")));
            }
            std::fs::write(&path, table::to_csv(columns, rows))?;
            let all = saved_rows(&self.db, &job.id).unwrap_or_default();
            std::fs::write(folder.join("all.csv"), table::to_csv(columns, &all))?;
            Ok(path)
        };
        match write() {
            Ok(path) => tracing::info!("assistant: saved {} rows to {}", rows.len(), path.display()),
            Err(error) => tracing::warn!("assistant: cannot write the CSV for '{}': {error}", job.name),
        }
    }

    async fn turn(&self, model: &str, messages: &[Value], options: ChatOptions) -> Result<crate::llm::ToolTurn, String> {
        let attempt = self.llm.chat_tools(model, messages.to_vec(), tool_schemas(), options, None).await;
        match attempt {
            Err(error) if error.contains(REASONING_EXHAUSTED) => {
                let retry = ChatOptions { thinking: false, effort: Effort::High, ..options };
                self.llm.chat_tools(model, messages.to_vec(), tool_schemas(), retry, None).await
            }
            other => other,
        }
    }

    fn system_prompt(&self) -> String {
        let now = Local::now();
        let standing = crate::standing_instructions(&self.db).map(|t| format!("\n\n{t}")).unwrap_or_default();
        format!(
            "You are the user's personal research assistant inside Ferro64, working on a standing job while they are away. \
Now: {} (local time).

How to work:
- Research with the tools before you write. Search for what is new, then read the pages that matter. Several focused searches beat one broad one.
- Prefer primary and reputable sources: company filings and investor relations, exchange and regulator pages, established news outlets. Treat forums, social posts and promotional sites as rumours.
- Check dates. Say when news is old, and do not present it as new.
- Every price, change or percentage you state must come from stock_quote or price_history in this run. Never state a figure from memory.
- Cite sources inline as [S1], [S2], using the numbers the tools give. Do not invent sources or numbers. Prices and figures from stock_quote or price_history are not sources: write them plainly, without [S#].
- When evidence is thin or sources disagree, say so plainly. \"No clear opportunity today\" is a valid finding.
- save_note keeps a lesson for the next run of this job; suggest_reminder offers the user a calendar reminder. Use each only when it really helps.
- You cannot trade, buy, sell or contact anyone. You write a report; the user decides and acts. Never say you have done anything on their behalf.

When you are done researching, reply with the report itself and no tool call. Write in Markdown with short sections.
If the job involves stocks, funds or other securities, end the report with a fenced ```json block holding your picks, exactly in this shape:
{{\"picks\": [{{\"symbol\": \"TICKER\", \"action\": \"buy|hold|sell|watch|avoid\", \"conviction\": \"low|medium|high\", \"thesis\": \"one or two sentences\", \"risks\": \"the main risk\", \"sources\": [\"S1\"]}}]}}
Include only tickers you researched in this run. The block is read by the app; the user sees the picks as cards.{standing}",
            now.format("%A %-d %B %Y, %H:%M")
        )
    }

    fn task_prompt(&self, job: &Job) -> String {
        let mut prompt = format!("JOB: {}\n\n{}", job.name, job.instructions.trim());
        let columns = job.columns();
        if !columns.is_empty() {
            prompt.push_str(&format!(
                "\n\nTABLE: end the report with a fenced ```csv block. Its first line is exactly:\n{}\nOne row per item. Quote a field that contains a comma. Fill a field only from a page you read in this run; leave it empty if you could not find it, and never guess a phone number, address or name. Rows are matched on the first two columns, so give those in full.",
                columns.join(",")
            ));
            // The newest saved rows, as many as fit, so the model can skip them.
            let saved = saved_rows(&self.db, &job.id).unwrap_or_default();
            if !saved.is_empty() {
                let mut list = String::new();
                let mut shown = 0;
                for row in saved.iter().rev() {
                    let line = format!("- {}\n", row.iter().take(2).cloned().collect::<Vec<_>>().join(" | "));
                    if list.len() + line.len() > SAVED_LIST_CHARS {
                        break;
                    }
                    list.push_str(&line);
                    shown += 1;
                }
                let more = if shown < saved.len() { format!(" (the newest {shown} of {})", saved.len()) } else { String::new() };
                prompt.push_str(&format!("\n\nALREADY SAVED BY EARLIER RUNS{more}. Do not repeat these; find new ones. Repeats are dropped:\n{list}"));
            }
        }
        if let Ok(notes) = list_notes(&self.db, &job.id) {
            if !notes.is_empty() {
                prompt.push_str("\n\nYOUR NOTES FROM EARLIER RUNS (written by you; hints to check, not facts):\n");
                for n in notes.iter().take(MAX_NOTES) {
                    prompt.push_str(&format!("- {}\n", n.text));
                }
            }
        }
        if let Ok(Some((at, report))) = last_report(&self.db, &job.id) {
            let (text, _) = split_picks(&report);
            prompt.push_str(&format!(
                "\n\nYOUR PREVIOUS REPORT ({}), for continuity only. Prices and news have changed since; check again before repeating anything:\n{}",
                at.get(..10).unwrap_or(&at),
                clip(&text, PREVIOUS_REPORT_CHARS)
            ));
        }
        if let Ok(decisions) = list_decisions(&self.db, Some(&job.id), 15) {
            if !decisions.is_empty() {
                prompt.push_str("\n\nWHAT THE USER DID WITH YOUR EARLIER PICKS (most recent first):\n");
                for d in decisions {
                    let price = d.price.map(|p| format!(" at ${p:.2}")).unwrap_or_default();
                    let note = if d.note.trim().is_empty() { String::new() } else { format!(" — \"{}\"", clip(d.note.trim(), 200)) };
                    prompt.push_str(&format!("- {} {}: {}{}{}\n", d.decided_at.get(..10).unwrap_or(&d.decided_at), d.symbol, d.decision, price, note));
                }
                prompt.push_str("Take these into account: positions they hold, ideas they passed on and why.");
            }
        }
        prompt
    }
}

#[derive(Default)]
struct Asked {
    citations: bool,
    table: bool,
    repeats: bool,
}

/// What to ask for once more before accepting a report: citations, the
/// table the job wants, or new rows in place of ones saved before.
fn follow_up(content: &str, sources: &[SourceRef], columns: &[String], known: &HashSet<String>, asked: &mut Asked, more_research: bool) -> Option<String> {
    if !sources.is_empty() && !content.contains("[S") && !asked.citations {
        asked.citations = true;
        return Some("Rewrite the whole report with inline citations: put [S#] after every claim that comes from a source, using the numbers the tools gave. Drop any claim you cannot cite. Keep any picks or csv block at the end. Do not call tools.".into());
    }
    if columns.is_empty() {
        return None;
    }
    let (_, rows) = table::extract(content, columns);
    let Some(rows) = rows else {
        if asked.table {
            return None;
        }
        asked.table = true;
        return Some(format!("The report has no table. Write the whole report again and end it with a fenced ```csv block whose first line is exactly: {}", columns.join(",")));
    };
    let (_, repeats) = table::split_new(rows, known);
    if repeats.is_empty() || asked.repeats {
        return None;
    }
    asked.repeats = true;
    let list: Vec<String> = repeats.iter().take(20).map(|r| r.iter().take(2).cloned().collect::<Vec<_>>().join(" | ")).collect();
    Some(format!(
        "{} of your rows were already saved by earlier runs, so they would be dropped:\n- {}\n{}Then write the whole report again with only new rows in the csv block.",
        repeats.len(),
        list.join("\n- "),
        if more_research { "Research replacements for them with the tools. " } else { "Replace them with other rows you found, if any; do not invent rows. " }
    ))
}

/// Keeps the conversation within the context window by shortening the
/// oldest tool results first. The last few stay whole.
fn fit(messages: &mut [Value], budget: usize) {
    let size = |m: &Value| m["content"].as_str().map_or(0, str::len) + m.get("tool_calls").map_or(0, |c| c.to_string().len());
    let mut total: usize = messages.iter().map(size).sum();
    let tool_positions: Vec<usize> = (0..messages.len()).filter(|&i| messages[i]["role"] == "tool").collect();
    let keep_whole = 3;
    for &i in tool_positions.iter().rev().skip(keep_whole).rev() {
        if total <= budget {
            return;
        }
        let content = messages[i]["content"].as_str().unwrap_or("").to_string();
        if content.len() <= 500 {
            continue;
        }
        let short = format!("{} … [shortened to save room; the source numbers still hold]", clip(&content, 400));
        total -= content.len() - short.len();
        messages[i]["content"] = Value::from(short);
    }
}

// ---------- tools ----------

fn tool_schemas() -> Value {
    let tool = |name: &str, description: &str, properties: Value, required: &[&str]| {
        json!({"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}})
    };
    json!([
        tool("web_search", "Search the web. Returns numbered sources [S#] with titles, dates and text. Use specific queries, e.g. 'NVDA earnings guidance November 2026'.",
             json!({"query": {"type": "string"}}), &["query"]),
        tool("read_page", "Read the text of one public web page, e.g. a source from web_search worth reading in full.",
             json!({"url": {"type": "string"}}), &["url"]),
        tool("stock_quote", "Current quote for US-listed stocks or ETFs: last price, day change, day range, 52-week range. Up to 10 tickers.",
             json!({"symbols": {"type": "array", "items": {"type": "string"}, "description": "Tickers such as AAPL, MSFT, VOO"}}), &["symbols"]),
        tool("price_history", "Daily price history for one US-listed stock or ETF, summarized: change over 1 week, 1 month and the period, high and low, and 20/50/200-day averages.",
             json!({"symbol": {"type": "string"}, "days": {"type": "integer", "description": "Calendar days to look back, 30 to 365; default 180"}}), &["symbol"]),
        tool("save_note", "Keep a short note for future runs of this job: a lesson about where to look or what to avoid, e.g. 'Practo pages block reading; hospital sites work'. Not for news or prices, which go stale.",
             json!({"note": {"type": "string"}}), &["note"]),
        tool("suggest_reminder", "Suggest a calendar reminder for the user, e.g. to recheck a stock after its earnings date. It is only a suggestion shown under the report.",
             json!({"title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"}, "time": {"type": "string", "description": "HH:MM, optional"}, "note": {"type": "string"}}), &["title", "date"]),
    ])
}

struct Tools<'a> {
    client: &'a reqwest::Client,
    sources: Vec<SourceRef>,
    quotes: HashMap<String, Quote>,
    /// Longest tool result, from the context window.
    cap: usize,
    notes: Vec<String>,
    reminders: Vec<Reminder>,
}

impl Tools<'_> {
    async fn run(&mut self, call: &ToolCall) -> (String, Step) {
        let args: Value = serde_json::from_str(&call.arguments).unwrap_or(Value::Null);
        let text = |key: &str| args.get(key).and_then(Value::as_str).unwrap_or("").trim().to_string();
        let (input, result) = match call.name.as_str() {
            "web_search" => {
                let query = text("query");
                let result = if query.is_empty() { Err("Give a query".to_string()) } else { self.search(&query).await };
                (query, result)
            }
            "read_page" => {
                let url = text("url");
                let result = self.read(&url).await;
                (url, result)
            }
            "stock_quote" => {
                let symbols = symbols(&args);
                let result = self.quotes(&symbols).await;
                (symbols.join(", "), result)
            }
            "price_history" => {
                let symbol = text("symbol").to_uppercase();
                let days = args.get("days").and_then(Value::as_i64).unwrap_or(180).clamp(30, 365);
                let result = history(self.client, &symbol, days).await;
                (format!("{symbol}, {days} days"), result)
            }
            "save_note" => {
                let note = text("note");
                let result = self.note(&note);
                (note, result)
            }
            "suggest_reminder" => {
                let reminder = Reminder { title: text("title"), date: text("date"), time: Some(text("time")).filter(|t| !t.is_empty()), note: text("note") };
                let input = format!("{} on {}{}", reminder.title, reminder.date, reminder.time.as_deref().map(|t| format!(" {t}")).unwrap_or_default());
                let result = self.remind(reminder);
                (input, result)
            }
            other => (String::new(), Err(format!("There is no tool named '{other}'. Use web_search, read_page, stock_quote, price_history, save_note or suggest_reminder."))),
        };
        let ok = result.is_ok();
        let output = match result {
            Ok(text) => clip(&text, self.cap),
            Err(error) => format!("Error: {error}"),
        };
        let summary = clip(output.lines().next().unwrap_or(""), 160);
        (output, Step { tool: call.name.clone(), input, summary, ok, ms: 0 })
    }

    fn note(&mut self, note: &str) -> Result<String, String> {
        if note.is_empty() {
            return Err("Write the note".into());
        }
        if self.notes.len() >= NOTES_PER_RUN {
            return Err(format!("At most {NOTES_PER_RUN} notes a run; keep the most useful"));
        }
        self.notes.push(clip(note, NOTE_CHARS));
        Ok("Saved for future runs of this job (once this run finishes). The user can read and delete it.".into())
    }

    fn remind(&mut self, reminder: Reminder) -> Result<String, String> {
        if reminder.title.is_empty() {
            return Err("Give the reminder a title".into());
        }
        let date = chrono::NaiveDate::parse_from_str(&reminder.date, "%Y-%m-%d").map_err(|_| "date must look like 2026-12-03")?;
        let today = Local::now().date_naive();
        if date < today || date > today + chrono::Duration::days(400) {
            return Err("The date must be between today and about a year from now".into());
        }
        if reminder.time.as_deref().is_some_and(|t| parse_time(t).is_none()) {
            return Err("time must look like 09:30, or be left out".into());
        }
        if self.reminders.len() >= REMINDERS_PER_RUN {
            return Err(format!("At most {REMINDERS_PER_RUN} reminders a run"));
        }
        self.reminders.push(Reminder { title: clip(&reminder.title, 120), note: clip(&reminder.note, 500), ..reminder });
        Ok("Suggested. It is shown under the report; nothing is added to the calendar unless the user chooses to.".into())
    }

    fn source(&mut self, title: &str, url: &str, published: Option<String>) -> usize {
        if let Some(existing) = self.sources.iter().find(|s| s.url == url) {
            return existing.n;
        }
        let n = self.sources.len() + 1;
        self.sources.push(SourceRef { n, title: title.to_string(), url: url.to_string(), published });
        n
    }

    async fn search(&mut self, query: &str) -> Result<String, String> {
        let found = websearch::research(self.client, query).await?;
        let mut out = format!("{} results for \"{query}\":\n\n", found.len());
        let per = (self.cap / found.len().max(1)).clamp(300, 1_500);
        for s in &found {
            let date = s.published.as_deref().map(|d| d.get(..10).unwrap_or(d).to_string());
            let n = self.source(&s.title, &s.url, date.clone());
            let body = if s.text.len() > s.snippet.len() { &s.text } else { &s.snippet };
            out.push_str(&format!("[S{n}] {}\n{}{}\n{}\n\n", s.title, s.url, date.map(|d| format!(" · published {d}")).unwrap_or_default(), clip(body, per)));
        }
        Ok(out)
    }

    async fn read(&mut self, url: &str) -> Result<String, String> {
        let max = PAGE_CHARS.min(self.cap);
        let mut text = websearch::fetch_page(self.client, url, max).await?;
        // A page built by JavaScript (many directories are) arrives nearly
        // empty; render it in a headless browser. Only after the plain
        // fetch reached a public page without being redirected elsewhere.
        if text.trim().len() < 400 {
            if let Some(rendered) = render_public(url).await.filter(|t| t.len() > text.trim().len()) {
                text = clip(&rendered, max);
            }
        }
        if text.trim().len() < 80 {
            return Err("The page had no readable text (it may block automated reading)".into());
        }
        let title = url::Url::parse(url).ok().and_then(|u| u.host_str().map(str::to_string)).unwrap_or_else(|| url.to_string());
        let n = self.source(&title, url, None);
        Ok(format!("[S{n}] {url}\n\n{text}"))
    }

    async fn quotes(&mut self, symbols: &[String]) -> Result<String, String> {
        if symbols.is_empty() {
            return Err("Give one or more tickers".into());
        }
        let lookups = symbols.iter().map(|s| quote(self.client, s));
        let results = futures_util::future::join_all(lookups).await;
        let mut out = String::from("Quotes from Nasdaq's public data (may be delayed):\n");
        for (symbol, result) in symbols.iter().zip(results) {
            match result {
                Ok(q) => {
                    out.push_str(&q.line());
                    self.quotes.insert(symbol.clone(), q);
                }
                Err(error) => out.push_str(&format!("{symbol}: no quote ({error})\n")),
            }
        }
        Ok(out)
    }
}

/// The text of a public page after its scripts ran, from a Chrome, Edge or
/// Brave already on this machine, with a throwaway profile (no cookies or
/// sign-ins) and local names unreachable. None when there is no browser or
/// it fails.
async fn render_public(url: &str) -> Option<String> {
    let browser = crate::coder::find_browser()?;
    let profile = std::env::temp_dir().join(format!("ferro64-read-{}", Uuid::new_v4().simple()));
    let mut command = tokio::process::Command::new(browser);
    command
        .args(["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--mute-audio"])
        .arg("--host-resolver-rules=MAP localhost ~NOTFOUND, MAP *.localhost ~NOTFOUND, MAP *.local ~NOTFOUND, MAP *.internal ~NOTFOUND")
        .arg(format!("--user-data-dir={}", profile.display()))
        .args(["--virtual-time-budget=8000", "--dump-dom", url])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);
    let child = command.spawn().ok()?;
    let pid = child.id();
    let result = tokio::time::timeout(Duration::from_secs(30), child.wait_with_output()).await;
    if result.is_err() {
        if let Some(pid) = pid {
            crate::coder::kill_tree(pid);
        }
    }
    let _ = std::fs::remove_dir_all(&profile);
    let output = result.ok()?.ok()?;
    let text = crate::coder::visible_text(&String::from_utf8_lossy(&output.stdout));
    (!text.trim().is_empty()).then_some(text)
}

fn symbols(args: &Value) -> Vec<String> {
    let raw: Vec<String> = match args.get("symbols") {
        Some(Value::Array(list)) => list.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        Some(Value::String(text)) => text.split([',', ' ']).map(str::to_string).collect(),
        _ => args.get("symbol").and_then(Value::as_str).map(|s| vec![s.to_string()]).unwrap_or_default(),
    };
    let mut out: Vec<String> = raw.into_iter().map(|s| s.trim().trim_start_matches('$').to_uppercase()).filter(|s| valid_symbol(s)).collect();
    out.dedup();
    out.truncate(10);
    out
}

fn valid_symbol(symbol: &str) -> bool {
    (1..=10).contains(&symbol.len()) && symbol.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
}

// ---------- market data (Nasdaq's public quote API, no key) ----------

#[derive(Debug, Clone)]
struct Quote {
    symbol: String,
    name: String,
    price: f64,
    change: String,
    percent: String,
    at: String,
    day_range: String,
    year_range: String,
    market: String,
}

impl Quote {
    fn line(&self) -> String {
        format!(
            "{} ({}): ${:.2}, {} ({}) · as of {} · day {} · 52-week {} · market {}\n",
            self.symbol, self.name, self.price, self.change, self.percent, self.at, self.day_range, self.year_range, self.market
        )
    }
}

fn money(text: &str) -> Option<f64> {
    text.trim().trim_start_matches('$').replace(',', "").parse().ok()
}

async fn nasdaq(client: &reqwest::Client, symbol: &str, path: &str, query: &[(&str, String)]) -> Result<Value, String> {
    if !valid_symbol(symbol) {
        return Err(format!("'{symbol}' is not a ticker"));
    }
    // Funds are a different asset class at the same address.
    for class in ["stocks", "etf"] {
        let mut params: Vec<(&str, String)> = vec![("assetclass", class.to_string())];
        params.extend(query.iter().cloned());
        let response = client
            .get(format!("https://api.nasdaq.com/api/quote/{symbol}/{path}"))
            .query(&params)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| format!("Nasdaq is unreachable: {e}"))?;
        if !response.status().is_success() {
            return Err(format!("Nasdaq returned {}", response.status()));
        }
        let body: Value = response.json().await.map_err(|e| format!("Nasdaq sent an unreadable reply: {e}"))?;
        if body.get("data").is_some_and(|d| !d.is_null()) {
            return Ok(body["data"].clone());
        }
    }
    Err("not found on Nasdaq (US listings only)".into())
}

async fn quote(client: &reqwest::Client, symbol: &str) -> Result<Quote, String> {
    let data = nasdaq(client, symbol, "info", &[]).await?;
    let s = |pointer: &str| data.pointer(pointer).and_then(Value::as_str).unwrap_or("").to_string();
    let price = money(&s("/primaryData/lastSalePrice")).ok_or("no price in the reply")?;
    Ok(Quote {
        symbol: symbol.to_string(),
        name: s("/companyName"),
        price,
        change: s("/primaryData/netChange"),
        percent: s("/primaryData/percentageChange"),
        at: s("/primaryData/lastTradeTimestamp"),
        day_range: s("/keyStats/dayrange/value"),
        year_range: s("/keyStats/fiftyTwoWeekHighLow/value"),
        market: s("/marketStatus"),
    })
}

/// A summary the model can quote: the arithmetic is done here.
async fn history(client: &reqwest::Client, symbol: &str, days: i64) -> Result<String, String> {
    if symbol.is_empty() {
        return Err("Give a ticker".into());
    }
    let from = (Local::now().date_naive() - chrono::Duration::days(days)).to_string();
    let data = nasdaq(client, symbol, "historical", &[("fromdate", from), ("limit", "9999".into())]).await?;
    let rows = data.pointer("/tradesTable/rows").and_then(Value::as_array).ok_or("no price history in the reply")?;
    // Newest first in the reply; oldest first here.
    let mut closes: Vec<(String, f64)> = rows
        .iter()
        .filter_map(|r| Some((r.get("date")?.as_str()?.to_string(), money(r.get("close")?.as_str()?)?)))
        .collect();
    closes.reverse();
    Ok(summarize(symbol, &closes))
}

fn summarize(symbol: &str, closes: &[(String, f64)]) -> String {
    let Some((last_date, last)) = closes.last().cloned() else {
        return format!("{symbol}: no trading days in that period");
    };
    let pct = |from: f64| (last / from - 1.0) * 100.0;
    let back = |n: usize| closes.len().checked_sub(n + 1).map(|i| closes[i].1);
    let avg = |n: usize| (closes.len() >= n).then(|| closes[closes.len() - n..].iter().map(|c| c.1).sum::<f64>() / n as f64);
    let (hi_date, hi) = closes.iter().cloned().fold((String::new(), f64::MIN), |a, c| if c.1 > a.1 { c } else { a });
    let (lo_date, lo) = closes.iter().cloned().fold((String::new(), f64::MAX), |a, c| if c.1 < a.1 { c } else { a });
    let mut out = format!("{symbol} daily closes, {} trading days from {} to {last_date}:\n", closes.len(), closes[0].0);
    out.push_str(&format!("last close ${last:.2}\n"));
    for (label, n) in [("1 week", 5), ("1 month", 21), ("3 months", 63)] {
        if let Some(then) = back(n) {
            out.push_str(&format!("change over {label}: {:+.1}%\n", pct(then)));
        }
    }
    out.push_str(&format!("change over the period: {:+.1}%\n", pct(closes[0].1)));
    out.push_str(&format!("period high ${hi:.2} on {hi_date}, low ${lo:.2} on {lo_date}; last close is {:.1}% below the high\n", (1.0 - last / hi) * 100.0));
    for n in [20, 50, 200] {
        if let Some(a) = avg(n) {
            out.push_str(&format!("{n}-day average ${a:.2} (last close {} it by {:.1}%)\n", if last >= a { "above" } else { "below" }, (last / a - 1.0).abs() * 100.0));
        }
    }
    let recent: Vec<String> = closes.iter().rev().take(10).rev().map(|(d, c)| format!("{d} ${c:.2}")).collect();
    out.push_str(&format!("last 10 closes: {}\n", recent.join(", ")));
    out
}

// ---------- the report ----------

/// Takes the fenced picks block off the end of a report and reads it.
/// A missing or broken block leaves the report as it is, with no picks.
fn split_picks(report: &str) -> (String, Vec<Pick>) {
    #[derive(Deserialize)]
    struct Doc {
        #[serde(default)]
        picks: Vec<Pick>,
    }
    let mut search_end = report.len();
    while let Some(open) = report[..search_end].rfind("```") {
        let before = &report[..open];
        let Some(start) = before.rfind("```") else { break };
        let inner = &report[start + 3..open];
        let body = inner.trim_start_matches("json").trim_start_matches("JSON").trim();
        if body.contains("\"picks\"") {
            if let Ok(doc) = serde_json::from_str::<Doc>(body) {
                let mut picks: Vec<Pick> = doc
                    .picks
                    .into_iter()
                    .map(|mut p| {
                        p.symbol = p.symbol.trim().trim_start_matches('$').to_uppercase();
                        p.action = p.action.trim().to_lowercase();
                        p.conviction = p.conviction.trim().to_lowercase();
                        p
                    })
                    .filter(|p| valid_symbol(&p.symbol))
                    .collect();
                picks.dedup_by(|a, b| a.symbol == b.symbol);
                let text = format!("{}{}", &report[..start], &report[open + 3..]);
                return (text.trim().to_string(), picks);
            }
        }
        search_end = start;
    }
    (report.trim().to_string(), vec![])
}

fn clip(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

/// A macOS notification when a run ends, so the report finds the user.
fn notify(run: &Run) {
    #[cfg(target_os = "macos")]
    {
        let text = match run.status.as_str() {
            "completed" if run.picks.is_empty() => "Your report is ready.".to_string(),
            "completed" => format!("Your report is ready: {} picks to review.", run.picks.len()),
            _ => format!("The run failed: {}", clip(run.error.as_deref().unwrap_or("unknown error"), 120)),
        };
        // Text goes in as arguments, never into the script itself.
        let _ = std::process::Command::new("osascript")
            .args(["-e", "on run argv", "-e", "display notification (item 1 of argv) with title (item 2 of argv)", "-e", "end run", &text, &format!("Ferro64 · {}", run.job_name)])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }
    #[cfg(not(target_os = "macos"))]
    let _ = run;
}

// ---------- HTTP ----------

pub fn routes(cfg: &mut web::ServiceConfig) {
    cfg.route("/api/assistant/jobs", web::get().to(jobs_handler))
        .route("/api/assistant/jobs", web::post().to(save_job_handler))
        .route("/api/assistant/jobs/{id}", web::delete().to(delete_job_handler))
        .route("/api/assistant/jobs/{id}/run", web::post().to(run_job_handler))
        .route("/api/assistant/runs", web::get().to(runs_handler))
        .route("/api/assistant/runs/{id}", web::get().to(run_handler))
        .route("/api/assistant/runs/{id}", web::delete().to(delete_run_handler))
        .route("/api/assistant/runs/{id}/csv", web::get().to(run_csv_handler))
        .route("/api/assistant/jobs/{id}/csv", web::get().to(job_csv_handler))
        .route("/api/assistant/jobs/{id}/notes", web::get().to(notes_handler))
        .route("/api/assistant/notes/{id}", web::delete().to(delete_note_handler))
        .route("/api/assistant/runs/{id}/reminders/{index}", web::get().to(reminder_handler))
        .route("/api/assistant/telegram", web::get().to(telegram_handler))
        .route("/api/assistant/telegram", web::put().to(telegram_save_handler))
        .route("/api/assistant/telegram/chats", web::post().to(telegram_chats_handler))
        .route("/api/assistant/telegram/test", web::post().to(telegram_test_handler))
        .route("/api/assistant/email", web::get().to(email_handler))
        .route("/api/assistant/email", web::put().to(email_save_handler))
        .route("/api/assistant/email", web::delete().to(email_delete_handler))
        .route("/api/assistant/email/test", web::post().to(email_test_handler))
        .route("/api/assistant/decisions", web::get().to(decisions_handler))
        .route("/api/assistant/decisions", web::post().to(decide_handler))
        .route("/api/assistant/decisions/{id}", web::delete().to(delete_decision_handler));
}

type Data = web::Data<Assistant>;

fn error(status: actix_web::http::StatusCode, message: impl std::fmt::Display) -> HttpResponse {
    HttpResponse::build(status).json(json!({"error": message.to_string()}))
}
fn bad(message: impl std::fmt::Display) -> HttpResponse {
    error(actix_web::http::StatusCode::BAD_REQUEST, message)
}
fn internal(message: impl std::fmt::Display) -> HttpResponse {
    error(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR, message)
}
fn missing(what: &str) -> HttpResponse {
    error(actix_web::http::StatusCode::NOT_FOUND, format!("{what} not found"))
}

async fn jobs_handler(a: Data) -> HttpResponse {
    let jobs = match list_jobs(&a.db) {
        Ok(jobs) => jobs,
        Err(e) => return internal(e),
    };
    let runs = list_runs(&a.db, None, 200).unwrap_or_default();
    let out: Vec<Value> = jobs
        .into_iter()
        .map(|job| {
            let last = runs.iter().find(|r| r.job_id == job.id);
            let rows = if job.columns().is_empty() { 0 } else { saved_rows(&a.db, &job.id).map(|r| r.len()).unwrap_or(0) };
            json!({"job": job, "rows": rows, "last_run": last.map(|r| json!({"id": r.id, "status": r.status, "started_at": r.started_at, "picks": r.picks.len()}))})
        })
        .collect();
    HttpResponse::Ok().json(out)
}

#[derive(Deserialize)]
struct JobInput {
    id: Option<String>,
    name: String,
    instructions: String,
    time: Option<String>,
    #[serde(default)]
    weekdays_only: bool,
    #[serde(default = "yes")]
    enabled: bool,
    model: Option<String>,
    table_columns: Option<String>,
}
fn yes() -> bool {
    true
}

async fn save_job_handler(a: Data, body: web::Json<JobInput>) -> HttpResponse {
    let input = body.into_inner();
    let name = input.name.trim().to_string();
    let instructions = input.instructions.trim().to_string();
    if name.is_empty() || name.chars().count() > 120 {
        return bad("Give the job a name of up to 120 characters");
    }
    if instructions.is_empty() || instructions.chars().count() > 20_000 {
        return bad("Write instructions of up to 20,000 characters");
    }
    let table_columns = input.table_columns.map(|c| table::parse_columns(&c)).filter(|c| !c.is_empty());
    if table_columns.as_ref().is_some_and(|c| c.len() < 2) {
        return bad("A table needs at least two columns; rows are matched on the first two");
    }
    let time = input.time.map(|t| t.trim().to_string()).filter(|t| !t.is_empty());
    if let Some(t) = &time {
        if parse_time(t).is_none() {
            return bad("Time must look like 07:30");
        }
    }
    let existing = match input.id.as_deref() {
        Some(id) => match get_job(&a.db, id) {
            Ok(Some(job)) => Some(job),
            Ok(None) => return missing("Job"),
            Err(e) => return internal(e),
        },
        None => None,
    };
    let now = Local::now();
    // A time already past today waits for the next day, rather than
    // starting a run the moment the job is saved.
    let passed = time.as_deref().and_then(parse_time).is_some_and(|t| now.time() >= t);
    let last_run_on = match (&existing, passed) {
        (Some(job), false) => job.last_run_on.clone().filter(|d| *d != now.date_naive().to_string()),
        (Some(job), true) => job.last_run_on.clone().or(Some(now.date_naive().to_string())),
        (None, true) => Some(now.date_naive().to_string()),
        (None, false) => None,
    };
    let job = Job {
        id: existing.as_ref().map(|j| j.id.clone()).unwrap_or_else(|| Uuid::new_v4().to_string()),
        name,
        instructions,
        time,
        weekdays_only: input.weekdays_only,
        enabled: input.enabled,
        model: input.model.filter(|m| !m.trim().is_empty()),
        created_at: existing.map(|j| j.created_at).unwrap_or_else(|| chrono::Utc::now().to_rfc3339()),
        last_run_on,
        table_columns: table_columns.map(|c| c.join(", ")),
    };
    match save_job(&a.db, &job) {
        Ok(()) => HttpResponse::Ok().json(job),
        Err(e) => internal(e),
    }
}

async fn delete_job_handler(a: Data, path: web::Path<String>) -> HttpResponse {
    let id = path.into_inner();
    let result = a.db.with_conn(|conn| {
        conn.execute("DELETE FROM assistant_decisions WHERE run_id IN (SELECT id FROM assistant_runs WHERE job_id = ?1)", params![id])?;
        conn.execute("DELETE FROM assistant_rows WHERE job_id = ?1", params![id])?;
        conn.execute("DELETE FROM assistant_notes WHERE job_id = ?1", params![id])?;
        conn.execute("DELETE FROM assistant_runs WHERE job_id = ?1 AND status NOT IN ('queued', 'running')", params![id])?;
        conn.execute("DELETE FROM assistant_jobs WHERE id = ?1", params![id])
    });
    match result {
        Ok(0) => missing("Job"),
        Ok(_) => HttpResponse::Ok().json(json!({"deleted": true})),
        Err(e) => internal(e),
    }
}

async fn run_job_handler(a: Data, path: web::Path<String>) -> HttpResponse {
    let job = match get_job(&a.db, &path.into_inner()) {
        Ok(Some(job)) => job,
        Ok(None) => return missing("Job"),
        Err(e) => return internal(e),
    };
    let busy = list_runs(&a.db, Some(&job.id), 5).unwrap_or_default().into_iter().any(|r| r.status == "queued" || r.status == "running");
    if busy {
        return bad("This job is already running");
    }
    match a.start(job, "manual").await {
        Ok(run) => HttpResponse::Ok().json(run),
        Err(e) => bad(e),
    }
}

#[derive(Deserialize)]
struct RunsQuery {
    job_id: Option<String>,
}

async fn runs_handler(a: Data, query: web::Query<RunsQuery>) -> HttpResponse {
    match list_runs(&a.db, query.job_id.as_deref(), 60) {
        // The list leaves out the long parts; a single run has them.
        Ok(runs) => HttpResponse::Ok().json(
            runs.into_iter()
                .map(|r| json!({"id": r.id, "job_id": r.job_id, "job_name": r.job_name, "status": r.status, "trigger": r.trigger, "model": r.model,
                    "started_at": r.started_at, "finished_at": r.finished_at, "picks": r.picks, "steps": r.steps.len(), "sources": r.sources.len(), "error": r.error,
                    "rows": r.table.as_ref().map(|t| t.rows.len())}))
                .collect::<Vec<_>>(),
        ),
        Err(e) => internal(e),
    }
}

async fn run_handler(a: Data, path: web::Path<String>) -> HttpResponse {
    let id = path.into_inner();
    let run = match get_run(&a.db, &id) {
        Ok(Some(run)) => run,
        Ok(None) => return missing("Run"),
        Err(e) => return internal(e),
    };
    let decisions = a
        .db
        .with_conn(|conn| {
            let mut stmt = conn.prepare("SELECT symbol, decision, note, id FROM assistant_decisions WHERE run_id = ?1")?;
            let rows = stmt.query_map(params![id], |r| Ok(json!({"symbol": r.get::<_, String>(0)?, "decision": r.get::<_, String>(1)?, "note": r.get::<_, String>(2)?, "id": r.get::<_, String>(3)?})))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
        })
        .unwrap_or_default();
    HttpResponse::Ok().json(json!({"run": run, "decisions": decisions}))
}

async fn delete_run_handler(a: Data, path: web::Path<String>) -> HttpResponse {
    let id = path.into_inner();
    let result = a.db.with_conn(|conn| {
        let n = conn.execute("DELETE FROM assistant_runs WHERE id = ?1 AND status NOT IN ('queued', 'running')", params![id])?;
        if n > 0 {
            conn.execute("DELETE FROM assistant_decisions WHERE run_id = ?1", params![id])?;
        }
        Ok(n)
    });
    match result {
        Ok(0) => bad("Run not found, or still running"),
        Ok(_) => HttpResponse::Ok().json(json!({"deleted": true})),
        Err(e) => internal(e),
    }
}

#[derive(Deserialize)]
struct DecisionsQuery {
    #[serde(default)]
    live: bool,
}

async fn decisions_handler(a: Data, query: web::Query<DecisionsQuery>) -> HttpResponse {
    let mut decisions = match list_decisions(&a.db, None, 200) {
        Ok(d) => d,
        Err(e) => return internal(e),
    };
    if query.live {
        let mut unique: Vec<String> = decisions.iter().map(|d| d.symbol.clone()).collect();
        unique.sort();
        unique.dedup();
        unique.truncate(30);
        let quotes = futures_util::future::join_all(unique.iter().map(|s| quote(&a.client, s))).await;
        let prices: HashMap<String, f64> = unique.into_iter().zip(quotes).filter_map(|(s, q)| Some((s, q.ok()?.price))).collect();
        for d in &mut decisions {
            d.current = prices.get(&d.symbol).copied();
        }
    }
    HttpResponse::Ok().json(decisions)
}

#[derive(Deserialize)]
struct DecideInput {
    run_id: String,
    symbol: String,
    decision: String,
    #[serde(default)]
    note: String,
    price: Option<f64>,
}

async fn decide_handler(a: Data, body: web::Json<DecideInput>) -> HttpResponse {
    let input = body.into_inner();
    if !["bought", "sold", "passed", "watching"].contains(&input.decision.as_str()) {
        return bad("Decision must be bought, sold, passed or watching");
    }
    let run = match get_run(&a.db, &input.run_id) {
        Ok(Some(run)) => run,
        Ok(None) => return missing("Run"),
        Err(e) => return internal(e),
    };
    let symbol = input.symbol.trim().to_uppercase();
    let Some(pick) = run.picks.iter().find(|p| p.symbol == symbol) else {
        return bad(format!("{symbol} is not a pick in this report"));
    };
    let note: String = input.note.chars().take(2_000).collect();
    let price = input.price.filter(|p| p.is_finite() && *p > 0.0).or(pick.price);
    let result = a.db.with_conn(|conn| {
        conn.execute(
            "INSERT INTO assistant_decisions (id, run_id, symbol, decision, note, price, decided_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(run_id, symbol) DO UPDATE SET decision = excluded.decision, note = excluded.note, price = excluded.price, decided_at = excluded.decided_at",
            params![Uuid::new_v4().to_string(), run.id, symbol, input.decision, note, price, chrono::Utc::now().to_rfc3339()],
        )
    });
    match result {
        Ok(_) => HttpResponse::Ok().json(json!({"saved": true})),
        Err(e) => internal(e),
    }
}

async fn delete_decision_handler(a: Data, path: web::Path<String>) -> HttpResponse {
    let id = path.into_inner();
    match a.db.with_conn(|conn| conn.execute("DELETE FROM assistant_decisions WHERE id = ?1", params![id])) {
        Ok(0) => missing("Decision"),
        Ok(_) => HttpResponse::Ok().json(json!({"deleted": true})),
        Err(e) => internal(e),
    }
}

fn csv_download(name: &str, body: String) -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/csv; charset=utf-8")
        .insert_header(("Content-Disposition", format!("attachment; filename=\"{name}.csv\"")))
        .body(body)
}

async fn run_csv_handler(a: Data, path: web::Path<String>) -> HttpResponse {
    match get_run(&a.db, &path.into_inner()) {
        Ok(Some(run)) => match run.table {
            Some(t) => {
                let date = run.started_at.get(..10).unwrap_or("run").to_string();
                csv_download(&format!("{}-{date}", table::slug(&run.job_name)), table::to_csv(&t.columns, &t.rows))
            }
            None => missing("Table"),
        },
        Ok(None) => missing("Run"),
        Err(e) => internal(e),
    }
}

async fn job_csv_handler(a: Data, path: web::Path<String>) -> HttpResponse {
    let job = match get_job(&a.db, &path.into_inner()) {
        Ok(Some(job)) => job,
        Ok(None) => return missing("Job"),
        Err(e) => return internal(e),
    };
    let columns = job.columns();
    if columns.is_empty() {
        return missing("Table");
    }
    match saved_rows(&a.db, &job.id) {
        Ok(rows) => csv_download(&format!("{}-all", table::slug(&job.name)), table::to_csv(&columns, &rows)),
        Err(e) => internal(e),
    }
}

async fn notes_handler(a: Data, path: web::Path<String>) -> HttpResponse {
    match list_notes(&a.db, &path.into_inner()) {
        Ok(notes) => HttpResponse::Ok().json(notes),
        Err(e) => internal(e),
    }
}

async fn delete_note_handler(a: Data, path: web::Path<String>) -> HttpResponse {
    let id = path.into_inner();
    match a.db.with_conn(|conn| conn.execute("DELETE FROM assistant_notes WHERE id = ?1", params![id])) {
        Ok(0) => missing("Note"),
        Ok(_) => HttpResponse::Ok().json(json!({"deleted": true})),
        Err(e) => internal(e),
    }
}

/// One suggested reminder as a calendar file. Opening it asks Calendar (or
/// any calendar app) to add the event, so the user confirms each one.
async fn reminder_handler(a: Data, path: web::Path<(String, usize)>) -> HttpResponse {
    let (id, index) = path.into_inner();
    let run = match get_run(&a.db, &id) {
        Ok(Some(run)) => run,
        Ok(None) => return missing("Run"),
        Err(e) => return internal(e),
    };
    let Some(reminder) = run.reminders.get(index) else { return missing("Reminder") };
    let Some(body) = ics(reminder, &run.job_name) else { return bad("The reminder's date is not valid") };
    HttpResponse::Ok()
        .content_type("text/calendar; charset=utf-8")
        .insert_header(("Content-Disposition", format!("attachment; filename=\"{}.ics\"", table::slug(&reminder.title))))
        .body(body)
}

fn ics_text(text: &str) -> String {
    text.replace('\\', "\\\\").replace(';', "\\;").replace(',', "\\,").replace(['\r', '\n'], "\\n")
}

fn ics(reminder: &Reminder, job_name: &str) -> Option<String> {
    let date = chrono::NaiveDate::parse_from_str(&reminder.date, "%Y-%m-%d").ok()?;
    let (start, end) = match reminder.time.as_deref().and_then(parse_time) {
        // Floating local time: it stays at 9:30 wherever the user is.
        Some(time) => {
            let at = date.and_time(time);
            (format!("DTSTART:{}", at.format("%Y%m%dT%H%M%S")), format!("DTEND:{}", (at + chrono::Duration::minutes(30)).format("%Y%m%dT%H%M%S")))
        }
        None => (format!("DTSTART;VALUE=DATE:{}", date.format("%Y%m%d")), format!("DTEND;VALUE=DATE:{}", (date + chrono::Duration::days(1)).format("%Y%m%d"))),
    };
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
    let description = ics_text(&format!("{}\n\nSuggested by your Ferro64 assistant ({job_name}).", reminder.note));
    let title = ics_text(&reminder.title);
    Some(format!(
        "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//Ferro64//Assistant//EN\r\nBEGIN:VEVENT\r\nUID:{}@ferro64\r\nDTSTAMP:{stamp}\r\n{start}\r\n{end}\r\nSUMMARY:{title}\r\nDESCRIPTION:{description}\r\nBEGIN:VALARM\r\nACTION:DISPLAY\r\nDESCRIPTION:{title}\r\nTRIGGER:-PT15M\r\nEND:VALARM\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n",
        Uuid::new_v4()
    ))
}

async fn telegram_handler(a: Data) -> HttpResponse {
    let (token, chat) = telegram::settings(&a.db);
    HttpResponse::Ok().json(json!({"token": token.as_deref().map(telegram::hint), "chat": chat}))
}

#[derive(Deserialize)]
struct TelegramInput {
    /// "" removes it; absent keeps it.
    token: Option<String>,
    chat: Option<String>,
}

async fn telegram_save_handler(a: Data, body: web::Json<TelegramInput>) -> HttpResponse {
    let input = body.into_inner();
    if let Some(token) = input.token.map(|t| t.trim().to_string()) {
        if !token.is_empty() && !telegram::valid_token(&token) {
            return bad("That does not look like a bot token. BotFather gives one like 123456789:AA…");
        }
        if let Err(e) = a.db.set_setting(telegram::TOKEN_KEY, &token) {
            return internal(e);
        }
        if token.is_empty() {
            let _ = a.db.set_setting(telegram::CHAT_KEY, "");
        }
    }
    if let Some(chat) = input.chat.map(|c| c.trim().to_string()) {
        if !chat.is_empty() && !telegram::valid_chat(&chat) {
            return bad("A chat id is a number, like 123456789");
        }
        if let Err(e) = a.db.set_setting(telegram::CHAT_KEY, &chat) {
            return internal(e);
        }
    }
    telegram_handler(a).await
}

async fn telegram_chats_handler(a: Data) -> HttpResponse {
    let (Some(token), _) = telegram::settings(&a.db) else { return bad("Save the bot token first") };
    match telegram::recent_chats(&a.client, &token).await {
        Ok(chats) if chats.is_empty() => bad("No messages yet. Open your bot in Telegram, send it \"hi\", then try again."),
        Ok(chats) => HttpResponse::Ok().json(chats),
        Err(e) => bad(e),
    }
}

async fn telegram_test_handler(a: Data) -> HttpResponse {
    let (Some(token), Some(chat)) = telegram::settings(&a.db) else { return bad("Save the bot token and pick your chat first") };
    match telegram::send_text(&a.client, &token, &chat, "Ferro64 is connected. Your assistant's reports will arrive here.").await {
        Ok(()) => HttpResponse::Ok().json(json!({"sent": true})),
        Err(e) => bad(e),
    }
}

/// Email settings without the password, and the provider presets.
async fn email_handler(a: Data) -> HttpResponse {
    let saved = email::load(&a.db);
    let presets: Vec<Value> = email::PRESETS.iter().map(|p| json!({"id": p.id, "name": p.name, "host": p.host, "port": p.port, "security": p.security, "help": p.help})).collect();
    HttpResponse::Ok().json(json!({
        "settings": saved.map(|s| json!({"provider": s.provider, "host": s.host, "port": s.port, "security": s.security, "username": s.username, "from": s.from, "to": s.to, "has_password": !s.password.is_empty()})),
        "presets": presets,
    }))
}

#[derive(Deserialize)]
struct EmailInput {
    provider: String,
    host: String,
    port: u16,
    security: String,
    #[serde(default)]
    username: String,
    /// Absent or empty keeps the saved one.
    password: Option<String>,
    from: String,
    to: String,
}

async fn email_save_handler(a: Data, body: web::Json<EmailInput>) -> HttpResponse {
    let input = body.into_inner();
    let kept = email::load(&a.db).map(|s| s.password).unwrap_or_default();
    // App passwords are shown with spaces (Gmail: "abcd efgh ijkl mnop").
    let password = input.password.map(|p| p.split_whitespace().collect::<String>()).filter(|p| !p.is_empty()).unwrap_or(kept);
    let settings = email::Settings {
        provider: input.provider.trim().to_string(),
        host: input.host.trim().to_string(),
        port: input.port,
        security: input.security.trim().to_string(),
        username: input.username.trim().to_string(),
        password,
        from: input.from.trim().to_string(),
        to: input.to.trim().to_string(),
    };
    if let Err(e) = email::validate(&settings) {
        return bad(e);
    }
    if !settings.username.is_empty() && settings.password.is_empty() {
        return bad("Enter the app password for that account");
    }
    match email::save(&a.db, Some(&settings)) {
        Ok(()) => email_handler(a).await,
        Err(e) => internal(e),
    }
}

async fn email_delete_handler(a: Data) -> HttpResponse {
    match email::save(&a.db, None) {
        Ok(()) => HttpResponse::Ok().json(json!({"deleted": true})),
        Err(e) => internal(e),
    }
}

async fn email_test_handler(a: Data) -> HttpResponse {
    let Some(settings) = email::load(&a.db) else { return bad("Save your email settings first") };
    let body = "Ferro64 is connected. Your assistant's reports will arrive by email, with the full report, its sources and any CSV attached.\n\n—\nSent by your Ferro64 assistant, from your own computer.";
    match email::send(&settings, "Ferro64 · email test", body, None).await {
        Ok(()) => HttpResponse::Ok().json(json!({"sent": true, "to": settings.to})),
        Err(e) => bad(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn job(time: Option<&str>, weekdays_only: bool, last: Option<&str>) -> Job {
        Job {
            id: "j".into(),
            name: "n".into(),
            instructions: "i".into(),
            time: time.map(str::to_string),
            weekdays_only,
            enabled: true,
            model: None,
            created_at: String::new(),
            last_run_on: last.map(str::to_string),
            table_columns: None,
        }
    }

    #[test]
    fn a_job_runs_once_a_day_after_its_time() {
        // Wednesday 25 November 2026.
        let at = |h, m| Local.with_ymd_and_hms(2026, 11, 25, h, m, 0).unwrap();
        assert!(!is_due(&job(Some("07:30"), false, None), at(7, 29)));
        assert!(is_due(&job(Some("07:30"), false, None), at(7, 30)));
        assert!(is_due(&job(Some("07:30"), false, Some("2026-11-24")), at(9, 0)));
        assert!(!is_due(&job(Some("07:30"), false, Some("2026-11-25")), at(9, 0)));
        assert!(!is_due(&job(None, false, None), at(9, 0)));
        let saturday = Local.with_ymd_and_hms(2026, 11, 28, 9, 0, 0).unwrap();
        assert!(!is_due(&job(Some("07:30"), true, None), saturday));
        assert!(is_due(&job(Some("07:30"), false, None), saturday));
        let mut off = job(Some("07:30"), false, None);
        off.enabled = false;
        assert!(!is_due(&off, at(9, 0)));
    }

    #[test]
    fn only_one_process_claims_a_day_and_live_runs_are_left_alone() {
        let dir = std::env::temp_dir().join(format!("ferro64-assistant-{}", Uuid::new_v4()));
        let db = Database::new(dir.join("t.db")).unwrap();
        init(&db).unwrap();
        save_job(&db, &job(Some("07:30"), false, None)).unwrap();
        assert!(claim_today(&db, "j", "2026-11-25").unwrap());
        assert!(!claim_today(&db, "j", "2026-11-25").unwrap());
        assert!(claim_today(&db, "j", "2026-11-26").unwrap());
        let mut run = Run { id: "r".into(), job_id: "j".into(), job_name: "n".into(), status: "running".into(), trigger: "manual".into(), model: "m".into(),
            started_at: "2020-01-01T00:00:00Z".into(), finished_at: None, report: None, sources: vec![], steps: vec![], picks: vec![], error: None, table: None, reminders: vec![] };
        // Started long ago but saved just now: another process is on it.
        save_run(&db, &run).unwrap();
        init(&db).unwrap();
        assert_eq!(get_run(&db, "r").unwrap().unwrap().status, "running");
        // Not saved for an hour: cut off.
        run.status = "running".into();
        save_run(&db, &run).unwrap();
        db.with_conn(|c| c.execute("UPDATE assistant_runs SET updated_at = '2020-01-01T00:00:00Z'", [])).unwrap();
        close_stale_runs(&db).unwrap();
        assert_eq!(get_run(&db, "r").unwrap().unwrap().status, "failed");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn picks_come_off_the_end_of_the_report() {
        let report = "## Today\nNVDA looks stretched [S2].\n\n```json\n{\"picks\": [{\"symbol\": \"$nvda\", \"action\": \"Watch\", \"conviction\": \"low\", \"thesis\": \"t\", \"risks\": \"r\", \"sources\": [\"S2\"]}, {\"symbol\": \"not a ticker!\"}]}\n```\n";
        let (text, picks) = split_picks(report);
        assert_eq!(text, "## Today\nNVDA looks stretched [S2].");
        assert_eq!(picks.len(), 1);
        assert_eq!(picks[0].symbol, "NVDA");
        assert_eq!(picks[0].action, "watch");
        // Code blocks that are not picks stay, and a broken block is left alone.
        let other = "Text\n```rust\nfn main() {}\n```\nmore";
        assert_eq!(split_picks(other), (other.to_string(), vec![]));
        let broken = "Text\n```json\n{\"picks\": [\n```";
        assert_eq!(split_picks(broken).1, vec![]);
    }

    #[test]
    fn history_figures_are_computed_not_guessed() {
        let closes: Vec<(String, f64)> = (0..60).map(|i| (format!("d{i}"), 100.0 + i as f64)).collect();
        let text = summarize("TEST", &closes);
        assert!(text.contains("last close $159.00"));
        assert!(text.contains("change over 1 week: +3.2%"), "{text}"); // 159 / 154
        assert!(text.contains("change over the period: +59.0%"));
        assert!(text.contains("20-day average $149.50 (last close above it by 6.4%)"));
        assert!(!text.contains("200-day"));
    }

    #[test]
    fn reminders_are_checked_and_become_calendar_files() {
        let client = reqwest::Client::new();
        let mut tools = Tools { client: &client, sources: vec![], quotes: HashMap::new(), cap: 1000, notes: vec![], reminders: vec![] };
        let soon = (Local::now().date_naive() + chrono::Duration::days(8)).to_string();
        let r = |date: &str, time: Option<&str>| Reminder { title: "Recheck NVDA; earnings, guidance".into(), date: date.into(), time: time.map(str::to_string), note: "line one\nline two".into() };
        assert!(tools.remind(r("2020-01-01", None)).is_err());
        assert!(tools.remind(r(&soon, Some("9am"))).is_err());
        assert!(tools.remind(r(&soon, Some("09:30"))).is_ok());
        let body = ics(&tools.reminders[0], "Stocks").unwrap();
        assert!(body.contains(&format!("DTSTART:{}T093000", soon.replace('-', ""))));
        assert!(body.contains("SUMMARY:Recheck NVDA\\; earnings\\, guidance\r\n"), "{body}");
        assert!(body.contains("line one\\nline two"));
        let all_day = ics(&r(&soon, None), "Stocks").unwrap();
        assert!(all_day.contains("DTSTART;VALUE=DATE:"));
        for _ in 0..NOTES_PER_RUN {
            assert!(tools.note("Practo blocks reading").is_ok());
        }
        assert!(tools.note("one too many").is_err());
    }

    #[test]
    fn tickers_are_cleaned_and_bounded() {
        assert_eq!(symbols(&json!({"symbols": ["aapl", "$msft", "rm -rf /", "BRK.B"]})), ["AAPL", "MSFT", "BRK.B"]);
        assert_eq!(symbols(&json!({"symbols": "voo, qqq"})), ["VOO", "QQQ"]);
        assert!(!valid_symbol("../x") && !valid_symbol(""));
    }

    #[test]
    fn old_tool_results_shrink_first() {
        let mut messages = vec![json!({"role": "system", "content": "s"})];
        for _ in 0..6 {
            messages.push(json!({"role": "tool", "content": "x".repeat(5_000)}));
        }
        fit(&mut messages, 18_000);
        assert!(messages[1]["content"].as_str().unwrap().len() < 600);
        assert_eq!(messages[6]["content"].as_str().unwrap().len(), 5_000);
        let total: usize = messages.iter().map(|m| m["content"].as_str().unwrap().len()).sum();
        assert!(total <= 18_000);
    }
}
