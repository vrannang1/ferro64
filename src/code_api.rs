//! HTTP routes for the Code view: projects, files, agent runs, approvals,
//! undo, command permissions and git.
use crate::coder::{self, Coder, Decision, RunRequest};
use crate::db::Project;
use actix_web::{http::StatusCode, web, HttpResponse};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use futures_util::StreamExt as _;
use tokio::sync::mpsc;

type Reply = Result<HttpResponse, actix_web::Error>;

fn fail(status: StatusCode, message: impl Into<String>) -> actix_web::Error {
    actix_web::error::InternalError::from_response(
        "",
        HttpResponse::build(status).json(json!({"error": message.into()})),
    )
    .into()
}
fn bad(message: impl Into<String>) -> actix_web::Error {
    fail(StatusCode::BAD_REQUEST, message)
}
fn internal(error: impl std::fmt::Display) -> actix_web::Error {
    fail(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
}

pub fn routes(cfg: &mut web::ServiceConfig) {
    cfg.route("/api/code/projects", web::get().to(list_projects))
        .route("/api/code/projects", web::post().to(add_project))
        .route("/api/code/projects/{id}", web::delete().to(remove_project))
        .route("/api/code/projects/{id}/tree", web::get().to(tree))
        .route("/api/code/projects/{id}/file", web::get().to(read_file))
        .route("/api/code/projects/{id}/file", web::put().to(save_file))
        .route("/api/code/projects/{id}/runs", web::get().to(list_runs))
        .route("/api/code/projects/{id}/permissions", web::get().to(permissions))
        .route("/api/code/projects/{id}/permissions", web::delete().to(revoke_permission))
        .route("/api/code/projects/{id}/git", web::post().to(git))
        .route("/api/code/projects/{id}/terminal", web::get().to(terminal))
        .route("/api/code/runs/{id}", web::get().to(get_run))
        .route("/api/code/runs/{id}/undo", web::post().to(undo))
        .route("/api/code/runs/{id}/stop", web::post().to(stop_run))
        .route("/api/code/run", web::post().to(run))
        .route("/api/code/projects/{id}/notes", web::get().to(list_notes))
        .route("/api/code/projects/{id}/brief", web::get().to(get_brief))
        .route("/api/code/projects/{id}/brief", web::put().to(save_brief))
        .route("/api/code/projects/{id}/brief", web::delete().to(delete_brief))
        .route("/api/code/notes/settings", web::post().to(notes_settings))
        .route("/api/code/notes/{id}", web::delete().to(delete_note))
        .route("/api/code/notes/{id}/restore", web::post().to(restore_note))
        .route("/api/code/approve", web::post().to(approve));
}

fn project(coder: &Coder, id: &str) -> Result<(Project, PathBuf), actix_web::Error> {
    let project = coder.db.project(id).map_err(internal)?.ok_or_else(|| fail(StatusCode::NOT_FOUND, "Project not found"))?;
    let root = PathBuf::from(&project.root);
    if !root.is_dir() {
        return Err(bad(format!("The project folder {} no longer exists", project.root)));
    }
    Ok((project, root))
}

async fn list_projects(coder: web::Data<Coder>) -> Reply {
    Ok(HttpResponse::Ok().json(coder.db.list_projects().map_err(internal)?))
}

#[derive(Deserialize)]
struct AddProject {
    path: String,
    name: Option<String>,
}

/// Folders too broad or too sensitive to hand to an agent.
fn refused_root(root: &Path) -> bool {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let system = ["/System", "/usr", "/bin", "/sbin", "/etc", "/private/etc", "/private/var", "/Library", "/Applications", "/opt"];
    root.parent().is_none()
        || home.as_deref() == Some(root)
        || ["/private", "/private/tmp", "/tmp", "/Volumes", "/Users"].iter().any(|p| root == Path::new(p))
        || system.iter().any(|p| root.starts_with(p))
        || home.is_some_and(|h| root.starts_with(h.join("Library")) || root.starts_with(h.join(".ssh")))
}

async fn add_project(coder: web::Data<Coder>, body: web::Json<AddProject>) -> Reply {
    let raw = body.path.trim();
    let expanded = match raw.strip_prefix("~/") {
        Some(rest) => std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default().join(rest),
        None => PathBuf::from(raw),
    };
    if !expanded.is_absolute() {
        return Err(bad("Give the project folder's full path, for example ~/Sandbox/my-app"));
    }
    let root = expanded.canonicalize().map_err(|e| bad(format!("Cannot open {}: {e}", expanded.display())))?;
    if !root.is_dir() {
        return Err(bad(format!("{} is not a folder", root.display())));
    }
    if refused_root(&root) {
        return Err(bad(format!("{} is too broad or a system folder; choose a project folder inside it", root.display())));
    }
    if let Some(existing) = coder.db.list_projects().map_err(internal)?.into_iter().find(|p| Path::new(&p.root) == root) {
        return Ok(HttpResponse::Ok().json(existing));
    }
    let name = body
        .name
        .as_deref()
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| root.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "project".into()));
    let project = Project {
        id: uuid::Uuid::new_v4().to_string(),
        name: name.chars().take(80).collect(),
        root: root.to_string_lossy().to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    coder.db.add_project(&project).map_err(internal)?;
    Ok(HttpResponse::Created().json(project))
}

async fn remove_project(coder: web::Data<Coder>, id: web::Path<String>) -> Reply {
    if coder.db.remove_project(&id).map_err(internal)? {
        Ok(HttpResponse::Ok().json(json!({"removed": *id, "files": "untouched"})))
    } else {
        Err(fail(StatusCode::NOT_FOUND, "Project not found"))
    }
}

#[derive(Deserialize)]
struct PathQuery {
    #[serde(default)]
    path: String,
}

async fn tree(coder: web::Data<Coder>, id: web::Path<String>, query: web::Query<PathQuery>) -> Reply {
    let (_, root) = project(&coder, &id)?;
    let dir = coder::resolve(&root, &query.path).map_err(bad)?;
    let entries = coder::list_dir(&root, &dir).map_err(bad)?;
    Ok(HttpResponse::Ok().json(entries))
}

async fn read_file(coder: web::Data<Coder>, id: web::Path<String>, query: web::Query<PathQuery>) -> Reply {
    let (_, root) = project(&coder, &id)?;
    let path = coder::resolve(&root, &query.path).map_err(bad)?;
    let bytes = std::fs::metadata(&path).map_err(|e| bad(format!("Cannot open {}: {e}", query.path)))?;
    if bytes.is_dir() || bytes.len() > 2_000_000 {
        return Err(bad("Only text files up to 2 MB open in the editor"));
    }
    let content = std::fs::read(&path).map_err(internal)?;
    match String::from_utf8(content) {
        Ok(text) if !text.contains('\0') => Ok(HttpResponse::Ok().json(json!({"path": query.path, "content": text}))),
        _ => Err(bad("Binary file; it cannot be edited as text")),
    }
}

#[derive(Deserialize)]
struct SaveFile {
    path: String,
    content: String,
}

/// The user's own edit from the editor: no approval, not part of any run.
async fn save_file(coder: web::Data<Coder>, id: web::Path<String>, body: web::Json<SaveFile>) -> Reply {
    let (_, root) = project(&coder, &id)?;
    let path = coder::resolve(&root, &body.path).map_err(bad)?;
    if path.is_dir() {
        return Err(bad("That is a folder"));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(internal)?;
    }
    std::fs::write(&path, &body.content).map_err(internal)?;
    Ok(HttpResponse::Ok().json(json!({"saved": body.path, "bytes": body.content.len()})))
}

async fn list_runs(coder: web::Data<Coder>, id: web::Path<String>) -> Reply {
    let runs = coder.db.list_code_runs(&id).map_err(internal)?;
    let rows: Vec<Value> = runs
        .into_iter()
        .map(|r| json!({"id": r.id, "task": r.task, "model": r.model, "status": r.status, "created_at": r.created_at, "updated_at": r.updated_at}))
        .collect();
    Ok(HttpResponse::Ok().json(rows))
}

async fn get_run(coder: web::Data<Coder>, id: web::Path<String>) -> Reply {
    let run = coder.db.code_run(&id).map_err(internal)?.ok_or_else(|| fail(StatusCode::NOT_FOUND, "Run not found"))?;
    Ok(HttpResponse::Ok().json(run))
}

async fn stop_run(coder: web::Data<Coder>, id: web::Path<String>) -> Reply {
    if coder.stop(&id) {
        Ok(HttpResponse::Ok().json(serde_json::json!({"status": "stopping"})))
    } else {
        Err(fail(StatusCode::NOT_FOUND, "That run is not running"))
    }
}

async fn undo(coder: web::Data<Coder>, id: web::Path<String>) -> Reply {
    Ok(HttpResponse::Ok().json(coder.undo(&id).map_err(bad)?))
}

async fn get_brief(coder: web::Data<Coder>, id: web::Path<String>) -> Reply {
    let (_, root) = project(&coder, &id)?;
    let brief = coder.db.brief(&id).map_err(internal)?;
    let Some(brief) = brief else { return Ok(HttpResponse::Ok().json(json!({"text": null}))) };
    let now = web::block(move || crate::brief::snapshot(&root)).await.map_err(internal)?;
    let stale = crate::brief::is_stale(&brief, &now);
    Ok(HttpResponse::Ok().json(json!({"text": brief.text, "edited": brief.edited, "stale": stale, "updated_at": brief.updated_at})))
}

#[derive(Deserialize)]
struct BriefEdit {
    text: String,
}

/// The user's version, kept until they ask for a new one.
async fn save_brief(coder: web::Data<Coder>, id: web::Path<String>, body: web::Json<BriefEdit>) -> Reply {
    let (_, root) = project(&coder, &id)?;
    let text = body.text.trim().to_string();
    if text.is_empty() || text.chars().count() > 10_000 {
        return Err(fail(StatusCode::BAD_REQUEST, "The summary must be between 1 and 10,000 characters"));
    }
    let now = web::block(move || crate::brief::snapshot(&root)).await.map_err(internal)?;
    coder.db.save_brief(&crate::db::Brief { project_id: id.to_string(), text, files_json: serde_json::to_string(&now.files).map_err(internal)?, key_hash: now.key_hash, edited: true, updated_at: chrono::Utc::now().to_rfc3339() }).map_err(internal)?;
    Ok(HttpResponse::Ok().json(json!({"status": "saved"})))
}

/// The next run on this project writes a fresh one.
async fn delete_brief(coder: web::Data<Coder>, id: web::Path<String>) -> Reply {
    coder.db.delete_brief(&id).map_err(internal)?;
    Ok(HttpResponse::Ok().json(json!({"status": "deleted"})))
}

async fn list_notes(coder: web::Data<Coder>, id: web::Path<String>) -> Reply {
    let items = coder.db.all_notes(&id).map_err(internal)?;
    Ok(HttpResponse::Ok().json(json!({"enabled": crate::notes::enabled(&coder.db), "items": items})))
}

#[derive(Deserialize)]
struct NotesSettings {
    enabled: bool,
}

async fn notes_settings(coder: web::Data<Coder>, body: web::Json<NotesSettings>) -> Reply {
    crate::notes::set_enabled(&coder.db, body.enabled).map_err(internal)?;
    Ok(HttpResponse::Ok().json(json!({"enabled": body.enabled})))
}

async fn delete_note(coder: web::Data<Coder>, id: web::Path<String>) -> Reply {
    if coder.db.delete_note(&id).map_err(internal)? {
        Ok(HttpResponse::Ok().json(json!({"status": "deleted"})))
    } else {
        Err(fail(StatusCode::NOT_FOUND, "Note not found"))
    }
}

async fn restore_note(coder: web::Data<Coder>, id: web::Path<String>) -> Reply {
    if coder.db.restore_note(&id).map_err(internal)? {
        Ok(HttpResponse::Ok().json(json!({"status": "active"})))
    } else {
        Err(fail(StatusCode::NOT_FOUND, "Note not found"))
    }
}

async fn permissions(coder: web::Data<Coder>, id: web::Path<String>) -> Reply {
    Ok(HttpResponse::Ok().json(coder.db.allowed_commands(&id).map_err(internal)?))
}

#[derive(Deserialize)]
struct Prefix {
    prefix: String,
}

async fn revoke_permission(coder: web::Data<Coder>, id: web::Path<String>, body: web::Json<Prefix>) -> Reply {
    let removed = coder.db.revoke_command(&id, &body.prefix).map_err(internal)?;
    Ok(HttpResponse::Ok().json(json!({"removed": removed})))
}

#[derive(Deserialize)]
struct Approve {
    run_id: String,
    call_id: String,
    decision: Decision,
    note: Option<String>,
}

async fn approve(coder: web::Data<Coder>, body: web::Json<Approve>) -> Reply {
    let body = body.into_inner();
    if coder.approvals.answer(&body.run_id, &body.call_id, body.decision, body.note) {
        Ok(HttpResponse::Ok().json(json!({"status": "answered"})))
    } else {
        Err(fail(StatusCode::GONE, "That request is no longer waiting (the run ended or was answered)"))
    }
}

/// Streams a run as server-sent events: run, status, reasoning, content,
/// tool, approval, change, tool_result, then done or error. Closing the
/// connection stops the run.
async fn run(coder: web::Data<Coder>, body: web::Json<RunRequest>) -> Reply {
    let (events, receiver) = mpsc::unbounded_channel();
    let coder = coder.into_inner();
    let request = body.into_inner();
    tokio::spawn(async move { coder.run(request, events).await });
    let events = futures_util::StreamExt::map(tokio_stream::wrappers::UnboundedReceiverStream::new(receiver), |(name, data): coder::Event| {
        web::Bytes::from(format!("event: {name}\ndata: {data}\n\n"))
    });
    // A comment every 15 s: while the agent waits for an approval nothing
    // else is sent, and only a write reveals that the page has gone, which
    // drops the receiver and ends the run instead of leaving it waiting.
    let heartbeat = futures_util::StreamExt::map(
        tokio_stream::wrappers::IntervalStream::new(tokio::time::interval_at(
            tokio::time::Instant::now() + std::time::Duration::from_secs(15),
            std::time::Duration::from_secs(15),
        )),
        |_| web::Bytes::from_static(b": ping\n\n"),
    );
    // The run ends when its events end; the heartbeat must not keep it open.
    let stream = futures_util::StreamExt::map(
        tokio_stream::StreamExt::merge(events.map(Some).chain(futures_util::stream::once(async { None })), heartbeat.map(Some))
            .take_while(|item| futures_util::future::ready(item.is_some()))
            .map(|item| item.unwrap_or_default()),
        Ok::<_, actix_web::Error>,
    );
    Ok(HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(stream))
}

/// A shell in the project folder, over a WebSocket (see terminal.rs).
async fn terminal(coder: web::Data<Coder>, id: web::Path<String>, req: actix_web::HttpRequest, body: web::Payload) -> Reply {
    let (_, root) = project(&coder, &id)?;
    crate::terminal::open(req, body, root).await
}

#[derive(Deserialize)]
struct Git {
    /// status, diff, log, commit or push
    action: String,
    message: Option<String>,
}

async fn git_run(root: &Path, args: &[&str]) -> (bool, String) {
    let run = tokio::process::Command::new("git")
        .args(args)
        .current_dir(root)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_PAGER", "cat")
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true)
        .output();
    match tokio::time::timeout(std::time::Duration::from_secs(120), run).await {
        Ok(Ok(out)) => {
            let mut text = String::from_utf8_lossy(&out.stdout).to_string();
            text.push_str(&String::from_utf8_lossy(&out.stderr));
            (out.status.success(), text)
        }
        Ok(Err(e)) => (false, format!("Cannot run git: {e}")),
        Err(_) => (false, "git did not finish within 2 minutes".to_string()),
    }
}

/// Git actions the user clicks in the Code view; they never come from the model.
async fn git(coder: web::Data<Coder>, id: web::Path<String>, body: web::Json<Git>) -> Reply {
    let (_, root) = project(&coder, &id)?;
    if !git_run(&root, &["rev-parse", "--is-inside-work-tree"]).await.0 {
        return Err(bad("This project is not a git repository; run `git init` in it first"));
    }
    let (ok, output) = match body.action.as_str() {
        "status" => git_run(&root, &["status", "--short", "--branch"]).await,
        "diff" => {
            let (ok, stat) = git_run(&root, &["diff", "--stat", "HEAD"]).await;
            let (_, full) = git_run(&root, &["diff", "HEAD"]).await;
            let full: String = full.chars().take(60_000).collect();
            (ok, format!("{stat}\n{full}"))
        }
        "log" => git_run(&root, &["log", "--oneline", "-15"]).await,
        "commit" => {
            let message = body.message.as_deref().map(str::trim).filter(|m| !m.is_empty()).ok_or_else(|| bad("Write a commit message"))?;
            let (added, out) = git_run(&root, &["add", "-A"]).await;
            if !added {
                (false, out)
            } else {
                git_run(&root, &["commit", "-m", message]).await
            }
        }
        "push" => git_run(&root, &["push"]).await,
        _ => return Err(bad("action must be status, diff, log, commit or push")),
    };
    Ok(HttpResponse::Ok().json(json!({"ok": ok, "output": output.trim_end()})))
}
