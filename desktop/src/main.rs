//! Ferro64 in a native window. Reuses a Ferro64 server already running on
//! its port, or starts one and stops it on quit, so there is never a second
//! server (and a second model in memory).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

/// The server this app started, stopped on quit.
struct Server(Mutex<Option<Child>>);

fn port() -> u16 {
    std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(3042)
}

/// True when a Ferro64 server answers on the port.
fn ferro64_running(port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_millis(500)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
    let request = format!("GET /api/status HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = String::new();
    let _ = stream.read_to_string(&mut response);
    response.contains("\"app\":\"Ferro64\"")
}

/// The server binary: FERRO64_SERVER, next to this app (in the .app
/// bundle), or the repository's release build.
fn server_binary() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("FERRO64_SERVER") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(PathBuf::from)) {
        candidates.push(dir.join(if cfg!(windows) { "ferro64.exe" } else { "ferro64" }));
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(if cfg!(windows) { "../target/release/ferro64.exe" } else { "../target/release/ferro64" }));
    candidates.into_iter().find(|p| p.is_file())
}

fn start_server(port: u16) -> Result<Option<Child>, String> {
    if ferro64_running(port) {
        return Ok(None);
    }
    let binary = server_binary().ok_or("The Ferro64 server was not found. Build it with `cargo build --release` in the Ferro64 folder.")?;
    let log = std::fs::File::create(std::env::temp_dir().join("ferro64-desktop-server.log")).ok();
    let mut command = Command::new(&binary);
    command.env("PORT", port.to_string()).env("HOST", "127.0.0.1").env("FERRO64_EXIT_WITH_PARENT", "1");
    if let Some(log) = log {
        if let Ok(copy) = log.try_clone() {
            command.stdout(copy);
        }
        command.stderr(log);
    }
    let mut child = command.spawn().map_err(|e| format!("Could not start {}: {e}", binary.display()))?;
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if ferro64_running(port) {
            return Ok(Some(child));
        }
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!("The Ferro64 server exited ({status}). Its log is in {}.", std::env::temp_dir().join("ferro64-desktop-server.log").display()));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let _ = child.kill();
    Err("The Ferro64 server did not start within a minute.".to_string())
}

/// Asks the server to shut down and returns at once, so the window and Dock
/// icon go away immediately while the server and model stop in the background.
/// A small watchdog outlives this app and forces them after 8 seconds.
fn stop_server(child: Child) {
    let pid = child.id();
    // Unix: a polite TERM lets the server stop llama-server itself. Windows has
    // no TERM, so the server and everything it started are ended together.
    #[cfg(unix)]
    {
        let _ = Command::new("kill").args(["-TERM", &pid.to_string()]).status();
        let watchdog = format!("for i in 1 2 3 4 5 6 7 8; do kill -0 {pid} 2>/dev/null || exit 0; sleep 1; done; pkill -KILL -P {pid}; kill -KILL {pid}");
        let _ = Command::new("sh").args(["-c", &watchdog]).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn();
    }
    #[cfg(windows)]
    let _ = Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).status();
}

fn main() {
    let port = port();
    let (page, child) = match start_server(port) {
        Ok(child) => (WebviewUrl::External(format!("http://127.0.0.1:{port}/").parse().expect("valid URL")), child),
        Err(error) => (WebviewUrl::App(format!("index.html?error={}", encode(&error)).into()), None),
    };
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(Server(Mutex::new(child)))
        .setup(move |app| {
            WebviewWindowBuilder::new(app, "main", page.clone())
                .title("Ferro64")
                .inner_size(1440.0, 920.0)
                .min_inner_size(720.0, 520.0)
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build the Ferro64 window");
    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            if let Some(child) = handle.state::<Server>().0.lock().unwrap().take() {
                stop_server(child);
            }
        }
    });
}

fn encode(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}
