//! A real terminal in the Code view: the user's login shell in a
//! pseudo-terminal, started in the project folder, streamed over a
//! WebSocket. The browser sends keystrokes as binary frames and
//! `{"resize":[cols,rows]}` as text; the shell's output comes back as binary.
//!
//! A shell is full access to this Mac, and the server has no login, so the
//! connection is accepted only from Ferro64's own page: the Host must be
//! loopback and the Origin must match it. That stops other websites (and DNS
//! rebinding) from opening a shell through 127.0.0.1.
use actix_web::{http::header, web, HttpRequest, HttpResponse};
use portable_pty::{CommandBuilder, PtySize};
use std::io::{Read, Write};
use std::path::PathBuf;

/// True when a request comes from this server's own page.
pub fn same_origin(host: Option<&str>, origin: Option<&str>) -> bool {
    let (Some(host), Some(origin)) = (host, origin) else { return false };
    let loopback = host.starts_with("127.0.0.1:") || host.starts_with("localhost:") || host == "127.0.0.1" || host == "localhost";
    loopback && (origin == format!("http://{host}") || origin == format!("https://{host}"))
}

/// The user's shell: $SHELL, else zsh or bash, whichever exists; PowerShell
/// on Windows.
pub fn shell() -> String {
    if cfg!(windows) {
        return "powershell.exe".to_string();
    }
    std::env::var("SHELL")
        .ok()
        .filter(|s| s.starts_with('/') && std::path::Path::new(s).exists())
        .or_else(|| ["/bin/zsh", "/bin/bash", "/usr/bin/bash", "/bin/sh"].into_iter().find(|s| std::path::Path::new(s).exists()).map(str::to_string))
        .unwrap_or_else(|| "/bin/sh".to_string())
}

pub async fn open(req: HttpRequest, body: web::Payload, root: PathBuf) -> Result<HttpResponse, actix_web::Error> {
    let header_str = |name| req.headers().get(name).and_then(|v: &header::HeaderValue| v.to_str().ok());
    if !same_origin(header_str(header::HOST), header_str(header::ORIGIN)) {
        return Ok(HttpResponse::Forbidden().body("The terminal only opens from Ferro64's own page"));
    }
    let (response, mut session, mut messages) = actix_ws::handle(&req, body)?;

    let pty = portable_pty::native_pty_system()
        .openpty(PtySize { rows: 24, cols: 80, pixel_width: 0, pixel_height: 0 })
        .map_err(actix_web::error::ErrorInternalServerError)?;
    let mut command = CommandBuilder::new(shell());
    if !cfg!(windows) {
        command.arg("-l");
    }
    command.cwd(&root);
    command.env("TERM", "xterm-256color");
    command.env("COLORTERM", "truecolor");
    let mut child = pty.slave.spawn_command(command).map_err(actix_web::error::ErrorInternalServerError)?;
    drop(pty.slave);
    let mut reader = pty.master.try_clone_reader().map_err(actix_web::error::ErrorInternalServerError)?;
    let mut writer = pty.master.take_writer().map_err(actix_web::error::ErrorInternalServerError)?;
    let master = pty.master;

    // The pty read blocks, so it gets its own thread.
    let (output, mut output_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    std::thread::spawn(move || {
        let mut buffer = [0u8; 8192];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if output.blocking_send(buffer[..n].to_vec()).is_err() {
                        break;
                    }
                }
            }
        }
    });

    actix_web::rt::spawn(async move {
        loop {
            tokio::select! {
                chunk = output_rx.recv() => match chunk {
                    Some(bytes) => {
                        if session.binary(bytes).await.is_err() { break; }
                    }
                    // The shell exited.
                    None => break,
                },
                message = messages.recv() => match message {
                    Some(Ok(actix_ws::Message::Binary(bytes))) => {
                        if writer.write_all(&bytes).is_err() { break; }
                    }
                    Some(Ok(actix_ws::Message::Text(text))) => {
                        if let Ok(control) = serde_json::from_str::<serde_json::Value>(&text) {
                            if let Some([cols, rows]) = control["resize"].as_array().map(|a| [a.first().and_then(|v| v.as_u64()), a.get(1).and_then(|v| v.as_u64())]) {
                                let size = PtySize { cols: cols.unwrap_or(80).clamp(10, 1000) as u16, rows: rows.unwrap_or(24).clamp(4, 500) as u16, pixel_width: 0, pixel_height: 0 };
                                let _ = master.resize(size);
                            }
                        }
                    }
                    Some(Ok(actix_ws::Message::Ping(bytes))) => { let _ = session.pong(&bytes).await; }
                    Some(Ok(actix_ws::Message::Close(_))) | None | Some(Err(_)) => break,
                    _ => {}
                },
            }
        }
        // The page closed or the shell ended: stop the shell and its children.
        let _ = child.kill();
        let _ = child.wait();
        let _ = session.close(None).await;
    });
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_apps_own_page_may_open_a_shell() {
        assert!(same_origin(Some("127.0.0.1:3042"), Some("http://127.0.0.1:3042")));
        assert!(same_origin(Some("localhost:3042"), Some("http://localhost:3042")));
        // Another site, a missing Origin, or a rebound DNS name are refused.
        assert!(!same_origin(Some("127.0.0.1:3042"), Some("https://evil.example")));
        assert!(!same_origin(Some("127.0.0.1:3042"), None));
        assert!(!same_origin(Some("evil.example:3042"), Some("http://evil.example:3042")));
        assert!(!same_origin(Some("127.0.0.1:3042"), Some("http://127.0.0.1:9999")));
        assert!(!same_origin(None, Some("http://127.0.0.1:3042")));
    }
}
