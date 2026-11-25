//! Reports by email, through the user's own mail account (SMTP). Ferro64
//! sends them after a run; the model has no tool for it and cannot choose
//! the recipients. Settings, including the app password, are kept in
//! SQLite on this machine and sent only to the mail server, encrypted.

use crate::db::Database;
use lettre::message::{header::ContentType, Attachment, Mailbox, MultiPart, SinglePart};
use lettre::transport::smtp::authentication::Credentials;
use lettre::transport::smtp::client::{Tls, TlsParameters};
use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const KEY: &str = "assistant_email";
const MAX_RECIPIENTS: usize = 10;

/// A mail provider's server, so the user only picks a name.
pub struct Preset {
    pub id: &'static str,
    pub name: &'static str,
    pub host: &'static str,
    pub port: u16,
    pub security: &'static str,
    pub help: &'static str,
}

pub const PRESETS: &[Preset] = &[
    Preset { id: "gmail", name: "Gmail", host: "smtp.gmail.com", port: 587, security: "starttls",
        help: "Turn on 2-Step Verification, then create an app password at myaccount.google.com/apppasswords." },
    Preset { id: "icloud", name: "iCloud Mail", host: "smtp.mail.me.com", port: 587, security: "starttls",
        help: "Create an app-specific password at account.apple.com → Sign-In and Security. Use your @icloud.com address." },
    Preset { id: "outlook", name: "Outlook.com", host: "smtp-mail.outlook.com", port: 587, security: "starttls",
        help: "Turn on two-step verification, then create an app password in your Microsoft account's security settings. Work accounts may need IT to allow SMTP." },
    Preset { id: "proton", name: "Proton Mail (Bridge)", host: "127.0.0.1", port: 1025, security: "starttls",
        help: "Proton needs its Bridge app running on this Mac. Copy the SMTP port and the Bridge password (not your Proton password) from Bridge's mailbox settings." },
    Preset { id: "other", name: "Other server", host: "", port: 587, security: "starttls",
        help: "For a work or agency mail relay: ask IT for the server, port and whether it needs a sign-in." },
];

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Settings {
    pub provider: String,
    pub host: String,
    pub port: u16,
    /// starttls (port 587), tls (port 465) or none (a relay inside a network).
    pub security: String,
    /// The sign-in, usually the email address; empty for a relay without one.
    pub username: String,
    pub password: String,
    pub from: String,
    /// Comma-separated.
    pub to: String,
}

pub fn load(db: &Database) -> Option<Settings> {
    db.setting(KEY).ok().flatten().and_then(|s| serde_json::from_str(&s).ok())
}

pub fn save(db: &Database, settings: Option<&Settings>) -> Result<(), String> {
    let value = settings.map(|s| serde_json::to_string(s).unwrap_or_default()).unwrap_or_default();
    db.set_setting(KEY, &value).map_err(|e| e.to_string())
}

/// Checks settings before they are saved, so a test or a run does not fail
/// on a typo.
pub fn validate(s: &Settings) -> Result<(), String> {
    if s.host.trim().is_empty() || s.host.contains(char::is_whitespace) {
        return Err("Enter the mail server, like smtp.gmail.com".into());
    }
    if s.port == 0 {
        return Err("Enter the server's port, usually 587".into());
    }
    if !["starttls", "tls", "none"].contains(&s.security.as_str()) {
        return Err("Security must be starttls, tls or none".into());
    }
    if s.security == "none" && !s.password.is_empty() {
        return Err("A password would be sent unencrypted. Use starttls or tls with a sign-in.".into());
    }
    s.from.parse::<Mailbox>().map_err(|_| format!("'{}' is not an email address", s.from))?;
    let to = recipients(&s.to)?;
    if to.is_empty() {
        return Err("Enter at least one address to send to".into());
    }
    Ok(())
}

fn recipients(list: &str) -> Result<Vec<Mailbox>, String> {
    let to: Vec<&str> = list.split([',', ';', '\n']).map(str::trim).filter(|a| !a.is_empty()).collect();
    if to.len() > MAX_RECIPIENTS {
        return Err(format!("At most {MAX_RECIPIENTS} addresses"));
    }
    to.into_iter().map(|a| a.parse::<Mailbox>().map_err(|_| format!("'{a}' is not an email address"))).collect()
}

fn is_this_machine(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

fn transport(s: &Settings) -> Result<AsyncSmtpTransport<Tokio1Executor>, String> {
    let tls = |host: &str| {
        // Proton Bridge on this Mac uses a self-signed certificate; it is
        // accepted only for this machine, never for a server elsewhere.
        TlsParameters::builder(host.to_string())
            .dangerous_accept_invalid_certs(is_this_machine(host))
            .build_rustls()
            .map_err(|e| format!("TLS setup failed: {e}"))
    };
    let mut builder = AsyncSmtpTransport::<Tokio1Executor>::builder_dangerous(s.host.trim()).port(s.port).timeout(Some(Duration::from_secs(30)));
    builder = match s.security.as_str() {
        "starttls" => builder.tls(Tls::Required(tls(s.host.trim())?)),
        "tls" => builder.tls(Tls::Wrapper(tls(s.host.trim())?)),
        _ => builder.tls(Tls::None),
    };
    if !s.username.trim().is_empty() {
        builder = builder.credentials(Credentials::new(s.username.trim().to_string(), s.password.clone()));
    }
    Ok(builder.build())
}

/// Sends one email, with an optional CSV attachment.
pub async fn send(s: &Settings, subject: &str, body: &str, csv: Option<(String, Vec<u8>)>) -> Result<(), String> {
    validate(s)?;
    let mut message = Message::builder().from(s.from.parse::<Mailbox>().map_err(|e| e.to_string())?).subject(subject);
    for to in recipients(&s.to)? {
        message = message.to(to);
    }
    let text = SinglePart::plain(body.to_string());
    let email = match csv {
        Some((name, bytes)) => message.multipart(
            MultiPart::mixed()
                .singlepart(text)
                .singlepart(Attachment::new(name).body(bytes, ContentType::parse("text/csv; charset=utf-8").map_err(|e| e.to_string())?)),
        ),
        None => message.singlepart(text),
    }
    .map_err(|e| format!("Could not build the email: {e}"))?;
    transport(s)?.send(email).await.map(|_| ()).map_err(|e| explain(&e.to_string()))
}

/// Mail servers answer sign-in problems tersely; say what to check.
fn explain(error: &str) -> String {
    let lower = error.to_lowercase();
    if lower.contains("535") || lower.contains("534") || lower.contains("authentication") || lower.contains("credentials") {
        format!("The mail server refused the sign-in. Use an app password, not your normal password. ({error})")
    } else if lower.contains("connection refused") {
        format!("Nothing answered at that server and port. For Proton, check that Bridge is running. ({error})")
    } else {
        format!("Sending failed: {error}")
    }
}

/// The email for a finished run: the whole report, picks and every source.
pub fn report(run: &super::Run) -> (String, String) {
    let date = run.started_at.get(..10).unwrap_or("");
    let subject = if run.status == "completed" {
        format!("{} · {date}", run.job_name)
    } else {
        format!("{} · {date} · run failed", run.job_name)
    };
    if run.status != "completed" {
        return (subject, format!("The run failed: {}\n\nOpen Ferro64's assistant view for details.", run.error.as_deref().unwrap_or("unknown error")));
    }
    let mut body = run.report.clone().unwrap_or_default();
    if !run.picks.is_empty() {
        body.push_str("\n\nPICKS (suggestions; you decide)\n");
        for p in &run.picks {
            let price = p.price.map(|x| format!(" at ${x:.2}")).unwrap_or_default();
            body.push_str(&format!("- {} {}{price}{}: {}\n", p.symbol, p.action, if p.conviction.is_empty() { String::new() } else { format!(", {} conviction", p.conviction) }, p.thesis));
        }
    }
    if let Some(t) = &run.table {
        body.push_str(&format!("\n{} new rows in the attached CSV{}.\n", t.rows.len(), if t.repeats.is_empty() { String::new() } else { format!("; {} already saved were left out", t.repeats.len()) }));
    }
    if !run.sources.is_empty() {
        body.push_str("\nSOURCES\n");
        for s in &run.sources {
            body.push_str(&format!("[S{}] {} — {}\n", s.n, s.title, s.url));
        }
    }
    body.push_str("\n—\nSent by your Ferro64 assistant, from your own computer.");
    (subject, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gmail() -> Settings {
        Settings { provider: "gmail".into(), host: "smtp.gmail.com".into(), port: 587, security: "starttls".into(), username: "me@example.com".into(), password: "abcd efgh ijkl mnop".into(), from: "me@example.com".into(), to: "me@example.com, Team <team@example.com>".into() }
    }

    #[test]
    fn settings_are_checked_before_saving() {
        assert!(validate(&gmail()).is_ok());
        assert!(validate(&Settings { to: "not an address".into(), ..gmail() }).is_err());
        assert!(validate(&Settings { to: String::new(), ..gmail() }).is_err());
        assert!(validate(&Settings { host: "smtp gmail".into(), ..gmail() }).is_err());
        assert!(validate(&Settings { security: "none".into(), ..gmail() }).is_err(), "a password must not go out unencrypted");
        assert!(validate(&Settings { security: "none".into(), username: String::new(), password: String::new(), host: "relay.agency.example".into(), port: 25, ..gmail() }).is_ok());
        let many = (0..11).map(|i| format!("a{i}@example.com")).collect::<Vec<_>>().join(",");
        assert!(validate(&Settings { to: many, ..gmail() }).is_err());
    }

    #[test]
    fn only_this_machine_skips_certificate_checks() {
        assert!(is_this_machine("127.0.0.1") && is_this_machine("localhost"));
        assert!(!is_this_machine("smtp.gmail.com") && !is_this_machine("127.0.0.1.evil.example"));
    }

    #[test]
    fn every_preset_but_other_is_complete() {
        for p in PRESETS {
            assert!(p.port > 0 && !p.help.is_empty());
            assert_eq!(p.host.is_empty(), p.id == "other");
        }
    }
}
