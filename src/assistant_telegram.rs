//! Reports to the user's own Telegram chat. Ferro64 sends them after a run;
//! the model has no tool for it and cannot choose where they go. The bot
//! token and chat are set in the assistant view and kept in SQLite.

use crate::db::Database;
use serde_json::{json, Value};

pub const TOKEN_KEY: &str = "assistant_telegram_token";
pub const CHAT_KEY: &str = "assistant_telegram_chat";
/// Telegram's limit is 4,096 characters a message.
const MESSAGE_CHARS: usize = 3_900;

pub fn valid_token(token: &str) -> bool {
    let Some((id, secret)) = token.split_once(':') else { return false };
    !id.is_empty() && id.chars().all(|c| c.is_ascii_digit()) && secret.len() >= 30 && secret.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

pub fn valid_chat(chat: &str) -> bool {
    chat.trim_start_matches('-').len() >= 3 && chat.trim_start_matches('-').chars().all(|c| c.is_ascii_digit())
}

pub fn settings(db: &Database) -> (Option<String>, Option<String>) {
    let get = |k| db.setting(k).ok().flatten().filter(|v: &String| !v.is_empty());
    (get(TOKEN_KEY), get(CHAT_KEY))
}

pub fn hint(token: &str) -> String {
    let tail: String = token.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
    format!("…{tail}")
}

fn api(token: &str, method: &str) -> String {
    format!("https://api.telegram.org/bot{token}/{method}")
}

/// Telegram's reply, or its error. The token is part of the address, so it
/// is kept out of any error text that could reach a log.
async fn call(request: reqwest::RequestBuilder) -> Result<Value, String> {
    let response = request.send().await.map_err(|e| format!("Telegram is unreachable: {}", e.without_url()))?;
    let body: Value = response.json().await.map_err(|e| format!("Telegram sent an unreadable reply: {}", e.without_url()))?;
    if body["ok"].as_bool() != Some(true) {
        return Err(format!("Telegram: {}", body["description"].as_str().unwrap_or("request refused")));
    }
    Ok(body["result"].clone())
}

pub async fn send_text(client: &reqwest::Client, token: &str, chat: &str, text: &str) -> Result<(), String> {
    let text: String = text.chars().take(4_000).collect();
    let request = client.post(api(token, "sendMessage")).json(&json!({"chat_id": chat, "text": text, "disable_web_page_preview": true}));
    call(request).await.map(|_| ())
}

pub async fn send_file(client: &reqwest::Client, token: &str, chat: &str, name: &str, bytes: Vec<u8>, caption: &str) -> Result<(), String> {
    let part = reqwest::multipart::Part::bytes(bytes).file_name(name.to_string()).mime_str("text/csv").map_err(|e| e.to_string())?;
    let form = reqwest::multipart::Form::new().text("chat_id", chat.to_string()).text("caption", caption.chars().take(1_000).collect::<String>()).part("document", part);
    call(client.post(api(token, "sendDocument")).multipart(form)).await.map(|_| ())
}

/// The chats that recently messaged the bot, so the user can pick theirs
/// after sending it "hi".
pub async fn recent_chats(client: &reqwest::Client, token: &str) -> Result<Vec<Value>, String> {
    let updates = call(client.get(api(token, "getUpdates"))).await?;
    let mut chats: Vec<Value> = vec![];
    for update in updates.as_array().into_iter().flatten() {
        let chat = update.pointer("/message/chat").or_else(|| update.pointer("/channel_post/chat"));
        let Some(chat) = chat else { continue };
        let id = chat["id"].as_i64().map(|i| i.to_string()).unwrap_or_default();
        if id.is_empty() || chats.iter().any(|c| c["id"] == id) {
            continue;
        }
        let name = chat["title"].as_str().map(str::to_string).unwrap_or_else(|| {
            [chat["first_name"].as_str(), chat["last_name"].as_str()].into_iter().flatten().collect::<Vec<_>>().join(" ")
        });
        let name = if name.is_empty() { chat["username"].as_str().unwrap_or("chat").to_string() } else { name };
        chats.push(json!({"id": id, "name": name}));
    }
    Ok(chats)
}

/// The message for a finished run: the report, the picks and where the
/// sources are. Long reports are cut with a pointer back to Ferro64.
pub fn message(run: &super::Run) -> String {
    let date = run.started_at.get(..10).unwrap_or("");
    if run.status != "completed" {
        return format!("{} · {date}\nThe run failed: {}", run.job_name, run.error.as_deref().unwrap_or("unknown error"));
    }
    let mut tail = String::new();
    if !run.picks.is_empty() {
        tail.push_str("\n\nPicks (suggestions; you decide):\n");
        for p in &run.picks {
            let price = p.price.map(|x| format!(" at ${x:.2}")).unwrap_or_default();
            tail.push_str(&format!("• {} {}{price}{}\n", p.symbol, p.action, if p.conviction.is_empty() { String::new() } else { format!(", {} conviction", p.conviction) }));
        }
    }
    if let Some(t) = &run.table {
        tail.push_str(&format!("\n{} new rows{} (CSV attached)\n", t.rows.len(), if t.repeats.is_empty() { String::new() } else { format!(", {} already saved left out", t.repeats.len()) }));
    }
    let sources: Vec<String> = run.sources.iter().take(8).map(|s| format!("[S{}] {}", s.n, s.url)).collect();
    if !sources.is_empty() {
        tail.push_str(&format!("\nSources:\n{}", sources.join("\n")));
        if run.sources.len() > 8 {
            tail.push_str(&format!("\n…and {} more in Ferro64", run.sources.len() - 8));
        }
    }
    let head = format!("{} · {date}\n\n", run.job_name);
    let report = run.report.as_deref().unwrap_or("");
    let room = MESSAGE_CHARS.saturating_sub(head.chars().count() + tail.chars().count() + 40);
    let body: String = if report.chars().count() > room {
        format!("{}…\n(cut short; the full report is in Ferro64)", report.chars().take(room).collect::<String>())
    } else {
        report.to_string()
    };
    format!("{head}{body}{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_and_chats_are_checked() {
        assert!(valid_token("123456789:AAHfiqksKZ8WmR2zSjiQ7_v4TMAKdiHm9T0"));
        assert!(!valid_token("123456789"));
        assert!(!valid_token("abc:AAHfiqksKZ8WmR2zSjiQ7_v4TMAKdiHm9T0"));
        assert!(!valid_token("1:short"));
        assert!(valid_chat("123456789") && valid_chat("-1001234567890"));
        assert!(!valid_chat("@me") && !valid_chat("12"));
        assert_eq!(hint("123:abcdefWXYZ"), "…WXYZ");
    }
}
