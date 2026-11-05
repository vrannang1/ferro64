//! Models on someone else's computer, for the coding agent. Off until a key is
//! added, and every request sends the project's code to that vendor.
//!
//! Most vendors speak the OpenAI chat format. Claude has its own Messages API, so
//! that format is translated here; the agent's transcript stays OpenAI-shaped.

use crate::db::Database;
use crate::llm::{ChatMessagePayload, ChatOptions, ChatOutcome, ChatStats, Emitter, ToolCall, ToolTurn};
use actix_web::{http::StatusCode, web, HttpResponse};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::{LazyLock, RwLock};
use std::time::{Duration, Instant};

const SETTING: &str = "hosted_models";
/// A run stops here, input and output together, unless changed in the tab.
pub const DEFAULT_TOKEN_LIMIT: u64 = 10_000_000;
const RETRY_WAITS: [u64; 4] = [5, 15, 45, 90];
const PREFIX: &str = "api:";

#[derive(Clone, Copy, PartialEq)]
enum Format {
    OpenAi,
    /// OpenAI's own newer models: `max_completion_tokens`, default temperature.
    OpenAiNew,
    Claude,
}

pub struct Vendor {
    pub id: &'static str,
    pub name: &'static str,
    env: &'static str,
    chat_url: &'static str,
    models_url: &'static str,
    keys_url: &'static str,
    format: Format,
}

pub const VENDORS: &[Vendor] = &[
    Vendor { id: "novita", name: "Novita", env: "NOVITA_API_KEY", chat_url: "https://api.novita.ai/openai/v1/chat/completions", models_url: "https://api.novita.ai/openai/v1/models", keys_url: "https://novita.ai/settings/key-management", format: Format::OpenAi },
    Vendor { id: "groq", name: "Groq", env: "GROQ_API_KEY", chat_url: "https://api.groq.com/openai/v1/chat/completions", models_url: "https://api.groq.com/openai/v1/models", keys_url: "https://console.groq.com/keys", format: Format::OpenAi },
    Vendor { id: "openai", name: "OpenAI", env: "OPENAI_API_KEY", chat_url: "https://api.openai.com/v1/chat/completions", models_url: "https://api.openai.com/v1/models", keys_url: "https://platform.openai.com/api-keys", format: Format::OpenAiNew },
    Vendor { id: "anthropic", name: "Claude", env: "ANTHROPIC_API_KEY", chat_url: "https://api.anthropic.com/v1/messages", models_url: "https://api.anthropic.com/v1/models?limit=1000", keys_url: "https://platform.claude.com/settings/keys", format: Format::Claude },
    Vendor { id: "gemini", name: "Gemini", env: "GEMINI_API_KEY", chat_url: "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", models_url: "https://generativelanguage.googleapis.com/v1beta/openai/models", keys_url: "https://aistudio.google.com/apikey", format: Format::OpenAi },
    Vendor { id: "xai", name: "xAI", env: "XAI_API_KEY", chat_url: "https://api.x.ai/v1/chat/completions", models_url: "https://api.x.ai/v1/models", keys_url: "https://console.x.ai", format: Format::OpenAi },
    Vendor { id: "custom", name: "Custom", env: "", chat_url: "", models_url: "", keys_url: "", format: Format::OpenAi },
];

/// The environment names of the vendors' keys, which commands must never see.
pub fn key_names() -> impl Iterator<Item = &'static str> {
    VENDORS.iter().map(|vendor| vendor.env).filter(|name| !name.is_empty())
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct SavedVendor {
    #[serde(default)]
    key: String,
    #[serde(default)]
    models: Vec<String>,
    /// Custom only: the address that ends before /chat/completions.
    #[serde(default)]
    url: String,
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct Saved {
    #[serde(default)]
    vendors: BTreeMap<String, SavedVendor>,
    #[serde(default)]
    token_limit: Option<u64>,
    /// What other tools send to use the proxy. Not a vendor's key.
    #[serde(default)]
    proxy_token: String,
}

/// Keys from the environment are read once, before commands' copies are cleaned.
static ENV_KEYS: LazyLock<BTreeMap<&'static str, String>> = LazyLock::new(|| {
    VENDORS.iter().filter(|v| !v.env.is_empty()).filter_map(|v| std::env::var(v.env).ok().filter(|k| !k.trim().is_empty()).map(|k| (v.id, k.trim().to_string()))).collect()
});
static SAVED: LazyLock<RwLock<Saved>> = LazyLock::new(|| RwLock::new(Saved::default()));
static CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder().connect_timeout(Duration::from_secs(15)).read_timeout(Duration::from_secs(600)).build().expect("HTTP client initialization failed")
});

/// Hidden unless asked for: Ferro64 is local, and this costs money per token.
pub fn enabled() -> bool {
    std::env::var("FERRO64_API_MODELS").is_ok_and(|value| matches!(value.trim(), "on" | "1" | "true"))
}

/// Loads what was saved. Call at startup, before commands can run.
pub fn load(db: &Database) {
    LazyLock::force(&ENV_KEYS);
    let mut saved: Saved = db.setting(SETTING).ok().flatten().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default();
    if saved.proxy_token.is_empty() {
        saved.proxy_token = format!("ferro64-{}", uuid::Uuid::new_v4().simple());
        if let Err(error) = store(db, saved.clone()) {
            tracing::warn!("Could not save the proxy token: {error}");
        }
    }
    if let Ok(mut current) = SAVED.write() {
        *current = saved;
    }
}

fn saved() -> Saved {
    SAVED.read().map(|saved| saved.clone()).unwrap_or_default()
}

fn store(db: &Database, saved: Saved) -> Result<(), String> {
    db.set_setting(SETTING, &serde_json::to_string(&saved).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if let Ok(mut current) = SAVED.write() {
        *current = saved;
    }
    Ok(())
}

pub fn token_limit() -> u64 {
    saved().token_limit.unwrap_or(DEFAULT_TOKEN_LIMIT)
}

fn vendor(id: &str) -> Option<&'static Vendor> {
    VENDORS.iter().find(|vendor| vendor.id == id)
}

/// A saved key wins, so what is typed in the tab is what is used.
fn key_for(saved: &Saved, id: &str) -> Option<String> {
    saved.vendors.get(id).map(|v| v.key.clone()).filter(|k| !k.is_empty()).or_else(|| ENV_KEYS.get(id).cloned())
}

pub fn is_hosted(model: &str) -> bool {
    model.starts_with(PREFIX)
}

fn parse(model: &str) -> Option<(&str, &str)> {
    model.strip_prefix(PREFIX)?.split_once(':').filter(|(_, name)| !name.is_empty())
}

/// Whether this hosted model can be used now.
pub fn available(model: &str) -> bool {
    let saved = saved();
    enabled() && parse(model).is_some_and(|(id, name)| {
        key_for(&saved, id).is_some() && saved.vendors.get(id).is_some_and(|v| v.models.iter().any(|m| m == name)) && (id != "custom" || saved.vendors.get(id).is_some_and(|v| !v.url.is_empty()))
    })
}

struct Target {
    name: String,
    format: Format,
    url: String,
    key: String,
}

fn target(model: &str) -> Result<Target, String> {
    let saved = saved();
    let (id, name) = parse(model).ok_or_else(|| format!("'{model}' is not a hosted model"))?;
    let vendor = vendor(id).ok_or_else(|| format!("Unknown vendor '{id}'"))?;
    let key = key_for(&saved, id).ok_or_else(|| format!("No API key for {}. Add one in the API models tab.", vendor.name))?;
    let url = if id == "custom" {
        format!("{}/chat/completions", saved.vendors.get(id).map(|v| v.url.as_str()).unwrap_or("").trim_end_matches('/'))
    } else {
        vendor.chat_url.to_string()
    };
    Ok(Target { name: name.to_string(), format: vendor.format, url, key })
}

fn authorized(request: reqwest::RequestBuilder, format: Format, key: &str) -> reqwest::RequestBuilder {
    match format {
        Format::Claude => request.header("x-api-key", key).header("anthropic-version", "2023-06-01"),
        _ => request.bearer_auth(key),
    }
}

/// What a vendor said went wrong, short, with the key taken out if it was echoed.
fn safe_error(text: &str, key: &str) -> String {
    let detail = serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|v| v.pointer("/error/message").and_then(Value::as_str).map(str::to_string).or_else(|| v.get("message").and_then(Value::as_str).map(str::to_string)))
        .unwrap_or_else(|| text.to_string());
    detail.replace(key, "[key]").chars().take(400).collect()
}

// ---------- Claude's format ----------

fn claude_tools(tools: &Value) -> Value {
    Value::from(
        tools
            .as_array()
            .into_iter()
            .flatten()
            .map(|tool| {
                let function = &tool["function"];
                // Large file writes stream as they are written, instead of a long silence.
                json!({"name": function["name"], "description": function["description"], "input_schema": function["parameters"], "eager_input_streaming": true})
            })
            .collect::<Vec<_>>(),
    )
}

/// Splits an OpenAI-shaped transcript into Claude's system text and messages.
fn claude_messages(messages: &[Value]) -> (String, Vec<Value>) {
    let mut system = Vec::new();
    let mut out: Vec<Value> = Vec::new();
    for message in messages {
        let text = message["content"].as_str().unwrap_or("");
        match message["role"].as_str().unwrap_or("") {
            "system" => system.push(text.to_string()),
            "assistant" => {
                let mut blocks = Vec::new();
                if !text.trim().is_empty() {
                    blocks.push(json!({"type": "text", "text": text}));
                }
                for call in message["tool_calls"].as_array().into_iter().flatten() {
                    let input = call["function"]["arguments"].as_str().and_then(|a| serde_json::from_str::<Value>(a).ok()).filter(Value::is_object).unwrap_or_else(|| json!({}));
                    blocks.push(json!({"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": input}));
                }
                if !blocks.is_empty() {
                    out.push(json!({"role": "assistant", "content": blocks}));
                }
            }
            "tool" => {
                let block = json!({"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": text});
                // Results of one turn's calls belong in a single user message.
                match out.last_mut().filter(|last| last["role"] == "user" && last["content"].is_array()) {
                    Some(last) => last["content"].as_array_mut().expect("checked above").push(block),
                    None => out.push(json!({"role": "user", "content": [block]})),
                }
            }
            _ => out.push(json!({"role": "user", "content": text})),
        }
    }
    (system.join("\n\n"), out)
}

/// Older Claude models reject `output_config.effort`.
fn claude_takes_effort(model: &str) -> bool {
    !["claude-3", "-4-5", "-4-1", "-4-0", "sonnet-4-2", "opus-4-2"].iter().any(|old| model.contains(old))
}

/// Models where a safety decline can be rerun on another model automatically.
fn claude_takes_fallback(model: &str) -> bool {
    ["claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"].contains(&model)
}

fn claude_body(name: &str, messages: &[Value], tools: Option<&Value>, options: ChatOptions) -> Value {
    let (system, messages) = claude_messages(messages);
    // No temperature: current Claude models reject it. Top-level cache_control
    // caches the conversation so far, which each step resends.
    let mut body = json!({"model": name, "max_tokens": options.max_tokens(true), "stream": true, "messages": messages, "cache_control": {"type": "ephemeral"}});
    if !system.is_empty() {
        body["system"] = Value::from(system);
    }
    if let Some(tools) = tools {
        body["tools"] = claude_tools(tools);
    }
    if claude_takes_effort(name) {
        body["output_config"] = json!({"effort": options.effort.as_str()});
    }
    if claude_takes_fallback(name) {
        body["fallbacks"] = Value::from("default");
    }
    body
}

/// Applies one line of Claude's stream.
fn apply_claude(line: &str, emitter: &mut Emitter, stats: &mut ChatStats, calls: &mut Vec<ToolCall>, blocks: &mut BTreeMap<u64, usize>) -> Result<bool, String> {
    let Some(data) = line.trim().strip_prefix("data:") else { return Ok(false) };
    let Ok(value) = serde_json::from_str::<Value>(data.trim()) else { return Ok(false) };
    match value["type"].as_str().unwrap_or("") {
        "message_start" => {
            let usage = &value["message"]["usage"];
            let read = usage["cache_read_input_tokens"].as_u64().unwrap_or(0);
            let total = usage["input_tokens"].as_u64().unwrap_or(0) + read + usage["cache_creation_input_tokens"].as_u64().unwrap_or(0);
            stats.prompt_tokens = Some(total);
            stats.cached_tokens = Some(read);
        }
        "content_block_start" if value["content_block"]["type"] == "tool_use" => {
            calls.push(ToolCall {
                id: value["content_block"]["id"].as_str().unwrap_or("").to_string(),
                name: value["content_block"]["name"].as_str().unwrap_or("").to_string(),
                arguments: String::new(),
            });
            blocks.insert(value["index"].as_u64().unwrap_or(0), calls.len() - 1);
        }
        "content_block_delta" => {
            let delta = &value["delta"];
            match delta["type"].as_str().unwrap_or("") {
                "text_delta" => emitter.content(delta["text"].as_str().unwrap_or("")),
                "thinking_delta" => emitter.reasoning(delta["thinking"].as_str().unwrap_or("")),
                "input_json_delta" => {
                    if let Some(call) = blocks.get(&value["index"].as_u64().unwrap_or(0)).and_then(|i| calls.get_mut(*i)) {
                        call.arguments.push_str(delta["partial_json"].as_str().unwrap_or(""));
                    }
                }
                _ => {}
            }
        }
        "message_delta" => {
            if let Some(output) = value["usage"]["output_tokens"].as_u64() {
                stats.completion_tokens = Some(output);
            }
            match value["delta"]["stop_reason"].as_str() {
                Some("refusal") => {
                    let why = value["delta"]["stop_details"]["explanation"].as_str().unwrap_or("no reason given");
                    return Err(format!("Claude declined this request: {why}"));
                }
                Some("max_tokens") => stats.finish_reason = Some("length".to_string()),
                Some("tool_use") => stats.finish_reason = Some("tool_calls".to_string()),
                Some(_) => stats.finish_reason = Some("stop".to_string()),
                None => {}
            }
        }
        "message_stop" => return Ok(true),
        "error" => return Err(format!("Claude: {}", value["error"]["message"].as_str().unwrap_or("request failed"))),
        _ => {}
    }
    Ok(false)
}

// ---------- The OpenAI format ----------

fn openai_body(target: &Target, messages: &[Value], tools: Option<&Value>, options: ChatOptions) -> Value {
    let mut body = json!({"model": target.name, "messages": messages, "stream": true, "stream_options": {"include_usage": true}});
    if target.format == Format::OpenAiNew {
        body["max_completion_tokens"] = Value::from(options.max_tokens(true));
    } else {
        body["max_tokens"] = Value::from(options.max_tokens(true));
        body["temperature"] = Value::from(options.temperature.value());
    }
    if let Some(tools) = tools {
        body["tools"] = tools.clone();
    }
    body
}

fn apply_openai(line: &str, emitter: &mut Emitter, stats: &mut ChatStats, calls: &mut Vec<ToolCall>) -> Result<bool, String> {
    let Some(data) = line.trim().strip_prefix("data:") else { return Ok(false) };
    let data = data.trim();
    if data == "[DONE]" {
        return Ok(true);
    }
    let Ok(value) = serde_json::from_str::<Value>(data) else { return Ok(false) };
    if let Some(error) = value.get("error") {
        return Err(error["message"].as_str().unwrap_or("request failed").to_string());
    }
    if let Some(choice) = value.pointer("/choices/0") {
        let delta = &choice["delta"];
        // Vendors differ in the name of this field.
        if let Some(text) = delta["reasoning_content"].as_str().or_else(|| delta["reasoning"].as_str()) {
            emitter.reasoning(text);
        }
        if let Some(text) = delta["content"].as_str() {
            emitter.content(text);
        }
        for part in delta["tool_calls"].as_array().into_iter().flatten() {
            let index = part["index"].as_u64().unwrap_or(calls.len() as u64) as usize;
            if calls.len() <= index {
                calls.resize(index + 1, ToolCall::default());
            }
            let call = &mut calls[index];
            if let Some(id) = part["id"].as_str().filter(|id| !id.is_empty()) {
                call.id = id.to_string();
            }
            if let Some(name) = part.pointer("/function/name").and_then(Value::as_str) {
                call.name.push_str(name);
            }
            if let Some(arguments) = part.pointer("/function/arguments").and_then(Value::as_str) {
                call.arguments.push_str(arguments);
            }
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            stats.finish_reason = Some(reason.to_string());
        }
    }
    if let Some(usage) = value.get("usage").filter(|usage| usage.is_object()) {
        stats.prompt_tokens = usage["prompt_tokens"].as_u64().or(stats.prompt_tokens);
        stats.completion_tokens = usage["completion_tokens"].as_u64().or(stats.completion_tokens);
        stats.cached_tokens = usage.pointer("/prompt_tokens_details/cached_tokens").and_then(Value::as_u64).or(stats.cached_tokens);
    }
    Ok(false)
}

// ---------- Sending ----------

fn can_retry(status: u16) -> bool {
    matches!(status, 408 | 429 | 500..=599)
}

/// Sends with waits for busy vendors and dropped connections. A rejected setting
/// comes back once more without it, because models disagree about these.
async fn send(target: &Target, mut body: Value, emitter: &mut Emitter) -> Result<reqwest::Response, String> {
    let mut waits = RETRY_WAITS.iter();
    loop {
        let mut request = authorized(CLIENT.post(&target.url), target.format, &target.key).json(&body);
        if body.get("fallbacks").is_some() {
            request = request.header("anthropic-beta", "server-side-fallback-2026-07-01");
        }
        let failure = match request.send().await {
            Ok(response) if response.status().is_success() => return Ok(response),
            Ok(response) => {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                let detail = safe_error(&text, &target.key);
                if status.as_u16() == 400 {
                    let lower = detail.to_lowercase();
                    let dropped: Vec<&str> = ["temperature", "stream_options", "output_config", "fallbacks"].into_iter().filter(|field| lower.contains(field) && body.get(*field).is_some()).collect();
                    if !dropped.is_empty() {
                        for field in dropped {
                            body.as_object_mut().expect("the body is an object").remove(field);
                        }
                        continue;
                    }
                }
                if !can_retry(status.as_u16()) {
                    return Err(format!("The vendor refused the request ({status}): {detail}"));
                }
                format!("{status}: {detail}")
            }
            // reqwest's error can name the URL but never the headers.
            Err(error) => format!("could not reach the vendor: {error}"),
        };
        match waits.next() {
            Some(wait) => {
                emitter.status(format!("the vendor is busy; trying again in {wait}s"));
                tokio::time::sleep(Duration::from_secs(*wait)).await;
            }
            None => return Err(format!("The vendor kept failing: {failure}")),
        }
    }
}

async fn read_stream(target: &Target, mut response: reqwest::Response, emitter: &mut Emitter) -> Result<(ChatStats, Vec<ToolCall>), String> {
    emitter.status("generating");
    let (mut stats, mut calls, mut blocks, mut buffer) = (ChatStats::default(), Vec::new(), BTreeMap::new(), Vec::new());
    'read: while let Some(chunk) = response.chunk().await.map_err(|error| format!("The vendor's stream was interrupted: {error}"))? {
        buffer.extend_from_slice(&chunk);
        while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = buffer.drain(..=newline).collect();
            let line = String::from_utf8_lossy(&line);
            let done = match target.format {
                Format::Claude => apply_claude(&line, emitter, &mut stats, &mut calls, &mut blocks),
                _ => apply_openai(&line, emitter, &mut stats, &mut calls),
            }
            .map_err(|error| error.replace(&target.key, "[key]"))?;
            if done || emitter.closed {
                break 'read;
            }
        }
    }
    emitter.finish();
    Ok((stats, calls))
}

pub async fn chat_tools(model: &str, messages: &[Value], tools: &Value, options: ChatOptions, emitter: &mut Emitter) -> Result<ToolTurn, String> {
    let target = target(model)?;
    let start = Instant::now();
    let tools = Some(tools).filter(|tools| tools.as_array().is_some_and(|list| !list.is_empty()));
    let body = match target.format {
        Format::Claude => claude_body(&target.name, messages, tools, options),
        _ => openai_body(&target, messages, tools, options),
    };
    let response = send(&target, body, emitter).await?;
    let (stats, calls) = read_stream(&target, response, emitter).await?;
    Ok(ToolTurn {
        content: emitter.content.clone(),
        reasoning: emitter.reasoning.trim().to_string(),
        tool_calls: calls.into_iter().filter(|call| !call.name.is_empty()).collect(),
        duration_ms: start.elapsed().as_millis() as u64,
        stats,
        cancelled: emitter.closed,
    })
}

pub async fn chat(model: &str, messages: &[ChatMessagePayload], options: ChatOptions, mut emitter: Emitter) -> Result<ChatOutcome, String> {
    let target = target(model)?;
    let start = Instant::now();
    let messages: Vec<Value> = messages.iter().map(|m| json!({"role": m.role, "content": m.content})).collect();
    let body = match target.format {
        Format::Claude => claude_body(&target.name, &messages, None, options),
        _ => openai_body(&target, &messages, None, options),
    };
    let response = send(&target, body, &mut emitter).await?;
    let (stats, _) = read_stream(&target, response, &mut emitter).await?;
    let mut outcome = crate::llm::finalize(&emitter, stats)?;
    outcome.duration_ms = start.elapsed().as_millis() as u64;
    Ok(outcome)
}

// ---------- One address for every model ----------

fn proxy_error(status: StatusCode, message: impl Into<String>) -> HttpResponse {
    HttpResponse::build(status).json(json!({"error": {"message": message.into(), "type": "ferro64"}}))
}

/// This spends the owner's credit, so it takes this machine and the tab's token.
/// The token also keeps out web pages, and commands an agent runs.
fn proxy_refused(request: &actix_web::HttpRequest) -> Option<HttpResponse> {
    if !request.peer_addr().is_some_and(|peer| peer.ip().is_loopback()) {
        return Some(proxy_error(StatusCode::FORBIDDEN, "The proxy only answers this machine."));
    }
    let token = saved().proxy_token;
    let sent = request.headers().get("authorization").and_then(|value| value.to_str().ok()).and_then(|value| value.strip_prefix("Bearer ")).unwrap_or("");
    (token.is_empty() || sent.trim() != token).then(|| proxy_error(StatusCode::UNAUTHORIZED, "Send the token from the API models tab as the API key."))
}

/// Agents may send content as a list of parts. Text is joined; anything else is
/// refused rather than dropped without a word.
fn proxy_messages(messages: &Value) -> Result<Vec<Value>, String> {
    let list = messages.as_array().filter(|list| !list.is_empty()).ok_or("messages must be a list with something in it")?;
    let mut out = Vec::with_capacity(list.len());
    for message in list {
        let mut message = message.clone();
        if message["role"] == "developer" {
            message["role"] = Value::from("system");
        }
        if let Some(parts) = message["content"].as_array() {
            let mut text = Vec::new();
            for part in parts {
                match part["text"].as_str().filter(|_| part["type"] == "text") {
                    Some(piece) => text.push(piece),
                    None => return Err("The proxy takes text only.".to_string()),
                }
            }
            message["content"] = Value::from(text.join("\n"));
        } else if message["content"].is_null() {
            message["content"] = Value::from("");
        }
        out.push(message);
    }
    Ok(out)
}

fn proxy_calls(calls: &[ToolCall]) -> Vec<Value> {
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            let id = if call.id.is_empty() { format!("call_{index}") } else { call.id.clone() };
            json!({"index": index, "id": id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}})
        })
        .collect()
}

fn proxy_usage(stats: &ChatStats) -> Value {
    let (input, output) = (stats.prompt_tokens.unwrap_or(0), stats.completion_tokens.unwrap_or(0));
    json!({"prompt_tokens": input, "completion_tokens": output, "total_tokens": input + output, "prompt_tokens_details": {"cached_tokens": stats.cached_tokens.unwrap_or(0)}})
}

fn proxy_finish(turn: &ToolTurn) -> String {
    let default = if turn.tool_calls.is_empty() { "stop" } else { "tool_calls" };
    turn.stats.finish_reason.clone().unwrap_or_else(|| default.to_string())
}

fn proxy_chunk(id: &str, model: &str, delta: Value, finish: Option<&str>, usage: Option<Value>) -> web::Bytes {
    let choices = if delta.is_null() { json!([]) } else { json!([{"index": 0, "delta": delta, "finish_reason": finish}]) };
    let mut chunk = json!({"id": id, "object": "chat.completion.chunk", "created": chrono::Utc::now().timestamp(), "model": model, "choices": choices});
    if let Some(usage) = usage {
        chunk["usage"] = usage;
    }
    web::Bytes::from(format!("data: {chunk}\n\n"))
}

async fn proxy_chat(request: actix_web::HttpRequest, coder: web::Data<crate::coder::Coder>, body: web::Json<Value>) -> HttpResponse {
    if let Some(refusal) = proxy_refused(&request) {
        return refusal;
    }
    let model = body["model"].as_str().unwrap_or("").to_string();
    let model = match coder.choose_model(Some(&model)).await {
        Ok(model) => model,
        Err(error) => return proxy_error(StatusCode::NOT_FOUND, error),
    };
    let messages = match proxy_messages(&body["messages"]) {
        Ok(messages) => messages,
        Err(error) => return proxy_error(StatusCode::BAD_REQUEST, error),
    };
    let tools = body.get("tools").filter(|tools| tools.is_array()).cloned().unwrap_or_else(|| json!([]));
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let options = ChatOptions::default();
    let coder = coder.into_inner();
    if body["stream"].as_bool() != Some(true) {
        return match coder.llm.chat_tools(&model, messages, tools, options, None).await {
            Ok(turn) => {
                let calls = proxy_calls(&turn.tool_calls);
                let mut message = json!({"role": "assistant", "content": turn.content});
                if !calls.is_empty() {
                    message["tool_calls"] = Value::from(calls);
                }
                HttpResponse::Ok().json(json!({
                    "id": id, "object": "chat.completion", "created": chrono::Utc::now().timestamp(), "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": proxy_finish(&turn)}], "usage": proxy_usage(&turn.stats),
                }))
            }
            Err(error) => proxy_error(StatusCode::BAD_GATEWAY, error),
        };
    }
    let (out, receiver) = tokio::sync::mpsc::unbounded_channel::<web::Bytes>();
    tokio::spawn(async move {
        let (sink, mut events) = tokio::sync::mpsc::unbounded_channel();
        let call = {
            let (coder, model) = (coder.clone(), model.clone());
            tokio::spawn(async move { coder.llm.chat_tools(&model, messages, tools, options, Some(sink)).await })
        };
        while let Some(event) = events.recv().await {
            let delta = match event {
                crate::llm::StreamEvent::Content(text) => json!({"role": "assistant", "content": text}),
                crate::llm::StreamEvent::Reasoning(text) => json!({"role": "assistant", "reasoning_content": text}),
                crate::llm::StreamEvent::Status(_) => continue,
            };
            // The client left: dropping `events` tells the call to stop.
            if out.send(proxy_chunk(&id, &model, delta, None, None)).is_err() {
                return;
            }
        }
        match call.await.map_err(|error| error.to_string()).and_then(|result| result) {
            Ok(turn) => {
                let calls = proxy_calls(&turn.tool_calls);
                if !calls.is_empty() {
                    let _ = out.send(proxy_chunk(&id, &model, json!({"role": "assistant", "tool_calls": calls}), None, None));
                }
                let _ = out.send(proxy_chunk(&id, &model, json!({}), Some(&proxy_finish(&turn)), None));
                let _ = out.send(proxy_chunk(&id, &model, Value::Null, None, Some(proxy_usage(&turn.stats))));
            }
            Err(error) => {
                let _ = out.send(web::Bytes::from(format!("data: {}\n\n", json!({"error": {"message": error}}))));
            }
        }
        let _ = out.send(web::Bytes::from_static(b"data: [DONE]\n\n"));
    });
    let stream = futures_util::StreamExt::map(tokio_stream::wrappers::UnboundedReceiverStream::new(receiver), Ok::<_, actix_web::Error>);
    HttpResponse::Ok().content_type("text/event-stream").insert_header(("Cache-Control", "no-cache")).insert_header(("X-Accel-Buffering", "no")).streaming(stream)
}

/// Hosted models switched on in the tab, and the chat models on this machine.
async fn proxy_models(request: actix_web::HttpRequest, coder: web::Data<crate::coder::Coder>) -> HttpResponse {
    if let Some(refusal) = proxy_refused(&request) {
        return refusal;
    }
    let all = saved();
    let mut models: Vec<Value> = Vec::new();
    for (id, vendor) in &all.vendors {
        if key_for(&all, id).is_some() {
            models.extend(vendor.models.iter().map(|name| json!({"id": format!("{PREFIX}{id}:{name}"), "object": "model", "owned_by": id})));
        }
    }
    for model in coder.llm.models().await.unwrap_or_default() {
        if model.installed && model.capabilities.iter().any(|c| c == "chat") {
            models.push(json!({"id": model.id, "object": "model", "owned_by": "local"}));
        }
    }
    HttpResponse::Ok().json(json!({"object": "list", "data": models}))
}

// ---------- The API models tab ----------

type Reply = Result<HttpResponse, actix_web::Error>;

fn fail(status: StatusCode, message: impl Into<String>) -> actix_web::Error {
    actix_web::error::InternalError::from_response("", HttpResponse::build(status).json(json!({"error": message.into()}))).into()
}

fn hint(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    if chars.len() < 12 {
        return "saved".to_string();
    }
    format!("…{}", chars[chars.len() - 4..].iter().collect::<String>())
}

/// Never includes a key: only where it came from and its last characters.
async fn list(request: actix_web::HttpRequest) -> Reply {
    let saved = saved();
    let here = request.peer_addr().is_some_and(|peer| peer.ip().is_loopback());
    let vendors: Vec<Value> = VENDORS
        .iter()
        .map(|vendor| {
            let mine = saved.vendors.get(vendor.id).cloned().unwrap_or_default();
            let source = if !mine.key.is_empty() { "saved" } else if ENV_KEYS.contains_key(vendor.id) { "environment" } else { "" };
            json!({
                "id": vendor.id, "name": vendor.name, "env": vendor.env, "keys_url": vendor.keys_url,
                "key_source": source, "key_hint": key_for(&saved, vendor.id).map(|key| hint(&key)),
                "models": mine.models, "url": mine.url,
            })
        })
        .collect();
    Ok(HttpResponse::Ok().json(json!({"vendors": vendors, "token_limit": token_limit(), "proxy_token": if here { saved.proxy_token.as_str() } else { "" }})))
}

#[derive(Deserialize)]
struct VendorUpdate {
    key: Option<String>,
    models: Option<Vec<String>>,
    url: Option<String>,
}

fn valid_url(url: &str) -> bool {
    let url = url.trim();
    url.starts_with("https://") || url.starts_with("http://localhost") || url.starts_with("http://127.0.0.1")
}

async fn update(request: actix_web::HttpRequest, coder: web::Data<crate::coder::Coder>, id: web::Path<String>, payload: web::Json<VendorUpdate>) -> Reply {
    let bad = |message: &str| fail(StatusCode::BAD_REQUEST, message);
    if vendor(&id).is_none() {
        return Err(fail(StatusCode::NOT_FOUND, "Unknown vendor"));
    }
    let mut all = saved();
    let entry = all.vendors.entry(id.to_string()).or_default();
    if let Some(key) = &payload.key {
        let key = key.trim();
        if key.len() > 400 || key.contains(char::is_whitespace) {
            return Err(bad("That does not look like an API key"));
        }
        entry.key = key.to_string();
    }
    if let Some(models) = &payload.models {
        let mut clean: Vec<String> = models.iter().map(|m| m.trim().to_string()).filter(|m| !m.is_empty()).collect();
        clean.dedup();
        if clean.len() > 40 || clean.iter().any(|m| m.len() > 200) {
            return Err(bad("Too many models, or a name too long"));
        }
        entry.models = clean;
    }
    if let Some(url) = &payload.url {
        // A key must not travel in the clear, other than to this machine.
        if !url.trim().is_empty() && !valid_url(url) {
            return Err(bad("The address must start with https://, or http://localhost for this machine"));
        }
        entry.url = url.trim().trim_end_matches('/').to_string();
    }
    store(&coder.db, all).map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    list(request).await
}

#[derive(Deserialize)]
struct LimitUpdate {
    token_limit: u64,
}

async fn update_limit(request: actix_web::HttpRequest, coder: web::Data<crate::coder::Coder>, payload: web::Json<LimitUpdate>) -> Reply {
    if payload.token_limit < 10_000 {
        return Err(fail(StatusCode::BAD_REQUEST, "The limit must be at least 10,000 tokens"));
    }
    let mut all = saved();
    all.token_limit = Some(payload.token_limit);
    store(&coder.db, all).map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    list(request).await
}

/// Asks the vendor what it offers. Costs nothing, and proves the key works.
async fn vendor_models(id: web::Path<String>) -> Reply {
    let vendor = vendor(&id).ok_or_else(|| fail(StatusCode::NOT_FOUND, "Unknown vendor"))?;
    let all = saved();
    let key = key_for(&all, vendor.id).ok_or_else(|| fail(StatusCode::BAD_REQUEST, "Add a key first"))?;
    let url = if vendor.id == "custom" {
        format!("{}/models", all.vendors.get("custom").map(|v| v.url.as_str()).unwrap_or(""))
    } else {
        vendor.models_url.to_string()
    };
    let response = authorized(CLIENT.get(&url), vendor.format, &key).send().await.map_err(|e| fail(StatusCode::BAD_GATEWAY, format!("Could not reach {}: {e}", vendor.name)))?;
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(fail(StatusCode::BAD_GATEWAY, format!("{} said {status}: {}", vendor.name, safe_error(&text, &key))));
    }
    let value: Value = serde_json::from_str(&text).unwrap_or_default();
    let mut models: Vec<String> = value["data"].as_array().into_iter().flatten().filter_map(|m| m["id"].as_str()).map(|id| id.trim_start_matches("models/").to_string()).collect();
    models.sort();
    Ok(HttpResponse::Ok().json(json!({"models": models})))
}

pub fn routes(cfg: &mut web::ServiceConfig) {
    if !enabled() {
        return;
    }
    cfg.route("/api/hosted", web::get().to(list))
        .route("/api/hosted", web::put().to(update_limit))
        .route("/api/hosted/{id}", web::put().to(update))
        .route("/api/hosted/{id}/models", web::get().to(vendor_models))
        .route("/v1/models", web::get().to(proxy_models))
        // Other agents send whole conversations.
        .service(web::resource("/v1/chat/completions").app_data(web::JsonConfig::default().limit(64 * 1024 * 1024)).route(web::post().to(proxy_chat)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_ids_split_at_the_first_two_colons_only() {
        assert_eq!(parse("api:novita:qwen/qwen3:latest"), Some(("novita", "qwen/qwen3:latest")));
        assert_eq!(parse("api:novita:"), None);
        assert_eq!(parse("qwen3.6:35b-a3b"), None);
        assert!(is_hosted("api:groq:x") && !is_hosted("qwen3.6:35b-a3b"));
    }

    #[test]
    fn every_key_name_is_hidden_from_commands() {
        for name in key_names() {
            assert!(crate::coder::is_own_setting(name), "{name}");
        }
        assert_eq!(key_names().count(), 6);
    }

    #[test]
    fn a_transcript_becomes_claudes_format() {
        let messages = vec![
            json!({"role": "system", "content": "rules"}),
            json!({"role": "user", "content": "fix it"}),
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"x\"}"}},
                {"id": "b", "type": "function", "function": {"name": "list_dir", "arguments": "not json"}},
            ]}),
            json!({"role": "tool", "tool_call_id": "a", "content": "one"}),
            json!({"role": "tool", "tool_call_id": "b", "content": "two"}),
            json!({"role": "assistant", "content": ""}),
            json!({"role": "user", "content": "[automatic] go on"}),
        ];
        let (system, out) = claude_messages(&messages);
        assert_eq!(system, "rules");
        assert_eq!(out.len(), 4, "the empty assistant turn is dropped: {out:?}");
        assert_eq!(out[1]["content"][0]["input"]["path"], "x");
        assert_eq!(out[1]["content"][1]["input"], json!({}), "bad arguments become an empty object");
        assert_eq!(out[2]["content"].as_array().unwrap().len(), 2, "one turn's results share a message");
        assert_eq!(out[2]["content"][1]["tool_use_id"], "b");
        assert_eq!(out[3]["content"], "[automatic] go on");
    }

    #[test]
    fn claude_requests_leave_out_what_current_models_reject() {
        let tools = json!([{"type": "function", "function": {"name": "n", "description": "d", "parameters": {"type": "object"}}}]);
        let body = claude_body("claude-opus-5-5", &[json!({"role": "user", "content": "hi"})], Some(&tools), ChatOptions::default());
        assert!(body.get("temperature").is_none());
        assert!(body.get("thinking").is_none());
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(body["tools"][0]["eager_input_streaming"], true);
        assert_eq!(body["fallbacks"], "default");
        let old = claude_body("claude-haiku-4-5", &[json!({"role": "user", "content": "hi"})], None, ChatOptions::default());
        assert!(old.get("output_config").is_none() && old.get("fallbacks").is_none());
    }

    #[test]
    fn claudes_stream_gives_text_calls_and_token_counts() {
        let mut emitter = Emitter::new(None);
        let (mut stats, mut calls, mut blocks) = (ChatStats::default(), Vec::new(), BTreeMap::new());
        for line in [
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":10,"cache_read_input_tokens":90,"cache_creation_input_tokens":5}}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#,
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"read_file"}}"#,
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"pa"}}"#,
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"th\":1}"}}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}"#,
        ] {
            assert!(!apply_claude(line, &mut emitter, &mut stats, &mut calls, &mut blocks).unwrap());
        }
        emitter.finish();
        assert_eq!(emitter.content, "hello");
        assert_eq!((calls[0].id.as_str(), calls[0].name.as_str(), calls[0].arguments.as_str()), ("t1", "read_file", "{\"path\":1}"));
        assert_eq!((stats.prompt_tokens, stats.cached_tokens, stats.completion_tokens), (Some(105), Some(90), Some(7)));
        assert_eq!(stats.finish_reason.as_deref(), Some("tool_calls"));
        let refused = r#"data: {"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"explanation":"no"}}}"#;
        assert!(apply_claude(refused, &mut emitter, &mut stats, &mut calls, &mut blocks).unwrap_err().contains("declined"));
    }

    #[test]
    fn the_openai_stream_gives_calls_and_token_counts() {
        let mut emitter = Emitter::new(None);
        let (mut stats, mut calls) = (ChatStats::default(), Vec::new());
        for line in [
            r#"data: {"choices":[{"delta":{"reasoning":"hm","tool_calls":[{"index":0,"id":"c","function":{"name":"read_file","arguments":"{}"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":80}}}"#,
        ] {
            assert!(!apply_openai(line, &mut emitter, &mut stats, &mut calls).unwrap());
        }
        assert!(apply_openai("data: [DONE]", &mut emitter, &mut stats, &mut calls).unwrap());
        assert_eq!(calls[0].name, "read_file");
        assert_eq!((stats.prompt_tokens, stats.completion_tokens, stats.cached_tokens), (Some(100), Some(5), Some(80)));
    }

    #[test]
    fn openais_own_models_use_their_token_setting() {
        let target = Target { name: "m".into(), format: Format::OpenAiNew, url: String::new(), key: String::new() };
        let body = openai_body(&target, &[], None, ChatOptions::default());
        assert!(body.get("max_completion_tokens").is_some() && body.get("max_tokens").is_none() && body.get("temperature").is_none());
    }

    #[test]
    fn the_proxy_takes_text_parts_and_refuses_the_rest() {
        let messages = proxy_messages(&json!([
            {"role": "developer", "content": "rules"},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "assistant", "content": null, "tool_calls": []},
        ])).unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["content"], "a
b");
        assert_eq!(messages[2]["content"], "");
        assert!(proxy_messages(&json!([{"role": "user", "content": [{"type": "image_url"}]}])).is_err());
        assert!(proxy_messages(&json!([])).is_err());
    }

    #[test]
    fn errors_and_the_tab_do_not_show_a_key() {
        assert_eq!(safe_error(r#"{"error":{"message":"bad key sk-secret"}}"#, "sk-secret"), "bad key [key]");
        assert_eq!(hint("abcdefghijklmnop"), "…mnop");
        assert_eq!(hint("short"), "saved");
        assert!(valid_url("https://x.ai") && valid_url("http://localhost:8000") && !valid_url("http://example.com"));
    }
}
