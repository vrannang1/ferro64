//! Web search for chat: find pages and hand their text to the model as
//! numbered sources. The engine, first match wins:
//!   KEENABLE_API_KEY  Keenable with an account (100K free requests a month)
//!   BRAVE_API_KEY     Brave Search
//!   SEARXNG_URL       a self-hosted SearXNG with the JSON format enabled
//!   (nothing set)     Keenable's keyless endpoint (1,000 an hour per IP),
//!                     then DuckDuckGo's HTML page if Keenable is down.
//! Results with only a short snippet get their page read for more text.

use reqwest::{redirect, Client, Url};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

/// Pages read when their snippet is too short to answer from.
const PAGES_READ: usize = 4;
const SHORT_SNIPPET: usize = 400;
const SNIPPET_CHARS: usize = 2_000;
const RESULTS_KEPT: usize = 8;
const PAGE_CHARS: usize = 3_500;
const PAGE_BYTES: usize = 3 * 1024 * 1024;
/// Upper bound on the source text added to the prompt.
pub const SOURCE_CHARS: usize = RESULTS_KEPT * 300 + PAGES_READ * PAGE_CHARS;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Source {
    pub title: String,
    pub url: String,
    pub snippet: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published: Option<String>,
    #[serde(skip_serializing, default)]
    pub text: String,
}

pub fn client() -> Client {
    Client::builder()
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15")
        .timeout(Duration::from_secs(10))
        .redirect(redirect::Policy::custom(|attempt| {
            if attempt.previous().len() > 5 || !is_public(attempt.url()) {
                attempt.stop()
            } else {
                attempt.follow()
            }
        }))
        .build()
        .unwrap_or_default()
}

/// A Keenable key pasted in Settings (kept in SQLite). A key in the
/// environment or .env wins over it.
static SAVED_KEENABLE_KEY: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

pub fn set_saved_keenable_key(key: Option<String>) {
    *SAVED_KEENABLE_KEY.write().unwrap() = key.map(|k| k.trim().to_string()).filter(|k| !k.is_empty());
}

fn setting(name: &str) -> Option<String> {
    let env = std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    if env.is_none() && name == "KEENABLE_API_KEY" {
        return SAVED_KEENABLE_KEY.read().unwrap().clone();
    }
    env
}

/// Where the Keenable key comes from: "env", "settings" or "none".
pub fn keenable_key_source() -> &'static str {
    if std::env::var("KEENABLE_API_KEY").is_ok_and(|v| !v.trim().is_empty()) {
        "env"
    } else if SAVED_KEENABLE_KEY.read().unwrap().is_some() {
        "settings"
    } else {
        "none"
    }
}

/// The key's last four characters, to show which key is set without
/// revealing it.
pub fn keenable_key_hint() -> Option<String> {
    setting("KEENABLE_API_KEY").map(|k| {
        let tail: String = k.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
        format!("…{tail}")
    })
}

/// The configured engine, for status and docs.
pub fn engine_name() -> &'static str {
    if setting("KEENABLE_API_KEY").is_some() {
        "Keenable"
    } else if setting("BRAVE_API_KEY").is_some() {
        "Brave Search"
    } else if setting("SEARXNG_URL").is_some() {
        "SearXNG"
    } else {
        "Keenable (keyless)"
    }
}

async fn search(client: &Client, query: &str) -> Result<Vec<Source>, String> {
    if let Some(key) = setting("KEENABLE_API_KEY") {
        return search_keenable(client, Some(&key), query).await;
    }
    if let Some(key) = setting("BRAVE_API_KEY") {
        return search_brave(client, &key, query).await;
    }
    if let Some(base) = setting("SEARXNG_URL") {
        return search_searxng(client, &base, query).await;
    }
    match search_keenable(client, None, query).await {
        Ok(results) if !results.is_empty() => Ok(results),
        Ok(_) => search_duckduckgo(client, query).await,
        Err(keenable) => search_duckduckgo(client, query)
            .await
            .map_err(|ddg| format!("{keenable}; fallback: {ddg}")),
    }
}

/// Searches, then reads the top pages in parallel. Pages that fail to load
/// keep their snippet.
pub async fn research(client: &Client, query: &str) -> Result<Vec<Source>, String> {
    let key = query.trim().to_lowercase();
    if let Some((at, sources)) = CACHE.lock().unwrap().get(&key) {
        if at.elapsed() < CACHE_FOR {
            return Ok(sources.clone());
        }
    }
    let sources = search_and_read(client, query).await?;
    let mut cache = CACHE.lock().unwrap();
    cache.retain(|_, (at, _)| at.elapsed() < CACHE_FOR);
    cache.insert(key, (Instant::now(), sources.clone()));
    Ok(sources)
}

/// Free search pages rate-limit automated use, so a repeated question reuses
/// its results for a while.
const CACHE_FOR: Duration = Duration::from_secs(30 * 60);
static CACHE: LazyLock<Mutex<HashMap<String, (Instant, Vec<Source>)>>> = LazyLock::new(Default::default);

async fn search_and_read(client: &Client, query: &str) -> Result<Vec<Source>, String> {
    let mut sources = search(client, query).await?;
    sources.retain(|s| Url::parse(&s.url).is_ok_and(|u| is_public(&u)));
    sources.dedup_by(|a, b| a.url == b.url);
    sources.truncate(RESULTS_KEPT);
    if sources.is_empty() {
        return Err(format!("{} returned no results", engine_name()));
    }
    official_first(&mut sources, query);
    let thin: Vec<usize> = (0..sources.len())
        .filter(|&i| sources[i].snippet.len() < SHORT_SNIPPET)
        .take(PAGES_READ)
        .collect();
    let reads = thin.iter().map(|&i| read_page(client, &sources[i].url));
    let texts = futures_util::future::join_all(reads).await;
    for (i, text) in thin.into_iter().zip(texts) {
        sources[i].text = text.unwrap_or_default();
    }
    Ok(sources)
}

/// Moves results whose site name matches a word of the query (directus.io for
/// "does directus support…") ahead of blogs and third-party pages. The sort
/// is stable, so the engine's order holds otherwise.
fn official_first(sources: &mut [Source], query: &str) {
    let words: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 4)
        .map(str::to_lowercase)
        .collect();
    sources.sort_by_key(|s| {
        let site = Url::parse(&s.url)
            .ok()
            .and_then(|u| u.host_str().map(|h| h.trim_start_matches("www.").to_lowercase()))
            .unwrap_or_default();
        let name = site.split('.').next().unwrap_or("");
        !words.iter().any(|w| name == w || site.starts_with(&format!("{w}.")) || site.contains(&format!(".{w}.")))
    });
}

/// The prompt block: numbered sources with page text where it was read,
/// kept within SOURCE_CHARS.
pub fn prompt_block(sources: &[Source]) -> String {
    let mut out = String::new();
    for (i, s) in sources.iter().enumerate() {
        let mut head = format!("[{}] {}\nURL: {}\n", i + 1, s.title, s.url);
        if let Some(date) = s.published.as_deref() {
            head.push_str(&format!("Published: {}\n", date.get(..10).unwrap_or(date)));
        }
        let body = if s.text.len() > s.snippet.len() { &s.text } else { &s.snippet };
        let room = SOURCE_CHARS.saturating_sub(out.len() + head.len() + 2);
        if room < 200 {
            break;
        }
        out.push_str(&head);
        out.push_str(&truncate(body, room));
        out.push_str("\n\n");
    }
    out
}

async fn search_keenable(client: &Client, key: Option<&str>, query: &str) -> Result<Vec<Source>, String> {
    let body = serde_json::json!({"query": query, "max_results": RESULTS_KEPT, "snippet_max_length": SNIPPET_CHARS});
    let request = match key {
        Some(key) => client.post("https://api.keenable.ai/v1/search").header("X-API-Key", key),
        None => client.post("https://api.keenable.ai/v1/search/public").header("X-Keenable-Title", "Ferro64"),
    };
    let response = request
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Keenable is unreachable: {e}"))?;
    match response.status().as_u16() {
        200..=299 => {}
        402 => return Err("Keenable's free monthly requests are used up".into()),
        429 => return Err("Keenable's hourly limit is reached; try again later or set KEENABLE_API_KEY".into()),
        code => return Err(format!("Keenable returned {code}")),
    }
    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Keenable sent an unreadable reply: {e}"))?;
    Ok(json_results(&body, "/results", "title", "url", "snippet", Some("published_at")))
}

async fn search_searxng(client: &Client, base: &str, query: &str) -> Result<Vec<Source>, String> {
    let response = client
        .get(format!("{}/search", base.trim_end_matches('/')))
        .query(&[("q", query), ("format", "json")])
        .send()
        .await
        .map_err(|e| format!("SearXNG at {base} is unreachable: {e}"))?;
    if response.status() == reqwest::StatusCode::FORBIDDEN {
        return Err("SearXNG refused JSON. Add `json` under search.formats in its settings.yml".into());
    }
    if !response.status().is_success() {
        return Err(format!("SearXNG returned {}", response.status()));
    }
    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("SearXNG sent an unreadable reply: {e}"))?;
    Ok(json_results(&body, "/results", "title", "url", "content", Some("publishedDate")))
}

/// Reads a JSON result list into sources; Brave's titles and snippets carry
/// <strong> markup, so text is taken from HTML.
fn json_results(body: &serde_json::Value, list: &str, title: &str, url: &str, snippet: &str, published: Option<&str>) -> Vec<Source> {
    let text = |r: &serde_json::Value, key: &str| clean(&html_text(r.get(key).and_then(|v| v.as_str()).unwrap_or("")));
    body.pointer(list)
        .and_then(|r| r.as_array())
        .map(|results| {
            results
                .iter()
                .filter_map(|r| {
                    Some(Source {
                        title: text(r, title),
                        url: r.get(url)?.as_str()?.to_string(),
                        snippet: text(r, snippet),
                        published: published
                            .and_then(|k| r.get(k))
                            .and_then(|v| v.as_str())
                            .filter(|v| !v.is_empty())
                            .map(str::to_string),
                        text: String::new(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

async fn search_brave(client: &Client, key: &str, query: &str) -> Result<Vec<Source>, String> {
    let response = client
        .get("https://api.search.brave.com/res/v1/web/search")
        .query(&[("q", query), ("count", "10")])
        .header("X-Subscription-Token", key)
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(|e| format!("Brave Search is unreachable: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("Brave Search returned {}", response.status()));
    }
    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Brave Search sent an unreadable reply: {e}"))?;
    Ok(json_results(&body, "/web/results", "title", "url", "description", None))
}

async fn search_duckduckgo(client: &Client, query: &str) -> Result<Vec<Source>, String> {
    let response = client
        .post("https://html.duckduckgo.com/html/")
        .form(&[("q", query), ("kl", "us-en")])
        .send()
        .await
        .map_err(|e| format!("DuckDuckGo is unreachable: {e}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "DuckDuckGo returned {}. It limits automated searches; set BRAVE_API_KEY for a steadier source.",
            response.status()
        ));
    }
    let html = response.text().await.map_err(|e| e.to_string())?;
    let results = parse_duckduckgo(&html);
    if results.is_empty() && html.contains("anomaly") {
        return Err("DuckDuckGo asked for a captcha. Try again in a minute, or set BRAVE_API_KEY.".into());
    }
    Ok(results)
}

fn parse_duckduckgo(html: &str) -> Vec<Source> {
    let doc = Html::parse_document(html);
    let result = Selector::parse(".result").unwrap();
    let link = Selector::parse("a.result__a").unwrap();
    let snippet = Selector::parse(".result__snippet").unwrap();
    doc.select(&result)
        .filter(|r| !r.value().classes().any(|c| c == "result--ad"))
        .filter_map(|r| {
            let a = r.select(&link).next()?;
            let url = duckduckgo_target(a.value().attr("href")?)?;
            Some(Source {
                title: clean(&a.text().collect::<String>()),
                url,
                snippet: r
                    .select(&snippet)
                    .next()
                    .map(|s| clean(&s.text().collect::<String>()))
                    .unwrap_or_default(),
                published: None,
                text: String::new(),
            })
        })
        .collect()
}

/// DuckDuckGo links go through a redirect that carries the target in `uddg`.
fn duckduckgo_target(href: &str) -> Option<String> {
    let absolute = if href.starts_with("//") { format!("https:{href}") } else { href.to_string() };
    let url = Url::parse(&absolute).ok()?;
    if url.host_str().is_some_and(|h| h.ends_with("duckduckgo.com")) {
        let target = url.query_pairs().find(|(k, _)| k == "uddg")?.1.into_owned();
        // Ads point back to duckduckgo.com/y.js.
        return Url::parse(&target).ok().filter(|u| !u.host_str().is_some_and(|h| h.ends_with("duckduckgo.com"))).map(|u| u.to_string());
    }
    Some(url.to_string())
}

async fn read_page(client: &Client, url: &str) -> Result<String, String> {
    let mut response = client.get(url).send().await.map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(response.status().to_string());
    }
    let html_page = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_none_or(|v| v.contains("html"));
    if !html_page {
        return Err("not an HTML page".into());
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
        bytes.extend_from_slice(&chunk);
        if bytes.len() > PAGE_BYTES {
            break;
        }
    }
    Ok(page_text(&String::from_utf8_lossy(&bytes)))
}

/// Readable text of a page: headings, paragraphs, list items, table cells and
/// code from its main content, falling back to the whole body.
fn page_text(html: &str) -> String {
    let doc = Html::parse_document(html);
    let blocks = Selector::parse("h1, h2, h3, h4, p, li, td, th, pre, blockquote, dt, dd").unwrap();
    let skip = ["nav", "header", "footer", "aside", "form", "script", "style", "noscript"];
    let roots = ["article", "main", "[role=main]", "body"];
    for root in roots {
        let Some(root) = doc.select(&Selector::parse(root).unwrap()).next() else {
            continue;
        };
        let mut out = String::new();
        for block in root.select(&blocks) {
            // Nested matches (li inside li, p inside td) would repeat text.
            let nested_or_chrome = block.ancestors().filter_map(scraper::ElementRef::wrap).any(|a| {
                let name = a.value().name();
                skip.contains(&name) || (a.id() != root.id() && blocks.matches(&a) && a.id() != block.id())
            });
            if nested_or_chrome {
                continue;
            }
            let text = clean(&block.text().collect::<String>());
            if text.len() < 3 {
                continue;
            }
            out.push_str(&text);
            out.push('\n');
            if out.len() >= PAGE_CHARS {
                break;
            }
        }
        if out.len() > 200 || root.value().name() == "body" {
            return truncate(&out, PAGE_CHARS);
        }
    }
    String::new()
}

fn html_text(fragment: &str) -> String {
    Html::parse_fragment(fragment).root_element().text().collect()
}

fn clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.trim().to_string();
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", text[..end].trim())
}

/// Only fetch public http(s) addresses, so a search result or redirect cannot
/// reach this machine or the local network.
fn is_public(url: &Url) -> bool {
    if !matches!(url.scheme(), "http" | "https") {
        return false;
    }
    match url.host() {
        Some(url::Host::Domain(host)) => {
            let host = host.to_ascii_lowercase();
            !(host == "localhost" || host.ends_with(".localhost") || host.ends_with(".local") || host.ends_with(".internal"))
        }
        Some(url::Host::Ipv4(ip)) => public_ip(IpAddr::V4(ip)),
        Some(url::Host::Ipv6(ip)) => public_ip(IpAddr::V6(ip)),
        None => false,
    }
}

fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => !(v4.is_private() || v4.is_loopback() || v4.is_link_local() || v4.is_unspecified() || v4.is_broadcast() || v4.octets()[0] == 100 && v4.octets()[1] >= 64 && v4.octets()[1] < 128),
        IpAddr::V6(v6) => !(v6.is_loopback() || v6.is_unspecified() || (v6.segments()[0] & 0xfe00) == 0xfc00 || (v6.segments()[0] & 0xffc0) == 0xfe80),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duckduckgo_results_unwrap_redirects_and_skip_ads() {
        let html = r#"<div class="result result--ad"><a class="result__a" href="//duckduckgo.com/y.js?ad=1">Ad</a></div>
            <div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdirectus.io%2Fdocs%2Fconfiguration%2Fdatabase&amp;rut=x">Database | <b>Directus</b></a>
            <a class="result__snippet">Supported: pg, mysql,  sqlite3.</a></div>"#;
        let results = parse_duckduckgo(html);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url, "https://directus.io/docs/configuration/database");
        assert_eq!(results[0].title, "Database | Directus");
        assert_eq!(results[0].snippet, "Supported: pg, mysql, sqlite3.");
    }

    #[test]
    fn page_text_prefers_main_content_and_skips_chrome() {
        let html = "<html><body><nav><li>Home</li></nav><main><h1>Databases</h1><p>Directus supports Postgres and MySQL.</p>\
            <ul><li>SQLite <ul><li>nested</li></ul></li></ul></main><footer><p>Copyright</p></footer></body></html>";
        let text = page_text(html);
        assert!(text.contains("Databases\nDirectus supports Postgres and MySQL."));
        assert!(!text.contains("Home") && !text.contains("Copyright"));
        assert_eq!(text.matches("nested").count(), 1);
    }

    #[test]
    fn api_results_parse_and_the_prompt_stays_in_budget() {
        let body = serde_json::json!({"results": [
            {"title": "Help | Directus Docs", "url": "https://docs.directus.io/support", "snippet": "Does Directus support Mongo? Not currently.", "published_at": "2024-07-11T21:59:52Z"},
            {"title": "No URL"},
            {"title": "<b>Big</b>", "url": "https://example.com/", "snippet": "x".repeat(SOURCE_CHARS * 2)}
        ]});
        let sources = json_results(&body, "/results", "title", "url", "snippet", Some("published_at"));
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[1].title, "Big");
        let block = prompt_block(&sources);
        assert!(block.starts_with("[1] Help | Directus Docs\nURL: https://docs.directus.io/support\nPublished: 2024-07-11\nDoes Directus support Mongo? Not currently."));
        assert!(block.len() <= SOURCE_CHARS + 10);
    }

    #[test]
    fn the_products_own_site_is_read_first() {
        let source = |url: &str| Source { title: String::new(), url: url.into(), snippet: String::new(), published: None, text: String::new() };
        let mut list = vec![source("https://blog.example.com/directus-mongo"), source("https://github.com/someone/directus-mongo"), source("https://docs.directus.io/db"), source("https://directus.com/features")];
        official_first(&mut list, "does Directus support MongoDB");
        let urls: Vec<&str> = list.iter().map(|s| s.url.as_str()).collect();
        assert_eq!(urls, ["https://docs.directus.io/db", "https://directus.com/features", "https://blog.example.com/directus-mongo", "https://github.com/someone/directus-mongo"]);
    }

    #[test]
    fn only_public_addresses_are_fetched() {
        for bad in ["http://localhost:3042/", "http://127.0.0.1/", "http://192.168.1.4/", "http://[::1]/", "file:///etc/passwd", "http://printer.local/"] {
            assert!(!is_public(&Url::parse(bad).unwrap()), "{bad}");
        }
        assert!(is_public(&Url::parse("https://directus.io/docs").unwrap()));
    }
}
