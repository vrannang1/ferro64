//! Compiling after edits, without the model having to remember. What comes back
//! is short: the first errors, and the real API when an error names something
//! that does not exist. Small models guess library APIs from memory, and an error
//! alone only tells them the guess was wrong.

use regex::Regex;
use std::path::Path;
use std::sync::LazyLock;

const LINES_KEPT: usize = 40;
const CHARS_KEPT: usize = 3_000;
const API_CHARS: usize = 1_500;
pub const CHECK_SECS: u64 = 180;

/// FERRO64_AUTOCHECK=off turns this off, to compare runs with and without it.
pub fn enabled() -> bool {
    std::env::var("FERRO64_AUTOCHECK").map_or(true, |value| value != "off")
}

/// The project's compile step, without running its tests or app.
pub fn command_for(root: &Path) -> Option<&'static str> {
    let has = |name: &str| root.join(name).is_file();
    let has_ext = |ext: &str| std::fs::read_dir(root).is_ok_and(|dir| dir.flatten().any(|f| f.path().extension().is_some_and(|e| e == ext)));
    if has("Cargo.toml") {
        Some("cargo check --quiet --message-format=short")
    } else if has("mix.exs") {
        Some("mix compile")
    } else if has("go.mod") {
        Some("go build ./...")
    } else if has("rebar.config") {
        Some("rebar3 compile")
    } else if has("stack.yaml") {
        Some("stack build --fast")
    } else if has_ext("cabal") {
        Some("cabal build -v1")
    } else if has("Package.swift") {
        Some("swift build")
    } else if has_ext("csproj") || has_ext("sln") {
        Some("dotnet build --nologo -v q")
    } else if has("tsconfig.json") {
        Some("npx --no-install tsc --noEmit")
    } else if has("pyproject.toml") || has("requirements.txt") || has("main.py") || has("app.py") {
        Some("python3 -m compileall -q .")
    } else {
        None
    }
}

/// Colours out, and a line repeated in a row collapsed into one.
pub fn squeeze(output: &str) -> String {
    static ANSI: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\x1b\[[0-9;?]*[ -/]*[@-~]").unwrap());
    let clean = ANSI.replace_all(output, "");
    let mut lines: Vec<String> = Vec::new();
    let mut repeats = 0;
    for line in clean.lines() {
        if lines.last().is_some_and(|last| last == line) {
            repeats += 1;
            continue;
        }
        if repeats > 0 {
            lines.push(format!("[the line above repeated {repeats} more times]"));
            repeats = 0;
        }
        lines.push(line.to_string());
    }
    if repeats > 0 {
        lines.push(format!("[the line above repeated {repeats} more times]"));
    }
    lines.join("\n")
}

/// The start of a compiler's report, which is where the real errors are.
pub fn first_errors(output: &str) -> String {
    let squeezed = squeeze(output);
    let mut kept: String = squeezed.lines().take(LINES_KEPT).collect::<Vec<_>>().join("\n");
    if kept.chars().count() > CHARS_KEPT {
        kept = kept.chars().take(CHARS_KEPT).collect();
    }
    if kept.len() < squeezed.len() {
        kept.push_str("\n[more errors cut; fix these first, because later ones often follow from them]");
    }
    kept
}

/// Names an error says do not exist, as written, such as `Data.UUID` with
/// `generate`. The library is a guess from these, and a wrong guess costs nothing.
pub fn missing_names(output: &str) -> Vec<String> {
    static PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
        [
            // Haskell
            r"Module [‘'`]([\w.]+)[’'`] does not export [‘'`](\w+)[’'`]",
            r"Could not find module [‘'`]([\w.]+)[’'`]",
            // Rust
            r"unresolved import `([\w:]+)`",
            r"(?:function|associated item|method) named `(\w+)` found for (?:\w+ )?`&?([\w:]+)",
            // Go
            r"undefined: ([\w.]+)",
            r"([\w.]+) undefined \(type",
            // Elixir and Erlang
            r"function ([\w.:]+)/\d+ (?:is )?undefined",
            r"([A-Z][\w.]+)\.(\w+)/\d+ is undefined",
            // Swift and TypeScript
            r"has no member '(\w+)'",
            r"no exported member '(\w+)'",
        ]
        .iter()
        .map(|pattern| Regex::new(pattern).unwrap())
        .collect()
    });
    let mut found = Vec::new();
    for pattern in PATTERNS.iter() {
        for captures in pattern.captures_iter(output) {
            let name = captures.iter().skip(1).flatten().map(|m| m.as_str()).collect::<Vec<_>>().join(" ");
            if !found.contains(&name) {
                found.push(name);
            }
        }
    }
    found.truncate(3);
    found
}

/// Library names a missing name might belong to: `Data.UUID` could be `uuid` or
/// `data`, and `axum::Router` is `axum`.
pub fn library_guesses(name: &str) -> Vec<String> {
    let mut guesses: Vec<String> = Vec::new();
    for word in name.split_whitespace() {
        let parts: Vec<&str> = word.split(|c| c == '.' || c == ':').filter(|p| !p.is_empty()).collect();
        for part in parts.iter().rev().chain(parts.first()) {
            let lower = part.to_lowercase();
            if lower.len() > 1 && !guesses.contains(&lower) {
                guesses.push(lower);
            }
        }
    }
    guesses
}

/// What a library really offers, for the report.
pub fn real_api(library: &Path, name: &str) -> String {
    let map = crate::brief::code_map(library, name);
    let short: String = map.chars().take(API_CHARS).collect();
    format!("\n\n[Ferro64] What the installed library {} really defines. Use these, not names from memory:\n{}", library.file_name().and_then(|n| n.to_str()).unwrap_or("library"), short.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_finds_the_compile_step() {
        let dir = std::env::temp_dir().join(format!("ferro64-check-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(command_for(&dir), None);
        std::fs::write(dir.join("App.csproj"), "").unwrap();
        assert_eq!(command_for(&dir), Some("dotnet build --nologo -v q"));
        std::fs::remove_file(dir.join("App.csproj")).unwrap();
        std::fs::write(dir.join("snippets.cabal"), "").unwrap();
        assert_eq!(command_for(&dir), Some("cabal build -v1"));
        std::fs::write(dir.join("mix.exs"), "").unwrap();
        assert_eq!(command_for(&dir), Some("mix compile"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn repeated_lines_and_colours_go() {
        assert_eq!(squeeze("\x1b[31merror\x1b[0m\na\na\na\nb"), "error\na\n[the line above repeated 2 more times]\nb");
        assert_eq!(squeeze("a\na"), "a\n[the line above repeated 1 more times]");
    }

    #[test]
    fn a_long_report_keeps_its_start() {
        let long: String = (0..200).map(|i| format!("error {i}\n")).collect();
        let kept = first_errors(&long);
        assert!(kept.contains("error 0") && !kept.contains("error 150") && kept.contains("fix these first"));
        assert_eq!(first_errors("one error"), "one error");
    }

    #[test]
    fn errors_from_our_tests_name_what_is_missing() {
        assert_eq!(missing_names("Module ‘Data.UUID’ does not export ‘generate’"), vec!["Data.UUID generate"]);
        assert_eq!(missing_names("error[E0432]: unresolved import `uuid::generate`"), vec!["uuid::generate"]);
        assert_eq!(missing_names("no function or associated item named `new_v9` found for struct `Uuid`"), vec!["new_v9 Uuid"]);
        assert_eq!(missing_names("./main.go:9:2: undefined: uuid.Generate"), vec!["uuid.Generate"]);
        assert!(missing_names("warning: everything is fine").is_empty());
    }

    #[test]
    fn library_guesses_come_from_the_name() {
        assert_eq!(library_guesses("Data.UUID generate")[0], "uuid");
        assert!(library_guesses("Data.UUID generate").contains(&"data".to_string()));
        assert_eq!(library_guesses("uuid::generate")[1], "uuid");
    }
}
