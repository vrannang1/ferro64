//! Refuses an edit that adds syntax errors, before it reaches the disk. Compilers
//! catch these too, but only after a broken file is saved, and small models can
//! then lose many steps on it. Grammars can lag behind a language, so this
//! compares against the file as it was, and a repeated attempt is let through.

use std::path::Path;
use tree_sitter::{Language, Node, Parser};

fn language(path: &str) -> Option<Language> {
    let extension = Path::new(path).extension()?.to_str()?;
    Some(match extension {
        "rs" => tree_sitter_rust::LANGUAGE.into(),
        "go" => tree_sitter_go::LANGUAGE.into(),
        "py" => tree_sitter_python::LANGUAGE.into(),
        "ex" | "exs" => tree_sitter_elixir::LANGUAGE.into(),
        "hs" => tree_sitter_haskell::LANGUAGE.into(),
        "swift" => tree_sitter_swift::LANGUAGE.into(),
        "cs" => tree_sitter_c_sharp::LANGUAGE.into(),
        "js" | "mjs" | "jsx" => tree_sitter_javascript::LANGUAGE.into(),
        "ts" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        _ => return None,
    })
}

fn collect(node: Node, found: &mut Vec<usize>) {
    if node.is_error() || node.is_missing() {
        found.push(node.start_position().row + 1);
        return;
    }
    if node.has_error() {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            collect(child, found);
        }
    }
}

/// The lines of each syntax error, or None for a language with no grammar here.
fn error_lines(path: &str, text: &str) -> Option<Vec<usize>> {
    let mut parser = Parser::new();
    parser.set_language(&language(path)?).ok()?;
    let tree = parser.parse(text, None)?;
    let mut found = Vec::new();
    collect(tree.root_node(), &mut found);
    Some(found)
}

/// What to tell the model if this edit adds syntax errors.
pub fn new_errors(path: &str, before: Option<&str>, after: &str) -> Option<String> {
    let lines = error_lines(path, after)?;
    let had = before.and_then(|text| error_lines(path, text)).map_or(0, |found| found.len());
    if lines.len() <= had {
        return None;
    }
    let shown: Vec<String> = lines.iter().take(3).map(usize::to_string).collect();
    let context = after.lines().nth(lines[0].saturating_sub(1)).unwrap_or("").trim();
    let context: String = context.chars().take(120).collect();
    Some(format!("That was not saved: it would add a syntax error to {path} at line {} (`{context}`). Check brackets, indentation and whether the text was cut off, then send the corrected edit.", shown.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broken_code_is_caught_in_each_language() {
        for (path, broken) in [
            ("a.rs", "fn main( {"),
            ("a.go", "package a\nfunc f( {"),
            ("a.py", "def f(:\n  pass"),
            ("a.ex", "defmodule A do\n def f( do\n"),
            ("a.hs", "module A where\nf = (1 +"),
            ("a.swift", "func f( {"),
            ("a.cs", "class A { void F( { }"),
            ("a.ts", "function f( {"),
        ] {
            assert!(new_errors(path, None, broken).is_some(), "{path}");
        }
    }

    #[test]
    fn good_code_passes_in_each_language() {
        for (path, good) in [
            ("a.rs", "pub fn f(x: &str) -> String { x.to_string() }\n"),
            ("a.go", "package a\n\nfunc F() int { return 1 }\n"),
            ("a.py", "def f(x):\n    return x\n"),
            ("a.ex", "defmodule A do\n  def f(x), do: x\nend\n"),
            ("a.hs", "module A (f) where\n\nf :: Int -> Int\nf x = x + 1\n"),
            ("a.swift", "func f() -> Int { return 1 }\n"),
            ("a.cs", "public static class A { public static int F(int x) => x + 1; }\n"),
            ("a.ts", "export const f = (x: number): number => x;\n"),
        ] {
            assert_eq!(new_errors(path, None, good), None, "{path}");
        }
    }

    #[test]
    fn only_new_errors_count_and_unknown_languages_pass() {
        let broken = "fn main( {";
        assert_eq!(new_errors("a.rs", Some(broken), &format!("{broken}\n// note")), None, "already broken before this edit");
        assert_eq!(new_errors("a.erl", None, "anything ((("), None);
        let message = new_errors("a.py", Some("x = 1\n"), "def f(:\n").unwrap();
        assert!(message.contains("a.py") && message.contains("line 1"));
    }
}
