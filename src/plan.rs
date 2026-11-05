//! The coding agent's plan. Ferro64 holds it, not the conversation: it is shown
//! to the user, repeated to the model on every request so trimming cannot lose
//! it, and an item only counts as done with a command that really passed.

use serde::{Deserialize, Serialize};
use serde_json::Value;

const MAX_ITEMS: usize = 30;
const MAX_TEXT: usize = 200;
const MAX_PROOF: usize = 500;
const MAX_SHARED_PROOF: usize = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanItem {
    pub text: String,
    /// "todo", "doing", "done" or "dropped".
    pub status: String,
    /// For done, the command that proved it; for dropped, the reason.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub proof: String,
}

impl PlanItem {
    fn settled(&self) -> bool {
        self.status == "done" || self.status == "dropped"
    }
}

/// Every item is done or dropped. An empty plan is not settled work.
pub fn settled(plan: &[PlanItem]) -> bool {
    !plan.is_empty() && plan.iter().all(PlanItem::settled)
}

pub fn done_count(plan: &[PlanItem]) -> usize {
    plan.iter().filter(|item| item.status == "done").count()
}

/// Commands that only look around. They may run before a plan exists, but they
/// prove nothing: a folder listing does not show that anything works.
pub fn only_looks(command: &str) -> bool {
    const LOOKING: &[&str] = &["ls", "pwd", "cd", "cat", "head", "tail", "find", "echo", "printf", "true", ":", "which", "wc", "tree", "stat", "file", "sleep"];
    // The & in 2>&1 is a redirect, not the end of a command.
    let command = command.replace("2>&1", "").replace("1>&2", "");
    command
        .split(|c| c == '|' || c == ';' || c == '&')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .all(|part| {
            let words: Vec<&str> = part.split_whitespace().collect();
            let first = words.first().copied().unwrap_or("");
            LOOKING.contains(&first)
                || (first == "git" && matches!(words.get(1), Some(&("status" | "log" | "diff" | "show" | "branch"))))
                || (words.len() == 2 && words[1] == "--version")
        })
}

/// Whether a request only asks for an answer. A wrong "no" leaves the model's own
/// judgment; a wrong "yes" costs one rewording, and the refusal says why.
pub fn only_answers(request: &str) -> bool {
    const HANDS_OFF: &[&str] = &["do not change anything", "don't change anything", "do not edit anything", "don't edit anything", "do not modify anything", "don't modify anything", "do not change any code", "don't change any code", "change nothing", "make no changes"];
    const ASKS_FOR_WORK: &[&str] = &["can you", "could you", "would you", "will you", "can we", "could we", "please", "let's", "go ahead", "i want you", "i need you", "why don't you", "how about"];
    const QUESTION: &[&str] = &["what", "why", "how", "where", "when", "which", "who", "whose", "is", "isn", "are", "aren", "was", "were", "do", "does", "doesn", "did", "should", "can", "could", "will"];
    let text = request.trim().to_lowercase().replace('\u{2019}', "'");
    let sentences: Vec<&str> = text.split(['.', '!', '?', '\n']).map(str::trim).filter(|s| !s.is_empty()).collect();
    if sentences.iter().any(|sentence| HANDS_OFF.contains(sentence)) {
        return true;
    }
    let first_word = text.split(|c: char| !c.is_alphanumeric()).next().unwrap_or("");
    text.ends_with('?') && QUESTION.contains(&first_word) && !ASKS_FOR_WORK.iter().any(|phrase| text.contains(phrase))
}

/// Whether a request plainly asks for files to change. Then a run that changed
/// nothing did not finish, whatever the model says.
pub fn asks_for_change(request: &str) -> bool {
    const WORK: &[&str] = &["add", "fix", "write", "create", "implement", "build", "remove", "delete", "rename", "refactor", "update", "change", "make"];
    !only_answers(request) && request.to_lowercase().split(|c: char| !c.is_alphanumeric()).any(|word| WORK.contains(&word))
}

/// Commands an answer-only request may run: looking, and tests or builds, whose
/// output is the point. A redirect or a delete can change files whatever comes first.
pub fn changes_nothing(command: &str) -> bool {
    const CHECKING: &[&str] = &[
        "grep", "rg", "mix test", "mix compile", "cargo test", "cargo check", "go test", "go vet", "npm test", "npm run test", "yarn test", "pnpm test", "pytest", "python -m pytest",
        "python3 -m pytest", "cabal test", "cabal build", "stack test", "stack build", "dotnet test", "dotnet build", "swift test", "swift build", "rebar3 eunit", "rebar3 ct", "rebar3 compile",
    ];
    let command = command.replace("2>&1", "").replace("1>&2", "");
    if command.contains('>') || command.contains("-delete") || command.contains("-exec") {
        return false;
    }
    command.split(['|', ';', '&', '\n']).map(str::trim).filter(|part| !part.is_empty()).all(|part| {
        let part = part.split_whitespace().collect::<Vec<_>>().join(" ");
        only_looks(&part) || CHECKING.iter().any(|check| part == *check || part.starts_with(&format!("{check} ")))
    })
}

/// Whether the exit code speaks for the whole command. In `mix test | head` or
/// `mix test; true` it comes from the last part, so a failure looks like a pass,
/// and `server &` always succeeds.
pub fn exit_counts(command: &str) -> bool {
    let chars: Vec<char> = command.trim().chars().collect();
    let (mut quote, mut i) = (None, 0);
    while i < chars.len() {
        let c = chars[i];
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) if c == '\\' => i += 1,
            Some(_) => {}
            None => match c {
                '\\' => i += 1,
                '\'' | '"' => quote = Some(c),
                ';' | '|' | '\n' => return false,
                '&' if chars.get(i + 1) == Some(&'&') => i += 1,
                '&' if chars.get(i + 1) == Some(&'>') || (i > 0 && chars[i - 1] == '>') => {}
                '&' => return false,
                _ => {}
            },
        }
        i += 1;
    }
    true
}

/// Commands that show something compiles or installs, not that it works.
fn only_builds(command: &str) -> bool {
    const BUILDING: &[&str] = &[
        "mix compile", "mix deps.get", "mix deps.compile", "cargo build", "cargo check", "cargo fetch", "npm install", "npm i", "npm ci", "npm run build",
        "yarn install", "yarn build", "pnpm install", "pnpm build", "npx tsc", "npx vite build", "vite build", "tsc", "go build", "go vet", "go mod",
        "dotnet build", "dotnet restore", "cabal build", "cabal update", "stack build", "swift build", "swift package resolve", "rebar3 compile", "rebar3 get-deps",
        "erlc", "ghc", "pip install", "pip3 install", "uv sync", "python -m venv", "python3 -m venv", "podman build", "podman compose config", "podman-compose config", "docker build", "docker compose config",
    ];
    let command = command.replace("2>&1", "");
    command.split("&&").map(str::trim).filter(|part| !part.is_empty()).all(|part| {
        let part = part.split_whitespace().collect::<Vec<_>>().join(" ");
        part.starts_with("cd ") || part.starts_with("source ") || BUILDING.iter().any(|b| part == *b || part.starts_with(&format!("{b} ")))
    })
}

fn about_containers(item: &str) -> bool {
    let item = item.to_lowercase();
    ["dockerfile", "docker-compose", "docker compose", "container", "podman"].iter().any(|word| item.contains(word))
}

/// An item about getting set up can be proved by a build; anything else has to run.
fn is_setup(item: &str) -> bool {
    let item = item.to_lowercase();
    // "Wire the server and set up routes" is a feature, so these count only at the start.
    let opening = item.split_once(':').filter(|(label, _)| label.starts_with("step")).map_or(item.as_str(), |(_, rest)| rest.trim_start());
    let sets_up = ["setup", "set up", "initial", "init "].iter().any(|word| opening.starts_with(word));
    sets_up || ["project", "scaffold", "install", "dependenc", "compil", "dockerfile", "docker-compose", "docker compose", "container"].iter().any(|word| item.contains(word))
}

/// Checks a whole new plan against the old one. `passed` holds the commands that
/// succeeded since the last file change. Returns the plan to keep, or what the
/// model has to correct.
pub fn apply(previous: &[PlanItem], args: &Value, passed: &[String]) -> Result<Vec<PlanItem>, String> {
    let items = args["items"].as_array().ok_or("Pass items: a list of {text, status, proof}.")?;
    if items.is_empty() {
        return Err("The plan needs at least one item.".to_string());
    }
    if items.len() > MAX_ITEMS {
        return Err(format!("Keep the plan to {MAX_ITEMS} items or fewer."));
    }
    let mut plan = Vec::with_capacity(items.len());
    for item in items {
        let text = clipped(item["text"].as_str().unwrap_or(""), MAX_TEXT);
        if text.is_empty() {
            return Err("Every item needs text.".to_string());
        }
        let status = item["status"].as_str().unwrap_or("todo").to_string();
        let proof = clipped(item["proof"].as_str().unwrap_or(""), MAX_PROOF);
        let before = previous.iter().find(|p| p.text == text);
        match status.as_str() {
            "todo" | "doing" => {}
            "done" => {
                // Proved earlier and unchanged: later edits do not undo it.
                let already = before.is_some_and(|p| p.status == "done" && p.proof == proof);
                if !already {
                    if proof.is_empty() {
                        return Err(format!("\"{text}\" cannot be done without proof: run a command that exercises it, then pass that exact command as proof."));
                    }
                    if only_looks(&proof) {
                        return Err(format!("`{proof}` only looks around, so it does not prove \"{text}\". Run something that exercises it: the tests, the build, or a request to the running app."));
                    }
                    // Models often add the output after the command, so contain, not equal.
                    if !passed.iter().any(|command| proof.contains(command.as_str())) {
                        if !exit_counts(&proof) {
                            return Err(format!("`{proof}` cannot be proof: with a pipe, `;` or `&`, the exit code comes from only the last part, so a failure can look like a pass. Run the command on its own, then pass that as proof for \"{text}\"."));
                        }
                        let usable = if passed.is_empty() { "none yet".to_string() } else { passed.iter().map(|c| format!("`{c}`")).collect::<Vec<_>>().join(", ") };
                        return Err(format!("`{proof}` is not a command that passed since your last file change, so it does not prove \"{text}\". Commands that count right now: {usable}. Run one that exercises this item, then pass it as proof."));
                    }
                    if about_containers(&text) && !proof.contains("docker") && !proof.contains("podman") && !proof.contains("compose") {
                        return Err(format!("`{proof}` does not touch containers, so it does not prove \"{text}\". Build or check it with Podman or Docker. If that cannot be done here, mark it dropped with the reason in proof."));
                    }
                    if !is_setup(&text) && passed.iter().filter(|command| proof.contains(command.as_str())).all(|command| only_builds(command)) {
                        return Err(format!("`{proof}` shows that it builds, not that \"{text}\" works. Run it and exercise this item: its tests, or a request to the running app that checks the result."));
                    }
                }
            }
            "dropped" => {
                if proof.is_empty() {
                    return Err(format!("\"{text}\" can only be dropped with a reason in proof, and the user will see it."));
                }
            }
            other => return Err(format!("Status {other:?} is not valid. Use todo, doing, done or dropped.")),
        }
        plan.push(PlanItem { text, status, proof });
    }
    // A broad test run can back up a couple of items, not a whole plan.
    for item in plan.iter().filter(|item| item.status == "done" && !is_setup(&item.text)) {
        let newly = !previous.iter().any(|p| p.text == item.text && p.status == "done" && p.proof == item.proof);
        let sharing = plan.iter().filter(|other| other.status == "done" && !is_setup(&other.text) && other.proof == item.proof).count();
        if newly && sharing > MAX_SHARED_PROOF {
            return Err(format!("`{}` is already the proof for {MAX_SHARED_PROOF} items, so it does not also prove \"{}\". Run something that exercises this item specifically.", item.proof, item.text));
        }
    }
    // The plan can grow, but nothing leaves it unnoticed.
    if let Some(gone) = previous.iter().find(|p| !plan.iter().any(|item| item.text == p.text)) {
        return Err(format!("\"{}\" is missing. Keep every item with its exact text; if one no longer applies, mark it dropped with the reason in proof.", gone.text));
    }
    Ok(plan)
}

/// As `apply`, but one refused item does not throw out the rest: a done item
/// without good proof stays open and the others are kept. Returns the plan and
/// what was refused. Anything wrong with the plan as a whole is still an error.
pub fn apply_keeping(previous: &[PlanItem], args: &Value, passed: &[String]) -> Result<(Vec<PlanItem>, Vec<String>), String> {
    let mut args = args.clone();
    let mut refused = Vec::new();
    if let Some(items) = args["items"].as_array_mut() {
        let mut accepted: Vec<Value> = Vec::new();
        for item in items.iter_mut() {
            if item["status"].as_str() != Some("done") {
                continue;
            }
            let text = clipped(item["text"].as_str().unwrap_or(""), MAX_TEXT);
            let before: Vec<PlanItem> = previous.iter().filter(|p| p.text == text).cloned().collect();
            // Checked with the items already accepted, so sharing a proof counts.
            let mut trial = accepted.clone();
            trial.push(item.clone());
            let earlier: Vec<PlanItem> = previous.iter().filter(|p| accepted.iter().any(|a| a["text"].as_str().map(str::trim) == Some(p.text.as_str()))).cloned().chain(before).collect();
            match apply(&earlier, &serde_json::json!({"items": trial}), passed) {
                Ok(_) => accepted.push(item.clone()),
                Err(why) => {
                    refused.push(why);
                    item["status"] = Value::from("doing");
                    item["proof"] = Value::from("");
                }
            }
        }
    }
    apply(previous, &args, passed).map(|plan| (plan, refused))
}

/// A plan for a model that will not write one: the request itself as the only new
/// item. It still has to be proved like any other.
pub fn with_request(previous: &[PlanItem], request: &str) -> Vec<PlanItem> {
    let mut plan = previous.to_vec();
    let text = clipped(request, MAX_TEXT);
    if !plan.iter().any(|item| item.text == text) {
        plan.push(PlanItem { text, status: "doing".to_string(), proof: String::new() });
    }
    plan
}

/// There is no finished plan to judge changes against yet.
pub fn open_or_empty(plan: &[PlanItem]) -> bool {
    !settled(plan)
}

/// What the model is told at the end of every request.
pub fn reminder(goal: &str, plan: &[PlanItem]) -> String {
    let lines: Vec<String> = plan
        .iter()
        .map(|item| {
            let mark = match item.status.as_str() {
                "done" => "[x]",
                "doing" => "[>]",
                "dropped" => "[-]",
                _ => "[ ]",
            };
            format!("{mark} {}", item.text)
        })
        .collect();
    let next = plan.iter().find(|item| !item.settled()).map_or_else(
        || "Every item is settled; if the user asked for something new, add it to the plan first.".to_string(),
        |item| format!("Work on: {}. Fix an error only as far as this item needs; do not rework other parts or start over.", item.text),
    );
    format!("[Ferro64 plan] Goal: {}\n{}\n{next}", clipped(goal, 600), lines.join("\n"))
}

/// The items still open, for the prompt that keeps the agent going.
pub fn open_items(plan: &[PlanItem]) -> String {
    plan.iter().filter(|item| !item.settled()).map(|item| item.text.as_str()).collect::<Vec<_>>().join("; ")
}

fn clipped(text: &str, limit: usize) -> String {
    text.trim().chars().take(limit).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_questions_and_hands_off_requests_are_answer_only() {
        for request in [
            "What did we do on this project last time, and is anything left over?",
            "Why does the buzzer return 500?",
            "How do I add a route?",
            "Should we add a cache?",
            "Review lib/a.ex. Do not change anything. Tell me the real problems.",
            "Review this. Don\u{2019}t change anything.",
        ] {
            assert!(only_answers(request), "{request}");
        }
        for request in [
            "Every API call returns 500. Fix it.",
            "Can you fix the login?",
            "What is broken? Please fix it?",
            "Write tests for a.ex. Do not change the controller.",
            "Rename this and do not change anything else.",
            "The login fails. Why?",
            "Add a route",
        ] {
            assert!(!only_answers(request), "{request}");
        }
    }

    #[test]
    fn a_request_for_work_is_told_apart() {
        for request in ["Median gives the wrong answer. Fix it.", "Add parse_duration to durations.py", "Can you write tests?"] {
            assert!(asks_for_change(request), "{request}");
        }
        for request in ["What did we do last time?", "can you check what is wrong with this", "How do I add a route?", "Explain this file"] {
            assert!(!asks_for_change(request), "{request}");
        }
    }

    #[test]
    fn an_answer_only_request_can_look_and_test_but_not_change() {
        for command in ["ls -la", "git log --oneline -5", "mix test 2>&1 | tail -30", "grep -rn Repo lib", "cd sub && cargo test"] {
            assert!(changes_nothing(command), "{command}");
        }
        for command in ["mix precommit", "mix format", "echo x > a.txt", "find . -name a -delete", "rm a.txt", "mix test && mix format", "git checkout ."] {
            assert!(!changes_nothing(command), "{command}");
        }
    }
    use serde_json::json;

    fn item(text: &str, status: &str, proof: &str) -> PlanItem {
        PlanItem { text: text.into(), status: status.into(), proof: proof.into() }
    }

    #[test]
    fn looking_around_is_not_proof() {
        for command in ["ls -la", "cd app && ls", "cat a | head", "git status", "pwd; echo hi"] {
            assert!(only_looks(command), "{command}");
        }
        for command in ["mix test", "cd app && mix test", "ls && mix test", "git commit -m x", "npm run build"] {
            assert!(!only_looks(command), "{command}");
        }
    }

    #[test]
    fn a_redirect_and_a_version_check_are_still_looking() {
        for command in ["which mix 2>&1", "cat a 2>&1 | head", "mix --version 2>&1; which elixir 2>&1", "node --version"] {
            assert!(only_looks(command), "{command}");
        }
        for command in ["mix test 2>&1", "mix test --version-check", "rm --version x"] {
            assert!(!only_looks(command), "{command}");
        }
    }

    #[test]
    fn only_a_whole_commands_exit_code_counts() {
        for command in ["mix test", "cd app && mix test 2>&1", "mix test &> out.txt", "grep 'a|b' x", "python3 -c 'a; b'", "curl -f localhost:4000/a?x=1\\&y=2"] {
            assert!(exit_counts(command), "{command}");
        }
        for command in ["mix test | tail", "mix test; true", "mix test || true", "mix phx.server &", "mix test\ntrue"] {
            assert!(!exit_counts(command), "{command}");
        }
    }

    #[test]
    fn a_build_only_proves_setup() {
        let args = |text: &str, proof: &str| json!({"items": [{"text": text, "status": "done", "proof": proof}]});
        let built = vec!["cd app && mix compile 2>&1".to_string()];
        let refused = apply(&[], &args("Build REST API endpoints", "cd app && mix compile 2>&1"), &built).unwrap_err();
        assert!(refused.contains("shows that it builds"), "{refused}");
        assert!(apply(&[], &args("Create Phoenix project", "cd app && mix compile 2>&1"), &built).is_ok());
        let ran = vec!["cd app && mix compile".to_string(), "cd app && mix test".to_string()];
        assert!(apply(&[], &args("Build REST API endpoints", "cd app && mix test"), &ran).is_ok());
        assert!(only_builds("cd web && npm run build") && only_builds("cargo build --release"));
        assert!(!only_builds("cargo test") && !only_builds("mix compile && mix test") && !only_builds("npm run dev"));
    }

    #[test]
    fn a_request_can_stand_in_for_a_plan_but_still_needs_proof() {
        let plan = with_request(&[], "Every API call returns 500. Fix it.");
        assert_eq!((plan.len(), plan[0].status.as_str()), (1, "doing"));
        assert!(!settled(&plan));
        // Earlier items stay, and the same request is not added twice.
        let earlier = vec![item("old work", "done", "t")];
        let plan = with_request(&earlier, "new request");
        assert_eq!(plan.len(), 2);
        assert_eq!(with_request(&plan, "new request").len(), 2);
        // A model can finish it the normal way, and only with a command that passed.
        let args = json!({"items": [{"text": "Every API call returns 500. Fix it.", "status": "done", "proof": "mix test"}]});
        assert!(apply(&with_request(&[], "Every API call returns 500. Fix it."), &args, &["mix test".to_string()]).is_ok());
        assert!(apply(&with_request(&[], "Every API call returns 500. Fix it."), &args, &[]).is_err());
    }

    #[test]
    fn setting_up_a_feature_is_still_a_feature() {
        for item in ["Wire Cowboy to serve static files and set up routes", "Build the API and setup auth"] {
            assert!(!is_setup(item), "{item}");
        }
        for item in ["Set up Rust backend", "Setup Vue with Vite", "Step 1: Set up the database file", "Create the project", "Install dependencies", "Write the Dockerfile"] {
            assert!(is_setup(item), "{item}");
        }
    }

    #[test]
    fn a_refused_item_does_not_throw_out_the_good_ones() {
        let passed = vec!["rebar3 compile".to_string()];
        let args = json!({"items": [
            {"text": "Create the project", "status": "done", "proof": "rebar3 compile"},
            {"text": "REST API works", "status": "done", "proof": "rebar3 compile"},
            {"text": "Vue page", "status": "todo"},
        ]});
        let (plan, refused) = apply_keeping(&[], &args, &passed).unwrap();
        assert_eq!(plan[0].status, "done");
        assert_eq!(plan[1].status, "doing");
        assert!(plan[1].proof.is_empty());
        assert_eq!(plan[2].status, "todo");
        assert_eq!(refused.len(), 1);
        assert!(refused[0].contains("shows that it builds"), "{refused:?}");
        // The same plan with nothing wrong: nothing refused.
        let fine = json!({"items": [{"text": "Create the project", "status": "done", "proof": "rebar3 compile"}]});
        assert!(apply_keeping(&[], &fine, &passed).unwrap().1.is_empty());
        // Earlier proof still holds, and a lost item is still an error.
        let previous = vec![item("a", "done", "t"), item("b", "todo", "")];
        let kept = json!({"items": [{"text": "a", "status": "done", "proof": "t"}, {"text": "b", "status": "todo"}]});
        assert!(apply_keeping(&previous, &kept, &[]).unwrap().1.is_empty());
        let lost = json!({"items": [{"text": "a", "status": "done", "proof": "t"}]});
        assert!(apply_keeping(&previous, &lost, &[]).is_err());
    }

    #[test]
    fn a_shared_proof_is_refused_only_past_the_limit() {
        let passed = vec!["t".to_string()];
        let done = |text: &str| json!({"text": text, "status": "done", "proof": "t"});
        let (plan, refused) = apply_keeping(&[], &json!({"items": [done("one"), done("two"), done("three")]}), &passed).unwrap();
        assert_eq!(crate::plan::done_count(&plan), 2);
        assert_eq!(plan[2].status, "doing");
        assert!(refused[0].contains("already the proof"), "{refused:?}");
    }

    #[test]
    fn a_piped_proof_is_refused_for_that_reason() {
        let args = json!({"items": [{"text": "Create the project", "status": "done", "proof": "swift build 2>&1 | tail -5"}]});
        let refused = apply(&[], &args, &[]).unwrap_err();
        assert!(refused.contains("exit code comes from only the last part"), "{refused}");
    }

    #[test]
    fn one_command_cannot_prove_the_whole_plan() {
        let passed = vec!["npx vitest run".to_string()];
        let done = |text: &str| json!({"text": text, "status": "done", "proof": "npx vitest run"});
        assert!(apply(&[], &json!({"items": [done("Frontend works"), done("Frontend tests pass")]}), &passed).is_ok());
        let refused = apply(&[], &json!({"items": [done("Frontend works"), done("Frontend tests pass"), done("Backend works")]}), &passed).unwrap_err();
        assert!(refused.contains("already the proof"), "{refused}");
        // Items proved earlier stay proved.
        let previous = vec![item("a", "done", "t"), item("b", "done", "t"), item("c", "done", "t")];
        let same = json!({"items": [{"text":"a","status":"done","proof":"t"},{"text":"b","status":"done","proof":"t"},{"text":"c","status":"done","proof":"t"}]});
        assert!(apply(&previous, &same, &[]).is_ok());
    }

    #[test]
    fn a_container_item_needs_a_container_command() {
        let args = |proof: &str| json!({"items": [{"text": "Create Dockerfile and compose", "status": "done", "proof": proof}]});
        let refused = apply(&[], &args("npx vitest run"), &["npx vitest run".to_string()]).unwrap_err();
        assert!(refused.contains("does not touch containers"), "{refused}");
        assert!(apply(&[], &args("podman compose config"), &["podman compose config".to_string()]).is_ok());
    }

    #[test]
    fn done_needs_a_command_that_passed() {
        let args = |proof: &str| json!({"items": [{"text": "tests pass", "status": "done", "proof": proof}]});
        let passed = vec!["mix test".to_string()];
        assert!(apply(&[], &args(""), &passed).unwrap_err().contains("without proof"));
        assert!(apply(&[], &args("ls -la"), &["ls -la".to_string()]).unwrap_err().contains("only looks around"));
        let refused = apply(&[], &args("npm test"), &passed).unwrap_err();
        assert!(refused.contains("Commands that count right now: `mix test`"), "{refused}");
        assert!(apply(&[], &args("npm test"), &[]).unwrap_err().contains("none yet"));
        assert_eq!(apply(&[], &args("mix test"), &passed).unwrap()[0].status, "done");
        // The command with its output after it still counts.
        assert_eq!(apply(&[], &args("mix test -> 14 tests, 0 failures"), &passed).unwrap()[0].status, "done");
    }

    #[test]
    fn a_done_item_stays_done_after_later_edits() {
        let previous = vec![item("tests pass", "done", "mix test"), item("buzzer", "todo", "")];
        let args = json!({"items": [
            {"text": "tests pass", "status": "done", "proof": "mix test"},
            {"text": "buzzer", "status": "doing"},
        ]});
        assert_eq!(apply(&previous, &args, &[]).unwrap()[1].status, "doing");
    }

    #[test]
    fn items_cannot_quietly_disappear() {
        let previous = vec![item("buzzer", "todo", "")];
        let gone = json!({"items": [{"text": "something easier", "status": "todo"}]});
        assert!(apply(&previous, &gone, &[]).unwrap_err().contains("missing"));
        let no_reason = json!({"items": [{"text": "buzzer", "status": "dropped"}]});
        assert!(apply(&previous, &no_reason, &[]).unwrap_err().contains("with a reason"));
        let dropped = json!({"items": [{"text": "buzzer", "status": "dropped", "proof": "the user removed it"}]});
        assert!(settled(&apply(&previous, &dropped, &[]).unwrap()));
    }

    #[test]
    fn a_plan_is_settled_only_when_every_item_is() {
        assert!(!settled(&[]));
        assert!(!settled(&[item("a", "done", "x"), item("b", "todo", "")]));
        assert!(settled(&[item("a", "done", "x"), item("b", "dropped", "why")]));
    }

    #[test]
    fn the_reminder_carries_the_goal_and_the_next_item() {
        let plan = [item("database works", "done", "mix test"), item("buzzer", "todo", "")];
        let text = reminder("Quiz Night", &plan);
        assert!(text.contains("Goal: Quiz Night"));
        assert!(text.contains("[x] database works"));
        assert!(text.contains("[ ] buzzer"));
        assert!(text.contains("Work on: buzzer"));
    }
}
