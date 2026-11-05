# Ideas for Ferro64

What to build next, and what to take from other open source projects. Nothing here is committed to or scheduled. Ferro64's rules still apply: one service, no extra daemons, no Docker requirement, and models come only through Settings.

The notes on other projects come from reading their READMEs and docs on 2026-11-03. None of them were run, and their benchmark numbers are their own.

## Fix first

These came from a long agent run building an Elixir app.

1. **Runs that survive a page close, without overnight mode.** An overnight run does. A watched run still stops when its page connection drops, which is deliberate while it needs approvals.

## Built

- **A plan the agent has to follow** (`src/plan.rs`). It writes a plan before changing anything, the plan stays on screen and goes with every request, an item only counts as done with a command that passed since the last file change, and a run with open items ends as "unfinished". The step limit resets each time an item is finished. A plain `ls` no longer counts as checking a change. A piped or chained command cannot be proof, because its exit code hides failures, a build only proves setup items, one proof closes at most two feature items, and a container item needs a container command. After a larger build the agent is asked once to start the real app from a clean state. Still to prove on a large build.

- **Agent notes** (`src/notes.rs`), design from [Graft](https://github.com/AEndrix03/Graft), Apache-2.0, no code copied. Notes are matched on meaning and shared wording, recalled when a run starts, and saved after a plan is proved, only once a command has passed. A changed source file flags a note, and a replaced note can be restored. The Notes panel reads, deletes and turns them off. The thresholds come from 18 hand-written examples, so they need more data.

- **Progress during long waits.** Reading a long history back in now shows "reading N%", in chat and in Code.

- **Nested code blocks in chat.** A fence closes only on a bare fence at least as long as the one that opened it.

- **Checking pages in a real browser** (`open_page` in `src/coder.rs`). A build can return 200 and still be blank, as Home Lab Board was. The tool opens a local page in a Chrome, Chromium, Edge or Brave already on the machine and returns what rendered and any browser errors. Local addresses only, approved like a command.

- **Working rules in the agent prompt**, adapted from [andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills), MIT, itself derived from Andrej Karpathy's observations: the least code that does the job, an existing library over a hand-written one, a named check for each plan item, stated assumptions, and a reproducing test for a bug. The rules guide the model; the plan checks are what enforce anything. Whether they help is still being measured.

- **Spreadsheets in chat** (`src/sheet.rs`). Attach a CSV, TSV, XLSX, XLS or ODS and ask about it, with follow-ups in the same chat. The model writes read-only queries and Ferro64 runs them on an in-memory copy, so figures are calculated, and the queries are listed under the answer.

- **Overnight mode** (`src/overnight.rs`). Rule-based approvals, the macOS system sandbox, and automatic continuing. Not sandboxed on Linux or Windows yet.

- **Project knowledge across sessions** (`src/brief.rs`, `src/code_index.rs`). A summary written on the first read, and the code related to the task found before the model starts, only for strong matches: weak ones led the model away from a real bug in testing. The 0.55 cutoff comes from one project, so it needs more data.

## Take from other projects

In suggested order.

### 1. Reviewer checklist (from agency-agents)

[agency-agents](https://github.com/msitarzewski/agency-agents), MIT.

A large library of role prompts. Adapt the Code Reviewer's rules for the review stage in Agents: be specific, explain why, mark issues as blockers or suggestions, and give all feedback in one review.

- **Skip:** the personality text and bulk installs. They use up context on a local model and add little.
- **Size:** very small; a prompt change.

### 2. Approve the plan before rendering video (idea from OpenMontage)

[OpenMontage](https://github.com/calesthio/openmontage), AGPL-3.0.

Show a scene-by-scene plan to approve before anything renders. A 10-second clip takes about 9 minutes, so catching a bad plan early saves real time, and it fits continue-from-last-frame.

- **Do not use:** its code. It is a large Python project built around cloud providers, its local video runs through ComfyUI, and AGPL code cannot go into this MIT/Apache project without changing its terms.

### 3. Code map (maybe, from codebase-memory-mcp)

[codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp), MIT.

Answers structural questions that semantic search cannot: what calls this function, and what a change affects. Ferro64 has no MCP client, but the tool has a command-line mode, so the app could run it per job with no daemon.

- **When:** wait for a large codebase to test on. It will not help much on small apps.

## API models (hidden)

Built and tested once on Novita, then hidden: per-token billing is the wrong fit for a local app. Set `FERRO64_API_MODELS=on` to bring back the tab, the hosted entries in the Code view's model list, and the `/v1` proxy. Off, none of these exist and saved keys are not used.

Gaps if it comes back: the first-read summary isn't counted in a run's tokens, and the Code view's thinking switch isn't sent to vendors (about 90% of billed output in the one real run was hidden thinking).

**API models** is optional, and off until you add a key. Add keys for Novita, Groq, OpenAI, Claude, Gemini, xAI or any OpenAI-style address, pick which of their models to offer, and they appear in the Code view's model list. These are not local: while one is selected, your project's code goes to that vendor and you pay for every request.

**This gets expensive, so it is not the everyday way to use Ferro64.** An API keeps no memory, so every step of a run resends the whole conversation, and models bill for hidden thinking as well. In one test, 12 minutes spent diagnosing a small app cost about $0.25 at $2 / $6 per million tokens, before it had changed a line, and a few fixes a day can pass the price of a flat subscription. For everyday coding, use a local model and your own compiler, and take what it can't solve to a subscription you already pay for. Keep API keys for when you are truly stuck.

Each run shows the tokens it used and stops at a limit you set (10 million by default). Keys can also come from `NOVITA_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` and `XAI_API_KEY`, and commands the agent runs never see them.

The same tab gives one OpenAI-style address, `/v1/chat/completions`, so another tool can reach any of those models, or a local one, by name. It answers this machine only and needs the token shown in the tab. The run limit does not apply to it.

## Four "algorithms" for small models (reviewed, not built)

The names (shift-reduce gating, PID loop, thermodynamic condensation, monadic paging) are mostly labels on ordinary engineering. Nothing here adds knowledge a small model lacks. The useful question is which failure each one reaches. In our eight-app tests the failures were mainly wrong library APIs guessed from memory, type errors, and code that compiles but fails at runtime. Syntax errors were rarely what stopped a run.

| Idea | Use | What to build | What to leave out |
|---|---|---|---|
| 1. AST token gating | About 20% | Parse a file with tree-sitter before saving an edit, and refuse an edit that adds syntax errors | Halting mid-stream and injecting a correction. Code arrives inside a JSON tool argument, unfinished code always looks invalid, and the compiler catches syntax in seconds anyway |
| 2. Compiler feedback loop | About 70%, the best of the four | Run the project's check automatically after edits, return only the first few errors, parsed where the tool offers JSON. When an error names a missing function or module, attach the real API from the installed library | Checking in a scratch file: most code cannot compile outside its project. The "PID" framing adds nothing; retries already stop on stalls |
| 3. Context condensation | About 60% | Remove repeated output lines. Show signatures first and bodies on request. Collapse old file reads, which made up most of the context in the one run measured | The 1,500-token target: the first request alone was 2.8K tokens. Removing ANSI colours is cheap but rarely matters, since commands run without a terminal |
| 4. Predictive VRAM paging | About 10% | For agent runs, load the next stage's model as the last one finishes | Prediction and the Markov model, since the stage order is already known. Keeping a 3B router loaded: small models were the ones that failed, and a bad router sends work the wrong way |

Results on an M1 Max, six small tasks with hidden tests, one run per setting:

| Model | Before fixes, check on / off | After fixes | Speed |
|---|---|---|---|
| Qwen3.5 4B | 4/6, 4/6 | 6/6 | 43 tok/s |
| Qwen3.5 9B | 6/6, 6/6 | 5/6 | 34 tok/s |
| Gemma 4 E4B | 2/6, 2/6 | 3/6 | 48 tok/s |
| Granite 4.1 8B | 1/6, 1/6 | not rerun | 34 tok/s |

Built: the compile check with API lookup (`check.rs`, off with `FERRO64_AUTOCHECK=off`), the save check (`syntax.rs`; no Erlang), repeated-line removal, and a nudge when a run that should change files has only written text. The compile check did not change pass counts on tasks this small, and the API lookup never fired. Skipped: idea 4, since a cached model loads in 1.5 to 4 seconds. Not built: signatures before bodies from idea 3.
