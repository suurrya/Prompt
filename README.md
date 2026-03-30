# IT Helpdesk Agent Benchmark

> **What we're building:** Four functionally identical IT Helpdesk AI agents whose
> only variable is the *prompting strategy* they use. The goal is to rigorously
> measure which technique produces the most accurate tool-selection decisions.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [File Reference](#3-file-reference)
4. [The Four Experiments](#4-the-four-experiments)
5. [The 13 Tools](#5-the-13-tools)
6. [Installation & Setup](#6-installation--setup)
7. [Running the Benchmark](#7-running-the-benchmark)
8. [Running the NiceGUI Chat UI](#8-running-the-nicegui-chat-ui)
9. [Understanding the Results](#9-understanding-the-results)
10. [Design Decisions](#10-design-decisions)

---

## 1. Project Overview

This project answers the question:

> *Does the way you prompt an LLM significantly affect how accurately it
> selects tools in a real-world agentic scenario?*

We test four prompting strategies — Static Few-Shot, Static Chain-of-Thought,
Dynamic Few-Shot, and Dynamic Chain-of-Thought — each applied to a simulated
IT Helpdesk agent built on the `smolagents` framework. Every agent shares the
same LLM backend (`gpt-4o-mini`), the same 13 tools, and the same knowledge
base. The only thing that changes is what the agent reads in its system prompt.

Accuracy is measured against a 20-case labelled test suite spanning 8
categories (auth, network, hardware, software, security, access, status,
user_info) and 3 difficulty tiers (easy, medium, hard).

---

## 2. Architecture

```
it_helpdesk_agents/
│
├── tools.py                        ← 13 shared @tool functions (constant)
├── knowledge_base.py               ← Static KB articles + keyword search
├── requirements.txt                ← All dependencies (including nicegui)
├── .env.example                    ← API key template
│
├── project_1_few_shot/
│   ├── prompts.py                  ← Terse few-shot prompt (no reasoning)
│   └── agents.py                   ← Static-prompt ITHelpdeskAgent
│
├── project_2_chain_of_thought/
│   ├── prompts.py                  ← CoT prompt with reasoning framework
│   └── agents.py                   ← Static-prompt ITHelpdeskAgent
│
├── project_3_dynamic_few_shot/
│   ├── prompts.py                  ← Example DB + TF-IDF selector + template
│   └── agents.py                   ← __call__ overridden, rebuilds prompt per query
│
├── project_4_dynamic_cot/
│   ├── prompts.py                  ← CoT Example DB + TF-IDF selector + template
│   └── agents.py                   ← __call__ overridden, rebuilds CoT prompt per query
│
├── evaluation/
│   ├── test_cases.py               ← 20 labelled test cases
│   └── run_eval.py                 ← Benchmark runner + report generator
│
└── ui/
    └── app.py                      ← NiceGUI 4-panel chat interface
```

**Controlled variables (same across all agents):**
- LLM model: `gpt-4o-mini` via `OpenAIServerModel`
- Tools: all 13 from `tools.py`
- Knowledge base: `knowledge_base.py`
- API key: `.env`

**Independent variable (changes per experiment):**
- The system prompt — its content, structure, and whether it's static or dynamic.

---

## 3. File Reference

### `tools.py`
The single source of truth for every tool available to all four agents.
Contains 13 `@tool`-decorated functions grouped by domain:

| Domain | Tools |
|---|---|
| Ticket Management | `create_ticket`, `escalate_ticket` |
| Knowledge Base | `lookup_knowledge_base` |
| Auth | `reset_password` |
| User & Account | `get_user_info`, `lookup_user_account` |
| Infrastructure | `check_system_status`, `schedule_maintenance` |
| Billing | `process_refund` |
| Memory / History | `store_resolved_ticket`, `save_ticket_to_long_term_memory`, `get_user_long_term_memory`, `get_customer_history` |

Each function's docstring is the tool description the LLM reads — it explains
*when* to use the tool, not just *what* it does. Well-written docstrings are
as important as the prompting strategy for accurate selection.

Memory tools use an in-process Python dict (`_LONG_TERM_MEMORY`) as a
drop-in replacement for Chroma/vector DBs, keeping the benchmark dependency-free
and reproducible.

### `knowledge_base.py`
A static list of 15 IT support articles covering common issues (VPN, MFA,
screen flicker, printer, Office, Teams, etc.). Provides a `search_knowledge_base()`
function that ranks articles by keyword overlap — deterministic and fast, no
embedding API call required.

### `project_*/prompts.py`
Each experiment has its own `prompts.py` with a philosophy and design tailored
to that technique. See [Section 4](#4-the-four-experiments) for details.

### `project_*/agents.py`
Wraps `smolagents.ToolCallingAgent`. All four expose the same interface:
`ITHelpdeskAgent()(user_query) → str`. Experiments 1 & 2 set the system
prompt once at `__init__`; Experiments 3 & 4 override `__call__` to
rebuild the prompt dynamically for every query.

### `evaluation/test_cases.py`
20 labelled IT helpdesk queries, each annotated with:
- `expected_tool` — the single tool that should be called first
- `category` — the problem domain
- `difficulty` — easy / medium / hard
- `notes` — why this case is interesting or where agents might fail

### `evaluation/run_eval.py`
The benchmark engine. For each agent:
1. Runs every test case via `agent(query)`
2. Extracts the first tool called from `agent._agent.memory.steps` (or
   `agent._last_agent.logs` for dynamic agents), with fallback regex scan
3. Compares against `expected_tool`
4. Prints a side-by-side comparison table with per-category and
   per-difficulty breakdowns
5. Saves full results to `evaluation/results.json`

Supports `--debug` mode which prints full Python tracebacks for any error
instead of silently recording it as a miss.

### `ui/app.py`
A NiceGUI single-page app that shows all four chatbots side-by-side in one
browser window. A shared input bar broadcasts each query to all four agents
simultaneously via `asyncio.gather`, so you can watch the agents respond in
parallel and directly compare latency and answer quality.

---

## 4. The Four Experiments

### Experiment 1 — Static Few-Shot (`project_1_few_shot`)

**Philosophy:** keep the prompt as short and direct as possible. Show the
model concrete input→output pairs and trust it to generalise by pattern-matching.

**Prompt structure:**
```
[Role sentence]
[Compact tool reference table]
[18 fixed User: "..." → tool_call() pairs]
[Call the tool now.]
```

**Strengths:** minimal token cost, fully predictable prompt, easy to debug.

**Weaknesses:** examples may not match the incoming query closely; no
reasoning scaffold means the model may pattern-match incorrectly on
surface-level similarity.

---

### Experiment 2 — Static Chain-of-Thought (`project_2_chain_of_thought`)

**Philosophy:** teach the model *how to think*, not just what to do.
Every example includes a multi-line `Thought:` block that works through
the decision step-by-step before committing to a tool.

**Prompt structure:**
```
[Role sentence]
[Tool list]
[REASONING FRAMEWORK — 3-step decision guide]
[10 rich CoT examples with Thought: + Action:]
[Apply the framework and call the tool.]
```

**Key addition vs. Exp 1:** a mandatory reasoning framework block that
prompts the model to classify the problem type, pick the right tool tier,
and apply safety rules (e.g. "never create_ticket for outages without
checking check_system_status first") even for queries with no matching example.

**Strengths:** better accuracy on ambiguous or multi-step queries; the
reasoning trace catches errors before they propagate to the final action.

**Weaknesses:** higher token cost per call; examples are fewer (quality over quantity).

---

### Experiment 3 — Dynamic Few-Shot (`project_3_dynamic_few_shot`)

**Philosophy:** make the examples *contextually relevant* rather than fixed.

**How it works:**
1. At call-time, `build_system_prompt(user_query)` is called.
2. TF-IDF cosine similarity ranks all 35 entries in `EXAMPLE_DATABASE`
   against the incoming query (bigrams, stop-word filtered).
3. The top 4 most similar examples are injected into a lean template.
4. A fresh `ToolCallingAgent` is instantiated with that prompt.

**Prompt structure:**
```
[Role sentence]
[Tool list]
[5 safety rules]
[TOP 4 DYNAMICALLY SELECTED examples]
[Call the tool.]
```

**Strengths:** the model always sees examples close to the actual problem —
dramatically reduces the "topic gap" that hurts static prompting on edge cases.

**Weaknesses:** small runtime overhead per call (TF-IDF transform + agent
re-init); no reasoning traces means the model still relies on pattern-matching.

---

### Experiment 4 — Dynamic Chain-of-Thought (`project_4_dynamic_cot`)

**Philosophy:** combine dynamic relevance (Exp 3) with reasoning depth (Exp 2).
This is the most advanced and highest-hypothesised-accuracy strategy.

**How it works:**
1. TF-IDF ranks all 28 entries in `COT_EXAMPLE_DATABASE`.
2. Top 3 are selected (fewer than Exp 3 because CoT examples are ~3× longer).
3. Injected into a template that also includes a "DIAGNOSTIC QUESTIONS" block
   that elicits reasoning even for queries with no close match.

**Prompt structure:**
```
[Role sentence]
[Tiered tool list (TIER 1–4)]
[DIAGNOSTIC QUESTIONS — 5 questions to answer before acting]
[TOP 3 DYNAMICALLY SELECTED CoT examples with Thought: + Action:]
[Reason through and call the tool.]
```

**Strengths:** best of both worlds — relevant examples AND deliberate
reasoning. Expected to win on hard cases (security escalations, outage
vs. local fault, KB-resolvable vs. ticket-required boundary).

**Weaknesses:** highest token cost; most complex to maintain (example DB
needs CoT traces per entry).

---

## 5. The 13 Tools

| Tool | When to use | Category |
|---|---|---|
| `lookup_knowledge_base(query)` | First choice for documented, self-service issues | KB |
| `create_ticket(category, priority, summary, email)` | Hands-on work, access provisioning, installs | Ticket |
| `escalate_ticket(ticket_id, reason, escalate_to)` | After create_ticket for security/P1 incidents | Ticket |
| `reset_password(email, method)` | User actively locked out, cannot access self-service portal | Auth |
| `get_user_info(email)` | Directory lookup — department, manager, devices, status | User |
| `lookup_user_account(email)` | Subscription tier, block status, billing context | User |
| `check_system_status(service_name)` | Before any outage-related ticket — check if already known | Status |
| `schedule_maintenance(asset_id, type, date, email)` | Physical on-site work (RAM upgrade, screen, battery) | Hardware |
| `process_refund(reservation_id)` | Direct refund for a known reservation ID | Billing |
| `store_resolved_ticket(user_id, summary)` | Brief archive of a closed ticket | Memory |
| `save_ticket_to_long_term_memory(user_id, summary, resolution)` | Full outcome archive (issue + fix) | Memory |
| `get_user_long_term_memory(user_id)` | Full history for a returning user | Memory |
| `get_customer_history(user_id)` | Quick past-issues summary before triaging | Memory |

---

## 6. Installation & Setup

### Prerequisites
- Python 3.10+
- An OpenAI API key with access to `gpt-4o-mini`

### Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `smolagents`, `openai`, `scikit-learn`, `numpy`,
`python-dotenv`, and `nicegui`.

### Configure your API key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```
OPENAI_API_KEY=sk-your-real-key-here
```

---

## 7. Running the Benchmark

All commands are run from the `it_helpdesk_agents/` directory.

```bash
# Full benchmark — all 4 experiments × 20 test cases
python evaluation/run_eval.py

# Run only specific experiments
python evaluation/run_eval.py --experiments 3 4

# Skip an experiment
python evaluation/run_eval.py --skip 1

# Run a subset of test cases
python evaluation/run_eval.py --tests TC-001 TC-005 TC-014

# Verbose mode — print agent reasoning traces
python evaluation/run_eval.py --verbose

# Debug mode — print full error tracebacks + step-log diagnostics
# Run this first if you see NONE or 0% across the board
python evaluation/run_eval.py --debug --experiments 1 --tests TC-001
```

### Troubleshooting 0% accuracy

If every result shows `[NONE]` and 0.0s latency, the agent is crashing before
making an API call. Run `--debug` to see the full traceback:

```bash
python evaluation/run_eval.py --debug --experiments 1 --tests TC-001
```

Common causes:
| Symptom | Cause | Fix |
|---|---|---|
| `AuthenticationError` | Wrong API key | Check `.env` |
| `TypeError: unexpected keyword argument 'system_prompt'` | Old smolagents | Already fixed — use `prompt_templates` dict then patch |
| `AssertionError: Some prompt templates are missing` | Only `system_prompt` key passed | Already fixed — init agent first, then patch `agent.prompt_templates["system_prompt"]` |
| `ModuleNotFoundError` | Missing dep | `pip install -r requirements.txt` |

---

## 8. Running the NiceGUI Chat UI

```bash
python ui/app.py
```

Then open **http://localhost:8080** in your browser.

### What you'll see

Four side-by-side chat panels, one per prompting strategy, colour-coded:
- 🟣 Indigo — Experiment 1 (Static Few-Shot)
- 🔵 Sky — Experiment 2 (Static Chain-of-Thought)
- 🟢 Emerald — Experiment 3 (Dynamic Few-Shot)
- 🟡 Amber — Experiment 4 (Dynamic Chain-of-Thought)

A shared input bar at the bottom sends each query to **all four agents
simultaneously** via `asyncio.gather + ThreadPoolExecutor(max_workers=4)`.
Response latency is shown under each bubble so you can compare thinking time directly.

---

## 9. Understanding the Results

### What the benchmark measures
The evaluator checks whether the **first tool called** matches `expected_tool`
in `test_cases.py`. This is intentional — in a production helpdesk, calling
the wrong first tool wastes time (e.g. creating a ticket before checking for
a known outage) even if subsequent tools are correct.

### What to look for

**Easy cases (TC-001, TC-002, TC-004…)** should be correct in all four agents.
If any agent fails easy cases, check for an API error (run `--debug`).

**Medium cases** separate static from dynamic — dynamic agents should do better
because they select topically close examples.

**Hard cases (TC-013, TC-014, TC-015)** are where CoT matters most:
- TC-013 (Outlook → check email status, not KB) requires understanding that a
  server hint means "check outage first"
- TC-014/015 (phishing/ransomware → create_ticket not lookup_knowledge_base)
  require security-domain reasoning, not just keyword matching

### Reading the summary table

```
ACCURACY     0% (0/20)  65% (13/20)  70% (14/20)  85% (17/20)
              Exp 1       Exp 2        Exp 3         Exp 4
```

The hypothesis is Exp 4 > Exp 3 > Exp 2 > Exp 1. The per-category and
per-difficulty breakdowns show exactly *where* each technique gains or loses.

---

## 10. Design Decisions

| Decision | Rationale |
|---|---|
| `gpt-4o-mini` as the constant LLM | Cheap, fast, widely available — isolates prompting as the variable |
| TF-IDF over embedding similarity | Deterministic, no API call, fully reproducible across runs |
| `smolagents ToolCallingAgent` | Native tool-calling interface; schema auto-generated from docstrings |
| In-memory dict for long-term memory | Removes Chroma/DB dependency; benchmark stays portable |
| First-tool-called as the accuracy metric | Reflects real-world cost of wrong initial routing decisions |
| 20 test cases across 3 difficulty tiers | Large enough to show meaningful differences; small enough to run cheaply |
| Agent init + prompt patch (not `prompt_templates={...}`) | Compatible with all smolagents versions; avoids AssertionError on missing keys |
