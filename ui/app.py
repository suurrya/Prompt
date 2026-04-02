
"""
ui/app.py  —  IT Helpdesk Prompt Technique Showcase
=====================================================
Four panels side-by-side. Same query → four different prompting strategies.
Each response is rendered as a structured, annotated dossier.

Persistence
-----------
All user chat history is stored in app.storage.user using plain dictionaries.
When a user returns, their entire conversation history is restored.

Run:
    cd it_helpdesk_agents && python ui/app.py
    Open http://localhost:8000
"""

from __future__ import annotations
import importlib # Purposes: Used to dynamically 'hot-load' the 4 different agent experiments at runtime.
import asyncio, html as _html, os, re, sys, time # Purposes: Core utilities for async runs, security, and timing comparisons.
from concurrent.futures import ThreadPoolExecutor # Purposes: Allows running multiple AI calls in parallel without blocking the UI.
from dotenv import load_dotenv # Purposes: Loads project-wide environment variables from .env.
from nicegui import app, ui # Purposes: The 'Web Engine' that renders the beautiful, reactive dashboard.
import markdown # Purposes: Converts technical AI thoughts into pretty, readable text.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))

_executor = ThreadPoolExecutor(max_workers=4)
_agents: dict[int, object] | None = None

# 2. EXPERIMENT SETTINGS
# ----------------------
# This dictionary defines the 4 experiments we are demonstrating. 
# Each has a name, a color, and a short "pitch" explaining its technique.
# Purposes: The "Central Config" for the UI. 
# It maps each project folder to a visual style (colors, icons, and 'pitch' text).
EXPERIMENTS = {
    1: {
        "short": "Static Few-Shot",
        "color": "#6366f1",
        "bg": "#eef2ff",
        "border": "#c7d2fe",
        "icon": "🗂️",
        "tag": "STATIC · NO REASONING",
        "pitch": "Uses a fixed list of examples. Fast, but limited to what it's been shown.",
    },
    2: {
        "short": "Static Chain-of-Thought",
        "color": "#0ea5e9",
        "bg": "#e0f2fe",
        "border": "#bae6fd",
        "icon": "🧠",
        "tag": "STATIC · WITH REASONING",
        "pitch": "Fixed examples, but each includes 'Thought' steps to show the AI's logic.",
    },
    3: {
        "short": "Dynamic Few-Shot",
        "color": "#10b981",
        "bg": "#d1fae5",
        "border": "#a7f3d0",
        "icon": "🔍",
        "tag": "DYNAMIC · NO REASONING",
        "pitch": "Picks the most relevant examples for your specific question using TF-IDF.",
    },
    4: {
        "short": "Dynamic Chain-of-Thought",
        "color": "#f59e0b",
        "bg": "#fef3c7",
        "border": "#fde68a",
        "icon": "✨",
        "tag": "DYNAMIC · WITH REASONING",
        "pitch": "The most advanced strategy: Dynamic examples PLUS step-by-step reasoning.",
    },
}

# 3. HELPER FUNCTIONS FOR DISPLAY
# -------------------------------

# This helps us pick an emoji icon based on which tool the AI decided to use.
TOOL_EMOJI = {
    "lookup_knowledge_base": "📖",
    "create_ticket": "🎫",
    "escalate_ticket": "🚨",
    "reset_password": "🔑",
    "get_user_info": "👤",
    "lookup_user_account": "💳",
    "check_system_status": "📡",
    "schedule_maintenance": "🔧",
    "process_refund": "💰",
    "store_resolved_ticket": "💾",
    "save_ticket_to_long_term_memory": "🗄️",
    "get_user_long_term_memory": "🕐",
    "get_customer_history": "📋",
}

def get_human_friendly_tool_summary(tool_name: str, arguments: dict) -> str:
    """
    This function converts a technical tool call into a nice English sentence.
    For example: 'create_ticket(priority="high")' becomes 'I've created a high priority ticket.'
    """
    # Define templates for each tool
    templates = {
        "reset_password": "I've initiated a password reset for {user_email} via {method}.",
        "lookup_knowledge_base": "I've searched the knowledge base for: \"{query}\".",
        "create_ticket": "I've created a {priority} {category} ticket: \"{summary}\".",
        "escalate_ticket": "I've escalated ticket {ticket_id} to {escalate_to}.",
        "check_system_status": "I've checked the live status of the {service_name} system.",
        "get_user_info": "I've retrieved directory details for {user_email}.",
        "lookup_user_account": "I've looked up the account status for {email}.",
        "schedule_maintenance": "I've booked a {maintenance_type} slot for {asset_id}.",
        "process_refund": "I've initiated a refund for reservation {reservation_id}.",
    }
    
    # Try to find the template for this tool
    template = templates.get(tool_name)
    if not template:
        # If we don't have a template, just return a generic message
        return f"I've decided to use the {tool_name} tool."
    
    # Use a default value if an argument is missing to avoid crashing
    safe_args = {
        "user_email": arguments.get("user_email", "the account"),
        "method": arguments.get("method", "email"),
        "query": arguments.get("query", "your issue"),
        "priority": arguments.get("priority", "medium"),
        "category": arguments.get("category", "IT"),
        "summary": arguments.get("summary", "Issue reported"),
        "ticket_id": arguments.get("ticket_id", "INC-XXXX"),
        "escalate_to": arguments.get("escalate_to", "the specialist team"),
        "service_name": arguments.get("service_name", "IT"),
        "email": arguments.get("email", "the user"),
        "maintenance_type": arguments.get("maintenance_type", "maintenance"),
        "asset_id": arguments.get("asset_id", "the device"),
        "reservation_id": arguments.get("reservation_id", "REF-XXXX"),
    }
    
    # Return the formatted string
    return template.format(**safe_args)

def parse_argument_string(args_str: str) -> dict:
    """
    Takes a string like 'user_email="bob@corp.com", method="sms"'
    and converts it into a Python dictionary: {'user_email': 'bob@corp.com', 'method': 'sms'}
    """
    args = {}
    if not args_str:
        return args
        
    # We split the string by commas that are followed by a 'key=' pattern.
    # This is slightly complex but ensures we don't break on commas inside quotes.
    pairs = re.split(r',\s*(?=\w+\s*=)', args_str)
    for part in pairs:
        if "=" in part:
            key, value = part.split("=", 1)
            # Remove whitespace and quotes from the key and value
            args[key.strip()] = value.strip().strip('"\'')
    return args

def load_all_experiment_agents() -> dict[int, object]:
    """
    Imports and initializes the 4 different AI agents used in our experiments.
    """
    agents = {}
    # Each experiment is located in its own folder
    experiment_paths = {
        1: "project_1_few_shot.agents", 
        2: "project_2_chain_of_thought.agents",
        3: "project_3_dynamic_few_shot.agents", 
        4: "project_4_dynamic_cot.agents",
    }
    
    for exp_id, path in experiment_paths.items():
        # Dynamically import the module
        module = importlib.import_module(path)
        # Create an instance of the agent class
        agents[exp_id] = module.ITHelpdeskAgent(verbose=False)
    
    return agents

def escape_html_text(text: str) -> str:
    """A safe way to convert text into HTML-friendly characters (to prevent hacking/errors)."""
    return _html.escape(str(text))

def parse_dossier(text: str) -> dict:
    """
    Purposes: This is the 'Regex Extraction Engine'.
    It takes the raw text 'dossier' from the agents and breaks 
    it back into data (the reasoning, the tool name, and the arguments).
    """
    out = {"examples": [], "thought": "", "tool_name": "", "tool_args_str": "",
           "error": "", "error_type": ""}

    # ── Errors ────────────────────────────────────────────────────────────
    if "❌" in text or text.strip().startswith("⚠️"):
        m = re.search(r"❌\s*Error[:\s]*\n?`?([^`]*)`?", text, re.DOTALL)
        msg = m.group(1).strip() if m else text.strip()
        if not msg or msg in ('""', "''", ""):
            out["error"] = "Agent reached maximum steps without completing the task."
            out["error_type"] = "max_steps"
        elif "max step" in msg.lower():
            out["error"] = msg; out["error_type"] = "max_steps"
        elif "not in the tool" in msg.lower() or "schema" in msg.lower():
            out["error"] = msg; out["error_type"] = "schema"
        else:
            out["error"] = msg or repr(text); out["error_type"] = "general"
        return out

    # ── Examples (Exp 3 & 4) ─────────────────────────────────────────────
    examples = re.search(r"(?:Examples selected|CoT Examples selected)[^\n]*:\n(.*?)\n\n", text, re.DOTALL)
    if examples:
        for line in examples.group(1).split("\n"):
            line = line.strip().lstrip("- ").strip("`").rstrip(".").strip()
            if line:
                out["examples"].append(line)

    # ── Thought — try every known format ─────────────────────────────────
    # 1. Fenced ```markdown block (Exp 2 & 4 format)
    tm = re.search(r"```(?:markdown)?\n(.*?)```", text, re.DOTALL)
    if tm and tm.group(1).strip():
        out["thought"] = tm.group(1).strip()

    # 2. Content between ### Chain of Thought and ### Decision (Exp 2)
    if not out["thought"]:
        cot = re.search(
            r"###\s*(?:Chain of Thought|Final Chain of Thought)\s*\n+(.*?)(?=###|\Z)",
            text, re.DOTALL | re.IGNORECASE
        )
        if cot:
            candidate = re.sub(r"```(?:markdown)?|```", "", cot.group(1)).strip()
            if candidate:
                out["thought"] = candidate

    # 3. Inline "Thought: ..." line (Exp 3 sometimes)
    if not out["thought"]:
        it = re.search(r"Thought:\s*(.+?)(?=\n\n|###|\Z)", text, re.DOTALL)
        if it and it.group(1).strip():
            out["thought"] = it.group(1).strip()

    # 4. Q1/Q2/Q3 structured reasoning block anywhere in the text (model
    #    sometimes outputs the diagnostic framework directly without a label)
    if not out["thought"]:
        qblock = re.search(
            r"((?:Q[1-5][:\.\s].*?)(?:\n.*?){2,}(?:→|->|Action:).*)",
            text, re.DOTALL
        )
        if qblock:
            out["thought"] = qblock.group(1).strip()

    # 5. Any paragraph before ### Decision that isn't a section header
    #    (catch free-form reasoning the model produced)
    if not out["thought"]:
        before_decision = re.search(
            r"###\s*(?:Dynamic (?:Few-Shot|CoT) Selection[^\n]*\n.*?)?\n\n(.+?)(?=###\s*Decision|\Z)",
            text, re.DOTALL | re.IGNORECASE
        )
        if before_decision:
            candidate = before_decision.group(1).strip()
            # Only use it if it looks like deliberation, not just the examples list
            if len(candidate) > 40 and not candidate.startswith("-"):
                out["thought"] = candidate

    # ── Tool call — only from the Decision section ────────────────────────
    dm = re.search(r"###\s*Decision\s*\n(.*?)(?=###|\Z)", text, re.DOTALL | re.IGNORECASE)
    search = dm.group(1) if dm else text
    tm2 = re.search(r"`(\w+)\(([^`]*)\)`", search)
    if tm2:
        out["tool_name"] = tm2.group(1)
        out["tool_args_str"] = tm2.group(2).strip()

    return out

# ── HTML builders ──────────────────────────────────────────────────────────

def _response_card_html(tool_name: str, args_str: str, color: str, latency: float,
                        panel_idx: int) -> str:
    """Main visible card — clean human-readable decision."""
    emoji  = TOOL_EMOJI.get(tool_name, "⚙️")
    answer = escape_html_text(get_human_friendly_tool_summary(tool_name, parse_argument_string(args_str)))
    cid    = f"details-{panel_idx}"
    return (
        f'<div style="background:{color}12;border:1.5px solid {color}50;border-radius:10px;'
        f'padding:11px 13px;margin:0 0 6px;">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;">'
        f'<span style="font-size:20px;">{emoji}</span>'
        f'<span style="font-size:13.5px;font-weight:700;color:{color};font-family:monospace;">'
        f'{escape_html_text(tool_name)}</span>'
        f'<span style="margin-left:auto;font-size:10.5px;color:#9ca3af;background:#f3f4f6;'
        f'border-radius:99px;padding:2px 7px;">{latency:.1f}s</span>'
        f'</div>'
        f'<div style="font-size:13px;color:#1e293b;line-height:1.55;">{answer}</div>'
        f'<div style="margin-top:8px;">'
        f'<button onclick="var d=document.getElementById(\'{cid}\');'
        f'd.style.display=d.style.display===\'none\'?\'block\':\'none\';" '
        f'style="font-size:10.5px;color:{color};background:none;border:none;cursor:pointer;'
        f'padding:0;font-weight:600;">▸ Show reasoning &amp; details</button>'
        f'</div></div>'
        f'<div id="{cid}" style="display:none;">'
    )

def _details_html(parsed: dict, color: str, exp_id: int) -> str:
    """
    Educational reasoning section — explains HOW each prompt technique
    arrived at its decision, with per-technique context.
    """
    parts = []
    meta = EXPERIMENTS[exp_id]

    # ── Per-technique "How I reasoned" banner ─────────────────────────────
    HOW_I_REASONED = {
        1: ("🗂️ How Static Few-Shot works",
            "This agent has no reasoning step. It scans its fixed list of "
            "(query → tool) examples and calls whichever tool the closest "
            "match demonstrated. No deliberation — pure pattern matching."),
        2: ("🧠 How Static Chain-of-Thought works",
            "Before acting, this agent was instructed to work through a "
            "5-question diagnostic framework (problem type → KB resolvable? "
            "→ outage? → security? → physical?). The reasoning trace below "
            "shows exactly how it classified your query."),
        3: ("🔍 How Dynamic Few-Shot works",
            "At call-time, TF-IDF cosine similarity ranked the entire example "
            "database and injected only the most relevant matches into the "
            "prompt. The examples below are what the agent actually saw — "
            "not a fixed list, but the closest matches to your specific query."),
        4: ("✨ How Dynamic Chain-of-Thought works",
            "This is the most advanced strategy: TF-IDF retrieves the most "
            "relevant CoT-annotated examples, then the agent reasons through "
            "the same 5-question framework. The examples AND the reasoning "
            "trace both adapt to each query."),
    }

    title, explanation = HOW_I_REASONED[exp_id]
    parts.append(
        f'<div style="background:{color}0a;border:1px solid {color}30;'
        f'border-radius:8px;padding:9px 12px;margin-bottom:10px;">'
        f'<div style="font-size:11px;font-weight:700;color:{color};margin-bottom:4px;">'
        f'{title}</div>'
        f'<div style="font-size:11px;color:#475569;line-height:1.6;">{explanation}</div>'
        f'</div>'
    )

    # ── TF-IDF examples (Exp 3 & 4) ──────────────────────────────────────
    if parsed["examples"]:
        chips = ""
        for e in parsed["examples"]:
            chips += (
                f'<div style="background:#fff;border:1px solid {color}30;'
                f'border-left:3px solid {color};border-radius:4px;'
                f'padding:4px 9px;margin-bottom:4px;font-size:11px;'
                f'color:#374151;font-family:monospace;line-height:1.4;">'
                f'{escape_html_text(e[:80])}{"…" if len(e) > 80 else ""}</div>'
            )
        parts.append(
            f'<div style="margin-bottom:10px;">'
            f'<div style="font-size:10px;font-weight:700;color:{color};'
            f'text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px;">'
            f'🔍 Examples retrieved for this query</div>'
            f'<div style="font-size:10.5px;color:#64748b;margin-bottom:5px;">'
            f'These were selected by TF-IDF similarity — not fixed, but the closest '
            f'matches from the database to <em>your</em> exact query.</div>'
            f'{chips}</div>'
        )
    elif exp_id in (3, 4):
        # Explain why no examples are shown even for dynamic agents
        parts.append(
            f'<div style="font-size:11px;color:#94a3b8;font-style:italic;margin-bottom:8px;">'
            f'No retrieved examples available for this response.</div>'
        )

    # ── Reasoning trace ────────────────────────────────────────────────────
    if parsed["thought"]:
        thought_label = {
            1: "Agent output (no structured reasoning)",
            2: "Diagnostic reasoning trace",
            3: "Agent's deliberation",
            4: "Diagnostic reasoning trace",
        }.get(exp_id, "Agent reasoning")

        thought_note = {
            1: "Exp 1 has no reasoning step — any text here is incidental model output.",
            2: "The agent worked through Q1→Q5 before choosing the tool.",
            3: "Dynamic selection provided context, but reasoning here is informal.",
            4: "Both relevant examples AND structured reasoning informed the decision.",
        }.get(exp_id, "")

        lines_html = ""
        for line in parsed["thought"].split("\n"):
            line = line.strip()
            if not line:
                continue
            # Q1: / Q2: style lines — highlight the question number
            qm = re.match(r"^(Q[1-5][\.:]\s*)(.+)", line)
            if qm:
                lines_html += (
                    f'<div style="display:flex;gap:6px;margin-bottom:5px;align-items:baseline;">'
                    f'<span style="font-size:10.5px;font-weight:700;color:{color};'
                    f'min-width:24px;flex-shrink:0;">{escape_html_text(qm.group(1).rstrip())}</span>'
                    f'<span style="font-size:11.5px;color:#1e293b;">{escape_html_text(qm.group(2))}</span>'
                    f'</div>'
                )
            # → or -> decision arrows
            elif re.match(r"^\s*[→\-\>]", line):
                lines_html += (
                    f'<div style="display:flex;gap:6px;margin-bottom:5px;margin-top:2px;'
                    f'background:{color}15;border-radius:5px;padding:4px 7px;">'
                    f'<span style="font-size:13px;color:{color};">→</span>'
                    f'<span style="font-size:11.5px;font-weight:600;color:{color};">'
                    f'{escape_html_text(re.sub(r"^[→\->]+\s*", "", line))}</span>'
                    f'</div>'
                )
            # Bullet points
            elif line.startswith(("•", "-", "*")):
                lines_html += (
                    f'<div style="display:flex;gap:5px;margin-bottom:4px;">'
                    f'<span style="color:{color};flex-shrink:0;margin-top:1px;">›</span>'
                    f'<span style="font-size:11.5px;color:#374151;">'
                    f'{escape_html_text(line.lstrip("•-* "))}</span></div>'
                )
            # Numbered lines (1. / 2. etc)
            elif re.match(r"^\d+[\.\:]", line):
                parts_kv = re.split(r"[\.\:]\s*", line, 1)
                num  = parts_kv[0]
                rest = parts_kv[1] if len(parts_kv) > 1 else ""
                lines_html += (
                    f'<div style="display:flex;gap:6px;margin-bottom:5px;">'
                    f'<span style="font-size:10.5px;font-weight:700;color:{color};'
                    f'min-width:18px;flex-shrink:0;">{escape_html_text(num)}.</span>'
                    f'<span style="font-size:11.5px;color:#374151;">{escape_html_text(rest)}</span>'
                    f'</div>'
                )
            else:
                lines_html += (
                    f'<div style="font-size:11.5px;color:#374151;margin-bottom:4px;">'
                    f'{escape_html_text(line)}</div>'
                )

        parts.append(
            f'<div style="margin-bottom:10px;">'
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;">'
            f'<div style="font-size:10px;font-weight:700;color:{color};'
            f'text-transform:uppercase;letter-spacing:.07em;">🧠 {thought_label}</div>'
            f'</div>'
            f'<div style="font-size:10.5px;color:#64748b;font-style:italic;margin-bottom:6px;">'
            f'{thought_note}</div>'
            f'<div style="border-left:3px solid {color};background:{color}0a;'
            f'border-radius:0 6px 6px 0;padding:9px 12px;line-height:1.6;">'
            f'{lines_html}</div></div>'
        )
    elif exp_id in (2, 4):
        # These experiments are supposed to have reasoning — explain why it's missing
        parts.append(
            f'<div style="background:#fef9c3;border:1px solid #fde047;border-radius:6px;'
            f'padding:8px 10px;margin-bottom:8px;">'
            f'<div style="font-size:10.5px;color:#92400e;">'
            f'⚠️ No reasoning trace captured. The model may not have produced a '
            f'"Thought:" block this time — this can happen with smaller models that '
            f'skip the reasoning step. Try upgrading to llama-3.3-70b-instruct.</div>'
            f'</div>'
        )

    # ── Raw tool arguments ────────────────────────────────────────────────
    if parsed["tool_name"] and parsed["tool_args_str"]:
        arg_rows = ""
        for part in re.split(r',\s*(?=\w+\s*=)', parsed["tool_args_str"]):
            kv = part.split("=", 1)
            if len(kv) == 2:
                k, v = kv[0].strip(), kv[1].strip().strip('"\'')
                arg_rows += (
                    f'<div style="display:flex;gap:8px;align-items:baseline;margin-top:3px;">'
                    f'<span style="font-size:10px;color:{color};min-width:84px;'
                    f'font-family:monospace;">{escape_html_text(k)}</span>'
                    f'<span style="font-size:10.5px;color:#374151;background:#f9fafb;'
                    f'border:1px solid #e5e7eb;border-radius:3px;padding:1px 6px;'
                    f'font-family:monospace;">"{escape_html_text(v)}"</span></div>'
                )
        parts.append(
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
            f'border-radius:6px;padding:8px 10px;">'
            f'<div style="font-size:10px;font-weight:700;color:#64748b;'
            f'text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px;">'
            f'📎 Raw tool arguments</div>'
            f'{arg_rows}</div>'
        )

    return "".join(parts)

def _error_card_html(msg: str, etype: str, exp_id: int, latency: float) -> str:
    if etype == "max_steps":
        icon, title = "⏱️", "Agent reached max steps"
        body = ("The model exhausted its step budget before calling a tool. "
                "Upgrade to meta/llama-3.3-70b-instruct in agents.py.")
    elif etype == "schema":
        icon, title = "📋", "Wrong tool arguments"
        body = f"Model called a tool with missing/incorrect parameters: {escape_html_text(msg)}"
    else:
        icon, title = "❌", "Agent error"
        body = escape_html_text(msg) if msg else "An unknown error occurred."
    return (
        f'<div style="background:#fef2f2;border:1.5px solid #fecaca;border-radius:8px;'
        f'padding:11px 13px;margin:0 0 6px;">'
        f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:5px;">'
        f'<span style="font-size:17px;">{icon}</span>'
        f'<span style="font-size:13px;font-weight:700;color:#b91c1c;">{title}</span>'
        f'<span style="margin-left:auto;font-size:10.5px;color:#9ca3af;background:#f3f4f6;'
        f'border-radius:99px;padding:2px 7px;">{latency:.1f}s</span>'
        f'</div>'
        f'<div style="font-size:11.5px;color:#7f1d1d;line-height:1.5;">{body}</div></div>'
    )

def render_response(raw: str, exp_id: int, latency: float, panel_idx: int) -> str:
    parsed = parse_dossier(raw)
    color  = EXPERIMENTS[exp_id]["color"]
    if parsed["error"]:
        return _error_card_html(parsed["error"], parsed["error_type"], exp_id, latency)
    if not parsed["tool_name"]:
        return (f'<div style="background:#fef9c3;border:1px solid #fde047;border-radius:8px;'
                f'padding:10px 12px;font-size:12px;color:#854d0e;margin:0 0 6px;">'
                f'⚠️ No tool was called — agent may have reached max steps.</div>')
    html = _response_card_html(parsed["tool_name"], parsed["tool_args_str"], color, latency, panel_idx)
    html += _details_html(parsed, color, exp_id)
    html += "</div>"   # close the collapsible div
    return html

# ── Page ───────────────────────────────────────────────────────────────────
@ui.page("/")
async def index():
    global _agents

    # ── Per-user persistent storage ──────────────────────────────────────
    app.storage.user.setdefault("experiment_chat_histories", {str(k): [] for k in EXPERIMENTS})
    app.storage.user.setdefault("is_processing", False)
    experiment_chat_histories: dict = app.storage.user["experiment_chat_histories"]

    # ── Global styles ─────────────────────────────────────────────────────
    ui.add_head_html("""
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap" rel="stylesheet">
    <style>
      *,*::before,*::after{box-sizing:border-box}
      body{background:#f1f5f9;font-family:'DM Sans',sans-serif;margin:0}
      .nicegui-content{padding:0!important}
      ::-webkit-scrollbar{width:4px}
      ::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:4px}
      .panel-card{background:#fff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.07);
        display:flex;flex-direction:column;min-width:0;overflow:hidden}
      .bubble-user{background:#1e293b;color:#f8fafc;border-radius:14px 14px 3px 14px;
        padding:8px 13px;font-size:13px;max-width:85%;line-height:1.5;word-break:break-word}
      @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
      .dot{animation:pulse 1.2s ease-in-out infinite}
      @keyframes slideDown{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:none}}
      .cbanner{animation:slideDown .3s ease}
      .tech-tag{font-size:9.5px;font-weight:700;letter-spacing:.08em;padding:2px 7px;
        border-radius:99px;font-family:monospace}
      /* Floating End Chat button */
      #end-chat-fab{
        position:fixed;bottom:82px;right:24px;z-index:9999;
        background:#ef4444;color:#fff;border:none;border-radius:10px;
        padding:9px 16px;font-size:13px;font-weight:600;cursor:pointer;
        box-shadow:0 4px 12px rgba(239,68,68,.4);font-family:'DM Sans',sans-serif;
        display:flex;align-items:center;gap:6px;
      }
      #end-chat-fab:hover{background:#dc2626}
    </style>""")

    # ── Header ────────────────────────────────────────────────────────────
    with ui.element("div").style(
        "background:#1e293b;padding:13px 20px;display:flex;align-items:center;"
        "gap:12px;border-bottom:1px solid #334155;flex-wrap:wrap;"
    ):
        ui.html('<span style="font-size:19px;">🖥️</span>')
        with ui.element("div"):
            ui.html('<div style="font-family:\'DM Serif Display\',serif;font-size:17px;color:#f8fafc;">'
                    'IT Helpdesk Agent Benchmark</div>')
            ui.html('<div style="font-size:10.5px;color:#94a3b8;margin-top:1px;">'
                    'Same query · Four prompting strategies · Watch them reason differently</div>')
        ui.element("div").style("flex:1")
        for eid, meta in EXPERIMENTS.items():
            ui.html(f'<div style="display:flex;align-items:center;gap:4px;">'
                    f'<span style="width:7px;height:7px;border-radius:50%;background:{meta["color"]};'
                    f'display:inline-block;"></span>'
                    f'<span style="font-size:10.5px;color:#cbd5e1;">Exp {eid}</span></div>')

    # ── Comparison banner ─────────────────────────────────────────────────
    comparison_row = ui.element("div").style("padding:0 14px;")
    comparison_row.set_visibility(False)

    # ── Four panels ───────────────────────────────────────────────────────
    experiment_ui_elements: dict[int, dict] = {}

    with ui.element("div").style(
        "display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;"
        "padding:9px 14px;height:calc(100vh - 164px);"
    ):
        for exp_id, meta in EXPERIMENTS.items():
            with ui.element("div").classes("panel-card"):
                # Header
                with ui.element("div").style("padding:10px 12px 8px;border-bottom:1px solid #f1f5f9;"):
                    ui.html(
                        f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:4px;">'
                        f'<span style="font-size:14px;">{meta["icon"]}</span>'
                        f'<span style="font-size:12.5px;font-weight:700;color:#1e293b;">'
                        f'Exp {exp_id}: {meta["short"]}</span>'
                        f'<span class="tech-tag" style="background:{meta["bg"]};color:{meta["color"]};'
                        f'border:1px solid {meta["border"]};margin-left:auto;">'
                        f'{meta["tag"]}</span></div>'
                    )
                    ui.html(
                        f'<div style="font-size:10.5px;color:#64748b;line-height:1.5;'
                        f'border-left:2px solid {meta["color"]}60;padding-left:6px;">'
                        f'{meta["pitch"]}</div>'
                    )

                # Scrollable body
                with ui.scroll_area().style("flex:1;background:#fafafa;") as scroll:
                    with ui.element("div").style(
                        "padding:10px;display:flex;flex-direction:column;gap:7px;"
                    ) as container:
                        stored = experiment_chat_histories.get(str(exp_id), [])
                        if stored:
                            for i, msg in enumerate(stored):
                                if msg["role"] == "user":
                                    ui.html(f'<div style="display:flex;justify-content:flex-end;">'
                                            f'<div class="bubble-user">{escape_html_text(msg["text"])}</div></div>')
                                else:
                                    raw_val = msg.get("raw", msg.get("text",""))
                                    rendered = render_response(raw_val, exp_id, msg.get("latency",0), i)
                                    ui.html(f'<div>{rendered}</div>')
                        else:
                            ui.html('<div style="text-align:center;padding:18px 0;">'
                                    '<div style="font-size:24px;margin-bottom:6px;">💬</div>'
                                    '<div style="font-size:11px;color:#94a3b8;">Send a query to see this agent respond</div>'
                                    '</div>')

                        experiment_ui_elements[exp_id] = {"container": container, "scroll": scroll}

    # ── Input bar ─────────────────────────────────────────────────────────
    with ui.element("div").style("background:#fff;border-top:1px solid #e2e8f0;padding:12px 18px;"):
        with ui.element("div").style(
            "display:flex;align-items:center;gap:9px;max-width:840px;margin:0 auto;"
        ):
            query_input = (
                ui.input(placeholder="Ask an IT question — e.g. 'I forgot my password and I'm locked out'")
                .style("flex:1;font-size:13px;")
                .props("outlined dense clearable")
            )
            send_btn = (
                ui.button("Send to all 4", icon="send")
                .style("background:#1e293b;color:#f8fafc;border-radius:8px;"
                       "font-size:13px;font-weight:600;padding:7px 14px;white-space:nowrap;")
                .props("no-caps unelevated")
            )
            ui.button(icon="delete_sweep", on_click=lambda: None).props("flat round dense") \
              .style("color:#94a3b8;").tooltip("Clear all (use End Chat button →)")



    # ── Helper functions ──────────────────────────────────────────────────
    msg_counter = [0]  # mutable counter for unique collapsible IDs

    def _add_user_bubble(exp_id: int, query: str):
        refs   = experiment_ui_elements[exp_id]
        stored = experiment_chat_histories[str(exp_id)]
        if not stored:
            refs["container"].clear()
        stored.append({"role": "user", "text": query})
        with refs["container"]:
            ui.html(f'<div style="display:flex;justify-content:flex-end;">'
                    f'<div class="bubble-user">{escape_html_text(query)}</div></div>')
        refs["scroll"].scroll_to(percent=1.0, duration=0.3)

    def _add_thinking(exp_id: int):
        c = EXPERIMENTS[exp_id]["color"]
        refs = experiment_ui_elements[exp_id]
        with refs["container"]:
            el = ui.html(
                f'<div style="display:flex;align-items:center;gap:5px;padding:3px 0;">'
                f'<span class="dot" style="width:6px;height:6px;border-radius:50%;background:{c};display:inline-block;"></span>'
                f'<span class="dot" style="width:6px;height:6px;border-radius:50%;background:{c};display:inline-block;animation-delay:.2s;"></span>'
                f'<span class="dot" style="width:6px;height:6px;border-radius:50%;background:{c};display:inline-block;animation-delay:.4s;"></span>'
                f'<span style="font-size:10.5px;color:#94a3b8;margin-left:3px;">{EXPERIMENTS[exp_id]["short"]} thinking…</span>'
                f'</div>'
            )
        refs["scroll"].scroll_to(percent=1.0, duration=0.2)
        return el

    AGENT_MESSAGES = {
        "create_ticket": "I have created a support ticket for this issue. The team will look into it shortly.",
        "escalate_ticket": "I have escalated this issue to the appropriate high-priority queue.",
        "lookup_knowledge_base": "I searched the knowledge base to find the relevant troubleshooting steps.",
        "reset_password": "I have initiated the password reset process for the account.",
        "get_user_info": "I've retrieved the requested user account details from the directory.",
        "lookup_user_account": "I looked up the user's account status in our system.",
        "check_system_status": "I checked our system monitors to see if there is an ongoing outage.",
        "schedule_maintenance": "I have scheduled the requested maintenance/hardware swap.",
        "process_refund": "I have processed the refund request.",
        "store_resolved_ticket": "I've successfully closed and archived this ticket.",
        "save_ticket_to_long_term_memory": "I've saved the resolution details to the user's history.",
        "get_user_long_term_memory": "I retrieved the user's long-term history.",
        "get_customer_history": "I checked the user's previous IT contact history."
    }

    def _render_agent_response(container, parsed: dict, exp_id: int, latency: float):
        # 1. Determine the conversational AI message based on the tool
        tool = parsed.get("tool_name")
        if parsed.get("error"):
            ai_message = "I encountered an error while processing this request."
        elif tool:
            # Fallback to a generic message if the tool isn't in our dictionary
            ai_message = AGENT_MESSAGES.get(tool, f"I have executed the `{tool}` tool to help with this.")
        else:
            ai_message = "I analyzed the request but couldn't determine the correct action."

        # Convert raw markdown to HTML
        # We assume 'raw' text is accessible or already parsed for HTML
        # Using the simplified dossier view if tool exists
        with container:
            with ui.column().classes('bubble bubble-agent').style('gap: 8px; padding: 12px;'):
                ui.html(f'<div style="font-size: 0.95em; color: #1e293b;">{ai_message}</div>')
                with ui.expansion('Show Reasoning & Details').classes('w-full').props('dense expand-separator').style('color: #6366f1; font-weight: 500; font-size: 0.9em;'):
                    color = EXPERIMENTS[exp_id]["color"]
                    details = _details_html(parsed, color, exp_id)
                    ui.html(details).classes('text-xs overflow-x-auto').style('background: #f8fafc; padding: 10px; border-radius: 6px; border: 1px solid #e2e8f0; color: #334155; margin-top: 4px;')

    def _add_agent_msg(exp_id: int, raw: str, latency: float):
        refs   = experiment_ui_elements[exp_id]
        stored = experiment_chat_histories[str(exp_id)]
        stored.append({"role": "agent", "raw": raw, "latency": latency})
        parsed = parse_dossier(raw)
        _render_agent_response(refs["container"], parsed, exp_id, latency)
        refs["scroll"].scroll_to(percent=1.0, duration=0.3)

    def _show_comparison(results: list[dict]):
        comparison_row.set_visibility(True)
        comparison_row.clear()
        with comparison_row:
            tools    = [r["tool"] for r in results if r["tool"]]
            all_same = len(set(tools)) == 1 if tools else False
            with ui.element("div").classes("cbanner").style(
                "background:#fff;border:1px solid #e2e8f0;border-radius:9px;"
                "padding:10px 14px;margin:6px 0 0;display:flex;align-items:center;gap:10px;flex-wrap:wrap;"
            ):
                ui.html('<div style="font-size:10px;font-weight:700;color:#475569;'
                        'letter-spacing:.06em;text-transform:uppercase;white-space:nowrap;">'
                        '⚡ Tool decision comparison</div>')
                for r in sorted(results, key=lambda x: x["exp_id"]):
                    meta  = EXPERIMENTS[r["exp_id"]]
                    tool  = r["tool"] or "—"
                    emoji = TOOL_EMOJI.get(tool, "❓") if tool != "—" else "❓"
                    border = ("border:1.5px solid #fca5a5;background:#fef2f2;"
                              if r.get("is_error") else
                              f"border:1.5px solid {meta['color']};")
                    ui.html(
                        f'<div style="display:flex;align-items:center;gap:5px;'
                        f'background:{meta["bg"]};{border}border-radius:6px;padding:4px 8px;">'
                        f'<span style="font-size:10.5px;font-weight:700;color:{meta["color"]};">Exp {r["exp_id"]}</span>'
                        f'<span style="font-size:13px;">{emoji}</span>'
                        f'<span style="font-size:11px;font-weight:600;color:#1e293b;font-family:monospace;">{escape_html_text(tool)}</span>'
                        f'<span style="font-size:9.5px;color:#9ca3af;">{r["latency"]:.1f}s</span>'
                        f'</div>'
                    )
                verdict = ('<span style="color:#059669;font-weight:600;">✓ All agents agree</span>'
                           if all_same and tools else
                           '<span style="color:#dc2626;font-weight:600;">⚡ Agents disagree — check reasoning</span>')
                ui.html(f'<div style="font-size:11px;margin-left:auto;">{verdict}</div>')


    def _do_end_chat():
        experiment_chat_histories.update({str(k): [] for k in EXPERIMENTS})
        app.storage.user["is_processing"] = False
        for exp_id in EXPERIMENTS:
            refs = experiment_ui_elements[exp_id]
            refs["container"].clear()
            with refs["container"]:
                ui.html('<div style="text-align:center;padding:18px 0;">'
                        '<div style="font-size:24px;margin-bottom:6px;">💬</div>'
                        '<div style="font-size:11px;color:#94a3b8;">Send a query to see this agent respond</div>'
                        '</div>')
        comparison_row.set_visibility(False)
        comparison_row.clear()
        send_btn.enable()
        query_input.enable()
        ui.notify("Chat history cleared.", type="info", timeout=2)

    def _shutdown_app():
        ui.notify("Shutting down server...", type="warning")
        app.shutdown() # This acts like Ctrl+C

    ui.button("End Session", icon="power_settings_new", on_click=_shutdown_app).style(
        "position: fixed; bottom: 24px; right: 24px; z-index: 50; "
        "background: #334155; color: #f8fafc; border-radius: 99px; "
        "padding: 10px 20px; font-weight: 600; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"
    ).props("no-caps")


    # Also wire the clear button in the input bar
    for btn in [b for b in ui.context.client.elements.values()
                if getattr(b, '_props', {}).get('icon') == 'delete_sweep']:
        btn.on_click(_do_end_chat)
        break

    # ── Send handler ─────────────────────────────────────────────────────
    # Purposes: The 'Async Orchestrator'. This function triggers all 4 agent experiments simultaneously.
    async def handle_send():
        global _agents
        # Purposes: Prevents 'Double Sending' if a request is already in progress.
        if app.storage.user.get("is_processing"):
            return
        query = (query_input.value or "").strip()
        # Purposes: Guard clause for empty queries.
        if not query:
            ui.notify("Please type a question first.", type="warning"); return

        # Purposes: Lazy-loading logic. Only loads agents from disk the first time you click Send.
        if _agents is None:
            ui.notify("Loading agents…", timeout=2)
            try:
                # Purposes: Offloads the slow 'import' calls to a background thread to keep the UI smooth.
                _agents = await asyncio.get_event_loop().run_in_executor(
                    _executor, load_all_experiment_agents
                )
            except Exception as exc:
                ui.notify(f"Failed to load agents: {exc}", type="negative", timeout=0)
                return

        app.storage.user["is_processing"] = True
        query_input.disable(); send_btn.disable()
        query_input.set_value("")
        comparison_row.set_visibility(False); comparison_row.clear()

        for exp_id in EXPERIMENTS:
            _add_user_bubble(exp_id, query)
        thinking = {exp_id: _add_thinking(exp_id) for exp_id in EXPERIMENTS}

        # Purposes: The 'Parallel Engine'. 
        # This nested function defines how a SINGLE experiment is run and recorded.
        async def _run(exp_id: int) -> dict:
            loop = asyncio.get_event_loop()
            t0 = time.perf_counter() # Purposes: Starts the stopwatch for this experiment.
            try:
                # Purposes: The core AI call. Runs in a background thread to prevent UI freezing.
                raw = await loop.run_in_executor(_executor, _agents[exp_id], query)
            except Exception as exc:
                raw = f"❌ Error:\n`{str(exc) or repr(exc)}`"
            latency = time.perf_counter() - t0 # Purposes: Stops the stopwatch.
            thinking[exp_id].delete() # Purposes: Removes the 'thinking...' dots.
            _add_agent_msg(exp_id, str(raw), latency) # Purposes: Renders the final dossier card.
            parsed = parse_dossier(str(raw)) # Purposes: Converts text result into data for the summary banner.
            return {"exp_id": exp_id, "tool": parsed["tool_name"],
                    "latency": latency, "is_error": bool(parsed["error"])}

        # Purposes: The 'Racing Logic'. runs all 4 _run() functions at once.
        results = sorted(
            list(await asyncio.gather(*[_run(eid) for eid in EXPERIMENTS])),
            key=lambda x: x["exp_id"]
        )
        # Purposes: Shows the '⚡ Tool decision comparison' banner at the top.
        _show_comparison(results)
        app.storage.user["is_processing"] = False
        query_input.enable(); send_btn.enable()
        query_input.run_method("focus")

    send_btn.on_click(handle_send)
    query_input.on("keydown.enter", handle_send)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ in ("__main__", "__mp_main__"):
    api_key = (os.environ.get("NVIDIA_API_KEY")
               or os.environ.get("OPENAI_API_KEY") or "")
    if not api_key or "your" in api_key:
        print("\n[ERROR] API key missing — add NVIDIA_API_KEY or OPENAI_API_KEY to .env\n")
        sys.exit(1)

    print("\n  IT Helpdesk Agent Benchmark UI")
    print("  http://localhost:8000")
    print("  Chat history: .nicegui/storage-user-<UUID>.json per user")
    print("  End Chat button: floating bottom-right of the page\n")

    ui.run(
        title="IT Helpdesk Agent Benchmark",
        favicon="🖥️", port=8000, reload=False, dark=False,
        storage_secret="itbenchmark_storage_secret_change_me_in_production",
    )