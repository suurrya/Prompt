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
import markdown
import asyncio, html as _html, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from nicegui import app, ui
import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))

_executor = ThreadPoolExecutor(max_workers=4)
_agents: dict[int, object] | None = None

# ── Experiment metadata ────────────────────────────────────────────────────
EXPERIMENTS = {
    1: {"short":"Static Few-Shot",       "color":"#6366f1","bg":"#eef2ff","border":"#c7d2fe","icon":"🗂️",
        "tag":"STATIC · NO REASONING",
        "pitch":"Fixed examples baked in. The model pattern-matches against a hardcoded query→tool list. "
                "Fast and predictable, but blind to queries it has never seen."},
    2: {"short":"Static Chain-of-Thought","color":"#0ea5e9","bg":"#e0f2fe","border":"#bae6fd","icon":"🧠",
        "tag":"STATIC · WITH REASONING",
        "pitch":"Same fixed prompt as Exp 1, but every example shows a Thought: step before the tool call. "
                "The model is taught to reason explicitly before acting."},
    3: {"short":"Dynamic Few-Shot",       "color":"#10b981","bg":"#d1fae5","border":"#a7f3d0","icon":"🔍",
        "tag":"DYNAMIC · NO REASONING",
        "pitch":"At call-time, TF-IDF picks the top-4 examples closest to the user's query. "
                "Prompt changes every call — more relevant examples, fewer edge-case errors."},
    4: {"short":"Dynamic Chain-of-Thought","color":"#f59e0b","bg":"#fef3c7","border":"#fde68a","icon":"✨",
        "tag":"DYNAMIC · WITH REASONING",
        "pitch":"Combines dynamic selection (Exp 3) with CoT traces (Exp 2). "
                "Most relevant examples AND step-by-step reasoning — highest expected accuracy."},
}

TOOL_EMOJI = {
    "lookup_knowledge_base":"📖","create_ticket":"🎫","escalate_ticket":"🚨",
    "reset_password":"🔑","get_user_info":"👤","lookup_user_account":"💳",
    "check_system_status":"📡","schedule_maintenance":"🔧","process_refund":"💰",
    "store_resolved_ticket":"💾","save_ticket_to_long_term_memory":"🗄️",
    "get_user_long_term_memory":"🕐","get_customer_history":"📋",
}

TOOL_RESPONSES = {
    "reset_password":   lambda a: f"I've initiated a password reset for {a.get('user_email','the account')} via {a.get('method','email')}.",
    "lookup_knowledge_base": lambda a: "I've searched the knowledge base: \"" + a.get('query','your issue') + "\".",
    "create_ticket":    lambda a: f"I've created a {a.get('priority','')} {a.get('category','')} ticket: \"{a.get('summary','Issue reported')}\".",
    "escalate_ticket":  lambda a: f"I've escalated ticket {a.get('ticket_id','')} to {a.get('escalate_to','the specialist team')}.",
    "check_system_status": lambda a: f"I've checked the live status of the {a.get('service_name','service')} system.",
    "get_user_info":    lambda a: f"I've retrieved directory details for {a.get('user_email','the user')}.",
    "lookup_user_account": lambda a: f"I've looked up the subscription status for {a.get('email','the user')}.",
    "schedule_maintenance": lambda a: f"I've booked a {a.get('maintenance_type','maintenance')} slot for {a.get('asset_id','the device')}.",
    "process_refund":   lambda a: f"I've initiated a refund for reservation {a.get('reservation_id','')}.",
    "store_resolved_ticket": lambda a: "I've saved a resolution summary to this user's history.",
    "save_ticket_to_long_term_memory": lambda a: "I've archived the full ticket outcome for future reference.",
    "get_user_long_term_memory": lambda a: f"I've retrieved the full history for user {a.get('user_id','')}.",
    "get_customer_history": lambda a: f"I've pulled up past contact history for user {a.get('user_id','')}.",
}

def _human_response(tool_name: str, args_str: str) -> str:
    """Convert a tool call into a clean, human-readable sentence."""
    args: dict = {}
    if args_str:
        for part in re.split(r',\s*(?=\w+\s*=)', args_str):
            kv = part.split("=", 1)
            if len(kv) == 2:
                args[kv[0].strip()] = kv[1].strip().strip('"\'')
    fn = TOOL_RESPONSES.get(tool_name)
    return fn(args) if fn else f"I've called {tool_name} to handle your request."

# ── Agent loading ──────────────────────────────────────────────────────────
def _load_agents() -> dict[int, object]:
    import importlib
    result = {}
    for exp_id, path in {
        1:"project_1_few_shot.agents", 2:"project_2_chain_of_thought.agents",
        3:"project_3_dynamic_few_shot.agents", 4:"project_4_dynamic_cot.agents",
    }.items():
        mod = importlib.import_module(path)
        result[exp_id] = mod.ITHelpdeskAgent(verbose=False)
    return result

# ── Dossier parser ─────────────────────────────────────────────────────────
def _esc(s: str) -> str:
    return _html.escape(str(s))

def parse_dossier(text: str) -> dict:
    out = {"examples":[],"thought":"","tool_name":"","tool_args_str":"",
           "error":"","error_type":""}

    if "❌" in text or text.strip().startswith("⚠️"):
        m = re.search(r"❌\s*Error[:\s]*\n?`?([^`]*)`?", text, re.DOTALL)
        msg = m.group(1).strip() if m else text.strip()
        if not msg or msg in ('""',"''",""):
            out["error"] = "Agent reached maximum steps without completing the task."
            out["error_type"] = "max_steps"
        elif "max step" in msg.lower():
            out["error"] = msg; out["error_type"] = "max_steps"
        elif "not in the tool" in msg.lower() or "schema" in msg.lower():
            out["error"] = msg; out["error_type"] = "schema"
        else:
            out["error"] = msg or repr(text); out["error_type"] = "general"
        return out

    ex = re.search(r"(?:Examples selected|CoT Examples selected)[^\n]*:\n(.*?)\n\n", text, re.DOTALL)
    if ex:
        for line in ex.group(1).split("\n"):
            line = line.strip().lstrip("- ").strip("`").rstrip(".").strip()
            if line: out["examples"].append(line)

    tm = re.search(r"```markdown\n(.*?)```", text, re.DOTALL)
    if tm:
        out["thought"] = tm.group(1).strip()
    else:
        it = re.search(r"Thought:\s*(.+?)(?=\n\n|\Z)", text, re.DOTALL)
        if it: out["thought"] = it.group(1).strip()

    dm = re.search(r"###\s*Decision\s*\n(.*?)(?=###|\Z)", text, re.DOTALL|re.IGNORECASE)
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
    answer = _esc(_human_response(tool_name, args_str))
    cid    = f"details-{panel_idx}"
    return (
        f'<div style="background:{color}12;border:1.5px solid {color}50;border-radius:10px;'
        f'padding:11px 13px;margin:0 0 6px;">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;">'
        f'<span style="font-size:20px;">{emoji}</span>'
        f'<span style="font-size:13.5px;font-weight:700;color:{color};font-family:monospace;">'
        f'{_esc(tool_name)}</span>'
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

def _details_html(parsed: dict, color: str) -> str:
    """Collapsible technical section — examples, reasoning, raw args."""
    parts = []

    if parsed["examples"]:
        chips = "".join(
            f'<span style="display:inline-block;background:#fff;border:1px solid {color}40;'
            f'border-left:3px solid {color};color:#374151;font-size:10.5px;padding:2px 7px;'
            f'border-radius:4px;margin:2px 3px 2px 0;font-family:monospace;">'
            f'{_esc(e[:62])}{"…" if len(e)>62 else ""}</span>'
            for e in parsed["examples"]
        )
        parts.append(
            f'<div style="margin-bottom:8px;">'
            f'<div style="font-size:9.5px;font-weight:700;color:{color};text-transform:uppercase;'
            f'letter-spacing:.07em;margin-bottom:4px;">🔍 TF-IDF retrieved examples</div>'
            f'<div>{chips}</div></div>'
        )

    if parsed["thought"]:
        lines = ""
        for line in parsed["thought"].split("\n"):
            line = line.strip()
            if not line: continue
            if line.startswith(("•","-","*")):
                lines += (f'<div style="display:flex;gap:5px;margin-bottom:3px;">'
                          f'<span style="color:{color};flex-shrink:0;">›</span>'
                          f'<span>{_esc(line.lstrip("•-* "))}</span></div>')
            elif re.match(r"^\d+[\.\:]", line):
                num, rest = re.split(r"[\.\:]\s*", line, 1) if re.search(r"[\.\:]", line) else (line, "")
                lines += (f'<div style="display:flex;gap:5px;margin-bottom:3px;">'
                          f'<span style="color:{color};font-weight:700;flex-shrink:0;">{_esc(num)}.</span>'
                          f'<span>{_esc(rest)}</span></div>')
            else:
                lines += f'<div style="margin-bottom:3px;">{_esc(line)}</div>'
        parts.append(
            f'<div style="border-left:3px solid {color};background:{color}0d;'
            f'border-radius:0 6px 6px 0;padding:8px 11px;margin-bottom:8px;">'
            f'<div style="font-size:9.5px;font-weight:700;color:{color};text-transform:uppercase;'
            f'letter-spacing:.07em;margin-bottom:5px;">🧠 Chain of thought</div>'
            f'<div style="font-size:11.5px;color:#374151;line-height:1.6;font-style:italic;">'
            f'{lines}</div></div>'
        )

    if parsed["tool_name"] and parsed["tool_args_str"]:
        arg_rows = ""
        for part in re.split(r',\s*(?=\w+\s*=)', parsed["tool_args_str"]):
            kv = part.split("=", 1)
            if len(kv) == 2:
                k, v = kv[0].strip(), kv[1].strip().strip('"\'')
                arg_rows += (
                    f'<div style="display:flex;gap:6px;align-items:baseline;margin-top:2px;">'
                    f'<span style="font-size:10px;color:{color};min-width:80px;font-family:monospace;">{_esc(k)}</span>'
                    f'<span style="font-size:10.5px;color:#374151;background:#f9fafb;'
                    f'border:1px solid #e5e7eb;border-radius:3px;padding:1px 5px;font-family:monospace;">'
                    f'"{_esc(v)}"</span></div>'
                )
        parts.append(
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:8px 10px;">'
            f'<div style="font-size:9.5px;font-weight:700;color:#64748b;text-transform:uppercase;'
            f'letter-spacing:.07em;margin-bottom:4px;">📎 Raw tool arguments</div>'
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
        body = f"Model called a tool with missing/incorrect parameters: {_esc(msg)}"
    else:
        icon, title = "❌", "Agent error"
        body = _esc(msg) if msg else "An unknown error occurred."
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
    html += _details_html(parsed, color)
    html += "</div>"   # close the collapsible div
    return html

# ── Page ───────────────────────────────────────────────────────────────────
@ui.page("/")
async def index():
    global _agents

    # ── Per-user persistent storage ──────────────────────────────────────
    app.storage.user.setdefault("panel_messages", {str(k): [] for k in EXPERIMENTS})
    app.storage.user.setdefault("is_processing", False)
    panel_messages: dict = app.storage.user["panel_messages"]

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
    panel_refs: dict[int, dict] = {}

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
                        stored = panel_messages.get(str(exp_id), [])
                        if stored:
                            for i, msg in enumerate(stored):
                                if msg["role"] == "user":
                                    ui.html(f'<div style="display:flex;justify-content:flex-end;">'
                                            f'<div class="bubble-user">{_esc(msg["text"])}</div></div>')
                                else:
                                    raw_val = msg.get("raw", msg.get("text",""))
                                    rendered = render_response(raw_val, exp_id, msg.get("latency",0), i)
                                    ui.html(f'<div>{rendered}</div>')
                        else:
                            ui.html('<div style="text-align:center;padding:18px 0;">'
                                    '<div style="font-size:24px;margin-bottom:6px;">💬</div>'
                                    '<div style="font-size:11px;color:#94a3b8;">Send a query to see this agent respond</div>'
                                    '</div>')

                        panel_refs[exp_id] = {"container": container, "scroll": scroll}

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

    # ── Floating End Chat button (bottom-right) ───────────────────────────
    ui.add_body_html("""
    <button id="end-chat-fab" onclick="document.dispatchEvent(new CustomEvent('end-chat'))">
      ✕ End Chat
    </button>
    <script>
      document.addEventListener('end-chat', function() {
        // Calls the NiceGUI server-side handler via a hidden element
        if (window.__endChatFn) window.__endChatFn();
      });
    </script>
    """)

    # ── Helper functions ──────────────────────────────────────────────────
    msg_counter = [0]  # mutable counter for unique collapsible IDs

    def _add_user_bubble(exp_id: int, query: str):
        refs   = panel_refs[exp_id]
        stored = panel_messages[str(exp_id)]
        if not stored:
            refs["container"].clear()
        stored.append({"role": "user", "text": query})
        with refs["container"]:
            ui.html(f'<div style="display:flex;justify-content:flex-end;">'
                    f'<div class="bubble-user">{_esc(query)}</div></div>')
        refs["scroll"].scroll_to(percent=1.0, duration=0.3)

    def _add_thinking(exp_id: int):
        c = EXPERIMENTS[exp_id]["color"]
        refs = panel_refs[exp_id]
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

    def _add_response(exp_id: int, raw: str, latency: float):
        container = panel_refs[exp_id]["container"]
        scroll = panel_refs[exp_id]["scroll"]
        parsed = parse_dossier(raw)
        
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
        html_content = markdown.markdown(raw, extensions=['fenced_code', 'tables'])
        
        with container:
            # 2. Use Native NiceGUI components instead of raw HTML for the toggle
            with ui.column().classes('bubble bubble-agent').style('gap: 8px; padding: 12px;'):
                # The conversational AI message
                ui.html(f'<div style="font-size: 0.95em; color: #1e293b;">{ai_message}</div>')
                
                # The native dropdown toggle that actually works
                with ui.expansion('Show Reasoning & Details').classes('w-full').props('dense expand-separator').style('color: #6366f1; font-weight: 500; font-size: 0.9em;'):
                    ui.html(html_content).classes('text-xs overflow-x-auto').style('background: #f8fafc; padding: 10px; border-radius: 6px; border: 1px solid #e2e8f0; color: #334155; margin-top: 4px;')
                    
        # Auto-scroll to bottom
        scroll.scroll_to(percent=1.0, duration=0.3)

        _show_comparison(results)
        app.storage.user["is_processing"] = False
        query_input.enable(); send_btn.enable()
        query_input.run_method("focus")

    def _do_end_chat():
        panel_messages.update({str(k): [] for k in EXPERIMENTS})
        app.storage.user["is_processing"] = False
        for exp_id in EXPERIMENTS:
            refs = panel_refs[exp_id]
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

    ui.button("End Chat", icon="power_settings_new", on_click=_shutdown_app).style(
        "position: fixed; bottom: 24px; right: 24px; z-index: 50; "
        "background: #ef4444; color: #f8fafc; border-radius: 99px; "
        "padding: 10px 20px; font-weight: 600; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
    ).props("no-caps")

    # Wire end-chat FAB to server-side handler
    ui.add_body_html("<script>window.__endChatFn = () => {};</script>")
    ui.on("end-chat", _do_end_chat)
    # Also wire the clear button in the input bar
    for btn in [b for b in ui.context.client.elements.values()
                if getattr(b, '_props', {}).get('icon') == 'delete_sweep']:
        btn.on_click(_do_end_chat)
        break

    # ── Send handler ─────────────────────────────────────────────────────
    async def handle_send():
        global _agents
        if app.storage.user.get("is_processing"):
            return
        query = (query_input.value or "").strip()
        if not query:
            ui.notify("Please type a question first.", type="warning"); return

        if _agents is None:
            ui.notify("Loading agents…", timeout=2)
            try:
                _agents = await asyncio.get_event_loop().run_in_executor(
                    _executor, _load_agents
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

        async def _run(exp_id: int) -> dict:
            loop = asyncio.get_event_loop()
            t0 = time.perf_counter()
            try:
                raw = await loop.run_in_executor(_executor, _agents[exp_id], query)
            except Exception as exc:
                raw = f"❌ Error:\n`{str(exc) or repr(exc)}`"
            latency = time.perf_counter() - t0
            thinking[exp_id].delete()
            _add_response(exp_id, str(raw), latency)
            parsed = parse_dossier(str(raw))
            return {"exp_id": exp_id, "tool": parsed["tool_name"],
                    "latency": latency, "is_error": bool(parsed["error"])}

        results = sorted(
            list(await asyncio.gather(*[_run(eid) for eid in EXPERIMENTS])),
            key=lambda x: x["exp_id"]
        )
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