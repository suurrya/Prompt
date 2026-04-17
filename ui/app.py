"""
ui/app.py
=========
NiceGUI page definition and all inner handler closures.
Contains only the @ui.page("/") route and the functions that close over UI elements.

To start the server, run:  python ui/main.py
"""

from __future__ import annotations
import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# ── Ensure project root is on sys.path before any ui.* imports ────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
from nicegui import app, ui

# ── Bootstrap ─────────────────────────────────────────────────────────────────
from ui.config import ROOT, EXPERIMENTS, TOOL_EMOJI, AGENT_MESSAGES
from ui.parsing import escape_html_text, parse_dossier
from ui.html_builders import details_html
from ui.agent_loader import load_all_experiment_agents, load_email_asset_options

load_dotenv(os.path.join(ROOT, ".env"))

executor = ThreadPoolExecutor(max_workers=4)
agents: dict[int, object] | None = None


# ── Page ──────────────────────────────────────────────────────────────────────
@ui.page("/")
async def index():
    global agents

    # ── Per-user persistent storage ───────────────────────────────────────────
    app.storage.user.setdefault("experiment_chat_histories", {str(k): [] for k in EXPERIMENTS})
    app.storage.user["is_processing"] = False  # Always reset on page load (clears stale state from crashed sessions)
    experiment_chat_histories: dict = app.storage.user["experiment_chat_histories"]

    # ── Global CSS ────────────────────────────────────────────────────────────
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
    </style>""")

    # ── Header ────────────────────────────────────────────────────────────────
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

    # ── Comparison banner ─────────────────────────────────────────────────────
    comparison_row = ui.element("div").style("padding:0 14px;")
    comparison_row.set_visibility(False)

    # ── Step 1: Selection card (visible on load) ──────────────────────────────
    email_asset_options = load_email_asset_options()
    with ui.card().style(
        "width:100%;border-radius:0;box-shadow:none;"
        "border-bottom:1px solid #e2e8f0;padding:32px 40px;background:#fff;"
    ) as selection_card:
        ui.html(
            '<div style="font-size:20px;font-weight:700;color:#1e293b;margin-bottom:6px;">Welcome</div>'
            '<div style="font-size:13px;color:#64748b;margin-bottom:20px;">'
            'Select your user account and asset to begin the helpdesk session.</div>'
        )
        with ui.row().style("align-items:center;gap:12px;"):
            selection_dropdown = (
                ui.select(
                    options=["-- Select user/asset --"] + email_asset_options,
                    value="-- Select user/asset --",
                    label="User / Asset",
                )
                .style("width:380px;font-size:13px;")
                .props("outlined dense")
            )
            continue_btn = (
                ui.button("Continue", icon="arrow_forward")
                .props("no-caps unelevated")
                .style(
                    "background:#1e293b;color:#f8fafc;border-radius:8px;"
                    "font-size:13px;font-weight:600;padding:8px 18px;"
                )
            )
            continue_btn.disable()

    # ── Agent response renderer (defined before panels so restore path can use it) ──
    def render_agent_response(container, parsed: dict, exp_id: int, latency: float):
        tool = parsed.get("tool_name")
        if parsed.get("error_type") == "no_tool":
            ai_message = "I could not determine a tool to call for this request."
        elif parsed.get("error"):
            ai_message = "I encountered an error while processing this request."
        elif tool:
            ai_message = AGENT_MESSAGES.get(tool, f"I have executed the `{tool}` tool to help with this.")
        else:
            ai_message = "I analyzed the request but couldn't determine the correct action."
        with container:
            with ui.column().style("gap:8px;padding:12px;"):
                ui.html(f'<div style="font-size:13px;color:#1e293b;">{ai_message}</div>')
                with ui.expansion("Show Reasoning & Details").props("dense expand-separator").style(
                    f"color:{EXPERIMENTS[exp_id]['color']};font-weight:500;font-size:0.9em;"
                ):
                    color   = EXPERIMENTS[exp_id]["color"]
                    details = details_html(parsed, color, exp_id, agents)
                    ui.html(details).style(
                        "background:#f8fafc;padding:10px;border-radius:6px;"
                        "border:1px solid #e2e8f0;color:#334155;margin-top:4px;font-size:11px;"
                    )

    # ── Four panels grid (hidden on start) ───────────────────────────────────
    experiment_ui_elements: dict[int, dict] = {}

    with ui.element("div").style(
        "display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;"
        "padding:9px 14px;height:calc(100vh - 164px);"
    ) as panels_wrapper:
        for exp_id, meta in EXPERIMENTS.items():
            with ui.element("div").classes("panel-card"):
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
                with ui.scroll_area().style("flex:1;background:#fafafa;") as scroll:
                    with ui.element("div").style(
                        "padding:10px;display:flex;flex-direction:column;gap:7px;"
                    ) as container:
                        stored = experiment_chat_histories.get(str(exp_id), [])
                        if stored:
                            for msg in stored:
                                if msg["role"] == "user":
                                    ui.html(f'<div style="display:flex;justify-content:flex-end;">'
                                            f'<div class="bubble-user">{escape_html_text(msg["text"])}</div></div>')
                                else:
                                    raw_val = msg.get("raw", msg.get("text", ""))
                                    render_agent_response(
                                        container,
                                        parse_dossier(raw_val),
                                        exp_id,
                                        msg.get("latency", 0),
                                    )
                        else:
                            ui.html('<div style="text-align:center;padding:18px 0;">'
                                    '<div style="font-size:24px;margin-bottom:6px;">💬</div>'
                                    '<div style="font-size:11px;color:#94a3b8;">Send a query to see this agent respond</div>'
                                    '</div>')
                        experiment_ui_elements[exp_id] = {"container": container, "scroll": scroll}

    # ── Input bar (hidden on start) ───────────────────────────────────────────
    with ui.element("div").style(
        "background:#fff;border-top:1px solid #e2e8f0;padding:12px 18px;"
    ) as input_wrapper:
        with ui.element("div").style(
            "display:flex;align-items:center;gap:9px;max-width:840px;margin:0 auto;"
        ):
            query_input = (
                ui.input(placeholder="Ask an IT question")
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

    # ── Step 2: Hide main chat until Continue is clicked ──────────────────────
    panels_wrapper.set_visibility(False)
    input_wrapper.set_visibility(False)

    # ── Step 2: Selection card state management ───────────────────────────────
    def on_selection_change(e):
        if e.value and e.value != "-- Select user/asset --":
            continue_btn.enable()
        else:
            continue_btn.disable()

    def on_continue():
        val: str = selection_dropdown.value
        if not val or val == "-- Select user/asset --":
            return
        parts      = val.split("  ->  ")
        email      = parts[0].strip()
        asset_id   = parts[1].strip() if len(parts) > 1 else ""
        model_info = parts[2].strip() if len(parts) > 2 else asset_id
        user_name  = email.split("@")[0].replace(".", " ").title()

        app.storage.user["selected_email"] = email
        app.storage.user["selected_asset"] = asset_id
        app.storage.user["selected_name"]  = user_name
        app.storage.user["selected_model"] = model_info

        # Reset chat histories so the welcome message is cleared on the first send,
        # regardless of any leftover data from a previous browser session.
        experiment_chat_histories.update({str(k): [] for k in EXPERIMENTS})

        selection_card.set_visibility(False)
        panels_wrapper.set_visibility(True)
        input_wrapper.set_visibility(True)

        welcome = f"Hello {user_name}, what help do you need with your {model_info} (Asset #{asset_id})?"
        for eid in EXPERIMENTS:
            refs = experiment_ui_elements[eid]
            refs["container"].clear()
            with refs["container"]:
                ui.html(
                    f'<div style="text-align:center;padding:22px 12px;">'
                    f'<div style="font-size:26px;margin-bottom:8px;">👋</div>'
                    f'<div style="font-size:13px;font-weight:600;color:#1e293b;">'
                    f'{escape_html_text(welcome)}</div>'
                    f'</div>'
                )

    selection_dropdown.on_value_change(on_selection_change)
    continue_btn.on_click(on_continue)

    # ── Chat helper closures ──────────────────────────────────────────────────
    def add_user_bubble(exp_id: int, query: str):
        refs   = experiment_ui_elements[exp_id]
        stored = experiment_chat_histories[str(exp_id)]
        if not stored:
            refs["container"].clear()
        stored.append({"role": "user", "text": query})
        with refs["container"]:
            ui.html(f'<div style="display:flex;justify-content:flex-end;">'
                    f'<div class="bubble-user">{escape_html_text(query)}</div></div>')
        refs["scroll"].scroll_to(percent=1.0, duration=0.3)

    def add_thinking(exp_id: int):
        c    = EXPERIMENTS[exp_id]["color"]
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

    def add_agent_msg(exp_id: int, raw: str, latency: float):
        refs   = experiment_ui_elements[exp_id]
        stored = experiment_chat_histories[str(exp_id)]
        stored.append({"role": "agent", "raw": raw, "latency": latency})
        render_agent_response(refs["container"], parse_dossier(raw), exp_id, latency)
        refs["scroll"].scroll_to(percent=1.0, duration=0.3)

    def show_comparison(results: list[dict]):
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
                    meta   = EXPERIMENTS[r["exp_id"]]
                    tool   = r["tool"] or "—"
                    emoji  = TOOL_EMOJI.get(tool, "❓") if tool != "—" else "❓"
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

    def do_end_chat():
        experiment_chat_histories.update({str(k): [] for k in EXPERIMENTS})
        app.storage.user["is_processing"] = False
        try:
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
        except (ValueError, RuntimeError):
            pass

    def shutdown_app():
        ui.notify("Shutting down server...", type="warning")
        app.shutdown()

    # ── End Session button ────────────────────────────────────────────────────
    with ui.element("div").style(
        "position:fixed;bottom:24px;right:24px;z-index:50;"
        "display:flex;gap:12px;align-items:center;"
    ):
        ui.button("End Session", icon="power_settings_new", on_click=shutdown_app) \
            .props("no-caps color=negative") \
            .style("border-radius:99px;padding:10px 20px;font-weight:600;"
                   "box-shadow:0 4px 12px rgba(0,0,0,0.15);")

    # Wire the clear (delete_sweep) button in the input bar
    for btn in [b for b in ui.context.client.elements.values()
                if getattr(b, "_props", {}).get("icon") == "delete_sweep"]:
        btn.on_click(do_end_chat)
        break

    # ── Send handler ──────────────────────────────────────────────────────────
    def safe_set_not_processing():
        """Write is_processing=False, silently ignoring disconnected-session errors."""
        try:
            app.storage.user["is_processing"] = False
        except (AssertionError, KeyError, RuntimeError):
            pass

    async def handle_send():
        global agents
        try:
            await handle_send_inner()
        except (ValueError, RuntimeError, AssertionError):
            safe_set_not_processing()

    async def handle_send_inner():
        global agents
        if app.storage.user.get("is_processing"):
            return
        query = (query_input.value or "").strip()
        if not query:
            ui.notify("Please type a question first.", type="warning")
            return

        if query.strip().lower() == "run test_cases":
            from evaluation.test_cases import get_test_cases
            user_email = app.storage.user.get("selected_email", "")
            asset_id   = app.storage.user.get("selected_asset", "")
            test_cases = get_test_cases(user_email, asset_id)
            query_input.set_value("")
            ui.notify(f"Running {len(test_cases)} personalised test cases…", timeout=3)
            for tc in test_cases:
                query_input.set_value(tc["query"])
                await handle_send_inner()
            query_input.set_value("")
            return

        if agents is None:
            ui.notify("Loading agents…", timeout=2)
            try:
                agents = await asyncio.get_event_loop().run_in_executor(
                    executor, load_all_experiment_agents
                )
            except Exception as exc:
                ui.notify(f"Failed to load agents: {exc}", type="negative", timeout=0)
                return

        app.storage.user["is_processing"] = True
        query_input.disable()
        send_btn.disable()
        query_input.set_value("")
        comparison_row.set_visibility(False)
        comparison_row.clear()

        try:
            for exp_id in EXPERIMENTS:
                add_user_bubble(exp_id, query)
            thinking = {exp_id: add_thinking(exp_id) for exp_id in EXPERIMENTS}
        except (ValueError, RuntimeError, AssertionError):
            safe_set_not_processing()
            return

        async def run_experiment(exp_id: int) -> dict:
            loop = asyncio.get_event_loop()
            t0   = time.perf_counter()

            # Prepend user context so agents use real values instead of <email>/<id>
            email_val  = app.storage.user.get("selected_email", "")
            asset_val  = app.storage.user.get("selected_asset", "")
            model_val  = app.storage.user.get("selected_model", "")
            ctx_parts = []
            if email_val: ctx_parts.append(f"user_email={email_val}")
            if asset_val: ctx_parts.append(f"asset_id={asset_val}")
            if model_val: ctx_parts.append(f"device={model_val}")
            enriched_query = (
                f"[Context: {', '.join(ctx_parts)}]\n{query}"
                if ctx_parts else query
            )

            try:
                raw = await loop.run_in_executor(executor, agents[exp_id], enriched_query)
            except Exception as exc:
                raw = f"❌ Error:\n`{str(exc) or repr(exc)}`"
            latency = time.perf_counter() - t0
            try:
                thinking[exp_id].delete()
                add_agent_msg(exp_id, str(raw), latency)
            except (ValueError, RuntimeError):
                pass
            parsed = parse_dossier(str(raw))
            return {"exp_id": exp_id, "tool": parsed["tool_name"],
                    "latency": latency, "is_error": bool(parsed["error"])}

        results = sorted(
            list(await asyncio.gather(*[run_experiment(eid) for eid in EXPERIMENTS])),
            key=lambda x: x["exp_id"],
        )
        try:
            show_comparison(results)
            query_input.run_method("focus")
        except (ValueError, RuntimeError, AssertionError):
            pass
        finally:
            # Always re-enable input — even if show_comparison raised
            try:
                query_input.enable()
                send_btn.enable()
            except (ValueError, RuntimeError, AssertionError):
                pass
            safe_set_not_processing()

    send_btn.on_click(handle_send)
    query_input.on("keydown.enter", handle_send)
