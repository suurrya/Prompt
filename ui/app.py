"""
ui/app.py
==========
NiceGUI app that runs all four IT Helpdesk agents side-by-side in one browser window.

Layout
------
  ┌───────────────────────────────────────────────────────────────────┐
  │  🖥  IT Helpdesk Agent Benchmark                    [ clear all ] │
  ├────────────────┬────────────────┬────────────────┬────────────────┤
  │ Exp 1          │ Exp 2          │ Exp 3          │ Exp 4          │
  │ Static         │ Static CoT     │ Dynamic        │ Dynamic CoT    │
  │ Few-Shot       │                │ Few-Shot       │                │
  ├────────────────┼────────────────┼────────────────┼────────────────┤
  │  chat area     │  chat area     │  chat area     │  chat area     │
  │  (scrollable)  │  (scrollable)  │  (scrollable)  │  (scrollable)  │
  ├────────────────┴────────────────┴────────────────┴────────────────┤
  │  [ shared input field                      ]  [ Send to all  ]    │
  └───────────────────────────────────────────────────────────────────┘

Each panel shows streamed messages from its own agent.
The shared input broadcasts the same query to all four agents simultaneously.

Usage
-----
    cd it_helpdesk_agents
    python ui/app.py

Then open  http://localhost:8080  in your browser.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from dotenv import load_dotenv
from nicegui import app, ui

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))

# ── Lazy-load agents (imported once, instantiated once per session) ───────────
def _load_agents() -> dict[int, object]:
    """
    Import and instantiate all four agents.
    Called once when the first user message is sent so the UI renders
    immediately even if smolagents takes a moment to import.
    """
    agents = {}
    modules = {
        1: "project_1_few_shot.agents",
        2: "project_2_chain_of_thought.agents",
        3: "project_3_dynamic_few_shot.agents",
        4: "project_4_dynamic_cot.agents",
    }
    for exp_id, module_path in modules.items():
        import importlib
        mod = importlib.import_module(module_path)
        agents[exp_id] = mod.ITHelpdeskAgent(verbose=False)
    return agents


# ── Global state (per-server, not per-session — fine for local use) ──────────
_agents: dict[int, object] | None = None
_executor = ThreadPoolExecutor(max_workers=4)

EXPERIMENTS = {
    1: {"label": "Exp 1 — Static Few-Shot",          "color": "#6366f1"},  # indigo
    2: {"label": "Exp 2 — Static Chain-of-Thought",  "color": "#0ea5e9"},  # sky
    3: {"label": "Exp 3 — Dynamic Few-Shot",         "color": "#10b981"},  # emerald
    4: {"label": "Exp 4 — Dynamic CoT",              "color": "#f59e0b"},  # amber
}


# ── Per-panel chat state ──────────────────────────────────────────────────────

class ChatPanel:
    """
    Encapsulates all UI elements and state for a single agent's chat panel.
    """

    def __init__(self, exp_id: int, meta: dict):
        self.exp_id = exp_id
        self.meta = meta
        self.messages: list[dict] = []      # {"role": "user"|"agent", "text": str}
        self.is_thinking = False

        self._msg_container: ui.column | None = None
        self._spinner: ui.spinner | None = None

    # ── Build NiceGUI elements ────────────────────────────────────────────

    def build(self) -> None:
        """Render the full panel inside whatever parent container is active."""
        color = self.meta["color"]
        label = self.meta["label"]

        with ui.card().classes("flex-1 min-w-0 flex flex-col h-full").style(
            "border-top: 3px solid " + color + "; min-height: 520px; max-height: 520px;"
        ):
            # Header
            with ui.row().classes("items-center gap-2 px-3 pt-3 pb-1"):
                ui.element("div").style(
                    f"width:10px;height:10px;border-radius:50%;background:{color};"
                )
                ui.label(label).classes("text-sm font-semibold text-gray-700")
                ui.space()
                ui.button(
                    icon="delete_outline",
                    on_click=self._clear,
                ).props("flat dense size=sm").tooltip("Clear this panel")

            ui.separator()

            # Scrollable message area
            with ui.scroll_area().classes("flex-1 px-3 py-2").style(
                "height: 380px; overflow-y: auto;"
            ) as scroll:
                self._scroll = scroll
                self._msg_container = ui.column().classes("w-full gap-2")
                with self._msg_container:
                    ui.label("Ask a question below to start.").classes(
                        "text-gray-400 text-xs italic"
                    )

            # Spinner (hidden until agent is thinking)
            with ui.row().classes("items-center gap-2 px-3 pb-2"):
                self._spinner = ui.spinner(size="sm").classes("text-gray-400")
                self._status_label = ui.label("").classes("text-xs text-gray-400")
                self._spinner.set_visibility(False)

    # ── Message rendering ─────────────────────────────────────────────────

    def _render_message(self, role: str, text: str, latency: float | None = None) -> None:
        """Append a chat bubble to the message container."""
        is_user = role == "user"
        bubble_classes = (
            "rounded-lg px-3 py-2 text-sm max-w-full break-words "
            + ("bg-indigo-50 text-indigo-900 self-end ml-6" if is_user
               else "bg-gray-100 text-gray-800 self-start mr-6")
        )
        with self._msg_container:
            with ui.column().classes(
                "w-full items-end" if is_user else "w-full items-start"
            ):
                ui.label(text).classes(bubble_classes)
                if latency is not None:
                    ui.label(f"{latency:.1f}s").classes(
                        "text-gray-400 text-xs mt-0.5"
                    )

    def append_user(self, text: str) -> None:
        self._render_message("user", text)
        self._scroll.scroll_to(percent=1.0)

    def append_agent(self, text: str, latency: float) -> None:
        self._render_message("agent", text, latency)
        self._status_label.set_text("")
        self._scroll.scroll_to(percent=1.0)

    def set_thinking(self, thinking: bool) -> None:
        self.is_thinking = thinking
        self._spinner.set_visibility(thinking)
        self._status_label.set_text("Thinking…" if thinking else "")

    def _clear(self) -> None:
        self._msg_container.clear()
        with self._msg_container:
            ui.label("Ask a question below to start.").classes(
                "text-gray-400 text-xs italic"
            )
        self._status_label.set_text("")
        self._spinner.set_visibility(False)

    # ── Agent call ────────────────────────────────────────────────────────

    async def send(self, query: str, agent) -> None:
        """
        Run the agent in a thread pool and update the UI on completion.
        Called concurrently for all four panels from the shared input handler.
        """
        self.append_user(query)
        self.set_thinking(True)

        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()
        try:
            response = await loop.run_in_executor(_executor, agent, query)
        except Exception as exc:
            response = f"⚠️ Error: {exc}"
        latency = time.perf_counter() - t0

        self.set_thinking(False)
        self.append_agent(str(response), latency)


# ── Main page ─────────────────────────────────────────────────────────────────

@ui.page("/")
async def index():
    # ── Load agents lazily on first page visit ────────────────────────────
    global _agents
    if _agents is None:
        try:
            _agents = await asyncio.get_event_loop().run_in_executor(
                _executor, _load_agents
            )
        except Exception as exc:
            ui.notify(f"Failed to load agents: {exc}", type="negative", timeout=0)
            _agents = {}

    panels: dict[int, ChatPanel] = {}

    # ── Page-level styles ─────────────────────────────────────────────────
    ui.add_head_html("""
    <style>
      body { background: #f8fafc; }
      .nicegui-content { padding: 0 !important; }
    </style>
    """)

    # ── Header ────────────────────────────────────────────────────────────
    with ui.header().classes("bg-white shadow-sm px-6 py-3 items-center"):
        with ui.row().classes("items-center gap-3 w-full"):
            ui.icon("computer").classes("text-indigo-600 text-2xl")
            ui.label("IT Helpdesk Agent Benchmark").classes(
                "text-lg font-bold text-gray-800"
            )
            ui.badge("4 agents").props("outline color=indigo")
            ui.space()
            ui.button(
                "Clear all",
                icon="delete_sweep",
                on_click=lambda: [p._clear() for p in panels.values()],
            ).props("flat").classes("text-gray-500")

    # ── Body: 4 panels in a row ───────────────────────────────────────────
    with ui.element("div").classes(
        "flex flex-row gap-3 px-4 pt-4"
    ).style("height: calc(100vh - 180px);"):
        for exp_id, meta in EXPERIMENTS.items():
            panel = ChatPanel(exp_id, meta)
            panel.build()
            panels[exp_id] = panel

    # ── Shared input bar ──────────────────────────────────────────────────
    with ui.footer().classes("bg-white border-t px-6 py-3"):
        with ui.row().classes("items-center gap-3 w-full"):
            query_input = (
                ui.input(placeholder="Type your IT question and press Enter or click Send…")
                .classes("flex-1")
                .props("outlined dense clearable")
            )
            send_btn = ui.button("Send to all", icon="send").props("color=indigo")

            # Legend chips
            for exp_id, meta in EXPERIMENTS.items():
                ui.chip(
                    f"Exp {exp_id}",
                    color="white",
                ).style(
                    f"border: 2px solid {meta['color']}; color: {meta['color']}; "
                    "font-size: 11px; height: 24px;"
                )

    # ── Send handler ──────────────────────────────────────────────────────
    async def handle_send() -> None:
        query = (query_input.value or "").strip()
        if not query:
            ui.notify("Please type a question first.", type="warning")
            return

        if not _agents:
            ui.notify("Agents not loaded — check your .env file.", type="negative")
            return

        # Disable input while processing
        query_input.disable()
        send_btn.disable()
        query_input.set_value("")

        # Fire all 4 agents concurrently
        await asyncio.gather(*[
            panels[exp_id].send(query, _agents[exp_id])
            for exp_id in EXPERIMENTS
            if exp_id in _agents
        ])

        query_input.enable()
        send_btn.enable()
        query_input.run_method("focus")

    send_btn.on_click(handle_send)
    query_input.on("keydown.enter", handle_send)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ in ("__main__", "__mp_main__"):
    # Validate API key before starting the server
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        print("\n[ERROR] OPENAI_API_KEY is missing or still set to the placeholder.")
        print("        Copy .env.example → .env, add your real key, then re-run.\n")
        sys.exit(1)

    print("\nStarting IT Helpdesk Agent Benchmark UI…")
    print("Open  http://localhost:8080  in your browser.\n")

    ui.run(
        title="IT Helpdesk Agent Benchmark",
        favicon="🖥",
        port=8080,
        reload=False,          # set True during development
        dark=False,
    )
