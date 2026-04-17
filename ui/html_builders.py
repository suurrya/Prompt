"""
ui/html_builders.py
===================
All HTML generation functions for the benchmark UI.
Returns HTML strings — no NiceGUI elements are created here.

Note: details_html accepts an optional `agents` dict so it can check the
active model ID without importing the global from app.py (avoids circular imports).
"""

from __future__ import annotations
import re

from ui.config import EXPERIMENTS
from ui.parsing import escape_html_text, get_human_friendly_tool_summary, parse_argument_string, parse_dossier


def response_card_html(
    tool_name: str, args_str: str, color: str, panel_idx: int
) -> str:
    """Main visible card — clean human-readable tool decision with a collapsible details toggle."""
    answer = escape_html_text(
        get_human_friendly_tool_summary(tool_name, parse_argument_string(args_str))
    )
    cid = f"details-{panel_idx}"
    return (
        f'<div style="background:{color}12;border:1.5px solid {color}50;border-radius:10px;'
        f'padding:11px 13px;margin:0 0 6px;">'
        f'<div style="font-size:13px;color:#1e293b;line-height:1.55;">{answer}</div>'
        f'<div style="margin-top:8px;">'
        f'<button onclick="var d=document.getElementById(\'{cid}\');'
        f'd.style.display=d.style.display===\'none\'?\'block\':\'none\';" '
        f'style="font-size:10.5px;color:{color};background:none;border:none;cursor:pointer;'
        f'padding:0;font-weight:600;">▸ Show reasoning &amp; details</button>'
        f'</div></div>'
        f'<div id="{cid}" style="display:none;">'
    )


def details_html(
    parsed: dict, color: str, exp_id: int, agents: dict | None = None
) -> str:
    """
    Educational reasoning section — explains HOW each prompt technique arrived at its decision.
    Pass the loaded `agents` dict so the model-ID warning can check the active model.
    """
    parts = []
    meta = EXPERIMENTS[exp_id]  # noqa: F841  (used via exp_id below)

    # ── Error detail (show actual exception when agent failed) ────────────────
    if parsed.get("error"):
        parts.append(
            f'<div style="background:#fef2f2;border:1px solid #fecaca;'
            f'border-radius:6px;padding:8px 10px;margin-bottom:10px;">'
            f'<div style="font-size:10px;font-weight:700;color:#b91c1c;'
            f'text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;">'
            f'❌ Error detail</div>'
            f'<div style="font-size:11px;color:#7f1d1d;font-family:monospace;'
            f'word-break:break-word;line-height:1.5;">'
            f'{escape_html_text(parsed["error"])}</div>'
            f'</div>'
        )

    # ── Per-technique "How I reasoned" banner ─────────────────────────────────
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

    # ── TF-IDF examples (Exp 3 & 4) ──────────────────────────────────────────
    if parsed["examples"]:
        chips = "".join(
            f'<div style="background:#fff;border:1px solid {color}30;'
            f'border-left:3px solid {color};border-radius:4px;'
            f'padding:4px 9px;margin-bottom:4px;font-size:11px;'
            f'color:#374151;font-family:monospace;line-height:1.4;">'
            f'{escape_html_text(e[:80])}{"…" if len(e) > 80 else ""}</div>'
            for e in parsed["examples"]
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
        parts.append(
            f'<div style="font-size:11px;color:#94a3b8;font-style:italic;margin-bottom:8px;">'
            f'No retrieved examples available for this response.</div>'
        )

    # ── Reasoning trace ───────────────────────────────────────────────────────
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
            qm = re.match(r"^(Q[1-5][\.:]\s*)(.+)", line)
            if qm:
                lines_html += (
                    f'<div style="display:flex;gap:6px;margin-bottom:5px;align-items:baseline;">'
                    f'<span style="font-size:10.5px;font-weight:700;color:{color};'
                    f'min-width:24px;flex-shrink:0;">{escape_html_text(qm.group(1).rstrip())}</span>'
                    f'<span style="font-size:11.5px;color:#1e293b;">{escape_html_text(qm.group(2))}</span>'
                    f'</div>'
                )
            elif re.match(r"^\s*[→\-\>]", line):
                lines_html += (
                    f'<div style="display:flex;gap:6px;margin-bottom:5px;margin-top:2px;'
                    f'background:{color}15;border-radius:5px;padding:4px 7px;">'
                    f'<span style="font-size:13px;color:{color};">→</span>'
                    f'<span style="font-size:11.5px;font-weight:600;color:{color};">'
                    f'{escape_html_text(re.sub(r"^[→\\->]+\\s*", "", line))}</span>'
                    f'</div>'
                )
            elif line.startswith(("•", "-", "*")):
                lines_html += (
                    f'<div style="display:flex;gap:5px;margin-bottom:4px;">'
                    f'<span style="color:{color};flex-shrink:0;margin-top:1px;">›</span>'
                    f'<span style="font-size:11.5px;color:#374151;">'
                    f'{escape_html_text(line.lstrip("•-* "))}</span></div>'
                )
            elif re.match(r"^\d+[\.\:]", line):
                kv = re.split(r"[\.\:]\s*", line, 1)
                num, rest = kv[0], kv[1] if len(kv) > 1 else ""
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
        active_model = ""
        if agents and exp_id in agents:
            active_model = getattr(getattr(agents[exp_id], "_model", None), "model_id", "")
        if "llama-3.3-70b-instruct" not in active_model:
            parts.append(
                f'<div style="background:#fef9c3;border:1px solid #fde047;border-radius:6px;'
                f'padding:8px 10px;margin-bottom:8px;">'
                f'<div style="font-size:10.5px;color:#92400e;">'
                f'⚠️ No reasoning trace captured. The model may not have produced a '
                f'"Thought:" block this time — this can happen with smaller models that '
                f'skip the reasoning step. Try upgrading to llama-3.3-70b-instruct.</div>'
                f'</div>'
            )

    # ── Raw tool arguments ────────────────────────────────────────────────────
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


def error_card_html(msg: str, etype: str, latency: float) -> str:
    """Red-themed error card shown when an agent fails."""
    if etype == "max_steps":
        icon, title = "⏱️", "Agent reached max steps"
        body = (
            "The model exhausted its step budget before calling a tool. "
            "Upgrade to meta/llama-3.3-70b-instruct in agents.py."
        )
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


def render_response(
    raw: str, exp_id: int, latency: float, panel_idx: int,
    agents: dict | None = None,
) -> str:
    """
    Orchestrates the full response HTML from raw agent output.
    Pass `agents` to enable the active-model warning in details_html.
    """
    parsed = parse_dossier(raw)
    color  = EXPERIMENTS[exp_id]["color"]
    if parsed["error"]:
        return error_card_html(parsed["error"], parsed["error_type"], latency)
    if not parsed["tool_name"]:
        return (
            f'<div style="background:#fef9c3;border:1px solid #fde047;border-radius:8px;'
            f'padding:10px 12px;font-size:12px;color:#854d0e;margin:0 0 6px;">'
            f'⚠️ No tool was called — agent may have reached max steps.</div>'
        )
    html  = response_card_html(parsed["tool_name"], parsed["tool_args_str"], color, panel_idx)
    html += details_html(parsed, color, exp_id, agents)
    html += "</div>"  # close the collapsible div
    return html
