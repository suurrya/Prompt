"""
ui/parsing.py
=============
Text-processing utilities: HTML escaping, tool-argument parsing,
and the regex dossier engine that extracts structured data from raw agent output.
"""

from __future__ import annotations
import html as _html
import re


def escape_html_text(text: str) -> str:
    """Escape text to HTML-safe characters (prevents XSS and rendering errors)."""
    return _html.escape(str(text))


def get_human_friendly_tool_summary(tool_name: str, arguments: dict) -> str:
    """
    Converts a technical tool call into a plain-English sentence.
    Example: create_ticket(priority="high") → "I've created a high priority IT ticket."
    """
    templates = {
        "reset_password":        "I've initiated a password reset for {user_email} via {method}.",
        "lookup_knowledge_base": "I've searched the knowledge base for: \"{query}\".",
        "create_ticket":         "I've created a {priority} {category} ticket: \"{summary}\".",
        "escalate_ticket":       "I've escalated ticket {ticket_id} to {escalate_to}.",
        "check_system_status":   "I've checked the live status of the {service_name} system.",
        "get_user_info":         "I've retrieved directory details for {user_email}.",
        "lookup_user_account":   "I've looked up the account status for {email}.",
        "schedule_maintenance":  "I've booked a {maintenance_type} slot for {asset_id}.",
        "process_refund":        "I've initiated a refund for reservation {reservation_id}.",
    }
    template = templates.get(tool_name)
    if not template:
        return f"I've decided to use the {tool_name} tool."

    safe_args = {
        "user_email":       arguments.get("user_email",       "the account"),
        "method":           arguments.get("method",           "email"),
        "query":            arguments.get("query",            "your issue"),
        "priority":         arguments.get("priority",         "medium"),
        "category":         arguments.get("category",         "IT"),
        "summary":          arguments.get("summary",          "Issue reported"),
        "ticket_id":        arguments.get("ticket_id",        "INC-XXXX"),
        "escalate_to":      arguments.get("escalate_to",      "the specialist team"),
        "service_name":     arguments.get("service_name",     "IT"),
        "email":            arguments.get("email",            "the user"),
        "maintenance_type": arguments.get("maintenance_type", "maintenance"),
        "asset_id":         arguments.get("asset_id",         "the device"),
        "reservation_id":   arguments.get("reservation_id",   "REF-XXXX"),
    }
    return template.format(**safe_args)


def parse_argument_string(args_str: str) -> dict:
    """
    Converts a raw argument string like 'user_email="bob@corp.com", method="sms"'
    into a Python dict: {'user_email': 'bob@corp.com', 'method': 'sms'}.
    """
    args: dict = {}
    if not args_str:
        return args
    pairs = re.split(r',\s*(?=\w+\s*=)', args_str)
    for part in pairs:
        if "=" in part:
            key, value = part.split("=", 1)
            args[key.strip()] = value.strip().strip('"\'')
    return args


def parse_dossier(text: str) -> dict:
    """
    Regex extraction engine.
    Takes raw agent output text and returns a structured dict with keys:
      examples, thought, tool_name, tool_args_str, error, error_type.
    """
    out = {
        "examples": [], "thought": "", "tool_name": "",
        "tool_args_str": "", "error": "", "error_type": "",
    }

    # ── Errors ────────────────────────────────────────────────────────────────
    if "❌" in text:
        m = re.search(r"❌\s*Error[:\s]*\n?`?([^`]*)`?", text, re.DOTALL)
        msg = m.group(1).strip() if m else text.strip()
        if not msg or msg in ('""', "''", ""):
            out["error"] = "Agent reached maximum steps without completing the task."
            out["error_type"] = "max_steps"
        elif "max step" in msg.lower():
            out["error"] = msg
            out["error_type"] = "max_steps"
        elif "not in the tool" in msg.lower() or "schema" in msg.lower():
            out["error"] = msg
            out["error_type"] = "schema"
        else:
            out["error"] = msg or repr(text)
            out["error_type"] = "general"
        return out

    if text.strip().startswith("⚠️"):
        out["error"] = "No tool was called."
        out["error_type"] = "no_tool"
        return out

    # ── Examples (Exp 3 & 4) ─────────────────────────────────────────────────
    examples = re.search(
        r"(?:Examples selected|CoT Examples selected)[^\n]*:\n(.*?)\n\n", text, re.DOTALL
    )
    if examples:
        for line in examples.group(1).split("\n"):
            line = line.strip().lstrip("- ").strip("`").rstrip(".").strip()
            if line:
                out["examples"].append(line)

    # ── Thought — try every known format ─────────────────────────────────────
    # 1. Fenced ```markdown block (Exp 2 & 4)
    tm = re.search(r"```(?:markdown)?\n(.*?)```", text, re.DOTALL)
    if tm and tm.group(1).strip():
        out["thought"] = tm.group(1).strip()

    # 2. ### Chain of Thought section (Exp 2)
    if not out["thought"]:
        cot = re.search(
            r"###\s*(?:Chain of Thought|Final Chain of Thought)\s*\n+(.*?)(?=###|\Z)",
            text, re.DOTALL | re.IGNORECASE,
        )
        if cot:
            candidate = re.sub(r"```(?:markdown)?|```", "", cot.group(1)).strip()
            if candidate:
                out["thought"] = candidate

    # 3. Inline "Thought: ..." line (Exp 3)
    if not out["thought"]:
        it = re.search(r"Thought:\s*(.+?)(?=\n\n|###|\Z)", text, re.DOTALL)
        if it and it.group(1).strip():
            out["thought"] = it.group(1).strip()

    # 4. Q1/Q2/Q3 structured reasoning block
    if not out["thought"]:
        qblock = re.search(
            r"((?:Q[1-5][:\.\s].*?)(?:\n.*?){2,}(?:→|->|Action:).*)", text, re.DOTALL
        )
        if qblock:
            out["thought"] = qblock.group(1).strip()

    # 5. Free-form paragraph before ### Decision
    if not out["thought"]:
        before_decision = re.search(
            r"###\s*(?:Dynamic (?:Few-Shot|CoT) Selection[^\n]*\n.*?)?\n\n(.+?)(?=###\s*Decision|\Z)",
            text, re.DOTALL | re.IGNORECASE,
        )
        if before_decision:
            candidate = before_decision.group(1).strip()
            if len(candidate) > 40 and not candidate.startswith("-"):
                out["thought"] = candidate

    # ── Tool call — only from the Decision section ────────────────────────────
    dm = re.search(r"###\s*Decision\s*\n(.*?)(?=###|\Z)", text, re.DOTALL | re.IGNORECASE)
    search = dm.group(1) if dm else text
    tm2 = re.search(r"`(\w+)\(([^`]*)\)`", search)
    if tm2:
        out["tool_name"]     = tm2.group(1)
        out["tool_args_str"] = tm2.group(2).strip()

    return out
