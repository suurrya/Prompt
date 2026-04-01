"""
_tool_extract.py  —  shared utility for robust tool-call extraction.

smolagents' ToolCallingAgent requires the LLM to output structured JSON tool calls.
Small models (≤8B) often output tool calls in plain text instead:

    Action: create_ticket(category="access", priority="high", ...)

In those cases step.tool_calls is empty, so we fall back to scanning the raw
model output text with a regex.

Import in all four agents:
    from _tool_extract import extract_tool_calls
"""
from __future__ import annotations
import re

# Every tool name the agents can call (mirrors tools.py)
KNOWN_TOOLS = {
    "lookup_knowledge_base", "create_ticket", "escalate_ticket",
    "reset_password", "get_user_info", "lookup_user_account",
    "check_system_status", "schedule_maintenance", "process_refund",
    "store_resolved_ticket", "save_ticket_to_long_term_memory",
    "get_user_long_term_memory", "get_customer_history",
}


class ToolCall:
    """Minimal stand-in for smolagents ToolCall objects."""
    def __init__(self, name: str, arguments: dict):
        self.name      = name
        self.arguments = arguments


def parse_args_from_text(args_text: str) -> dict:
    """
    Parse    key="value", key="value"   into a plain dict.
    Also handles unquoted values:  key=value
    """
    result = {}
    # Quoted values
    for k, v in re.findall(r'(\w+)\s*=\s*"([^"]*)"', args_text):
        result[k] = v
    # Unquoted values (only if key not already found)
    for k, v in re.findall(r"(\w+)\s*=\s*'([^']*)'", args_text):
        result.setdefault(k, v)
    for k, v in re.findall(r"(\w+)\s*=\s*([^,\s)\n\"'][^\s,)]*)", args_text):
        result.setdefault(k, v)
    return result


def scan_text_for_tool(text: str) -> "FakeToolCall | None":
    """
    Try several patterns to find a tool call in raw model output text:
      1. `tool_name(args)` backtick format
      2. Action: tool_name(args)
      3. Bare  tool_name(args)  on its own line
    """
    patterns = [
        r"`(\w+)\(([^`]*)\)`",                        # `tool_name(args)`
        r"(?:Action|action)\s*:\s*(\w+)\(([^)\n]*)\)",# Action: tool_name(args)
        r"^\s*(\w+)\(([^)\n]*)\)\s*$",                # bare on its own line
    ]
    for pat in patterns:
        flags = re.MULTILINE if "^" in pat else 0
        for m in re.finditer(pat, text, flags):
            name = m.group(1)
            if name in KNOWN_TOOLS:
                args = parse_args_from_text(m.group(2))
                return ToolCall(name, args)
    return None


def extract_tool_calls(steps) -> list:
    """
    Return a list of tool calls from the smolagents step list.

    Strategy (in priority order):
      1. step.tool_calls  — populated when the LLM outputs structured JSON.
      2. step.tool_name   — older smolagents API.
      3. Regex scan of step.model_output_message.content — catches plain-text
         "Action: tool_name(args)" outputs from smaller models.

    Returns a list of real smolagents ToolCall objects or _FakeToolCall objects.
    The calling agent code only accesses .name and .arguments on each item,
    so both types are compatible.
    """
    # Pass 1: structured tool_calls
    for step in reversed(list(steps)):
        tcs = getattr(step, "tool_calls", None)
        if tcs:
            return list(tcs)
        tn = getattr(step, "tool_name", None)
        if isinstance(tn, str) and tn in KNOWN_TOOLS:
            ta = getattr(step, "tool_arguments", None) or {}
            return [ToolCall(tn, ta if isinstance(ta, dict) else {})]

    # Pass 2: scan raw model output text
    for step in reversed(list(steps)):
        msg = getattr(step, "model_output_message", None)
        if msg:
            content = getattr(msg, "content", "") or ""
            fake = scan_text_for_tool(content)
            if fake:
                return [fake]
        # Also try string attrs (older smolagents)
        for attr in vars(step) if hasattr(step, "__dict__") else []:
            val = str(getattr(step, attr, "") or "")
            fake = scan_text_for_tool(val)
            if fake:
                return [fake]

    return []
