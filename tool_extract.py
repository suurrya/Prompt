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
import re # Purposes: Used to scan model output text for complex 'Action: ...' patterns.

# Purposes: A definitive whitelist of all allowed tool names. 
# This prevents the agent from hallucinating or calling invalid functions.
KNOWN_TOOLS = {
    "lookup_knowledge_base", "create_ticket", "escalate_ticket",
    "reset_password", "get_user_info", "lookup_user_account",
    "check_system_status", "schedule_maintenance", "process_refund",
    "store_resolved_ticket", "save_ticket_to_long_term_memory",
    "get_user_long_term_memory", "get_customer_history",
}


class ToolCall:
    """Minimal stand-in for smolagents ToolCall objects."""
    # Purposes: Allows us to treat manually parsed regex matches as if they were 
    # real objects from the smolagents framework.
    def __init__(self, name: str, arguments: dict):
        self.name = name # The name of the tool (e.g., 'create_ticket').
        self.arguments = arguments # The dictionary of parameters (e.g., {'priority': 'high'}).


def parse_args_from_text(args_text: str) -> dict:
    """
    Parse    key="value", key="value"   into a plain dict.
    Also handles unquoted values:  key=value
    """
    # Purposes: Initializes an empty dictionary to hold the final parsed parameters.
    result = {}
    # Quoted values
    # Purposes: Finds pairs like priority="high" where double quotes are used.
    for k, v in re.findall(r'(\w+)\s*=\s*"([^"]*)"', args_text):
        result[k] = v
    # Unquoted values (only if key not already found)
    # Purposes: Finds pairs like priority='high' using single quotes as a fallback.
    for k, v in re.findall(r"(\w+)\s*=\s*'([^']*)'", args_text):
        result.setdefault(k, v)
    # Purposes: Final fallback for unquoted values like count=5 or status=open.
    for k, v in re.findall(r"(\w+)\s*=\s*([^,\s)\n\"'][^\s,)]*)", args_text):
        result.setdefault(k, v)
    # Purposes: Returns the complete dictionary of arguments found in the text.
    return result


def scan_text_for_tool(text: str) -> "FakeToolCall | None":
    """
    Try several patterns to find a tool call in raw model output text:
      1. `tool_name(args)` backtick format
      2. Action: tool_name(args)
      3. Bare  tool_name(args)  on its own line
      4. **Tool:** tool_name(args)         — CoT bold-label format
      5. "call the `tool_name` tool"       — prose description format
      6. Bare tool_name with no args       — minimal output (e.g. "schedule_maintenance")
    """
    patterns = [
        r"`(\w+)\(([^`]*)\)`",                              # `tool_name(args)`
        r"(?:Action|action)\s*:\s*`?(\w+)\(([^)\n]*)\)`?", # Action: tool_name(args)
        r"^\s*(\w+)\(([^)\n]*)\)\s*$",                     # bare on its own line
        r"\*\*Tool:\*\*\s*(\w+)\(([^)]*)\)",               # **Tool:** tool_name(args)
        r"(?:call(?:ing)?)\s+(?:the\s+)?`(\w+)`\s+tool",   # "call the `tool_name` tool"
        r"^\s*(\w+)\s*$",                                   # bare tool name, no args
    ]
    # Purposes: Loops through several common patterns small models use to indicate a tool call.
    for pat in patterns:
        flags = re.MULTILINE if "^" in pat else 0 # Purpose: Correctly handle line-anchored (^) patterns.
        # Purposes: Finds all matches for the current pattern in the given text.
        for m in re.finditer(pat, text, flags):
            name = m.group(1) # Purpose: The tool name located in capture group 1.
            # Purposes: Verifies that the found command is actually a valid tool in our kit.
            if name in KNOWN_TOOLS:
                # Purposes: Parses the string inside the parentheses into a dictionary of arguments.
                args = parse_args_from_text(m.group(2)) if m.lastindex >= 2 else {}
                # Purposes: Returns a compatible ToolCall object for the framework to execute.
                return ToolCall(name, args)

    # Last resort: scan for any known tool name mentioned in the text.
    # Purposes: Catches truncated CoT outputs where the model reasoned about a tool but the Action line never appeared.
    for tool in KNOWN_TOOLS:
        if re.search(rf"\b{tool}\b", text):
            return ToolCall(tool, {})

    return None # Purposes: Returns None if no valid tool pattern was discovered.


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
    # Purposes: We search backwards to find the most recent decision made by the model.
    for step in reversed(list(steps)):
        # Purposes: Attempts to retrieve the standard 'tool_calls' attribute from the step.
        tcs = getattr(step, "tool_calls", None)
        if tcs:
            return list(tcs) # Purposes: Found structured tool calls (success for larger models).
        # Purposes: Fallback to an older attribute name used in some versions of the library.
        tn = getattr(step, "tool_name", None)
        if isinstance(tn, str) and tn in KNOWN_TOOLS:
            ta = getattr(step, "tool_arguments", None) or {}
            # Purposes: Creates a compatible object if only raw name/args strings are found.
            return [ToolCall(tn, ta if isinstance(ta, dict) else {})]

    # Pass 2: scan raw model output text
    # Purposes: If structured passion failed, search the raw text blocks for tool patterns.
    for step in reversed(list(steps)):
        # Purposes: Checks the model's textual response message.
        msg = getattr(step, "model_output_message", None)
        if msg:
            content = getattr(msg, "content", "") or ""
            # Purposes: Uses our regex-based 'brain' to find a tool call in the message text.
            fake = scan_text_for_tool(content)
            if fake:
                return [fake] # Purposes: Found a tool through text parsing (success for 8B models).
        # Purposes: Final attempt to scan every string-like attribute in the step object.
        for attr in vars(step) if hasattr(step, "__dict__") else []:
            val = str(getattr(step, attr, "") or "")
            # Purposes: Re-runs the scan on any other available hidden text data.
            fake = scan_text_for_tool(val)
            if fake:
                return [fake] 

    # Purposes: Returns an empty list if no tool call (structured or text) was found in the memory.
    return []
