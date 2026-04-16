import re # Purposes: Used to search and extract tool patterns (Action: ...) from raw text.
import json # Purposes: Fallback parser for cases where the model outputs structured JSON.
import uuid # Purposes: Generates unique session IDs for each tool call in the framework.
import time # Purposes: Provides the 'sleep' function for the API retry logic.
import datetime # Purposes: Timestamps each model call in the debug log.
import threading # Purposes: Provides the Semaphore used to serialise concurrent API calls.
import concurrent.futures # Purposes: Provides ThreadPoolExecutor and Future for enforcing a hard per-call timeout.
from smolagents import OpenAIServerModel # Purposes: The base class that allows us to connect to the NVIDIA/OpenAI API.
from smolagents.models import ChatMessageToolCall, ChatMessageToolCallFunction # Purposes: Data structures used to represent tools in the agent's memory.

# Purposes: Hard ceiling on how long a single LLM API call may run.
# Any call that exceeds this will raise TimeoutError instead of blocking forever.
# Set conservatively at 45 s — well above normal latency (~2–10 s) but below the
# pathological 150-200 s values seen when NIMs is under heavy load.
API_TIMEOUT_SECONDS = 45

# Purposes: NVIDIA NIMs free tier processes requests serially — sending all 4 experiments
# at the same time causes every request after the first to queue for minutes.
# This semaphore allows only 1 API call to be in-flight at a time, eliminating queue wait.
# The UI still shows all 4 "thinking" indicators simultaneously; only the API calls are serialised.
_NVIDIA_API_SEMAPHORE = threading.Semaphore(1)


def _estimate_tokens(text: str) -> int:
    # Purposes: Provides a fast, dependency-free token count approximation.
    # The OpenAI rule-of-thumb is ~4 characters per token for English text.
    # We use max(1, ...) so we never return zero for non-empty strings.
    return max(1, len(text) // 4)


def _messages_to_text(messages) -> str:
    # Purposes: Collapses the full messages list into one string so _estimate_tokens
    # can operate on the entire conversation context in a single pass.
    parts = []
    # Purposes: Iterates over every message object in the list.
    for m in messages:
        # Purposes: Supports both smolagents ChatMessage objects (attribute access)
        # and plain dicts (key access) so the function works regardless of format.
        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
        # Purposes: Only appends non-empty content to avoid inflating the estimate.
        if content:
            parts.append(str(content))
    # Purposes: Joins all content pieces with a space — token count is length-based
    # so the separator choice doesn't affect accuracy meaningfully.
    return " ".join(parts)

# TextToolParserModel

#
# This class acts as a universal adapter or a "compatibility shim" to make
# a "legacy" text-based model API compatible with the modern, tool-native
# smol-agents framework.
#
# Its primary role is to solve the problem that the NVIDIA API endpoint for
# Llama 3 does not support the OpenAI-standard structured JSON tool-calling format.
# This class fakes that support through text parsing and adds critical resilience.
#
# Key Responsibilities:
#   1. Disables Native Tools (`supports_native_tools = False`): Forces smol-agents
#      to use simple text-based prompting instead of JSON.
#
#   2. Adds Resilient Retries (in `generate`): Automatically retries API calls
#      on transient server errors (like 500 or "DEGRADED"), making the app
#      more robust.
#
#   3. Parses Text to Tools (in `parse_tool_calls`): This is its main job.
#      It uses a series of regex patterns to find and extract tool calls
#      (like `Action: create_ticket(...)`) from the model's raw text output,
#      then manually constructs the structured ToolCall objects that the
#      rest of the framework expects.
#

class TextToolParserModel(OpenAIServerModel):
    """
    Custom model wrapper for the Hugging Face Inference Router.
    This model does NOT support native tool calls on the HF free tier 
    Inference Router for some models (e.g., Qwen2.5-Coder-7B). 
    This wrapper forces prompt-based tool calling and handles manual parsing.
    """
    
    # Purposes: Forces the agent framework to use "Text Mode" instead of "JSON Mode".
    # This ensures that the system prompt includes tool definitions as text instructions.
    supports_native_tools = False

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        """
        Overrides the standard generate which strips tools_to_call_from.
        This ensures that 'tools' and 'tool_choice' are NOT sent to the HF Router API.
        """
        kwargs.pop('tools_to_call_from', None)

        # Purposes: Standard LLM APIs might error out; we set a 3-attempt limit for robustness.
        max_retries = 3
        # Purposes: The starting wait time (2 seconds) before retrying an API call.
        base_delay_seconds = 2

        # Purposes: Estimate input token cost once before retrying — messages never change between attempts.
        input_tokens = _estimate_tokens(_messages_to_text(messages))

        # Purposes: Loop through the retry logic until success or max_retries reached.
        for attempt in range(max_retries):
            try:
                # Purposes: Record the moment we start waiting for the semaphore so we can
                # separate "queue wait" time from actual API generation time in the debug log.
                t_wait_start = time.perf_counter()

                # Purposes: Acquire the semaphore before touching the network.
                # Only one thread may hold it at a time, so concurrent experiments
                # take turns rather than all flooding the NVIDIA NIMs queue at once.
                with _NVIDIA_API_SEMAPHORE:
                    # Purposes: Semaphore acquired — queue wait is now over. Record how long we waited.
                    wait_elapsed = time.perf_counter() - t_wait_start

                    # Pass stop_sequences and response_format through to the parent so
                    # smolagents' built-in stop tokens are not silently dropped.
                    # Purposes: Start the wall-clock timer immediately before the API call.
                    t_api_start = time.perf_counter()

                    # Purposes: Submit the blocking API call to a single-worker thread pool.
                    # This lets us enforce API_TIMEOUT_SECONDS without killing the main thread.
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _executor:
                        # Purposes: Dispatch the parent generate() onto the background thread.
                        _future = _executor.submit(
                            super().generate, messages,
                            stop_sequences=stop_sequences,
                            response_format=response_format,
                            **kwargs
                        )
                        # Purposes: Block until the result arrives OR the timeout fires.
                        # concurrent.futures.TimeoutError is raised if the API exceeds the limit.
                        try:
                            result = _future.result(timeout=API_TIMEOUT_SECONDS)
                        except concurrent.futures.TimeoutError:
                            # Purposes: Cancel the pending future (best-effort) and surface a clear error.
                            _future.cancel()
                            raise TimeoutError(
                                f"LLM API did not respond within {API_TIMEOUT_SECONDS}s — "
                                f"skipping this call."
                            )

                # Purposes: Stop the API timer as soon as the full response is received.
                api_elapsed = time.perf_counter() - t_api_start

                # Purposes: Extract the model's raw text output for token estimation.
                output_text = getattr(result, "content", "") or ""

                # Purposes: Bundle all per-call statistics into a dict stored on self
                # so parse_tool_calls() — called immediately after — can read them
                # without needing them passed as arguments.
                self._last_call_stats = {
                    "timestamp":    datetime.datetime.now().strftime("%H:%M:%S"),  # Wall-clock time of the call
                    "wait_s":       wait_elapsed,                                   # Time spent waiting for the semaphore (queue delay)
                    "latency_s":    api_elapsed,                                    # Actual API round-trip time (generation only)
                    "tokens_in":    input_tokens,                                   # Estimated prompt token count
                    "tokens_out":   _estimate_tokens(output_text),                  # Estimated completion token count
                    "tokens_total": input_tokens + _estimate_tokens(output_text),   # Combined total
                }
                return result
            
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for known, temporary server-side errors
                is_transient_error = (
                    isinstance(e, TimeoutError) or
                    ("degraded" in error_str and "cannot be invoked" in error_str) or
                    ("500" in error_str and "internal server error" in error_str) or
                    ("timeout" in error_str) or
                    ("timed out" in error_str) or
                    ("rate limit" in error_str) or
                    ("429" in error_str)
                )

                if is_transient_error and attempt < max_retries - 1:
                    delay = base_delay_seconds * (2 ** attempt)
                    
                    print(f"\n[WARNING] API reported a transient server error. "
                          f"Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    
                    time.sleep(delay)
                    # The loop will then continue to the next attempt
                else:
                    # If it's a different error, or we've run out of retries,
                    # raise the exception to fail the call.
                    print(f"\n[ERROR] API call failed after {attempt + 1} attempts.")
                    raise e
        
        # Purposes: Final fail-safe if all retries are exhausted.
        raise Exception("API call failed permanently after multiple retries.")

    def parse_tool_calls(self, message):
        """
        A robust parser that extracts tool calls from text when native JSON tool calls are missing.
        Supports:
          - Arrow format: → tool_name(arg="val")
          - Action prefix: Action: tool_name(arg="val")
          - Markdown blocks: ```python tool_name(...) ```
          - JSON formats: {"action": "...", "parameters": {...}}
        """
        # If the model surprisingly returned native tool_calls, use them.
        if message.tool_calls:
            return message

        content = (message.content or "").strip()
        
        # 0. Debug Log — writes a structured entry per LLM call with timing and token stats.
        try:
            # Purposes: Retrieve the stats dict populated by generate(); fall back to empty dict
            # if parse_tool_calls is somehow called without a prior generate() (e.g. native tool path).
            stats = getattr(self, "_last_call_stats", {})

            # Purposes: Pull each metric out of the stats dict with a safe default so the
            # log never crashes even when stats are partially missing.
            ts         = stats.get("timestamp",    "??:??:??")  # Wall-clock time
            wait       = stats.get("wait_s",       0.0)         # Semaphore queue wait (how long we waited for our turn)
            latency    = stats.get("latency_s",    0.0)         # Actual API generation time (network + model compute)
            tokens_in  = stats.get("tokens_in",    "?")         # Estimated prompt tokens
            tokens_out = stats.get("tokens_out",   "?")         # Estimated completion tokens
            tokens_tot = stats.get("tokens_total", "?")         # Combined total

            with open("debug_model_output.txt", "a") as f:
                # Purposes: Visual separator makes it easy to find the start of each call in the log.
                f.write(f"\n{'─' * 60}\n")
                # Purposes: Timestamp lets you correlate log entries with UI latency readings.
                f.write(f"  Time      : {ts}\n")
                # Purposes: Shows how long this call waited for the serialising semaphore (queue delay).
                # A high value here means another experiment was already talking to the API.
                f.write(f"  Wait      : {wait:.2f}s  (semaphore / queue delay)\n")
                # Purposes: Shows only the real API round-trip time (model compute + network).
                # A high value here means NVIDIA NIMs itself was slow.
                f.write(f"  Latency   : {latency:.2f}s  (API generation time)\n")
                # Purposes: Token counts show prompt cost (IN) and generation cost (OUT) separately.
                f.write(f"  Tokens IN : ~{tokens_in}  (estimated from prompt length)\n")
                f.write(f"  Tokens OUT: ~{tokens_out}  (estimated from response length)\n")
                # Purposes: Total gives a single number for quick billing / rate-limit awareness.
                f.write(f"  Tokens TOT: ~{tokens_tot}\n")
                f.write(f"  Output    :\n{content}\n")
        except Exception:
            # Purposes: Debug logging must never crash the agent — silently skip on any I/O error.
            pass
            
        # 0. Pre-process: Strip markdown code blocks if the entire content is wrapped in one
        # or if there's a block at the end.
        content = re.sub(r"```(?:python|json)?\s*(.*?)\s*```", r"\1", content, flags=re.DOTALL).strip()
        # Purposes: A whitelist of tool names. If the AI hallucinates a name NOT in this list,
        # the parser will ignore it, preventing "Undefined Tool" errors.
        valid_tools = {
            "lookup_knowledge_base", "create_ticket", "escalate_ticket", "reset_password",
            "get_user_info", "lookup_user_account", "check_system_status", "schedule_maintenance",
            "process_refund", "store_resolved_ticket", "save_ticket_to_long_term_memory",
            "get_user_long_term_memory", "get_customer_history"
        }
        
        # Purposes: A temporary list to store any valid tool calls we find in the text.
        tool_calls = []

        # Purposes: Shared regex for parsing key="value" pairs from a tool's argument string.
        # Supports double-quoted, single-quoted, and bare unquoted values.
        arg_pattern = re.compile(r"""
            (\w+)               # Capture Group 1: The argument key (e.g., 'priority')
            \s*=\s*             # Equals sign with optional whitespace
            (?:                 # Non-capturing group for different value formats
                "([^"]*)"       # Capture Group 2: Double-quoted string
                |               # OR
                '([^']*)'       # Capture Group 3: Single-quoted string
                |               # OR
                ([^,\s\)]+)     # Capture Group 4: Unquoted value (e.g., numbers, booleans)
            )
        """, re.VERBOSE)

        # Purposes: Positional-argument fallback for single-parameter tools.
        # If the model forgets the 'key=' part, we map the bare value to the known parameter name.
        parameter_mapping = {
            "check_system_status": "service_name",
            "lookup_knowledge_base": "query",
            "get_user_info": "user_email",
            "lookup_user_account": "email",
            "get_user_long_term_memory": "user_id",
            "get_customer_history": "user_id"
        }

        def _parse_args(args_str: str, name: str) -> dict:
            # Purposes: Extracts key/value pairs from the raw argument string.
            args = {}
            for k, v1, v2, v3 in arg_pattern.findall(args_str):
                args[k] = v1 or v2 or v3
            # Purposes: Single-arg positional fallback when no key=value pairs were found.
            if not args and args_str.strip():
                clean_arg = args_str.strip(' \'"')
                if name in parameter_mapping:
                    args[parameter_mapping[name]] = clean_arg
            return args

        def _make_tool_call(name: str, args_str: str) -> ChatMessageToolCall:
            # Purposes: Constructs the structured ToolCall object the agent framework expects.
            return ChatMessageToolCall(
                id=str(uuid.uuid4()),
                type="function",
                function=ChatMessageToolCallFunction(name=name, arguments=_parse_args(args_str, name))
            )

        # ── Pass 1: Prefer explicit "Action:" prefix ─────────────────────────────
        # Purposes: CoT agents (Exp 2 & 4) write "Action: toolname(args)" at the end
        # of their reasoning trace. This is the most reliable signal — it is the model's
        # deliberate final decision, not text echoed from the system-prompt framework.
        # Matching this first prevents false positives from lines like
        # "→ create_ticket(priority=critical) THEN escalate_ticket." that appear
        # verbatim inside the echoed diagnostic framework earlier in the output.
        action_pattern = re.compile(r"""
            Action:\s*          # Requires the literal "Action:" prefix
            ([a-zA-Z_]\w*)      # Capture Group 1: tool_name
            \s*\(
            (.*?)               # Capture Group 2: raw args
            \)
        """, re.VERBOSE | re.DOTALL)

        for match in action_pattern.finditer(content):
            # Purposes: Reject hallucinated tool names not in the whitelist.
            name = match.group(1).strip()
            if name not in valid_tools:
                continue
            tool_calls.append(_make_tool_call(name, match.group(2).strip()))
            break  # Purposes: Only take the first "Action:"-prefixed call.

        # ── Pass 2: Fallback — scan all bare patterns, keep the LAST valid one ──
        # Purposes: Exp 1 / 3 / 4 output bare "toolname(args)" with no "Action:" prefix.
        # We take the LAST match (not the first) because:
        #   • The system-prompt framework text is echoed near the TOP of the output and
        #     contains false-positive patterns like "→ create_ticket(priority=critical)".
        #   • The model's actual decision is always at the BOTTOM of its response.
        # Using reversed() ensures framework echoes are skipped in favour of the true action.
        if not tool_calls:
            bare_pattern = re.compile(r"""
                (?:→\s*)?           # Optional arrow prefix (but NOT requiring "Action:")
                ([a-zA-Z_]\w*)      # Capture Group 1: tool_name
                \s*\(
                (.*?)               # Capture Group 2: raw args
                \)
            """, re.VERBOSE | re.DOTALL)

            all_matches = list(bare_pattern.finditer(content))
            for match in reversed(all_matches):  # Purposes: Iterate from last match to first.
                name = match.group(1).strip()
                if name not in valid_tools:
                    continue
                tool_calls.append(_make_tool_call(name, match.group(2).strip()))
                break  # Purposes: Keep only the last valid tool call found.

        if tool_calls:
            message.tool_calls = tool_calls
            return message

        # 2. Handle JSON formats (fallback for Qwen action/parameters or OpenAI name/arguments)
        if '"action":' in content or '"name":' in content:
            try:
                # Extract the first JSON object from the text block
                json_match = re.search(r"(\{.*\})", content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                    name = data.get("action") or data.get("name")
                    args = data.get("parameters") or data.get("arguments") or {}
                    if name:
                        message.tool_calls = [
                            ChatMessageToolCall(
                                id=str(uuid.uuid4()), 
                                type="function", 
                                function=ChatMessageToolCallFunction(name=name, arguments=args)
                            )
                        ]
                        return message
            except Exception:
                pass

        # If no custom parsing succeeded, return as-is.
        return message
