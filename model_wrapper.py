import re
import json
import uuid
from smolagents import OpenAIServerModel
from smolagents.models import ChatMessageToolCall, ChatMessageToolCallFunction

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
    
    # Explicitly indicate that this model doesn't support native OpenAI-style tool calls
    supports_native_tools = False

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        """
        Overrides the standard generate which strips tools_to_call_from.
        This ensures that 'tools' and 'tool_choice' are NOT sent to the HF Router API.
        """
        kwargs.pop('tools_to_call_from', None)
        
        max_retries = 3
        base_delay_seconds = 2 

        for attempt in range(max_retries):
            try:
                # Attempt to make the API call
                return super().generate(messages, **kwargs)
            
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for known, temporary server-side errors
                is_transient_error = (
                    ("degraded" in error_str and "cannot be invoked" in error_str) or
                    ("500" in error_str and "internal server error" in error_str)
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
        
        # This part should ideally not be reached, but it's a safeguard.
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
        
        # 0. Debug Log
        try:
            with open("debug_model_output.txt", "a") as f:
                f.write(f"\n--- MODEL OUTPUT ---\n{content}\n")
        except:
            pass
            
        # 0. Pre-process: Strip markdown code blocks if the entire content is wrapped in one
        # or if there's a block at the end.
        content = re.sub(r"```(?:python|json)?\s*(.*?)\s*```", r"\1", content, flags=re.DOTALL).strip()        # 1. Handle Arrow/Action format or raw tool call: (?:→|Action:)?\s*tool_name(arg="val")
        # We enforce that the tool name matches one of our known valid tools to avoid matching random text
        valid_tools = {
            "lookup_knowledge_base", "create_ticket", "escalate_ticket", "reset_password",
            "get_user_info", "lookup_user_account", "check_system_status", "schedule_maintenance",
            "process_refund", "store_resolved_ticket", "save_ticket_to_long_term_memory",
            "get_user_long_term_memory", "get_customer_history"
        }
        
        tool_calls = []
        
        # This regex is the "Brain" of our manual tool parsing.
        # It looks for patterns like: Action: create_ticket(priority="high")
        # It handles optional prefixes (→ or Action:) and extracts the tool name and its arguments.
        tool_pattern = re.compile(r"""
            (?:→\s*|Action:\s*)?    # Optional prefix like '→ ' or 'Action: '
            ([a-zA-Z_]\w*)          # Capture Group 1: The tool_name (must start with letter/underscore)
            \s*\(                   # Opening parenthesis
            (                       # Capture Group 2: The raw arguments string
                .*?                 # Non-greedy match for everything inside
            )
            \)                      # Closing parenthesis
        """, re.VERBOSE | re.DOTALL)
        
        for match in tool_pattern.finditer(content):
            name = match.group(1).strip()
            
            # Skip if it's not a valid tool name
            if name not in valid_tools:
                continue
                
            args_str = match.group(2).strip()
            
            # Efficiently extract key="value" pairs from the arguments string
            args = {}
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
            
            for k, v1, v2, v3 in arg_pattern.findall(args_str):
                args[k] = v1 or v2 or v3

            # Fallback: if no key=value was found, but there is a string, it might be a positional arg.
            # Map it to the known single parameter for single-argument tools.
            if not args and args_str.strip():
                clean_arg = args_str.strip(' \'"')
                # These mappings help beginners understand that even if the AI forgets
                # the "key=" part, we can still recover the intent for simple tools.
                parameter_mapping = {
                    "check_system_status": "service_name",
                    "lookup_knowledge_base": "query",
                    "get_user_info": "user_email",
                    "lookup_user_account": "email",
                    "get_user_long_term_memory": "user_id",
                    "get_customer_history": "user_id"
                }
                if name in parameter_mapping:
                    args[parameter_mapping[name]] = clean_arg

            tool_calls.append(
                ChatMessageToolCall(
                    id=str(uuid.uuid4()), 
                    type="function", 
                    function=ChatMessageToolCallFunction(name=name, arguments=args)
                )
            )
            # Enforce single-step execution (max_steps=1).
            # The LLM may output multiple tool calls in a row (e.g. 1. check_status(...) 2. create_ticket(...)).
            # We ONLY want the first one so the agent executes it, observes the result, then decides the next step.
            break

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
