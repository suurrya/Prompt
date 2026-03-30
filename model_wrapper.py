import re
import json
import uuid
from smolagents import OpenAIServerModel
from smolagents.models import ChatMessageToolCall, ChatMessageToolCallFunction

class HFRouterModel(OpenAIServerModel):
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
        return super().generate(
            messages, 
            stop_sequences=stop_sequences, 
            response_format=response_format, 
            tools_to_call_from=None, 
            **kwargs
        )

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
        
        # 0. Pre-process: Strip markdown code blocks if the entire content is wrapped in one
        # or if there's a block at the end.
        content = re.sub(r"```(?:python|json)?\s*(.*?)\s*```", r"\1", content, flags=re.DOTALL).strip()

        # 1. Handle Arrow/Action format: (→|Action:)\s*tool_name(arg="val")
        # Mandatory prefix to avoid matching prompt descriptions like "articles (TIER 1)"
        tool_calls = []
        pattern = r"(?:→|Action:)\s*(\w+)\s*\((.*?)\)"
        
        # We use re.finditer to handle sequential tool calls (e.g., security → escalate)
        for match in re.finditer(pattern, content, re.DOTALL):
            name = match.group(1).strip()
            args_str = match.group(2).strip()
            
            # Efficiently extract key="value" pairs from the arguments string
            # Handles: key="val", key='val', key=val
            args = {}
            arg_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s\)]+))'
            for k, v1, v2, v3 in re.findall(arg_pattern, args_str):
                args[k] = v1 or v2 or v3

            tool_calls.append(
                ChatMessageToolCall(
                    id=str(uuid.uuid4()), 
                    type="function", 
                    function=ChatMessageToolCallFunction(name=name, arguments=args)
                )
            )

        if tool_calls:
            message.tool_calls = tool_calls
            return message

        # 2. Handle JSON formats (fallback for Qwen action/parameters or OpenAI name/arguments)
        if '"action":' in content or '"name":' in content:
            try:
                # Extract the first JSON object from the text block
                j_match = re.search(r"(\{.*\})", content, re.DOTALL)
                if j_match:
                    data = json.loads(j_match.group(1))
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
