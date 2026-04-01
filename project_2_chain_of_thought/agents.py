"""
project_2_chain_of_thought/agents.py
======================================
Experiment 2 — Static Chain-of-Thought (CoT) Prompting

Architecture is identical to Experiment 1 (a static system prompt set once
at __init__) — the only difference is that the system prompt now contains
examples with explicit "Thought:" reasoning traces, teaching the model to
deliberate before selecting a tool.

This additional reasoning step costs a few extra tokens but generally
improves accuracy for ambiguous queries where simple pattern-matching fails.
"""

import os
import re
import sys
import json

from dotenv import load_dotenv
from smolagents import ToolCallingAgent, AgentMaxStepsError

# Allow running from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Custom wrapper for the Hugging Face Inference Router
from model_wrapper import TextToolParserModel
from tool_extract import extract_tool_calls
from tools import ALL_TOOLS
from project_2_chain_of_thought.prompts import SYSTEM_PROMPT

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class ITHelpdeskAgent:
    """
    Experiment 2: Static Chain-of-Thought Prompting.

    The system prompt contains examples that include an explicit Thought:
    step showing how to reason about tool selection before committing.
    """

    EXPERIMENT_NAME = "Static Chain-of-Thought"

    def __init__(self, model_id: str = "meta/llama3-8b-instruct", verbose: bool = False):
        self.verbose = verbose
        self._model = TextToolParserModel(
            model_id=model_id,
            api_base="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
        self._agent = ToolCallingAgent(
            tools=ALL_TOOLS,
            model=self._model,
            max_steps=1,
            verbosity_level=1 if self.verbose else 0,
        )
        self._agent.prompt_templates["system_prompt"] = SYSTEM_PROMPT

    def __call__(self, user_query: str) -> str:
        """
        Process a user query.

        The static CoT prompt is used unchanged for every call — the only
        runtime variation comes from the model's internal reasoning trace,
        which the CoT examples in the prompt explicitly elicit.
        """
        if self.verbose:
            print(f"\n[{self.EXPERIMENT_NAME}] Query: {user_query}")
        
        try:
            # --- Step 4: Execute and get the tool calls from the result ---
            try:
                self._agent.run(user_query)
            except AgentMaxStepsError:
                pass
            
            # --- Step 5: Extract the reasoning and build the dossier ---
            thought = ""
            tool_calls = extract_tool_calls(self._agent.memory.steps)
            
            # Robustly find the last action step that contains reasoning and tool calls
            for step in reversed(self._agent.memory.steps):
                if hasattr(step, "model_output_message") and step.model_output_message:
                    model_output = step.model_output_message.content
                    thought_match = re.search(r"Thought:(.*?)(?:Action:|$)", model_output, re.DOTALL)
                    if thought_match:
                        thought = thought_match.group(1).strip()
                        break

            dossier = "### Chain of Thought\n\n"
            dossier += f"```markdown\n{thought}\n```\n\n"
            dossier += "### Decision\n\n"

            if not tool_calls:
                dossier += "⚠️ Decision: No tool was called."
            else:
                for call in tool_calls:
                    # Robustly handle different smolagents ToolCall formats
                    tool_name = getattr(call, "name", None)
                    tool_args = getattr(call, "arguments", None)
                    if not tool_name and hasattr(call, "function"):
                        tool_name = call.function.name
                        tool_args = call.function.arguments
                    
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}
                    elif not isinstance(tool_args, dict):
                        tool_args = {}
                        
                    args_str = ", ".join(f'{k}="{v}"' for k, v in tool_args.items())
                    dossier += f"✅ Tool Call:\n`{tool_name}({args_str})`\n"

            return dossier

        except Exception as e:
            return f"❌ Error:\n`{str(e)}`"