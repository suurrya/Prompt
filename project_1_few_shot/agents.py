"""
project_1_few_shot/agents.py
=============================
Experiment 1 — Static Few-Shot Prompting

The ITHelpdeskAgent here is the simplest possible implementation:
  • Build the system prompt once at class instantiation from the static string
    defined in prompts.py.
  • Pass it unchanged to ToolCallingAgent.
  • Every user query is processed with exactly the same set of examples.

No runtime logic, no dynamic selection. The model must generalize from the
fixed examples in SYSTEM_PROMPT regardless of how similar they are to the
actual incoming query.
"""
import os
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
from project_1_few_shot.prompts import SYSTEM_PROMPT

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class ITHelpdeskAgent:
    """
    Experiment 1: Static Few-Shot Prompting.

    The system prompt contains a hand-crafted, fixed set of
    (User query → Tool call) examples that never change.
    """

    EXPERIMENT_NAME = "Static Few-Shot"

    def __init__(self, model_id: str = "meta/llama3-8b-instruct", verbose: bool = False):
        self._model = TextToolParserModel(
            model_id=model_id,
            api_base="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
        self._agent = ToolCallingAgent(
                tools=ALL_TOOLS,
                model=self._model,
                max_steps=1,
                verbosity_level=1 if verbose else 0,
            )
        # Override only the system_prompt key so the other required keys
        # (planning, managed_agent, final_answer, …) stay at their defaults.
        self._agent.prompt_templates["system_prompt"] = SYSTEM_PROMPT
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public interface — identical signature across all four experiments
    # ------------------------------------------------------------------

    def __call__(self, user_query: str) -> str:
        """
        Process a user query and return the agent's final response.

        The system prompt is static — it was set once at __init__ and
        does not change between calls.
        """
        if self.verbose:
            print(f"\n[{self.EXPERIMENT_NAME}] Query: {user_query}")

        try:
            # We still need to get the raw tool calls from the agent's plan
            try:
                self._agent.run(user_query)
            except AgentMaxStepsError:
                pass
            
            # Robustly find the last action step that contains tool calls
            tool_calls = extract_tool_calls(self._agent.memory.steps)

            if not tool_calls:
                return "⚠️ Decision: No tool was called."

            # Format the output
            dossier = "### Decision\n\n"
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