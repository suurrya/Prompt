"""
project_1_few_shot/agents.py
=============================
Experiment 1 — Static Few-Shot Prompting

Hypothesis: 
  A simple, fixed set of high-quality examples is sufficient for a 
  large model (like Llama 3) to generalize and solve basic IT 
  helpdesk queries accurately without needing complex reasoning.

Methodology:
  1. Build the system prompt once at class instantiation from the static string 
     defined in prompts.py.
  2. Pass it unchanged to ToolCallingAgent.
  3. Every user query is processed with exactly the same set of examples.

Pros: Extremely fast (lowest latency), predictable, and easy to maintain.
Cons: Limited by the "context window" (cannot include too many examples) 
      and may fail on edge cases not covered by the fixed example set.
"""
import os # Purposes: Used to access environment variables like API keys safely.
import sys # Purposes: Modifies the system path so we can import modules from the parent directory.
import json # Purposes: Required to parse tool arguments from the model's text response.

from dotenv import load_dotenv # Purposes: Loads secret keys from a .env file into the environment.
from smolagents import ToolCallingAgent, AgentMaxStepsError # Purposes: The core engine for calling tools and handling step limits.

# Allow running from any working directory.
# Purposes: Ensures that the 'tools' and 'model_wrapper' modules can be found by Python.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Custom wrapper for the Hugging Face Inference Router
# Purposes: Imports our specialized model client and extraction utilities.
from model_wrapper import TextToolParserModel
from tool_extract import extract_tool_calls
from tools import ALL_TOOLS
from project_1_few_shot.prompts import SYSTEM_PROMPT

# Purposes: Boots the environment settings for the entire script.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class ITHelpdeskAgent:
    """
    Experiment 1: Static Few-Shot Prompting.

    The system prompt contains a hand-crafted, fixed set of
    (User query → Tool call) examples that never change.
    """

    EXPERIMENT_NAME = "Static Few-Shot"

    # Section 1: Initialization
    def __init__(self, model_id: str = "meta/llama3-8b-instruct", verbose: bool = False):
        # Purposes: Creates the 'Translator' client that handles API calls and text-to-tool parsing.
        self._model = TextToolParserModel(
            model_id=model_id,
            api_base="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
        # Purposes: Initializes the agent with the model and tools, and sets a strict step limit to enforce a single decision per query.
        self._agent = ToolCallingAgent(
                tools=ALL_TOOLS,
                model=self._model,
                max_steps=1, # Purposes: Enforces a single decision per query for predictability.
                verbosity_level=1 if verbose else 0,
            )
        # Override only the system_prompt key so the other required keys
        # (planning, managed_agent, final_answer, …) stay at their defaults.
        # Purposes: This is the "Static Injection". We force the agent to use our hand-picked examples.
        self._agent.prompt_templates["system_prompt"] = SYSTEM_PROMPT
        # Purposes: Stores the logging preference for the __call__ method.
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public interface — identical signature across all four experiments
    # ------------------------------------------------------------------

    # Section 2: Execution Logic
    def __call__(self, user_query: str) -> str:
        """
        Process a user query and return the agent's final response.
        """
        # Purposes: Logs the incoming user request if 'verbose' mode is active.
        if self.verbose:
            print(f"\n[{self.EXPERIMENT_NAME}] Query: {user_query}")

        try:
            # We still need to get the raw tool calls from the agent's plan
            try:
                # Purposes: Triggers the model to generate a response based on the static prompt.
                self._agent.run(user_query)
            except AgentMaxStepsError:
                # Purposes: We expect this! max_steps=1 forces the agent to stop after one decision.
                pass
            
            # Robustly find the last action step that contains tool calls
            # Purposes: Scans the agent's short-term memory to extract the tool it decided to use.
            tool_calls = extract_tool_calls(self._agent.memory.steps)

            # Purposes: If no valid tool command was found, return a warning instead of a crash.
            if not tool_calls:
                return "⚠️ Decision: No tool was called."

            # Format the output
            # Purposes: Builds the 'Dossier' report displayed in the User Interface.
            dossier = "### Decision\n\n"
            for call in tool_calls:
                # Robustly handle different smolagents ToolCall formats
                tool_name = getattr(call, "name", None) # Purpose: Extracts the tool name (e.g., 'check_system_status').
                tool_args = getattr(call, "arguments", None) # Purpose: Extracts the parameters (e.g., {'service_name': 'vpn'}).
                # Purposes: Final checks for older or structured framework objects.
                if not tool_name and hasattr(call, "function"):
                    tool_name = call.function.name
                    tool_args = call.function.arguments
                
                # Purposes: Ensures that arguments are in a clean dictionary format for display.
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}
                elif not isinstance(tool_args, dict):
                    tool_args = {}
                    
                # Purposes: Joins the key="value" pairs into a pretty string for the report.
                args_str = ", ".join(f'{k}="{v}"' for k, v in tool_args.items())
                dossier += f"✅ Tool Call:\n`{tool_name}({args_str})`\n"
            # Purposes: Sends the final report back to the UI controller.
            return dossier

        except Exception as e:
            return f"❌ Error:\n`{str(e)}`"