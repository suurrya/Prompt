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

from dotenv import load_dotenv
from smolagents import ToolCallingAgent

# Allow running from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Custom wrapper for the Hugging Face Inference Router
from model_wrapper import HFRouterModel
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

    def __init__(self, model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct", verbose: bool = False):
        model = HFRouterModel(
            model_id=model_id,
            api_base="https://router.huggingface.co/v1",
            api_key=os.environ["HUGGING_FACE_API_KEY"],
        )
        self._agent = ToolCallingAgent(
            tools=ALL_TOOLS,
            model=model,
            max_steps=2,
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
        return self._agent.run(user_query)
