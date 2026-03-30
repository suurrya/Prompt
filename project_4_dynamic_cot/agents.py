"""
project_4_dynamic_cot/agents.py
=================================
Experiment 4 — Dynamic Chain-of-Thought (CoT) Prompting

The most advanced strategy: combines dynamic example selection (Experiment 3)
with chain-of-thought reasoning traces (Experiment 2).

For each incoming query:
  1. The TF-IDF selector finds the top-k CoT examples most similar to the query.
  2. A system prompt is assembled that presents those examples with their
     full Thought: → Action: reasoning traces.
  3. A fresh ToolCallingAgent is initialised with that just-in-time prompt.
  4. The agent runs and returns the result.

Hypothesis: by seeing examples that are BOTH contextually close AND
demonstrate deliberate reasoning, the model makes the fewest tool-selection
errors — especially on edge cases and ambiguous queries.
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
from project_4_dynamic_cot.prompts import build_system_prompt

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))



class ITHelpdeskAgent:
    """
    Experiment 4: Dynamic Chain-of-Thought Prompting.

    Overrides __call__() to build a new system prompt on every request,
    injecting the most semantically similar CoT examples from the database.
    This is the most sophisticated of the four agents.
    """

    EXPERIMENT_NAME = "Dynamic Chain-of-Thought"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        top_k_examples: int = 2,
        verbose: bool = False,
    ):
        self._model_id = model_id
        self._top_k = top_k_examples
        self._api_key = os.environ["HUGGING_FACE_API_KEY"]
        self.verbose = verbose

        self._model = HFRouterModel(
            model_id=model_id,
            api_base="https://router.huggingface.co/v1",
            api_key=self._api_key,
        )

    # ------------------------------------------------------------------
    # Overridden __call__ — dynamic CoT prompt on every request
    # ------------------------------------------------------------------

    def __call__(self, user_query: str) -> str:
        """
        For each query:
          1. Select the most relevant CoT examples via TF-IDF similarity.
          2. Build a system prompt with those examples including Thought: traces.
          3. Initialise a ToolCallingAgent with the tailored prompt.
          4. Run and return the result.
        """
        # Step 1 & 2: Dynamic CoT prompt construction
        dynamic_prompt = build_system_prompt(user_query, top_k=self._top_k)

        if self.verbose:
            print(f"\n[{self.EXPERIMENT_NAME}] Query: {user_query}")
            print(f"[{self.EXPERIMENT_NAME}] Prompt chars: {len(dynamic_prompt)}")

        # Step 3: Fresh agent with the tailored CoT prompt
        agent = ToolCallingAgent(
            tools=ALL_TOOLS,
            model=self._model,
            max_steps=2,
            verbosity_level=1 if self.verbose else 0,
        )
        agent.prompt_templates["system_prompt"] = dynamic_prompt
        self._last_agent = agent  # expose for log inspection by the evaluator

        # Step 4: Execute
        return agent.run(user_query)
