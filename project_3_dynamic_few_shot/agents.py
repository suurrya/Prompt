"""
project_3_dynamic_few_shot/agents.py
======================================
Experiment 3 — Dynamic Few-Shot Prompting

The key architectural difference from Experiments 1 & 2:
  • __call__() is overridden.
  • For EACH incoming query it:
      1. Calls build_system_prompt(user_query) to perform a TF-IDF similarity
         search and construct a tailored prompt.
      2. Re-initialises the underlying ToolCallingAgent with that new prompt.
      3. Runs the agent.

This "just-in-time" prompting means the examples the model sees are always
the ones most semantically similar to the actual problem — not a fixed set
that may be irrelevant.

Trade-off: a small runtime overhead per call (vectoriser transform + agent
re-init) in exchange for meaningfully better example relevance.
"""

import os
import sys

from dotenv import load_dotenv
from smolagents import ToolCallingAgent, OpenAIServerModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import ALL_TOOLS
from project_3_dynamic_few_shot.prompts import build_system_prompt

load_dotenv()


class ITHelpdeskAgent:
    """
    Experiment 3: Dynamic Few-Shot Prompting.

    Overrides __call__() to rebuild the system prompt from scratch for
    every user query, selecting the most contextually relevant examples
    via TF-IDF cosine similarity before each LLM call.
    """

    EXPERIMENT_NAME = "Dynamic Few-Shot"

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        top_k_examples: int = 4,
        verbose: bool = False,
    ):
        self._model_id = model_id
        self._top_k = top_k_examples
        self._api_key = os.environ["OPENAI_API_KEY"]
        self.verbose = verbose

        # We build the model once; only the system_prompt changes per call.
        self._model = OpenAIServerModel(
            model_id=model_id,
            api_key=self._api_key,
        )

    # ------------------------------------------------------------------
    # Overridden __call__ — the core of the dynamic-prompting strategy
    # ------------------------------------------------------------------

    def __call__(self, user_query: str) -> str:
        """
        For each incoming query:
          1. Dynamically select the top-k most relevant few-shot examples.
          2. Build a tailored system prompt embedding those examples.
          3. Instantiate a fresh ToolCallingAgent with that prompt.
          4. Run the agent and return the result.
        """
        # Step 1 & 2: Dynamic prompt construction
        dynamic_prompt = build_system_prompt(user_query, top_k=self._top_k)

        if self.verbose:
            print(f"\n[{self.EXPERIMENT_NAME}] Query: {user_query}")
            print(f"[{self.EXPERIMENT_NAME}] System prompt length: {len(dynamic_prompt)} chars")

        # Step 3: Re-initialise agent with the tailored prompt
        agent = OpenAIServerModel(
            tools=ALL_TOOLS,
            model=self._model,
            system_prompt=dynamic_prompt,
            verbosity_level=1 if self.verbose else 0,
        )

        # Step 4: Execute
        return agent.run(user_query)
