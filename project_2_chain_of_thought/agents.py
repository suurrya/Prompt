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
import sys

from dotenv import load_dotenv
from smolagents import ToolCallingAgent, OpenAIServerModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import ALL_TOOLS
from project_2_chain_of_thought.prompts import SYSTEM_PROMPT

load_dotenv()


class ITHelpdeskAgent:
    """
    Experiment 2: Static Chain-of-Thought Prompting.

    The system prompt contains examples that include an explicit Thought:
    step showing how to reason about tool selection before committing.
    """

    EXPERIMENT_NAME = "Static Chain-of-Thought"

    def __init__(self, model_id: str = "gpt-4o-mini", verbose: bool = False):
        model = OpenAIServerModel(
            model_id=model_id,
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self._agent = OpenAIServerModel(
            tools=ALL_TOOLS,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            verbosity_level=1 if verbose else 0,
        )
        self.verbose = verbose

    def __call__(self, user_query: str) -> str:
        """
        Process a user query.

        The static CoT prompt is used unchanged for every call — the only
        runtime variation comes from the model's internal reasoning trace,
        which the CoT examples in the prompt explicitly elicit.
        """
        if self.verbose:
            print(f"\n[{self.EXPERIMENT_NAME}] Query: {user_query}")
        return self._agent.run(user_query)
