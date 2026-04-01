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
import json

from dotenv import load_dotenv
from smolagents import ToolCallingAgent, AgentMaxStepsError

# Allow running from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Custom wrapper for the Hugging Face Inference Router
import re
from model_wrapper import HFRouterModel
from tool_extract import extract_tool_calls
from tools import ALL_TOOLS
from .prompts import build_system_prompt, select_cot_examples

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
        model_id: str = "meta/llama3-8b-instruct",
        top_k_examples: int = 2,
        verbose: bool = False,
    ):
        self._model_id = model_id
        self._top_k = top_k_examples
        self._api_key = os.environ["NVIDIA_API_KEY"]
        self.verbose = verbose

        self._model = HFRouterModel(
            model_id=model_id,
            api_base="https://integrate.api.nvidia.com/v1",
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
        if self.verbose:
            print(f"\n[{self.EXPERIMENT_NAME}] Query: {user_query}")

        try:
            # --- Step 1: Get the dynamically selected examples for the dossier ---
            selected = select_cot_examples(user_query, top_k=self._top_k)

            # --- Step 2: Build the dynamic prompt (same as your original code) ---
            dynamic_prompt = build_system_prompt(user_query, top_k=self._top_k)

            # --- Step 3: Instantiate the agent (same as your original code) ---
            agent = ToolCallingAgent(
                tools=ALL_TOOLS,
                model=self._model,
                max_steps=1,
                verbosity_level=1 if self.verbose else 0,
            )
            agent.prompt_templates["system_prompt"] = dynamic_prompt
            self._last_agent = agent # Expose for evaluator

            # --- Step 4: Execute and get both the text response and the tool calls ---
            try:
                agent.run(user_query)
            except AgentMaxStepsError:
                pass
            
            # --- Step 5: Extract the reasoning and build the dossier ---
            thought = ""
            tool_calls = extract_tool_calls(self._last_agent.memory.steps)
            
            # Robustly find the last action step that contains reasoning and tool calls
            for step in reversed(self._last_agent.memory.steps):
                if hasattr(step, "model_output_message") and step.model_output_message:
                    model_output = step.model_output_message.content
                    thought_match = re.search(r"Thought:(.*?)(?:Action:|$)", model_output, re.DOTALL)
                    if thought_match:
                        thought = thought_match.group(1).strip()
                        break
            
            dossier = "### Dynamic CoT Selection\n\n"
            dossier += "CoT Examples selected for this query:\n"
            for ex in selected:
                dossier += f"- `{ex['query'][:60].strip()}...`\n"
            dossier += "\n"

            dossier += "### Final Chain of Thought\n\n"
            dossier += f"```markdown\n{thought}\n```\n\n"

            dossier += "### Decision\n\n"
            if not tool_calls:
                dossier += "⚠️ Decision: No tool was called."
            else:
                import json
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