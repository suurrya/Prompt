"""
project_3_dynamic_few_shot/agents.py
======================================
Experiment 3 — Dynamic Few-Shot Prompting

Hypothesis:
  Instead of a fixed set of examples, selecting only the most 
  semantically relevant matches for each query (using TF-IDF similarity) 
  will provide the model with a better "mental model" of the 
  requested task, leading to higher accuracy and lower noise.

Methodology:
  1. For EACH incoming query, it performs a search against a database 
     of hundreds of example (query → tool_call) pairs.
  2. It selects the Top-K (default 4) most similar matches.
  3. It "just-in-time" constructs a tailored system prompt.
  4. It initializes a fresh agent with this custom-built prompt.

Pros: Higher contextual relevance; can support a nearly infinite 
      vocalbulary of tools/scenarios by just adding more examples.
Cons: Requires a vector store or similarity engine (TF-IDF here) and 
      adds a small compute overhead to every query.
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
from .prompts import build_system_prompt, select_examples

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))



class ITHelpdeskAgent:
    """
    Experiment 3: Dynamic Few-Shot Prompting.

    Overrides __call__() to rebuild the system prompt from scratch for
    every user query, selecting the most contextually relevant examples
    via TF-IDF(Term Frequency–Inverse Document Frequency) cosine similarity before each LLM call.
    """

    EXPERIMENT_NAME = "Dynamic Few-Shot"

    def __init__(
        self,
        model_id: str = "meta/llama3-8b-instruct",
        top_k_examples: int = 4,
        verbose: bool = False,
    ):

        self._model_id = model_id
        self._top_k = top_k_examples
        self._api_key = os.environ["NVIDIA_API_KEY"]
        self.verbose = verbose

        # We build the model once; only the system_prompt changes per call.
        self._model = TextToolParserModel(
            model_id=model_id,
            api_base="https://integrate.api.nvidia.com/v1",
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
        if self.verbose:
            print(f"\n[{self.EXPERIMENT_NAME}] Query: {user_query}")

        try:
            # --- Step 1: Get the dynamically selected examples for the dossier ---
            selected = select_examples(user_query, top_k=self._top_k)

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
            self._last_agent = agent  # Expose for evaluator

            # --- Step 4: Execute and get the tool calls from the result ---
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

            # --- Step 6: Build the Pretty-Printed Dossier ---
            dossier = "### Dynamic Few-Shot Selection\n\n"
            dossier += "Examples selected for this query:\n"
            for ex in selected:
                # Show a truncated version of the query to keep it clean
                dossier += f"- `{ex['query'][:60].strip()}...`\n"
            dossier += "\n"
            
            dossier += "### Decision\n\n"
            if thought:
                dossier += f"Thought: {thought}\n\n"
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