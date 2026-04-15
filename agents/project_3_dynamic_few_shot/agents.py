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

import os # Purposes: Used to retrieve secure API credentials for NVIDIA.
import re # Purposes: Scans the model's textual response for reasoning and action blocks.
import sys # Purposes: Adjusts the path so we can import 'tools' and 'model_wrapper' from root.
import json # Purposes: Standard library for cleaning and formatting tool arguments.

from dotenv import load_dotenv # Purposes: Loads project-wide environment variables from .env.
from smolagents import ToolCallingAgent, AgentMaxStepsError # Purposes: The worker engine that executes the final plan.

# Allow running from any working directory.
# Purposes: Ensures local package imports (tools, models) work regardless of start folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Custom wrapper for the Hugging Face Inference Router
# Purposes: Imports the compatibility wrapper and the retrieval logic for prompts.
from core.model_wrapper import TextToolParserModel
from core.tool_extract import extract_tool_calls
from core.tools import ALL_TOOLS
from .prompts import build_system_prompt, select_examples

# Purposes: Initializes the backend environment.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))



class ITHelpdeskAgent:
    """
    Experiment 3: Dynamic Few-Shot Prompting.

    Overrides __call__() to rebuild the system prompt from scratch for
    every user query, selecting the most contextually relevant examples
    via TF-IDF(Term Frequency–Inverse Document Frequency) cosine similarity before each LLM call.
    """

    EXPERIMENT_NAME = "Dynamic Few-Shot"

    # Section 1: Setup & Initialization
    def __init__(
        self,
        model_id: str = "meta/llama-3.1-8b-instruct",
        top_k_examples: int = 4, # Purpose: Controls the number of retrieved examples (default 4).
        verbose: bool = False, # Purpose: Controls detailed terminal logging.
    ):

        # Purposes: Stores the model path (e.g., Llama 3 8B).
        self._model_id = model_id
        # Purposes: Stores the 'K' parameter for our nearest-neighbor search.
        self._top_k = top_k_examples
        # Purposes: Securely pulls the API key from the system environment.
        self._api_key = os.environ["NVIDIA_API_KEY"]
        self.verbose = verbose

        # We build the model once; only the system_prompt changes per call.
        # Purposes: Initializes the 'Translator' client.
        self._model = TextToolParserModel(
            model_id=model_id,
            api_base="https://integrate.api.nvidia.com/v1",
            api_key=self._api_key,
            max_tokens=256,    # Tight cap — Exp 3 only needs a bare tool call (~10 tokens)
            temperature=0,     # Greedy decoding: no sampling overhead, fully deterministic
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
            # Purposes: Performs a TF-IDF search to find the 4 examples most similar to the user's query.
            selected = select_examples(user_query, top_k=self._top_k)

            # --- Step 2: Build the dynamic prompt (same as your original code) ---
            # Purposes: Constructs a new system prompt 'Just-in-Time' using the selected examples.
            dynamic_prompt = build_system_prompt(user_query, top_k=self._top_k)

            # --- Step 3: Instantiate the agent (same as your original code) ---
            # Purposes: Creates a fresh worker agent for this specific query (Late Binding).
            agent = ToolCallingAgent(
                tools=ALL_TOOLS,
                model=self._model,
                max_steps=1, # Purposes: Enforces a single-decision pass.
                verbosity_level=1 if self.verbose else 0,
            )
            # Purposes: Injects the dynamically generated prompt into the worker.
            agent.prompt_templates["system_prompt"] = dynamic_prompt
            self._last_agent = agent  # Expose for evaluator

            # --- Step 4: Execute and get the tool calls from the result ---
            try:
                # Purposes: Triggers the model to generate the tool call using the custom prompt.
                agent.run(user_query)
            except AgentMaxStepsError:
                # Purposes: Gracefully exits after the mandatory 1-step limit.
                pass
            
            # --- Step 5: Extract the reasoning and build the dossier ---
            # Purposes: Initializes a variable to store extracted reasoning.
            thought = ""
            # Purposes: Scans the agent's memory to find which tool it chose.
            tool_calls = extract_tool_calls(self._last_agent.memory.steps)
            
            # Robustly find the last action step that contains reasoning and tool calls
            for step in reversed(self._last_agent.memory.steps):
                if hasattr(step, "model_output_message") and step.model_output_message:
                    model_output = step.model_output_message.content
                    # Purposes: Extracts 'Thought:' string from the model output.
                    thought_match = re.search(r"Thought:(.*?)(?:Action:|$)", model_output, re.DOTALL)
                    if thought_match:
                        thought = thought_match.group(1).strip()
                        break

            # --- Step 6: Build the Pretty-Printed Dossier ---
            # Purposes: Includes a section showing which few-shot examples were 'Retrieved' for this query.
            dossier = "### Dynamic Few-Shot Selection\n\n"
            dossier += "Examples selected for this query:\n"
            for ex in selected:
                # Show a truncated version of the query to keep it clean
                dossier += f"- `{ex['query'][:60].strip()}...`\n"
            dossier += "\n"
            
            dossier += "### Decision\n\n"
            if thought:
                dossier += f"Thought: {thought}\n\n"
            # Purposes: Warning if no tool was found in the text response.
            if not tool_calls:
                dossier += "⚠️ Decision: No tool was called."
            else:
                for call in tool_calls:
                    # Robustly handle different smolagents ToolCall formats
                    tool_name = getattr(call, "name", None) # Extracted tool name.
                    tool_args = getattr(call, "arguments", None) # Extracted parameters.
                    # Purposes: Checks sub-objects (function/name) if the base attributes are missing.
                    if not tool_name and hasattr(call, "function"):
                        tool_name = call.function.name
                        tool_args = call.function.arguments
                    
                    # Purposes: Clears decoding errors if the model output malformed JSON strings.
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}
                    elif not isinstance(tool_args, dict) or tool_args is None:
                        tool_args = {}
                        
                    # Purposes: Formats the call into the standardized report style for the UI.
                    args_str = ", ".join(f'{k}="{v}"' for k, v in tool_args.items())
                    dossier += f"✅ Tool Call:\n`{tool_name}({args_str})`\n"
        
            # Purposes: Returns the dynamically constructed report to the dashboard.
            return dossier

        except Exception as e:
            return f"❌ Error:\n`{str(e)}`"