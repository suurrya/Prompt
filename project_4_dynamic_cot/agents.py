"""
project_4_dynamic_cot/agents.py
=================================
Experiment 4 — Dynamic Chain-of-Thought (CoT) Prompting

Hypothesis:
  Combining dynamic example selection (Experiment 3) with full 
  Chain-of-Thought reasoning traces (Experiment 2) creates the 
  ultimate "Expert System." The model sees contextually relevant 
  matches AND is taught to deliberate, resulting in the highest accuracy 
  and fewset hallucinations for IT helpdesk tasks.

Methodology:
  1. For each incoming query, a TF-IDF selector finds the top-k 
     CoT-annotated examples (usually 2 to avoid context bloat).
  2. A system prompt is assembled presenting these examples with their 
     full Reasoning → Action traces.
  3. A fresh ToolCallingAgent is initialized with this "just-in-time" prompt.
  4. The model reasons through its own diagnostic framework before selecting a tool.

Pros: Highest reliability; best generalization to "novel" edge cases.
Cons: Highest token consumption and highest latency of all four strategies.
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
from model_wrapper import TextToolParserModel
from tool_extract import extract_tool_calls
from tools import ALL_TOOLS
from .prompts import build_system_prompt, select_cot_examples

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))



# Section 1: Setup
# This part sets up the agent's initial state and configuration when it's first created.

# Defines a new blueprint for an object called ITHelpdeskAgent.
class ITHelpdeskAgent:
    """
    Experiment 4: Dynamic Chain-of-Thought Prompting.

    Overrides __call__() to build a new system prompt on every request,
    injecting the most semantically similar CoT examples from the database.
    This is the most sophisticated of the four agents.
    """

    # A class-level variable that gives a name to this agent's strategy.
    EXPERIMENT_NAME = "Dynamic Chain-of-Thought"

    # The constructor method. This code runs automatically once when you create an instance of the agent 
    # (e.g., my_agent = ITHelpdeskAgent()).
    # Parameters: model_id, top_k_examples, verbose are configurations you can set.
    def __init__(self,
        model_id: str = "meta/llama3-8b-instruct",
        top_k_examples: int = 2,
        verbose: bool = False,
    ):
        # Stores the name of the AI model to be used (e.g., "llama3-8b-instruct").
        self._model_id = model_id
        # Stores the number of examples to fetch for the dynamic prompt.
        self._top_k = top_k_examples
        # Securely retrieves the API key needed to communicate with the AI model from the computer's environment variables.
        self.api_key = os.environ["NVIDIA_API_KEY"]
        # A flag to control whether the agent prints detailed step-by-step information as it runs.
        self.verbose = verbose

        # This is a key line. It creates an instance of another object, likely from a library like smolagents. 
        # This _model object is the low-level client responsible for actually making API calls to the AI service (like NVIDIA's API).
        self._model = TextToolParserModel(
            model_id=model_id,
            api_base="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key,
        )

    # ------------------------------------------------------------------
    # Overridden __call__ — dynamic CoT prompt on every request
    # ------------------------------------------------------------------

    # Section 2: __call__ Method (The Main Logic)
    # This special method makes the object behave like a function. 
    # When you have an instance my_agent, you can "call" it directly like my_agent("some user query"), and this code will run.
    # Defines the main entry point for processing a user query. It takes the user_query and is expected to return a single string (the dossier).
    def __call__(self, user_query: str) -> str:
        """
        For each query:
          1. Select the most relevant CoT examples via TF-IDF(Term Frequency–Inverse Document Frequency) similarity.
          2. Build a system prompt with those examples including Thought: traces.
          3. Initialise a ToolCallingAgent with the tailored prompt.
          4. Run and return the result.
        """
        # Step 1 & 2: Dynamic CoT prompt construction
        # A simple check. If verbose was set to True during initialization, it prints the incoming query.
        if self.verbose:
            print(f"\n[{self.EXPERIMENT_NAME}] Query: {user_query}")

        # This is a crucial error-handling block. The code inside try: is executed. 
        # If any error occurs during the process, the program doesn't crash. 
        # Instead, it jumps to the except block, formats the error message, and returns that.
        try:
            # --- Step 1: Get the dynamically selected examples for the dossier ---
            # Function: Calls the select_examples function (or a variant of it).
            # Purpose: To find the top_k most semantically similar examples from the database based on the user_query.
            # Variable (selected): Contains a list of dictionaries, where each dictionary is a complete example (query, thought, tool_call).
            selected = select_cot_examples(user_query, top_k=self._top_k)

            # --- Step 2: Build the dynamic prompt (same as your original code) ---
            # Function: Calls the build_system_prompt function.
            # Purpose: To take the selected examples and construct the full, formatted system prompt string.
            # Variable (dynamic_prompt): Contains the final, single string of the system prompt, ready to be used by the AI model.
            dynamic_prompt = build_system_prompt(user_query, top_k=self._top_k)

            # --- Step 3: Instantiate the agent (same as your original code) ---
            # What it does: Creates an instance of another agent, likely a lower-level agent from a library like smolagents that knows how to follow instructions and call tools.
            # Purpose: This ToolCallingAgent is the "worker bee." You give it a prompt and a set of tools, and it does the actual thinking.
            # Variable (agent): An object ready to run, but its default prompt hasn't been replaced yet.
            agent = ToolCallingAgent(
                tools=ALL_TOOLS,
                model=self._model,
                max_steps=1,
                verbosity_level=1 if self.verbose else 0,
            )
            # What it does: This is a critical step. It reaches into the ToolCallingAgent and replaces its default system prompt with the dynamic_prompt we just built. 
            # This "tunes" the worker agent for this specific query.
            agent.prompt_templates["system_prompt"] = dynamic_prompt
            # What it does: Saves a reference to this fully configured ToolCallingAgent.
            # Purpose: To allow us to inspect its internal memory after it has run to see what it was thinking.
            self._last_agent = agent # Expose for evaluator

            # --- Step 4: Execute and get both the text response and the tool calls ---
            try:
                # Function: This is the main execution command. It tells the ToolCallingAgent: 
                # "Here is the user's query. Use the prompt I gave you, think step-by-step, and call a tool if necessary."
                # Behind the scenes: This triggers the API call to the LLM. 
                agent.run(user_query)
            except AgentMaxStepsError:
                pass
            
            # --- Step 5: Extract the reasoning and build the dossier ---
            # Purpose: After the agent has run, these lines prepare to extract the results from its memory.
            # Variables: thought is an empty string to be filled, and tool_calls will contain a list of the tool calls made.
            thought = ""
            # extract_tool_calls Function: A helper function that sifts through the agent's memory (self._last_agent.memory.steps) to find what tool(s) were actually called.
            tool_calls = extract_tool_calls(self._last_agent.memory.steps)
            
            # Robustly find the last action step that contains reasoning and tool calls
            # Purpose: To find the AI's reasoning (Thought:). It cleverly searches backwards through the agent's memory steps 
            # to find the last thing the model "thought" before it acted. 
            for step in reversed(self._last_agent.memory.steps):
                if hasattr(step, "model_output_message") and step.model_output_message:
                    model_output = step.model_output_message.content
                    # It uses re.search (a regular expression) to reliably extract the text that comes after "Thought:".
                    thought_match = re.search(r"Thought:(.*?)(?:Action:|$)", model_output, re.DOTALL)
                    if thought_match:
                        thought = thought_match.group(1).strip()
                        break
            
            # Purpose: To build a detailed, human-readable report string about the entire process.
            # dossier: A multi-line string that is constructed piece by piece, including which examples were selected, the final chain of thought from the AI, and the final decision.
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

            # The final, formatted report string is returned as the output of the ITHelpdeskAgent's call.
            return dossier

        except Exception as e:
            return f"❌ Error:\n`{str(e)}`"