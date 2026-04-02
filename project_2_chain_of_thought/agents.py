"""
project_2_chain_of_thought/agents.py
======================================
Experiment 2 — Static Chain-of-Thought (CoT) Prompting

Hypothesis:
  Forcing a model to "think" (reason step-by-step) before selecting a tool
  significantly improves accuracy, especially for complex or multi-step
  IT queries where a simple pattern match might focus on the wrong detail.

Methodology:
  1. Identical to Experiment 1 (static system prompt set once at __init__).
  2. The key difference is that the prompt examples now include explicit 
     "Thought:" reasoning traces.
  3. This teaches the model to deliberate on problem classification (outage? 
     security? etc.) before committing to a tool call.

Pros: Higher reliability and better handling of ambiguous user intents.
Cons: Slightly higher token cost and latency due to generating the 
      reasoning trace before the tool call.
"""

import os # Purposes: Used to retrieve API keys from environment variables.
import re # Purposes: Crucial for Experiment 2 to extract the 'Thought:' trace from the model's response.
import sys # Purposes: Modifies the path so we can import utilities from the root directory.
import json # Purposes: Used to clean up and format the tool arguments for the final dossier.

from dotenv import load_dotenv # Purposes: Loads local configurations from the .env file.
from smolagents import ToolCallingAgent, AgentMaxStepsError # Purposes: The orchestration engine for our IT agent.

# Allow running from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Custom wrapper for the Hugging Face Inference Router
from model_wrapper import TextToolParserModel
from tool_extract import extract_tool_calls
from tools import ALL_TOOLS
from project_2_chain_of_thought.prompts import SYSTEM_PROMPT

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class ITHelpdeskAgent:
    """
    Experiment 2: Static Chain-of-Thought Prompting.

    The system prompt contains examples that include an explicit Thought:
    step showing how to reason about tool selection before committing.
    """

    EXPERIMENT_NAME = "Static Chain-of-Thought"

    # Section 1: Setup & Initialization
    def __init__(self, model_id: str = "meta/llama3-8b-instruct", verbose: bool = True):
        # Purposes: Stores the logging preference.
        self.verbose = verbose
        # Purposes: Creates the translator client that decodes text into tool calls.
        self._model = TextToolParserModel(
            model_id=model_id,
            api_base="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
        # Purposes: Initializes the worker agent configured to stop after 1 decision.
        self._agent = ToolCallingAgent(
            tools=ALL_TOOLS,
            model=self._model,
            max_steps=1, # Purposes: Prevents the agent from looping indefinitely.
            verbosity_level=1 if self.verbose else 0,
        )
        # Purposes: Replaces the default prompt with our Static CoT template (includes 'Thought:' steps).
        self._agent.prompt_templates["system_prompt"] = SYSTEM_PROMPT

    # Section 2: Execution Logic (The Call)
    def __call__(self, user_query: str) -> str:
        """
        Process a user query.
        """
        # Purposes: Logs the request for debugging in the terminal.
        if self.verbose:
            print(f"\n[{self.EXPERIMENT_NAME}] Query: {user_query}")
        
        try:
            # --- Step 4: Execute and get the tool calls from the result ---
            try:
                # Purposes: Runs the agent, which generates the 'Thought:' and 'Action:' text.
                self._agent.run(user_query)
            except AgentMaxStepsError:
                # Purposes: Catching the expected stop after 1 step.
                pass
            
            # --- Step 5: Extract the reasoning and build the dossier ---
            # Purposes: Initializes a variable to store the extracted reasoning trace.
            thought = ""
            # Purposes: Uses the centralized utility to find which tool the model chose in its memory.
            tool_calls = extract_tool_calls(self._agent.memory.steps)
            
            # Robustly find the last action step that contains reasoning and tool calls
            # Purposes: Scans backwards through memory messages to find where the model 'thought' about its action.
            for step in reversed(self._agent.memory.steps):
                if hasattr(step, "model_output_message") and step.model_output_message:
                    model_output = step.model_output_message.content
                    # Purposes: Uses Regex to grab everything between 'Thought:' and 'Action:'.
                    thought_match = re.search(r"Thought:(.*?)(?:Action:|$)", model_output, re.DOTALL)
                    if thought_match:
                        thought = thought_match.group(1).strip()
                        break

            # Purposes: Constructs the final summary dossier with separate sections for Reasoning and Decision.
            dossier = "### Chain of Thought\n\n"
            dossier += f"```markdown\n{thought}\n```\n\n" # Display the AI's internal deliberation.
            dossier += "### Decision\n\n"

            # Purposes: Warning if the model output text but missed calling a known tool.
            if not tool_calls:
                dossier += "⚠️ Decision: No tool was called."
            else:
                for call in tool_calls:
                    # Robustly handle different smolagents ToolCall formats
                    tool_name = getattr(call, "name", None) # Extracted tool name.
                    tool_args = getattr(call, "arguments", None) # Extracted parameters.
                    # Purposes: Checks function sub-objects if the base attributes are empty.
                    if not tool_name and hasattr(call, "function"):
                        tool_name = call.function.name
                        tool_args = call.function.arguments
                    
                    # Purposes: Cleans up arguments so they appear properly in the UI.
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}
                    elif not isinstance(tool_args, dict):
                        tool_args = {}
                        
                    # Purposes: Combines keys and values into a readable 'tool_call(arg="val")' string.
                    args_str = ", ".join(f'{k}="{v}"' for k, v in tool_args.items())
                    dossier += f"✅ Tool Call:\n`{tool_name}({args_str})`\n"

            # Purposes: Returns the full report to the evaluation script or the UI dashboard.
            return dossier

        except Exception as e:
            return f"❌ Error:\n`{str(e)}`"