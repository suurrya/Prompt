import os
import sys
import json
import re

# Add the current directory to sys.path so we can import projects
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Explicitly load .env from the root of the project
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# Ensure API key is set
if not os.environ.get("NVIDIA_API_KEY"):
    print("[ERROR] NVIDIA_API_KEY not found. Please check your .env file.")
    sys.exit(1)

from project_1_few_shot.agents import ITHelpdeskAgent as Agent1
from project_2_chain_of_thought.agents import ITHelpdeskAgent as Agent2
from project_3_dynamic_few_shot.agents import ITHelpdeskAgent as Agent3
from project_4_dynamic_cot.agents import ITHelpdeskAgent as Agent4
from tool_extract import extract_tool_calls

EXPERIMENTS = [
    {"id": 1, "class": Agent1, "name": "Static Few-Shot"},
    {"id": 2, "class": Agent2, "name": "Static CoT"},
    {"id": 3, "class": Agent3, "name": "Dynamic Few-Shot"},
    {"id": 4, "class": Agent4, "name": "Dynamic CoT"},
]

def run_test(query: str, experiment_id: int = None):
    """
    Runs a single query against one or all 4 agent experiments.
    """
    to_run = [e for e in EXPERIMENTS if experiment_id is None or e["id"] == experiment_id]
    
    print(f"\n{'='*60}")
    print(f"QUERY: \"{query}\"")
    print(f"{'='*60}\n")

    for exp in to_run:
        print(f"--- Running Experiment {exp['id']}: {exp['name']} ---")
        agent_instance = exp["class"](verbose=False)
        
        try:
            # The __call__ method returns a 'dossier' string
            dossier = agent_instance(query)
            
            # Use the agent's internal memory to pull raw details
            # Most agents store their ToolCallingAgent in self._agent or self._last_agent
            agent_obj = getattr(agent_instance, "_agent", getattr(agent_instance, "_last_agent", None))
            
            if agent_obj and agent_obj.memory.steps:
                last_step = agent_obj.memory.steps[-1]
                
                # Extract Thought
                thought = "No thought found."
                if hasattr(last_step, "model_output_message") and last_step.model_output_message:
                    content = last_step.model_output_message.content
                    match = re.search(r"Thought:(.*?)(?:Action:|$)", content, re.DOTALL)
                    if match:
                        thought = match.group(1).strip()
                
                # Extract Action
                tool_calls = extract_tool_calls(agent_obj.memory.steps)
                action = "No tool called."
                if tool_calls:
                    call = tool_calls[0]
                    name = getattr(call, "name", getattr(getattr(call, "function", None), "name", "unknown"))
                    args = getattr(call, "arguments", getattr(getattr(call, "function", None), "arguments", {}))
                    action = f"{name}({args})"
                
                print(f"THOUGHT: {thought[:200]}..." if len(thought) > 200 else f"THOUGHT: {thought}")
                print(f"ACTION:  {action}")
            else:
                print("Dossier Output:\n", dossier)
                
        except Exception as e:
            print(f"EXPERIMENT {exp['id']} FAILED: {e}")
        print("-" * 40)

if __name__ == "__main__":
    # Example usage:
    # 1. Provide a query as a command line argument
    # 2. Or use the default one below
    test_query = sys.argv[1] if len(sys.argv) > 1 else "Outlook stopped receiving emails — is there an outage?"
    
    # Run against all 4
    run_test(test_query)
