import os
import sys

# Try to find .env, otherwise use system env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure API key is set
if not os.environ.get("NVIDIA_API_KEY"):
    print("NVIDIA_API_KEY not found in env.")

from project_1_few_shot.agents import ITHelpdeskAgent as Agent1

query = "Outlook stopped receiving emails since this morning — is it a server issue?"

print("--- Project 1 ---")
a1 = Agent1(verbose=True)
try:
    res1 = a1(query)
    print("Result:\n", res1)
    print("\n--- RAW MEMORY STEPS ---")
    for step in a1._agent.memory.steps:
        print(f"\n{type(step)}")
        if hasattr(step, "model_input_messages"):
            # Print last prompt msg
            pass
        if hasattr(step, "model_output_message"):
            print("MODEL OUTPUT:", repr(step.model_output_message.content))
        if hasattr(step, "tool_calls"):
            print("TOOL CALLS:", getattr(step, "tool_calls", None))
        if hasattr(step, "error"):
            print("ERROR:", step.error)
            
except Exception as e:
    print("Exception:", e)
