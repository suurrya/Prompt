Project Deep Dive: The IT Helpdesk Agent Benchmark
1. Project Objective & Philosophy
Good morning. The goal of this document is to provide a comprehensive overview of the IT Helpdesk Agent Benchmark project, covering our primary objective, system architecture, a detailed breakdown of each component, and a live example demonstrating how it all comes together.

At its core, this project is not just about building a single AI agent. It's about building a benchmark application to empirically test which AI prompting strategies are the most effective. We aim to answer the question:

What is the best way to instruct a language model to reliably choose the correct tool for a given IT helpdesk task?

2. Core Architecture: "Plan, Don't Run"
After extensive debugging of model unreliability and crashes (AssertionError), we landed on a stable and robust architecture we call "Plan, Don't Run." This architecture is the key to the system's stability, as it leverages the model's powerful reasoning (the "Plan") without getting bogged down by the fragile, multi-step execution (the "Run").

Here is the high-level data flow:

UI (ui/app.py): A user submits a query through the web interface. app.py sends this same query concurrently to all four of our experimental agents.

Agent (agents.py): Each of the four agents takes the query and applies its unique prompting strategy to construct a system prompt.

The "Single-Step Plan": The agent then calls the Language Model (LLM) for a single step. Critically, we force it to stop immediately after it makes its decision by setting max_steps=1 and catching the expected AgentMaxStepsError. This prevents the model from getting stuck in conversational loops.

Robust Parsing (tool_extract.py): The agent's memory now contains the raw text output from the model. We pass this memory to our custom extract_tool_calls utility. This is our safety net—it can parse a properly formatted tool call, but it can also find the tool call in messy, plain-text output that smaller models often produce.

Dossier Creation: The agent's __call__ method formats all this information—the selected examples (if any), the model's thought process, and the extracted tool call—into a single, rich Markdown string we call a "Reasoning Dossier."

UI Rendering (ui/app.py): The UI receives this dossier string. It doesn't just display it; it parses it again to create the beautiful, interactive card view with a human-friendly summary, a collapsible details section, and the comparison banner.

3. Code Deep Dive: The Role of Each File
The project is organized into three main areas: the core plumbing, the experiments themselves, and the evaluation suite.

Part 1: The Core "Plumbing" (Shared Infrastructure)
These are the foundational files that make the system work, regardless of which agent is running.

model_wrapper.py (The Universal Adapter):

The "Why": The NVIDIA API for Llama 3 does not support the modern, structured JSON tool-calling format that libraries like smol-agents expect.

What it does: The TextToolParserModel class acts as a Compatibility Shim.

It sets supports_native_tools = False to force text-based prompting.

It contains a resilient retry loop with exponential backoff to automatically handle transient 500 or DEGRADED server errors from the API.

Its main job is in parse_tool_calls, where it uses a sophisticated Regex Parser to find tool calls like Action: create_ticket(...) in the model's plain text output and manually re-formats them into the structured objects the framework expects.

tools.py (The Agent's "API Surface"):

What it does: This file defines the agent's capabilities—the actual Python functions for reset_password, check_system_status, etc.

The Secret Sauce: The LLM never sees the Python code inside these functions. It only reads the docstrings. A clear, well-written docstring is absolutely critical for the agent to understand when and how to use each tool.

tool_extract.py (The "Safety Net"):

What it does: This is a custom utility that robustly parses the agent's memory after a run. Because smaller models often fail to produce perfect formatting, this function tries multiple strategies in order of reliability to find the tool call, ensuring we capture the model's intent even if its formatting is imperfect.

Part 2: The 4 Experiments (The Agents)
This is where we test our hypotheses. We've organized the four agents into a hierarchy of increasing intelligence and complexity.

Experiment	Strategy	Pros	Cons
1	Static Few-Shot	Uses a fixed list of ~20 hardcoded examples in the system prompt.	Fastest, lowest latency.	Fails on new or ambiguous edge cases.
2	Static CoT	Uses the same fixed examples, but each includes a Thought: section to teach reasoning.	Better classification and reasoning.	Slower (more tokens to generate).
3	Dynamic Few-Shot	Uses TF-IDF similarity to find the most relevant examples "just-in-time" for each query.	Great context; handles hundreds of tools.	Adds metadata overhead; sensitive to data quality.
4	Dynamic CoT	Combines just-in-time example selection with full reasoning traces.	Highest Accuracy & Most Robust.	Slowest and uses the most tokens.
 
The __call__ method in each of the four agents.py files implements these unique strategies while conforming to our stable 'Plan, Don't Run' architecture.

Part 3: Interaction & Evaluation (The Dashboard and Judge)
ui/app.py (The Dashboard):

Built with the NiceGUI framework.

It runs all 4 agents concurrently using Python's asyncio and ThreadPoolExecutor.

Its main job is rendering. It takes the dossier strings from the agents and uses a series of helper functions to build the final interactive UI, complete with educational explanations and the "Agents Agree/Disagree" banner.

reproduce_errors.py (The Debugger):

A simple terminal-based script. If an agent is behaving strangely in the UI, we use this to bypass the pretty formatting and see the raw, unfiltered output from the model, which is essential for debugging prompt issues.

evaluation/run_eval.py (The Judge):

This is our automated test harness. It runs the 20+ scenarios defined in test_cases.py against all four agents and produces a final accuracy score for each, giving us the quantitative data to prove which prompting strategy is superior.

4. The Prompting Techniques in Action: A Case Study
This screenshot from a live test run perfectly illustrates the value of our benchmark.

The Query: "My Wi-Fi keeps dropping every hour in the office."
The Correct Action: lookup_knowledge_base (Tier 1 self-service first).



Experiment 1: Static Few-Shot (The "Flashcard Student")
Result: Success.

Analysis: The user's query was a very close match to one of its hardcoded examples, so simple pattern-matching worked perfectly. It's fast and effective for common problems.

Experiment 2: Static Chain-of-Thought (The "Student Who Shows Their Work")
Result: Failure (chose create_ticket).

Analysis: This shows that just adding a reasoning framework isn't a magic bullet. Without a perfectly relevant static example, the reasoning can lead the model astray, causing it to focus on "dropping" as a "fault" and incorrectly escalate.

Experiment 3: Dynamic Few-Shot (The "Just-in-Time Flashcard Student")
Result: Failure (chose create_ticket).

Analysis: This is a critical insight. The TF-IDF search found the perfect example, but the agent still failed because the example itself in our database was wrong. It taught the agent the incorrect action, proving this technique is highly sensitive to data quality.

Experiment 4: Dynamic Chain-of-Thought (The "Open-Book Test Student")
Result: Success.

Analysis: This is the most impressive result. It saw the same flawed example as Agent 3, but it didn't blindly follow it. Its internal reasoning framework forced it to ask, "Can the user self-resolve with KB guidance?" This principle allowed it to override the flawed example and make the correct, nuanced decision. This proves that combining dynamic examples with a reasoning framework makes an agent more robust.

5. Conclusion and Next Steps
In summary, we have built a stable and powerful benchmark application. The "Plan, Don't Run" architecture has solved our stability issues, and the UI now provides a clear, educational view into the AI's decision-making process.

Our next steps are clear:

Prompt Refinement: Continue to refine the examples in our prompts.py files, using the results from the run_eval.py script to target our weakest areas and push all agents towards our 95% accuracy goal.

Stateful Memory: The current agents are stateless. The next major architectural evolution will be to implement conversational memory, allowing the agents to understand follow-up questions and perform multi-turn tasks.