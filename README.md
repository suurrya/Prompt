# IT Helpdesk Agent — Prompting Strategy Benchmark

A controlled experiment comparing four prompting strategies for LLM tool-selection accuracy,
built on the `smolagents` framework.

## Project Layout

```
it_helpdesk_agents/
├── .env                          # OPENAI_API_KEY lives here
├── tools.py                      # Shared @tool definitions (constant across all agents)
├── knowledge_base.py             # Static knowledge store queried by lookup_knowledge_base
├── project_1_few_shot/           # Experiment 1 – Static Few-Shot
│   ├── prompts.py
│   └── agents.py
├── project_2_chain_of_thought/   # Experiment 2 – Static Chain-of-Thought
│   ├── prompts.py
│   └── agents.py
├── project_3_dynamic_few_shot/   # Experiment 3 – Dynamic Few-Shot
│   ├── prompts.py
│   └── agents.py
├── project_4_dynamic_cot/        # Experiment 4 – Dynamic Chain-of-Thought
│   ├── prompts.py
│   └── agents.py
└── evaluation/
    ├── test_cases.py             # 20 labelled queries with expected tool calls
    └── run_eval.py               # Runs all four agents and prints a comparison report
```

## Setup

```bash
pip install smolagents openai scikit-learn python-dotenv
```

Copy `.env.example` → `.env` and add your key:

```
OPENAI_API_KEY=sk-...
```

## Running a single agent

```python
from project_1_few_shot.agents import ITHelpdeskAgent
agent = ITHelpdeskAgent()
print(agent("My laptop screen keeps flickering"))
```

## Running the full benchmark

```bash
cd evaluation
python run_eval.py
```

The report prints per-agent accuracy plus a side-by-side tool-call comparison for
every test case so you can inspect mismatches directly.

## Prompting Strategies at a Glance

| # | Strategy | Examples | Reasoning trace |
|---|----------|----------|-----------------|
| 1 | Static Few-Shot | Fixed in system prompt | ✗ |
| 2 | Static CoT | Fixed in system prompt | ✓ (Thought:) |
| 3 | Dynamic Few-Shot | Selected at runtime via TF-IDF | ✗ |
| 4 | Dynamic CoT | Selected at runtime via TF-IDF | ✓ (Thought:) |
