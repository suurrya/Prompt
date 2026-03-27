"""
evaluation/run_eval.py
=======================
Benchmark runner for the four IT Helpdesk prompting experiments.

Usage
-----
    cd it_helpdesk_agents
    python evaluation/run_eval.py

    # Run only a subset of test cases (by ID):
    python evaluation/run_eval.py --tests TC-001 TC-005 TC-014

    # Run only specific experiments:
    python evaluation/run_eval.py --experiments 1 3

    # Skip a slow experiment:
    python evaluation/run_eval.py --skip 2

    # Verbose mode (prints each agent's tool call trace):
    python evaluation/run_eval.py --verbose

Output
------
  Per-agent accuracy + a side-by-side comparison table so mismatches are
  immediately visible. Results are also saved to evaluation/results.json.

How accuracy is measured
------------------------
  After each agent call, the runner inspects `agent._agent.logs` (smolagents
  stores a list of Step objects). The FIRST ToolCall step's tool_name is
  extracted and compared against test_cases.expected_tool.

  If the framework changes its log format, the fallback extracts the tool
  name from the string representation of the agent's response.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.test_cases import TEST_CASES

# ── Agent registry ────────────────────────────────────────────────────────────
EXPERIMENTS = {
    1: ("project_1_few_shot", "Static Few-Shot"),
    2: ("project_2_chain_of_thought", "Static Chain-of-Thought"),
    3: ("project_3_dynamic_few_shot", "Dynamic Few-Shot"),
    4: ("project_4_dynamic_cot", "Dynamic Chain-of-Thought"),
}

TOOL_NAMES = [
    "create_ticket",
    "escalate_ticket",
    "lookup_knowledge_base",
    "reset_password",
    "get_user_info",
    "check_system_status",
    "schedule_maintenance",
]


# ── Tool-call extraction ──────────────────────────────────────────────────────

def _extract_first_tool(agent_instance, response_text: str) -> Optional[str]:
    """
    Extract the name of the first tool called during an agent run.

    Strategy (in order of preference):
      1. Inspect agent.logs for ActionStep / ToolCall objects (smolagents API).
      2. Regex search of the response_text for known tool names.
    """
    # Strategy 1: inspect logs
    try:
        logs = agent_instance._agent.logs  # list of Step objects
        for step in logs:
            # smolagents >= 1.x stores tool calls in step.tool_calls
            if hasattr(step, "tool_calls") and step.tool_calls:
                return step.tool_calls[0].name
            # older smolagents stores in step.tool_name
            if hasattr(step, "tool_name") and step.tool_name:
                return step.tool_name
    except Exception:
        pass

    # Strategy 2: regex on response text
    for tool_name in TOOL_NAMES:
        if re.search(rf"\b{re.escape(tool_name)}\b", response_text):
            return tool_name

    return None


# ── Single-agent evaluation ───────────────────────────────────────────────────

def evaluate_agent(
    experiment_id: int,
    test_cases: list[dict],
    verbose: bool = False,
) -> dict:
    """
    Load and run a single agent against all test cases.

    Returns a result dict:
        {
            "experiment_id": int,
            "name": str,
            "accuracy": float,        # 0.0 – 1.0
            "results": [              # per-case results
                {
                    "id": str,
                    "query": str,
                    "expected": str,
                    "actual": str | None,
                    "correct": bool,
                    "latency_s": float,
                    "response": str,
                }
            ]
        }
    """
    module_path, exp_name = EXPERIMENTS[experiment_id]
    print(f"\n{'='*70}")
    print(f"  Experiment {experiment_id}: {exp_name}")
    print(f"{'='*70}")

    # Dynamic import so we can control which experiments run
    module = importlib.import_module(f"{module_path}.agents")
    agent = module.ITHelpdeskAgent(verbose=verbose)

    case_results = []
    correct_count = 0

    for tc in test_cases:
        print(f"  [{tc['id']}] {tc['query'][:60]}...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            response = agent(tc["query"])
        except Exception as exc:
            response = f"ERROR: {exc}"

        latency = time.perf_counter() - t0
        actual_tool = _extract_first_tool(agent, response)
        correct = actual_tool == tc["expected_tool"]
        if correct:
            correct_count += 1

        status_icon = "✓" if correct else "✗"
        print(f"{status_icon}  [{actual_tool or 'NONE'} | expected: {tc['expected_tool']}]  ({latency:.1f}s)")

        case_results.append({
            "id": tc["id"],
            "query": tc["query"],
            "expected": tc["expected_tool"],
            "actual": actual_tool,
            "correct": correct,
            "latency_s": round(latency, 2),
            "response": response,
        })

    accuracy = correct_count / len(test_cases) if test_cases else 0.0
    print(f"\n  → Accuracy: {correct_count}/{len(test_cases)} = {accuracy:.0%}")

    return {
        "experiment_id": experiment_id,
        "name": exp_name,
        "accuracy": round(accuracy, 4),
        "correct": correct_count,
        "total": len(test_cases),
        "results": case_results,
    }


# ── Summary report ────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict], test_cases: list[dict]) -> None:
    """Print a formatted side-by-side comparison table."""
    print("\n\n" + "═" * 90)
    print("  BENCHMARK SUMMARY")
    print("═" * 90)

    # Header
    col_w = 26
    header = f"{'Test Case':<12} {'Expected Tool':<25}"
    for r in all_results:
        short = r["name"][:col_w]
        header += f" {short:<{col_w}}"
    print(header)
    print("─" * 90)

    # One row per test case
    tc_map = {tc["id"]: tc for tc in test_cases}
    for tc_id in [tc["id"] for tc in test_cases]:
        tc = tc_map[tc_id]
        row = f"{tc_id:<12} {tc['expected_tool']:<25}"
        for r in all_results:
            res_map = {res["id"]: res for res in r["results"]}
            res = res_map.get(tc_id)
            if res:
                actual = res["actual"] or "NONE"
                icon = "✓" if res["correct"] else "✗"
                cell = f"{icon} {actual}"
            else:
                cell = "N/A"
            row += f" {cell:<{col_w}}"
        print(row)

    # Footer: accuracy + avg latency
    print("─" * 90)
    acc_row = f"{'ACCURACY':<12} {'':<25}"
    lat_row = f"{'AVG LATENCY':<12} {'':<25}"
    for r in all_results:
        avg_lat = sum(res["latency_s"] for res in r["results"]) / max(len(r["results"]), 1)
        acc_row += f" {r['accuracy']:.0%} ({r['correct']}/{r['total']}){'':<{col_w - 12}}"
        lat_row += f" {avg_lat:.1f}s{'':<{col_w - 5}}"
    print(acc_row)
    print(lat_row)
    print("═" * 90)

    # Rank
    ranked = sorted(all_results, key=lambda x: x["accuracy"], reverse=True)
    print("\n  RANKING:")
    for i, r in enumerate(ranked, 1):
        print(f"    #{i}  Experiment {r['experiment_id']}: {r['name']}  →  {r['accuracy']:.0%}")

    # Category breakdown
    categories = list({tc["category"] for tc in test_cases})
    print("\n  ACCURACY BY CATEGORY:")
    cat_header = f"  {'Category':<15}"
    for r in all_results:
        cat_header += f" {r['name'][:20]:<22}"
    print(cat_header)
    for cat in sorted(categories):
        cat_ids = {tc["id"] for tc in test_cases if tc["category"] == cat}
        row = f"  {cat:<15}"
        for r in all_results:
            cat_results = [res for res in r["results"] if res["id"] in cat_ids]
            if cat_results:
                c = sum(1 for res in cat_results if res["correct"])
                row += f" {c}/{len(cat_results)}{'':<20}"
            else:
                row += f" {'N/A':<22}"
        print(row)

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run IT Helpdesk agent benchmark.")
    parser.add_argument(
        "--experiments", nargs="+", type=int, default=[1, 2, 3, 4],
        help="Which experiments to run (1-4). Default: all.",
    )
    parser.add_argument(
        "--skip", nargs="+", type=int, default=[],
        help="Experiment IDs to skip.",
    )
    parser.add_argument(
        "--tests", nargs="+", default=None,
        help="Specific test case IDs to run (e.g. TC-001 TC-005).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each agent's reasoning trace.",
    )
    parser.add_argument(
        "--output", default="evaluation/results.json",
        help="Path to save JSON results. Default: evaluation/results.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Filter experiments
    experiments_to_run = [e for e in args.experiments if e not in args.skip]
    if not experiments_to_run:
        print("No experiments selected. Exiting.")
        sys.exit(0)

    # Filter test cases
    test_cases = TEST_CASES
    if args.tests:
        test_cases = [tc for tc in TEST_CASES if tc["id"] in args.tests]
        if not test_cases:
            print(f"No matching test cases found for: {args.tests}")
            sys.exit(1)

    print(f"\nIT Helpdesk Agent Benchmark")
    print(f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiments: {experiments_to_run}")
    print(f"Test cases : {len(test_cases)}")

    all_results = []
    for exp_id in experiments_to_run:
        result = evaluate_agent(exp_id, test_cases, verbose=args.verbose)
        all_results.append(result)

    print_summary(all_results, test_cases)

    # Save results
    output_path = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "run_at": datetime.now().isoformat(),
                "experiments": experiments_to_run,
                "test_case_count": len(test_cases),
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
