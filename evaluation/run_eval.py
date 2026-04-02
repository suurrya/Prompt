"""
evaluation/run_eval.py
=======================
Benchmark runner for the four IT Helpdesk prompting experiments.

Usage
-----
    cd it_helpdesk_agents
    python evaluation/run_eval.py                            # all 4 agents, all 20 tests
    python evaluation/run_eval.py --experiments 3 4          # only dynamic agents
    python evaluation/run_eval.py --tests TC-001 TC-005      # specific test cases
    python evaluation/run_eval.py --skip 2                   # skip one experiment
    python evaluation/run_eval.py --verbose                  # print agent reasoning traces
    python evaluation/run_eval.py --debug                    # print full error tracebacks

How accuracy is measured
------------------------
After each agent.run() call, the runner inspects the smolagents step log
to find the FIRST tool called. It tries every known log attribute path
across smolagents versions (memory.steps, logs, tool_calls, tool_name).
As a fallback it regex-scans the response text for any known tool name.
"""

from __future__ import annotations

import argparse # Purposes: Handles command-line flags like --verbose or --experiments.
import importlib # Purposes: Dynamically loads the agent class from each project folder.
import json # Purposes: Saves the final accuracy report to results.json.
import os # Purposes: Manages file paths for .env and results.
import re # Purposes: The 'Detective' that finds tool names in the agent's text response.
import sys # Purposes: Configures the python path so imports work correctly.
import time # Purposes: Measures the 'Latency' (the clock time) for each AI response.
import traceback # Purposes: Captures and displays detailed error info if an agent crashes.
from datetime import datetime # Purposes: Timestamps the benchmark run for data logging.
from typing import Optional # Purposes: Type hinting for the 'Tool Detective' (might return None).
from dotenv import load_dotenv # Purposes: Loads project-wide environment variables from .env.

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))

from evaluation.test_cases import TEST_CASES

EXPERIMENTS = {
    1: ("project_1_few_shot",          "Static Few-Shot"),
    2: ("project_2_chain_of_thought",  "Static Chain-of-Thought"),
    3: ("project_3_dynamic_few_shot",  "Dynamic Few-Shot"),
    4: ("project_4_dynamic_cot",       "Dynamic Chain-of-Thought"),
}

TOOL_NAMES = [
    "create_ticket", "escalate_ticket", "lookup_knowledge_base",
    "reset_password", "get_user_info", "lookup_user_account",
    "check_system_status", "schedule_maintenance", "process_refund",
    "store_resolved_ticket", "save_ticket_to_long_term_memory",
    "get_user_long_term_memory", "get_customer_history",
]


# ── Pre-flight check ──────────────────────────────────────────────────────────

def preflight_check() -> None:
    """Validate prerequisites and exit with a clear message if anything is wrong."""

    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))

    hf_key = os.environ.get("NVIDIA_API_KEY", "")
    
    if not hf_key or hf_key == "your_nvidia_api_key_here":
        print("\n[ERROR] NVIDIA_API_KEY is missing or invalid.")
        print("        Please add your token to your .env file.\n")
        sys.exit(1)

    selected_key = "NVIDIA_API_KEY"

    try:
        import smolagents
        print(f"[OK] smolagents {smolagents.__version__} installed")
    except ImportError:
        print("\n[ERROR] smolagents is not installed. Run: pip install -r requirements.txt\n")
        sys.exit(1)

    try:
        import sklearn  # noqa: F401
        print("[OK] scikit-learn installed")
    except ImportError:
        print("\n[ERROR] scikit-learn not installed. Run: pip install -r requirements.txt\n")
        sys.exit(1)

    print(f"[OK] {selected_key} found\n")


# ── Tool-call extraction ──────────────────────────────────────────────────────

def _get_agent_steps(agent_instance) -> list:
    """
    Retrieve the step list from a smolagents agent, trying all known
    attribute paths across smolagents versions.

    Experiments 1 & 2: agent._agent  (set once at __init__)
    Experiments 3 & 4: agent._last_agent  (set after each __call__)
    """
    # Purposes: Searches for the 'Memory' of the agent. 
    # Different experiments store the 'worker bee' in different attributes.
    for attr in ("_last_agent", "_agent"):
        inner = getattr(agent_instance, attr, None)
        if inner is None:
            continue
        # smolagents >= 1.8 → agent.memory.steps
        # Purposes: The modern way smolagents keeps track of its thoughts and actions.
        if hasattr(inner, "memory") and hasattr(inner.memory, "steps"):
            return list(inner.memory.steps)
        # smolagents 1.x → agent.logs
        # Purposes: The legacy way agents recorded their history.
        if hasattr(inner, "logs"):
            return list(inner.logs)
    return []


def _extract_first_tool(agent_instance, response_text: str, debug: bool = False) -> Optional[str]:
    """
    Return the name of the first tool called, or None if undetectable.

    Tries (in order):
      1. Walk smolagents step logs for ToolCall / tool_name attributes.
      2. Scan any string attribute of each step for a known tool name.
      3. Regex-scan the response text.
    """
    # Purposes: Grabs the history of what the agent did.
    steps = _get_agent_steps(agent_instance)
    if debug and steps:
        print(f"      [DEBUG] {len(steps)} steps found in agent log")

    # Purposes: Phase 1 - Look for 'Structured' tool calls (the ideal way).
    for step in steps:
        # smolagents >= 1.x: ActionStep.tool_calls → list[ToolCall]
        tool_calls = getattr(step, "tool_calls", None)
        if tool_calls:
            first = tool_calls[0]
            name = getattr(first, "name", None) or (
                first.get("name") if isinstance(first, dict) else None
            )
            if name and name in TOOL_NAMES:
                return name

        # Older smolagents: ActionStep.tool_name
        # Purposes: Fallback for older library versions.
        tool_name = getattr(step, "tool_name", None)
        if isinstance(tool_name, str) and tool_name in TOOL_NAMES:
            return tool_name

        # Scan string-valued attributes as a last resort
        # IMPORTANT: skip model_input_messages to avoid picking up names from prompt examples
        # Purposes: Phase 2 - If structured calls aren't there, hunt for tool names in the raw text output.
        for attr in vars(step) if hasattr(step, "__dict__") else []:
            if attr == "model_input_messages":
                continue
            val = str(getattr(step, attr, "") or "")
            for tn in TOOL_NAMES:
                if re.search(rf"\b{re.escape(tn)}\b", val):
                    if debug:
                        print(f"      [DEBUG] found '{tn}' in step.{attr}")
                    return tn

    # Fallback: scan response text
    # Purposes: Phase 3 - Ultimate fallback. If nothing's in the log, we just check the final text the agent returned.
    for tn in TOOL_NAMES:
        if re.search(rf"\b{re.escape(tn)}\b", response_text):
            return tn

    if debug:
        print(f"      [DEBUG] no tool found in {len(steps)} steps or response text")
        print(f"      [DEBUG] response[:200] = {response_text[:200]!r}")
    return None


# ── Single-agent evaluation ───────────────────────────────────────────────────

def evaluate_agent(
    experiment_id: int,
    test_cases: list[dict],
    verbose: bool = False,
    debug: bool = False,
) -> dict:
    """Load and run one agent against all test cases. Returns a result dict."""
    module_path, exp_name = EXPERIMENTS[experiment_id]
    bar = "=" * 70
    # Purposes: Prints the header for this experiment's run.
    print(f"\n{bar}\n  Experiment {experiment_id}: {exp_name}\n{bar}")

    # ── Import agent — let errors surface, don't swallow them ────────────
    try:
        # Purposes: Dynamically imports the agent. This keeps Experiment 1 separated from Experiment 4.
        module = importlib.import_module(f"{module_path}.agents")
        agent = module.ITHelpdeskAgent(verbose=verbose)
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
        # Purposes: A 'Safety Net'. We run the agent on its own thread so we can kill it if it hangs.
        _pool = ThreadPoolExecutor(max_workers=1)
        CALL_TIMEOUT = 15
    except Exception:
        print(f"\n[FATAL] Could not instantiate Experiment {experiment_id} agent:")
        traceback.print_exc()
        return {
            "experiment_id": experiment_id, "name": exp_name,
            "accuracy": 0.0, "correct": 0, "total": len(test_cases),
            "results": [], "fatal_error": traceback.format_exc(),
        }

    case_results = []
    correct_count = 0

    # Purposes: Loops through every test case (the Questions) to see if the AI (the Student) gets them right.
    for tc in test_cases:
        print(f"  [{tc['id']}] {tc['query'][:62]}…", end=" ", flush=True)
        t0 = time.perf_counter()
        error_detail: Optional[str] = None

        try:
            # Purposes: Sends the query to the agent and waits for a response (with a 15-second cutoff).
            future = _pool.submit(agent, tc["query"])
            response = future.result(timeout=CALL_TIMEOUT)
        except FuturesTimeout:
            error_detail = f"TIMEOUT after {CALL_TIMEOUT}s"
            response = f"ERROR: {error_detail}"
            # Cancel the hung future (best-effort)
            future.cancel()
        except Exception as exc:
            error_detail = repr(exc)
            response = f"ERROR: {exc}"
            if debug:
                print()
                traceback.print_exc()

        latency = time.perf_counter() - t0 # Purposes: Measures how many seconds the agent 'thought' for.
        # Purposes: The moment of truth. Extracts what tool the agent picked and compares it to the answer key.
        actual_tool = _extract_first_tool(agent, response, debug=debug)
        correct = actual_tool == tc["expected_tool"]
        if correct:
            correct_count += 1

        icon = "✓" if correct else "✗"
        err_suffix = f"  ← {error_detail}" if error_detail else ""
        print(f"{icon}  [{actual_tool or 'NONE'} | expected: {tc['expected_tool']}]  ({latency:.1f}s){err_suffix}")

        case_results.append({
            "id": tc["id"],
            "query": tc["query"],
            "expected": tc["expected_tool"],
            "actual": actual_tool,
            "correct": correct,
            "latency_s": round(latency, 2),
            "response": response[:500],   # cap length in JSON output
            "error": error_detail,
        })

    # Purposes: Final Scorecard. Calculates the percentage of correct answers.
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
    col_w = 26
    bar90 = "═" * 100
    dash90 = "─" * 100

    print(f"\n\n{bar90}")
    print("  BENCHMARK SUMMARY")
    print(bar90)

    header = f"{'Test Case':<12} {'Expected Tool':<28}"
    for r in all_results:
        header += f" {r['name'][:col_w]:<{col_w}}"
    print(header)
    print(dash90)

    tc_map = {tc["id"]: tc for tc in test_cases}
    for tc in test_cases:
        row = f"{tc['id']:<12} {tc['expected_tool']:<28}"
        for r in all_results:
            res = {res["id"]: res for res in r["results"]}.get(tc["id"])
            if res:
                actual = res["actual"] or "NONE"
                cell = f"{'✓' if res['correct'] else '✗'} {actual}"
            else:
                cell = "N/A"
            row += f" {cell:<{col_w}}"
        print(row)

    print(dash90)
    acc_row = f"{'ACCURACY':<12} {'':<28}"
    lat_row = f"{'AVG LATENCY':<12} {'':<28}"
    for r in all_results:
        lats = [res["latency_s"] for res in r["results"]]
        avg_lat = sum(lats) / max(len(lats), 1)
        acc_row += f" {r['accuracy']:.0%} ({r['correct']}/{r['total']}){'':<{col_w - 12}}"
        lat_row += f" {avg_lat:.1f}s{'':<{col_w - 5}}"
    print(acc_row)
    print(lat_row)
    print(bar90)

    # Ranking
    ranked = sorted(all_results, key=lambda x: x["accuracy"], reverse=True)
    print("\n  RANKING:")
    medals = ["🥇", "🥈", "🥉", "  "]
    for i, r in enumerate(ranked):
        m = medals[i] if i < 3 else "  "
        print(f"    {m}  Experiment {r['experiment_id']}: {r['name']}  →  {r['accuracy']:.0%}")

    # Per-category breakdown
    categories = sorted({tc["category"] for tc in test_cases})
    print("\n  ACCURACY BY CATEGORY:")
    cat_hdr = f"  {'Category':<15}"
    for r in all_results:
        cat_hdr += f" {r['name'][:22]:<24}"
    print(cat_hdr)
    for cat in categories:
        cat_ids = {tc["id"] for tc in test_cases if tc["category"] == cat}
        row = f"  {cat:<15}"
        for r in all_results:
            cat_res = [res for res in r["results"] if res["id"] in cat_ids]
            c = sum(1 for res in cat_res if res["correct"])
            row += f" {c}/{len(cat_res)}{'':<22}" if cat_res else f" {'N/A':<24}"
        print(row)

    # Per-difficulty breakdown
    difficulties = sorted({tc["difficulty"] for tc in test_cases})
    print("\n  ACCURACY BY DIFFICULTY:")
    dif_hdr = f"  {'Difficulty':<15}"
    for r in all_results:
        dif_hdr += f" {r['name'][:22]:<24}"
    print(dif_hdr)
    for dif in difficulties:
        dif_ids = {tc["id"] for tc in test_cases if tc["difficulty"] == dif}
        row = f"  {dif:<15}"
        for r in all_results:
            dif_res = [res for res in r["results"] if res["id"] in dif_ids]
            c = sum(1 for res in dif_res if res["correct"])
            row += f" {c}/{len(dif_res)}{'':<22}" if dif_res else f" {'N/A':<24}"
        print(row)

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run IT Helpdesk agent benchmark.")
    parser.add_argument("--experiments", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--skip", nargs="+", type=int, default=[])
    parser.add_argument("--tests", nargs="+", default=None,
                        help="Specific test IDs, e.g. TC-001 TC-005")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each agent's full reasoning trace")
    parser.add_argument("--debug", action="store_true",
                        help="Print full error tracebacks and step-log diagnostics")
    parser.add_argument("--output", default="evaluation/results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    preflight_check()

    experiments_to_run = [e for e in args.experiments if e not in args.skip]
    if not experiments_to_run:
        print("No experiments selected. Exiting.")
        sys.exit(0)

    test_cases = TEST_CASES
    if args.tests:
        test_cases = [tc for tc in TEST_CASES if tc["id"] in args.tests]
        if not test_cases:
            print(f"No matching test cases for: {args.tests}")
            sys.exit(1)

    print(f"IT Helpdesk Agent Benchmark")
    print(f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiments: {experiments_to_run}")
    print(f"Test cases : {len(test_cases)}")

    all_results = []
    for exp_id in experiments_to_run:
        result = evaluate_agent(exp_id, test_cases,
                                verbose=args.verbose, debug=args.debug)
        all_results.append(result)

    print_summary(all_results, test_cases)

    output_path = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Purposes: The 'Report Preservation'. Saves all the hard work we just did so it can be viewed in the UI.
    with open(output_path, "w") as f:
        json.dump({
            "run_at": datetime.now().isoformat(),
            "experiments": experiments_to_run,
            "test_case_count": len(test_cases),
            "results": all_results,
        }, f, indent=2)
    # Purposes: Final message to the developer.
    print(f"Results saved → {output_path}\n")


if __name__ == "__main__":
    main()
