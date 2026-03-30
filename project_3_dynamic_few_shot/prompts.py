"""
project_3_dynamic_few_shot/prompts.py (Re-architected for Multi-Layer Dynamism)
=======================================
Strategy: A highly dynamic, multi-layered prompt that incorporates user context,
conversation history, chained outputs, and relevant few-shot examples in a
token-efficient manner.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Example Database (Can be slightly trimmed for cost) ──────────────────────
# Keep this database rich, but we can trim some of the more repetitive entries
# if max token savings are needed. For now, we'll assume it's the same.
EXAMPLE_DATABASE: list[dict] = [
    # ... (Your existing 30+ examples go here) ...
    # Auth
    {"query": "I forgot my password and can't log in", "tool_call": 'reset_password(user_email="<email>", method="email")'},
    {"query": "Account locked after too many wrong password attempts", "tool_call": 'reset_password(user_email="<email>", method="sms")'},
    # Network / VPN
    {"query": "I can't connect to the VPN from home", "tool_call": 'lookup_knowledge_base(query="VPN connection remote work setup")'},
    # Hardware
    {"query": "Laptop screen flickering badly", "tool_call": 'lookup_knowledge_base(query="laptop screen flickering display driver")'},
    {"query": "Printer is offline and won't print", "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Printer offline", user_email="<email>")'},
    {"query": "Book a RAM upgrade for my workstation", "tool_call": 'schedule_maintenance(asset_id="<id>", maintenance_type="ram_upgrade", preferred_date="<date>", user_email="<email>")'},
    # ... etc.
]

# ── TF-IDF Selector (No changes needed here) ─────────────────────────────────
def _build_index():
    queries = [ex["query"] for ex in EXAMPLE_DATABASE]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform(queries)
    return vec, mat

_VECTORIZER, _MATRIX = _build_index()

def select_examples(user_query: str, top_k: int = 3) -> list[dict]: # Reduced top_k to 3
    qvec = _VECTORIZER.transform([user_query])
    scores = cosine_similarity(qvec, _MATRIX).flatten()
    indices = scores.argsort()[::-1][:top_k]
    return [EXAMPLE_DATABASE[i] for i in indices]

# ── New Modular Prompt Builder ───────────────────────────────────────────────
def build_system_prompt(
    user_query: str,
    user: dict | None = None,         # ADD: Optional runtime user context
    history: list | None = None,      # ADD: Optional conversation history
    previous_output: str | None = None, # ADD: Optional chained context
    top_k: int = 3                    # Default to 3 examples to save tokens
) -> str:
    """
    Dynamically constructs a token-efficient system prompt by assembling
    multiple layers of context only when they are available.
    """
    prompt_parts = []

    # --- Core Instruction (Minimal) ---
    prompt_parts.append("You are an IT agent. Call the correct tool for the user's request. Output only the tool call.")

    # --- Layer 1: Chained Context (Most Immediate) ---
    if previous_output:
        prompt_parts.append(f"\nCONTEXT FROM PREVIOUS STEP:\n{previous_output}")

    # --- Layer 2: User and Conversation Context ---
    # Combine user and history into a single, efficient block
    if user or history:
        context_block = "CURRENT CONTEXT:\n"
        if user:
            context_block += f"- User: {user.get('name', 'N/A')} ({user.get('role', 'N/A')})\n"
        if history:
            # Show only the last few turns of conversation to save tokens
            for turn in history[-3:]: 
                context_block += f"- {turn.get('role', 'unknown')}: {turn.get('text', '')}\n"
        prompt_parts.append(context_block)
    
    # --- Layer 3: Dynamic Few-Shot Examples (Most Relevant) ---
    examples = select_examples(user_query, top_k=top_k)
    if examples:
        examples_block = "RELEVANT EXAMPLES:\n"
        for ex in examples:
            examples_block += f"User: \"{ex['query']}\" → {ex['tool_call']}\n"
        prompt_parts.append(examples_block)

    # --- Final Assembly ---
    # Join all available parts into a single prompt string.
    # The double newline is a strong separator for the model.
    return "\n\n".join(prompt_parts)

