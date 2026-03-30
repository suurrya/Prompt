"""
project_4_dynamic_cot/prompts.py  — IMPROVED
==============================================
Dynamic Chain-of-Thought (CoT) Prompting
-----------------------------------------
Improvements over original Experiment 4:

  1. Thought format aligned to the reasoning checklist in the template.
     Every Thought now follows the same 5-question structure the template
     instructs the model to use — removing the contradiction between what
     the template says and what the examples demonstrate.

  2. Similarity threshold added to select_cot_examples().
     Examples below min_score=0.1 are filtered out so the model never
     sees weakly relevant demonstrations that could mislead it.

  3. Fallback example added.
     If no example clears the threshold, one hardcoded general example
     is injected so the examples block is never completely empty.

  4. Thought rendering is multi-line aware.
     Each line of a structured Thought is indented individually, making
     the reasoning trace easy for the model to parse.

  5. Thoughts rewritten in active, present-tense deliberative style.
     Every Thought now reads as live reasoning-in-progress, not a past
     tense summary — matching the inferencing style the model must produce.
"""

from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------------------------------------------------------
# CoT Example Database
# Thought format MUST match the 5-question checklist in the prompt template:
#   Problem type → Self-resolve? → Known outage? → Physical work? → Security risk?
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Optimized CoT Example Database (Telegraphic Thoughts)
# ---------------------------------------------------------------------------
COT_EXAMPLE_DATABASE: list[dict] = [
    # Path 1: Auth
    {
        "query": "I forgot my password and I'm locked out",
        "thought": "1.Type:auth; 2.KB?:no(locked out); 3.Outage?:no; 4.Physical?:no; 5.Security?:no. -> Decision:reset_password.",
        "tool_call": 'reset_password(user_email="<user_email>", method="email")',
    },
    # Path 2: KB
    {
        "query": "How do I set up two-factor authentication?",
        "thought": "1.Type:how-to; 2.KB?:yes(MFA setup); 3.Outage?:no; 4.Physical?:no; 5.Security?:no. -> Decision:lookup_knowledge_base.",
        "tool_call": 'lookup_knowledge_base(query="MFA two-factor authentication setup")',
    },
    # Path 3: Outage
    {
        "query": "Nobody on my floor can access SharePoint today.",
        "thought": "1.Type:outage(team-wide); 2.KB?:no(infra); 3.Outage?:yes(MUST check status first); 4.Physical?:no; 5.Security?:no. -> Decision:check_system_status.",
        "tool_call": 'check_system_status(service_name="sharepoint")',
    },
    # Path 4: Security
    {
        "query": "I clicked a link in a suspicious email.",
        "thought": "1.Type:security(phishing); 2.KB?:no; 3.Outage?:no; 4.Physical?:no; 5.Security?:YES-CRITICAL. -> Decision:create ticket, then escalate.",
        "tool_call": 'create_ticket(category="security", priority="critical", summary="User clicked phishing link", user_email="<user_email>") → escalate_ticket(ticket_id="<ticket_id>", reason="Phishing", escalate_to="security-team")',
    },
    # Path 5: Hardware
    {
        "query": "My laptop battery dies in under an hour.",
        "thought": "1.Type:hardware(fault); 2.KB?:no(physical); 3.Outage?:no; 4.Physical?:yes(battery); 5.Security?:no. -> Decision:create_ticket.",
        "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Laptop battery degraded", user_email="<user_email>")',
    },
    # Path 6: Software
    {
        "query": "I need Adobe Acrobat Pro installed.",
        "thought": "1.Type:software(install); 2.KB?:no(license); 3.Outage?:no; 4.Physical?:no; 5.Security?:no. -> Decision:create_ticket.",
        "tool_call": 'create_ticket(category="software", priority="low", summary="Software installation request: Adobe Acrobat Pro", user_email="<user_email>")',
    },
]


# COT_EXAMPLE_DATABASE: list[dict] = [

#     # Path 1: User is locked out -> Direct Action
#     {
#         "query": "I forgot my password and I'm locked out",
#         "thought": (
#             "Problem type: authentication — credential failure.\n"
#             "Can user self-resolve with KB? No — they are locked out.\n"
#             "Known outage? No.\n"
#             "Physical work needed? No.\n"
#             "Security risk? No.\n"
#             "Decision: reset_password directly."
#         ),
#         "tool_call": 'reset_password(user_email="<user_email>", method="email")',
#     },
#     # Path 2: "How to" question -> Knowledge Base
#     {
#         "query": "How do I set up two-factor authentication?",
#         "thought": (
#             "Problem type: how-to / guidance.\n"
#             "Can user self-resolve? Yes — MFA setup is documented in the KB.\n"
#             "Known outage? Not applicable.\n"
#             "Physical work needed? No.\n"
#             "Security risk? No.\n"
#             "Decision: lookup_knowledge_base for the MFA setup article."
#         ),
#         "tool_call": 'lookup_knowledge_base(query="MFA two-factor authentication setup")',
#     },
#     # Path 3: Team-wide issue -> System Status Check
#     {
#         "query": "Nobody on my floor can access SharePoint today.",
#         "thought": (
#             "Problem type: outage — team-wide symptom.\n"
#             "Can user self-resolve? No — implies infrastructure issue.\n"
#             "Known outage? Very likely — I MUST check_system_status before creating a ticket.\n"
#             "Physical work needed? Maybe, but depends on status check.\n"
#             "Security risk? No.\n"
#             "Decision: check_system_status for 'sharepoint' first."
#         ),
#         "tool_call": 'check_system_status(service_name="sharepoint")',
#     },
#     # Path 4: Security Incident -> Create Ticket AND Escalate
#     {
#         "query": "I clicked a link in a suspicious email that asked for my password.",
#         "thought": (
#             "Problem type: security incident — active phishing.\n"
#             "Can user self-resolve? No — requires immediate security team intervention.\n"
#             "Known outage? Not applicable.\n"
#             "Physical work needed? No.\n"
#             "Security risk? YES — CRITICAL. Credentials may be compromised.\n"
#             "Decision: create_ticket for audit, then immediately escalate_ticket."
#         ),
#         "tool_call": (
#             'create_ticket(category="security", priority="critical", summary="User clicked phishing link", user_email="<user_email>") '
#             '→ escalate_ticket(ticket_id="<ticket_id>", reason="Active phishing attempt", escalate_to="security-team")'
#         ),
#     },
#     # Path 5: Physical Hardware Fault -> Create Ticket
#     {
#         "query": "My laptop battery drains in under an hour.",
#         "thought": (
#             "Problem type: hardware fault — battery degradation.\n"
#             "Can user self-resolve? No — this requires physical replacement.\n"
#             "Known outage? Not applicable.\n"
#             "Physical work needed? Yes — a technician needs to replace the battery.\n"
#             "Security risk? No.\n"
#             "Decision: create_ticket under hardware."
#         ),
#         "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Laptop battery degraded", user_email="<user_email>")',
#     },
#     # Path 6: Software Install Request -> Create Ticket
#     {
#         "query": "I need Adobe Acrobat Pro installed on my laptop.",
#         "thought": (
#             "Problem type: software — installation request.\n"
#             "Can user self-resolve? No — requires license and IT deployment.\n"
#             "Known outage? Not applicable.\n"
#             "Physical work needed? No.\n"
#             "Security risk? No.\n"
#             "Decision: create_ticket under software."
#         ),
#         "tool_call": 'create_ticket(category="software", priority="low", summary="Software installation request — Adobe Acrobat Pro", user_email="<user_email>")',
#     },
#     # Path 7: Access Request -> Create Ticket
#     {
#         "query": "I need access to the Finance department shared drive.",
#         "thought": (
#             "Problem type: access request.\n"
#             "Can user self-resolve? No — requires manager approval and IT action.\n"
#             "Known outage? Not applicable.\n"
#             "Physical work needed? No.\n"
#             "Security risk? No.\n"
#             "Decision: create_ticket under access."
#         ),
#         "tool_call": 'create_ticket(category="access", priority="medium", summary="Access request — Finance shared drive", user_email="<user_email>")',
#     },
#     # Path 8: User History Request -> Get History Tool
#     {
#         "query": "What issues has user 'jdoe' had before?",
#         "thought": (
#             "Problem type: history — user is asking for their own data.\n"
#             "Can user self-resolve? Not applicable.\n"
#             "Known outage? No.\n"
#             "Physical work needed? No.\n"
#             "Security risk? No.\n"
#             "Decision: get_customer_history is the direct tool for this."
#         ),
#         "tool_call": 'get_customer_history(user_id="jdoe")',
#     },
# ]

# ---------------------------------------------------------------------------
# Fallback example — used when no example clears the similarity threshold
# ---------------------------------------------------------------------------

FALLBACK_EXAMPLE: dict = {
    "query": "General IT issue",
    "thought": "1.Type:unclear; 2.KB?:yes(default); 3.Outage?:check if service-related; 4.Physical?:unknown; 5.Security?:escalate if any risk. -> Decision:lookup_knowledge_base.",
    "tool_call": 'lookup_knowledge_base(query="<describe the issue in detail>")',
}

# ---------------------------------------------------------------------------
# TF-IDF selector — same mechanism as Experiment 3, now with score threshold
# ---------------------------------------------------------------------------

def _build_cot_corpus() -> tuple[list[str], TfidfVectorizer, np.ndarray]:
    """Vectorise example queries once at import time and cache."""
    queries = [ex["query"] for ex in COT_EXAMPLE_DATABASE]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(queries)
    return queries, vectorizer, matrix


_COT_CORPUS_QUERIES, _COT_VECTORIZER, _COT_CORPUS_MATRIX = _build_cot_corpus()


def select_cot_examples(
    user_query: str,
    top_k: int = 2,
    min_score: float = 0.1,        # NEW: filter out weakly relevant examples
) -> list[dict]:
    """
    Return up to top_k CoT examples most similar to user_query.

    Only examples scoring >= min_score are returned. If none clear the
    threshold, an empty list is returned and the caller falls back to
    the FALLBACK_EXAMPLE.

    Args:
        user_query:  The raw incoming user message.
        top_k:       Maximum number of examples to return.
        min_score:   Minimum TF-IDF cosine similarity to qualify.

    Returns:
        List of matching example dicts, ordered by descending similarity.
    """
    query_vec = _COT_VECTORIZER.transform([user_query])
    scores = cosine_similarity(query_vec, _COT_CORPUS_MATRIX).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [
        COT_EXAMPLE_DATABASE[i]
        for i in top_indices
        if scores[i] >= min_score   # threshold gate
    ]


# ---------------------------------------------------------------------------
# Prompt template
# The How-to-Reason checklist MATCHES the Thought format used in every example.
# ---------------------------------------------------------------------------

# _COT_PROMPT_TEMPLATE = """\
# You are an expert IT agent. Reason through the user's request using the 5 questions, then call the correct tool.

# ## How to Reason (Chain-of-Thought)
# 1. Problem type: (auth / hardware / software / network / security / access / outage)
# 2. Can user self-resolve with KB?
# 3. Possible service outage? (if yes -> check_system_status FIRST)
# 4. Requires physical IT work?
# 5. Security risk?

# ## Relevant Examples
# {examples_block}

# Now, apply the reasoning framework to the user's request and call the tool.
# """

_COT_PROMPT_TEMPLATE = """\
You are an expert IT agent. For the user's request, reason through these 5 steps:
1. Problem type? 2. Self-resolve with KB? 3. Possible outage? 4. Physical work? 5. Security risk?
Then, call the correct tool. Output a 'Thought:' block and an 'Action:' block.

TOOLS: lookup_knowledge_base, create_ticket, escalate_ticket, reset_password, get_user_info, check_system_status, schedule_maintenance

## Relevant Examples
{examples_block}
"""


# def _render_example(i: int, ex: dict) -> list[str]:
#     """Render a single CoT example into indented, multi-line text."""
#     lines = [
#         f"---",
#         f"Example {i}",
#         f'User: "{ex["query"]}"',
#         "Thought:",
#     ]
#     for thought_line in ex["thought"].strip().split("\n"):
#         lines.append(f"  {thought_line.strip()}")
#     lines.append(f"Action: {ex['tool_call']}")
#     lines.append("")
#     return lines

def _render_example(ex: dict) -> list[str]:
    """Render a single CoT example into a compact, multi-line format."""
    # No "Example X" or "---" to save tokens
    return [
        f'User: "{ex["query"]}"',
        f'Thought: {ex["thought"]}',
        f'Action: {ex["tool_call"]}',
    ]


def build_system_prompt(user_query: str, top_k: int = 2) -> str:
    """
    Dynamically construct the CoT system prompt:
      1. Select top_k most relevant examples above the similarity threshold.
      2. Fall back to FALLBACK_EXAMPLE if nothing qualifies.
      3. Render each example with structured, multi-line Thought formatting.
      4. Inject into the template.

    Args:
        user_query: The raw incoming user message.
        top_k:      Maximum number of CoT examples to inject.

    Returns:
        A fully assembled system prompt string with dynamic CoT examples.
    """
    examples = select_cot_examples(user_query, top_k=top_k)

    # Fallback: if no example cleared the threshold, use the general one
    if not examples:
        examples = [FALLBACK_EXAMPLE]

    lines: list[str] = []
    for ex in examples:
        lines.extend(_render_example(ex))

    examples_block = "\n".join(lines).strip()
    return _COT_PROMPT_TEMPLATE.format(examples_block=examples_block)