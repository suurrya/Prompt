"""
project_3_dynamic_few_shot/prompts.py
=======================================
Dynamic Few-Shot Prompting
---------------------------
Instead of a fixed example bank, this module:
  1. Stores a larger database of (query, tool_call) pairs.
  2. Provides a `select_examples()` function that, at call-time, ranks
     the stored examples by TF-IDF cosine similarity to the incoming query
     and returns the top-k most relevant ones.
  3. Provides a `build_system_prompt()` function that assembles the final
     prompt from a template + the dynamically selected examples.

Why this beats static few-shot:
  When a user asks about "Outlook email sync failures", a static prompt
  that only has a VPN example will force the model to generalise across a
  large topic gap. A dynamic selector can surface the email-specific example
  instead, giving the model a much closer pattern to match against.
"""

from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------------------------------------------------------
# Example database — larger than the static bank so the selector has room
# to pick the genuinely closest matches.
# ---------------------------------------------------------------------------
EXAMPLE_DATABASE: list[dict] = [
    # Auth / Passwords
    {
        "query": "I forgot my password and can't log in",
        "tool_call": 'reset_password(user_email="<user_email>", method="email")',
    },
    {
        "query": "My account is locked out after too many failed attempts",
        "tool_call": 'reset_password(user_email="<user_email>", method="sms")',
    },
    {
        "query": "How do I set up two-factor authentication?",
        "tool_call": 'lookup_knowledge_base(query="MFA two-factor authentication setup")',
    },
    # Network / VPN
    {
        "query": "I can't connect to the VPN from home",
        "tool_call": 'lookup_knowledge_base(query="VPN connection remote home setup")',
    },
    {
        "query": "Wi-Fi keeps dropping on my laptop in the office",
        "tool_call": 'lookup_knowledge_base(query="Wi-Fi wireless network dropping corporate")',
    },
    {
        "query": "The internet is completely down on my floor",
        "tool_call": 'check_system_status(service_name="internet")',
    },
    # Hardware
    {
        "query": "My laptop screen is flickering badly",
        "tool_call": 'lookup_knowledge_base(query="laptop screen flickering display issue")',
    },
    {
        "query": "My printer shows as offline and won't print",
        "tool_call": (
            'create_ticket(category="hardware", priority="medium", '
            'summary="Printer offline — user cannot print", user_email="<user_email>")'
        ),
    },
    {
        "query": "My computer is very slow and keeps freezing",
        "tool_call": 'lookup_knowledge_base(query="computer slow freezing performance")',
    },
    {
        "query": "My laptop battery drains in under an hour",
        "tool_call": (
            'create_ticket(category="hardware", priority="medium", '
            'summary="Laptop battery drains unusually fast", user_email="<user_email>")'
        ),
    },
    {
        "query": "I need a RAM upgrade for my workstation",
        "tool_call": (
            'schedule_maintenance(asset_id="<asset_id>", maintenance_type="ram_upgrade", '
            'preferred_date="<date>", user_email="<user_email>")'
        ),
    },
    {
        "query": "My keyboard keys are sticking and some aren't responding",
        "tool_call": (
            'create_ticket(category="hardware", priority="medium", '
            'summary="Keyboard keys sticking or unresponsive", user_email="<user_email>")'
        ),
    },
    # Software
    {
        "query": "Microsoft Office keeps crashing when I open Excel",
        "tool_call": 'lookup_knowledge_base(query="Microsoft Office Excel crash M365")',
    },
    {
        "query": "I need Adobe Photoshop installed on my machine",
        "tool_call": (
            'create_ticket(category="software", priority="low", '
            'summary="Software installation request — Adobe Photoshop", user_email="<user_email>")'
        ),
    },
    {
        "query": "Outlook is not syncing my emails",
        "tool_call": 'lookup_knowledge_base(query="Outlook email sync not working Exchange")',
    },
    # Teams / Collaboration
    {
        "query": "Teams crashes every time I join a video call",
        "tool_call": 'lookup_knowledge_base(query="Microsoft Teams crash video call audio")',
    },
    {
        "query": "My microphone doesn't work in Teams meetings",
        "tool_call": 'lookup_knowledge_base(query="Teams microphone not working meeting audio")',
    },
    # Security
    {
        "query": "I received a suspicious email asking for my password",
        "tool_call": (
            'create_ticket(category="security", priority="critical", '
            'summary="Suspected phishing email", user_email="<user_email>") '
            '→ escalate_ticket(reason="Phishing attempt", escalate_to="security-team")'
        ),
    },
    {
        "query": "I think my laptop has a virus",
        "tool_call": (
            'create_ticket(category="security", priority="high", '
            'summary="Suspected malware infection", user_email="<user_email>") '
            '→ escalate_ticket(reason="Possible malware", escalate_to="security-team")'
        ),
    },
    # Access
    {
        "query": "I need access to the Finance shared drive",
        "tool_call": (
            'create_ticket(category="access", priority="medium", '
            'summary="Access request — Finance shared drive", user_email="<user_email>")'
        ),
    },
    {
        "query": "My colleague needs read access to the Marketing SharePoint site",
        "tool_call": (
            'create_ticket(category="access", priority="medium", '
            'summary="Access request — Marketing SharePoint read access", user_email="<user_email>")'
        ),
    },
    # System Status
    {
        "query": "Is SharePoint down? I can't open any documents",
        "tool_call": 'check_system_status(service_name="sharepoint")',
    },
    {
        "query": "Our CRM system seems to be having issues",
        "tool_call": 'check_system_status(service_name="crm")',
    },
    {
        "query": "Are there any known outages affecting email today?",
        "tool_call": 'check_system_status(service_name="email")',
    },
    # Onboarding
    {
        "query": "I'm a new employee and need help setting up my laptop",
        "tool_call": 'lookup_knowledge_base(query="new employee IT onboarding laptop setup")',
    },
    # User Info
    {
        "query": "Can you tell me what devices are assigned to john.smith@company.com?",
        "tool_call": 'get_user_info(user_email="john.smith@company.com")',
    },
]

# ---------------------------------------------------------------------------
# TF-IDF similarity selector
# ---------------------------------------------------------------------------

def _build_corpus() -> tuple[list[str], TfidfVectorizer, np.ndarray]:
    """Vectorise the example queries once and cache the result."""
    queries = [ex["query"] for ex in EXAMPLE_DATABASE]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(queries)
    return queries, vectorizer, matrix


_CORPUS_QUERIES, _VECTORIZER, _CORPUS_MATRIX = _build_corpus()


def select_examples(user_query: str, top_k: int = 4) -> list[dict]:
    """
    Return the `top_k` examples from EXAMPLE_DATABASE whose stored query is
    most similar to `user_query` according to TF-IDF cosine similarity.

    Args:
        user_query: The raw user message.
        top_k: Number of examples to return.

    Returns:
        List of example dicts (keys: 'query', 'tool_call'), ordered by
        descending similarity.
    """
    query_vec = _VECTORIZER.transform([user_query])
    scores = cosine_similarity(query_vec, _CORPUS_MATRIX).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [EXAMPLE_DATABASE[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are an expert IT Helpdesk assistant. Your sole responsibility is to select
and call the most appropriate tool to resolve the user's IT problem.

## Available Tools
- lookup_knowledge_base  : Search self-service articles (use FIRST for common issues).
- create_ticket          : Log a new support ticket for hands-on work.
- escalate_ticket        : Escalate a ticket to a specialist team.
- reset_password         : Initiate a password reset for a locked-out user.
- get_user_info          : Retrieve account and device info from the directory.
- check_system_status    : Check if a service is currently up or experiencing an outage.
- schedule_maintenance   : Book a physical maintenance appointment for a device.

## Tool Selection Rules
1. Prefer lookup_knowledge_base for well-documented, self-service issues.
2. Use reset_password when the user is explicitly locked out or cannot log in.
3. Use check_system_status BEFORE creating a ticket for service-outage reports.
4. Use create_ticket for hardware faults, access requests, or issues needing hands-on work.
5. Use escalate_ticket immediately for confirmed security incidents.
6. Use schedule_maintenance only for physical, on-site hardware work.
7. Use get_user_info to retrieve a user's profile when needed for context.

## Most Relevant Examples for This Query
{examples_block}

Now call the correct tool for the user's request. Do NOT explain your choice —
just call the tool.
"""


def build_system_prompt(user_query: str, top_k: int = 4) -> str:
    """
    Dynamically construct the system prompt by:
      1. Selecting the top-k most similar examples for user_query.
      2. Rendering them into the template.

    Args:
        user_query: The incoming user message.
        top_k: How many examples to inject.

    Returns:
        A fully assembled system prompt string.
    """
    examples = select_examples(user_query, top_k=top_k)
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}")
        lines.append(f'User: "{ex["query"]}"')
        lines.append(f"Action: {ex['tool_call']}")
        lines.append("")
    examples_block = "\n".join(lines).strip()
    return _PROMPT_TEMPLATE.format(examples_block=examples_block)
