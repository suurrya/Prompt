"""
project_3_dynamic_few_shot/prompts.py
=======================================
Strategy: Dynamic Few-Shot Prompting
--------------------------------------
Philosophy: make the examples CONTEXTUALLY RELEVANT rather than fixed.
At call-time, TF-IDF cosine similarity ranks the example bank and inserts
only the top-k examples most similar to the incoming query.

Design decisions:
  • The EXAMPLE_DATABASE is larger (30 entries) so the selector has
    more candidates to choose from — increasing the chance of a near-match.
  • Examples are written as compact stimulus→response pairs (no Thought:)
    — the value here comes from RELEVANCE, not reasoning depth.
  • The prompt TEMPLATE is intentionally lean; all the intelligence comes
    from which examples get selected, not from meta-instructions.
  • TF-IDF with bigrams (ngram_range=(1,2)) captures two-word phrases like
    "screen flickering" or "VPN connection" that single-word TF-IDF misses.
  • top_k=4 balances context richness against prompt bloat.

Hypothesis: dynamic selection outperforms static few-shot on queries that
fall outside the static example set, because the closest match is always
injected rather than the same fixed examples every time.
"""

from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Example database ─────────────────────────────────────────────────────────

EXAMPLE_DATABASE: list[dict] = [
    # Auth
    {"query": "I forgot my password and can't log in",
     "tool_call": 'reset_password(user_email="<email>", method="email")'},
    {"query": "Account locked after too many wrong password attempts",
     "tool_call": 'reset_password(user_email="<email>", method="sms")'},
    {"query": "How do I set up two-factor authentication?",
     "tool_call": 'lookup_knowledge_base(query="MFA two-factor authentication setup")'},
    {"query": "My colleague is locked out of Active Directory",
     "tool_call": 'reset_password(user_email="<email>", method="manual")'},

    # Network / VPN
    {"query": "I can't connect to the VPN from home",
     "tool_call": 'lookup_knowledge_base(query="VPN connection remote work setup")'},
    {"query": "Wi-Fi keeps dropping in the office",
     "tool_call": 'lookup_knowledge_base(query="Wi-Fi wireless network dropping corporate")'},
    {"query": "Nobody on the 3rd floor has internet",
     "tool_call": 'check_system_status(service_name="internet")'},

    # Hardware
    {"query": "Laptop screen flickering badly",
     "tool_call": 'lookup_knowledge_base(query="laptop screen flickering display driver")'},
    {"query": "Printer is offline and won't print",
     "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Printer offline", user_email="<email>")'},
    {"query": "Computer is very slow and freezing",
     "tool_call": 'lookup_knowledge_base(query="computer slow freezing performance")'},
    {"query": "Laptop battery drains in under an hour",
     "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Laptop battery degraded", user_email="<email>")'},
    {"query": "Book a RAM upgrade for my workstation",
     "tool_call": 'schedule_maintenance(asset_id="<id>", maintenance_type="ram_upgrade", preferred_date="<date>", user_email="<email>")'},
    {"query": "Keyboard keys are stuck or not responding",
     "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Keyboard unresponsive", user_email="<email>")'},
    {"query": "Laptop keeps overheating and shutting down",
     "tool_call": 'lookup_knowledge_base(query="laptop overheating thermal shutdown")'},
    {"query": "Screen replacement needed for my laptop",
     "tool_call": 'schedule_maintenance(asset_id="<id>", maintenance_type="screen_replacement", preferred_date="<date>", user_email="<email>")'},

    # Software
    {"query": "Microsoft Excel crashes when I open it",
     "tool_call": 'lookup_knowledge_base(query="Microsoft Excel crash M365 Office repair")'},
    {"query": "I need Adobe Photoshop installed on my laptop",
     "tool_call": 'create_ticket(category="software", priority="low", summary="Software install — Adobe Photoshop", user_email="<email>")'},
    {"query": "I need Slack and Zoom installed",
     "tool_call": 'create_ticket(category="software", priority="low", summary="Software install — Slack and Zoom", user_email="<email>")'},
    {"query": "Outlook is not syncing emails",
     "tool_call": 'lookup_knowledge_base(query="Outlook email sync Exchange troubleshoot")'},
    {"query": "Teams audio and microphone not working in meetings",
     "tool_call": 'lookup_knowledge_base(query="Teams microphone audio not working meeting")'},

    # Security
    {"query": "I received a suspicious phishing email",
     "tool_call": ('create_ticket(category="security", priority="critical", '
                   'summary="Suspected phishing email", user_email="<email>") '
                   '→ escalate_ticket(ticket_id="<id>", reason="Phishing attempt", escalate_to="security-team")')},
    {"query": "My laptop may be infected with malware or ransomware",
     "tool_call": ('create_ticket(category="security", priority="critical", '
                   'summary="Suspected malware infection", user_email="<email>") '
                   '→ escalate_ticket(ticket_id="<id>", reason="Malware suspected", escalate_to="security-team")')},

    # Access
    {"query": "I need access to the Finance shared drive",
     "tool_call": 'create_ticket(category="access", priority="medium", summary="Access request — Finance shared drive", user_email="<email>")'},
    {"query": "My new hire needs an AD account and laptop",
     "tool_call": 'create_ticket(category="access", priority="high", summary="New hire onboarding — AD account and laptop", user_email="<email>")'},
    {"query": "I need read access to the Legal SharePoint library",
     "tool_call": 'create_ticket(category="access", priority="medium", summary="Access request — Legal SharePoint read", user_email="<email>")'},

    # System status
    {"query": "Is SharePoint down? Nobody can access files",
     "tool_call": 'check_system_status(service_name="sharepoint")'},
    {"query": "CRM has been throwing 500 errors all morning",
     "tool_call": 'check_system_status(service_name="crm")'},
    {"query": "Are there any email outages today?",
     "tool_call": 'check_system_status(service_name="email")'},

    # User / account / billing
    {"query": "Look up account details for alice.jones@company.com",
     "tool_call": 'get_user_info(user_email="alice.jones@company.com")'},
    {"query": "Check subscription status for bob@company.com",
     "tool_call": 'lookup_user_account(email="bob@company.com")'},
    {"query": "Process a refund for reservation RES-00123",
     "tool_call": 'process_refund(reservation_id="RES-00123")'},

    # Memory / history
    {"query": "What issues has user jdoe had before?",
     "tool_call": 'get_customer_history(user_id="jdoe")'},
    {"query": "Retrieve the full history for user u001",
     "tool_call": 'get_user_long_term_memory(user_id="u001")'},
    {"query": "The ticket is resolved. Archive it to the user's history.",
     "tool_call": 'store_resolved_ticket(user_id="<user_id>", summary="<summary>")'},
    {"query": "Save the full resolution outcome for this user.",
     "tool_call": 'save_ticket_to_long_term_memory(user_id="<user_id>", summary="<summary>", resolution="<resolution>")'},
]

# ── TF-IDF selector ──────────────────────────────────────────────────────────

def _build_index():
    queries = [ex["query"] for ex in EXAMPLE_DATABASE]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform(queries)
    return vec, mat

_VECTORIZER, _MATRIX = _build_index()


def select_examples(user_query: str, top_k: int = 4) -> list[dict]:
    """
    Return the top_k EXAMPLE_DATABASE entries whose stored query is most
    similar to user_query by TF-IDF cosine similarity.
    """
    qvec = _VECTORIZER.transform([user_query])
    scores = cosine_similarity(qvec, _MATRIX).flatten()
    indices = scores.argsort()[::-1][:top_k]
    return [EXAMPLE_DATABASE[i] for i in indices]


# ── Prompt template (filled at runtime by build_system_prompt) ──────────────

_TEMPLATE = """\
You are an IT Helpdesk agent. Select and call the correct tool for the user's request.

AVAILABLE TOOLS:
  lookup_knowledge_base        – self-service KB articles
  create_ticket                – new helpdesk ticket
  escalate_ticket              – escalate to specialist
  reset_password               – password reset
  get_user_info                – AD directory lookup
  lookup_user_account          – account + subscription status
  check_system_status          – live service status
  schedule_maintenance         – physical maintenance appointment
  process_refund               – billing refund
  store_resolved_ticket        – archive resolved issue (brief)
  save_ticket_to_long_term_memory – archive full outcome
  get_user_long_term_memory    – retrieve full user history
  get_customer_history         – quick past-issues summary

RULES:
  1. Use lookup_knowledge_base FIRST for documented, self-service issues.
  2. Use reset_password when the user is actively locked out.
  3. Run check_system_status BEFORE creating a ticket for service outages.
  4. After create_ticket for security incidents, always escalate_ticket.
  5. Use schedule_maintenance for explicit physical upgrade/repair requests.

MOST RELEVANT EXAMPLES FOR THIS QUERY:
{examples_block}

Call the correct tool now.
"""


def build_system_prompt(user_query: str, top_k: int = 4) -> str:
    """
    Dynamically construct the system prompt by selecting the top_k examples
    most similar to user_query and injecting them into the template.
    """
    examples = select_examples(user_query, top_k=top_k)
    lines: list[str] = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"[{i}] User: \"{ex['query']}\"")
        lines.append(f"    → {ex['tool_call']}")
        lines.append("")
    return _TEMPLATE.format(examples_block="\n".join(lines).strip())
