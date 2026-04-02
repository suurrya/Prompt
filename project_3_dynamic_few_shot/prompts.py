"""
project_3_dynamic_few_shot/prompts.py (Re-architected for Multi-Layer Dynamism)
=======================================
Strategy: Dynamic Few-Shot Prompting
--------------------------------------
Philosophy: make the examples CONTEXTUALLY RELEVANT rather than fixed.
At call-time, TF-IDF(Term Frequency–Inverse Document Frequency) cosine similarity ranks the example bank and inserts
only the top-k examples most similar to the incoming query.

Design decisions:
  • The EXAMPLE_DATABASE is larger (30 entries) so the selector has
    more candidates to choose from — increasing the chance of a near-match.
  • Examples are written as compact stimulus→response pairs (no Thought:)
    — the value here comes from RELEVANCE, not reasoning depth.
  • The prompt TEMPLATE is intentionally lean; all the intelligence comes
    from which examples get selected, not from meta-instructions.
  • TF-IDF(Term Frequency–Inverse Document Frequency) with bigrams (ngram_range=(1,2)) captures two-word phrases like
    "screen flickering" or "VPN connection" that single-word TF-IDF(Term Frequency–Inverse Document Frequency) misses.
  • top_k=4 balances context richness against prompt bloat.

Hypothesis: dynamic selection outperforms static few-shot on queries that
fall outside the static example set, because the closest match is always
injected rather than the same fixed examples every time.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EXAMPLE_DATABASE: list[dict] = [

    # ── AUTH ──────────────────────────────────────────────────────────────
    {"query": "I forgot my password and I'm locked out of my computer",
     "tool_call": 'reset_password(user_email="<email>", method="email")'},
    {"query": "I'm completely locked out and can't log in",
     "tool_call": 'reset_password(user_email="<email>", method="email")'},
    {"query": "Account locked after too many wrong password attempts",
     "tool_call": 'reset_password(user_email="<email>", method="sms")'},
    # TC-003: indirect lockout via colleague report
    {"query": "My colleague says I've been locked out of AD after some failed logins",
     "tool_call": 'reset_password(user_email="<email>", method="manual")'},
    {"query": "My colleague says my AD account is locked out",
     "tool_call": 'reset_password(user_email="<email>", method="manual")'},
    {"query": "How do I set up two-factor authentication on my work account",
     "tool_call": 'lookup_knowledge_base(query="MFA two-factor authentication setup work account")'},
    {"query": "How do I enrol in MFA",
     "tool_call": 'lookup_knowledge_base(query="MFA enrollment setup authenticator")'},

    # ── NETWORK ───────────────────────────────────────────────────────────
    {"query": "I can't seem to connect to the VPN it worked yesterday",
     "tool_call": 'lookup_knowledge_base(query="VPN connection troubleshoot AnyConnect remote")'},
    {"query": "VPN keeps disconnecting when I work from home",
     "tool_call": 'lookup_knowledge_base(query="VPN connection remote work setup AnyConnect")'},
    # TC-005: floor-wide outage
    {"query": "Nobody on the 3rd floor has any internet right now",
     "tool_call": 'check_system_status(service_name="internet")'},
    {"query": "The whole office has no internet this morning",
     "tool_call": 'check_system_status(service_name="internet")'},
    # TC-006: single-user Wi-Fi
    {"query": "My Wi-Fi keeps dropping every hour in the office",
     "tool_call": 'lookup_knowledge_base(query="Wi-Fi wireless dropping corporate office troubleshoot")'},

    # ── HARDWARE ──────────────────────────────────────────────────────────
    # TC-007: screen flicker — KB documented
    {"query": "The display on my laptop is flickering badly it's very distracting",
     "tool_call": 'lookup_knowledge_base(query="laptop screen flickering display driver refresh rate")'},
    {"query": "My laptop screen keeps flickering and the display is unstable",
     "tool_call": 'lookup_knowledge_base(query="laptop screen flickering display driver")'},
    # TC-008: battery dead — needs ticket
    {"query": "My laptop battery only lasts 40 minutes now even when fully charged",
     "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Laptop battery degraded — 40 min runtime", user_email="<email>")'},
    {"query": "Laptop battery dies in under an hour after full charge",
     "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Laptop battery degraded — needs replacement", user_email="<email>")'},
    # TC-009: schedule RAM upgrade
    {"query": "Can you book a slot to upgrade my workstation's RAM it needs more memory",
     "tool_call": 'schedule_maintenance(asset_id="<id>", maintenance_type="ram_upgrade", preferred_date="<date>", user_email="<email>")'},
    {"query": "Please book a RAM upgrade for my workstation",
     "tool_call": 'schedule_maintenance(asset_id="<id>", maintenance_type="ram_upgrade", preferred_date="<date>", user_email="<email>")'},
    # TC-010: slow desktop — KB
    {"query": "My desktop is very slow apps are freezing and it's barely usable",
     "tool_call": 'lookup_knowledge_base(query="computer desktop slow freezing performance CPU RAM disk")'},
    {"query": "Computer keeps freezing and is very slow",
     "tool_call": 'lookup_knowledge_base(query="computer freezing slow performance troubleshoot")'},

    # ── SOFTWARE ──────────────────────────────────────────────────────────
    # TC-011: Excel crash — KB
    {"query": "Excel crashes immediately every time I try to open it",
     "tool_call": 'lookup_knowledge_base(query="Microsoft Excel crash M365 Office Quick Repair")'},
    {"query": "Microsoft Office keeps crashing when I open Word or Excel",
     "tool_call": 'lookup_knowledge_base(query="Microsoft Office crash M365 repair cache")'},
    # Mail merge how-to
    {"query": "How do I do a mail merge in Microsoft Word",
     "tool_call": 'lookup_knowledge_base(query="Microsoft Word mail merge steps tutorial")'},
    # TC-012: install request — ticket
    {"query": "I need Slack and Zoom installed on my company laptop",
     "tool_call": 'create_ticket(category="software", priority="low", summary="Software install request — Slack and Zoom", user_email="<email>")'},
    {"query": "Please install Adobe Acrobat on my machine",
     "tool_call": 'create_ticket(category="software", priority="low", summary="Software install request — Adobe Acrobat", user_email="<email>")'},
    # TC-013: Outlook server hint — check status first (HARD)
    {"query": "Outlook stopped receiving emails since this morning is it a server issue",
     "tool_call": 'check_system_status(service_name="email")'},
    {"query": "Outlook not working since this morning could it be a server problem",
     "tool_call": 'check_system_status(service_name="email")'},

    # ── SECURITY ──────────────────────────────────────────────────────────
    # TC-014: phishing with "reset password" wording (HARD — keyword trap)
    # The phrase "reset my password" here means PHISHING, not a real auth issue.
    {"query": "I got an email telling me to reset my password via a link that looks fake",
     "tool_call": ('create_ticket(category="security", priority="critical", '
                   'summary="Suspected phishing — fake password reset link", user_email="<email>") '
                   '→ escalate_ticket(ticket_id="<id>", reason="Phishing link mimicking password reset", escalate_to="security-team")')},
    {"query": "Received suspicious email with a link asking me to verify or reset my password",
     "tool_call": ('create_ticket(category="security", priority="critical", '
                   'summary="Suspected phishing email — credential harvesting link", user_email="<email>") '
                   '→ escalate_ticket(ticket_id="<id>", reason="Phishing attempt", escalate_to="security-team")')},
    # TC-015: ransomware (files renamed) (HARD)
    {"query": "Files on my desktop have been renamed and I can't open any of them",
     "tool_call": ('create_ticket(category="security", priority="critical", '
                   'summary="Suspected ransomware — files renamed and inaccessible", user_email="<email>") '
                   '→ escalate_ticket(ticket_id="<id>", reason="Ransomware suspected — isolation required", escalate_to="security-team")')},
    {"query": "All my files were renamed and I cannot open them there is a red warning screen",
     "tool_call": ('create_ticket(category="security", priority="critical", '
                   'summary="Suspected ransomware — files encrypted", user_email="<email>") '
                   '→ escalate_ticket(ticket_id="<id>", reason="Ransomware attack", escalate_to="security-team")')},
    {"query": "I received a suspicious phishing email",
     "tool_call": ('create_ticket(category="security", priority="critical", '
                   'summary="Suspected phishing email", user_email="<email>") '
                   '→ escalate_ticket(ticket_id="<id>", reason="Phishing attempt", escalate_to="security-team")')},

    # ── ACCESS ────────────────────────────────────────────────────────────
    # TC-016
    {"query": "I need read access to the Legal department's SharePoint library",
     "tool_call": 'create_ticket(category="access", priority="medium", summary="Access request — Legal SharePoint read access", user_email="<email>")'},
    {"query": "I need access to the Finance shared drive",
     "tool_call": 'create_ticket(category="access", priority="medium", summary="Access request — Finance shared drive", user_email="<email>")'},
    # TC-017
    {"query": "My new hire starts Monday and needs an AD account and laptop",
     "tool_call": 'create_ticket(category="access", priority="high", summary="New hire onboarding — AD account and laptop by Monday", user_email="<email>")'},

    # ── SYSTEM STATUS ─────────────────────────────────────────────────────
    # TC-018
    {"query": "Is SharePoint currently experiencing any outages",
     "tool_call": 'check_system_status(service_name="sharepoint")'},
    {"query": "Is there a known issue with SharePoint today",
     "tool_call": 'check_system_status(service_name="sharepoint")'},
    # TC-019
    {"query": "Our CRM has been throwing 500 errors for the whole team since 9am",
     "tool_call": 'check_system_status(service_name="crm")'},
    {"query": "The CRM is giving 500 Internal Server errors for everyone",
     "tool_call": 'check_system_status(service_name="crm")'},
    {"query": "Are there any email outages today",
     "tool_call": 'check_system_status(service_name="email")'},

    # ── USER / ACCOUNT / BILLING ──────────────────────────────────────────
    # TC-020
    {"query": "Can you look up the account details for alice.jones@company.com",
     "tool_call": 'get_user_info(user_email="alice.jones@company.com")'},
    {"query": "What devices does john.smith@company.com have and what department is he in",
     "tool_call": 'get_user_info(user_email="john.smith@company.com")'},
    {"query": "Check the subscription status for bob@company.com",
     "tool_call": 'lookup_user_account(email="bob@company.com")'},
    {"query": "Process a refund for reservation RES-00123",
     "tool_call": 'process_refund(reservation_id="RES-00123")'},

    # ── MEMORY / HISTORY ──────────────────────────────────────────────────
    {"query": "What issues has user jdoe had before",
     "tool_call": 'get_customer_history(user_id="jdoe")'},
    {"query": "Save the resolution to this user's long-term history",
     "tool_call": 'store_resolved_ticket(user_id="<user_id>", summary="<summary>")'},
]

# ── TF-IDF(Term Frequency–Inverse Document Frequency) selector ───────────────────────────────────────────────────────

def _build_index():
    # saves all the queries in the EXAMPLE_DATABASE in a list
    queries = [example["query"] for example in EXAMPLE_DATABASE]
    # Creates and configures a tool to convert text into numerical vectors, ignoring common English words and using both single words and two-word phrases.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    # Converts the list of queries into a matrix of TF-IDF scores
    tfidf_matrix = vectorizer.fit_transform(queries)
    return vectorizer, tfidf_matrix 
    # vec is a trained, reusable tool permanently stores all the knowledge learned from queries
    # mat is a matrix where the matrix the documents and the columes is the the unique terms from the vocabulary
    # the value of the matrix is the tf-idf score of the term in the document

# IDF(term) = log [ (Total number of documents + 1) / (Number of documents containing the term + 1) ] + 1

_VECTORIZER, _MATRIX = _build_index()


def select_examples(user_query: str, top_k: int = 4, min_score: float = 0.2) -> list[dict]:
    """
    Return up to top_k examples, but only if they meet a minimum
    cosine similarity score. This prevents irrelevant examples from being
    injected into the prompt and confusing the model.
    """
    if not user_query.strip():
        return []
        
    # takes the query text and converts it into a numerical vector.    
    query_vector = _VECTORIZER.transform([user_query])

    # calculates how similar the new user query vector is to every example vector in our database
    similarity_scores = cosine_similarity(query_vector, _MATRIX).flatten()
    
    # Get the indices of the top k examples
    # This sorts the scores but returns the original indices of the items in sorted order from highest score to lowest and takes the highest 2 examples
    top_indices = similarity_scores.argsort()[::-1][:top_k]
    
    # --- The NEW Logic: Filter by score ---
    final_examples = []
    for i in top_indices:
        # Only include the example if its score is above our threshold
        if similarity_scores[i] >= min_score:
            final_examples.append(EXAMPLE_DATABASE[i])
            
    return final_examples



# ── Prompt template ───────────────────────────────────────────────────────

_TEMPLATE = """\
You are an IT Helpdesk agent. Call the single most appropriate tool.

TOOLS:
  lookup_knowledge_base(query)                           — KB / how-to articles
  create_ticket(category, priority, summary, user_email) — new support ticket
  escalate_ticket(ticket_id, reason, escalate_to)        — escalate to specialist
  reset_password(user_email, method)                     — password reset
  get_user_info(user_email)                              — AD directory / device lookup
  lookup_user_account(email)                             — subscription / billing
  check_system_status(service_name)                      — live service status
  schedule_maintenance(asset_id, type, date, user_email) — physical maintenance
  process_refund(reservation_id)                         — billing refund
  get_customer_history / get_user_long_term_memory / store_resolved_ticket / save_ticket_to_long_term_memory

RULES:
  1. Team-wide / service-wide issues → check_system_status FIRST.
  2. "is it a server issue?" / "server problem?" → check_system_status FIRST.
  3. Phishing / suspicious email / fake link → create_ticket(critical) then escalate_ticket. NEVER reset_password.
  4. Files renamed / can't open files → create_ticket(critical, ransomware) then escalate_ticket.
  5. User or colleague locked out → reset_password (do not use KB).
  6. "Book a slot" / "upgrade my RAM/screen" → schedule_maintenance.
  7. Install request (not a how-to) → create_ticket.

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
        lines.append(f'[{i}] User: "{ex["query"]}"')
        lines.append(f'    → {ex["tool_call"]}')
        lines.append("")
    return _TEMPLATE.format(examples_block="\n".join(lines).strip())
    