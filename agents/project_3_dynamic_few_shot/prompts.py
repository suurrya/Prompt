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
from sklearn.feature_extraction.text import TfidfVectorizer # Purposes: The 'Indexer' that converts human queries into mathematical scores.
from sklearn.metrics.pairwise import cosine_similarity # Purposes: The 'Comparison Brain' that finds the most similar examples in 3D-space.

# Purposes: This is the "Warehouse" of example IT scenarios.
# Unlike Experiment 1, this bank is too large for the prompt, so we 
# only 'Ship' the most relevant ones for each query.
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

    # ── HARDWARE (extra variety) ──────────────────────────────────────────
    {"query": "I dropped my laptop and the screen cracked and is black",
     "tool_call": 'create_ticket(category="hardware", priority="high", summary="Laptop screen cracked — display black after drop", user_email="<email>")'},
    {"query": "My laptop screen is cracked and the display is completely gone",
     "tool_call": 'create_ticket(category="hardware", priority="high", summary="Cracked laptop screen — needs replacement", user_email="<email>")'},
    {"query": "Half the keys on my keyboard have stopped responding",
     "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Laptop keyboard keys unresponsive — hardware fault", user_email="<email>")'},
    {"query": "My keyboard is not working properly some keys are broken",
     "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Keyboard hardware fault — keys unresponsive", user_email="<email>")'},
    {"query": "My docking station stopped charging my laptop and USB ports don't work",
     "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Docking station fault — no charging or USB", user_email="<email>")'},
    {"query": "My external monitor shows no signal even though it is plugged in",
     "tool_call": 'lookup_knowledge_base(query="external monitor no signal HDMI DisplayPort connection troubleshoot")'},
    {"query": "External monitor not detecting my laptop via HDMI",
     "tool_call": 'lookup_knowledge_base(query="external monitor HDMI not detected display settings troubleshoot")'},
    {"query": "Can you book a screen replacement for my laptop it has dead pixels",
     "tool_call": 'schedule_maintenance(asset_id="<id>", maintenance_type="screen_replacement", preferred_date="<date>", user_email="<email>")'},
    {"query": "Please book a battery swap for my laptop",
     "tool_call": 'schedule_maintenance(asset_id="<id>", maintenance_type="battery_swap", preferred_date="<date>", user_email="<email>")'},
    {"query": "My USB drive is not being recognised when I plug it in",
     "tool_call": 'lookup_knowledge_base(query="USB drive not recognised device manager troubleshoot")'},

    # ── SOFTWARE (extra variety) ──────────────────────────────────────────
    {"query": "My camera and microphone don't work in Microsoft Teams calls",
     "tool_call": 'lookup_knowledge_base(query="Microsoft Teams camera microphone not working audio video settings")'},
    {"query": "Teams video call has no audio and my camera is not showing",
     "tool_call": 'lookup_knowledge_base(query="Teams camera audio not working call settings troubleshoot")'},
    {"query": "Teams has been completely unavailable for our whole department",
     "tool_call": 'check_system_status(service_name="teams")'},
    {"query": "OneDrive keeps showing a sync error and my files are not uploading",
     "tool_call": 'lookup_knowledge_base(query="OneDrive sync error files not uploading troubleshoot reset")'},
    {"query": "OneDrive is stuck syncing and not finishing",
     "tool_call": 'lookup_knowledge_base(query="OneDrive stuck syncing cache reset sign out troubleshoot")'},
    {"query": "None of us can print to the main office printer this morning",
     "tool_call": 'check_system_status(service_name="printer")'},
    {"query": "My laptop cannot find the office printer it was working last week",
     "tool_call": 'lookup_knowledge_base(query="office printer not found driver install network printer troubleshoot")'},
    {"query": "I can't install software on my own I need admin rights",
     "tool_call": 'create_ticket(category="software", priority="low", summary="Software install request — requires admin elevation", user_email="<email>")'},

    # ── NETWORK (extra variety) ───────────────────────────────────────────
    {"query": "My Ethernet cable is plugged in but my laptop shows no network",
     "tool_call": 'lookup_knowledge_base(query="Ethernet cable no network connection laptop troubleshoot")'},
    {"query": "Browser says unable to connect to everything but ping works",
     "tool_call": 'lookup_knowledge_base(query="browser proxy settings unable to connect network troubleshoot")'},
    {"query": "How do I configure my VPN to use the new server address",
     "tool_call": 'lookup_knowledge_base(query="VPN client configuration new server AnyConnect setup")'},
    {"query": "The internet has been really slow for everyone in the building since lunch",
     "tool_call": 'check_system_status(service_name="internet")'},

    # ── AUTH (extra variety) ──────────────────────────────────────────────
    {"query": "My password expired and I cannot log in",
     "tool_call": 'reset_password(user_email="<email>", method="email")'},
    {"query": "I lost my phone and cannot pass the MFA check I am locked out",
     "tool_call": 'create_ticket(category="access", priority="high", summary="MFA bypass needed — user lost MFA device", user_email="<email>")'},
    {"query": "The SSO login page is not loading for anyone in the office",
     "tool_call": 'check_system_status(service_name="sso")'},

    # ── SECURITY (extra variety) ──────────────────────────────────────────
    {"query": "I found a USB stick plugged into my workstation that I did not put there",
     "tool_call": ('create_ticket(category="security", priority="critical", '
                   'summary="Suspicious USB device found in workstation", user_email="<email>") '
                   '→ escalate_ticket(ticket_id="<id>", reason="Possible hardware implant or exfiltration device", escalate_to="security-team")')},
    {"query": "I accidentally emailed a file with customer data to my personal Gmail",
     "tool_call": ('create_ticket(category="security", priority="critical", '
                   'summary="Data leak — customer data emailed to personal account", user_email="<email>") '
                   '→ escalate_ticket(ticket_id="<id>", reason="Potential data breach — personal email exfiltration", escalate_to="security-team")')},
    {"query": "There is a suspicious unknown process using 100% CPU on my computer",
     "tool_call": 'lookup_knowledge_base(query="unknown suspicious process CPU usage malware check Task Manager")'},

    # ── ACCESS (extra variety) ────────────────────────────────────────────
    {"query": "I need write access to the Finance shared drive",
     "tool_call": 'create_ticket(category="access", priority="medium", summary="Access request — Finance shared drive write access", user_email="<email>")'},
    {"query": "I need read access to the production database for reporting",
     "tool_call": 'create_ticket(category="access", priority="medium", summary="Database access request — production read access for reporting", user_email="<email>")'},
    {"query": "I am travelling to Germany and need a VPN profile for that region",
     "tool_call": 'create_ticket(category="access", priority="medium", summary="VPN profile request — Germany region for travel", user_email="<email>")'},

    # ── STATUS (extra variety) ────────────────────────────────────────────
    {"query": "Is Microsoft Teams having any issues right now",
     "tool_call": 'check_system_status(service_name="teams")'},
    {"query": "The ERP system is not loading for any of us in the finance team",
     "tool_call": 'check_system_status(service_name="erp")'},
    {"query": "Is there a known issue with the ERP today",
     "tool_call": 'check_system_status(service_name="erp")'},

    # ── USER INFO (extra variety) ─────────────────────────────────────────
    {"query": "Can you look up my own account and what devices I am assigned",
     "tool_call": 'get_user_info(user_email="<email>")'},
]

# ── TF-IDF(Term Frequency–Inverse Document Frequency) selector ───────────────────────────────────────────────────────

def _build_index():
    # saves all the queries in the EXAMPLE_DATABASE in a list
    # Purposes: Collects every 'query' string from our database to be indexed.
    queries = [example["query"] for example in EXAMPLE_DATABASE]
    # Creates and configures a tool to convert text into numerical vectors, ignoring common English words and using both single words and two-word phrases.
    # Purposes: Initializes the 'Translator' tool. Bigrams (1,2) ensure we catch phrases like "VPN down".
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    # Converts the list of queries into a matrix of TF-IDF scores
    # Purposes: Performs the heavy-lifting of turning every example query into a vector of numbers.
    tfidf_matrix = vectorizer.fit_transform(queries)
    return vectorizer, tfidf_matrix 
    # vec is a trained, reusable tool permanently stores all the knowledge learned from queries
    # mat is a matrix where the matrix the documents and the columes is the the unique terms from the vocabulary
    # the value of the matrix is the tf-idf score of the term in the document

# IDF(term) = log [ (Total number of documents + 1) / (Number of documents containing the term + 1) ] + 1

FS_VECTORIZER, FS_MATRIX = _build_index()


def select_examples(user_query: str, top_k: int = 4, min_score: float = 0.2) -> list[dict]:
    """
    Return up to top_k examples, but only if they meet a minimum
    cosine similarity score. This prevents irrelevant examples from being
    injected into the prompt and confusing the model.
    """
    # Purposes: Guard clause for empty or whitespace-only queries.
    if not user_query.strip():
        return []
        
    # takes the query text and converts it into a numerical vector.    
    # Purposes: Converts the USER'S query into the same mathematical format as the database.
    query_vector = FS_VECTORIZER.transform([user_query])

    # calculates how similar the new user query vector is to every example vector in our database
    # Purposes: Measures the 'Angle' between the user query and every database example.
    similarity_scores = cosine_similarity(query_vector, FS_MATRIX).flatten()
    
    # Get the indices of the top k examples
    # This sorts the scores but returns the original indices of the items in sorted order from highest score to lowest and takes the highest 2 examples
    # Purposes: Ranks the database by similarity and picks the top 'K' candidates.
    top_indices = similarity_scores.argsort()[::-1][:top_k]
    
    # --- The NEW Logic: Filter by score ---
    # Purposes: List to hold only the 'Quality' matches (above the threshold).
    final_examples = []
    for i in top_indices:
        # Only include the example if its score is above our threshold
        # Purposes: Discards 'garbage' matches that aren't actually relevant.
        if similarity_scores[i] >= min_score:
            final_examples.append(EXAMPLE_DATABASE[i])
            
    return final_examples



# ── Prompt template ───────────────────────────────────────────────────────

# Purposes: The 'Skeleton' of the prompt. Everything is fixed except the {examples_block}.
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
    # Purposes: Calls our selector 'Brain' to find the best examples for this specific query.
    examples = select_examples(user_query, top_k=top_k)
    lines: list[str] = []
    # Purposes: Loops through the chosen examples to format them for the prompt string.
    for i, ex in enumerate(examples, 1):
        lines.append(f'[{i}] User: "{ex["query"]}"')
        lines.append(f'    → {ex["tool_call"]}')
        lines.append("")
    # Purposes: Injects the formatted examples into the {examples_block} of our template.
    return _TEMPLATE.format(examples_block="\n".join(lines).strip())
    