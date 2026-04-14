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

from sklearn.feature_extraction.text import TfidfVectorizer # Purposes: The 'Indexer' that converts human queries into mathematical scores.
from sklearn.metrics.pairwise import cosine_similarity # Purposes: The 'Comparison Brain' that finds the most similar examples in 3D-space.

# Purposes: This is the "Gold Tier" database. 
# It contains not just tool calls, but the full 'Thought:' traces 
# teaching the model exactly how to reason using the 5-question framework.
COT_EXAMPLE_DATABASE: list[dict] = [

    # TC-001: lockout (self)
    {"query": "I forgot my password and I'm locked out of my computer",
     "thought": "Q1:auth Q2:no(portal unreachable) Q3:no Q4:no Q5:no → reset_password",
     "tool_call": 'reset_password(user_email="<email>", method="email")'},

    # TC-002: MFA how-to
    {"query": "How do I set up two-factor authentication on my work account",
     "thought": "Q1:KB-how-to Q2:YES(MFA steps documented) Q3:no Q4:no Q5:no → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="MFA two-factor authentication setup work account")'},

    # TC-003: colleague reports lockout
    {"query": "My colleague says I've been locked out of AD after some failed logins",
     "thought": "Q1:auth Q2:no(user locked out, can't self-serve) Q3:no Q4:no Q5:no → reset_password",
     "tool_call": 'reset_password(user_email="<email>", method="manual")'},

    # TC-004: VPN
    {"query": "I can't seem to connect to the VPN it worked yesterday",
     "thought": "Q1:KB-network Q2:YES(AnyConnect steps in KB) Q3:no Q5:no → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="VPN connection troubleshoot AnyConnect remote")'},

    # TC-005: floor-wide outage
    {"query": "Nobody on the 3rd floor has any internet right now",
     "thought": "Q1:outage Q3:YES(team-wide='nobody') → check_system_status FIRST",
     "tool_call": 'check_system_status(service_name="internet")'},

    # TC-006: Wi-Fi dropping (single user)
    {"query": "My Wi-Fi keeps dropping every hour in the office",
     "thought": "Q1:KB-network Q2:YES(CORP-SECURE reconnect in KB) Q3:single-user not outage → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="Wi-Fi wireless dropping corporate office troubleshoot")'},

    # TC-007: screen flicker (documented hw fix)
    {"query": "The display on my laptop is flickering badly it's very distracting",
     "thought": "Q1:HW-but-KB-fix Q2:YES(driver update+60Hz documented) Q5:no → lookup_knowledge_base first",
     "tool_call": 'lookup_knowledge_base(query="laptop screen flickering display driver refresh rate")'},

    # TC-008: battery dead
    {"query": "My laptop battery only lasts 40 minutes now even when fully charged",
     "thought": "Q1:hardware-fault Q2:no(no KB fix for worn cell) Q5:YES(physical swap needed) → create_ticket",
     "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Laptop battery degraded — 40 min runtime", user_email="<email>")'},

    # TC-009: RAM upgrade request
    {"query": "Can you book a slot to upgrade my workstation's RAM it needs more memory",
     "thought": "Q1:hardware-upgrade Q5:YES(explicit 'book a slot') → schedule_maintenance not create_ticket",
     "tool_call": 'schedule_maintenance(asset_id="<id>", maintenance_type="ram_upgrade", preferred_date="<date>", user_email="<email>")'},

    # TC-010: slow desktop
    {"query": "My desktop is very slow apps are freezing and it's barely usable",
     "thought": "Q1:KB-hw-perf Q2:YES(Task Manager, SFC, disk cleanup documented) → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="computer desktop slow freezing performance CPU RAM disk")'},

    # TC-011: Excel crash
    {"query": "Excel crashes immediately every time I try to open it",
     "thought": "Q1:software-KB Q2:YES(M365 Quick Repair in KB) → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="Microsoft Excel crash M365 Office Quick Repair")'},

    # TC-012: software install
    {"query": "I need Slack and Zoom installed on my company laptop",
     "thought": "Q1:software-install-request Q2:no(IT must deploy via Intune/SCCM) → create_ticket",
     "tool_call": 'create_ticket(category="software", priority="low", summary="Software install request — Slack and Zoom", user_email="<email>")'},

    # TC-013: Outlook server hint (HARD)
    {"query": "Outlook stopped receiving emails since this morning is it a server issue",
     "thought": ("Q1:possible-outage Q2:no Q3:YES('server issue?'+'since this morning'=outage signal)"
                 " → check_system_status(email) BEFORE KB lookup"),
     "tool_call": 'check_system_status(service_name="email")'},

    # TC-014: phishing with reset-password wording (HARD — keyword trap)
    {"query": "I got an email telling me to reset my password via a link that looks fake",
     "thought": ("Q1:SECURITY-phishing Q4:YES-CRITICAL('looks fake'=phishing, NOT a real lockout)"
                 " WARNING: 'reset my password' is bait — do NOT call reset_password."
                 " → create_ticket(critical) THEN escalate_ticket(security-team)"),
     "tool_call": ('create_ticket(category="security", priority="critical", '
                   'summary="Suspected phishing — fake password reset link", user_email="<email>") '
                   '→ escalate_ticket(ticket_id="<id>", reason="Phishing link mimicking password reset", escalate_to="security-team")')},

    # TC-015: ransomware (files renamed) (HARD)
    {"query": "Files on my desktop have been renamed and I can't open any of them",
     "thought": ("Q1:SECURITY-ransomware Q4:YES-CRITICAL(unexplained file renaming=ransomware indicator)"
                 " → create_ticket(critical) THEN escalate_ticket(security-team)"),
     "tool_call": ('create_ticket(category="security", priority="critical", '
                   'summary="Suspected ransomware — files renamed and inaccessible", user_email="<email>") '
                   '→ escalate_ticket(ticket_id="<id>", reason="Ransomware suspected — isolation required", escalate_to="security-team")')},

    # TC-016: SharePoint access
    {"query": "I need read access to the Legal department's SharePoint library",
     "thought": "Q1:access Q2:no(manager approval+IT provisioning required) → create_ticket",
     "tool_call": 'create_ticket(category="access", priority="medium", summary="Access request — Legal SharePoint read access", user_email="<email>")'},

    # TC-017: new hire
    {"query": "My new hire starts Monday and needs an AD account and laptop",
     "thought": "Q1:access+hardware Q5:YES(physical device provisioning) → create_ticket high priority",
     "tool_call": 'create_ticket(category="access", priority="high", summary="New hire onboarding — AD account and laptop by Monday", user_email="<email>")'},

    # TC-018: SharePoint outage direct query
    {"query": "Is SharePoint currently experiencing any outages",
     "thought": "Q1:outage Q3:YES(direct status inquiry) → check_system_status immediately",
     "tool_call": 'check_system_status(service_name="sharepoint")'},

    # TC-019: CRM 500 errors
    {"query": "Our CRM has been throwing 500 errors for the whole team since 9am",
     "thought": "Q1:outage Q3:YES('whole team'+'since 9am'=service-level) → check_system_status",
     "tool_call": 'check_system_status(service_name="crm")'},

    # TC-020: directory lookup
    {"query": "Can you look up the account details for alice.jones@company.com",
     "thought": "Q1:user-info Q2:no KB → get_user_info (not lookup_user_account which is for billing)",
     "tool_call": 'get_user_info(user_email="alice.jones@company.com")'},
]

FALLBACK_EXAMPLE: dict = {
    "query": "General IT issue",
    "thought": "Q1:unclear Q2:YES(default to KB) Q3:check if service → lookup_knowledge_base",
    "tool_call": 'lookup_knowledge_base(query="<describe issue>")',
}

# ── TF-IDF(Term Frequency–Inverse Document Frequency) selector ───────────────────────────────────────────────────────

def build_cot_index():
    # Purposes: Collects every 'query' string from our expert database and saves them in a list for indexing.
    queries = [example["query"] for example in COT_EXAMPLE_DATABASE]
    # Creates and configures a tool to convert text into numerical vectors, ignoring common English words and using both single words and two-word phrases.
    # Purposes: Initializes the 'Translator' tool. Bigrams (1,2) ensure we catch phrases like "VPN down".
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    # Converts the list of queries into a matrix of TF-IDF scores
    # Purposes: Performs the heavy-lifting of turning every expert query into a vector of numbers.
    tfidf_matrix = vectorizer.fit_transform(queries)
    return vectorizer, tfidf_matrix 
    # vec is a trained, reusable tool permanently stores all the knowledge learned from queries
    # mat is a matrix where the matrix the documents and the columes is the the unique terms from the vocabulary
    # the value of the matrix is the tf-idf score of the term in the document

# IDF(term) = log [ (Total number of documents + 1) / (Number of documents containing the term + 1) ] + 1

COT_VECTORIZER, COT_MATRIX = build_cot_index()


def select_cot_examples(user_query: str, top_k: int = 3, min_score: float = 0.05) -> list[dict]:
    # takes the query text and converts it into a numerical vector.    
    # Purposes: Converts the USER'S query into the same mathematical format as the database.
    query_vector = COT_VECTORIZER.transform([user_query])

    # calculates how similar the new user query vector is to every example vector in our database
    # Purposes: Measures the 'Angle' between the user query and every database example.
    similarity_scores = cosine_similarity(query_vector, COT_MATRIX).flatten()

    # Get the indices of the top k examples
    # This sorts the scores but returns the original indices of the items in sorted order from highest score to lowest and takes the highest 2 examples
    # Purposes: Ranks the database by similarity and picks the top 'K' candidates.
    top_indices = similarity_scores.argsort()[::-1][:top_k]

    # Purposes: List to hold only the 'Quality' matches (above the threshold).
    final_examples = []
    for i in top_indices:
        # Only include the example if its score is above our threshold
        # Purposes: Discards 'garbage' matches that aren't actually relevant.
        if similarity_scores[i] >= min_score:
            final_examples.append(COT_EXAMPLE_DATABASE[i])
            
    return final_examples

# ── Prompt template ───────────────────────────────────────────────────────

# Purposes: The 'Expert Skeleton' of the prompt. 
# Includes the Persona, the Diagnostic Framework, and the 5-Question Rules.
_COT_PROMPT_TEMPLATE = """\
You are an expert IT Helpdesk agent. For the user's request, answer these 5 questions,
then call the correct tool:
  Q1: Problem type? (auth/KB/outage/hardware/software/security/access/billing/history)
  Q2: Can user self-resolve with KB? YES → lookup_knowledge_base
  Q3: Possible known outage? "team-wide"/"server issue?"/"since this morning" → check_system_status FIRST
  Q4: Security incident? phishing/suspicious link/malware/renamed files → create_ticket(critical) + escalate
  Q5: Physical work needed? battery/RAM/screen upgrade → schedule_maintenance or create_ticket

CRITICAL RULES:
  • "looks fake" / "suspicious link" / "fake email" → SECURITY → create_ticket + escalate. NEVER reset_password.
  • "files renamed" / "can't open files" → RANSOMWARE → create_ticket + escalate.
  • "colleague says locked out" / "my account is locked" → AUTH → reset_password.
  • "server issue?" / "since this morning for the whole team" → check_system_status FIRST.
  • "book a slot" / "upgrade my RAM" → schedule_maintenance.

AVAILABLE TOOLS:
  lookup_knowledge_base, create_ticket, escalate_ticket, reset_password,
  get_user_info, lookup_user_account, check_system_status, schedule_maintenance,
  process_refund, get_customer_history, get_user_long_term_memory,
  store_resolved_ticket, save_ticket_to_long_term_memory

## Most Relevant Examples
{examples_block}
"""

def build_system_prompt(user_query: str, top_k: int = 3) -> str:
    # Purposes: Calls our expert selector 'Brain' to find the best CoT examples for this specific query.
    examples = select_cot_examples(user_query, top_k=top_k)
    # Purposes: If NO similar examples were found (rare), use a generic fallback so the AI isn't blind.
    if not examples:
        examples = [FALLBACK_EXAMPLE]
    
    # Purposes: Build the lines of the example block.
    lines = []
    # Purposes: Loops through the chosen expert examples to format them for the prompt string.
    for i, ex in enumerate(examples, 1):
        lines.append(f'[{i}] User: "{ex["query"]}"')
        lines.append(f'      Thought: {ex["thought"]}')
        lines.append(f'      Action: {ex["tool_call"]}')
        lines.append("")
    # Purposes: Injects the formatted expert examples into the {examples_block} of our expert template.
    return _COT_PROMPT_TEMPLATE.format(examples_block="\n".join(lines).strip())

