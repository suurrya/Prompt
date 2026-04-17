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

    # ── HARDWARE extras ──────────────────────────────────────────────────────

    # Cracked screen (physical damage → ticket)
    {"query": "I dropped my laptop and the screen cracked the display is completely black now",
     "thought": "Q1:hardware-physical-damage Q2:no(no KB fix for cracked screen) Q5:YES(physical repair) → create_ticket high",
     "tool_call": 'create_ticket(category="hardware", priority="high", summary="Laptop screen cracked — display completely black after drop", user_email="<email>")'},

    # Cracked screen — explicit booking
    {"query": "My laptop screen has dead pixels can you book a screen replacement appointment",
     "thought": "Q1:hardware-upgrade Q5:YES(explicit 'book a … replacement') → schedule_maintenance not create_ticket",
     "tool_call": 'schedule_maintenance(asset_id="<id>", maintenance_type="screen_replacement", preferred_date="<date>", user_email="<email>")'},

    # Keyboard broken
    {"query": "Half the keys on my laptop keyboard have stopped responding",
     "thought": "Q1:hardware-fault Q2:no(partial keyboard failure=physical defect, not a driver issue) Q5:YES(physical repair) → create_ticket",
     "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Laptop keyboard keys unresponsive — hardware fault", user_email="<email>")'},

    # Keyboard unresponsive after spill
    {"query": "I spilled water on my keyboard and now several keys don't work",
     "thought": "Q1:hardware-physical-damage Q2:no Q5:YES → create_ticket medium",
     "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Keyboard damaged by liquid spill — keys unresponsive", user_email="<email>")'},

    # Docking station fault
    {"query": "My docking station stopped charging my laptop and the USB ports don't work",
     "thought": "Q1:hardware-fault Q2:no(docking station physical fault, not driver) Q5:YES → create_ticket",
     "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Docking station fault — no charging or USB output", user_email="<email>")'},

    # External monitor — no signal (KB)
    {"query": "My external monitor says No Signal even though it's plugged in",
     "thought": "Q1:hardware-KB Q2:YES(HDMI/DP re-seat + display settings documented) Q5:no → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="external monitor no signal HDMI DisplayPort connection troubleshoot")'},

    # External monitor — resolution wrong (KB)
    {"query": "My second monitor is displaying at the wrong resolution and I can't change it",
     "thought": "Q1:hardware-KB Q2:YES(display settings + driver steps in KB) → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="external monitor wrong resolution display settings driver")'},

    # Battery swap explicit booking
    {"query": "My battery is swelling and needs to be replaced can you schedule a battery swap",
     "thought": "Q1:hardware-safety-upgrade Q5:YES(user explicitly says 'schedule a battery swap') → schedule_maintenance",
     "tool_call": 'schedule_maintenance(asset_id="<id>", maintenance_type="battery_replacement", preferred_date="<date>", user_email="<email>")'},

    # USB device not recognised (KB)
    {"query": "My USB drive isn't being recognised when I plug it into my laptop",
     "thought": "Q1:hardware-KB Q2:YES(Device Manager + driver reinstall in KB) → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="USB drive not recognised device manager driver troubleshoot")'},

    # ── SOFTWARE extras ──────────────────────────────────────────────────────

    # Teams audio/video (KB)
    {"query": "My camera and microphone don't work in Microsoft Teams calls",
     "thought": "Q1:software-KB Q2:YES(Teams A/V settings + driver update documented) Q3:single-user, not department-wide → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="Microsoft Teams camera microphone not working audio video settings")'},

    # Teams frozen in calls (KB)
    {"query": "Teams freezes every time I join a video call and I have to restart it",
     "thought": "Q1:software-KB Q2:YES(Teams cache clear + graphics driver steps in KB) Q3:single-user → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="Microsoft Teams freezing video call crash restart fix")'},

    # Teams down for whole department (outage)
    {"query": "Teams has been completely unavailable for our whole department this afternoon",
     "thought": "Q1:outage Q3:YES('whole department'=service-level) → check_system_status FIRST",
     "tool_call": 'check_system_status(service_name="teams")'},

    # OneDrive sync error (KB)
    {"query": "OneDrive keeps showing a sync error and none of my files are uploading",
     "thought": "Q1:software-KB Q2:YES(sign-out/re-sign-in, selective sync, cache clear in KB) Q3:single-user → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="OneDrive sync error files not uploading troubleshoot reset")'},

    # OneDrive personal account conflict (KB)
    {"query": "OneDrive is mixing my personal and work files and I can't separate them",
     "thought": "Q1:software-KB Q2:YES(unlink personal account, selective sync documented) → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="OneDrive personal work account conflict unlink selective sync")'},

    # Printer offline — whole floor (outage)
    {"query": "None of us can print to the main printer on our floor this morning",
     "thought": "Q1:outage Q3:YES('none of us'=team-wide shared resource) → check_system_status FIRST",
     "tool_call": 'check_system_status(service_name="printer")'},

    # Printer driver missing (KB)
    {"query": "My laptop can't find the office printer it was working last week",
     "thought": "Q1:software-KB Q2:YES(printer driver install + network printer add documented) Q3:single-user → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="office printer not found driver install network printer troubleshoot")'},

    # Admin rights install (ticket)
    {"query": "I need admin rights to install a tool required for my project",
     "thought": "Q1:access-software Q2:no(IT must evaluate+approve admin rights) → create_ticket access medium",
     "tool_call": 'create_ticket(category="access", priority="medium", summary="Admin rights request — tool installation for project", user_email="<email>")'},

    # How-to mail merge (KB)
    {"query": "How do I do a mail merge in Microsoft Word",
     "thought": "Q1:KB-how-to Q2:YES(Word mail merge wizard documented) Q3-5:no → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="Microsoft Word mail merge steps tutorial")'},

    # ── NETWORK extras ──────────────────────────────────────────────────────

    # Building-wide internet slow (outage)
    {"query": "The internet has been really slow for everyone in the building since lunch",
     "thought": "Q1:outage Q3:YES('everyone in the building'=infrastructure-level) → check_system_status FIRST",
     "tool_call": 'check_system_status(service_name="internet")'},

    # Ethernet not working (KB)
    {"query": "My Ethernet cable is plugged in but my laptop says no network connection",
     "thought": "Q1:network-KB Q2:YES(ipconfig renew + driver reinstall documented) Q3:single-user → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="Ethernet cable no network connection laptop troubleshoot")'},

    # Browser proxy issue (KB)
    {"query": "I can ping websites but my browser says unable to connect to everything",
     "thought": "Q1:network-KB Q2:YES(proxy settings + Winsock reset documented) Q3:single-user → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="browser proxy settings unable to connect network troubleshoot")'},

    # VPN config how-to (KB)
    {"query": "How do I configure my VPN client to use the new server",
     "thought": "Q1:KB-how-to Q2:YES(AnyConnect server profile setup documented) Q3:no → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="VPN client configuration new server AnyConnect setup")'},

    # VPN profile for travel (ticket)
    {"query": "I'm travelling to Germany next week and need a VPN profile for that region",
     "thought": "Q1:access Q2:no(regional VPN profile requires IT to provision) → create_ticket access medium",
     "tool_call": 'create_ticket(category="access", priority="medium", summary="VPN profile request — Germany region for travel", user_email="<email>")'},

    # ── AUTH extras ─────────────────────────────────────────────────────────

    # Password expired (reset_password)
    {"query": "My password expired overnight and now I can't log in at all",
     "thought": "Q1:auth Q2:no(expired password blocks portal) Q3:no Q4:no → reset_password",
     "tool_call": 'reset_password(user_email="<email>", method="email")'},

    # SSO not loading for whole office (outage)
    {"query": "The SSO login page isn't loading for anyone in the office",
     "thought": "Q1:outage Q3:YES('anyone in the office'=company-wide SSO failure) → check_system_status FIRST",
     "tool_call": 'check_system_status(service_name="sso")'},

    # Lost MFA device (ticket — NOT reset_password)
    {"query": "I lost my phone and now I can't pass the MFA check I'm locked out",
     "thought": ("Q1:access-MFA-recovery Q2:no(user can't self-serve without MFA device)"
                 " RULE: do NOT call reset_password — a password reset won't fix missing MFA device."
                 " → create_ticket(access, high) so IT can manually bypass MFA"),
     "tool_call": 'create_ticket(category="access", priority="high", summary="MFA bypass needed — user lost MFA device and is locked out", user_email="<email>")'},

    # ── SECURITY extras ──────────────────────────────────────────────────────

    # Suspicious USB found (security)
    {"query": "I found a USB stick plugged into my workstation that I definitely didn't put there",
     "thought": ("Q1:SECURITY-hardware-implant Q4:YES-CRITICAL(unknown USB=possible exfiltration device)"
                 " → create_ticket(critical) THEN escalate_ticket(security-team). Do NOT touch device."),
     "tool_call": ('create_ticket(category="security", priority="critical", '
                   'summary="Suspicious USB device found in workstation", user_email="<email>") '
                   '→ escalate_ticket(ticket_id="<id>", reason="Possible hardware implant or data exfiltration attempt", escalate_to="security-team")')},

    # Accidental data leak (security)
    {"query": "I accidentally emailed a file with customer data to my personal Gmail account",
     "thought": ("Q1:SECURITY-data-breach Q4:YES-CRITICAL(customer data exfiltrated even if accidental)"
                 " → create_ticket(critical) THEN escalate_ticket for compliance notification"),
     "tool_call": ('create_ticket(category="security", priority="critical", '
                   'summary="Data leak — customer data emailed to personal Gmail", user_email="<email>") '
                   '→ escalate_ticket(ticket_id="<id>", reason="Potential data breach — personal email exfiltration", escalate_to="security-team")')},

    # Unknown process high CPU (KB first)
    {"query": "There's a process called svchost32.exe using 100% CPU that I don't recognise",
     "thought": "Q1:security-suspicious-process Q2:YES(Task Manager + malware check steps in KB) Q4:possible but KB first → lookup_knowledge_base",
     "tool_call": 'lookup_knowledge_base(query="unknown suspicious process CPU usage malware check Task Manager")'},

    # ── ACCESS extras ────────────────────────────────────────────────────────

    # Shared drive write access (ticket)
    {"query": "I need write access to the Finance shared drive on the server",
     "thought": "Q1:access Q2:no(shared drive permissions require manager approval + IT action) → create_ticket access medium",
     "tool_call": 'create_ticket(category="access", priority="medium", summary="Access request — Finance shared drive write access", user_email="<email>")'},

    # Production database access (ticket)
    {"query": "I need read access to the production database for my reporting work",
     "thought": "Q1:access-database Q2:no(DBA + manager approval required) → create_ticket access medium",
     "tool_call": 'create_ticket(category="access", priority="medium", summary="Database access request — production read access for reporting", user_email="<email>")'},

    # ── SYSTEM STATUS extras ─────────────────────────────────────────────────

    # Teams direct status query
    {"query": "Is Microsoft Teams having any issues right now",
     "thought": "Q1:outage Q3:YES(direct status inquiry) → check_system_status immediately",
     "tool_call": 'check_system_status(service_name="teams")'},

    # ERP down for team (outage)
    {"query": "The ERP system isn't loading for any of us in the finance team",
     "thought": "Q1:outage Q3:YES('any of us'=team-wide core business system) → check_system_status FIRST",
     "tool_call": 'check_system_status(service_name="erp")'},

    # ERP 500 errors (outage)
    {"query": "Our ERP has been throwing errors for the whole finance department since this morning",
     "thought": "Q1:outage Q3:YES('whole finance department'+'since this morning'=service-level) → check_system_status",
     "tool_call": 'check_system_status(service_name="erp")'},

    # ── BILLING / USER extras ────────────────────────────────────────────────

    # Own account info (get_user_info)
    {"query": "Can you look up my own account details and what devices I'm assigned",
     "thought": "Q1:user-info Q2:no KB → get_user_info with user's own email (not lookup_user_account)",
     "tool_call": 'get_user_info(user_email="<email>")'},

    # Refund request
    {"query": "I was charged incorrectly my reservation ID is RES-00456",
     "thought": "Q1:billing-refund Q2:no → process_refund with the provided reservation ID",
     "tool_call": 'process_refund(reservation_id="RES-00456")'},

    # Subscription lookup
    {"query": "Check the subscription status for bob@company.com",
     "thought": "Q1:billing-subscription Q2:no → lookup_user_account (billing context, not get_user_info)",
     "tool_call": 'lookup_user_account(email="bob@company.com")'},

    # Past issues history
    {"query": "What issues has user jdoe had before",
     "thought": "Q1:history Q2:no → get_customer_history to retrieve past tickets",
     "tool_call": 'get_customer_history(user_id="jdoe")'},
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
COT_PROMPT_TEMPLATE = """\
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
    return COT_PROMPT_TEMPLATE.format(examples_block="\n".join(lines).strip())

