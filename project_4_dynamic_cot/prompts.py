"""
project_4_dynamic_cot/prompts.py
==================================
Strategy: Dynamic Chain-of-Thought (CoT) Prompting  ← most advanced
------------------------------------------------------------------
Philosophy: the BEST of both worlds — examples are selected dynamically
(relevance from Exp 3) AND each one includes a full reasoning trace
(depth from Exp 2).

Design decisions:
  • COT_EXAMPLE_DATABASE holds 28 richly annotated entries. Each has:
      - query   : the example user message
      - thought : multi-line reasoning that works through the decision
      - tool_call: the correct action(s)
  • top_k=3 (not 4) because CoT examples are 3× longer than bare pairs —
    fewer examples keeps the prompt within a sensible context window.
  • The prompt template includes a "DIAGNOSTIC QUESTIONS" block that
    primes the model to emit its own reasoning trace even for queries
    that don't perfectly match any stored example.
  • Security examples explicitly model the two-step create→escalate
    sequence with reasoning that explains WHY both are needed.
  • Outage examples model the check_system_status → defer/ticket split.

Hypothesis: combining dynamic selection with CoT traces gives the highest
accuracy because the model sees BOTH a close-matching example AND a
demonstration of the reasoning process required to handle edge cases.
"""

from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── CoT Example Database ─────────────────────────────────────────────────────

COT_EXAMPLE_DATABASE: list[dict] = [
    # ── AUTH ──────────────────────────────────────────────────────────────
    {
        "query": "I forgot my password and I'm locked out of my computer",
        "thought": (
            "The user cannot authenticate — this is an active lockout.\n"
            "  • The self-service portal requires a valid session, so KB guidance is useless.\n"
            "  • reset_password is the direct action. Method = email (standard).\n"
            "  • No outage check needed; this is user-specific, not service-wide."
        ),
        "tool_call": 'reset_password(user_email="<email>", method="email")',
    },
    {
        "query": "My account keeps getting locked after a few failed attempts",
        "thought": (
            "Repeated lockouts suggest either a forgotten password or a device\n"
            "  caching stale credentials. Either way, the fix is a reset.\n"
            "  • reset_password via sms for higher assurance on repeated lockouts.\n"
            "  • After reset, the user should clear saved credentials."
        ),
        "tool_call": 'reset_password(user_email="<email>", method="sms")',
    },
    {
        "query": "How do I enroll in two-factor authentication?",
        "thought": (
            "This is a how-to question — the user is not locked out.\n"
            "  • MFA enrollment is fully documented in the knowledge base.\n"
            "  • lookup_knowledge_base is TIER 1 (self-service); no ticket needed.\n"
            "  • The user can follow the guide independently."
        ),
        "tool_call": 'lookup_knowledge_base(query="MFA two-factor authentication enrollment setup")',
    },

    # ── NETWORK ───────────────────────────────────────────────────────────
    {
        "query": "I can't connect to the VPN from home",
        "thought": (
            "VPN connectivity issues are common and well-documented.\n"
            "  • The KB covers AnyConnect config, DNS flush, credential issues.\n"
            "  • This is a single-user issue, not a floor-wide outage.\n"
            "  • TIER 1: surface the KB article before opening a ticket."
        ),
        "tool_call": 'lookup_knowledge_base(query="VPN connection remote home AnyConnect troubleshoot")',
    },
    {
        "query": "Nobody on our floor has any internet connection right now",
        "thought": (
            "Floor-wide loss of internet = infrastructure issue, not a user device.\n"
            "  • Opening individual tickets would create noise.\n"
            "  • check_system_status first — if there's a known incident, the team\n"
            "    just needs the ETA; a ticket is unnecessary.\n"
            "  • This is always the first move for multi-user outage reports."
        ),
        "tool_call": 'check_system_status(service_name="internet")',
    },
    {
        "query": "My Wi-Fi drops every hour in the office",
        "thought": (
            "Intermittent single-user Wi-Fi issues are documented.\n"
            "  • Re-enrolling on CORP-SECURE and checking 802.1X certs fixes most cases.\n"
            "  • TIER 1: KB first. This is self-serviceable."
        ),
        "tool_call": 'lookup_knowledge_base(query="Wi-Fi wireless dropping disconnecting office corporate")',
    },

    # ── HARDWARE ──────────────────────────────────────────────────────────
    {
        "query": "The display on my laptop keeps flickering",
        "thought": (
            "Screen flicker is almost always a GPU driver or refresh-rate issue.\n"
            "  • KB has step-by-step: update display drivers, set 60Hz refresh rate.\n"
            "  • 90% of cases are self-resolvable; no ticket needed upfront.\n"
            "  • If KB steps fail → user calls back → physical inspection ticket."
        ),
        "tool_call": 'lookup_knowledge_base(query="laptop screen flickering display driver refresh rate")',
    },
    {
        "query": "My laptop battery only lasts 40 minutes when fully charged",
        "thought": (
            "Battery degradation this severe = worn cell, not a software issue.\n"
            "  • There is no KB fix for a failing battery — physical swap required.\n"
            "  • create_ticket under 'hardware', medium priority.\n"
            "  • Technician will assess whether a swap or a new device is needed."
        ),
        "tool_call": 'create_ticket(category="hardware", priority="medium", summary="Laptop battery degraded — under 40 min runtime", user_email="<email>")',
    },
    {
        "query": "Can you book a RAM upgrade slot for my workstation?",
        "thought": (
            "The user is explicitly requesting a physical hardware upgrade.\n"
            "  • schedule_maintenance is the purpose-built tool for this.\n"
            "  • It books a workshop slot and sends confirmation — no ticket needed.\n"
            "  • maintenance_type = 'ram_upgrade'."
        ),
        "tool_call": 'schedule_maintenance(asset_id="<id>", maintenance_type="ram_upgrade", preferred_date="<date>", user_email="<email>")',
    },
    {
        "query": "My desktop is very slow and keeps freezing",
        "thought": (
            "Frequent freezing has documented causes: high RAM/CPU, full disk, OS corruption.\n"
            "  • KB covers Task Manager checks, SFC /scannow, Storage Sense.\n"
            "  • These are all self-resolvable steps the user can try first.\n"
            "  • TIER 1: lookup_knowledge_base before creating a ticket."
        ),
        "tool_call": 'lookup_knowledge_base(query="computer desktop slow freezing performance RAM CPU disk")',
    },
    {
        "query": "My laptop keeps overheating and shutting itself off",
        "thought": (
            "Thermal shutdown suggests a failing fan or dry heat paste.\n"
            "  • KB has basic tips (clean vents, power profile), worth surfacing.\n"
            "  • But persistent thermal shutdowns usually need physical inspection.\n"
            "  • Start with KB; a hardware ticket is likely the next step."
        ),
        "tool_call": 'lookup_knowledge_base(query="laptop overheating thermal shutdown fan hardware")',
    },
    {
        "query": "I need a screen replacement for my laptop",
        "thought": (
            "Screen replacement is unambiguously physical workshop work.\n"
            "  • schedule_maintenance with type='screen_replacement'.\n"
            "  • No KB lookup needed — user is explicitly requesting on-site service."
        ),
        "tool_call": 'schedule_maintenance(asset_id="<id>", maintenance_type="screen_replacement", preferred_date="<date>", user_email="<email>")',
    },

    # ── SOFTWARE ──────────────────────────────────────────────────────────
    {
        "query": "Excel crashes every time I try to open a file",
        "thought": (
            "Excel/M365 crashes are well-documented: Quick Repair → cache clear.\n"
            "  • TIER 1: KB has the exact fix. Self-serviceable.\n"
            "  • Only escalate to a ticket if Quick Repair and Online Repair both fail."
        ),
        "tool_call": 'lookup_knowledge_base(query="Microsoft Excel crash M365 Office Quick Repair")',
    },
    {
        "query": "I need Adobe Acrobat installed on my machine",
        "thought": (
            "Software installs require license procurement + Intune/SCCM deployment.\n"
            "  • No self-service path exists — this is always TIER 3.\n"
            "  • create_ticket under 'software', low priority (not blocking)."
        ),
        "tool_call": 'create_ticket(category="software", priority="low", summary="Software install request — Adobe Acrobat", user_email="<email>")',
    },
    {
        "query": "Outlook stopped receiving emails since this morning — could it be a server issue?",
        "thought": (
            "User is hinting at a server problem, not a client config issue.\n"
            "  • Before opening a ticket or KB lookup, check if email service is down.\n"
            "  • If check_system_status returns 'operational', THEN look at KB for\n"
            "    Outlook profile corruption or sync issues.\n"
            "  • Always rule out an outage first when the user mentions 'server'."
        ),
        "tool_call": 'check_system_status(service_name="email")',
    },
    {
        "query": "Microsoft Teams crashes every time I join a video call",
        "thought": (
            "Teams crash on call join = cache corruption or driver issue, both KB-documented.\n"
            "  • KB covers: clear Teams cache, update display/audio drivers.\n"
            "  • Self-serviceable. No ticket required upfront."
        ),
        "tool_call": 'lookup_knowledge_base(query="Microsoft Teams crash video call cache clear driver")',
    },

    # ── SECURITY ──────────────────────────────────────────────────────────
    {
        "query": "I received a suspicious email asking me to click a link and enter my credentials",
        "thought": (
            "This is an active phishing attempt — a security incident.\n"
            "  • Time is critical: if the user clicked the link, credentials may\n"
            "    already be compromised and the security team must act immediately.\n"
            "  • Step 1: create_ticket with category='security', priority='critical'\n"
            "    — this creates the audit trail and triggers SLA monitoring.\n"
            "  • Step 2: escalate_ticket to 'security-team' immediately.\n"
            "  • Both steps are REQUIRED — escalation alone doesn't create a ticket."
        ),
        "tool_call": ('create_ticket(category="security", priority="critical", '
                      'summary="Suspected phishing — user received credential-harvesting email", '
                      'user_email="<email>") '
                      '→ escalate_ticket(ticket_id="<ticket_id>", '
                      'reason="Active phishing attempt — possible credential compromise", '
                      'escalate_to="security-team")'),
    },
    {
        "query": "Files on my desktop have been renamed and I can't open them",
        "thought": (
            "Unexplained file renaming + inability to open = classic ransomware indicator.\n"
            "  • This is a critical security incident. Do NOT try KB steps.\n"
            "  • Step 1: create_ticket category='security', priority='critical'.\n"
            "  • Step 2: escalate_ticket to 'security-team' immediately.\n"
            "  • The device must be isolated; the security team handles the rest."
        ),
        "tool_call": ('create_ticket(category="security", priority="critical", '
                      'summary="Suspected ransomware — files renamed and inaccessible", '
                      'user_email="<email>") '
                      '→ escalate_ticket(ticket_id="<ticket_id>", '
                      'reason="Ransomware suspected — device requires isolation", '
                      'escalate_to="security-team")'),
    },

    # ── ACCESS ────────────────────────────────────────────────────────────
    {
        "query": "I need read access to the Legal department's SharePoint library",
        "thought": (
            "SharePoint access provisioning requires manager approval + IT action.\n"
            "  • No self-service path — this is always TIER 3.\n"
            "  • create_ticket under 'access', medium priority.\n"
            "  • The IT access team will request manager CC before provisioning."
        ),
        "tool_call": 'create_ticket(category="access", priority="medium", summary="Access request — Legal SharePoint read access", user_email="<email>")',
    },
    {
        "query": "My new hire starts Monday and needs an AD account and laptop",
        "thought": (
            "New hire provisioning = account creation + device assignment.\n"
            "  • This is a planned access + hardware task, not an emergency.\n"
            "  • create_ticket under 'access', high priority (time-sensitive — Monday).\n"
            "  • IT will coordinate with HR to complete setup before start date."
        ),
        "tool_call": 'create_ticket(category="access", priority="high", summary="New hire onboarding — AD account and laptop needed by Monday", user_email="<email>")',
    },

    # ── SYSTEM STATUS ─────────────────────────────────────────────────────
    {
        "query": "Is SharePoint currently experiencing any outages?",
        "thought": (
            "This is a direct outage inquiry — the user wants a live status check.\n"
            "  • check_system_status is always the first (and often only) action here.\n"
            "  • If status = 'outage', relay the ETA. If 'operational', investigate further."
        ),
        "tool_call": 'check_system_status(service_name="sharepoint")',
    },
    {
        "query": "The CRM has been throwing 500 errors for the whole team since 9am",
        "thought": (
            "Team-wide CRM errors since a specific time = likely service-level issue.\n"
            "  • check_system_status before creating individual tickets.\n"
            "  • If an incident is active, the error is already being worked.\n"
            "  • If status = 'operational', escalate to network-team for investigation."
        ),
        "tool_call": 'check_system_status(service_name="crm")',
    },

    # ── USER / ACCOUNT / BILLING ──────────────────────────────────────────
    {
        "query": "Can you look up the account details for alice.jones@company.com?",
        "thought": (
            "This is a directory lookup request.\n"
            "  • get_user_info returns department, manager, account status, devices.\n"
            "  • No ticket or KB lookup needed — this is a pure data retrieval."
        ),
        "tool_call": 'get_user_info(user_email="alice.jones@company.com")',
    },
    {
        "query": "Check the subscription tier for bob@company.com",
        "thought": (
            "This is a subscription/billing context lookup.\n"
            "  • lookup_user_account is purpose-built for subscription + block status.\n"
            "  • Use this (not get_user_info) when billing context is needed."
        ),
        "tool_call": 'lookup_user_account(email="bob@company.com")',
    },
    {
        "query": "Process a refund for reservation RES-00789",
        "thought": (
            "The user has a specific reservation ID and wants a refund.\n"
            "  • process_refund is the direct action. No ticket creation needed.\n"
            "  • If the refund fails (ID not found), THEN create a billing ticket."
        ),
        "tool_call": 'process_refund(reservation_id="RES-00789")',
    },

    # ── MEMORY / HISTORY ──────────────────────────────────────────────────
    {
        "query": "What issues has user jdoe contacted IT about before?",
        "thought": (
            "The user is asking for a quick history summary, not reporting a new issue.\n"
            "  • get_customer_history returns a brief past-issue summary.\n"
            "  • Use this for a fast context check before triaging the current request."
        ),
        "tool_call": 'get_customer_history(user_id="jdoe")',
    },
    {
        "query": "The issue is resolved. Please save it to the user's long-term history.",
        "thought": (
            "Post-resolution archiving — the ticket is closed.\n"
            "  • store_resolved_ticket is for brief, one-sentence summaries.\n"
            "  • Use save_ticket_to_long_term_memory if a full issue + resolution\n"
            "    detail is available. Choose based on available detail level."
        ),
        "tool_call": 'store_resolved_ticket(user_id="<user_id>", summary="<one-sentence summary>")',
    },
    {
        "query": "Archive the full outcome of this ticket including what fixed it",
        "thought": (
            "Full post-resolution archiving with both the problem and the fix.\n"
            "  • save_ticket_to_long_term_memory captures both summary AND resolution.\n"
            "  • This is richer than store_resolved_ticket and should be used when\n"
            "    the resolution steps are known and worth preserving."
        ),
        "tool_call": 'save_ticket_to_long_term_memory(user_id="<user_id>", summary="<summary>", resolution="<resolution>")',
    },
]

# ── TF-IDF selector ──────────────────────────────────────────────────────────

def _build_cot_index():
    queries = [ex["query"] for ex in COT_EXAMPLE_DATABASE]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform(queries)
    return vec, mat

_COT_VECTORIZER, _COT_MATRIX = _build_cot_index()


def select_cot_examples(user_query: str, top_k: int = 3) -> list[dict]:
    """
    Return the top_k COT_EXAMPLE_DATABASE entries whose stored query is most
    similar to user_query by TF-IDF cosine similarity.
    """
    qvec = _COT_VECTORIZER.transform([user_query])
    scores = cosine_similarity(qvec, _COT_MATRIX).flatten()
    indices = scores.argsort()[::-1][:top_k]
    return [COT_EXAMPLE_DATABASE[i] for i in indices]


# ── Prompt template ──────────────────────────────────────────────────────────

_COT_TEMPLATE = """\
You are a senior IT Helpdesk agent. Reason step-by-step through the user's
request, then call the single most appropriate tool (or sequence of tools
for multi-step incidents like security escalations).

AVAILABLE TOOLS:
  lookup_knowledge_base        – self-service KB articles (TIER 1 — try first)
  check_system_status          – live service status (check before outage tickets)
  reset_password               – password reset for locked-out users (TIER 2)
  process_refund               – billing refund (TIER 2)
  lookup_user_account          – subscription + account status (TIER 2)
  get_user_info                – AD directory lookup (TIER 2)
  create_ticket                – new helpdesk ticket (TIER 3)
  escalate_ticket              – escalate to specialist (TIER 3, after create_ticket)
  schedule_maintenance         – physical maintenance appointment (TIER 3)
  store_resolved_ticket        – archive brief resolved summary (TIER 4)
  save_ticket_to_long_term_memory – archive full issue + resolution (TIER 4)
  get_user_long_term_memory    – retrieve full user history (TIER 4)
  get_customer_history         – quick past-issues summary (TIER 4)

DIAGNOSTIC QUESTIONS — answer these before picking a tool:
  1. Is this AUTH / KB / OUTAGE / HARDWARE / SOFTWARE / SECURITY / ACCESS / BILLING / HISTORY?
  2. Can the user resolve this themselves with KB guidance? (→ TIER 1)
  3. Could this be a known outage? (→ check_system_status before anything else)
  4. Is there a security risk requiring immediate escalation? (→ create + escalate)
  5. Does this require physical work or manager approval? (→ TIER 3)

MOST RELEVANT WORKED EXAMPLES FOR THIS QUERY:
{examples_block}

Now reason through the user's request and call the correct tool(s).
"""


def build_system_prompt(user_query: str, top_k: int = 3) -> str:
    """
    Dynamically construct the CoT system prompt by selecting the most
    relevant reasoning-trace examples for the given user query.
    """
    examples = select_cot_examples(user_query, top_k=top_k)
    lines: list[str] = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"{'─'*60}")
        lines.append(f"Example {i}")
        lines.append(f"User: \"{ex['query']}\"")
        lines.append(f"Thought:\n  {ex['thought']}")
        lines.append(f"Action: {ex['tool_call']}")
        lines.append("")
    return _COT_TEMPLATE.format(examples_block="\n".join(lines).strip())
