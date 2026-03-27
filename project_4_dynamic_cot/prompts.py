"""
project_4_dynamic_cot/prompts.py
==================================
Dynamic Chain-of-Thought (CoT) Prompting  ← most advanced strategy
--------------------------------------------------------------------
Combines the two enhancements from Experiments 2 and 3:

  • DYNAMIC selection — at call-time, the top-k most relevant examples are
    chosen via TF-IDF cosine similarity (same mechanism as Experiment 3).

  • CHAIN-OF-THOUGHT traces — each example includes a "Thought:" section
    that models the step-by-step reasoning process (same structure as
    Experiment 2).

Expected outcome: the model receives examples that are BOTH semantically
close to the actual query AND demonstrate explicit deliberative reasoning,
giving it the best possible scaffold for accurate tool selection.
"""

from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------------------------------------------------------
# CoT Example Database
# Each entry: query text  +  a Thought: trace  +  the final tool call.
# ---------------------------------------------------------------------------
COT_EXAMPLE_DATABASE: list[dict] = [
    # Auth / Passwords
    {
        "query": "I forgot my password and can't log in",
        "thought": (
            "The user cannot authenticate — this is a credential issue. "
            "The self-service portal requires an active session, so KB guidance is not enough. "
            "I must initiate a password reset directly via the reset_password tool."
        ),
        "tool_call": 'reset_password(user_email="<user_email>", method="email")',
    },
    {
        "query": "My account is locked after too many wrong password attempts",
        "thought": (
            "Account lockout after repeated failures is a standard security policy response. "
            "The resolution is unlocking via a password reset — the user cannot self-serve "
            "because they are already locked out of the portal."
        ),
        "tool_call": 'reset_password(user_email="<user_email>", method="sms")',
    },
    {
        "query": "How do I enrol in two-factor authentication?",
        "thought": (
            "MFA enrolment is a how-to question with well-documented steps in the knowledge base. "
            "The user is not locked out — they just need guidance. "
            "lookup_knowledge_base is the right first action."
        ),
        "tool_call": 'lookup_knowledge_base(query="MFA two-factor authentication setup enrolment")',
    },
    # Network / VPN
    {
        "query": "I can't connect to the VPN from home",
        "thought": (
            "VPN connectivity issues are common and covered by documented troubleshooting steps "
            "(AnyConnect config, DNS flush, credentials). "
            "I should surface the KB article before raising a ticket — the user can likely self-fix."
        ),
        "tool_call": 'lookup_knowledge_base(query="VPN connection remote home troubleshoot")',
    },
    {
        "query": "The office Wi-Fi keeps disconnecting on my laptop",
        "thought": (
            "Intermittent Wi-Fi is documented — re-enrol on CORP-SECURE, check 802.1X cert. "
            "KB guidance covers this. No hands-on work needed yet."
        ),
        "tool_call": 'lookup_knowledge_base(query="Wi-Fi wireless network disconnecting corporate office")',
    },
    {
        "query": "The internet on our entire floor seems to be down",
        "thought": (
            "A floor-wide outage suggests infrastructure rather than a user-device issue. "
            "Before creating a ticket I should check if this is a known incident. "
            "check_system_status for 'internet' is the right first move."
        ),
        "tool_call": 'check_system_status(service_name="internet")',
    },
    # Hardware
    {
        "query": "My laptop screen keeps flickering",
        "thought": (
            "Screen flickering is typically a driver or refresh-rate issue — documented in KB. "
            "Directing the user to driver-update steps is faster than opening a ticket. "
            "If the KB steps don't help, they can call back for physical inspection."
        ),
        "tool_call": 'lookup_knowledge_base(query="laptop screen flickering display driver refresh rate")',
    },
    {
        "query": "My printer shows offline and won't print anything",
        "thought": (
            "Printer offline can sometimes be fixed by restarting the spooler (KB-resolvable). "
            "However, office printers are shared infrastructure — if the spooler fix fails, "
            "a technician may need to intervene. I'll check the KB first but the ticket "
            "may still be needed after the user tries self-help."
        ),
        "tool_call": 'lookup_knowledge_base(query="printer offline spooler fix troubleshoot")',
    },
    {
        "query": "My computer freezes every few minutes and is barely usable",
        "thought": (
            "Frequent freezing has multiple common causes: high RAM/CPU, full disk, OS corruption. "
            "These all have KB-documented remediation steps. "
            "I should surface those first before creating a ticket."
        ),
        "tool_call": 'lookup_knowledge_base(query="computer freezing slow performance unresponsive RAM CPU")',
    },
    {
        "query": "My laptop battery dies in under an hour even when fully charged",
        "thought": (
            "Battery degradation to under an hour is a hardware fault — likely a worn cell. "
            "This cannot be resolved remotely. The user will need a battery swap. "
            "Create a ticket under hardware so a technician can assess and schedule a replacement."
        ),
        "tool_call": (
            'create_ticket(category="hardware", priority="medium", '
            'summary="Laptop battery degraded — drains in under one hour", '
            'user_email="<user_email>")'
        ),
    },
    {
        "query": "I'd like to schedule a RAM upgrade for my workstation",
        "thought": (
            "The user is explicitly requesting a hardware upgrade — this is physical work. "
            "schedule_maintenance is the correct tool: it books a slot at the IT workshop "
            "and notifies the user. No ticket creation is needed; maintenance IS the ticket."
        ),
        "tool_call": (
            'schedule_maintenance(asset_id="<asset_id>", maintenance_type="ram_upgrade", '
            'preferred_date="<date>", user_email="<user_email>")'
        ),
    },
    # Software
    {
        "query": "Microsoft Excel crashes every time I try to open a file",
        "thought": (
            "Excel crashing is a common M365 issue — usually resolved by a Quick Repair "
            "or cache clear. KB has step-by-step guidance. "
            "Point the user to self-service before creating a ticket."
        ),
        "tool_call": 'lookup_knowledge_base(query="Microsoft Excel crash M365 Office repair")',
    },
    {
        "query": "I need Adobe Acrobat Pro installed on my laptop",
        "thought": (
            "Software installation requires license procurement and Intune/SCCM deployment. "
            "This is not self-serviceable — the IT portal must approve and deploy it. "
            "Create a ticket under 'software' with low priority."
        ),
        "tool_call": (
            'create_ticket(category="software", priority="low", '
            'summary="Software installation request — Adobe Acrobat Pro", '
            'user_email="<user_email>")'
        ),
    },
    {
        "query": "Outlook is not receiving any emails since this morning",
        "thought": (
            "Before assuming a client-side issue, I should check if the email service is down. "
            "If check_system_status shows 'operational', then the problem is likely a corrupted "
            "profile or sync issue — and the KB has steps for that. "
            "I'll check system status first."
        ),
        "tool_call": 'check_system_status(service_name="email")',
    },
    # Teams / Collaboration
    {
        "query": "Teams crashes whenever I join a video call",
        "thought": (
            "Teams crash on calls is a known issue — usually fixed by clearing the cache "
            "or updating drivers. The KB has these steps. Self-service is likely sufficient."
        ),
        "tool_call": 'lookup_knowledge_base(query="Microsoft Teams video call crash cache clear")',
    },
    {
        "query": "My microphone is not working in Teams meetings",
        "thought": (
            "Microphone issues in Teams are almost always a device-permission or device-selection "
            "problem in Teams Settings → Devices, or a Windows privacy setting. "
            "The KB covers this precisely. Direct the user to self-service."
        ),
        "tool_call": 'lookup_knowledge_base(query="Teams microphone not working permissions settings")',
    },
    # Security
    {
        "query": "I received a suspicious email asking me to click a link and enter credentials",
        "thought": (
            "This is a phishing attempt — a security incident. "
            "Time is critical: if the user clicked the link, credentials may already be compromised. "
            "I must (1) create a priority critical ticket for the audit trail, "
            "then (2) immediately escalate to the security team. "
            "These two actions should happen in sequence."
        ),
        "tool_call": (
            'create_ticket(category="security", priority="critical", '
            'summary="Suspected phishing email — credential compromise risk", '
            'user_email="<user_email>") '
            '→ escalate_ticket(ticket_id="<ticket_id>", '
            'reason="Active phishing attempt", escalate_to="security-team")'
        ),
    },
    {
        "query": "I think my machine has been infected with ransomware",
        "thought": (
            "Ransomware is a critical security incident. "
            "The device must be isolated immediately. "
            "I need to create a critical ticket and escalate to the security team at once. "
            "This is not a KB-resolvable situation."
        ),
        "tool_call": (
            'create_ticket(category="security", priority="critical", '
            'summary="Suspected ransomware infection — device requires immediate isolation", '
            'user_email="<user_email>") '
            '→ escalate_ticket(ticket_id="<ticket_id>", '
            'reason="Ransomware suspected", escalate_to="security-team")'
        ),
    },
    # Access
    {
        "query": "I need access to the Finance department shared drive",
        "thought": (
            "Shared drive access requires manager approval and manual provisioning — "
            "there is no self-service path. "
            "Create a ticket under 'access' with medium priority. "
            "The IT access team will request manager approval."
        ),
        "tool_call": (
            'create_ticket(category="access", priority="medium", '
            'summary="Access request — Finance shared drive", '
            'user_email="<user_email>")'
        ),
    },
    {
        "query": "My new colleague needs access to the Marketing SharePoint site",
        "thought": (
            "Provisioning access for a colleague is an access management task. "
            "Create a ticket; the requester's manager must approve before IT provisions access."
        ),
        "tool_call": (
            'create_ticket(category="access", priority="medium", '
            'summary="Access request — Marketing SharePoint for new colleague", '
            'user_email="<user_email>")'
        ),
    },
    # System Status
    {
        "query": "SharePoint seems to be down — nobody on my team can access files",
        "thought": (
            "A team-wide SharePoint issue suggests a service-level problem, not a user-device problem. "
            "I should check_system_status before creating individual tickets — "
            "if there is an active incident, the team just needs to wait for the ETA."
        ),
        "tool_call": 'check_system_status(service_name="sharepoint")',
    },
    {
        "query": "Our CRM has been throwing errors all morning",
        "thought": (
            "CRM errors affecting multiple users imply a possible service outage. "
            "check_system_status for 'crm' will confirm whether this is a known incident. "
            "If not, I'll create a ticket with the error details."
        ),
        "tool_call": 'check_system_status(service_name="crm")',
    },
    # Onboarding
    {
        "query": "I'm a new starter and need help setting up my work laptop",
        "thought": (
            "New employee onboarding is fully documented in the KB with a step-by-step checklist. "
            "Pointing the user to the onboarding article is faster than creating a ticket "
            "and empowers them to self-configure with the guide."
        ),
        "tool_call": 'lookup_knowledge_base(query="new employee IT onboarding laptop setup checklist")',
    },
    # User Info
    {
        "query": "Can you pull up the account details for jane.doe@company.com?",
        "thought": (
            "This is a directory lookup request. "
            "get_user_info is designed exactly for this: "
            "it returns department, manager, account status, and assigned devices."
        ),
        "tool_call": 'get_user_info(user_email="jane.doe@company.com")',
    },
]

# ---------------------------------------------------------------------------
# TF-IDF selector (same mechanism as Experiment 3)
# ---------------------------------------------------------------------------

def _build_cot_corpus() -> tuple[list[str], TfidfVectorizer, np.ndarray]:
    queries = [ex["query"] for ex in COT_EXAMPLE_DATABASE]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(queries)
    return queries, vectorizer, matrix


_COT_CORPUS_QUERIES, _COT_VECTORIZER, _COT_CORPUS_MATRIX = _build_cot_corpus()


def select_cot_examples(user_query: str, top_k: int = 3) -> list[dict]:
    """
    Return the `top_k` CoT examples whose stored query is most similar
    to `user_query` by TF-IDF cosine similarity.
    """
    query_vec = _COT_VECTORIZER.transform([user_query])
    scores = cosine_similarity(query_vec, _COT_CORPUS_MATRIX).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [COT_EXAMPLE_DATABASE[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_COT_PROMPT_TEMPLATE = """\
You are an expert IT Helpdesk assistant. Your role is to reason carefully about
the user's problem step-by-step and then call the single most appropriate tool.

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

## How to Reason (Chain-of-Thought)
For every request, think through these questions before choosing a tool:
  • What is the core problem type? (auth / hardware / software / network / security / access / outage)
  • Can the user self-resolve with KB guidance?
  • Could this be a known outage? (check_system_status first)
  • Does this require physical/hands-on IT work?
  • Is there a security risk that demands immediate escalation?

## Most Relevant Examples for This Query (with Reasoning Traces)
{examples_block}

Now reason through the user's request and call the correct tool.
"""


def build_system_prompt(user_query: str, top_k: int = 3) -> str:
    """
    Dynamically construct the CoT system prompt by selecting the most
    relevant reasoning-trace examples for the given user query.

    Args:
        user_query: The raw incoming user message.
        top_k: Number of CoT examples to inject.

    Returns:
        A fully assembled system prompt string with dynamic CoT examples.
    """
    examples = select_cot_examples(user_query, top_k=top_k)
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"---\nExample {i}")
        lines.append(f'User: "{ex["query"]}"')
        lines.append(f"Thought:\n  {ex['thought']}")
        lines.append(f"Action: {ex['tool_call']}")
        lines.append("")
    examples_block = "\n".join(lines).strip()
    return _COT_PROMPT_TEMPLATE.format(examples_block=examples_block)
