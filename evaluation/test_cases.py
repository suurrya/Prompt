"""
evaluation/test_cases.py
=========================
Ground-truth test suite for the four IT Helpdesk agents.

Each test case is a dict with:
    id          – unique identifier for the test
    query       – the raw user message sent to the agent
    expected_tool – the single tool name that should be the FIRST tool called
    category    – query category for segmented analysis
    difficulty  – 'easy', 'medium', or 'hard' (subjective labelling)
    notes       – why this case is interesting / where agents might fail

Coverage:
  • 20 distinct scenarios spread across all tool types
  • Mix of obvious, ambiguous, and multi-step cases
  • Includes edge cases that separate CoT from non-CoT agents
"""

TEST_CASES: list[dict] = [
    # ── Authentication / Password (3 cases) ──────────────────────────────
    {
        "id": "TC-001",
        "query": "I forgot my password and I'm locked out of my computer.",
        "expected_tool": "reset_password",
        "category": "auth",
        "difficulty": "easy",
        "notes": "Clear-cut lockout → reset_password. Agents that surface KB first fail here.",
    },
    {
        "id": "TC-002",
        "query": "How do I set up two-factor authentication on my work account?",
        "expected_tool": "lookup_knowledge_base",
        "category": "auth",
        "difficulty": "easy",
        "notes": "How-to question — KB is correct; reset_password is wrong.",
    },
    {
        "id": "TC-003",
        "query": "My colleague says I've been locked out of AD after some failed logins.",
        "expected_tool": "reset_password",
        "category": "auth",
        "difficulty": "medium",
        "notes": "Indirect lockout report — still requires reset_password, not a ticket.",
    },
    # ── Network (3 cases) ─────────────────────────────────────────────────
    {
        "id": "TC-004",
        "query": "I can't seem to connect to the VPN. It worked yesterday.",
        "expected_tool": "lookup_knowledge_base",
        "category": "network",
        "difficulty": "easy",
        "notes": "Classic VPN troubleshooting — KB covers Cisco AnyConnect steps.",
    },
    {
        "id": "TC-005",
        "query": "Nobody on the 3rd floor has any internet right now.",
        "expected_tool": "check_system_status",
        "category": "network",
        "difficulty": "medium",
        "notes": "Floor-wide outage → check_system_status before creating tickets.",
    },
    {
        "id": "TC-006",
        "query": "My Wi-Fi keeps dropping every hour in the office.",
        "expected_tool": "lookup_knowledge_base",
        "category": "network",
        "difficulty": "easy",
        "notes": "Single-user intermittent Wi-Fi → KB self-service guidance.",
    },
    # ── Hardware (4 cases) ────────────────────────────────────────────────
    {
        "id": "TC-007",
        "query": "The display on my laptop is flickering badly. It's very distracting.",
        "expected_tool": "lookup_knowledge_base",
        "category": "hardware",
        "difficulty": "easy",
        "notes": "Driver/refresh-rate fix is in the KB — self-service first.",
    },
    {
        "id": "TC-008",
        "query": "My laptop battery only lasts 40 minutes now even when fully charged.",
        "expected_tool": "create_ticket",
        "category": "hardware",
        "difficulty": "medium",
        "notes": "Battery degradation requires physical inspection/swap — ticket.",
    },
    {
        "id": "TC-009",
        "query": "Can you book a slot to upgrade my workstation's RAM? It needs more memory.",
        "expected_tool": "schedule_maintenance",
        "category": "hardware",
        "difficulty": "medium",
        "notes": "Explicit upgrade request → schedule_maintenance, not a plain ticket.",
    },
    {
        "id": "TC-010",
        "query": "My desktop is very slow, apps are freezing and it's barely usable.",
        "expected_tool": "lookup_knowledge_base",
        "category": "hardware",
        "difficulty": "easy",
        "notes": "Performance issues have KB guidance (Task Manager, SFC, disk cleanup).",
    },
    # ── Software (3 cases) ───────────────────────────────────────────────
    {
        "id": "TC-011",
        "query": "Excel crashes immediately every time I try to open it.",
        "expected_tool": "lookup_knowledge_base",
        "category": "software",
        "difficulty": "easy",
        "notes": "M365 Quick Repair is in the KB — self-service before ticketing.",
    },
    {
        "id": "TC-012",
        "query": "I need Slack and Zoom installed on my company laptop.",
        "expected_tool": "create_ticket",
        "category": "software",
        "difficulty": "easy",
        "notes": "Software installation always requires a ticket — cannot be self-served.",
    },
    {
        "id": "TC-013",
        "query": "Outlook stopped receiving emails since this morning — is it a server issue?",
        "expected_tool": "check_system_status",
        "category": "software",
        "difficulty": "hard",
        "notes": (
            "User hints at a server issue → check_system_status(email) first. "
            "Agents without CoT may jump to lookup_knowledge_base prematurely."
        ),
    },
    # ── Security (2 cases) ───────────────────────────────────────────────
    {
        "id": "TC-014",
        "query": "I got an email telling me to reset my password via a link that looks fake.",
        "expected_tool": "create_ticket",
        "category": "security",
        "difficulty": "hard",
        "notes": (
            "Phishing → create ticket (critical) then escalate. "
            "First tool is create_ticket. Agents that call lookup_knowledge_base fail."
        ),
    },
    {
        "id": "TC-015",
        "query": "Files on my desktop have been renamed and I can't open any of them.",
        "expected_tool": "create_ticket",
        "category": "security",
        "difficulty": "hard",
        "notes": "Likely ransomware — critical ticket then security escalation required.",
    },
    # ── Access (2 cases) ─────────────────────────────────────────────────
    {
        "id": "TC-016",
        "query": "I need read access to the Legal department's SharePoint library.",
        "expected_tool": "create_ticket",
        "category": "access",
        "difficulty": "easy",
        "notes": "Access provisioning always requires a ticket — no self-service path.",
    },
    {
        "id": "TC-017",
        "query": "My new hire starts Monday and needs an AD account and laptop.",
        "expected_tool": "create_ticket",
        "category": "access",
        "difficulty": "medium",
        "notes": "New-hire onboarding asset provisioning → create_ticket.",
    },
    # ── System Status (2 cases) ───────────────────────────────────────────
    {
        "id": "TC-018",
        "query": "Is SharePoint currently experiencing any outages?",
        "expected_tool": "check_system_status",
        "category": "status",
        "difficulty": "easy",
        "notes": "Direct outage enquiry → check_system_status.",
    },
    {
        "id": "TC-019",
        "query": "Our CRM has been throwing 500 errors for the whole team since 9am.",
        "expected_tool": "check_system_status",
        "category": "status",
        "difficulty": "medium",
        "notes": "Team-wide CRM errors → check_system_status(crm) before ticketing.",
    },
    # ── User Info (1 case) ────────────────────────────────────────────────
    {
        "id": "TC-020",
        "query": "Can you look up the account details for alice.jones@company.com?",
        "expected_tool": "get_user_info",
        "category": "user_info",
        "difficulty": "easy",
        "notes": "Explicit directory lookup → get_user_info.",
    },
]
