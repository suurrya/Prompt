"""
tools.py
========
Centralized toolkit shared by ALL four helpdesk agents.
Every function is decorated with @tool so smolagents can auto-generate
JSON schemas and pass them to the LLM.

Tools are grouped by domain:
  ── Ticket Management          create_ticket, escalate_ticket
  ── Knowledge Base             lookup_knowledge_base
  ── Password / Auth            reset_password
  ── User & Account             get_user_info, lookup_user_account
  ── Infrastructure & Status    check_system_status, schedule_maintenance
  ── Billing & Reservations     process_refund
  ── Memory / History           store_resolved_ticket,
                                save_ticket_to_long_term_memory,
                                get_user_long_term_memory,
                                get_customer_history

All external-DB calls from functions.py have been replaced with
deterministic simulated responses so the benchmark requires no database
or vector store — keeping results reproducible and portable.
"""

import json
import re
import uuid
import datetime
from smolagents import tool
from knowledge_base import search_knowledge_base as _kb_search

# ── In-process long-term memory store (replaces Chroma) ───────────────────────
_LONG_TERM_MEMORY: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# TICKET MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

@tool
def create_ticket(category: str, priority: str, summary: str, user_email: str) -> dict:
    """
    Creates a new IT support ticket in the helpdesk system.

    Use when a problem requires tracking, follow-up, or hands-on work by an
    IT technician — especially hardware faults, access provisioning, software
    installs, or security incidents the knowledge base cannot resolve.

    Args:
        category: Issue category — one of: 'hardware', 'software', 'network',
                  'security', 'access', 'email', 'billing', 'other'.
        priority: Ticket priority — one of: 'low', 'medium', 'high', 'critical'.
        summary: A concise one-sentence description of the problem.
        user_email: The requester's company email address.

    Returns:
        Dict with ticket_id, category, priority, summary, user_email,
        status ('open'), and created_at timestamp.
    """
    ticket_id = f"INC-{uuid.uuid4().hex[:6].upper()}"
    return {
        "ticket_id": ticket_id,
        "category": category,
        "priority": priority,
        "summary": summary,
        "user_email": user_email,
        "status": "open",
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


@tool
def escalate_ticket(ticket_id: str, reason: str, escalate_to: str) -> dict:
    """
    Escalates an existing ticket to a higher support tier or specialist team.

    Use when:
    - A security incident is confirmed (escalate_to='security-team').
    - The issue is a P1/critical outage (escalate_to='on-call-engineer').
    - Infrastructure changes are required (escalate_to='network-team').
    - SLA has been breached without resolution.

    Args:
        ticket_id: The ticket ID returned by create_ticket (e.g. 'INC-A1B2C3').
        reason: Short explanation of why escalation is needed.
        escalate_to: Target team — e.g. 'security-team', 'network-team',
                     'on-call-engineer', 'it-management'.

    Returns:
        Dict with ticket_id, escalated_to, reason, escalated_at, and status.
    """
    return {
        "ticket_id": ticket_id,
        "escalated_to": escalate_to,
        "reason": reason,
        "escalated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "escalated",
    }


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════

@tool
def lookup_knowledge_base(query: str) -> list:
    """
    Searches the IT knowledge base for self-service resolution articles.

    Use this tool FIRST for common, well-documented issues such as:
    password resets, VPN problems, printer issues, Office/Teams troubleshooting,
    software installation guidance, Wi-Fi connectivity, and general how-to
    questions. Prefer this over create_ticket when a KB article can empower
    the user to self-resolve immediately.

    Args:
        query: Natural-language description of the user's problem.

    Returns:
        List of matching articles, each with 'id', 'title', 'tags', and
        'content' (the resolution steps). Empty list if no match found.
    """
    return _kb_search(query)


# ══════════════════════════════════════════════════════════════════════════════
# PASSWORD / AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════

@tool
def reset_password(user_email: str, method: str = "email") -> dict:
    """
    Initiates a password reset for the specified user account.

    Use when a user is locked out, has forgotten their password, or explicitly
    requests a reset AND cannot use the self-service portal (e.g. no MFA device).

    Args:
        user_email: The user's company email address.
        method: Delivery method — 'email' (default), 'sms', or 'manual'
                (IT admin sets a temporary password directly).

    Returns:
        Dict confirming the reset was initiated, with user_email, method,
        a temporary_ticket_id for audit, and initiated_at timestamp.
    """
    return {
        "user_email": user_email,
        "action": "password_reset_initiated",
        "method": method,
        "temporary_ticket_id": f"PWD-{uuid.uuid4().hex[:6].upper()}",
        "initiated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


# ══════════════════════════════════════════════════════════════════════════════
# USER & ACCOUNT
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_user_info(user_email: str) -> dict:
    """
    Retrieves account and device information for a given user from the directory.

    Use to look up a user's department, manager, assigned assets, and account
    status before creating a ticket or taking an account action.

    Args:
        user_email: The user's company email address.

    Returns:
        Dict with user_email, full_name, department, manager_email,
        account_status, and assigned_devices list.
    """
    local_part = user_email.split("@")[0]
    return {
        "user_email": user_email,
        "full_name": local_part.replace(".", " ").title(),
        "department": "Engineering",
        "manager_email": f"manager@{user_email.split('@')[1]}",
        "account_status": "active",
        "assigned_devices": ["LAPTOP-7F3A", "PHONE-9C2B"],
    }


@tool
def lookup_user_account(email: str) -> str:
    """
    Looks up a user's account status, block status, and subscription tier by email.

    Use instead of get_user_info when you need subscription or billing context
    (e.g. to process a refund or verify entitlements). Validates email format
    before querying.

    Args:
        email: The user's email address.

    Returns:
        JSON string with user_id, name, is_blocked, subscription_tier, and
        subscription_status. Returns an error JSON if the email is invalid
        or the user is not found.
    """
    if not re.match(r"^[\w\.\-]+@[\w\.\-]+\.\w+$", email):
        return json.dumps({"status": "error", "message": "Invalid email format."})
    local_part = email.split("@")[0]
    return json.dumps({
        "user_id": f"USR-{abs(hash(email)) % 9000 + 1000}",
        "name": local_part.replace(".", " ").title(),
        "is_blocked": False,
        "subscription_tier": "premium",
        "subscription_status": "active",
    })


# ══════════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE & STATUS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def check_system_status(service_name: str) -> dict:
    """
    Checks the current operational status of a named IT service or system.

    Use BEFORE creating a ticket when a user reports a service is down or
    unreachable — the outage may already be known and under remediation.

    Args:
        service_name: Service to check — e.g. 'email', 'vpn', 'sharepoint',
                      'teams', 'internet', 'erp', 'crm'.

    Returns:
        Dict with service_name, status ('operational'/'degraded'/'outage'),
        last_checked timestamp, and optional incident_id + eta_minutes.
    """
    status_map = {
        "email": "operational", "vpn": "operational",
        "sharepoint": "degraded", "teams": "operational",
        "internet": "operational", "erp": "operational", "crm": "outage",
    }
    status = status_map.get(service_name.lower(), "operational")
    result: dict = {
        "service_name": service_name,
        "status": status,
        "last_checked": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if status == "outage":
        result.update({"incident_id": f"INC-ACTIVE-{service_name.upper()[:3]}", "eta_minutes": 45})
    elif status == "degraded":
        result.update({"incident_id": f"INC-DEGRADED-{service_name.upper()[:3]}", "eta_minutes": 20})
    return result


@tool
def schedule_maintenance(asset_id: str, maintenance_type: str, preferred_date: str, user_email: str) -> dict:
    """
    Schedules a maintenance appointment for a hardware asset.

    Use when a user needs physical or hands-on work on their device —
    screen replacements, RAM/battery upgrades, OS reinstalls, or hardware
    inspections that cannot be done remotely.

    Args:
        asset_id: Asset tag or device ID (e.g. 'LAPTOP-7F3A').
        maintenance_type: Type of work — 'screen_replacement', 'ram_upgrade',
                          'os_reinstall', 'battery_swap', 'hardware_inspection'.
        preferred_date: Requested date in YYYY-MM-DD format.
        user_email: User's email for confirmation notifications.

    Returns:
        Dict with maintenance_id, asset_id, maintenance_type, scheduled_date,
        location, user_email, and status.
    """
    return {
        "maintenance_id": f"MNT-{uuid.uuid4().hex[:6].upper()}",
        "asset_id": asset_id,
        "maintenance_type": maintenance_type,
        "requested_date": preferred_date,
        "scheduled_date": preferred_date,
        "location": "IT Workshop — Floor 2",
        "user_email": user_email,
        "status": "scheduled",
    }


# ══════════════════════════════════════════════════════════════════════════════
# BILLING & RESERVATIONS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def process_refund(reservation_id: str) -> str:
    """
    Processes a refund for a specific reservation or billing transaction.

    Use when a user requests a refund for a charge, reservation, or service
    that was incorrectly billed or cancelled. Validate the reservation ID
    before calling.

    Args:
        reservation_id: Unique reservation or transaction ID (e.g. 'RES-00123').

    Returns:
        JSON string with status ('success'/'error'), a confirmation message,
        and the refund_amount if successful.
    """
    if not reservation_id or not reservation_id.strip():
        return json.dumps({"status": "error", "message": "Reservation ID is required."})
    return json.dumps({
        "status": "success",
        "message": f"Refund initiated for reservation {reservation_id}.",
        "reservation_id": reservation_id,
        "refund_amount": 15.00,
        "processed_at": datetime.datetime.utcnow().isoformat() + "Z",
    })


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY / HISTORY
# ══════════════════════════════════════════════════════════════════════════════

@tool
def store_resolved_ticket(user_id: str, summary: str) -> str:
    """
    Saves a brief summary of a resolved issue to the user's long-term history.

    Call ONLY when a ticket is fully resolved or closed — not while the issue
    is still open. Records are retrievable via get_user_long_term_memory.

    Args:
        user_id: The user's ID or email used as the history key.
        summary: One-sentence summary of the issue and how it was resolved.

    Returns:
        Confirmation string indicating success.
    """
    _LONG_TERM_MEMORY.setdefault(user_id, []).append(summary)
    return f"Resolved-ticket summary saved for user '{user_id}'."


@tool
def save_ticket_to_long_term_memory(user_id: str, summary: str, resolution: str) -> str:
    """
    Archives the full outcome of a resolved ticket — both the issue summary
    and the resolution steps — to the user's long-term history.

    Use for detailed post-resolution archiving so future agents can retrieve
    context-rich history for repeat callers.

    Args:
        user_id: The user's ID or email used as the history key.
        summary: Description of the original issue.
        resolution: Steps taken or solution applied to resolve the issue.

    Returns:
        Confirmation string indicating success.
    """
    record = f"Issue: {summary} | Resolution: {resolution}"
    _LONG_TERM_MEMORY.setdefault(user_id, []).append(record)
    return f"Ticket outcome archived in long-term memory for user '{user_id}'."


@tool
def get_user_long_term_memory(user_id: str) -> str:
    """
    Retrieves the full long-term history of past issues and resolutions for
    a specific user, enabling context-aware support for returning callers.

    Use at the START of a conversation with a returning user to understand
    their history before diagnosing the current issue.

    Args:
        user_id: The user's ID or email — must match what was used when
                 records were stored.

    Returns:
        Numbered history entries separated by newlines, or a no-history notice.
    """
    records = _LONG_TERM_MEMORY.get(user_id, [])
    if not records:
        return f"No prior history found for user '{user_id}'."
    return "\n".join(f"[{i+1}] {r}" for i, r in enumerate(records))


@tool
def get_customer_history(user_id: str) -> str:
    """
    Returns a brief summary of issue categories a user has previously
    contacted IT about — useful for spotting repeat problems or patterns.

    Use for a quick context check before triaging a new request, particularly
    for frequent callers.

    Args:
        user_id: The user's ID or email.

    Returns:
        Plain-English summary of past issue categories, or a no-history notice.
    """
    records = _LONG_TERM_MEMORY.get(user_id, [])
    if not records:
        return f"No prior contact history on record for user '{user_id}'."
    return f"User '{user_id}' previously contacted IT about: {'; '.join(records[:3])}"


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE EXPORT — imported by all four agents
# ══════════════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    create_ticket,
    escalate_ticket,
    lookup_knowledge_base,
    reset_password,
    get_user_info,
    lookup_user_account,
    check_system_status,
    schedule_maintenance,
    process_refund,
    store_resolved_ticket,
    save_ticket_to_long_term_memory,
    get_user_long_term_memory,
    get_customer_history,
]
