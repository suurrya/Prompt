"""
tools.py
========
Centralized toolkit shared by ALL four helpdesk agents.
Every function is decorated with smolagents @tool so the framework can
auto-generate JSON schemas and pass them to the LLM.

Adding or removing a tool here instantly affects every agent — keeping
the capability surface constant while only the prompting strategy varies.
"""

import uuid
import datetime
from smolagents import tool
from knowledge_base import search_knowledge_base


# ── Ticket Management ──────────────────────────────────────────────────────

@tool
def create_ticket(
    category: str,
    priority: str,
    summary: str,
    user_email: str,
) -> dict:
    """
    Creates a new IT support ticket in the helpdesk system.

    Use this tool when a user reports a problem that requires tracking,
    follow-up, or hands-on work by an IT technician — especially hardware
    faults, access provisioning, or issues not resolved by the knowledge base.

    Args:
        category: Issue category. One of: 'hardware', 'software', 'network',
                  'security', 'access', 'email', 'other'.
        priority: Ticket priority. One of: 'low', 'medium', 'high', 'critical'.
        summary: A concise one-sentence description of the problem.
        user_email: The requester's company email address.

    Returns:
        A dict containing the generated ticket_id, category, priority,
        summary, user_email, status ('open'), and created_at timestamp.
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

    Use this tool when:
    - A security incident is confirmed (escalate_to='security-team').
    - The issue is a P1/critical outage (escalate_to='on-call-engineer').
    - The problem requires infrastructure changes (escalate_to='network-team').
    - The issue has breached SLA without resolution.

    Args:
        ticket_id: The ticket ID returned by create_ticket (e.g. 'INC-A1B2C3').
        reason: Short explanation of why escalation is needed.
        escalate_to: Target team. Common values: 'security-team',
                     'network-team', 'on-call-engineer', 'it-management'.

    Returns:
        A dict with ticket_id, escalated_to, reason, and escalated_at timestamp.
    """
    return {
        "ticket_id": ticket_id,
        "escalated_to": escalate_to,
        "reason": reason,
        "escalated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "escalated",
    }


# ── Knowledge Base ─────────────────────────────────────────────────────────

@tool
def lookup_knowledge_base(query: str) -> list:
    """
    Searches the IT knowledge base for self-service articles relevant to the query.

    Use this tool FIRST for common, well-documented issues such as:
    password resets, VPN problems, printer issues, Office/Teams troubleshooting,
    software installation guidance, Wi-Fi connectivity, and general how-to questions.

    This tool is preferred over create_ticket when a step-by-step resolution
    already exists, because it empowers the user to self-heal immediately.

    Args:
        query: A natural-language description of the user's problem.

    Returns:
        A list of matching knowledge-base articles, each containing:
        'id', 'title', 'tags', and 'content' (the resolution steps).
        Returns an empty list if no relevant articles are found.
    """
    return search_knowledge_base(query)


# ── User Account Actions ───────────────────────────────────────────────────

@tool
def reset_password(user_email: str, method: str = "email") -> dict:
    """
    Initiates a password reset for the specified user account.

    Use this tool when a user is locked out, has forgotten their password,
    or explicitly asks for a password reset — and they cannot use the
    self-service portal (e.g., they have no MFA device available).

    Args:
        user_email: The user's company email address.
        method: Delivery method for the reset link. One of:
                'email' (default) – sends link to registered email;
                'sms'             – sends OTP to registered mobile;
                'manual'          – IT admin sets a temp password directly.

    Returns:
        A dict confirming the reset was initiated, with user_email,
        method, a temporary_ticket_id for audit purposes, and timestamp.
    """
    return {
        "user_email": user_email,
        "action": "password_reset_initiated",
        "method": method,
        "temporary_ticket_id": f"PWD-{uuid.uuid4().hex[:6].upper()}",
        "initiated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


@tool
def get_user_info(user_email: str) -> dict:
    """
    Retrieves account and device information for a given user from the directory.

    Use this tool to look up a user's department, manager, assigned assets,
    and account status before creating a ticket or taking an account action.
    This helps pre-populate ticket fields accurately.

    Args:
        user_email: The user's company email address.

    Returns:
        A dict with: user_email, full_name, department, manager_email,
        account_status ('active'/'disabled'/'locked'), and assigned_devices list.
    """
    # Simulated directory response — replace with real LDAP/Graph API call.
    local_part = user_email.split("@")[0]
    return {
        "user_email": user_email,
        "full_name": local_part.replace(".", " ").title(),
        "department": "Engineering",
        "manager_email": f"manager@{user_email.split('@')[1]}",
        "account_status": "active",
        "assigned_devices": ["LAPTOP-7F3A", "PHONE-9C2B"],
    }


# ── Infrastructure & Status ────────────────────────────────────────────────

@tool
def check_system_status(service_name: str) -> dict:
    """
    Checks the current operational status of a named IT service or system.

    Use this tool when a user reports that a specific application or service
    is down or unreachable, before creating a ticket — the outage may already
    be known and under active remediation.

    Args:
        service_name: Name of the service to check. Examples: 'email',
                      'vpn', 'sharepoint', 'teams', 'internet', 'erp', 'crm'.

    Returns:
        A dict with service_name, status ('operational'/'degraded'/'outage'),
        last_checked timestamp, and an optional incident_id if an active
        incident exists.
    """
    # Simulated status map — in production, query a monitoring API.
    status_map = {
        "email": "operational",
        "vpn": "operational",
        "sharepoint": "degraded",
        "teams": "operational",
        "internet": "operational",
        "erp": "operational",
        "crm": "outage",
    }
    status = status_map.get(service_name.lower(), "operational")
    result = {
        "service_name": service_name,
        "status": status,
        "last_checked": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if status == "outage":
        result["incident_id"] = f"INC-ACTIVE-{service_name.upper()[:3]}"
        result["eta_minutes"] = 45
    elif status == "degraded":
        result["incident_id"] = f"INC-DEGRADED-{service_name.upper()[:3]}"
        result["eta_minutes"] = 20
    return result


@tool
def schedule_maintenance(
    asset_id: str,
    maintenance_type: str,
    preferred_date: str,
    user_email: str,
) -> dict:
    """
    Schedules a maintenance appointment for a hardware asset.

    Use this tool when a user needs physical or hands-on work performed on
    their device — such as hardware upgrades, screen replacements, battery
    swaps, or OS reinstalls that cannot be done remotely.

    Args:
        asset_id: Asset tag or device ID (e.g., 'LAPTOP-7F3A').
        maintenance_type: Type of work. Examples: 'screen_replacement',
                          'ram_upgrade', 'os_reinstall', 'battery_swap',
                          'hardware_inspection'.
        preferred_date: Requested date in YYYY-MM-DD format.
        user_email: The user's company email for confirmation notifications.

    Returns:
        A dict with a maintenance_id, asset_id, maintenance_type,
        scheduled_date (may differ from preferred if unavailable),
        location, and user_email.
    """
    return {
        "maintenance_id": f"MNT-{uuid.uuid4().hex[:6].upper()}",
        "asset_id": asset_id,
        "maintenance_type": maintenance_type,
        "requested_date": preferred_date,
        "scheduled_date": preferred_date,  # Assume slot is available.
        "location": "IT Workshop — Floor 2",
        "user_email": user_email,
        "status": "scheduled",
    }


# ── Convenience export ─────────────────────────────────────────────────────

ALL_TOOLS = [
    create_ticket,
    escalate_ticket,
    lookup_knowledge_base,
    reset_password,
    get_user_info,
    check_system_status,
    schedule_maintenance,
]
