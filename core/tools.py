"""
tools.py
========
Centralized toolkit shared by ALL four helpdesk agents.
Every function is decorated with smolagents @tool so the framework can
auto-generate JSON schemas and pass them to the LLM.

Adding or removing a tool here instantly affects every agent — keeping
the capability surface constant while only the prompting strategy varies.
"""

import uuid # Purposes: To generate unique, non-repeating identifiers for tickets and resets.
import datetime # Purposes: To capture the exact moment an action occurs for audit logs.
import sqlite3 # Purposes: To query the local assets.db inventory database.
import json # Purposes: To serialise SQL result rows into a JSON string for the LLM.
import os # Purposes: To resolve the absolute path to assets.db regardless of the working directory.
# Imports the @tool decorator from the smolagents library.
# This is what turns a regular Python function into a "Tool" that an AI can understand and use.
from smolagents import tool # Purposes: Required to register these functions with the AI framework.

# Absolute path to the SQLite database created by db/generate_db.py.
# __file__ is core/tools.py → parent is core/ → parent is project root → db/assets.db
_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "assets.db")


# Section 2: Ticket Management Tools
# This category handles the creation and escalation of IT support tickets.

@tool
def create_ticket(
    category: str,
    priority: str,
    summary: str,
    user_email: str,
) -> dict:
    """
    Creates a new IT support ticket in the helpdesk system.

    Purpose: Use this tool when a user reports a problem that requires tracking,
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
    # Purposes: Generates a unique Ticket ID using a hex string from uuid4 for tracking.
    ticket_id = f"INC-{uuid.uuid4().hex[:6].upper()}"
    # The final dictionary that represents the ticket data.
    # Purposes: Returns a structured record of the issue to be stored in the helpdesk database.
    return {
        "ticket_id": ticket_id, # The unique ID generated above.
        "category": category, # The classification of the issue (e.g., hardware).
        "priority": priority, # How urgent the issue is.
        "summary": summary, # A brief description for the technician.
        "user_email": user_email, # Who reported the problem.
        "status": "open", # Initial state for all new tickets.
        "created_at": datetime.datetime.utcnow().isoformat() + "Z", # ISO timestamp for auditing.
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

    What it does: 
    1. Records which team the ticket is being sent to.
    2. Stores the reason for the escalation.
    3. Updates the status to 'escalated' and adds a timestamp.
    """
    # Purposes: Returns a confirmation of the escalation, marking the ticket as handled by a specialist.
    return {
        "ticket_id": ticket_id, # Reference to the original ticket.
        "escalated_to": escalate_to, # The specialist team taking over.
        "reason": reason, # Why the level-1 agent couldn't solve it.
        "escalated_at": datetime.datetime.utcnow().isoformat() + "Z", # Timestamp of the hand-off.
        "status": "escalated", # New status to lock the ticket for L1 agents.
    }


# ── Knowledge Base ─────────────────────────────────────────────────────────
# Section 3: Knowledge Base Tools
# These tools allow the AI to search for self-service information before creating a ticket.

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
    # Purposes: In this experiment, we return an empty list to simulate a search with no results,
    # forcing the agent to decide between other tools or creating a ticket.
    return []


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
    # Purposes: Triggers the backend reset workflow and returns the status to the user.
    return {
        "user_email": user_email, # The account being reset.
        "action": "password_reset_initiated", # Status flag for the UI.
        "method": method, # How the reset link is being delivered.
        "temporary_ticket_id": f"PWD-{uuid.uuid4().hex[:6].upper()}", # Audit ID for the reset event.
        "initiated_at": datetime.datetime.utcnow().isoformat() + "Z", # Timestamp for security logs.
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
    # Purposes: Extracts the left side of the email to create a display name.
    local_part = user_email.split("@")[0]
    # Purposes: Returns the user's full profile to the agent for context-building.
    return {
        "user_email": user_email, # Primary key.
        "full_name": local_part.replace(".", " ").title(), # Formatted name (e.g., 'Alice Smith').
        "department": "Engineering", # Useful for routing tickets.
        "manager_email": f"manager@{user_email.split('@')[1]}", # Escalation contact.
        "account_status": "active", # Verifies if the user is even allowed to make requests.
        "assigned_devices": ["LAPTOP-7F3A", "PHONE-9C2B"], # Inventory for hardware checks.
    }


@tool
def lookup_user_account(email: str) -> dict:
    """
    Looks up a user's account subscription tier, billing status, and block flags.

    Use this tool when a user asks about their account standing, subscription
    level, billing issues, or whether their account has been suspended/blocked —
    distinct from directory lookups (use get_user_info for department/manager info).

    Args:
        email: The user's company email address.

    Returns:
        A dict with: email, subscription_tier, billing_status,
        account_blocked (bool), and block_reason (if applicable).
    """
    # Simulated account store — replace with real billing/subscription API call.
    return {
        "email": email,
        "subscription_tier": "enterprise",
        "billing_status": "current",
        "account_blocked": False,
        "block_reason": None,
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
    # Purposes: A lookup table simulating real-time system health checks.
    status_map = {
        "email": "operational",
        "vpn": "operational",
        "sharepoint": "degraded",
        "teams": "operational",
        "internet": "operational",
        "erp": "operational",
        "crm": "outage",
    }
    # Purposes: Fetches the status; defaults to 'operational' if the service is unknown.
    status = status_map.get(service_name.lower(), "operational")
    # Purposes: Prepares the basic response object.
    result = {
        "service_name": service_name,
        "status": status,
        "last_checked": datetime.datetime.utcnow().isoformat() + "Z", # Freshness indicator.
    }
    # Purposes: If a known issue exists, provide the Incident ID and ETA to prevent duplicate tickets.
    if status == "outage":
        result["incident_id"] = f"INC-ACTIVE-{service_name.upper()[:3]}"
        result["eta_minutes"] = 45 # Estimated resolution time.
    elif status == "degraded":
        result["incident_id"] = f"INC-DEGRADED-{service_name.upper()[:3]}"
        result["eta_minutes"] = 20 # Performance recovery time.
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
    # Purposes: Books the appointment in the maintenance database.
    return {
        "maintenance_id": f"MNT-{uuid.uuid4().hex[:6].upper()}", # Unique booking number.
        "asset_id": asset_id, # The hardware being fixed.
        "maintenance_type": maintenance_type, # The scope of work.
        "requested_date": preferred_date, # User's requested slot.
        "scheduled_date": preferred_date,  # Assume slot is available for this simulation.
        "location": "IT Workshop — Floor 2", # Instructions for the user.
        "user_email": user_email, # Contact for logistics.
        "status": "scheduled", # Initial state for the appointment.
    }


# ── Asset Database Query ───────────────────────────────────────────────────

@tool
def query_asset_database(sql: str) -> str:
    """
    Executes a read-only SQL SELECT query against the IT asset inventory database
    and returns the matching rows as a JSON string.

    Use this tool when a user or technician needs structured information about
    hardware assets — for example: finding assets by owner, checking warranty
    expiry dates, listing devices by type or location, or auditing retired equipment.

    DATABASE: assets.db  (tables: departments, employees, laptops, asset_assignments)

    TABLE: laptops  — all company laptops
        asset_id      INTEGER  -- primary key (auto-assigned integer, e.g. 1, 2, 3)
        serial_number TEXT     -- unique serial number (alternate key)
        manufacturer  TEXT     -- e.g. 'Dell', 'Apple', 'HP', 'Lenovo', 'Asus'
        model         TEXT     -- e.g. 'XPS 15 9530', 'MacBook Pro 14'
        assigned_to   INTEGER  -- emp_id FK → employees, or NULL if unassigned
        dept_id       INTEGER  -- FK → departments
        location      TEXT     -- office location string
        status        TEXT     -- 'active' | 'in_repair' | 'retired' | 'available'
        purchase_date TEXT     -- YYYY-MM-DD
        warranty_end  TEXT     -- YYYY-MM-DD

    TABLE: employees
        emp_id        INTEGER  -- primary key
        badge_number  TEXT     -- unique badge number (alternate key)
        email         TEXT     -- unique email address (alternate key)
        full_name     TEXT
        dept_id       INTEGER  -- FK → departments

    TABLE: departments
        dept_id       INTEGER  -- primary key
        dept_code     TEXT     -- unique short code, e.g. 'ENG', 'FIN' (alternate key)
        dept_name     TEXT     -- unique full name (alternate key)
        location      TEXT

    TABLE: asset_assignments  — assignment history (composite primary key)
        asset_id      INTEGER  -- composite PK part 1, FK → laptops
        emp_id        INTEGER  -- composite PK part 2, FK → employees
        assigned_date TEXT     -- composite PK part 3 (YYYY-MM-DD)
        returned_date TEXT     -- NULL if the laptop is still with the employee
        notes         TEXT

    Example queries:
        "SELECT l.asset_id, l.model, e.email FROM laptops l JOIN employees e ON l.assigned_to = e.emp_id"
        "SELECT * FROM laptops WHERE status = 'in_repair'"
        "SELECT l.asset_id, l.model, l.warranty_end FROM laptops l WHERE l.warranty_end < '2026-01-01'"
        "SELECT l.asset_id, l.model, e.email, d.dept_name FROM laptops l JOIN employees e ON l.assigned_to = e.emp_id JOIN departments d ON l.dept_id = d.dept_id WHERE l.status = 'active'"
        "SELECT * FROM asset_assignments WHERE emp_id = 1 ORDER BY assigned_date"

    Args:
        sql: A valid SQL SELECT statement targeting the 'assets' table.
             Only SELECT statements are permitted — INSERT/UPDATE/DELETE will be rejected.

    Returns:
        A JSON string representing a list of row objects (dicts). Each key is a
        column name. Returns an empty list "[]" if no rows match, or an error
        message string if the SQL is invalid.
    """
    # Guard against write operations — this tool is intentionally read-only.
    # Purposes: Prevents the agent from accidentally mutating inventory data.
    if not sql.strip().upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT statements are allowed."})

    # The try...except block is essential: if the agent constructs a malformed
    # SQL query (e.g. wrong column name, syntax error), sqlite3 raises an
    # OperationalError. Without this guard, the exception would crash the agent
    # mid-run. Instead, we catch it and return a descriptive error string so the
    # agent can self-correct and retry with a fixed query.
    try:
        conn = sqlite3.connect(_DB_PATH)
        # row_factory makes each row a dict (column_name → value) rather than
        # a plain tuple — this is what we want to serialise as JSON for the LLM.
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        # Returning a JSON string (not a Python object) is the correct choice
        # here: the LLM receives tool results as text, and structured JSON is
        # far easier for it to parse and reason about than repr(list_of_dicts).
        return json.dumps(rows, indent=2)
    except sqlite3.Error as e:
        # Surface the database error as a readable string so the agent knows
        # exactly what went wrong and can adjust its query on the next step.
        return json.dumps({"error": str(e)})


# ── Convenience export ─────────────────────────────────────────────────────

# Purposes: This export list is what the agents actually "see". 
# By adding a function to this list, we give every agent the ability to use that tool.
ALL_TOOLS = [
    create_ticket, # Capability 1: Formal tracking.
    escalate_ticket, # Capability 2: Priority tiering.
    lookup_knowledge_base, # Capability 3: Self-service.
    reset_password, # Capability 4: Account recovery.
    get_user_info, # Capability 5: Identity context (directory/assets).
    lookup_user_account, # Capability 6: Subscription / billing / block status.
    check_system_status, # Capability 7: Outage detection.
    schedule_maintenance, # Capability 8: Hardware repair.
    query_asset_database, # Capability 9: Structured asset inventory lookup.
]
