"""
knowledge_base.py
=================
Static knowledge store shared by ALL four agents.
Keeps the information layer constant so only the prompting strategy varies.

Each article is a dict with:
    id       – unique slug
    title    – short description
    tags     – list of keyword tags used for matching
    content  – the resolution text returned to the agent / user
"""

KNOWLEDGE_BASE: list[dict] = [
    # ── Password & Authentication ───────────────────────────────────────────
    {
        "id": "KB001",
        "title": "How to reset a forgotten password",
        "tags": ["password", "reset", "forgot", "login", "authentication", "locked out"],
        "content": (
            "To reset your password: (1) Visit the self-service portal at "
            "https://helpdesk.internal/reset. (2) Enter your company email. "
            "(3) Follow the link sent to your registered mobile number. "
            "If MFA is unavailable, contact IT with your employee ID for a manual reset."
        ),
    },
    {
        "id": "KB002",
        "title": "Multi-Factor Authentication (MFA) setup",
        "tags": ["mfa", "two-factor", "2fa", "authenticator", "otp", "authentication"],
        "content": (
            "Install Microsoft Authenticator or Google Authenticator. Open the IT portal, "
            "navigate to Security → MFA Setup, scan the QR code, then enter the 6-digit "
            "code to complete enrollment. Hardware tokens are available upon request."
        ),
    },
    # ── Network & VPN ──────────────────────────────────────────────────────
    {
        "id": "KB003",
        "title": "VPN connection troubleshooting",
        "tags": ["vpn", "network", "remote", "connect", "cisco", "anyconnect", "tunnel"],
        "content": (
            "1. Confirm internet connectivity first. 2. Open Cisco AnyConnect and use "
            "vpn.company.com as the server. 3. Use your AD credentials. "
            "If the handshake fails, flush DNS (`ipconfig /flushdns`) and retry. "
            "Persistent issues: raise a ticket with your IP address and error screenshot."
        ),
    },
    {
        "id": "KB004",
        "title": "Wi-Fi not connecting on corporate network",
        "tags": ["wifi", "wireless", "network", "internet", "ssid", "corporate"],
        "content": (
            "Ensure you are connecting to CORP-SECURE (not CORP-GUEST). "
            "Forget the network and reconnect supplying your AD credentials. "
            "If certificate errors appear, re-enroll the device via https://mdm.internal. "
            "For 802.1X errors open a ticket so network team can check the RADIUS logs."
        ),
    },
    # ── Hardware ───────────────────────────────────────────────────────────
    {
        "id": "KB005",
        "title": "Laptop screen flickering or display issues",
        "tags": ["screen", "display", "flicker", "monitor", "graphics", "laptop", "hdmi"],
        "content": (
            "Update GPU drivers via Device Manager → Display Adapters. "
            "Test with an external monitor to isolate panel vs GPU. "
            "Adjust refresh rate to 60 Hz (Display Settings → Advanced). "
            "If the issue persists after a driver update, log a hardware ticket for "
            "physical inspection."
        ),
    },
    {
        "id": "KB006",
        "title": "Printer not printing / offline status",
        "tags": ["printer", "print", "offline", "stuck", "queue", "spooler"],
        "content": (
            "1. Restart the Print Spooler service (`services.msc`). "
            "2. Clear the print queue (`C:\\Windows\\System32\\spool\\PRINTERS`). "
            "3. Re-add the printer using \\\\ printserver \\ <printer-name>. "
            "4. Ensure the correct driver is installed from https://drivers.internal."
        ),
    },
    {
        "id": "KB007",
        "title": "Computer running slowly or freezing",
        "tags": ["slow", "performance", "freeze", "hang", "cpu", "memory", "ram"],
        "content": (
            "Open Task Manager and check CPU/RAM usage. Common culprits: antivirus scans, "
            "Windows Update, or memory leaks. Run `sfc /scannow` for corruption. "
            "If RAM usage is consistently >90%, request a memory upgrade ticket. "
            "Disk-full machines should have temp files cleared via Storage Sense."
        ),
    },
    # ── Software & Applications ────────────────────────────────────────────
    {
        "id": "KB008",
        "title": "Microsoft 365 / Office apps not opening or crashing",
        "tags": ["office", "microsoft365", "outlook", "word", "excel", "crash", "m365"],
        "content": (
            "Run Quick Repair: Control Panel → Programs → Microsoft 365 → Change → Quick Repair. "
            "If the problem persists, run Online Repair (requires internet). "
            "Clear Office cache: %LocalAppData%\\Microsoft\\Office\\16.0\\. "
            "Sign out and back in to the Office account if licensing errors appear."
        ),
    },
    {
        "id": "KB009",
        "title": "Software installation request",
        "tags": ["install", "software", "application", "license", "request", "deploy"],
        "content": (
            "Submit a software request via the IT portal (https://helpdesk.internal/software). "
            "Include: software name, version, business justification, and manager approval. "
            "Approved requests are deployed within 2 business days via SCCM/Intune. "
            "Emergency installs can be escalated to your IT Business Partner."
        ),
    },
    # ── Email & Collaboration ──────────────────────────────────────────────
    {
        "id": "KB010",
        "title": "Outlook not sending or receiving emails",
        "tags": ["outlook", "email", "send", "receive", "sync", "exchange", "calendar"],
        "content": (
            "Check connectivity indicator in Outlook status bar. "
            "Try Send/Receive All (F9). If the profile is corrupted, run `outlook /resetnavpane`. "
            "For persistent sync issues, recreate the mail profile in Control Panel → Mail. "
            "Exchange server: mail.company.com (auto-discovered via AD)."
        ),
    },
    {
        "id": "KB011",
        "title": "Microsoft Teams audio or video issues",
        "tags": ["teams", "meeting", "audio", "video", "microphone", "camera", "call"],
        "content": (
            "In Teams: Settings → Devices — confirm the correct mic/speaker/camera are selected. "
            "Grant Teams microphone permission in Windows Privacy settings. "
            "Clear Teams cache: %AppData%\\Microsoft\\Teams (close Teams first). "
            "For persistent call-quality issues, run the Teams Network Assessment tool."
        ),
    },
    # ── Security & Access ──────────────────────────────────────────────────
    {
        "id": "KB012",
        "title": "Suspected phishing email or malware",
        "tags": ["phishing", "malware", "virus", "security", "suspicious", "email", "ransomware"],
        "content": (
            "DO NOT click links or open attachments. Forward the email as an attachment to "
            "security@company.com. If you clicked a link, disconnect from the network immediately "
            "and call the Security Hotline: +1-800-SEC-HELP. A P1 ticket will be auto-created."
        ),
    },
    {
        "id": "KB013",
        "title": "Request access to a shared drive or folder",
        "tags": ["access", "permissions", "shared drive", "folder", "sharepoint", "files"],
        "content": (
            "Shared drive access requires manager approval. Submit a request at "
            "https://helpdesk.internal/access with: resource path, access level (Read/Write), "
            "and manager CC. Access is provisioned within 4 business hours once approved."
        ),
    },
    # ── System Status / Outages ────────────────────────────────────────────
    {
        "id": "KB014",
        "title": "Checking current system status and outages",
        "tags": ["outage", "down", "status", "service", "disruption", "incident"],
        "content": (
            "Check the live status page at https://status.internal. Major incidents are also "
            "communicated via email and Teams channel #it-incidents. "
            "If a service is not listed, raise a ticket so monitoring can be verified."
        ),
    },
    # ── Onboarding & Offboarding ───────────────────────────────────────────
    {
        "id": "KB015",
        "title": "New employee IT onboarding checklist",
        "tags": ["onboarding", "new employee", "setup", "new hire", "account", "laptop"],
        "content": (
            "Day 1 checklist: (1) Collect laptop from IT desk with signed acceptance form. "
            "(2) Activate AD account using the welcome email. (3) Enroll MFA. "
            "(4) Install mandatory software via Company Portal. "
            "(5) Attend 30-min IT orientation session (booked automatically by HR)."
        ),
    },
]


def search_knowledge_base(query: str, top_k: int = 3) -> list[dict]:
    """
    Simple keyword-overlap search.
    Returns up to `top_k` articles ranked by how many of the query's words
    appear in the article's tags + title (lowercased).

    In production this would be a vector search; for this benchmark a
    deterministic keyword ranker keeps results reproducible without an
    embeddings API call.
    """
    query_tokens = set(query.lower().split())
    scored = []
    for article in KNOWLEDGE_BASE:
        searchable = " ".join(article["tags"]) + " " + article["title"].lower()
        score = sum(1 for token in query_tokens if token in searchable)
        if score > 0:
            scored.append((score, article))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [article for _, article in scored[:top_k]]
