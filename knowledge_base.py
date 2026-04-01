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
    # ── Password & Authentication ──────────────────────────────────────────
    {
        "id": "KB001",
        "title": "How to reset a forgotten password",
        "tags": ["password", "reset", "forgot", "login", "authentication", "locked out", "lockout", "ad"],
        "content": (
            "To reset your password: (1) Visit the self-service portal at "
            "https://helpdesk.internal/reset. (2) Enter your company email. "
            "(3) Follow the link sent to your registered mobile number. "
            "If MFA is unavailable, contact IT with your employee ID for a manual reset. "
            "Account lockouts after repeated failed attempts are unlocked automatically "
            "after 30 minutes, or immediately via IT reset."
        ),
    },
    {
        "id": "KB002",
        "title": "Multi-Factor Authentication (MFA) setup and enrollment",
        "tags": ["mfa", "two-factor", "2fa", "authenticator", "otp", "authentication", "setup", "enroll"],
        "content": (
            "Install Microsoft Authenticator or Google Authenticator. Open the IT portal, "
            "navigate to Security → MFA Setup, scan the QR code, then enter the 6-digit "
            "code to complete enrollment. Hardware tokens are available upon request from IT."
        ),
    },
    # ── Network & VPN ──────────────────────────────────────────────────────
    {
        "id": "KB003",
        "title": "VPN connection troubleshooting (Cisco AnyConnect)",
        "tags": ["vpn", "network", "remote", "connect", "cisco", "anyconnect", "tunnel", "disconnect"],
        "content": (
            "1. Confirm internet connectivity first. 2. Open Cisco AnyConnect and use "
            "vpn.company.com as the server. 3. Use your AD credentials. "
            "If the handshake fails, flush DNS (`ipconfig /flushdns`) and retry. "
            "Persistent issues: raise a ticket with your IP address and error screenshot."
        ),
    },
    {
        "id": "KB004",
        "title": "Wi-Fi not connecting or dropping on corporate network",
        "tags": ["wifi", "wireless", "network", "internet", "ssid", "corporate", "dropping", "disconnect"],
        "content": (
            "Ensure you are connecting to CORP-SECURE (not CORP-GUEST). "
            "Forget the network and reconnect supplying your AD credentials. "
            "If certificate errors appear, re-enroll the device via https://mdm.internal. "
            "For 802.1X errors open a ticket so the network team can check the RADIUS logs."
        ),
    },
    # ── Hardware ───────────────────────────────────────────────────────────
    {
        "id": "KB005",
        "title": "Laptop screen flickering or display issues",
        "tags": ["screen", "display", "flicker", "flickering", "monitor", "graphics", "laptop", "hdmi", "refresh"],
        "content": (
            "Update GPU drivers via Device Manager → Display Adapters → Update driver. "
            "Test with an external monitor to isolate panel vs GPU fault. "
            "Adjust refresh rate to 60 Hz (Display Settings → Advanced Display → Refresh Rate). "
            "If the issue persists after a driver update, log a hardware ticket for "
            "physical inspection."
        ),
    },
    {
        "id": "KB006",
        "title": "Printer not printing / offline status",
        "tags": ["printer", "print", "offline", "stuck", "queue", "spooler", "not printing"],
        "content": (
            "1. Restart the Print Spooler service (`services.msc`). "
            "2. Clear the print queue (C:\\Windows\\System32\\spool\\PRINTERS). "
            "3. Re-add the printer using \\\\printserver\\<printer-name>. "
            "4. Ensure the correct driver is installed from https://drivers.internal."
        ),
    },
    {
        "id": "KB007",
        "title": "Computer running slowly or freezing",
        "tags": ["slow", "performance", "freeze", "hang", "cpu", "memory", "ram", "freezing", "unresponsive"],
        "content": (
            "Open Task Manager (Ctrl+Shift+Esc) and check CPU/RAM usage. "
            "Common culprits: antivirus scans, Windows Update, or memory leaks. "
            "Run `sfc /scannow` in an elevated command prompt to check for corruption. "
            "If RAM usage is consistently above 90%, request a memory upgrade ticket. "
            "Disk-full machines: clear temp files via Settings → System → Storage Sense."
        ),
    },
    # ── Software & Applications ────────────────────────────────────────────
    {
        "id": "KB008",
        "title": "Microsoft 365 / Office apps not opening or crashing",
        "tags": ["office", "microsoft365", "m365", "outlook", "word", "excel", "crash", "not opening"],
        "content": (
            "Run Quick Repair: Control Panel → Programs → Microsoft 365 → Change → Quick Repair. "
            "If the problem persists, run Online Repair (requires internet). "
            "Clear Office cache: %LocalAppData%\\Microsoft\\Office\\16.0\\. "
            "Sign out and back into the Office account if licensing errors appear."
        ),
    },
    {
        "id": "KB009",
        "title": "Software installation request process",
        "tags": ["install", "software", "application", "license", "request", "deploy", "sccm", "intune"],
        "content": (
            "Submit a software request via the IT portal (https://helpdesk.internal/software). "
            "Include: software name, version, business justification, and manager approval. "
            "Approved requests are deployed within 2 business days via SCCM/Intune. "
            "Emergency installs can be escalated to your IT Business Partner."
        ),
    },
    {
        "id": "KB009b",
        "title": "Microsoft Word mail merge — step-by-step guide",
        "tags": ["word", "mail merge", "mailmerge", "merge", "microsoft", "office", "how to", "letters"],
        "content": (
            "Mail merge steps in Microsoft Word: (1) Open Word → Mailings tab → Start Mail Merge → Letters. "
            "(2) Click Select Recipients → Use an Existing List and browse to your Excel/CSV data file. "
            "(3) Insert Merge Fields where needed (e.g. <<FirstName>>, <<Address>>). "
            "(4) Preview Results to verify. (5) Finish & Merge → Print Documents or Edit Individual Documents. "
            "For email merge, choose E-mail Messages instead of Letters in step 1."
        ),
    },
    # ── Email & Collaboration ──────────────────────────────────────────────
    {
        "id": "KB010",
        "title": "Outlook not sending or receiving emails",
        "tags": ["outlook", "email", "send", "receive", "sync", "exchange", "calendar", "not receiving"],
        "content": (
            "Check connectivity indicator in Outlook status bar (bottom-right). "
            "Try Send/Receive All (F9). If the profile is corrupted, run `outlook /resetnavpane`. "
            "For persistent sync issues, recreate the mail profile in Control Panel → Mail → Show Profiles. "
            "Exchange server: mail.company.com (auto-discovered via Active Directory)."
        ),
    },
    {
        "id": "KB011",
        "title": "Microsoft Teams audio or video issues",
        "tags": ["teams", "meeting", "audio", "video", "microphone", "camera", "call", "mic", "headset"],
        "content": (
            "In Teams: Settings (…) → Devices — confirm the correct mic/speaker/camera are selected. "
            "Grant Teams microphone permission in Windows Privacy Settings → Microphone. "
            "Clear Teams cache: %AppData%\\Microsoft\\Teams (close Teams first, then delete the folder). "
            "For persistent call-quality issues, run the Teams Network Assessment tool."
        ),
    },
    # ── Security & Access ──────────────────────────────────────────────────
    {
        "id": "KB012",
        "title": "Suspected phishing email — what to do",
        "tags": ["phishing", "suspicious email", "fake link", "credential harvesting", "phishing email"],
        "content": (
            "DO NOT click any links or open attachments in the suspicious email. "
            "Forward the email as an attachment to security@company.com. "
            "If you clicked a link or entered credentials, disconnect from the network IMMEDIATELY "
            "and call the Security Hotline: +1-800-SEC-HELP. A P1 ticket will be auto-created. "
            "IT Security will investigate and notify you of any required password resets."
        ),
    },
    {
        "id": "KB012b",
        "title": "Ransomware and malware incident response",
        "tags": ["ransomware", "malware", "virus", "encrypted", "files renamed", "locked files",
                 "cannot open files", "red screen", "infection"],
        "content": (
            "If you see files renamed (e.g. to .locked, .encrypted) or cannot open files: "
            "(1) Do NOT attempt to restore files yourself. "
            "(2) Disconnect your device from all networks immediately (unplug ethernet, disable Wi-Fi). "
            "(3) Call IT Security Hotline: +1-800-SEC-HELP immediately. "
            "(4) Do NOT pay any ransom. "
            "IT will isolate the device, investigate the infection vector, and restore from backups."
        ),
    },
    {
        "id": "KB013",
        "title": "Request access to a shared drive or SharePoint",
        "tags": ["access", "permissions", "shared drive", "folder", "sharepoint", "files", "read access"],
        "content": (
            "Shared drive / SharePoint access requires manager approval. "
            "Submit a request at https://helpdesk.internal/access with: resource path, "
            "access level (Read/Write/Full), and manager CC. "
            "Access is provisioned within 4 business hours once approved."
        ),
    },
    # ── System Status / Outages ────────────────────────────────────────────
    {
        "id": "KB014",
        "title": "Checking current system status and known outages",
        "tags": ["outage", "down", "status", "service", "disruption", "incident", "not working"],
        "content": (
            "Check the live status page at https://status.internal. "
            "Major incidents are also communicated via email and Teams channel #it-incidents. "
            "If a service is not listed on the status page, raise a ticket so monitoring can be verified."
        ),
    },
    # ── Onboarding & Offboarding ───────────────────────────────────────────
    {
        "id": "KB015",
        "title": "New employee IT onboarding checklist",
        "tags": ["onboarding", "new employee", "setup", "new hire", "account", "laptop", "start"],
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
