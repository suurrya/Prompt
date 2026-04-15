"""
ui/config.py
============
All static constants shared across the UI modules.
Nothing here should import from other ui/ modules.
"""

import os

# ── Project root (one level above ui/) ───────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Experiment metadata ───────────────────────────────────────────────────────
# Central config mapping each experiment ID to its display style and description.
EXPERIMENTS = {
    1: {
        "short": "Static Few-Shot",
        "color": "#6366f1",
        "bg": "#eef2ff",
        "border": "#c7d2fe",
        "icon": "🗂️",
        "tag": "STATIC · NO REASONING",
        "pitch": "Uses a fixed list of examples. Fast, but limited to what it's been shown.",
    },
    2: {
        "short": "Static Chain-of-Thought",
        "color": "#0ea5e9",
        "bg": "#e0f2fe",
        "border": "#bae6fd",
        "icon": "🧠",
        "tag": "STATIC · WITH REASONING",
        "pitch": "Fixed examples, but each includes 'Thought' steps to show the AI's logic.",
    },
    3: {
        "short": "Dynamic Few-Shot",
        "color": "#10b981",
        "bg": "#d1fae5",
        "border": "#a7f3d0",
        "icon": "🔍",
        "tag": "DYNAMIC · NO REASONING",
        "pitch": "Picks the most relevant examples for your specific question using TF-IDF.",
    },
    4: {
        "short": "Dynamic Chain-of-Thought",
        "color": "#f59e0b",
        "bg": "#fef3c7",
        "border": "#fde68a",
        "icon": "✨",
        "tag": "DYNAMIC · WITH REASONING",
        "pitch": "The most advanced strategy: Dynamic examples PLUS step-by-step reasoning.",
    },
}

# ── Tool → emoji mapping ──────────────────────────────────────────────────────
TOOL_EMOJI = {
    "lookup_knowledge_base": "📖",
    "create_ticket": "🎫",
    "escalate_ticket": "🚨",
    "reset_password": "🔑",
    "get_user_info": "👤",
    "lookup_user_account": "💳",
    "check_system_status": "📡",
    "schedule_maintenance": "🔧",
    "process_refund": "💰",
    "store_resolved_ticket": "💾",
    "save_ticket_to_long_term_memory": "🗄️",
    "get_user_long_term_memory": "🕐",
    "get_customer_history": "📋",
    "query_asset_database": "🗄️",
}

# ── Conversational messages per tool ─────────────────────────────────────────
# Used by _render_agent_response to show a plain-English summary of what the agent did.
AGENT_MESSAGES = {
    "create_ticket": "I have created a support ticket for this issue. The team will look into it shortly.",
    "escalate_ticket": "I have escalated this issue to the appropriate high-priority queue.",
    "lookup_knowledge_base": "I searched the knowledge base to find the relevant troubleshooting steps.",
    "reset_password": "I have initiated the password reset process for the account.",
    "get_user_info": "I've retrieved the requested user account details from the directory.",
    "lookup_user_account": "I looked up the user's account status in our system.",
    "check_system_status": "I checked our system monitors to see if there is an ongoing outage.",
    "schedule_maintenance": "I have scheduled the requested maintenance/hardware swap.",
    "process_refund": "I have processed the refund request.",
    "store_resolved_ticket": "I've successfully closed and archived this ticket.",
    "save_ticket_to_long_term_memory": "I've saved the resolution details to the user's history.",
    "get_user_long_term_memory": "I retrieved the user's long-term history.",
    "get_customer_history": "I checked the user's previous IT contact history.",
    "query_asset_database": "I've queried the asset inventory database.",
}
