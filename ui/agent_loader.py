"""
ui/agent_loader.py
==================
Agent instantiation and database helpers.
  - load_all_experiment_agents()  : imports and creates the 4 ITHelpdeskAgent instances.
  - load_email_asset_options()    : queries assets.db for the user/asset dropdown.
"""

from __future__ import annotations
import importlib
import os
import sqlite3

from ui.config import ROOT

DB_PATH = os.path.join(ROOT, "db", "assets.db")


def load_all_experiment_agents() -> dict[int, object]:
    """
    Dynamically imports and instantiates the 4 experiment agents.
    Called once (lazily) on the first Send, then cached in app.py's _agents global.
    """
    experiment_paths = {
        1: "agents.project_1_few_shot.agents",
        2: "agents.project_2_chain_of_thought.agents",
        3: "agents.project_3_dynamic_few_shot.agents",
        4: "agents.project_4_dynamic_cot.agents",
    }
    agents: dict[int, object] = {}
    for exp_id, path in experiment_paths.items():
        module = importlib.import_module(path)
        agents[exp_id] = module.ITHelpdeskAgent(verbose=False)
    return agents


def load_email_asset_options() -> list[str]:
    """
    Queries assets.db for rows where assigned_to is set.
    Returns a list of strings formatted as 'email  ->  ASSET-ID' for the dropdown.
    Returns an empty list if the database does not exist or a query error occurs.
    """
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT e.email, l.asset_id, l.manufacturer, l.model "
            "FROM laptops l "
            "JOIN employees e ON l.assigned_to = e.emp_id "
            "WHERE l.status = 'active' "
            "ORDER BY e.email, l.asset_id"
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            f"{email}  ->  {asset_id}  ->  {manufacturer} {model}"
            for email, asset_id, manufacturer, model in rows
        ]
    except sqlite3.Error:
        return []
