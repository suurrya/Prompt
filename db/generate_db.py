"""
generate_db.py
==============
One-time setup script: creates assets.db in the project root and populates
the 'assets' table with sample IT inventory data.

Run once before starting any agent or the benchmark:
    python generate_db.py

Re-running is safe — the table is dropped and recreated each time so the
data stays consistent with what is defined below.
"""

# sqlite3 is part of Python's standard library — no install needed.
# It stores the entire database as a single file (assets.db), making the
# project fully portable and dependency-free.
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "assets.db")


def create_database() -> None:

    # Delete the old DB file if it exists, to start fresh
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # Connect to the database (this will create the file fresh)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Schema — every column the agent can reference in its SQL queries.
    # Keep column names simple and self-explanatory: the agent reads the
    # docstring of query_asset_database to learn these names.
    cursor.execute("""
        CREATE TABLE assets (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id         TEXT NOT NULL,        -- e.g. 'LAPTOP-7F3A'
            asset_type       TEXT NOT NULL,        -- laptop | server | phone | printer | monitor
            manufacturer     TEXT,                 -- Dell, Apple, HP, Lenovo …
            model            TEXT,                 -- e.g. 'XPS 15 9530'
            assigned_to      TEXT,                 -- user email, or NULL if unassigned
            department       TEXT,                 -- Engineering, Finance, HR …
            location         TEXT,                 -- office / data-centre location
            status           TEXT NOT NULL,        -- active | in_repair | retired | available
            purchase_date    TEXT,                 -- YYYY-MM-DD
            warranty_end     TEXT                  -- YYYY-MM-DD  ← key for warranty queries
        )
    """)

    # Sample inventory — enough variety to exercise different query patterns
    sample_assets = [
        # asset_id        type       manufacturer  model               assigned_to                   dept          location            status      purchased     warranty_end
        ("LAPTOP-7F3A",  "laptop",  "Dell",       "XPS 15 9530",      "alice.johnson@company.com",  "Engineering","London HQ",        "active",   "2022-03-15", "2025-03-15"),
        ("LAPTOP-9C2B",  "laptop",  "Apple",      "MacBook Pro 14",   "bob.smith@company.com",      "Design",     "London HQ",        "active",   "2023-06-01", "2026-06-01"),
        ("LAPTOP-3D4E",  "laptop",  "Lenovo",     "ThinkPad X1 Carbon","carol.white@company.com",   "Finance",    "Manchester Office","active",   "2021-11-20", "2024-11-20"),
        ("LAPTOP-5F6G",  "laptop",  "HP",         "EliteBook 840",    "dave.brown@company.com",     "HR",         "Manchester Office","in_repair","2020-08-10", "2023-08-10"),
        ("LAPTOP-1A2B",  "laptop",  "Dell",       "Latitude 5540",    None,                         None,         "IT Stockroom",     "available","2024-01-05", "2027-01-05"),
        ("SERVER-AA01",  "server",  "Dell",       "PowerEdge R750",   None,                         "IT",         "Data Centre Rack 1","active",  "2021-05-12", "2026-05-12"),
        ("SERVER-BB02",  "server",  "HPE",        "ProLiant DL380",   None,                         "IT",         "Data Centre Rack 2","active",  "2020-02-28", "2025-02-28"),
        ("SERVER-CC03",  "server",  "Supermicro", "SYS-620P",         None,                         "IT",         "Data Centre Rack 3","active",  "2019-09-15", "2024-09-15"),
        ("PHONE-9C2B",   "phone",   "Apple",      "iPhone 15 Pro",    "alice.johnson@company.com",  "Engineering","London HQ",        "active",   "2023-10-01", "2024-10-01"),
        ("PHONE-3D4E",   "phone",   "Samsung",    "Galaxy S24",       "eve.davis@company.com",      "Sales",      "London HQ",        "active",   "2024-02-14", "2025-02-14"),
        ("PRINTER-01",   "printer", "HP",         "LaserJet Pro 4001","",                           "Office",     "London HQ Floor 2","active",   "2022-07-22", "2025-07-22"),
        ("MONITOR-01",   "monitor", "Dell",       "UltraSharp 27",    "bob.smith@company.com",      "Design",     "London HQ",        "active",   "2023-03-01", "2026-03-01"),
        ("MONITOR-02",   "monitor", "LG",         "27UK850",          "carol.white@company.com",    "Finance",    "Manchester Office","retired",  "2018-06-15", "2021-06-15"),
        ("LAPTOP-7H8I",  "laptop",  "Dell",       "XPS 13 9340",      "frank.miller@company.com",   "Legal",      "London HQ",        "active",   "2023-08-20", "2026-08-20"),
        ("SERVER-DD04",  "server",  "Dell",       "PowerEdge R650",   None,                         "IT",         "Data Centre Rack 1","active",  "2022-11-30", "2025-11-30"),
    ]

    cursor.executemany("""
        INSERT INTO assets
            (asset_id, asset_type, manufacturer, model, assigned_to, department,
             location, status, purchase_date, warranty_end)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_assets)

    conn.commit()
    conn.close()
    print(f"[OK] Database created at: {DB_PATH}")
    print(f"[OK] {len(sample_assets)} asset records inserted into the 'assets' table.")


if __name__ == "__main__":
    create_database()
    
