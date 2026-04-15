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
            asset_id TEXT PRIMARY KEY,                 -- 1. Primary Key (Natural Key)
            serial_number TEXT NOT NULL UNIQUE,        -- 2. Alternate Key
            asset_type TEXT NOT NULL,
            manufacturer TEXT,
            model TEXT,
            assigned_to_id INTEGER,                    -- 3. Foreign Key
            location_id INTEGER,                       -- 4. Foreign Key
            status TEXT,
            purchase_date TEXT,
            warranty_end TEXT,
            
            FOREIGN KEY (assigned_to_id) REFERENCES employees(employee_id), -- Link to employees
            FOREIGN KEY (location_id) REFERENCES locations(location_id)     -- Link to locations
        )
    """)

    # cursor.execute("""
        
                   
    #     CREATE TABLE locations (
                   
    #         location_id INTEGER PRIMARY KEY AUTOINCREMENT, -- 1. Identity Key & Primary Key
    #         location_name TEXT NOT NULL UNIQUE,            -- 2. Alternate Key
    #         city TEXT NOT NULL
    #     );

    #     CREATE TABLE employees (
    #                 employee_id INTEGER PRIMARY KEY AUTOINCREMENT, -- 1. Identity Key & Primary Key
    #                 full_name TEXT NOT NULL,
    #                 email TEXT NOT NULL UNIQUE,                    -- 2. Alternate Key
    #                 department TEXT NOT NULL
    #     );     
    # """)

    # cursor.execute("""
                
    #         """)

    # Sample inventory — enough variety to exercise different query patterns
    sample_assets = [
        # asset_id        type       manufacturer  model               assigned_to                   dept          location            status      purchased     warranty_end
        ("LAPTOP-7F3A",  "laptop",  "Dell",       "XPS 15 9530",      "alice.johnson@company.com",  "Engineering","London HQ",        "active",   "2022-03-15", "2025-03-15"),
        ("LAPTOP-9C2B",  "laptop",  "Apple",      "MacBook Pro 14",   "bob.smith@company.com",      "Design",     "London HQ",        "active",   "2023-06-01", "2026-06-01"),
        ("LAPTOP-3D4E",  "laptop",  "Lenovo",     "ThinkPad X1 Carbon","carol.white@company.com",   "Finance",    "Manchester Office","active",   "2021-11-20", "2024-11-20"),
        ("LAPTOP-5F6G",  "laptop",  "HP",         "EliteBook 840",    "dave.brown@company.com",     "HR",         "Manchester Office","in_repair","2020-08-10", "2023-08-10"),
        ("LAPTOP-1A2B",  "laptop",  "Dell",       "Latitude 5540",    "John.doe@company.com",       "HR",         "IT Stockroom",     "available","2024-01-05", "2027-01-05"),
        ("PHONE-9C2B",   "phone",   "Apple",      "iPhone 15 Pro",    "alice.johnson@company.com",  "Engineering","London HQ",        "active",   "2023-10-01", "2024-10-01"),
        ("PHONE-3D4E",   "phone",   "Samsung",    "Galaxy S24",       "eve.davis@company.com",      "Sales",      "London HQ",        "active",   "2024-02-14", "2025-02-14"),
        ("LAPTOP-7H8I",  "laptop",  "Dell",       "XPS 13 9340",      "frank.miller@company.com",   "Legal",      "London HQ",        "active",   "2023-08-20", "2026-08-20"),
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
    
