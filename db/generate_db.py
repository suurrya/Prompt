"""
generate_db.py
==============
Creates assets.db with a fully normalised, multi-table schema that
demonstrates every relational key type:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  KEY TYPE        │ WHERE USED                                       │
  ├─────────────────────────────────────────────────────────────────────┤
  │  Primary Key     │ dept_id, emp_id, asset_id  (one per table)       │
  │  Identity Key    │ All three PKs — INTEGER AUTOINCREMENT            │
  │  Candidate Key   │ dept_code, dept_name, email, badge_number,       │
  │                  │ serial_number  (unique, not-null → could be PK)  │
  │  Alternate Key   │ Same columns — candidate keys NOT chosen as PK   │
  │  Foreign Key     │ employees.dept_id → departments                  │
  │                  │ laptops.assigned_to → employees                  │
  │                  │ laptops.dept_id → departments                    │
  │                  │ asset_assignments.asset_id → laptops             │
  │                  │ asset_assignments.emp_id → employees             │
  │  Composite Key   │ asset_assignments PK = (asset_id, emp_id,        │
  │                  │                         assigned_date)           │
  │  Super Key       │ Any superset of a candidate key, e.g.            │
  │                  │ {asset_id, serial_number}, {emp_id, email}       │
  └─────────────────────────────────────────────────────────────────────┘

Run once before starting any agent or the benchmark:
    python db/generate_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "assets.db")


def create_database() -> None:
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    # ── Table 1: departments ────────────────────────────────────────────
    # Primary Key  : dept_id  (Identity Key — AUTOINCREMENT)
    # Alternate Keys: dept_code, dept_name  (both Candidate Keys)
    # Super Keys   : {dept_id}, {dept_code}, {dept_name},
    #                {dept_id, dept_code}, {dept_id, dept_name}, …
    cursor.execute("""
        CREATE TABLE departments (
            dept_id   INTEGER PRIMARY KEY AUTOINCREMENT, -- Identity Key + Primary Key
            dept_code TEXT    NOT NULL UNIQUE,           -- Candidate Key → Alternate Key
            dept_name TEXT    NOT NULL UNIQUE,           -- Candidate Key → Alternate Key
            location  TEXT    NOT NULL
        )
    """)

    # ── Table 2: employees ──────────────────────────────────────────────
    # Primary Key  : emp_id  (Identity Key — AUTOINCREMENT)
    # Foreign Key  : dept_id → departments(dept_id)
    # Alternate Keys: email, badge_number  (both Candidate Keys)
    # Super Keys   : {emp_id}, {email}, {badge_number},
    #                {emp_id, email}, {emp_id, badge_number, dept_id}, …
    cursor.execute("""
        CREATE TABLE employees (
            emp_id       INTEGER PRIMARY KEY AUTOINCREMENT, -- Identity Key + Primary Key
            badge_number TEXT    NOT NULL UNIQUE,           -- Candidate Key → Alternate Key
            email        TEXT    NOT NULL UNIQUE,           -- Candidate Key → Alternate Key
            full_name    TEXT    NOT NULL,
            dept_id      INTEGER NOT NULL,                  -- Foreign Key
            FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
        )
    """)

    # ── Table 3: laptops  ───────────────────────────────────────────────
    # Primary Key  : asset_id  (Identity Key — AUTOINCREMENT, plain integer)
    # Foreign Keys : assigned_to → employees(emp_id)
    #                dept_id    → departments(dept_id)
    # Alternate Key: serial_number  (Candidate Key not chosen as PK)
    # Super Keys   : {asset_id}, {serial_number},
    #                {asset_id, serial_number}, {asset_id, model, manufacturer}, …
    cursor.execute("""
        CREATE TABLE laptops (
            asset_id      INTEGER PRIMARY KEY AUTOINCREMENT, -- Identity Key + Primary Key
            serial_number TEXT    NOT NULL UNIQUE,           -- Candidate Key → Alternate Key
            manufacturer  TEXT    NOT NULL,
            model         TEXT    NOT NULL,
            assigned_to   INTEGER,                           -- Foreign Key (nullable = unassigned)
            dept_id       INTEGER,                           -- Foreign Key
            location      TEXT,
            status        TEXT    NOT NULL DEFAULT 'available',
            purchase_date TEXT    NOT NULL,
            warranty_end  TEXT    NOT NULL,
            FOREIGN KEY (assigned_to) REFERENCES employees(emp_id),
            FOREIGN KEY (dept_id)    REFERENCES departments(dept_id)
        )
    """)

    # ── Table 4: asset_assignments  ─────────────────────────────────────
    # Composite Key: (asset_id, emp_id, assigned_date) — PRIMARY KEY
    # Foreign Keys : asset_id → laptops(asset_id)
    #                emp_id   → employees(emp_id)
    # Super Keys   : {asset_id, emp_id, assigned_date},
    #                {asset_id, emp_id, assigned_date, returned_date}, …
    cursor.execute("""
        CREATE TABLE asset_assignments (
            asset_id      INTEGER NOT NULL,  -- Composite Key part 1 + Foreign Key
            emp_id        INTEGER NOT NULL,  -- Composite Key part 2 + Foreign Key
            assigned_date TEXT    NOT NULL,  -- Composite Key part 3
            returned_date TEXT,              -- NULL means currently in use
            notes         TEXT,
            PRIMARY KEY (asset_id, emp_id, assigned_date),   -- Composite Key
            FOREIGN KEY (asset_id) REFERENCES laptops(asset_id),
            FOREIGN KEY (emp_id)   REFERENCES employees(emp_id)
        )
    """)

    # ── Seed: departments ───────────────────────────────────────────────
    departments = [
        # dept_code  dept_name                 location
        ("ENG",      "Engineering",            "London HQ"),
        ("DES",      "Design",                 "London HQ"),
        ("FIN",      "Finance",                "Manchester Office"),
        ("HR",       "Human Resources",        "Manchester Office"),
        ("SAL",      "Sales",                  "London HQ"),
        ("LEG",      "Legal",                  "London HQ"),
        ("MKT",      "Marketing",              "London HQ"),
        ("IT",       "Information Technology", "London HQ"),
        ("OPS",      "Operations",             "Birmingham Office"),
        ("PRO",      "Procurement",            "Birmingham Office"),
    ]
    cursor.executemany(
        "INSERT INTO departments (dept_code, dept_name, location) VALUES (?,?,?)",
        departments,
    )

    # ── Seed: employees ─────────────────────────────────────────────────
    # dept_id values match insertion order above (1=ENG … 10=PRO)
    employees = [
        # badge_number   email                              full_name              dept_id
        ("EMP-001", "alice.johnson@company.com",    "Alice Johnson",       1),  # Engineering
        ("EMP-002", "bob.smith@company.com",        "Bob Smith",           2),  # Design
        ("EMP-003", "carol.white@company.com",      "Carol White",         3),  # Finance
        ("EMP-004", "dave.brown@company.com",       "Dave Brown",          4),  # HR
        ("EMP-005", "eve.davis@company.com",        "Eve Davis",           5),  # Sales
        ("EMP-006", "frank.miller@company.com",     "Frank Miller",        6),  # Legal
        ("EMP-007", "grace.lee@company.com",        "Grace Lee",           7),  # Marketing
        ("EMP-008", "henry.wilson@company.com",     "Henry Wilson",        8),  # IT
        ("EMP-009", "isla.taylor@company.com",      "Isla Taylor",         9),  # Operations
        ("EMP-010", "jack.anderson@company.com",    "Jack Anderson",       10), # Procurement
        ("EMP-011", "karen.thomas@company.com",     "Karen Thomas",        1),  # Engineering
        ("EMP-012", "liam.jackson@company.com",     "Liam Jackson",        2),  # Design
        ("EMP-013", "mia.harris@company.com",       "Mia Harris",          3),  # Finance
        ("EMP-014", "noah.martin@company.com",      "Noah Martin",         4),  # HR
        ("EMP-015", "olivia.garcia@company.com",    "Olivia Garcia",       5),  # Sales
        ("EMP-016", "peter.martinez@company.com",   "Peter Martinez",      6),  # Legal
        ("EMP-017", "quinn.robinson@company.com",   "Quinn Robinson",      7),  # Marketing
        ("EMP-018", "rachel.clark@company.com",     "Rachel Clark",        8),  # IT
        ("EMP-019", "sam.rodriguez@company.com",    "Sam Rodriguez",       9),  # Operations
        ("EMP-020", "tina.lewis@company.com",       "Tina Lewis",          10), # Procurement
        ("EMP-021", "uma.walker@company.com",       "Uma Walker",          1),  # Engineering
        ("EMP-022", "victor.hall@company.com",      "Victor Hall",         2),  # Design
        ("EMP-023", "wendy.allen@company.com",      "Wendy Allen",         3),  # Finance
        ("EMP-024", "xander.young@company.com",     "Xander Young",        5),  # Sales
        ("EMP-025", "yara.hernandez@company.com",   "Yara Hernandez",      8),  # IT
    ]
    cursor.executemany(
        "INSERT INTO employees (badge_number, email, full_name, dept_id) VALUES (?,?,?,?)",
        employees,
    )

    # ── Seed: laptops ───────────────────────────────────────────────────
    # asset_id auto-assigned (1, 2, 3 …)
    # assigned_to uses emp_id (1–25); NULL = unassigned
    laptops = [
        # serial_number    manufacturer  model                  assigned_to  dept_id  location               status      purchase_date  warranty_end
        ("SN-DELL-001",   "Dell",       "XPS 15 9530",         1,  1,  "London HQ",         "active",    "2022-03-15", "2025-03-15"),
        ("SN-APPL-002",   "Apple",      "MacBook Pro 14",      2,  2,  "London HQ",         "active",    "2023-06-01", "2026-06-01"),
        ("SN-LENO-003",   "Lenovo",     "ThinkPad X1 Carbon",  3,  3,  "Manchester Office", "active",    "2021-11-20", "2024-11-20"),
        ("SN-HP-004",     "HP",         "EliteBook 840 G10",   4,  4,  "Manchester Office", "in_repair", "2020-08-10", "2023-08-10"),
        ("SN-DELL-005",   "Dell",       "Latitude 5540",       5,  5,  "London HQ",         "active",    "2024-01-05", "2027-01-05"),
        ("SN-DELL-006",   "Dell",       "XPS 13 9340",         6,  6,  "London HQ",         "active",    "2023-08-20", "2026-08-20"),
        ("SN-APPL-007",   "Apple",      "MacBook Pro 16",      7,  7,  "London HQ",         "active",    "2023-11-01", "2026-11-01"),
        ("SN-LENO-008",   "Lenovo",     "ThinkPad T14",        8,  8,  "London HQ",         "active",    "2022-07-14", "2025-07-14"),
        ("SN-HP-009",     "HP",         "EliteBook 1040 G10",  9,  9,  "Birmingham Office", "active",    "2023-04-22", "2026-04-22"),
        ("SN-DELL-010",   "Dell",       "Precision 5570",      10, 10, "Birmingham Office", "active",    "2022-12-10", "2025-12-10"),
        ("SN-DELL-011",   "Dell",       "Latitude 7440",       11, 1,  "London HQ",         "active",    "2024-02-28", "2027-02-28"),
        ("SN-APPL-012",   "Apple",      "MacBook Air M2",      12, 2,  "London HQ",         "active",    "2023-09-15", "2026-09-15"),
        ("SN-LENO-013",   "Lenovo",     "ThinkPad L14",        13, 3,  "Manchester Office", "active",    "2022-05-30", "2025-05-30"),
        ("SN-HP-014",     "HP",         "ProBook 450 G10",     14, 4,  "Manchester Office", "active",    "2023-01-17", "2026-01-17"),
        ("SN-DELL-015",   "Dell",       "XPS 15 9530",         15, 5,  "London HQ",         "active",    "2024-03-10", "2027-03-10"),
        ("SN-ASUS-016",   "Asus",       "ZenBook Pro 14",      16, 6,  "London HQ",         "active",    "2023-07-04", "2026-07-04"),
        ("SN-DELL-017",   "Dell",       "Latitude 5540",       17, 7,  "London HQ",         "active",    "2023-10-20", "2026-10-20"),
        ("SN-APPL-018",   "Apple",      "MacBook Pro 14",      18, 8,  "London HQ",         "active",    "2024-01-30", "2027-01-30"),
        ("SN-HP-019",     "HP",         "EliteBook 840 G10",   19, 9,  "Birmingham Office", "active",    "2022-09-09", "2025-09-09"),
        ("SN-LENO-020",   "Lenovo",     "ThinkPad X1 Carbon",  20, 10, "Birmingham Office", "active",    "2023-03-25", "2026-03-25"),
        ("SN-DELL-021",   "Dell",       "XPS 13 9340",         21, 1,  "London HQ",         "active",    "2024-04-15", "2027-04-15"),
        ("SN-APPL-022",   "Apple",      "MacBook Air M2",      22, 2,  "London HQ",         "active",    "2023-12-01", "2026-12-01"),
        ("SN-LENO-023",   "Lenovo",     "ThinkPad T14",        23, 3,  "Manchester Office", "active",    "2022-08-18", "2025-08-18"),
        ("SN-DELL-024",   "Dell",       "Precision 5570",      24, 5,  "London HQ",         "active",    "2023-06-11", "2026-06-11"),
        ("SN-ASUS-025",   "Asus",       "ExpertBook B9",       25, 8,  "London HQ",         "active",    "2024-05-05", "2027-05-05"),
        # Unassigned / pool laptops
        ("SN-DELL-026",   "Dell",       "Latitude 5540",       None, 8, "IT Stockroom",     "available", "2024-06-01", "2027-06-01"),
        ("SN-HP-027",     "HP",         "ProBook 450 G10",     None, 8, "IT Stockroom",     "available", "2024-06-01", "2027-06-01"),
        ("SN-LENO-028",   "Lenovo",     "ThinkPad L14",        None, 8, "IT Stockroom",     "available", "2023-11-15", "2026-11-15"),
        ("SN-APPL-029",   "Apple",      "MacBook Air M2",      None, 8, "IT Stockroom",     "available", "2024-07-10", "2027-07-10"),
        # In-repair laptops
        ("SN-DELL-030",   "Dell",       "XPS 15 9530",         None, 8, "IT Workshop",      "in_repair", "2021-04-20", "2024-04-20"),
        ("SN-HP-031",     "HP",         "EliteBook 840 G10",   None, 8, "IT Workshop",      "in_repair", "2020-11-30", "2023-11-30"),
        ("SN-LENO-032",   "Lenovo",     "ThinkPad X1 Carbon",  None, 8, "IT Workshop",      "in_repair", "2022-02-14", "2025-02-14"),
        # Retired laptops
        ("SN-DELL-033",   "Dell",       "Latitude 5400",       None, 8, "IT Stockroom",     "retired",   "2018-01-10", "2021-01-10"),
        ("SN-HP-034",     "HP",         "EliteBook 820 G4",    None, 8, "IT Stockroom",     "retired",   "2017-06-15", "2020-06-15"),
        ("SN-LENO-035",   "Lenovo",     "ThinkPad T470",       None, 8, "IT Stockroom",     "retired",   "2017-09-22", "2020-09-22"),
        ("SN-DELL-036",   "Dell",       "Inspiron 15 3000",    None, 8, "IT Stockroom",     "retired",   "2016-03-18", "2019-03-18"),
        ("SN-APPL-037",   "Apple",      "MacBook Pro 13 2019", None, 8, "IT Stockroom",     "retired",   "2019-05-01", "2022-05-01"),
        ("SN-ASUS-038",   "Asus",       "VivoBook 15",         None, 8, "IT Stockroom",     "retired",   "2018-08-30", "2021-08-30"),
        # Extras (active, assigned to first few employees for more history)
        ("SN-DELL-039",   "Dell",       "XPS 15 9560",         1,  1,  "London HQ",         "active",    "2024-08-01", "2027-08-01"),
        ("SN-APPL-040",   "Apple",      "MacBook Pro 16",      3,  3,  "Manchester Office", "active",    "2024-09-01", "2027-09-01"),
    ]
    cursor.executemany("""
        INSERT INTO laptops
            (serial_number, manufacturer, model, assigned_to, dept_id,
             location, status, purchase_date, warranty_end)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, laptops)

    # ── Seed: asset_assignments (Composite Key) ──────────────────────────
    # Records PAST assignments — returned_date filled means the laptop was given back.
    # (asset_id, emp_id, assigned_date) = Composite Primary Key
    assignments = [
        # asset_id  emp_id  assigned_date  returned_date   notes
        (1,  1,  "2022-03-15", None,         "Initial issue"),
        (2,  2,  "2023-06-01", None,         "Initial issue"),
        (3,  3,  "2021-11-20", None,         "Initial issue"),
        (4,  4,  "2020-08-10", "2024-01-15", "Returned for battery replacement"),
        (5,  5,  "2024-01-05", None,         "Initial issue"),
        (6,  6,  "2023-08-20", None,         "Initial issue"),
        (7,  7,  "2023-11-01", None,         "Initial issue"),
        (8,  8,  "2022-07-14", None,         "Initial issue"),
        (9,  9,  "2023-04-22", None,         "Initial issue"),
        (10, 10, "2022-12-10", None,         "Initial issue"),
        # Some laptops were previously assigned before current owner
        (1,  21, "2021-01-10", "2022-03-14", "Previous owner — transferred to Alice"),
        (3,  13, "2020-05-01", "2021-11-19", "Previous owner — transferred to Carol"),
        (30, 1,  "2023-06-01", "2024-01-01", "Old laptop before repair"),
        (31, 4,  "2019-03-01", "2020-11-29", "Old laptop before repair"),
    ]
    cursor.executemany("""
        INSERT INTO asset_assignments
            (asset_id, emp_id, assigned_date, returned_date, notes)
        VALUES (?,?,?,?,?)
    """, assignments)

    conn.commit()
    conn.close()

    print(f"[OK] Database created at: {DB_PATH}")
    print(f"[OK] {len(departments)} departments")
    print(f"[OK] {len(employees)} employees")
    print(f"[OK] {len(laptops)} laptops")
    print(f"[OK] {len(assignments)} assignment history records")


if __name__ == "__main__":
    create_database()
