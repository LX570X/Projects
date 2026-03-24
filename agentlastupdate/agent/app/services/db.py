"""
Purpose:
- Central SQLite connection/init helpers shared by repositories.

Libraries used:
- sqlite3: local relational database.
- contextlib/pathlib: safe cursor lifecycle and schema path resolution.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path


DB_PATH = Path("metadata.db")
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "models" / "sqlite_schema.sql"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_db_connection() as conn:
        conn.executescript(schema_sql)
        conn.commit()


@contextmanager
def db_cursor():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    finally:
        conn.close()
