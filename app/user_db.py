"""
SQLite database for user management.
Separate from MongoDB sensor data.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List
from contextlib import contextmanager

# Database file path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "users.db")


def get_connection():
    """Get SQLite connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Initialize the database with users table"""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                full_name TEXT,
                hashed_password TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                last_login TEXT
            )
        """)
        print("✅ SQLite users database initialized")


def get_user_by_username(username: str) -> Optional[dict]:
    """Get user by username"""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT * FROM users WHERE username = ?", 
            (username,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
    return None


def get_user_by_id(user_id: int) -> Optional[dict]:
    """Get user by ID"""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT * FROM users WHERE id = ?", 
            (user_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
    return None


def create_user(user_data: dict) -> dict:
    """Create a new user"""
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (username, email, full_name, hashed_password, role, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_data["username"],
                user_data.get("email"),
                user_data.get("full_name"),
                user_data["hashed_password"],
                user_data.get("role", "user"),
                1 if user_data.get("is_active", True) else 0,
                datetime.utcnow().isoformat()
            )
        )
        user_data["id"] = cursor.lastrowid
        user_data["created_at"] = datetime.utcnow()
        return user_data


def update_user_last_login(username: str):
    """Update user's last login time"""
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE username = ?",
            (datetime.utcnow().isoformat(), username)
        )


def get_all_users() -> List[dict]:
    """Get all users"""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM users ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]


def delete_user(user_id: int) -> bool:
    """Delete a user by ID"""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return cursor.rowcount > 0


def count_admins() -> int:
    """Count admin users"""
    with get_db() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        return cursor.fetchone()[0]


def user_exists(username: str) -> bool:
    """Check if username exists"""
    return get_user_by_username(username) is not None


# Initialize database on import
init_db()
