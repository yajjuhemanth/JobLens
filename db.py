"""
TrueNotice — SQLite persistence layer with User & Data Isolation
================================================================
Uses Python's stdlib ``sqlite3``. Stores user accounts (hashed with Werkzeug),
per-user applicant profiles, and isolated job notifications so TrueNotice provides
complete multi-user isolation: private dashboards, private watchlists, private
eligibility caching, and grounded Q&A.

Every query is parameterized. ``init_db()`` is idempotent and called once
at startup.
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime
from typing import Optional
from werkzeug.security import generate_password_hash, check_password_hash

# DB file lives alongside the app. Override with JOBLENS_DB for tests.
DB_PATH = os.environ.get(
    "JOBLENS_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "joblens.db"),
)


def _connect() -> sqlite3.Connection:
    """Open a connection with row access by column name and FK enforcement."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist and migrate schema safely."""
    with _connect() as conn:
        # 1. Users table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name          TEXT,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")

        # 2. Per-user applicant profiles
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id       TEXT PRIMARY KEY,
                dob           TEXT,
                category      TEXT,
                gender        TEXT,
                qualification TEXT,
                extra_json    TEXT,
                updated_at    TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        # 3. Notifications table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notifications (
                id               TEXT PRIMARY KEY,
                user_id          TEXT,
                created_at       TEXT NOT NULL,
                source_type      TEXT NOT NULL,
                source_url       TEXT,
                job_title        TEXT NOT NULL,
                data_json        TEXT NOT NULL,
                analysis_text    TEXT,
                last_date        TEXT,
                exam_date        TEXT,
                pinned           INTEGER NOT NULL DEFAULT 0,
                eligibility_json TEXT
            )
            """
        )

        # Schema migrations for existing databases
        notif_cols = [r["name"] for r in conn.execute("PRAGMA table_info(notifications)").fetchall()]
        if "user_id" not in notif_cols:
            conn.execute("ALTER TABLE notifications ADD COLUMN user_id TEXT")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_notifs_user ON notifications(user_id, pinned, created_at)")


# ─────────────────────────────────────────────────────────────────
#  User Authentication & Management
# ─────────────────────────────────────────────────────────────────

def _row_to_user(row: sqlite3.Row) -> dict:
    """Convert a user DB row to a safe dict (no password hash)."""
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"] or row["email"].split("@")[0].capitalize(),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_user(email: str, password: str, name: str = "") -> tuple[Optional[dict], Optional[str]]:
    """
    Register a new user account.
    Returns (user_dict, None) on success or (None, error_message) on failure.
    """
    email_clean = email.strip().lower()
    if not email_clean or "@" not in email_clean:
        return None, "A valid email address is required."
    if not password or len(password) < 6:
        return None, "Password must be at least 6 characters long."

    user_id = "usr_" + uuid.uuid4().hex[:16]
    now = datetime.utcnow().isoformat()
    pw_hash = generate_password_hash(password)
    display_name = name.strip() or email_clean.split("@")[0].capitalize()

    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO users (id, email, password_hash, name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, email_clean, pw_hash, display_name, now, now),
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return _row_to_user(row), None
    except sqlite3.IntegrityError:
        return None, "An account with this email already exists."
    except Exception as e:
        return None, f"Failed to create account: {e}"


def authenticate_user(email: str, password: str) -> Optional[dict]:
    """
    Validate credentials and return user dict if correct, else None.
    """
    email_clean = email.strip().lower()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email_clean,)).fetchone()
    if not row:
        return None
    if not check_password_hash(row["password_hash"], password):
        return None
    return _row_to_user(row)


def get_user_by_id(user_id: str) -> Optional[dict]:
    """Look up a user by ID."""
    if not user_id:
        return None
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row) if row else None


def get_user_by_email(email: str) -> Optional[dict]:
    """Look up a user by email."""
    if not email:
        return None
    email_clean = email.strip().lower()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email_clean,)).fetchone()
    return _row_to_user(row) if row else None


# ─────────────────────────────────────────────────────────────────
#  Notifications (Isolated per user)
# ─────────────────────────────────────────────────────────────────

def _row_to_notification(row: sqlite3.Row) -> dict:
    """Turn a DB row into a JSON-friendly dict with parsed sub-objects."""
    return {
        "id": row["id"],
        "user_id": row["user_id"] if "user_id" in row.keys() else None,
        "created_at": row["created_at"],
        "source_type": row["source_type"],
        "source_url": row["source_url"],
        "job_title": row["job_title"],
        "data": json.loads(row["data_json"]),
        "analysis_text": row["analysis_text"],
        "last_date": row["last_date"],
        "exam_date": row["exam_date"],
        "pinned": bool(row["pinned"]),
        "eligibility": json.loads(row["eligibility_json"]) if row["eligibility_json"] else None,
    }


def save_notification(
    user_id: str,
    data: dict,
    analysis_text: str = "",
    source_type: str = "url",
    source_url: str = "",
) -> str:
    """Persist an analysis result for a specific user. Returns the new notification id."""
    notif_id = str(uuid.uuid4())
    dates = data.get("dates") or {}
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO notifications
                (id, user_id, created_at, source_type, source_url, job_title,
                 data_json, analysis_text, last_date, exam_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                notif_id,
                user_id,
                datetime.utcnow().isoformat(),
                source_type,
                source_url,
                data.get("job_title") or "Job Notification",
                json.dumps(data),
                analysis_text,
                dates.get("last_date"),
                dates.get("exam_date"),
            ),
        )
    return notif_id


def list_notifications(user_id: str) -> list[dict]:
    """All notifications for the given user, pinned first, then newest first."""
    if not user_id:
        return []
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM notifications WHERE user_id = ? ORDER BY pinned DESC, created_at DESC",
            (user_id,),
        ).fetchall()
    return [_row_to_notification(r) for r in rows]


def get_notification(notif_id: str, user_id: Optional[str] = None) -> Optional[dict]:
    """
    Retrieve a notification by id.
    If user_id is provided, checks user ownership.
    If user_id is None, retrieves the notification (useful for direct shared links / sitemaps).
    """
    with _connect() as conn:
        if user_id:
            row = conn.execute(
                "SELECT * FROM notifications WHERE id = ? AND user_id = ?",
                (notif_id, user_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM notifications WHERE id = ?", (notif_id,)
            ).fetchone()
    return _row_to_notification(row) if row else None


def delete_notification(notif_id: str, user_id: str) -> bool:
    """Delete a notification owned by the user. Returns True if removed."""
    if not user_id:
        return False
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM notifications WHERE id = ? AND user_id = ?",
            (notif_id, user_id),
        )
    return cur.rowcount > 0


def set_pinned(notif_id: str, user_id: str, pinned: bool) -> bool:
    """Set the watchlist/pin flag for the user's notification."""
    if not user_id:
        return False
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE notifications SET pinned = ? WHERE id = ?",
            (1 if pinned else 0, notif_id),
        )
    return cur.rowcount > 0


def toggle_pinned(notif_id: str, user_id: str) -> Optional[bool]:
    """Flip the pin flag for the user's notification. Returns the new value, or None."""
    notif = get_notification(notif_id, user_id=user_id)
    if not notif:
        return None
    new_val = not notif["pinned"]
    set_pinned(notif_id, user_id, new_val)
    return new_val


def set_eligibility(notif_id: str, user_id: str, eligibility: dict) -> bool:
    """Cache an 'Am I eligible?' verdict on a user's notification."""
    if not user_id:
        return False
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE notifications SET eligibility_json = ? WHERE id = ? AND user_id = ?",
            (json.dumps(eligibility), notif_id, user_id),
        )
    return cur.rowcount > 0


def clear_all_eligibility(user_id: str) -> None:
    """Invalidate every cached verdict for this user (called when user's profile changes)."""
    if not user_id:
        return
    with _connect() as conn:
        conn.execute("UPDATE notifications SET eligibility_json = NULL WHERE user_id = ?", (user_id,))


def count_all_notifications() -> int:
    """Global aggregate count of notifications (for landing page stats)."""
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) as c FROM notifications").fetchone()
    return row["c"] if row else 0


def list_public_notifications(limit: int = 200) -> list[dict]:
    """Public list of recent notifications for sitemap and llms.txt."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM notifications ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_notification(r) for r in rows]


# ─────────────────────────────────────────────────────────────────
#  Profile (Isolated per user in user_profiles)
# ─────────────────────────────────────────────────────────────────

def get_profile(user_id: str) -> Optional[dict]:
    """The applicant profile for a specific user, or None if never set."""
    if not user_id:
        return None
    with _connect() as conn:
        row = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        return None
    return {
        "dob": row["dob"],
        "category": row["category"],
        "gender": row["gender"],
        "qualification": row["qualification"],
        "extra": json.loads(row["extra_json"]) if row["extra_json"] else {},
        "updated_at": row["updated_at"],
    }


def save_profile(user_id: str, dob: str, category: str, gender: str, qualification: str) -> dict:
    """Upsert the user's profile and invalidate cached eligibility for this user."""
    if not user_id:
        return {}
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_profiles (user_id, dob, category, gender, qualification, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                dob = excluded.dob,
                category = excluded.category,
                gender = excluded.gender,
                qualification = excluded.qualification,
                updated_at = excluded.updated_at
            """,
            (user_id, dob, category, gender, qualification, now),
        )
    clear_all_eligibility(user_id)
    return get_profile(user_id) or {}


def save_profile_extra(user_id: str, answers: dict) -> dict:
    """Merge dynamic eligibility answers into the user's profile."""
    if not user_id:
        return {}
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        row = conn.execute("SELECT extra_json FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
        extra = json.loads(row["extra_json"]) if row and row["extra_json"] else {}
        extra.update(answers)
        conn.execute(
            """
            INSERT INTO user_profiles (user_id, extra_json, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                extra_json = excluded.extra_json,
                updated_at = excluded.updated_at
            """,
            (user_id, json.dumps(extra), now),
        )
    clear_all_eligibility(user_id)
    return get_profile(user_id) or {}
