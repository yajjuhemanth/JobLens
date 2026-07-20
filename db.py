"""
TrueNotice — SQLite persistence layer
======================================
Uses Python's stdlib ``sqlite3`` (no extra dependency). Stores analyzed
job notifications and a single applicant profile so TrueNotice works like a
real, returning-user app: a Home dashboard, a watchlist, per-notification
eligibility caching, and grounded Q&A — all without re-running analysis.

Every query is parameterized. ``init_db()`` is idempotent and called once
at startup.
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime
from typing import Optional

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
    """Create tables if they don't exist. Safe to call repeatedly."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notifications (
                id               TEXT PRIMARY KEY,
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profile (
                id            INTEGER PRIMARY KEY CHECK (id = 1),
                dob           TEXT,
                category      TEXT,
                gender        TEXT,
                qualification TEXT,
                extra_json    TEXT,
                updated_at    TEXT
            )
            """
        )
        # extra_json was added after initial release — patch existing DBs.
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(profile)").fetchall()]
        if "extra_json" not in cols:
            conn.execute("ALTER TABLE profile ADD COLUMN extra_json TEXT")


# ─────────────────────────────────────────────────────────────────
#  Notifications
# ─────────────────────────────────────────────────────────────────

def _row_to_notification(row: sqlite3.Row) -> dict:
    """Turn a DB row into a JSON-friendly dict with parsed sub-objects."""
    return {
        "id": row["id"],
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
    data: dict,
    analysis_text: str = "",
    source_type: str = "url",
    source_url: str = "",
) -> str:
    """Persist an analysis result. Returns the new notification id."""
    notif_id = str(uuid.uuid4())
    dates = data.get("dates") or {}
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO notifications
                (id, created_at, source_type, source_url, job_title,
                 data_json, analysis_text, last_date, exam_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                notif_id,
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


def list_notifications() -> list[dict]:
    """All notifications, pinned first, then newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM notifications ORDER BY pinned DESC, created_at DESC"
        ).fetchall()
    return [_row_to_notification(r) for r in rows]


def get_notification(notif_id: str) -> Optional[dict]:
    """A single notification by id, or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM notifications WHERE id = ?", (notif_id,)
        ).fetchone()
    return _row_to_notification(row) if row else None


def delete_notification(notif_id: str) -> bool:
    """Delete a notification. Returns True if a row was removed."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM notifications WHERE id = ?", (notif_id,))
    return cur.rowcount > 0


def set_pinned(notif_id: str, pinned: bool) -> bool:
    """Set the watchlist/pin flag. Returns True if the row exists."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE notifications SET pinned = ? WHERE id = ?",
            (1 if pinned else 0, notif_id),
        )
    return cur.rowcount > 0


def toggle_pinned(notif_id: str) -> Optional[bool]:
    """Flip the pin flag. Returns the new value, or None if not found."""
    notif = get_notification(notif_id)
    if not notif:
        return None
    new_val = not notif["pinned"]
    set_pinned(notif_id, new_val)
    return new_val


def set_eligibility(notif_id: str, eligibility: dict) -> bool:
    """Cache an 'Am I eligible?' verdict on a notification."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE notifications SET eligibility_json = ? WHERE id = ?",
            (json.dumps(eligibility), notif_id),
        )
    return cur.rowcount > 0


def clear_all_eligibility() -> None:
    """Invalidate every cached verdict (call when the profile changes)."""
    with _connect() as conn:
        conn.execute("UPDATE notifications SET eligibility_json = NULL")


# ─────────────────────────────────────────────────────────────────
#  Profile (single-user local app — one row, id = 1)
# ─────────────────────────────────────────────────────────────────

def get_profile() -> Optional[dict]:
    """The applicant profile, or None if never set."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM profile WHERE id = 1").fetchone()
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


def save_profile(dob: str, category: str, gender: str, qualification: str) -> dict:
    """Upsert the single profile row and invalidate cached eligibility."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO profile (id, dob, category, gender, qualification, updated_at)
            VALUES (1, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                dob = excluded.dob,
                category = excluded.category,
                gender = excluded.gender,
                qualification = excluded.qualification,
                updated_at = excluded.updated_at
            """,
            (dob, category, gender, qualification, datetime.utcnow().isoformat()),
        )
    clear_all_eligibility()
    return get_profile()


def save_profile_extra(answers: dict) -> dict:
    """Merge dynamic, notification-specific eligibility answers (domicile,
    local language, experience, etc.) into the profile and invalidate cached
    eligibility verdicts, since the answer set changed."""
    with _connect() as conn:
        row = conn.execute("SELECT extra_json FROM profile WHERE id = 1").fetchone()
        extra = json.loads(row["extra_json"]) if row and row["extra_json"] else {}
        extra.update(answers)
        conn.execute(
            """
            INSERT INTO profile (id, extra_json, updated_at) VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                extra_json = excluded.extra_json,
                updated_at = excluded.updated_at
            """,
            (json.dumps(extra), datetime.utcnow().isoformat()),
        )
    clear_all_eligibility()
    return get_profile()
