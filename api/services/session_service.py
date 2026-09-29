"""
api/services/session_service.py
Service managing conversation metadata and user-visible chat history.
Persists sessions and ordered messages to a lightweight SQLite database at memory/conversations.db.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from api.schemas.sessions import (
    SessionDetail,
    SessionListResponse,
    SessionMessage,
    SessionSummary,
)

logger = logging.getLogger(__name__)

DB_PATH = Path("./memory/conversations.db")


def _get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row factory and WAL mode enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Initialize database tables for sessions and messages."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                citations TEXT,
                consensus_score REAL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
        conn.commit()


# Initialize database schema on module load
init_db()


def record_turn(
    session_id: str,
    query: str,
    answer: str,
    citations: Optional[List[str]] = None,
    consensus_score: Optional[float] = None,
) -> None:
    """
    Record a completed user and assistant turn into the session store.
    Updates the session title on the first turn and updates the updated_at timestamp.
    """
    if not session_id or not query:
        return

    now = datetime.now(timezone.utc).isoformat()
    citations_json = json.dumps(citations or [])

    # Derive clean title from the initial query (first 60 characters)
    title = query.strip()
    if len(title) > 60:
        title = title[:57] + "..."

    try:
        with _get_connection() as conn:
            # Upsert session
            conn.execute("""
                INSERT INTO sessions (session_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
            """, (session_id, title, now, now))

            # Insert user turn
            conn.execute("""
                INSERT INTO messages (session_id, role, content, timestamp, citations, consensus_score)
                VALUES (?, 'user', ?, ?, '[]', NULL)
            """, (session_id, query, now))

            # Insert assistant turn
            conn.execute("""
                INSERT INTO messages (session_id, role, content, timestamp, citations, consensus_score)
                VALUES (?, 'assistant', ?, ?, ?, ?)
            """, (session_id, answer, now, citations_json, consensus_score))

            conn.commit()
            logger.info(f"[SessionService] Recorded turn for session '{session_id}'")
    except Exception as e:
        logger.error(f"[SessionService] Failed to record turn: {e}", exc_info=True)


def list_sessions() -> SessionListResponse:
    """List all active conversation sessions ordered by most recent activity."""
    try:
        with _get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    s.session_id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
            """)
            rows = cursor.fetchall()

            summaries = [
                SessionSummary(
                    session_id=row["session_id"],
                    title=row["title"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    message_count=int(row["message_count"]),
                )
                for row in rows
            ]
            return SessionListResponse(sessions=summaries, total=len(summaries))
    except Exception as e:
        logger.error(f"[SessionService] Failed to list sessions: {e}", exc_info=True)
        return SessionListResponse(sessions=[], total=0)


def get_session_history(session_id: str) -> Optional[SessionDetail]:
    """Retrieve full chronological conversation history for a given session."""
    try:
        with _get_connection() as conn:
            sess_cursor = conn.execute(
                "SELECT session_id, title, created_at, updated_at FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            sess_row = sess_cursor.fetchone()
            if not sess_row:
                return None

            msg_cursor = conn.execute("""
                SELECT role, content, timestamp, citations, consensus_score
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
            """, (session_id,))
            msg_rows = msg_cursor.fetchall()

            messages = []
            for row in msg_rows:
                citations = []
                if row["citations"]:
                    try:
                        citations = json.loads(row["citations"])
                    except Exception:
                        citations = []
                messages.append(
                    SessionMessage(
                        role=row["role"],
                        content=row["content"],
                        timestamp=row["timestamp"],
                        citations=citations,
                        consensus_score=row["consensus_score"],
                    )
                )

            return SessionDetail(
                session_id=sess_row["session_id"],
                title=sess_row["title"],
                created_at=sess_row["created_at"],
                updated_at=sess_row["updated_at"],
                messages=messages,
            )
    except Exception as e:
        logger.error(f"[SessionService] Failed to get session history: {e}", exc_info=True)
        return None


def delete_session(session_id: str) -> bool:
    """
    Delete a specific conversation session and its associated messages.
    Does NOT wipe global semantic memory, user profile, or knowledge graph.
    """
    try:
        with _get_connection() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"[SessionService] Deleted session '{session_id}'")
            return deleted
    except Exception as e:
        logger.error(f"[SessionService] Failed to delete session: {e}", exc_info=True)
        return False
