"""
guardrails/rbac.py  —  P6-T2
Role-Based Access Control, Tool Access Control, and Policy Enforcement.

Concepts covered:
  - Role-Based Access Control (RBAC)   (3 roles: admin / analyst / viewer)
  - Tool Access Control                (per-role tool allowlist)
  - Policy Enforcement                 (policy violations logged + blocked)
  - Permission-Aware Tool Access       (MCP server reads role before executing)
  - Guardrail Policies                 (all policies in config/guardrail_policies.json)

Role hierarchy:
  admin   → full access: all docs, all tools, ingestion, web browsing
  analyst → standard access: own docs + RAG tools; NO web browser, NO ingestion
  viewer  → read-only: search + answer only; NO ingestion, NO tool calls

Policy config at /config/guardrail_policies.json — single source of truth.
All enforcement decisions logged to /logs/security_events.jsonl.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_POLICY_PATH = Path("./config/guardrail_policies.json")
_SECURITY_LOG = Path("./logs/security_events.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# Default policy config (written to disk on first use if missing)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_POLICIES = {
    "roles": {
        "admin": {
            "allowed_tools": [
                "document_search", "ingest_document", "ask_agent",
                "web_research", "web_search", "run_code", "fetch_url", "graph_rag",
            ],
            "allowed_doc_categories": ["research", "technical", "report", "general"],
            "can_ingest": True,
            "can_use_browser": True,
            "can_delete": True,
            "description": "Full access — all documents, all tools, all operations",
        },
        "analyst": {
            "allowed_tools": [
                "document_search", "ask_agent", "graph_rag", "run_code",
                "web_search", "web_research",
            ],
            "allowed_doc_categories": ["research", "technical", "report", "general"],
            "can_ingest": False,
            "can_use_browser": False,
            "can_delete": False,
            "description": "Standard access — RAG tools only, no ingestion or web browsing",
        },
        "viewer": {
            "allowed_tools": [
                "document_search",
            ],
            "allowed_doc_categories": ["general", "report"],
            "can_ingest": False,
            "can_use_browser": False,
            "can_delete": False,
            "description": "Read-only access — search and answer only",
        },
    },
    "global_rules": {
        "max_query_length": 1000,
        "max_results_per_query": 10,
        "require_citations": True,
        "allow_web_search_roles": ["admin"],
    },
}


def _load_policies() -> dict:
    """Load guardrail policies from disk. Write defaults if not found."""
    _POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _POLICY_PATH.exists():
        _POLICY_PATH.write_text(json.dumps(_DEFAULT_POLICIES, indent=2))
        logger.info(f"[RBAC] Default policies written to {_POLICY_PATH}")
    try:
        return json.loads(_POLICY_PATH.read_text())
    except Exception as e:
        logger.warning(f"[RBAC] Failed to load policies: {e} — using defaults")
        return _DEFAULT_POLICIES


def _log_policy_event(
    event_type: str,
    user_role: str,
    resource: str,
    reason: str,
    session_id: str = "",
) -> None:
    """Log a policy enforcement event to security_events.jsonl."""
    _SECURITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "user_role": user_role,
        "resource": resource,
        "reason": reason,
        "session_id": session_id,
    }
    with open(_SECURITY_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# RBAC check
# ─────────────────────────────────────────────────────────────────────────────

class RBACResult:
    """Result of an RBAC check."""
    __slots__ = ("allowed", "reason", "role", "policy_name")

    def __init__(self, allowed: bool, reason: str = "", role: str = "", policy_name: str = ""):
        self.allowed = allowed
        self.reason = reason
        self.role = role
        self.policy_name = policy_name

    def __bool__(self):
        return self.allowed


def check_tool_access(
    tool_name: str,
    user_role: str,
    session_id: str = "",
) -> RBACResult:
    """
    Check if a user role is allowed to invoke a specific tool.

    Called by:
      - Research Agent (P3-T4) before each tool call
      - MCP server (P10) before executing any tool
      - Browser Agent (P11) before web navigation

    Args:
        tool_name:   Name of the tool being invoked.
        user_role:   RBAC role of the current user.
        session_id:  For logging.

    Returns:
        RBACResult with .allowed (bool) and .reason (str if blocked).

    Concepts: Tool Access Control, Permission-Aware Tool Access, RBAC
    """
    policies = _load_policies()
    role_config = policies.get("roles", {}).get(user_role)

    if role_config is None:
        reason = f"Unknown role '{user_role}' — access denied by default"
        _log_policy_event("tool_blocked", user_role, tool_name, reason, session_id)
        return RBACResult(allowed=False, reason=reason, role=user_role)

    allowed_tools: List[str] = role_config.get("allowed_tools", [])
    norm_tool = "web_search" if tool_name == "web_research" else tool_name
    norm_allowed = [("web_search" if t == "web_research" else t) for t in allowed_tools]

    if norm_tool not in norm_allowed:
        reason = (
            f"Role '{user_role}' is not permitted to use tool '{tool_name}'. "
            f"Allowed tools: {allowed_tools}"
        )
        _log_policy_event("tool_blocked", user_role, tool_name, reason, session_id)
        logger.warning(f"[RBAC] BLOCKED: role={user_role} tool={tool_name}")
        return RBACResult(
            allowed=False,
            reason=reason,
            role=user_role,
            policy_name="tool_access_control",
        )

    logger.debug(f"[RBAC] ALLOWED: role={user_role} tool={tool_name}")
    return RBACResult(allowed=True, role=user_role)


def check_doc_category_access(
    doc_category: str,
    user_role: str,
    session_id: str = "",
) -> RBACResult:
    """
    Check if a user role can access documents of a specific category.

    Used by the retriever to filter results by doc_category before returning
    to the generator.

    Concept: RBAC, Policy Enforcement
    """
    policies = _load_policies()
    role_config = policies.get("roles", {}).get(user_role, {})
    allowed_categories: List[str] = role_config.get("allowed_doc_categories", [])

    if doc_category not in allowed_categories:
        reason = (
            f"Role '{user_role}' cannot access doc_category='{doc_category}'. "
            f"Allowed: {allowed_categories}"
        )
        _log_policy_event("doc_access_blocked", user_role, doc_category, reason, session_id)
        return RBACResult(
            allowed=False,
            reason=reason,
            role=user_role,
            policy_name="doc_category_access",
        )

    return RBACResult(allowed=True, role=user_role)


def check_ingestion_permission(
    user_role: str,
    session_id: str = "",
) -> RBACResult:
    """
    Check if a user role is allowed to ingest (add) documents.

    Concept: Policy Enforcement
    """
    policies = _load_policies()
    role_config = policies.get("roles", {}).get(user_role, {})
    can_ingest = role_config.get("can_ingest", False)

    if not can_ingest:
        reason = f"Role '{user_role}' does not have document ingestion permission."
        _log_policy_event("ingestion_blocked", user_role, "ingest_document", reason, session_id)
        return RBACResult(
            allowed=False,
            reason=reason,
            role=user_role,
            policy_name="ingestion_permission",
        )

    return RBACResult(allowed=True, role=user_role)


# ─────────────────────────────────────────────────────────────────────────────
# Filter retrieved chunks by RBAC (remove inaccessible doc categories)
# ─────────────────────────────────────────────────────────────────────────────

def filter_chunks_by_role(
    chunks,
    user_role,
    session_id="",
):
    """
    Filter retrieved chunks according to the user's
    document-category permissions.

    Missing or unknown document categories are denied
    rather than treated as public/general content.
    """

    filtered = []

    for chunk in chunks:

        metadata = chunk.get(
            "metadata",
            {},
        )

        doc_category = metadata.get(
            "doc_category"
        )

        # Fail closed:
        # missing category must never grant access.
        if not doc_category:
            logger.warning(
                "[RBAC] BLOCKED chunk with missing "
                "doc_category for role=%s",
                user_role,
            )
            continue

        result = check_doc_category_access(
            doc_category=doc_category,
            user_role=user_role,
            session_id=session_id,
        )

        if result.allowed:
            filtered.append(chunk)

    return filtered
# ─────────────────────────────────────────────────────────────────────────────
# Get role permissions summary (for Gradio UI / MCP context)
# ─────────────────────────────────────────────────────────────────────────────

def get_role_permissions(user_role: str) -> Dict:
    """Return the full permissions dict for a given role."""
    policies = _load_policies()
    return policies.get("roles", {}).get(user_role, {})
