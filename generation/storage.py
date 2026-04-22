"""
generation/storage.py

Supabase storage functions for plans and conversation memory.
"""

import os
from supabase import create_client

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)


# ─────────────────────────────────────────────
# PLAN STORAGE
# ─────────────────────────────────────────────

def save_plan(user_id: str, profile: dict, plan: dict, chunks: list) -> str:
    """Save generated plan and return the plan ID."""
    result = supabase.table("plans").insert({
        "user_id":     user_id,
        "profile":     profile,
        "plan":        plan,
        "chunks_used": chunks,
    }).execute()
    return result.data[0]["id"]


def get_latest_plan(user_id: str) -> dict | None:
    """Retrieve the most recent plan for a user."""
    result = (
        supabase.table("plans")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


# ─────────────────────────────────────────────
# CONVERSATION MEMORY  ← NEW for Upgrade 3
# ─────────────────────────────────────────────

def save_message(user_id: str, plan_id: str, role: str, message: str) -> None:
    """
    Save a single conversation turn to Supabase.

    Args:
        user_id:  UUID of the user
        plan_id:  UUID of the plan this conversation is about
        role:     'user' or 'assistant'
        message:  the message text
    """
    supabase.table("conversations").insert({
        "user_id": user_id,
        "plan_id": plan_id,
        "role":    role,
        "message": message,
    }).execute()


def get_conversation(user_id: str, plan_id: str, limit: int = 6) -> list[dict]:
    """
    Retrieve the last N conversation turns for a user+plan.
    Returns oldest first so the LLM sees them in order.

    Args:
        user_id:  UUID of the user
        plan_id:  UUID of the plan
        limit:    max number of messages to return (default 6 = 3 exchanges)

    Returns:
        list of {role, message} dicts ordered oldest → newest
    """
    result = (
        supabase.table("conversations")
        .select("role, message, created_at")
        .eq("user_id", user_id)
        .eq("plan_id", plan_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    # Reverse so oldest message is first
    messages = list(reversed(result.data or []))
    return [{"role": m["role"], "message": m["message"]} for m in messages]


def get_all_conversations(user_id: str, plan_id: str) -> list[dict]:
    """
    Retrieve ALL conversation turns for a user+plan.
    Used by the frontend to display the full chat history.
    """
    result = (
        supabase.table("conversations")
        .select("role, message, created_at")
        .eq("user_id", user_id)
        .eq("plan_id", plan_id)
        .order("created_at", desc=True)
        .execute()
    )
    messages = list(reversed(result.data or []))
    return [{"role": m["role"], "message": m["message"]} for m in messages]