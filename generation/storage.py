import os
from supabase import create_client

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)

def save_plan(user_id: str, profile: dict,
              plan: dict, chunks: list[dict]) -> str:
    """Save generated plan and return the plan ID."""
    result = supabase.table("plans").insert({
        "user_id": user_id,
        "profile": profile,
        "plan": plan,
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


def update_plan(plan_id: str, updated_plan: dict) -> None:
    """Replace a plan (used when user requests adjustments)."""
    supabase.table("plans").update({
        "plan": updated_plan,
        "version": supabase.rpc("increment", {"row_id": plan_id})
    }).eq("id", plan_id).execute()