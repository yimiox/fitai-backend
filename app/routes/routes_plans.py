"""
app/routes/routes_plans.py

FastAPI router for plan generation and adjustment.
Includes prompt injection defence on all free-text inputs.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from generation.pipeline import run_generation_pipeline, run_adjustment_pipeline
from generation.prompts import INJECTION_KEYWORDS

router = APIRouter()


# ─────────────────────────────────────────────
# INJECTION DEFENCE
# ─────────────────────────────────────────────

def injection_check(text: str) -> bool:
    """
    Returns True if the text contains prompt injection patterns.
    Case-insensitive. Checks against the shared INJECTION_KEYWORDS list.
    """
    text_lower = text.lower()
    for keyword in INJECTION_KEYWORDS:
        if keyword.lower() in text_lower:
            return True
    return False


def safe_text(text: str, field_name: str) -> str:
    """
    Validate a free-text field for injection attempts.
    Raises HTTPException 400 if injection detected.
    Returns the stripped text if safe.
    """
    cleaned = text.strip()
    if injection_check(cleaned):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input in '{field_name}': message contains disallowed content."
        )
    return cleaned


# ─────────────────────────────────────────────
# REQUEST SCHEMAS
# ─────────────────────────────────────────────

class GeneratePlanRequest(BaseModel):
    user_id: str
    profile: dict


class AdjustPlanRequest(BaseModel):
    user_id: str
    plan_id: str
    adjustment: str


# ─────────────────────────────────────────────
# POST /api/generate-plan
# ─────────────────────────────────────────────

@router.post("/generate-plan")
async def generate_plan(request: GeneratePlanRequest):

    # Validate required profile fields
    required_fields = [
        "goal", "experience", "budget_tier",
        "bmi", "tdee", "equipment",
        "health_conditions", "dietary_restrictions",
        "age", "sex", "height_cm", "weight_kg"
    ]
    missing = [f for f in required_fields if f not in request.profile]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required profile fields: {missing}"
        )

    # Validate goal value
    valid_goals = ["fat_loss", "muscle_gain", "endurance", "general", "recomp", "strength"]
    if request.profile.get("goal") not in valid_goals:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid goal. Must be one of: {valid_goals}"
        )

    # Injection check on any free-text profile fields
    for field in ["notes", "other_conditions", "other_restrictions"]:
        if field in request.profile and request.profile[field]:
            safe_text(str(request.profile[field]), field)

    try:
        plan = run_generation_pipeline(
            user_id=request.user_id,
            profile=request.profile
        )
        return plan

    except Exception as e:
        print(f"[routes_plans] Generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Plan generation failed: {str(e)}"
        )


# ─────────────────────────────────────────────
# POST /api/adjust-plan
# ─────────────────────────────────────────────

@router.post("/adjust-plan")
async def adjust_plan(request: AdjustPlanRequest):

    # Length checks
    if not request.adjustment.strip():
        raise HTTPException(status_code=422, detail="Adjustment text cannot be empty")
    if len(request.adjustment) > 500:
        raise HTTPException(status_code=422, detail="Adjustment text too long (max 500 characters)")

    # ── INJECTION CHECK — blocks before LLM is called ──────────────
    safe_text(request.adjustment, "adjustment")
    # ───────────────────────────────────────────────────────────────

    try:
        updated_plan = run_adjustment_pipeline(
            user_id=request.user_id,
            plan_id=request.plan_id,
            adjustment=request.adjustment
        )
        return updated_plan

    except Exception as e:
        print(f"[routes_plans] Adjustment failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Plan adjustment failed: {str(e)}"
        )


# ─────────────────────────────────────────────
# GET /api/plan/{user_id}
# ─────────────────────────────────────────────

@router.get("/plan/{user_id}")
async def get_plan(user_id: str):
    from generation.storage import get_latest_plan
    plan = get_latest_plan(user_id)
    if not plan:
        raise HTTPException(status_code=404, detail=f"No plan found for user {user_id}")
    return plan