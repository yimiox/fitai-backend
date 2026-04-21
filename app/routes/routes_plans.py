"""
app/routes/routes_plans.py
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from generation.pipeline import (
    run_generation_pipeline,
    run_adjustment_pipeline
)

router = APIRouter()


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

@router.post("/generate-plan")   # ← removed response_model — no strict validation
async def generate_plan(request: GeneratePlanRequest):

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

    valid_goals = ["fat_loss", "muscle_gain", "endurance", "general"]
    if request.profile.get("goal") not in valid_goals:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid goal. Must be one of: {valid_goals}"
        )

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

@router.post("/adjust-plan")   # ← removed response_model
async def adjust_plan(request: AdjustPlanRequest):

    if not request.adjustment.strip():
        raise HTTPException(status_code=422, detail="Adjustment text cannot be empty")

    if len(request.adjustment) > 500:
        raise HTTPException(status_code=422, detail="Adjustment text too long (max 500 characters)")

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
        raise HTTPException(
            status_code=404,
            detail=f"No plan found for user {user_id}"
        )
    return plan
