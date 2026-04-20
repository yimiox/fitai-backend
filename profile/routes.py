"""
backend/profile/routes.py
==========================
FastAPI routes for profile building, storing, and retrieving.

Endpoints:
  POST /profile/build         — Build + store a profile, return computed values
  GET  /profile/{profile_id}  — Retrieve a stored profile
  GET  /profile/              — List recent profiles (admin)
"""

import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client

from profile.builder import build_profile

load_dotenv()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/profile", tags=["profile"])


def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("Supabase credentials not set in .env")
    return create_client(url, key)


class QuestionnairePayload(BaseModel):
    """Matches the shape of data sent by the React frontend."""
    # Bio
    age: int
    sex: str
    height_cm: float
    weight_kg: float
    body_type: Optional[str] = None

    # Economic
    monthly_food_budget: float = 0
    currency: str = "INR"
    equipment: str = "bodyweight"
    supplement_budget: Optional[str] = None

    # Goals
    primary_goal: str = "general"
    experience_level: str = "beginner"
    timeline_weeks: int = 12
    training_days_per_week: Optional[int] = 4
    workout_duration: Optional[str] = None

    # Health
    conditions: Optional[list[str]] = []
    injuries: Optional[list[str]] = []
    diet_restrictions: Optional[list[str]] = []
    allergies: Optional[str] = None
    medication: Optional[str] = None

    # Lifestyle
    meals_per_day: Optional[int] = 3
    cooking_time_mins: Optional[str] = None
    work_type: str = "sedentary"
    sleep_hours: Optional[float] = 7.0
    stress_level: Optional[int] = 5
    training_time: Optional[str] = None
    notes: Optional[str] = None


@router.post("/build")
def build_and_store_profile(payload: QuestionnairePayload):
    """
    Main endpoint called by the React frontend after questionnaire completion.

    1. Builds computed profile from raw answers
    2. Stores both raw + computed in Supabase
    3. Returns profile_id + full computed data for the results screen

    The stored profile_id is used in Stage 4 to generate the plan.
    """
    try:
        # Build the profile
        profile = build_profile(payload.model_dump())

        # Store in Supabase
        supabase = get_supabase()
        row = {
            "raw_data":    profile.raw,
            "computed":    profile.computed,
            "rag_queries": profile.rag_queries,
            "llm_context": profile.llm_context,
        }
        result = supabase.table("user_profiles").insert(row).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to save profile")

        profile_id = result.data[0]["id"]
        logger.info(f"Profile stored: {profile_id}")

        return {
            "profile_id":  profile_id,
            "raw":         profile.raw,
            "computed":    profile.computed,
            "rag_queries": profile.rag_queries,
        }

    except Exception as e:
        logger.error(f"Profile build error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{profile_id}")
def get_profile(profile_id: str):
    """Retrieve a stored profile by ID."""
    supabase = get_supabase()
    result = (
        supabase.table("user_profiles")
        .select("*")
        .eq("id", profile_id)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Profile not found")
    return result.data


@router.get("/")
def list_profiles(limit: int = 20):
    """Admin: list most recent profiles."""
    supabase = get_supabase()
    result = (
        supabase.table("user_profiles")
        .select("id, created_at, computed")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []
