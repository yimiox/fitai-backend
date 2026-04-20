"""
tests/test_builder.py
=====================
Unit tests for the profile builder calculations.
Run with: python -m pytest tests/ -v
"""

import pytest
from profile.builder import (
    build_profile, _compute_bmi, _compute_bmr, _classify_budget,
    _build_risk_flags, _infer_activity,
)


# ── BMI tests ────────────────────────────────────────────────────────────────

def test_bmi_normal():
    bmi, cat = _compute_bmi(170, 70)
    assert 24 < bmi < 25
    assert cat == "Normal weight"

def test_bmi_obese():
    _, cat = _compute_bmi(165, 100)
    assert cat == "Obese"

def test_bmi_underweight():
    _, cat = _compute_bmi(180, 50)
    assert cat == "Underweight"


# ── BMR tests ─────────────────────────────────────────────────────────────────

def test_bmr_male():
    # 30yo male, 175cm, 80kg → approx 1882 kcal
    bmr = _compute_bmr(30, "Male", 175, 80)
    assert 1800 < bmr < 1950

def test_bmr_female():
    bmr = _compute_bmr(25, "Female", 165, 60)
    assert 1350 < bmr < 1550


# ── Budget tier ───────────────────────────────────────────────────────────────

def test_budget_low():
    assert _classify_budget(50) == "low"

def test_budget_mid():
    assert _classify_budget(150) == "mid"

def test_budget_high():
    assert _classify_budget(300) == "high"


# ── Risk flags ────────────────────────────────────────────────────────────────

def test_diabetes_flag():
    flags = _build_risk_flags(["Type 2 diabetes"], [], None)
    assert any("blood sugar" in f.lower() for f in flags)

def test_knee_injury_flag():
    flags = _build_risk_flags([], ["Knee issues"], None)
    assert any("knee" in f.lower() for f in flags)

def test_no_flags_for_none():
    flags = _build_risk_flags(["None"], ["None"], None)
    assert flags == []


# ── Full profile build ────────────────────────────────────────────────────────

SAMPLE_PAYLOAD = {
    "age": 28, "sex": "Male", "height_cm": 175, "weight_kg": 78,
    "monthly_food_budget": 5000, "currency": "INR",
    "equipment": "full_gym", "primary_goal": "muscle_gain",
    "experience_level": "beginner", "timeline_weeks": 12,
    "training_days_per_week": 4, "work_type": "sedentary",
    "sleep_hours": 7, "stress_level": 4, "meals_per_day": 3,
    "conditions": ["None"], "injuries": ["None"],
    "diet_restrictions": ["None"],
}

def test_full_build_returns_profile():
    profile = build_profile(SAMPLE_PAYLOAD)
    assert profile.computed["bmi"] > 0
    assert profile.computed["tdee"] > 1000
    assert profile.computed["protein_g"] > 0
    assert profile.computed["budget_tier"] in ("low", "mid", "high")
    assert len(profile.rag_queries) >= 3
    assert "USER PROFILE SUMMARY" in profile.llm_context

def test_protein_higher_for_fat_loss():
    fat_loss = build_profile({**SAMPLE_PAYLOAD, "primary_goal": "fat_loss"})
    endurance = build_profile({**SAMPLE_PAYLOAD, "primary_goal": "endurance"})
    assert fat_loss.computed["protein_g"] > endurance.computed["protein_g"]

def test_deficit_for_fat_loss():
    p = build_profile({**SAMPLE_PAYLOAD, "primary_goal": "fat_loss"})
    assert p.computed["calorie_direction"] == "deficit"
    assert p.computed["calorie_delta"] < 0

def test_surplus_for_muscle_gain():
    p = build_profile({**SAMPLE_PAYLOAD, "primary_goal": "muscle_gain"})
    assert p.computed["calorie_direction"] == "surplus"

def test_vegetarian_flag():
    p = build_profile({**SAMPLE_PAYLOAD, "diet_restrictions": ["Vegetarian"]})
    assert p.computed["is_vegetarian"] is True
    assert any("plant-based" in q for q in p.rag_queries)

def test_vegan_implies_vegetarian():
    p = build_profile({**SAMPLE_PAYLOAD, "diet_restrictions": ["Vegan"]})
    assert p.computed["is_vegan"] is True
    assert p.computed["is_vegetarian"] is True

def test_safety_floor_calories():
    # Extreme deficit should not drop below 1200 kcal
    p = build_profile({**SAMPLE_PAYLOAD, "weight_kg": 40, "primary_goal": "fat_loss"})
    assert p.computed["target_calories"] >= 1200
