"""
profile/builder.py
==================
STEP 3 BACKEND — Take raw questionnaire answers and compute a structured
user profile with all the derived values the LLM needs to generate a plan.

What we compute:
  - BMI + BMI category
  - BMR (Basal Metabolic Rate) using Mifflin-St Jeor equation
  - TDEE (Total Daily Energy Expenditure) — BMR × activity multiplier
  - Target calories (TDEE adjusted for goal)
  - Minimum protein intake (grams/day) based on goal + body weight
  - Budget tier (low / mid / high) normalized across currencies
  - Goal label and RAG query tags
  - Risk flags (conditions that constrain plan generation)

Everything returned here gets saved to Supabase and injected into the LLM prompt.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import math


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RawProfile:
    """Exactly what the questionnaire frontend sends — no computation yet."""
    # Bio
    age: int
    sex: str                       # "Male" | "Female" | "Other"
    height_cm: float
    weight_kg: float
    body_type: Optional[str] = None

    # Economic
    monthly_food_budget: float = 0.0
    currency: str = "INR"
    equipment: str = "bodyweight"
    supplement_budget: Optional[str] = None

    # Goals
    primary_goal: str = "general"
    experience_level: str = "beginner"
    timeline_weeks: int = 12
    training_days_per_week: int = 4
    workout_duration: Optional[str] = None

    # Health
    conditions: list[str] = field(default_factory=list)
    injuries: list[str] = field(default_factory=list)
    diet_restrictions: list[str] = field(default_factory=list)
    allergies: Optional[str] = None
    medication: Optional[str] = None

    # Lifestyle
    meals_per_day: int = 3
    cooking_time_mins: Optional[str] = None
    work_type: str = "sedentary"
    sleep_hours: float = 7.0
    stress_level: int = 5
    training_time: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ComputedProfile:
    """All derived values calculated from the raw questionnaire answers."""
    # Body composition
    bmi: float
    bmi_category: str              # "Underweight" | "Normal" | "Overweight" | "Obese"

    # Energy
    bmr: float                     # Basal Metabolic Rate (kcal/day)
    tdee: float                    # Total Daily Energy Expenditure (kcal/day)
    target_calories: float         # Adjusted for goal
    calorie_delta: int             # e.g. -500 (deficit) or +300 (surplus)
    calorie_direction: str         # "deficit" | "surplus" | "maintenance"

    # Macros
    protein_g: float               # Minimum grams of protein per day
    carbs_g: float                 # Approximate carb target
    fat_g: float                   # Approximate fat target

    # Budget
    budget_tier: str               # "low" | "mid" | "high"
    budget_usd_monthly: float      # Normalised to USD for comparisons
    currency: str
    monthly_food_budget: float

    # Goal metadata
    goal_label: str                # Human-readable goal name
    goal_tag: str                  # snake_case for RAG queries
    experience_level: str
    equipment_label: str

    # Flags for the LLM
    is_vegetarian: bool
    is_vegan: bool
    has_conditions: bool
    condition_list: list[str]
    injury_list: list[str]
    risk_flags: list[str]          # Conditions that affect exercise/diet selection


@dataclass
class BuiltProfile:
    """Full profile ready to store in Supabase and pass to the LLM."""
    raw: dict                      # Original questionnaire answers
    computed: dict                 # All derived values
    rag_queries: list[str]         # Pre-built retrieval queries for Stage 4
    llm_context: str               # Formatted text block for the LLM prompt


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Activity multipliers for TDEE (Mifflin-St Jeor standard)
ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,     # Desk job, little exercise
    "light":     1.375,   # Light activity 1–3 days/week
    "moderate":  1.55,    # Moderate activity 3–5 days/week
    "active":    1.725,   # Hard exercise 6–7 days/week
    "very_active": 1.9,   # Physical job + training
}

# Calorie adjustments by goal
GOAL_CALORIE_DELTA = {
    "fat_loss":    -500,
    "muscle_gain": +300,
    "recomp":        0,   # Maintenance calories, macro-shifted
    "endurance":   -200,
    "strength":    +200,
    "general":       0,
}

# Protein targets in g/kg bodyweight by goal
PROTEIN_TARGETS = {
    "fat_loss":    2.2,   # Higher protein during deficit preserves muscle
    "muscle_gain": 1.8,
    "recomp":      2.2,
    "endurance":   1.4,
    "strength":    2.0,
    "general":     1.6,
}

# Goal display labels
GOAL_LABELS = {
    "fat_loss":    "Fat loss",
    "muscle_gain": "Muscle gain",
    "recomp":      "Body recomposition",
    "endurance":   "Endurance / cardio",
    "strength":    "Strength / powerlifting",
    "general":     "General health",
}

# Rough USD conversion rates for budget tier classification
# (approximate — for tier classification only, not financial advice)
USD_RATES = {
    "INR": 0.012,  "USD": 1.0,   "GBP": 1.27, "EUR": 1.08,
    "AUD": 0.65,   "CAD": 0.73,  "SGD": 0.74, "AED": 0.27,
}

# Budget tiers in USD/month
BUDGET_TIERS = {
    "low":  (0, 80),
    "mid":  (80, 250),
    "high": (250, float("inf")),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main builder function
# ─────────────────────────────────────────────────────────────────────────────

def build_profile(raw_data: dict) -> BuiltProfile:
    """
    Take raw questionnaire data (dict from the API request body) and return
    a fully computed BuiltProfile ready to store and pass to the LLM.

    Args:
        raw_data: dict with questionnaire answers (matches RawProfile fields)

    Returns:
        BuiltProfile with raw answers, all computed values, and pre-built RAG queries
    """
    # Parse raw data into typed dataclass
    raw = _parse_raw(raw_data)

    # Compute everything
    bmi, bmi_cat      = _compute_bmi(raw.height_cm, raw.weight_kg)
    bmr               = _compute_bmr(raw.age, raw.sex, raw.height_cm, raw.weight_kg)
    activity          = _infer_activity(raw.work_type, raw.training_days_per_week, raw.stress_level)
    tdee              = round(bmr * ACTIVITY_MULTIPLIERS.get(activity, 1.375))
    delta             = GOAL_CALORIE_DELTA.get(raw.primary_goal, 0)
    target_cal        = max(1200, tdee + delta)   # Never below 1200 kcal (safety floor)
    protein_g         = round(raw.weight_kg * PROTEIN_TARGETS.get(raw.primary_goal, 1.6), 1)
    fat_g             = round(target_cal * 0.25 / 9, 1)          # 25% of calories from fat
    carbs_g           = round((target_cal - protein_g * 4 - fat_g * 9) / 4, 1)
    budget_usd        = raw.monthly_food_budget * USD_RATES.get(raw.currency, 1.0)
    budget_tier       = _classify_budget(budget_usd)
    conditions        = [c for c in (raw.conditions or []) if c != "None"]
    injuries          = [i for i in (raw.injuries or []) if i != "None"]
    diet_restrictions = [d for d in (raw.diet_restrictions or []) if d != "None"]
    is_vegan          = "Vegan" in diet_restrictions
    is_vegetarian     = is_vegan or "Vegetarian" in diet_restrictions
    risk_flags        = _build_risk_flags(conditions, injuries, raw.medication)

    computed = ComputedProfile(
        bmi=round(bmi, 1),
        bmi_category=bmi_cat,
        bmr=round(bmr),
        tdee=round(tdee),
        target_calories=round(target_cal),
        calorie_delta=delta,
        calorie_direction="deficit" if delta < 0 else "surplus" if delta > 0 else "maintenance",
        protein_g=protein_g,
        carbs_g=max(50, carbs_g),
        fat_g=fat_g,
        budget_tier=budget_tier,
        budget_usd_monthly=round(budget_usd, 2),
        currency=raw.currency,
        monthly_food_budget=raw.monthly_food_budget,
        goal_label=GOAL_LABELS.get(raw.primary_goal, raw.primary_goal),
        goal_tag=raw.primary_goal,
        experience_level=raw.experience_level,
        equipment_label=raw.equipment,
        is_vegetarian=is_vegetarian,
        is_vegan=is_vegan,
        has_conditions=len(conditions) > 0,
        condition_list=conditions,
        injury_list=injuries,
        risk_flags=risk_flags,
    )

    rag_queries   = _build_rag_queries(raw, computed, diet_restrictions)
    llm_context   = _format_llm_context(raw, computed, diet_restrictions)

    return BuiltProfile(
        raw=asdict(raw),
        computed=asdict(computed),
        rag_queries=rag_queries,
        llm_context=llm_context,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Calculation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_raw(data: dict) -> RawProfile:
    """Safely parse a dict into a RawProfile, with type coercion."""
    return RawProfile(
        age=int(data.get("age", 25)),
        sex=data.get("sex", "Male"),
        height_cm=float(data.get("height_cm", 170)),
        weight_kg=float(data.get("weight_kg", 70)),
        body_type=data.get("body_type"),
        monthly_food_budget=float(data.get("monthly_food_budget", 0) or 0),
        currency=data.get("currency", "INR"),
        equipment=data.get("equipment", "bodyweight"),
        supplement_budget=data.get("supplement_budget"),
        primary_goal=data.get("primary_goal", "general"),
        experience_level=data.get("experience_level", "beginner"),
        timeline_weeks=int(data.get("timeline_weeks", 12) or 12),
        training_days_per_week=int(data.get("training_days_per_week", 4) or 4),
        workout_duration=data.get("workout_duration"),
        conditions=data.get("conditions") or [],
        injuries=data.get("injuries") or [],
        diet_restrictions=data.get("diet_restrictions") or [],
        allergies=data.get("allergies"),
        medication=data.get("medication"),
        meals_per_day=int(data.get("meals_per_day", 3) or 3),
        cooking_time_mins=data.get("cooking_time_mins"),
        work_type=data.get("work_type", "sedentary"),
        sleep_hours=float(data.get("sleep_hours", 7) or 7),
        stress_level=int(data.get("stress_level", 5) or 5),
        training_time=data.get("training_time"),
        notes=data.get("notes"),
    )


def _compute_bmi(height_cm: float, weight_kg: float) -> tuple[float, str]:
    """
    BMI = weight(kg) / height(m)²
    Categories per WHO standard.
    """
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:   cat = "Underweight"
    elif bmi < 25.0: cat = "Normal weight"
    elif bmi < 30.0: cat = "Overweight"
    else:            cat = "Obese"
    return bmi, cat


def _compute_bmr(age: int, sex: str, height_cm: float, weight_kg: float) -> float:
    """
    Mifflin-St Jeor equation (1990) — most validated for general populations.

    Men:   BMR = 10W + 6.25H - 5A + 5
    Women: BMR = 10W + 6.25H - 5A - 161
    Other: average of both

    W = weight (kg), H = height (cm), A = age (years)
    """
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    if sex == "Male":   return base + 5
    if sex == "Female": return base - 161
    return base - 78   # Average for "Other"


def _infer_activity(work_type: str, training_days: int, stress_level: int) -> str:
    """
    Map work type + training frequency to an activity multiplier key.
    High stress slightly suppresses the multiplier (cortisol effect).
    """
    if work_type in ("active", "very_active") and training_days >= 5:
        return "very_active"
    if work_type in ("moderate", "active") or training_days >= 5:
        return "active"
    if work_type == "light" or training_days >= 3:
        return "moderate"
    if training_days >= 1:
        return "light"
    return "sedentary"


def _classify_budget(budget_usd: float) -> str:
    """Classify monthly food budget into low / mid / high tier."""
    for tier, (lo, hi) in BUDGET_TIERS.items():
        if lo <= budget_usd < hi:
            return tier
    return "mid"


def _build_risk_flags(conditions: list, injuries: list, medication: str) -> list[str]:
    """
    Return a list of plain-language flags that the LLM should respect
    when generating the exercise and diet plan.
    """
    flags = []
    cond_lower = [c.lower() for c in conditions]

    if "type 2 diabetes" in cond_lower:
        flags.append("Monitor blood sugar around training; avoid prolonged fasted cardio; limit high-GI carbs post-workout")
    if "hypertension" in cond_lower:
        flags.append("Avoid Valsalva manoeuvre; limit heavy isometric holds; include dedicated cardio; low sodium diet")
    if "high cholesterol" in cond_lower:
        flags.append("Favour unsaturated fats; include omega-3 rich foods; prioritise cardio in plan")
    if "pcos" in cond_lower:
        flags.append("Prioritise insulin sensitivity; include resistance training; moderate carbs; avoid extreme deficits")
    if "hypothyroidism" in cond_lower:
        flags.append("Avoid extreme caloric restriction; include iodine-rich foods; resistance training supports metabolism")
    if "heart disease" in cond_lower:
        flags.append("No high-intensity exercise without physician clearance; focus on low-moderate steady-state cardio")

    injury_lower = [i.lower() for i in injuries]
    if "lower back pain" in injury_lower:
        flags.append("Avoid heavy spinal loading (deadlifts, barbell squats); substitute with hip hinges, goblet squats, cable work")
    if "knee issues" in injury_lower:
        flags.append("Avoid deep knee flexion under load; substitute lunges/squats with leg press, wall sits, leg curls")
    if "shoulder impingement" in injury_lower:
        flags.append("Avoid overhead pressing; substitute with landmine press, incline DB press, cable flyes")

    if medication and "metabolism" in medication.lower():
        flags.append("Medication may affect metabolic rate; TDEE estimates may be less accurate — monitor progress and adjust")

    return flags


def _build_rag_queries(raw: RawProfile, computed: ComputedProfile, diet_restrictions: list) -> list[str]:
    """
    Pre-build the retrieval queries that Stage 4 will use to fetch paper chunks.
    Building them here (at profile time) means the LLM gets them ready-made.
    """
    goal    = raw.primary_goal
    exp     = raw.experience_level
    equip   = raw.equipment.replace("_", " ")
    veg_tag = "plant-based " if computed.is_vegetarian else ""
    budget  = computed.budget_tier

    return [
        f"{goal.replace('_', ' ')} training {exp} evidence-based",
        f"protein intake {goal.replace('_', ' ')} {exp} grams per kilogram",
        f"{veg_tag}high protein {budget} budget affordable foods",
        f"caloric {'deficit' if computed.calorie_delta < 0 else 'surplus'} {exp} body composition",
        f"exercise progression {equip} {exp} {goal.replace('_', ' ')}",
        f"sleep recovery muscle growth stress cortisol",
    ]


def _format_llm_context(raw: RawProfile, computed: ComputedProfile, diet_restrictions: list) -> str:
    """
    Format the user profile into a structured text block for injection
    into the LLM prompt in Stage 4.
    """
    flags_text = "\n".join(f"  - {f}" for f in computed.risk_flags) or "  - None"
    restrictions_text = ", ".join(diet_restrictions) if diet_restrictions else "None"
    conditions_text   = ", ".join(computed.condition_list) if computed.condition_list else "None"
    injuries_text     = ", ".join(computed.injury_list) if computed.injury_list else "None"

    return f"""
USER PROFILE SUMMARY
====================

BIOMETRICS
  Age: {raw.age} | Sex: {raw.sex} | Height: {raw.height_cm} cm | Weight: {raw.weight_kg} kg
  BMI: {computed.bmi} ({computed.bmi_category})

ENERGY TARGETS
  BMR: {computed.bmr} kcal/day
  TDEE: {computed.tdee} kcal/day
  Target calories: {computed.target_calories} kcal/day ({computed.calorie_direction})
  Protein target: {computed.protein_g}g/day (minimum)
  Carbohydrate target: {computed.carbs_g}g/day
  Fat target: {computed.fat_g}g/day

GOAL & TRAINING
  Primary goal: {computed.goal_label}
  Experience: {raw.experience_level}
  Training days/week: {raw.training_days_per_week}
  Equipment: {raw.equipment.replace('_', ' ')}
  Preferred duration: {raw.workout_duration or 'flexible'}
  Timeline: {raw.timeline_weeks} weeks

ECONOMICS
  Monthly food budget: {raw.currency} {raw.monthly_food_budget} (Budget tier: {computed.budget_tier})
  Supplement budget: {raw.supplement_budget or 'none'}

DIETARY
  Restrictions: {restrictions_text}
  Allergies/intolerances: {raw.allergies or 'none specified'}
  Meals per day: {raw.meals_per_day}
  Cooking time available: {raw.cooking_time_mins or 'not specified'}

HEALTH
  Medical conditions: {conditions_text}
  Injuries/pain areas: {injuries_text}
  Sleep: {raw.sleep_hours} hours/night
  Stress level: {raw.stress_level}/10
  Work activity: {raw.work_type}

CLINICAL FLAGS (must be respected in plan generation):
{flags_text}

ADDITIONAL NOTES FROM USER:
{raw.notes or 'None'}
""".strip()
