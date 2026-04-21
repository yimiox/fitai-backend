def assemble_user_message(profile: dict, chunks: list[dict], error_hint: str = None) -> str:
    """
    Build the full user message sent to the LLM.
    chunks: list of {id, paper_title, domain, text}
    """

    # 1. Format the user profile section
    profile_section = f"""
=== USER PROFILE ===
Age: {profile['age']}
Sex: {profile['sex']}
Height: {profile['height_cm']} cm
Weight: {profile['weight_kg']} kg
BMI: {profile['bmi']:.1f}
TDEE: {profile['tdee']} kcal/day
Goal: {profile['goal']}
Experience level: {profile['experience']}
Budget tier: {profile['budget_tier']} ({profile['monthly_food_budget']})
Equipment: {profile['equipment']}
Health conditions: {', '.join(profile['health_conditions']) or 'None'}
Dietary restrictions: {', '.join(profile['dietary_restrictions']) or 'None'}
Meals per day: {profile['meals_per_day']}
Cooking time available: {profile['cooking_time_mins']} mins/meal
"""

    # 2. Format the retrieved research chunks
    chunks_section = "\n=== RESEARCH EVIDENCE ===\n"
    for i, chunk in enumerate(chunks):
        chunk_id = f"REF_{i+1:02d}"
        chunk['citation_id'] = chunk_id  # tag for LLM to cite
        chunks_section += f"""
[{chunk_id}] {chunk['paper_title']} ({chunk['domain']})
{chunk['text']}
---"""

    # 3. Specify the exact JSON output schema
    schema_section = """
=== REQUIRED OUTPUT (valid JSON only) ===
{
  "workout_plan": {
    "weekly_schedule": [
      {
        "day": "Monday",
        "session_type": "Upper body strength",
        "duration_mins": 45,
        "exercises": [
          {
            "name": "Barbell bench press",
            "sets": 3,
            "reps": "8-10",
            "rest_secs": 90,
            "notes": "Keep elbows at 45 degrees",
            "citation_id": "REF_01"
          }
        ]
      }
    ],
    "weekly_frequency": 3,
    "progressive_overload_note": "Increase weight by 2.5kg when all sets completed"
  },
  "diet_plan": {
    "daily_calories": 1800,
    "macros": {
      "protein_g": 150,
      "carbs_g": 180,
      "fat_g": 60
    },
    "meal_plan": [
      {
        "day": "Day 1",
        "meals": [
          {
            "meal_type": "Breakfast",
            "name": "Oats with eggs",
            "calories": 450,
            "protein_g": 30,
            "ingredients": ["80g oats", "3 eggs", "200ml milk"],
            "estimated_cost": "low",
            "prep_time_mins": 10,
            "citation_id": "REF_03"
          }
        ]
      }
    ],
    "shopping_list": ["Oats", "Eggs", "Chicken breast", "Lentils"]
  },
  "citations": [
    {
      "citation_id": "REF_01",
      "paper_title": "Effects of resistance training on body composition",
      "relevant_finding": "3 sets of 8-10 reps at 70-80% 1RM optimises hypertrophy in beginners"
    }
  ],
  "safety_disclaimer": "Consult your doctor before starting if you have any medical conditions.",
  "personalization_notes": "Plan adjusted for low budget using eggs and lentils as primary protein sources."
}
"""
# # If this is a retry, tell Claude what went wrong last time
#     if error_hint:
#         user_message += f"""

# === CORRECTION REQUIRED ===
# Your previous response had this problem: {error_hint}
# Fix it and respond with ONLY valid JSON matching the schema above.
# """

#     return user_message

    return profile_section + chunks_section + schema_section