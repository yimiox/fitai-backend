def build_retrieval_queries(profile: dict) -> list[str]:
    """
    Convert a user profile into targeted search queries
    for the pgvector paper database.
    """
    queries = []
    goal = profile["goal"]          # e.g. "fat_loss"
    level = profile["experience"]   # e.g. "beginner"
    budget = profile["budget_tier"] # e.g. "low"
    conditions = profile.get("health_conditions", [])

    # --- Workout queries ---
    goal_map = {
        "fat_loss":     "resistance training fat loss",
        "muscle_gain":  "progressive overload hypertrophy",
        "endurance":    "aerobic training cardiovascular fitness",
        "general":      "general fitness exercise health benefits",
    }
    base_workout = goal_map.get(goal, "exercise benefits")
    queries.append(f"{level} {base_workout}")

    # Add condition-specific query if needed
    if "lower_back_pain" in conditions:
        queries.append("exercise lower back pain rehabilitation")
    if "diabetes" in conditions:
        queries.append("resistance training blood sugar insulin sensitivity")

    # --- Nutrition queries ---
    nutrition_map = {
        "fat_loss":    "caloric deficit protein intake weight loss",
        "muscle_gain": "protein synthesis muscle growth diet",
        "endurance":   "carbohydrate intake endurance performance",
        "general":     "balanced diet micronutrients health",
    }
    queries.append(nutrition_map.get(goal, "healthy diet nutrition"))

    # Budget-adjusted nutrition query
    if budget == "low":
        queries.append("high protein budget foods affordable nutrition")
    elif budget == "mid":
        queries.append("protein sources moderate cost meal planning")

    return queries  # Returns 2–4 queries


# Example usage:
profile = {
    "goal": "fat_loss",
    "experience": "beginner",
    "budget_tier": "low",
    "health_conditions": [],
}
queries = build_retrieval_queries(profile)
# Returns: [
#   "beginner resistance training fat loss",
#   "caloric deficit protein intake weight loss",
#   "high protein budget foods affordable nutrition"
# ]