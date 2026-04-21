import json
import re

class PlanValidationError(Exception):
    pass

def parse_and_validate(raw_text: str, source_chunks: list[dict]) -> dict:
    """
    Parse LLM output into a validated plan dict.
    Raises PlanValidationError if output is unusable.
    """

    # 1. Strip markdown fences if present (```json ... ```)
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

    # 2. Parse JSON
    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise PlanValidationError(f"LLM returned invalid JSON: {e}")

    # 3. Check required top-level keys
    required_keys = ["workout_plan", "diet_plan", "citations"]
    for key in required_keys:
        if key not in plan:
            raise PlanValidationError(f"Missing required key: {key}")

    # 4. Validate citations — every citation_id must match a real chunk
    valid_ids = {chunk["citation_id"] for chunk in source_chunks}
    for citation in plan.get("citations", []):
        cid = citation.get("citation_id", "")
        if cid not in valid_ids:
            raise PlanValidationError(
                f"Hallucinated citation: {cid} does not exist in source chunks"
            )

    # 5. Check citation coverage — every exercise must have a citation
    for day in plan["workout_plan"].get("weekly_schedule", []):
        for exercise in day.get("exercises", []):
            if not exercise.get("citation_id"):
                # Flag but don't fail — add a warning instead
                exercise["citation_warning"] = "No citation provided"

    # 6. Check calorie sanity (basic guard against nonsensical output)
    daily_cals = plan["diet_plan"].get("daily_calories", 0)
    if not (800 <= daily_cals <= 5000):
        raise PlanValidationError(
            f"Daily calories out of safe range: {daily_cals}"
        )

    return plan


# Attach full chunk text to citations for the frontend
def enrich_citations(plan: dict, chunks: list[dict]) -> dict:
    chunk_map = {c["citation_id"]: c for c in chunks}
    for citation in plan.get("citations", []):
        cid = citation["citation_id"]
        if cid in chunk_map:
            citation["chunk_text"] = chunk_map[cid]["text"]
            citation["paper_title"] = chunk_map[cid]["paper_title"]
    return plan