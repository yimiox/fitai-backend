"""
generation/parser.py

Parses and validates raw LLM output.
Detects:
  - Invalid JSON
  - Truncated responses
  - Missing required fields
  - Hallucinated citation IDs
  - Injection error responses from the LLM
  - Unsafe calorie values
"""

import json
import re


class PlanValidationError(Exception):
    pass


class InjectionDetectedError(Exception):
    """Raised when the LLM itself detected and refused an injection attempt."""
    pass


def parse_and_validate(raw_text: str, source_chunks: list[dict]) -> dict:
    """
    Parse LLM output into a validated plan dict.
    Raises PlanValidationError if output is unusable.
    Raises InjectionDetectedError if LLM flagged an injection attempt.
    """

    # 1. Strip markdown fences if present
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    cleaned = cleaned.strip()

    # 2. Parse JSON
    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Check for obvious truncation
        open_braces  = cleaned.count("{")
        close_braces = cleaned.count("}")
        if open_braces != close_braces:
            raise PlanValidationError(
                f"Response truncated — mismatched braces "
                f"({open_braces} open, {close_braces} close). "
                f"Increase max_tokens in llm_client.py."
            )
        raise PlanValidationError(f"LLM returned invalid JSON: {e}")

    # ── 3. INJECTION DETECTION ──────────────────────────────────────
    # Check if the LLM returned an injection refusal instead of a plan
    if "error_code" in plan and plan.get("error_code") == "INJECTION_DETECTED":
        raise InjectionDetectedError(
            plan.get("error", "Injection attempt detected and refused.")
        )

    # Also catch any generic error response
    if "error" in plan and len(plan) <= 2:
        raise InjectionDetectedError(
            plan.get("error", "Request refused by safety filter.")
        )
    # ───────────────────────────────────────────────────────────────

    # 4. Check required top-level keys
    required_keys = ["workout_plan", "diet_plan", "citations"]
    for key in required_keys:
        if key not in plan:
            raise PlanValidationError(f"Missing required key: {key}")

    # 5. Validate citations — every citation_id must match a real chunk
    valid_ids = {chunk["citation_id"] for chunk in source_chunks}
    for citation in plan.get("citations", []):
        cid = citation.get("citation_id", "")
        if cid not in valid_ids:
            raise PlanValidationError(
                f"Hallucinated citation: '{cid}' does not exist in source chunks. "
                f"Valid IDs: {sorted(valid_ids)}"
            )

    # 6. Check citation coverage on exercises — flag missing ones
    for day in plan["workout_plan"].get("weekly_schedule", []):
        for exercise in day.get("exercises", []):
            if not exercise.get("citation_id"):
                exercise["citation_warning"] = "No citation provided"

    # 7. Calorie sanity check
    daily_cals = plan["diet_plan"].get("daily_calories", 0)
    if daily_cals and not (800 <= daily_cals <= 6000):
        raise PlanValidationError(
            f"Daily calories out of safe range: {daily_cals} "
            f"(expected 800–6000 kcal)"
        )

    return plan


def enrich_citations(plan: dict, chunks: list[dict]) -> dict:
    """
    Attach full chunk text and paper metadata to each citation
    so the frontend can display the supporting evidence.
    """
    chunk_map = {c["citation_id"]: c for c in chunks}
    for citation in plan.get("citations", []):
        cid = citation.get("citation_id")
        if cid and cid in chunk_map:
            citation["chunk_text"]   = chunk_map[cid]["text"]
            citation["paper_title"]  = chunk_map[cid]["paper_title"]
            citation["domain"]       = chunk_map[cid].get("domain", "")
            citation["similarity"]   = round(chunk_map[cid].get("similarity", 0), 3)
    return plan