"""
generation/pipeline.py
Updated to handle InjectionDetectedError from parser.
"""

import time
import anthropic
import os

from generation.prompts import SYSTEM_PROMPT
from generation.prompt_assembler import assemble_user_message
from generation.llm_client import generate_plan
from generation.parser import parse_and_validate, enrich_citations, PlanValidationError, InjectionDetectedError
from app.services.retriever import retrieve_for_profile
from generation.storage import save_plan, get_latest_plan

MAX_RETRIES = 3


def generate_with_retry(profile: dict, chunks: list) -> dict:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[pipeline] Generation attempt {attempt}/{MAX_RETRIES}")

        try:
            raw_response = generate_plan(profile, chunks, error_hint=last_error)
            plan = parse_and_validate(raw_response, chunks)
            plan = enrich_citations(plan, chunks)
            print(f"[pipeline] Generation succeeded on attempt {attempt}")
            return plan

        except InjectionDetectedError as e:
            # Don't retry injection refusals — surface immediately
            print(f"[pipeline] Injection detected by LLM: {e}")
            raise Exception(f"Request refused by safety filter: {e}")

        except PlanValidationError as e:
            last_error = str(e)
            print(f"[pipeline] Attempt {attempt} failed validation: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(1.5)

        except anthropic.APIError as e:
            last_error = f"API error: {str(e)}"
            print(f"[pipeline] Attempt {attempt} API error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2)

    raise Exception(
        f"Plan generation failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def run_generation_pipeline(user_id: str, profile: dict) -> dict:
    print(f"[pipeline] Retrieving chunks for user {user_id}")

    retrieval_profile = {
        "goal":              profile.get("goal"),
        "experience":        profile.get("experience"),
        "budget_tier":       profile.get("budget_tier"),
        "equipment":         profile.get("equipment"),
        "conditions":        profile.get("health_conditions", []),
        "diet_restrictions": profile.get("dietary_restrictions", []),
    }

    chunks = retrieve_for_profile(retrieval_profile)

    if not chunks:
        raise Exception(
            "No research chunks found. "
            "Make sure papers have been ingested into the vector store."
        )

    print(f"[pipeline] Retrieved {len(chunks)} chunks")

    chunk_dicts = []
    for i, c in enumerate(chunks):
        chunk_dicts.append({
            "citation_id": f"REF_{i+1:02d}",
            "paper_title": c.paper_title,
            "domain":      ", ".join(c.domain_tags),
            "text":        c.chunk_text,
            "similarity":  c.similarity,
        })

    plan = generate_with_retry(profile, chunk_dicts)

    plan_id = save_plan(
        user_id=user_id,
        profile=profile,
        plan=plan,
        chunks=chunk_dicts
    )

    plan["plan_id"] = plan_id
    return plan


def run_adjustment_pipeline(user_id: str, plan_id: str, adjustment: str) -> dict:
    existing = get_latest_plan(user_id)
    if not existing:
        raise Exception(f"No plan found for user {user_id}")

    original_profile = existing["profile"]
    original_plan    = existing["plan"]

    adjusted_profile = {
        **original_profile,
        "_adjustment_request": adjustment,
        "_original_plan_summary": {
            "workout_days": len(
                original_plan.get("workout_plan", {})
                             .get("weekly_schedule", [])
            ),
            "daily_calories": original_plan.get("diet_plan", {})
                                           .get("daily_calories", 0),
            "goal": original_profile.get("goal"),
        }
    }

    print(f"[pipeline] Adjusting plan {plan_id}: '{adjustment}'")

    retrieval_profile = {
        "goal":              original_profile.get("goal"),
        "experience":        original_profile.get("experience"),
        "budget_tier":       original_profile.get("budget_tier"),
        "equipment":         original_profile.get("equipment"),
        "conditions":        original_profile.get("health_conditions", []),
        "diet_restrictions": original_profile.get("dietary_restrictions", []),
    }

    chunks = retrieve_for_profile(retrieval_profile)

    chunk_dicts = []
    for i, c in enumerate(chunks):
        chunk_dicts.append({
            "citation_id": f"REF_{i+1:02d}",
            "paper_title": c.paper_title,
            "domain":      ", ".join(c.domain_tags),
            "text":        c.chunk_text,
            "similarity":  c.similarity,
        })

    updated_plan = generate_with_retry(adjusted_profile, chunk_dicts)

    new_plan_id = save_plan(
        user_id=user_id,
        profile=original_profile,
        plan=updated_plan,
        chunks=chunk_dicts
    )

    updated_plan["plan_id"]       = new_plan_id
    updated_plan["adjusted_from"] = plan_id
    return updated_plan