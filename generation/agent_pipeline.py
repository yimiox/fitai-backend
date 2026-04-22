"""
generation/agent_pipeline.py

Agentic plan generation pipeline.
Uses LangChain AgentExecutor with direct anthropic SDK —
avoids langchain-anthropic version conflicts entirely.
"""

import os
import json
import time
import anthropic

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from generation.tools import ALL_TOOLS
from generation.prompts import SYSTEM_PROMPT
from generation.prompt_assembler import assemble_user_message
from generation.parser import (
    parse_and_validate,
    enrich_citations,
    PlanValidationError,
    InjectionDetectedError,
)
from generation.storage import save_plan, get_latest_plan
from app.services.retriever import retrieve_for_profile

MAX_RETRIES = 3

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ─────────────────────────────────────────────
# AGENT DECISION LOGIC (no LangChain LLM needed)
# We run the agent decisions ourselves using plain
# anthropic SDK — more reliable, no version issues
# ─────────────────────────────────────────────

def run_agentic_retrieval(profile: dict) -> tuple[list[dict], list[str]]:
    """
    Intelligent retrieval that adapts based on the user profile.
    Makes decisions about which searches to run and whether to retry.

    Returns:
        chunk_dicts:   list of chunk dicts ready for prompt assembly
        reasoning_log: list of strings describing each decision made
    """
    from app.services.retriever import retrieve

    reasoning_log = []
    all_chunks = []
    seen_texts = set()

    def add_chunks(results, label):
        added = 0
        for r in results:
            key = r.chunk_text[:100]
            if key not in seen_texts:
                seen_texts.add(key)
                all_chunks.append(r)
                added += 1
        return added

    goal       = profile.get("goal", "general")
    experience = profile.get("experience", "beginner")
    equipment  = profile.get("equipment", "gym")
    budget     = profile.get("budget_tier", "mid")
    conditions = [c for c in profile.get("health_conditions", []) if c and c != "None"]

    # ── STEP 1: Exercise retrieval ──────────────────────────────
    goal_keywords = {
        "fat_loss":    "resistance training fat loss body composition",
        "muscle_gain": "hypertrophy progressive overload resistance training",
        "endurance":   "cardiovascular aerobic training VO2 max",
        "recomp":      "body recomposition resistance training deficit",
        "strength":    "maximal strength powerlifting periodisation",
        "general":     "general fitness exercise health benefits",
    }.get(goal, "exercise training")

    exercise_query = f"{goal_keywords} {experience} {equipment}"
    exercise_results = retrieve(query=exercise_query, top_k=5, min_similarity=0.25)
    added = add_chunks(exercise_results, "exercise")

    avg_sim = (sum(r.similarity for r in exercise_results) / len(exercise_results)
               if exercise_results else 0)

    reasoning_log.append(
        f"Tool: retrieve_exercise | Query: '{exercise_query}' | "
        f"Found: {len(exercise_results)} chunks | Avg similarity: {avg_sim:.0%}"
    )

    # ── STEP 2: Check similarity — retry if too low ─────────────
    if avg_sim < 0.35 and exercise_results:
        broader_query = f"{goal} {experience} training"
        retry_results = retrieve(query=broader_query, top_k=5, min_similarity=0.2)
        add_chunks(retry_results, "exercise-retry")
        reasoning_log.append(
            f"Decision: Low similarity ({avg_sim:.0%}) detected → "
            f"retried with broader query '{broader_query}' | "
            f"Found: {len(retry_results)} additional chunks"
        )
    elif avg_sim >= 0.35:
        reasoning_log.append(
            f"Decision: Similarity {avg_sim:.0%} acceptable → proceeding without retry"
        )
    else:
        reasoning_log.append(
            f"Decision: No exercise chunks found → will rely on nutrition evidence"
        )

    # ── STEP 3: Nutrition retrieval ─────────────────────────────
    budget_context = {
        "low":  "affordable high protein budget foods",
        "mid":  "moderate cost protein sources meal planning",
        "high": "optimal nutrition performance foods",
    }.get(budget, "nutrition meal planning")

    nutrition_query = f"protein intake macronutrients {goal} {budget_context}"
    nutrition_results = retrieve(
        query=nutrition_query, top_k=4, min_similarity=0.25,
        domain_filter=["nutrition", "diet", "protein"]
    )
    if not nutrition_results:
        nutrition_results = retrieve(query=nutrition_query, top_k=4, min_similarity=0.2)

    add_chunks(nutrition_results, "nutrition")
    reasoning_log.append(
        f"Tool: retrieve_nutrition | Query: '{nutrition_query}' | "
        f"Found: {len(nutrition_results)} chunks"
    )

    # ── STEP 4: Extra budget pass if low ────────────────────────
    if budget == "low":
        budget_query = "high protein low cost foods eggs lentils oats budget diet"
        budget_results = retrieve(query=budget_query, top_k=3, min_similarity=0.2)
        add_chunks(budget_results, "budget-nutrition")
        reasoning_log.append(
            f"Decision: Low budget detected → extra nutrition search for affordable foods | "
            f"Found: {len(budget_results)} additional chunks"
        )

    # ── STEP 5: Medical retrieval for health conditions ─────────
    for condition in conditions:
        medical_query = f"exercise diet recommendations {condition} safety guidelines"
        medical_results = retrieve(query=medical_query, top_k=3, min_similarity=0.2)
        add_chunks(medical_results, f"medical-{condition}")
        reasoning_log.append(
            f"Tool: retrieve_medical | Condition: '{condition}' | "
            f"Query: '{medical_query}' | Found: {len(medical_results)} chunks"
        )

    if not conditions:
        reasoning_log.append(
            "Decision: No health conditions reported → skipping medical retrieval"
        )

    # ── STEP 6: Ensure minimum chunk count ──────────────────────
    if len(all_chunks) < 4:
        fallback_results = retrieve_for_profile({
            "goal":              goal,
            "experience":        experience,
            "budget_tier":       budget,
            "equipment":         equipment,
            "conditions":        conditions,
            "diet_restrictions": profile.get("dietary_restrictions", []),
        })
        add_chunks(fallback_results, "fallback")
        reasoning_log.append(
            f"Decision: Only {len(all_chunks)} chunks found → "
            f"ran fallback profile retrieval, added {len(fallback_results)} more"
        )

    if not all_chunks:
        raise Exception("Agentic retrieval found no relevant research chunks.")

    # Sort by similarity, cap at 10
    all_chunks.sort(key=lambda c: c.similarity, reverse=True)
    final_chunks = all_chunks[:10]

    reasoning_log.append(
        f"Final: Selected top {len(final_chunks)} chunks by similarity "
        f"(range: {final_chunks[0].similarity:.0%}–{final_chunks[-1].similarity:.0%})"
    )

    # Convert to dicts with citation IDs
    chunk_dicts = []
    for i, c in enumerate(final_chunks):
        chunk_dicts.append({
            "citation_id": f"REF_{i+1:02d}",
            "paper_title": c.paper_title,
            "domain":      ", ".join(c.domain_tags),
            "text":        c.chunk_text,
            "similarity":  c.similarity,
        })

    print(f"[agent] Retrieval complete: {len(chunk_dicts)} chunks, {len(reasoning_log)} decisions")
    return chunk_dicts, reasoning_log


# ─────────────────────────────────────────────
# PLAN GENERATION WITH RETRY
# ─────────────────────────────────────────────

def generate_with_retry(profile: dict, chunks: list[dict]) -> dict:
    from generation.llm_client import generate_plan
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[agent_pipeline] Generation attempt {attempt}/{MAX_RETRIES}")
        try:
            raw_response = generate_plan(profile, chunks, error_hint=last_error)
            plan = parse_and_validate(raw_response, chunks)
            plan = enrich_citations(plan, chunks)
            print(f"[agent_pipeline] Succeeded on attempt {attempt}")
            return plan

        except InjectionDetectedError as e:
            raise Exception(f"Request refused by safety filter: {e}")

        except PlanValidationError as e:
            last_error = str(e)
            print(f"[agent_pipeline] Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(1.5)

        except anthropic.APIError as e:
            last_error = f"API error: {str(e)}"
            if attempt < MAX_RETRIES:
                time.sleep(2)

    raise Exception(
        f"Plan generation failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


# ─────────────────────────────────────────────
# MAIN AGENTIC PIPELINE: New plan
# ─────────────────────────────────────────────

def run_agentic_generation_pipeline(user_id: str, profile: dict) -> dict:
    print(f"[agent_pipeline] Starting agentic retrieval for user {user_id}")

    chunk_dicts, reasoning_log = run_agentic_retrieval(profile)
    plan = generate_with_retry(profile, chunk_dicts)

    plan["agent_reasoning"] = reasoning_log
    plan["retrieval_stats"] = {
        "chunks_used":    len(chunk_dicts),
        "tool_calls":     len(reasoning_log),
        "avg_similarity": round(
            sum(c["similarity"] for c in chunk_dicts) / len(chunk_dicts), 3
        ) if chunk_dicts else 0,
    }

    plan_id = save_plan(
        user_id=user_id,
        profile=profile,
        plan=plan,
        chunks=chunk_dicts,
    )

    plan["plan_id"] = plan_id
    print(f"[agent_pipeline] Plan saved: {plan_id}")
    return plan


# ─────────────────────────────────────────────
# AGENTIC ADJUSTMENT PIPELINE
# ─────────────────────────────────────────────

def run_agentic_adjustment_pipeline(
    user_id: str, plan_id: str, adjustment: str
) -> dict:
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
        }
    }

    print(f"[agent_pipeline] Agentic adjustment: '{adjustment}'")

    chunk_dicts, reasoning_log = run_agentic_retrieval(original_profile)
    updated_plan = generate_with_retry(adjusted_profile, chunk_dicts)

    updated_plan["agent_reasoning"] = reasoning_log
    updated_plan["retrieval_stats"] = {
        "chunks_used":    len(chunk_dicts),
        "tool_calls":     len(reasoning_log),
        "avg_similarity": round(
            sum(c["similarity"] for c in chunk_dicts) / len(chunk_dicts), 3
        ) if chunk_dicts else 0,
    }

    new_plan_id = save_plan(
        user_id=user_id,
        profile=original_profile,
        plan=updated_plan,
        chunks=chunk_dicts,
    )

    updated_plan["plan_id"]       = new_plan_id
    updated_plan["adjusted_from"] = plan_id
    return updated_plan