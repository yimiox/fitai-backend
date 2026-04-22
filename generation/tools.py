"""
generation/tools.py

LangChain Tools that wrap the FitAI retrieval system.
These are used by the AgentExecutor in agent_pipeline.py
to make decisions about how to search for research evidence.

Tools available to the agent:
  1. retrieve_research    — search pgvector for relevant paper chunks
  2. check_similarity     — get average similarity score of last results
  3. retrieve_medical     — targeted search for health condition papers
  4. retrieve_nutrition   — targeted search for nutrition/diet papers
  5. retrieve_exercise    — targeted search for exercise/training papers
"""

import os
from langchain.tools import tool
from app.services.retriever import retrieve, retrieve_for_profile

# Shared state — agent stores last retrieval results here
# so check_similarity_tool can inspect them
_last_results = []


@tool
def retrieve_research(query: str) -> str:
    """
    Search the FitAI research paper database for chunks relevant to
    a fitness or nutrition query.

    Use this when you need evidence to support a workout or diet recommendation.
    Returns the top matching paper chunks with their similarity scores.

    Args:
        query: Natural language search query, e.g.
               'progressive overload beginners home workout'
               'protein intake fat loss high budget'
    """
    global _last_results

    results = retrieve(
        query=query,
        top_k=6,
        min_similarity=0.25,
    )

    _last_results = results

    if not results:
        return "No relevant research found for this query. Try a broader search term."

    output = f"Found {len(results)} relevant chunks:\n\n"
    for i, r in enumerate(results, 1):
        output += (
            f"[{i}] {r.paper_title} (similarity: {r.similarity:.0%})\n"
            f"    {r.chunk_text[:200]}...\n\n"
        )
    return output


@tool
def check_similarity() -> str:
    """
    Check the average similarity score of the most recent retrieval results.

    Use this AFTER retrieve_research to decide if the results are good enough.
    If average similarity is below 0.35, the results may be too generic —
    consider calling retrieve_research again with a more specific query,
    or broadening the query if results were 0.

    Returns: average similarity score and a recommendation.
    """
    global _last_results

    if not _last_results:
        return "No retrieval results to check. Call retrieve_research first."

    scores = [r.similarity for r in _last_results]
    avg = sum(scores) / len(scores)
    best = max(scores)
    worst = min(scores)

    recommendation = ""
    if avg >= 0.5:
        recommendation = "Results are highly relevant. Proceed with plan generation."
    elif avg >= 0.35:
        recommendation = "Results are reasonably relevant. Proceed but consider one more search."
    else:
        recommendation = (
            "Results have low similarity. Try retrieve_research with a BROADER query "
            "or different keywords before generating the plan."
        )

    return (
        f"Retrieval quality report:\n"
        f"  Results: {len(_last_results)} chunks\n"
        f"  Average similarity: {avg:.0%}\n"
        f"  Best: {best:.0%} | Worst: {worst:.0%}\n"
        f"  Recommendation: {recommendation}"
    )


@tool
def retrieve_medical(condition: str) -> str:
    """
    Search specifically for research papers about exercise and diet
    recommendations for a medical condition.

    Use this when the user has a health condition (diabetes, hypertension,
    asthma, obesity, back pain, heart disease, etc.) to find
    condition-specific evidence before generating their plan.

    Args:
        condition: The medical condition, e.g. 'diabetes', 'hypertension',
                   'lower back pain', 'asthma'
    """
    query = f"exercise diet recommendations {condition} safety guidelines"

    results = retrieve(
        query=query,
        top_k=4,
        min_similarity=0.2,
        domain_filter=None,   # search all domains for medical queries
    )

    if not results:
        return (
            f"No specific research found for '{condition}'. "
            f"Include a general safety disclaimer in the plan."
        )

    output = f"Medical evidence for '{condition}' ({len(results)} chunks):\n\n"
    for i, r in enumerate(results, 1):
        output += (
            f"[{i}] {r.paper_title} (similarity: {r.similarity:.0%})\n"
            f"    {r.chunk_text[:250]}...\n\n"
        )
    return output


@tool
def retrieve_nutrition(goal: str, budget: str) -> str:
    """
    Search specifically for nutrition and diet research papers.

    Use this to get evidence for meal plan recommendations.

    Args:
        goal:   The user's fitness goal, e.g. 'fat_loss', 'muscle_gain'
        budget: The user's budget tier, e.g. 'low', 'mid', 'high'
    """
    budget_context = {
        "low":  "affordable high protein budget foods",
        "mid":  "moderate cost protein sources meal planning",
        "high": "optimal nutrition performance foods",
    }.get(budget, "nutrition meal planning")

    query = f"protein intake macronutrients {goal} {budget_context}"

    results = retrieve(
        query=query,
        top_k=4,
        min_similarity=0.25,
        domain_filter=["nutrition", "diet", "protein"],
    )

    if not results:
        # Fall back to broader search without domain filter
        results = retrieve(query=query, top_k=4, min_similarity=0.2)

    if not results:
        return "No nutrition research found. Use general dietary guidelines."

    output = f"Nutrition evidence ({len(results)} chunks):\n\n"
    for i, r in enumerate(results, 1):
        output += (
            f"[{i}] {r.paper_title} (similarity: {r.similarity:.0%})\n"
            f"    {r.chunk_text[:250]}...\n\n"
        )
    return output


@tool
def retrieve_exercise(goal: str, experience: str, equipment: str) -> str:
    """
    Search specifically for exercise and training research papers.

    Use this to get evidence for workout recommendations.

    Args:
        goal:       The user's goal, e.g. 'fat_loss', 'muscle_gain', 'endurance'
        experience: The user's experience level, e.g. 'beginner', 'intermediate', 'advanced'
        equipment:  Available equipment, e.g. 'home', 'gym', 'bodyweight'
    """
    goal_keywords = {
        "fat_loss":    "resistance training fat loss body composition",
        "muscle_gain": "hypertrophy progressive overload resistance training",
        "endurance":   "cardiovascular aerobic training VO2 max",
        "recomp":      "body recomposition resistance training deficit",
        "strength":    "maximal strength powerlifting periodisation",
        "general":     "general fitness exercise health benefits",
    }.get(goal, "exercise training")

    query = f"{goal_keywords} {experience} {equipment}"

    results = retrieve(
        query=query,
        top_k=5,
        min_similarity=0.25,
        domain_filter=["strength training", "exercise", "hypertrophy", "endurance"],
    )

    if not results:
        results = retrieve(query=query, top_k=5, min_similarity=0.2)

    if not results:
        return "No exercise research found. Use standard training guidelines."

    output = f"Exercise evidence ({len(results)} chunks):\n\n"
    for i, r in enumerate(results, 1):
        output += (
            f"[{i}] {r.paper_title} (similarity: {r.similarity:.0%})\n"
            f"    {r.chunk_text[:250]}...\n\n"
        )
    return output


# Export all tools for the agent
ALL_TOOLS = [
    retrieve_research,
    check_similarity,
    retrieve_medical,
    retrieve_nutrition,
    retrieve_exercise,
]