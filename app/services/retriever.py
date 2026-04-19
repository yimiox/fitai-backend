"""
retrieval/retriever.py
======================
STEP 5 — Given a natural language query, find the most relevant paper chunks.

This is called at query time (when a user submits their questionnaire).
It does NOT re-read any PDFs — it just searches the vectors already stored.

Flow:
  1. Convert the query string into a vector (using same embedding model as ingestion)
  2. Run cosine similarity search against paper_chunks in Supabase
  3. Return the top-k most relevant chunks with their paper metadata

The results are then injected into the LLM prompt in Stage 4.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from supabase import Client

from ingest.embedder import embed_chunks
from ingest.uploader import get_supabase_client

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned from similarity search, with its similarity score."""
    paper_title: str
    domain_tags: list[str]
    chunk_text:  str
    similarity:  float        # 0.0 = unrelated, 1.0 = identical meaning


def retrieve(
    query: str,
    top_k: int = 6,
    domain_filter: Optional[list[str]] = None,
    min_similarity: float = 0.3,
    supabase: Optional[Client] = None,
) -> list[RetrievedChunk]:
    """
    Find the most relevant paper chunks for a given query.

    Args:
        query:          Natural language query, e.g. "protein intake for fat loss beginners"
        top_k:          How many chunks to return (6–8 is a good default)
        domain_filter:  Optional list of domain tags to restrict search,
                        e.g. ["nutrition"] to only search nutrition papers
        min_similarity: Discard chunks below this similarity threshold
        supabase:       Optional existing client

    Returns:
        List of RetrievedChunk, sorted by similarity descending

    Example:
        chunks = retrieve("progressive overload beginners", top_k=5)
        for c in chunks:
            print(f"[{c.similarity:.2f}] {c.paper_title}: {c.chunk_text[:200]}")
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")

    client = supabase or get_supabase_client()

    # Step 1: Embed the query using the same model used during ingestion
    logger.info(f"Embedding query: '{query[:80]}'")
    [query_vector] = embed_chunks([query])

    # Step 2: Call the match_chunks SQL function we defined in setup_db.sql
    logger.info(f"Searching for top {top_k} chunks "
                f"{'(domain: ' + str(domain_filter) + ')' if domain_filter else '(all domains)'}")

    params = {
        "query_embedding": query_vector,
        "match_count":     top_k,
        "domain_filter":   domain_filter,
    }

    result = client.rpc("match_chunks", params).execute()

    if result.data is None:
        logger.warning("No results from similarity search")
        return []

    # Step 3: Filter by minimum similarity and wrap in dataclass
    chunks = []
    for row in result.data:
        sim = float(row["similarity"])
        if sim < min_similarity:
            logger.debug(f"Skipping low-similarity chunk ({sim:.3f}): {row['paper_title'][:40]}")
            continue
        chunks.append(RetrievedChunk(
            paper_title=row["paper_title"],
            domain_tags=row.get("domain_tags") or [],
            chunk_text=row["chunk_text"],
            similarity=sim,
        ))

    logger.info(f"Retrieved {len(chunks)} relevant chunks "
                f"(similarity range: {chunks[0].similarity:.2f}–{chunks[-1].similarity:.2f})"
                if chunks else "Retrieved 0 chunks above similarity threshold")

    return chunks


def retrieve_for_profile(user_profile: dict, supabase: Optional[Client] = None) -> list[RetrievedChunk]:
    """
    Build multiple targeted queries from a user profile and retrieve chunks for all.
    Deduplicates results so the same chunk isn't returned twice.

    This is the main function called from the plan generation endpoint.

    Args:
        user_profile: The structured profile from the questionnaire, e.g.:
            {
                "goal":           "fat_loss",
                "experience":     "beginner",
                "budget_tier":    "low",
                "equipment":      "home",
                "diet_restrictions": ["vegetarian"],
                "conditions":     []
            }

    Returns:
        Deduplicated list of RetrievedChunk, sorted by similarity
    """
    queries = _build_queries_from_profile(user_profile)
    logger.info(f"Running {len(queries)} retrieval queries for user profile")

    seen_texts = set()
    all_chunks = []

    for query, domain_filter in queries:
        results = retrieve(
            query=query,
            top_k=4,
            domain_filter=domain_filter,
            min_similarity=0.25,
            supabase=supabase,
        )
        for chunk in results:
            # Deduplicate by first 100 chars of chunk text
            key = chunk.chunk_text[:100]
            if key not in seen_texts:
                seen_texts.add(key)
                all_chunks.append(chunk)

    # Sort by similarity, return top 10 overall
    all_chunks.sort(key=lambda c: c.similarity, reverse=True)
    final = all_chunks[:10]

    logger.info(f"Final retrieval: {len(final)} unique chunks across {len(queries)} queries")
    return final


def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into the text block that gets injected into the LLM prompt.
    Each chunk is numbered and labeled with its source paper.

    Output example:
        [1] Source: "Effects of Progressive Overload on Muscle Hypertrophy"
            "Progressive overload, defined as systematically increasing training volume..."

        [2] Source: "Dietary Protein and Body Composition"
            "Protein intakes of 1.6–2.2g/kg bodyweight were associated with..."
    """
    if not chunks:
        return "No relevant research chunks were found for this query."

    lines = ["RELEVANT RESEARCH EVIDENCE:\n"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f'[{i}] Source: "{chunk.paper_title}" (relevance: {chunk.similarity:.0%})')
        # Strip the [Source: ...] header we prepended during ingestion to avoid duplication
        text = chunk.chunk_text
        if text.startswith("[Source:"):
            text = "\n".join(text.split("\n")[2:]).strip()
        lines.append(f'    "{text[:600]}{"..." if len(text) > 600 else ""}"')
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Private: query builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_queries_from_profile(profile: dict) -> list[tuple[str, Optional[list[str]]]]:
    """
    Convert a user profile into a list of (query_string, domain_filter) tuples.
    More targeted queries = better retrieval = better plans.

    Returns list of (query, domain_filter) tuples.
    """
    goal        = profile.get("goal", "general health")
    experience  = profile.get("experience", "beginner")
    budget      = profile.get("budget_tier", "mid")
    equipment   = profile.get("equipment", "gym")
    conditions  = profile.get("conditions", [])
    restrictions = profile.get("diet_restrictions", [])

    goal_map = {
        "fat_loss":    "caloric deficit fat loss body composition",
        "muscle_gain": "hypertrophy muscle growth resistance training",
        "endurance":   "cardiovascular endurance aerobic training",
        "general":     "general health physical activity guidelines",
    }

    workout_query = f"{goal_map.get(goal, goal)} {experience} {equipment}"
    nutrition_query = f"protein intake macronutrients {goal} {experience}"
    budget_query = f"affordable high protein foods {'plant-based ' if 'vegetarian' in restrictions or 'vegan' in restrictions else ''}{budget} budget"

    queries = [
        (workout_query,   ["strength training", "hypertrophy", "endurance", "exercise"]),
        (nutrition_query, ["nutrition", "diet", "protein"]),
        (budget_query,    ["nutrition", "diet"]),
    ]

    # Add condition-specific queries if relevant
    for condition in conditions:
        if condition in ("diabetes", "hypertension", "obesity"):
            queries.append((
                f"exercise diet recommendations {condition}",
                None,  # Search all domains
            ))

    return queries
