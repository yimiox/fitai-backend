"""
ingest/uploader.py
==================
STEP 4 — Store paper metadata and chunk vectors in Supabase.

Two tables are written to:
  papers        — one row per paper (title, authors, DOI, domain tags)
  paper_chunks  — one row per chunk (text + embedding vector)

Both are inserted in batches to avoid hitting Supabase's request size limits.
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client, Client

from ingest.pdf_parser import ParsedPaper
from ingest.chunker import Chunk

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_BATCH_SIZE = 50   # Insert this many rows per API call


def get_supabase_client() -> Client:
    """Create and return a Supabase client. Call once and reuse."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env\n"
            "Get these from: Supabase dashboard → Settings → API"
        )

    return create_client(url, key)


def upload_paper(
    parsed: ParsedPaper,
    chunks: list[Chunk],
    vectors: list[list[float]],
    authors: Optional[str] = None,
    supabase: Optional[Client] = None,
) -> str:
    """
    Insert a paper and all its chunks+embeddings into Supabase.

    Args:
        parsed:   ParsedPaper from pdf_parser
        chunks:   List of Chunk from chunker
        vectors:  List of embedding vectors, same length and order as chunks
        authors:  Optional author string (e.g. "Smith et al.")
        supabase: Optional existing client (creates one if not provided)

    Returns:
        The paper_id UUID assigned by Supabase
    """
    assert len(chunks) == len(vectors), \
        f"Mismatch: {len(chunks)} chunks but {len(vectors)} vectors"

    client = supabase or get_supabase_client()

    # ── 1. Insert the paper record ────────────────────────────────────────
    paper_row = {
        "title":       parsed.title,
        "authors":     authors or parsed.authors,
        "year":        parsed.year,
        "doi":         parsed.doi,
        "domain_tags": parsed.domain_tags,
        "file_name":   parsed.file_name,
    }

    logger.info(f"Inserting paper: '{parsed.title[:60]}'")
    result = client.table("papers").insert(paper_row).execute()

    if not result.data:
        raise RuntimeError(f"Failed to insert paper: {result}")

    paper_id = result.data[0]["id"]
    logger.info(f"Paper inserted with id={paper_id}")

    # ── 2. Insert chunks in batches ───────────────────────────────────────
    total_inserted = 0

    for batch_start in range(0, len(chunks), SUPABASE_BATCH_SIZE):
        batch_chunks  = chunks[batch_start : batch_start + SUPABASE_BATCH_SIZE]
        batch_vectors = vectors[batch_start : batch_start + SUPABASE_BATCH_SIZE]

        rows = []
        for chunk, vector in zip(batch_chunks, batch_vectors):
            rows.append({
                "paper_id":    paper_id,
                "paper_title": chunk.paper_title,
                "domain_tags": chunk.domain_tags,
                "chunk_index": chunk.chunk_index,
                "chunk_text":  chunk.chunk_text,
                "token_count": chunk.token_count,
                "embedding":   vector,      # Supabase pgvector accepts Python lists
            })

        batch_result = client.table("paper_chunks").insert(rows).execute()

        if not batch_result.data:
            raise RuntimeError(f"Failed to insert chunk batch starting at {batch_start}")

        total_inserted += len(rows)
        logger.info(f"Inserted chunks {batch_start}–{batch_start + len(rows) - 1} "
                    f"({total_inserted}/{len(chunks)} total)")

    logger.info(f"Upload complete: {total_inserted} chunks stored for paper '{parsed.title[:60]}'")
    return paper_id


def paper_exists(file_name: str, supabase: Optional[Client] = None) -> bool:
    """
    Check if a paper with this filename was already ingested.
    Prevents duplicate ingestion if you run the script twice.
    """
    client = supabase or get_supabase_client()
    result = (
        client.table("papers")
        .select("id")
        .eq("file_name", file_name)
        .execute()
    )
    return len(result.data) > 0


def list_papers(supabase: Optional[Client] = None) -> list[dict]:
    """Return all ingested papers as a list of dicts (for admin UI)."""
    client = supabase or get_supabase_client()
    result = (
        client.table("papers")
        .select("id, title, authors, year, domain_tags, file_name, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


def delete_paper(paper_id: str, supabase: Optional[Client] = None) -> None:
    """
    Delete a paper and all its chunks (cascades via FK constraint).
    Useful when you need to re-ingest a paper after fixing an error.
    """
    client = supabase or get_supabase_client()
    client.table("papers").delete().eq("id", paper_id).execute()
    logger.info(f"Deleted paper {paper_id} and all its chunks")
