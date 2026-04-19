"""
ingest/chunker.py
=================
STEP 2 — Split the paper's raw text into overlapping chunks.

Why chunk?
  LLMs have context limits. We can't feed a 40-page paper into every query.
  Instead we split it into small, semantically meaningful pieces (~400 tokens),
  store each piece separately, and only retrieve the relevant ones at query time.

Why overlap?
  If a sentence is cut at the boundary between chunk 4 and chunk 5, overlap
  (50 tokens repeated) ensures the idea still appears fully in at least one chunk.

Output: list of Chunk objects, each with text + metadata.
"""

import logging
from dataclasses import dataclass
from typing import Generator

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants — tune these if output quality is poor
# ─────────────────────────────────────────────────────────────────────────────

# ~400 tokens × 4 chars/token ≈ 1600 chars per chunk
# This fits comfortably in the LLM context window even with many chunks
CHUNK_SIZE     = 1600   # characters
CHUNK_OVERLAP  = 200    # characters of overlap between adjacent chunks

# Minimum chunk size — discard chunks smaller than this (usually page remnants)
MIN_CHUNK_CHARS = 150


@dataclass
class Chunk:
    """A single text chunk from a paper, ready to be embedded."""
    chunk_text:  str
    chunk_index: int         # 0-based position within the paper
    token_count: int         # rough estimate (chars / 4)
    paper_title: str
    domain_tags: list[str]


def chunk_paper(raw_text: str, paper_title: str, domain_tags: list[str]) -> list[Chunk]:
    """
    Split a paper's full text into overlapping chunks.

    Args:
        raw_text:    Full extracted text from pdf_parser
        paper_title: Used to label each chunk for retrieval context
        domain_tags: Passed through to each chunk (for filtered retrieval)

    Returns:
        List of Chunk objects, in order
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Split on paragraph breaks first, then sentences, then words
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(raw_text)

    chunks = []
    for i, text in enumerate(raw_chunks):
        text = text.strip()

        # Skip tiny fragments (headers, footers that slipped through)
        if len(text) < MIN_CHUNK_CHARS:
            logger.debug(f"Skipping tiny chunk {i}: {repr(text[:50])}")
            continue

        # Prepend a context header so the LLM always knows the source,
        # even when a chunk is retrieved without surrounding context
        annotated_text = f"[Source: {paper_title}]\n\n{text}"

        chunks.append(Chunk(
            chunk_text=annotated_text,
            chunk_index=i,
            token_count=len(text) // 4,   # rough token estimate
            paper_title=paper_title,
            domain_tags=domain_tags,
        ))

    logger.info(f"Chunked '{paper_title}' into {len(chunks)} chunks "
                f"(avg {sum(c.token_count for c in chunks) // max(len(chunks),1)} tokens each)")

    return chunks


def chunk_preview(chunks: list[Chunk], n: int = 3) -> str:
    """
    Return a human-readable preview of the first n chunks.
    Useful for debugging — call this after chunking to sanity-check output.
    """
    lines = [f"Total chunks: {len(chunks)}\n"]
    for c in chunks[:n]:
        lines.append(f"--- Chunk {c.chunk_index} ({c.token_count} tokens) ---")
        lines.append(c.chunk_text[:300] + ("..." if len(c.chunk_text) > 300 else ""))
        lines.append("")
    return "\n".join(lines)
