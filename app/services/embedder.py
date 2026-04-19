"""
ingest/embedder.py
==================
STEP 3 — Convert each text chunk into a vector (embedding).

What is an embedding?
  A list of ~1536 numbers that captures the *meaning* of a piece of text.
  Two chunks that discuss similar ideas will have vectors that are close
  together in space (high cosine similarity). This is what lets us find
  relevant paper sections by meaning, not just keywords.

Two options:
  "openai"  — text-embedding-3-small from OpenAI
               Pros: state-of-the-art quality, easy API
               Cons: costs money (~$0.002 per 1M tokens — very cheap in practice)
               Dimension: 1536

  "local"   — sentence-transformers all-MiniLM-L6-v2 (runs on your machine)
               Pros: completely free, offline-capable
               Cons: slightly lower quality, slower on CPU
               Dimension: 384 (update your SQL to vector(384) if using this)

Set EMBEDDING_MODEL in .env to choose.
"""

import os
import logging
import time
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai")

# OpenAI batch limit — their API accepts up to 2048 inputs per request
OPENAI_BATCH_SIZE = 100


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_embedder() -> Callable[[list[str]], list[list[float]]]:
    """
    Returns the embedding function to use, based on EMBEDDING_MODEL env var.

    Usage:
        embed = get_embedder()
        vectors = embed(["some text", "another chunk"])
        # vectors[0] is a list of 1536 floats
    """
    if EMBEDDING_MODEL == "openai":
        return _openai_embedder()
    elif EMBEDDING_MODEL == "local":
        return _local_embedder()
    else:
        raise ValueError(f"Unknown EMBEDDING_MODEL: {EMBEDDING_MODEL}. Use 'openai' or 'local'.")


def embed_chunks(texts: list[str], progress_cb: Callable[[int, int], None] = None) -> list[list[float]]:
    """
    Embed a list of text strings. Handles batching automatically.

    Args:
        texts:       List of strings to embed (one per chunk)
        progress_cb: Optional callback(done, total) for progress reporting

    Returns:
        List of embedding vectors (one per input string), in order
    """
    embed_fn = get_embedder()
    all_vectors = []

    batch_size = OPENAI_BATCH_SIZE if EMBEDDING_MODEL == "openai" else 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(f"Embedding batch {i//batch_size + 1} / {(len(texts)-1)//batch_size + 1} "
                    f"({len(batch)} chunks)...")

        vectors = embed_fn(batch)
        all_vectors.extend(vectors)

        if progress_cb:
            progress_cb(min(i + batch_size, len(texts)), len(texts))

        # Polite rate-limiting between batches
        if EMBEDDING_MODEL == "openai" and i + batch_size < len(texts):
            time.sleep(0.3)

    return all_vectors


# ─────────────────────────────────────────────────────────────────────────────
# Embedding backends
# ─────────────────────────────────────────────────────────────────────────────

def _openai_embedder() -> Callable[[list[str]], list[list[float]]]:
    """Returns a function that calls OpenAI's embedding API."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in .env")

    client = OpenAI(api_key=api_key)

    def embed(texts: list[str]) -> list[list[float]]:
        # Clean: OpenAI rejects empty strings
        cleaned = [t.strip() or "empty" for t in texts]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=cleaned,
        )
        # Response comes back sorted by index
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    logger.info("Using OpenAI text-embedding-3-small (dim=1536)")
    return embed


def _local_embedder() -> Callable[[list[str]], list[list[float]]]:
    """
    Returns a function that uses a local sentence-transformer model.
    Downloads the model on first run (~90MB). Subsequent runs are instant.

    IMPORTANT: If you use this, change vector(1536) to vector(384) in setup_db.sql
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. Run:\n"
            "  pip install sentence-transformers\n"
            "Or switch to EMBEDDING_MODEL=openai in .env"
        )

    logger.info("Loading local sentence-transformer model (first run downloads ~90MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Local model loaded (dim=384). Remember to use vector(384) in Supabase.")

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    return embed
