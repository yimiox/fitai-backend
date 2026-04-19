"""
main.py + api/routes.py combined
=================================
FastAPI endpoints for the RAG pipeline.

Endpoints:
  POST /ingest         — Admin: upload and ingest a PDF
  GET  /papers         — Admin: list all ingested papers
  DELETE /papers/{id}  — Admin: delete a paper
  POST /retrieve       — Internal: retrieve chunks for a query (used by plan generator)
  GET  /health         — Health check
"""

import logging
import os
from pathlib import Path
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from scripts.pdf_parser import parse_pdf
from app.services.chunker import chunk_paper
from app.services.embedder import embed_chunks
from app.services.uploader import upload_paper, paper_exists, list_papers, delete_paper, get_supabase_client
from app.services.retriever import retrieve, retrieve_for_profile, format_chunks_for_prompt

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FitAI RAG API",
    description="Research paper ingestion and retrieval for FitAI",
    version="1.0.0",
)

# Allow requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://your-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 6
    domain_filter: Optional[list[str]] = None
    min_similarity: float = 0.3

class ProfileRetrieveRequest(BaseModel):
    """Retrieve chunks based on a full user profile (used in plan generation)."""
    goal: str               # "fat_loss" | "muscle_gain" | "endurance" | "general"
    experience: str         # "beginner" | "intermediate" | "advanced"
    budget_tier: str        # "low" | "mid" | "high"
    equipment: str          # "home" | "gym"
    conditions: list[str] = []
    diet_restrictions: list[str] = []

class ChunkResponse(BaseModel):
    paper_title: str
    domain_tags: list[str]
    chunk_text: str
    similarity: float


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "fitai-rag"}


@app.post("/ingest", summary="Admin: Upload and ingest a research paper PDF")
async def ingest_paper(
    file: UploadFile = File(..., description="The PDF file to ingest"),
    domain_tags: str = Form(..., description="Comma-separated domain tags, e.g. 'strength training,hypertrophy'"),
    authors: Optional[str] = Form(None, description="Author string, e.g. 'Smith et al. 2023'"),
):
    """
    Upload a research paper PDF and run the full ingestion pipeline.
    
    This endpoint:
    1. Saves the PDF temporarily
    2. Extracts and cleans text
    3. Splits into chunks
    4. Generates embeddings
    5. Stores everything in Supabase
    
    Takes 10–60 seconds depending on paper length.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    tags = [t.strip() for t in domain_tags.split(",") if t.strip()]
    if not tags:
        raise HTTPException(status_code=400, detail="At least one domain tag is required")

    supabase = get_supabase_client()

    # Check for duplicates
    if paper_exists(file.filename, supabase):
        raise HTTPException(
            status_code=409,
            detail=f"Paper '{file.filename}' already ingested. Delete it first to re-ingest."
        )

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run the pipeline
        parsed  = parse_pdf(tmp_path, domain_tags=tags)
        parsed.file_name = file.filename   # Use original filename
        if authors:
            parsed.authors = authors

        chunks  = chunk_paper(parsed.raw_text, parsed.title, tags)
        texts   = [c.chunk_text for c in chunks]
        vectors = embed_chunks(texts)

        paper_id = upload_paper(parsed, chunks, vectors, supabase=supabase)

        return {
            "success":    True,
            "paper_id":   paper_id,
            "title":      parsed.title,
            "chunk_count": len(chunks),
            "page_count": parsed.page_count,
            "year":       parsed.year,
            "doi":        parsed.doi,
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/papers", summary="Admin: List all ingested papers")
def get_papers():
    """Returns all research papers in the knowledge base."""
    return list_papers()


@app.delete("/papers/{paper_id}", summary="Admin: Delete a paper and all its chunks")
def remove_paper(paper_id: str):
    """
    Permanently deletes a paper and all its chunks from Supabase.
    Use this if you need to re-ingest a paper after fixing errors.
    """
    try:
        delete_paper(paper_id)
        return {"success": True, "deleted": paper_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=list[ChunkResponse], summary="Retrieve relevant chunks for a query")
def retrieve_chunks(req: RetrieveRequest):
    """
    Find the most relevant paper chunks for a free-text query.
    Used internally by the plan generation service.
    
    Example query: "progressive overload beginners home workout"
    """
    try:
        chunks = retrieve(
            query=req.query,
            top_k=req.top_k,
            domain_filter=req.domain_filter,
            min_similarity=req.min_similarity,
        )
        return [
            ChunkResponse(
                paper_title=c.paper_title,
                domain_tags=c.domain_tags,
                chunk_text=c.chunk_text,
                similarity=c.similarity,
            )
            for c in chunks
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve/profile", summary="Retrieve chunks for a full user profile")
def retrieve_for_user_profile(profile: ProfileRetrieveRequest):
    """
    Build multi-query retrieval from a user profile.
    Returns deduplicated chunks formatted for LLM injection.
    
    This is called by the plan generation endpoint before hitting the LLM.
    """
    try:
        chunks = retrieve_for_profile(profile.model_dump())
        formatted = format_chunks_for_prompt(chunks)
        return {
            "chunk_count": len(chunks),
            "formatted_context": formatted,
            "chunks": [
                {
                    "paper_title": c.paper_title,
                    "chunk_text":  c.chunk_text,
                    "similarity":  round(c.similarity, 3),
                }
                for c in chunks
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
