"""
run_ingestion.py
================
Command-line script to ingest a single research paper PDF into the RAG pipeline.

Usage:
    python run_ingestion.py --file path/to/paper.pdf --domain "strength training" --domain "hypertrophy"
    python run_ingestion.py --file paper.pdf --domain "nutrition" --authors "Smith et al. 2023"
    python run_ingestion.py --list       # Show all ingested papers
    python run_ingestion.py --delete <paper_id>

This is an admin-only tool. Users never see or run this.
Run it once per paper you want to add to the knowledge base.
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingestion")


def ingest(file_path: str, domain_tags: list[str], authors: str = None, force: bool = False):
    """Full ingestion pipeline: parse → chunk → embed → upload."""
    from pathlib import Path
    from scripts.pdf_parser import parse_pdf
    from app.services.chunker import chunk_paper, chunk_preview
    from app.services.embedder import embed_chunks
    from app.services.uploader import upload_paper, paper_exists, get_supabase_client

    path = Path(file_path)
    client = get_supabase_client()

    # ── Guard: already ingested? ──────────────────────────────────────────
    if not force and paper_exists(path.name, client):
        logger.warning(f"'{path.name}' is already in the database. "
                       "Use --force to re-ingest (will create duplicates).")
        return

    print(f"\n{'='*60}")
    print(f"  Ingesting: {path.name}")
    print(f"  Domain tags: {domain_tags}")
    print(f"{'='*60}\n")

    # ── Step 1: Parse PDF ─────────────────────────────────────────────────
    print("STEP 1/4 — Parsing PDF...")
    parsed = parse_pdf(path, domain_tags=domain_tags)
    if authors:
        parsed.authors = authors
    print(f"  ✓ Extracted {len(parsed.raw_text):,} characters from {parsed.page_count} pages")
    print(f"  ✓ Title: {parsed.title[:70]}")
    print(f"  ✓ Year: {parsed.year or 'unknown'}  DOI: {parsed.doi or 'not found'}")

    # ── Step 2: Chunk ─────────────────────────────────────────────────────
    print("\nSTEP 2/4 — Chunking text...")
    chunks = chunk_paper(parsed.raw_text, parsed.title, domain_tags)
    print(f"  ✓ Created {len(chunks)} chunks")
    print("\n  Preview of first 2 chunks:")
    print(chunk_preview(chunks, n=2))

    # ── Step 3: Embed ─────────────────────────────────────────────────────
    print("STEP 3/4 — Generating embeddings...")

    def progress(done, total):
        pct = int(done / total * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}% ({done}/{total})", end="", flush=True)

    texts = [c.chunk_text for c in chunks]
    vectors = embed_chunks(texts, progress_cb=progress)
    print(f"\n  ✓ Generated {len(vectors)} vectors (dim={len(vectors[0])})")

    # ── Step 4: Upload ────────────────────────────────────────────────────
    print("\nSTEP 4/4 — Uploading to Supabase...")
    paper_id = upload_paper(parsed, chunks, vectors, authors=authors, supabase=client)
    print(f"  ✓ Paper stored with id: {paper_id}")
    print(f"  ✓ {len(chunks)} chunks stored in paper_chunks table")

    print(f"\n{'='*60}")
    print(f"  Done! '{parsed.title[:60]}' is now searchable.")
    print(f"{'='*60}\n")


def list_papers():
    """Print all ingested papers in a readable format."""
    from ingest.uploader import list_papers as _list, get_supabase_client
    papers = _list(get_supabase_client())
    if not papers:
        print("No papers ingested yet.")
        return
    print(f"\n{'─'*80}")
    print(f"  {'ID':<38} {'Year':<6} {'Title':<40}")
    print(f"{'─'*80}")
    for p in papers:
        title = (p.get("title") or "Unknown")[:38]
        year  = str(p.get("year") or "?")
        pid   = p["id"]
        tags  = ", ".join(p.get("domain_tags") or [])
        print(f"  {pid:<38} {year:<6} {title}")
        print(f"  {'':38} {'':6} Tags: {tags}")
    print(f"{'─'*80}")
    print(f"  Total: {len(papers)} papers\n")


def delete_paper(paper_id: str):
    """Delete a paper and all its chunks."""
    from ingest.uploader import delete_paper as _delete, get_supabase_client
    confirm = input(f"Delete paper {paper_id} and ALL its chunks? [y/N] ")
    if confirm.lower() == "y":
        _delete(paper_id, get_supabase_client())
        print(f"Deleted paper {paper_id}")
    else:
        print("Cancelled.")


def test_retrieval(query: str):
    """Quick test of the retrieval system after ingestion."""
    from retrieval.retriever import retrieve, format_chunks_for_prompt
    print(f"\nTesting retrieval for: '{query}'")
    chunks = retrieve(query, top_k=3)
    print(format_chunks_for_prompt(chunks))


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FitAI research paper ingestion tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ingestion.py --file papers/progressive_overload.pdf --domain "strength training" --domain "hypertrophy"
  python run_ingestion.py --file papers/protein_intake.pdf --domain "nutrition" --authors "Morton et al. 2018"
  python run_ingestion.py --list
  python run_ingestion.py --test "protein intake fat loss beginners"
  python run_ingestion.py --delete abc123-...
        """
    )

    parser.add_argument("--file",    help="Path to PDF file to ingest")
    parser.add_argument("--domain",  help="Domain tag (repeat for multiple)", action="append", dest="domains")
    parser.add_argument("--authors", help="Author string, e.g. 'Smith et al. 2023'")
    parser.add_argument("--force",   help="Re-ingest even if already exists", action="store_true")
    parser.add_argument("--list",    help="List all ingested papers", action="store_true")
    parser.add_argument("--delete",  help="Delete a paper by ID", metavar="PAPER_ID")
    parser.add_argument("--test",    help="Test retrieval with a query string", metavar="QUERY")

    args = parser.parse_args()

    if args.list:
        list_papers()
    elif args.delete:
        delete_paper(args.delete)
    elif args.test:
        test_retrieval(args.test)
    elif args.file:
        if not args.domains:
            print("ERROR: --domain is required. Example: --domain 'strength training'")
            sys.exit(1)
        ingest(args.file, args.domains, args.authors, args.force)
    else:
        parser.print_help()
