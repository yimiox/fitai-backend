"""
ingest/pdf_parser.py
====================
STEP 1 — Extract clean text from a research paper PDF.

We try pdfplumber first (better for text-heavy papers with columns).
If it fails or returns empty, we fall back to PyMuPDF (fitz).

Output: a single string of clean text, plus basic metadata extracted
from the filename and first page.
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pdfplumber
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class ParsedPaper:
    """Holds everything extracted from a single PDF."""
    raw_text: str                     # Full text of the paper
    title: str                        # Best-guess title
    file_name: str                    # Original filename
    page_count: int                   # Number of pages
    domain_tags: list[str] = field(default_factory=list)  # Set by caller
    authors: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None


def parse_pdf(file_path: str | Path, domain_tags: list[str] = None) -> ParsedPaper:
    """
    Main entry point. Takes a path to a PDF and returns a ParsedPaper.

    Args:
        file_path:   Path to the .pdf file
        domain_tags: e.g. ['strength training', 'hypertrophy']

    Returns:
        ParsedPaper with extracted text and metadata
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    logger.info(f"Parsing PDF: {path.name}")

    # Try pdfplumber first
    text, page_count = _extract_with_pdfplumber(path)

    # Fall back to PyMuPDF if pdfplumber returns too little
    if len(text.strip()) < 500:
        logger.warning("pdfplumber returned sparse text — trying PyMuPDF")
        text, page_count = _extract_with_pymupdf(path)

    if len(text.strip()) < 100:
        raise ValueError(f"Could not extract meaningful text from {path.name}. "
                         "The PDF may be scanned/image-only. Use an OCR tool first.")

    # Clean the extracted text
    text = _clean_text(text)

    # Extract metadata from the first ~2000 chars
    title = _guess_title(text, path.stem)
    doi   = _extract_doi(text)
    year  = _extract_year(text)

    logger.info(f"Extracted {len(text)} chars from {page_count} pages. "
                f"Title guess: '{title[:60]}...'")

    return ParsedPaper(
        raw_text=text,
        title=title,
        file_name=path.name,
        page_count=page_count,
        domain_tags=domain_tags or [],
        doi=doi,
        year=year,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_with_pdfplumber(path: Path) -> tuple[str, int]:
    """Extract text page by page using pdfplumber."""
    pages_text = []
    page_count = 0
    try:
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=2, y_tolerance=3)
                if t:
                    pages_text.append(t)
    except Exception as e:
        logger.warning(f"pdfplumber error: {e}")

    return "\n\n".join(pages_text), page_count


def _extract_with_pymupdf(path: Path) -> tuple[str, int]:
    """Extract text using PyMuPDF as fallback."""
    pages_text = []
    page_count = 0
    try:
        doc = fitz.open(str(path))
        page_count = len(doc)
        for page in doc:
            t = page.get_text("text")
            if t:
                pages_text.append(t)
        doc.close()
    except Exception as e:
        logger.error(f"PyMuPDF error: {e}")

    return "\n\n".join(pages_text), page_count


def _clean_text(text: str) -> str:
    """
    Remove common PDF artifacts that hurt chunking quality:
    - Page numbers standing alone on a line
    - Running headers/footers (often repeat every page)
    - Excessive whitespace
    - Hyphenated line breaks (reattach words split across lines)
    """
    # Reattach hyphenated line breaks: "pro-\ntein" → "protein"
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Remove lines that are just a page number (e.g. "— 12 —" or just "12")
    text = re.sub(r"^\s*[—–-]?\s*\d{1,4}\s*[—–-]?\s*$", "", text, flags=re.MULTILINE)

    # Collapse 3+ newlines to 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove non-printable characters except newlines and tabs
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Collapse multiple spaces to one
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def _guess_title(text: str, fallback: str) -> str:
    """
    Tries to extract the paper title from the first few lines.
    Research papers usually start with a large-font title on page 1,
    which pdfplumber renders as the first non-empty line(s).
    """
    lines = [l.strip() for l in text[:2000].splitlines() if len(l.strip()) > 10]
    if lines:
        # The title is often the first long line (>20 chars) before "Abstract"
        for line in lines[:8]:
            if len(line) > 20 and not line.lower().startswith(("abstract", "introduction", "doi", "http")):
                return line
    return fallback.replace("_", " ").replace("-", " ").title()


def _extract_doi(text: str) -> Optional[str]:
    """Extract DOI if present anywhere in the text."""
    match = re.search(r"\b(10\.\d{4,9}/[^\s]+)", text)
    return match.group(1) if match else None


def _extract_year(text: str) -> Optional[int]:
    """Extract the most likely publication year (4-digit, 1990–2030)."""
    matches = re.findall(r"\b(19[9]\d|20[0-2]\d)\b", text[:3000])
    if matches:
        # Return the most frequently appearing year in the header region
        from collections import Counter
        return int(Counter(matches).most_common(1)[0][0])
    return None
