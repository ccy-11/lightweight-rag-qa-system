# -*- coding: utf-8 -*-
"""PDF loading module using PyPDF2."""
import os
from dataclasses import dataclass
from typing import List
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError


@dataclass
class PageText:
    text: str
    page_num: int
    doc_name: str


def load_single_pdf(pdf_path: str) -> List[PageText]:
    """Load a single PDF and return per-page text blocks."""
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    results: List[PageText] = []
    try:
        reader = PdfReader(pdf_path)
        if reader.is_encrypted:
            print(f"WARNING: PDF encrypted, skipping: {pdf_path}")
            return results
        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            text = raw.strip()
            if text:
                results.append(PageText(text=text, page_num=page_num, doc_name=doc_name))
    except PdfReadError as exc:
        print(f"WARNING: failed to read PDF {pdf_path}: {exc}")
    except FileNotFoundError:
        raise
    return results


def load_all_pdfs(pdf_dir: str) -> List[PageText]:
    """Load all PDF files in a directory."""
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    all_pages: List[PageText] = []
    for filename in sorted(os.listdir(pdf_dir)):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            all_pages.extend(load_single_pdf(pdf_path))
    print(f"Loaded {len(all_pages)} non-empty pages from {pdf_dir}")
    return all_pages
