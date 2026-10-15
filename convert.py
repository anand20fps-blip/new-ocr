#!/usr/bin/env python3
"""Extract text for content checks by rendering a PDF and running Windows Scan text.

Searchable and scanned PDFs take the same path: each page becomes a PNG, then
Windows.Media.Ocr reads the picture. That is the engine behind Photos > Scan text.
This does not build a document and does not use any text already stored in the PDF.

Usage:
    .\\.venv\\Scripts\\python.exe convert.py "D:\\path\\to\\file.pdf"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pymupdf

from scan_text import ensure_models, recognize, scan_engine

# Windows Scan text rejects bitmaps whose longest side is above this.
_MAX_SIDE = 10000


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a PDF to images and extract text with Windows Scan text."
    )
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: output/<pdf name>)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Render resolution (default: 300)",
    )
    return parser


def _page_dpi(page: pymupdf.Page, dpi: int) -> int:
    """Lower the render DPI when a page would exceed the Scan text size limit."""
    longest = max(page.rect.width, page.rect.height, 1) * dpi / 72
    if longest > _MAX_SIDE:
        return max(72, int(dpi * _MAX_SIDE / longest))
    return dpi


def render_pdf(pdf: Path, pages_dir: Path, dpi: int) -> list[Path]:
    """Write one PNG per page. The PDF text layer is ignored."""
    pages_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf)
    paths: list[Path] = []
    try:
        for index, page in enumerate(doc, start=1):
            out = pages_dir / f"page_{index:04d}.png"
            pixmap = page.get_pixmap(dpi=_page_dpi(page, dpi), alpha=False)
            pixmap.save(out)
            paths.append(out)
            print(f"image {index}/{doc.page_count}: {out.name}", flush=True)
    finally:
        doc.close()
    return paths


async def extract(pdf: Path, out_dir: Path, dpi: int) -> Path:
    pages_dir = out_dir / "pages"
    images = render_pdf(pdf, pages_dir, dpi)
    engine = await scan_engine()
    tessdata = ensure_models()
    parts: list[str] = []
    for index, image in enumerate(images, start=1):
        text = await recognize(image, engine, tessdata)
        page_txt = pages_dir / f"page_{index:04d}.txt"
        page_txt.write_text(text + "\n", encoding="utf-8")
        parts.append(f"----- Page {index} -----\n{text}")
        print(f"scan {index}/{len(images)}: {len(text)} characters", flush=True)
    document = out_dir / f"{pdf.stem}.txt"
    document.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return document


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    pdf = args.pdf
    if not pdf.is_file():
        print(f"PDF not found: {pdf}", file=sys.stderr)
        return 1
    if args.dpi < 72:
        print("DPI must be at least 72.", file=sys.stderr)
        return 1
    out_dir = args.output or (Path(__file__).resolve().parent / "output" / pdf.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    document = asyncio.run(extract(pdf, out_dir, args.dpi))
    print(f"text: {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
