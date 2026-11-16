from pathlib import Path

from app.config import settings
from app.services.chunking import chunk_pages
from app.services.ingestion import parse_pdf_pages
from app.services.memory_index import load_docs_into_memory


def test_3m_2018_capex_page_present() -> None:
    pdf_path = Path(settings.financebench_root) / "pdfs" / "3M_2018_10K.pdf"
    assert pdf_path.exists()
    pages = parse_pdf_pages(pdf_path)
    assert len(pages) > 59
    page = pages[59]
    lowered = page.text.lower()
    assert "purchases of property, plant and equipment" in lowered or "cash flow" in lowered
    chunks = chunk_pages(pages)
    matched = [chunk for chunk in chunks if chunk.page_start <= 59 <= chunk.page_end]
    assert matched
    assert any("1,577" in chunk.content or "1577" in chunk.content for chunk in matched)


def test_memory_index_finds_capex_evidence() -> None:
    index = load_docs_into_memory(["3M_2018_10K"])
    hits = index.search("FY2018 capital expenditure cash flow purchases of property", company="3M", limit=5)
    assert hits
    assert any(hit.page_start <= 59 <= hit.page_end for hit in hits)
