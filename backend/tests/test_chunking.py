from app.services.chunking import PageText, chunk_pages, is_statement_or_table_page


def test_keeps_statement_pages_whole() -> None:
    pages = [
        PageText(0, "Introduction narrative " * 20),
        PageText(
            1,
            "Consolidated Statement of Cash Flows\nYears ended December 31\n(Millions)\n2018 1577 2017 1373",
        ),
        PageText(2, "Short note."),
    ]
    chunks = chunk_pages(pages, token_budget=80)
    statement = next(chunk for chunk in chunks if chunk.page_start == 1)
    assert statement.page_start == statement.page_end == 1
    assert is_statement_or_table_page(statement.content)
