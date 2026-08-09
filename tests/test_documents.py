"""Filing text extraction and chunking tests.

10-K layouts vary more than the SEC's item numbering suggests, and each case here comes
from a real filing that broke the extractor.
"""

from __future__ import annotations

from edgar_desk.edgar.chunking import chunk_text
from edgar_desk.edgar.documents import html_to_text, split_sections

BODY = 'Competition is intense and our margins may suffer. ' * 40


def _standard_10k() -> str:
    """The common layout: a table of contents, then Item headings above the content."""
    toc = '\n'.join(
        [
            'Table of Contents',
            'Item 1. Business',
            'Item 1A. Risk Factors',
            "Item 7. Management's Discussion and Analysis",
            'Item 8. Financial Statements',
        ]
    )
    return '\n\n'.join(
        [
            toc,
            'Item 1. Business',
            BODY,
            'Item 1A. Risk Factors',
            BODY,
            "Item 7. Management's Discussion and Analysis",
            BODY,
            'Item 8. Financial Statements',
            'See the consolidated statements.',
        ]
    )


def test_html_to_text_strips_markup_and_entities() -> None:
    html = '<div><p>Risk&nbsp;Factors</p><script>ignore()</script><b>AT&amp;T</b></div>'
    text = html_to_text(html)
    assert 'Risk Factors' in text
    assert 'AT&T' in text
    assert 'ignore()' not in text
    assert '<' not in text


def test_split_sections_skips_table_of_contents() -> None:
    """The contents lists every Item before the body does; picking the last occurrence
    of each heading lands on the real section."""
    sections = split_sections(_standard_10k())
    items = {s.item for s in sections}
    assert {'1', '1A', '7'} <= items
    risk = next(s for s in sections if s.item == '1A')
    assert risk.text.startswith('Item 1A. Risk Factors')
    assert 'Competition is intense' in risk.text
    # The section must stop at the next Item, not swallow the rest of the filing.
    assert 'See the consolidated statements.' not in risk.text


def test_split_sections_ignores_unwanted_items() -> None:
    sections = split_sections(_standard_10k())
    assert all(s.item in {'1', '1A', '7', '7A'} for s in sections)


def test_running_header_layout_is_recovered() -> None:
    """Intel's 10-K never writes "Item 1A." above the content: its item list points at
    page numbers, and the section is marked only by a repeated page header."""
    pages = '\n\n'.join(f'Risk Factors\n\n{BODY}\npage {n}' for n in range(1, 8))
    filing = '\n\n'.join(
        [
            'Table of Contents',
            'Risk Factors',
            '37',
            'Other Key Information 52',
            'x' * 40000,  # distance between the contents entry and the section itself
            pages,
        ]
    )
    sections = split_sections(filing)
    assert [s.item for s in sections] == ['1A']
    # Must begin at the section, not at the far-away contents entry.
    assert 'Other Key Information' not in sections[0].text
    assert 'Competition is intense' in sections[0].text


def test_single_mention_is_not_treated_as_a_section() -> None:
    """One passing reference to a title is a cross-reference, not a running header."""
    filing = f'Some preamble\n\nRisk Factors\n\n{BODY}'
    assert split_sections(filing) == []


def test_chunking_respects_target_size_and_overlaps() -> None:
    text = '\n\n'.join(f'Paragraph {i}. {"word " * 120}' for i in range(20))
    chunks = chunk_text(text, target_chars=2000, overlap_chars=200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 2600 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunking_splits_an_oversized_paragraph() -> None:
    """A single risk factor can exceed the target on its own and must still be split."""
    giant = ' '.join(f'Sentence number {i} about a specific risk.' for i in range(400))
    chunks = chunk_text(giant, target_chars=1500, overlap_chars=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 2000 for c in chunks)


def test_tiny_trailing_chunk_is_dropped() -> None:
    assert chunk_text('too short') == []
