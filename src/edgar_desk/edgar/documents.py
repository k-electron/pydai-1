"""Extract narrative sections from 10-K filings.

10-Ks are filed as HTML with no semantic markup for their structure: the only reliable
signal that "Item 1A. Risk Factors" has begun is the text of the heading itself. So the
approach is to strip to plain text first, then split on Item headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Sections worth retrieving. Item 1A and Item 7 carry nearly all of the analysis-relevant
# prose; the rest of a 10-K is financial statements already captured as XBRL facts.
WANTED_ITEMS: dict[str, str] = {
    '1': 'Item 1. Business',
    '1A': 'Item 1A. Risk Factors',
    '7': "Item 7. Management's Discussion and Analysis",
    '7A': 'Item 7A. Quantitative and Qualitative Disclosures About Market Risk',
}

_SCRIPT_STYLE = re.compile(r'<(script|style)\b[^>]*>.*?</\1>', re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r'<[^>]+>')
_WS = re.compile(r'[ \t\r\f\v]+')
_BLANKS = re.compile(r'\n\s*\n\s*\n+')

# Matches "Item 1A." / "ITEM 7A" / "Item&#160;1A." at a line start, tolerating the
# non-breaking spaces and stray punctuation that survive HTML stripping.
_ITEM_HEADING = re.compile(
    r'^[\s\u00a0]*item[\s\u00a0]*([0-9]{1,2}[A-Za-z]?)[\s\u00a0]*[.\-:\u2014]?[\s\u00a0]*(.{0,90})$',
    re.IGNORECASE | re.MULTILINE,
)

_ENTITIES = {
    '&nbsp;': ' ',
    '&#160;': ' ',
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#39;': "'",
    '&apos;': "'",
    '&#8217;': '\u2019',
    '&#8216;': '\u2018',
    '&#8220;': '\u201c',
    '&#8221;': '\u201d',
    '&#8212;': '\u2014',
    '&#8211;': '\u2013',
}


@dataclass(frozen=True, slots=True)
class Section:
    item: str
    title: str
    text: str


def html_to_text(html: str) -> str:
    """Strip HTML to readable plain text, preserving paragraph breaks."""
    text = _SCRIPT_STYLE.sub(' ', html)
    text = re.sub(r'<(br|/p|/div|/tr|/h[1-6])\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = _TAG.sub(' ', text)
    for entity, replacement in _ENTITIES.items():
        text = text.replace(entity, replacement)
    text = re.sub(r'&#\d+;', ' ', text)
    text = text.replace('\u00a0', ' ')
    text = _WS.sub(' ', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return _BLANKS.sub('\n\n', text).strip()


# Some filers never write "Item 1A." above the content. Intel's 10-K is a cross-reference
# layout: its item list points at page ranges ("Risk Factors Pages 37 - 51") and the
# sections themselves are marked only by a running page header repeating the title.
RUNNING_HEADER_TITLES: dict[str, str] = {
    '1A': 'Risk Factors',
    '7': "Management's Discussion and Analysis",
    '7A': 'Quantitative and Qualitative Disclosures About Market Risk',
}

MIN_RUNNING_HEADER_REPEATS = 3


def _sections_from_running_headers(text: str) -> list[Section]:
    """Recover sections from a filing that marks them with repeated page headers.

    Where a title appears alone on a line many times over, those are the page headers of
    one multi-page section, so the span from the first to the last occurrence is the
    section body. A title appearing once or twice is a cross-reference, not a header.
    """
    sections: list[Section] = []
    for item, title in RUNNING_HEADER_TITLES.items():
        pattern = re.compile(rf'^[ \t]*{re.escape(title)}[ \t]*$', re.IGNORECASE | re.MULTILINE)
        positions = [m.start() for m in pattern.finditer(text)]
        if len(positions) < MIN_RUNNING_HEADER_REPEATS:
            continue

        cluster = _densest_cluster(positions)
        if len(cluster) < MIN_RUNNING_HEADER_REPEATS:
            continue

        start, last = cluster[0], cluster[-1]
        page_length = (last - start) // max(1, len(cluster) - 1)
        # Extend past the final header to cover that last page's content.
        end = min(len(text), last + page_length + 2000)
        body = text[start:end].strip()
        if len(body) >= 500:
            sections.append(Section(item=item, title=WANTED_ITEMS[item], text=body))
    return sections


def _densest_cluster(positions: list[int]) -> list[int]:
    """Keep only the evenly-spaced run of positions.

    The same title also appears in the table of contents, thousands of characters away
    from the section itself. Page headers repeat at roughly one page interval, so the
    contents entry shows up as an outlier gap and gets dropped.
    """
    if len(positions) < 3:
        return positions
    gaps = sorted(positions[i + 1] - positions[i] for i in range(len(positions) - 1))
    median_gap = gaps[len(gaps) // 2]
    limit = max(median_gap * 3, 1)

    runs: list[list[int]] = [[positions[0]]]
    for previous, current in zip(positions, positions[1:], strict=False):
        if current - previous <= limit:
            runs[-1].append(current)
        else:
            runs.append([current])
    return max(runs, key=len)


def split_sections(text: str) -> list[Section]:
    """Split plain 10-K text into its Item sections.

    A 10-K names each Item twice: once in the table of contents and once at the section
    itself. The table of contents entries come first and have almost no text after them,
    so taking the *last* occurrence of each Item heading skips the contents reliably
    without needing to detect the table itself.

    Falls back to running-header detection for filers that never write the Item number
    above the content.
    """
    matches = list(_ITEM_HEADING.finditer(text))
    if not matches:
        return _sections_from_running_headers(text)

    last_start: dict[str, int] = {}
    ordered: list[tuple[str, int]] = []
    for match in matches:
        item = match.group(1).upper()
        last_start[item] = match.start()
    for item, start in sorted(last_start.items(), key=lambda kv: kv[1]):
        ordered.append((item, start))

    sections: list[Section] = []
    for index, (item, start) in enumerate(ordered):
        if item not in WANTED_ITEMS:
            continue
        end = ordered[index + 1][1] if index + 1 < len(ordered) else len(text)
        body = text[start:end].strip()
        # A heading with almost nothing after it is a stray cross-reference, not a section.
        if len(body) < 500:
            continue
        sections.append(Section(item=item, title=WANTED_ITEMS[item], text=body))

    return sections or _sections_from_running_headers(text)
