"""Split section text into embeddable chunks.

Paragraph-aware with overlap: risk factors are written as self-contained paragraphs, so
splitting on paragraph boundaries keeps each chunk about one risk rather than straddling
two. The overlap keeps a claim that spans a boundary retrievable from either side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# bge-m3 handles 8192 tokens, but retrieval quality falls off well before that: a chunk
# holding five unrelated risks matches everything weakly and nothing strongly.
TARGET_CHARS = 2400
OVERLAP_CHARS = 300
MIN_CHARS = 200

_PARAGRAPH = re.compile(r'\n\s*\n')


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    text: str

    @property
    def approx_tokens(self) -> int:
        """Rough token estimate; English prose runs about four characters per token."""
        return max(1, len(self.text) // 4)


def _split_oversized(paragraph: str, limit: int) -> list[str]:
    """Break a single huge paragraph on sentence boundaries."""
    if len(paragraph) <= limit:
        return [paragraph]
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    out: list[str] = []
    current = ''
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > limit:
            out.append(current)
            current = sentence
        else:
            current = f'{current} {sentence}'.strip()
    if current:
        out.append(current)
    return out


def chunk_text(
    text: str,
    *,
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Chunk]:
    paragraphs: list[str] = []
    for raw in _PARAGRAPH.split(text):
        cleaned = raw.strip()
        if cleaned:
            paragraphs.extend(_split_oversized(cleaned, target_chars))

    chunks: list[str] = []
    current = ''
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > target_chars:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ''
            # Resume at a word boundary so the overlap does not start mid-token.
            if tail and ' ' in tail:
                tail = tail[tail.index(' ') + 1 :]
            current = f'{tail}\n\n{paragraph}'.strip()
        else:
            current = f'{current}\n\n{paragraph}'.strip() if current else paragraph

    if current:
        chunks.append(current)

    return [Chunk(index=i, text=c) for i, c in enumerate(chunks) if len(c) >= MIN_CHARS]
