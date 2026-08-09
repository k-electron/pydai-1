"""Ground truth pulled from the database, and the arithmetic for checking against it.

The reason this project uses EDGAR: expected answers are real reported figures, so a
numeric claim can be graded exactly instead of being handed to a judge model. Expectations
are read from the corpus at eval time rather than pasted into the file, so re-ingesting
newer filings updates the targets instead of breaking the suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import asyncpg

# Magnitude words and suffixes a model might use when writing a figure out.
_SCALES: dict[str, float] = {
    'trillion': 1e12,
    't': 1e12,
    'billion': 1e9,
    'bn': 1e9,
    'b': 1e9,
    'million': 1e6,
    'mm': 1e6,
    'm': 1e6,
    'thousand': 1e3,
    'k': 1e3,
}

_NUMBER = re.compile(
    r'(-?\$?\s?\d[\d,]*(?:\.\d+)?)\s*(trillion|billion|million|thousand|bn|mm|[tbmk])?\b',
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ExpectedFact:
    ticker: str
    concept: str
    fiscal_year: int
    value: float

    @property
    def label(self) -> str:
        return f'{self.ticker} {self.concept} FY{self.fiscal_year}'


def extract_numbers(text: str) -> list[float]:
    """Pull every number out of prose, resolving magnitude words.

    "$60.922 billion", "60,922 million" and "60.922B" all become 60922000000.0, so a
    claim can be checked without dictating how the model writes it.
    """
    values: list[float] = []
    for raw, suffix in _NUMBER.findall(text):
        cleaned = raw.replace('$', '').replace(',', '').strip()
        try:
            number = float(cleaned)
        except ValueError:
            continue
        values.append(number * _SCALES.get(suffix.lower(), 1.0) if suffix else number)
    return values


def mentions_value(text: str, expected: float, *, tolerance: float = 0.01) -> bool:
    """Whether `text` states `expected`, within a relative tolerance.

    The tolerance absorbs legitimate rounding: a model writing "$60.9 billion" for
    60,922,000,000 is correct, and demanding exactness would fail it.

    Bare magnitudes are also accepted, since a table of figures headed "in millions"
    prints 60,922 for the same number.
    """
    if expected == 0:
        return any(abs(v) < 1e-6 for v in extract_numbers(text))

    target = abs(expected)
    for value in extract_numbers(text):
        candidate = abs(value)
        if candidate == 0:
            continue
        for scale in (1.0, 1e3, 1e6, 1e9):
            if abs(candidate * scale - target) / target <= tolerance:
                return True
    return False


def mentions_ratio(
    text: str, numerator: float, denominator: float, *, tolerance: float = 0.05
) -> bool:
    """Whether `text` states the percentage `numerator / denominator`."""
    if not denominator:
        return False
    expected_pct = 100.0 * numerator / denominator
    return any(
        abs(value - expected_pct) / max(abs(expected_pct), 1e-9) <= tolerance
        for value in extract_numbers(text)
    )


async def load_facts(
    pool: asyncpg.Pool,
    *,
    tickers: list[str],
    concepts: list[str],
    fiscal_years: list[int],
) -> dict[tuple[str, str, int], ExpectedFact]:
    """Read expected values straight from the corpus."""
    rows = await pool.fetch(
        """
        SELECT ticker, concept, fiscal_year, value
        FROM xbrl_facts
        WHERE fiscal_period = 'FY'
          AND ticker = ANY($1) AND concept = ANY($2) AND fiscal_year = ANY($3)
        """,
        tickers,
        concepts,
        fiscal_years,
    )
    return {
        (r['ticker'], r['concept'], r['fiscal_year']): ExpectedFact(
            ticker=r['ticker'],
            concept=r['concept'],
            fiscal_year=r['fiscal_year'],
            value=float(r['value']),
        )
        for r in rows
    }


def brief_text(brief) -> str:
    """Flatten a brief into the text a numeric check should search."""
    parts = [brief.summary]
    parts.extend(f.claim for f in brief.findings)
    parts.extend(brief.caveats)
    return '\n'.join(parts)
