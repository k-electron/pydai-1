"""Ground truth matching tests.

Every numeric grade in the eval suite goes through this code, so a bug here would quietly
mark correct answers wrong (or worse, wrong answers right).
"""

from __future__ import annotations

import pytest

from edgar_desk.evals.ground_truth import extract_numbers, mentions_ratio, mentions_value

NVDA_FY2024_REVENUE = 60_922_000_000.0


@pytest.mark.parametrize(
    'text',
    [
        '$60.922 billion',
        '60.922B',
        'revenue of $60,922 million',
        'USD 60922000000',
        'about $60.9 billion',
        'roughly 60.92 bn',
    ],
)
def test_a_figure_is_recognized_however_it_is_written(text: str) -> None:
    """Models write numbers many ways, and the grade should not depend on the styling."""
    assert mentions_value(text, NVDA_FY2024_REVENUE)


@pytest.mark.parametrize(
    'text',
    ['$130.5 billion', '$6.09 billion', '609.22 billion', 'revenue grew 12%'],
)
def test_a_wrong_figure_is_not_accepted(text: str) -> None:
    assert not mentions_value(text, NVDA_FY2024_REVENUE)


def test_bare_magnitudes_are_accepted() -> None:
    """Filings tabulate in millions, so a bare "60,922" is the same number.

    This is a deliberate trade. Accepting unscaled magnitudes means a figure that is
    genuinely off by exactly 1000x would slip through, but refusing them would fail every
    correct answer written in the units filings actually use. For grading, a false
    negative is the worse error: it makes a passing agent look broken.
    """
    assert mentions_value('R&D was 60,922 (in millions)', NVDA_FY2024_REVENUE)
    assert mentions_value('revenue of $60,922', NVDA_FY2024_REVENUE)


def test_rounding_is_tolerated_but_a_different_number_is_not() -> None:
    assert mentions_value('$60.9 billion', NVDA_FY2024_REVENUE)
    assert not mentions_value('$62.0 billion', NVDA_FY2024_REVENUE)


def test_negative_values_are_matched() -> None:
    """Intel reported a net loss in FY2024; sign must survive extraction."""
    assert mentions_value('a net loss of $18.756 billion', -18_756_000_000.0)


def test_zero_is_handled() -> None:
    assert mentions_value('exactly 0', 0.0)
    assert not mentions_value('exactly 5', 0.0)


def test_ratio_matching() -> None:
    rd, revenue = 8_091_000_000.0, 34_639_000_000.0  # AMD FY2025: 23.36%
    assert mentions_ratio('R&D was 23.4% of revenue', rd, revenue)
    assert mentions_ratio('about 23% of revenue', rd, revenue)
    assert not mentions_ratio('R&D was 12% of revenue', rd, revenue)


def test_ratio_with_zero_denominator_is_false_not_an_error() -> None:
    assert not mentions_ratio('anything', 1.0, 0.0)


def test_extract_numbers_resolves_magnitudes() -> None:
    values = extract_numbers('revenue $1.5 billion, cost 300 million, headcount 4,200')
    assert 1.5e9 in values
    assert 3e8 in values
    assert 4200.0 in values


def test_extraction_ignores_prose_without_numbers() -> None:
    assert extract_numbers('no figures at all here') == []
