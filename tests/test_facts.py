"""XBRL normalization tests.

The value of choosing EDGAR is that its numbers are checkable, which only holds if
normalization is right. These use fixtures shaped like real `companyfacts` payloads,
including the specific quirks that produced wrong numbers during development.
"""

from __future__ import annotations

from datetime import date

from edgar_desk.edgar.facts import (
    fiscal_period_for,
    fiscal_year_for,
    normalize_company_facts,
)


def _payload(entries: list[dict], tag: str = 'Revenues', taxonomy: str = 'us-gaap') -> dict:
    return {'facts': {taxonomy: {tag: {'units': {'USD': entries}}}}}


def test_period_classified_by_duration_not_form() -> None:
    """A 10-K contains quarterly breakdowns as well as the annual figure, and the SEC
    labels every fact in it `FY`."""
    assert fiscal_period_for(date(2023, 1, 30), date(2024, 1, 28), 'FY') == 'FY'
    assert fiscal_period_for(date(2023, 5, 1), date(2023, 10, 29), 'FY') == 'YTD'
    assert fiscal_period_for(None, date(2024, 1, 28), 'FY') == 'PIT'


def test_quarters_are_fiscal_not_calendar() -> None:
    """NVIDIA's fiscal year ends in January, so its Q4 ends in January too. Calendar
    quarter arithmetic would call that Q1."""
    jan = 1
    assert fiscal_period_for(date(2023, 10, 30), date(2024, 1, 28), 'FY', jan) == 'Q4'
    assert fiscal_period_for(date(2024, 1, 29), date(2024, 4, 28), 'FY', jan) == 'Q1'
    assert fiscal_period_for(date(2024, 4, 29), date(2024, 7, 28), 'FY', jan) == 'Q2'

    sep = 9  # Apple
    assert fiscal_period_for(date(2023, 10, 1), date(2023, 12, 30), 'FY', sep) == 'Q1'
    assert fiscal_period_for(date(2024, 6, 30), date(2024, 9, 28), 'FY', sep) == 'Q4'


def test_fiscal_year_end_month_is_inferred_from_annual_periods() -> None:
    """A non-calendar fiscal year has to be detected, not assumed."""
    annual = [
        {'start': f'{y - 1}-01-30', 'end': f'{y}-01-28', 'val': y, 'fy': y, 'fp': 'FY'}
        for y in (2023, 2024, 2025)
    ]
    q4 = {'start': '2024-10-30', 'end': '2025-01-26', 'val': 1, 'fy': 2025, 'fp': 'FY'}
    facts = normalize_company_facts(_payload([*annual, q4]), cik='X', ticker='NVDA')
    by_period = {(f.fiscal_year, f.fiscal_period) for f in facts}
    assert (2025, 'Q4') in by_period
    assert (2025, 'FY') in by_period


def test_fiscal_year_comes_from_period_end() -> None:
    assert fiscal_year_for(date(2024, 1, 28)) == 2024
    assert fiscal_year_for(date(2024, 9, 28)) == 2024


def test_restated_period_collapses_to_one_row() -> None:
    """NVIDIA's FY2024 revenue appears in the FY2024, FY2025 and FY2026 10-Ks with
    fy=2024, 2025 and 2026. It is one fact, and it belongs to fiscal 2024."""
    shared = {'start': '2023-01-30', 'end': '2024-01-28', 'val': 60922000000, 'form': '10-K'}
    facts = normalize_company_facts(
        _payload(
            [
                {**shared, 'fy': 2024, 'fp': 'FY', 'accn': 'a-24', 'filed': '2024-02-21'},
                {**shared, 'fy': 2025, 'fp': 'FY', 'accn': 'a-25', 'filed': '2025-02-26'},
                {**shared, 'fy': 2026, 'fp': 'FY', 'accn': 'a-26', 'filed': '2026-02-25'},
            ]
        ),
        cik='0001045810',
        ticker='NVDA',
    )
    assert len(facts) == 1
    fact = facts[0]
    assert fact.fiscal_year == 2024
    assert fact.fiscal_period == 'FY'
    assert fact.value == 60922000000
    # The newest filing wins, so a restated value would be picked up.
    assert fact.accession == 'a-26'


def test_restatement_value_wins() -> None:
    shared = {'start': '2023-01-01', 'end': '2023-12-31', 'form': '10-K', 'fp': 'FY'}
    facts = normalize_company_facts(
        _payload(
            [
                {**shared, 'val': 100, 'fy': 2023, 'accn': 'orig', 'filed': '2024-01-15'},
                {**shared, 'val': 110, 'fy': 2024, 'accn': 'restated', 'filed': '2025-01-15'},
            ]
        ),
        cik='X',
        ticker='X',
    )
    assert [f.value for f in facts] == [110]
    assert facts[0].accession == 'restated'


def test_synonymous_tags_collapse_to_one_concept_row() -> None:
    """A filer reporting the same period under two revenue tags must not produce two
    rows, or any aggregate over the concept double-counts."""
    period = {'start': '2021-02-01', 'end': '2022-01-30', 'fy': 2022, 'fp': 'FY', 'form': '10-K'}
    payload = {
        'facts': {
            'us-gaap': {
                'Revenues': {
                    'units': {
                        'USD': [{**period, 'val': 26914000000, 'accn': 'a', 'filed': '2022-03-18'}]
                    }
                },
                'RevenueFromContractWithCustomerExcludingAssessedTax': {
                    'units': {
                        'USD': [{**period, 'val': 26914000000, 'accn': 'a', 'filed': '2022-03-18'}]
                    }
                },
            }
        }
    }
    facts = normalize_company_facts(payload, cik='0001045810', ticker='NVDA')
    revenue = [f for f in facts if f.concept == 'Revenue']
    assert len(revenue) == 1
    # The tuple order in CONCEPTS decides the winner.
    assert revenue[0].tag == 'RevenueFromContractWithCustomerExcludingAssessedTax'


def test_unrelated_tags_are_dropped() -> None:
    facts = normalize_company_facts(
        _payload(
            [{'start': '2023-01-01', 'end': '2023-12-31', 'val': 1, 'fy': 2023, 'fp': 'FY'}],
            tag='SomeObscureFootnoteTag',
        ),
        cik='X',
        ticker='X',
    )
    assert facts == []


def test_min_fiscal_year_filter() -> None:
    entries = [
        {'start': f'{y}-01-01', 'end': f'{y}-12-31', 'val': y, 'fy': y, 'fp': 'FY'}
        for y in (2015, 2020, 2024)
    ]
    facts = normalize_company_facts(_payload(entries), cik='X', ticker='X', min_fiscal_year=2018)
    assert sorted(f.fiscal_year for f in facts) == [2020, 2024]
