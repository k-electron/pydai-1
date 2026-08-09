"""Turn the SEC's `companyfacts` payload into clean, deduplicated rows.

Two things make this less trivial than it looks, and both are handled here rather than
being left for the SQL layer or the model to cope with:

1. The same economic fact is reported once per filing that includes it, so a five-year
   revenue history contains each year several times over.
2. Companies use different tags for the same concept -- `Revenues` for one filer and
   `RevenueFromContractWithCustomerExcludingAssessedTax` for another -- so a query for
   "revenue" has to know about synonyms.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date

# The concepts worth loading, grouped under a canonical name. NVIDIA alone reports 626
# us-gaap tags; almost all of them are footnote detail that adds noise to retrieval and
# gives the model more ways to pick the wrong number.
CONCEPTS: dict[str, tuple[str, ...]] = {
    'Revenue': (
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'Revenues',
        'SalesRevenueNet',
    ),
    'CostOfRevenue': ('CostOfRevenue', 'CostOfGoodsAndServicesSold'),
    'GrossProfit': ('GrossProfit',),
    'ResearchAndDevelopment': ('ResearchAndDevelopmentExpense',),
    'SellingGeneralAndAdministrative': (
        'SellingGeneralAndAdministrativeExpense',
        'GeneralAndAdministrativeExpense',
    ),
    'OperatingIncome': ('OperatingIncomeLoss',),
    'NetIncome': ('NetIncomeLoss', 'ProfitLoss'),
    'EarningsPerShareDiluted': ('EarningsPerShareDiluted',),
    'OperatingCashFlow': ('NetCashProvidedByUsedInOperatingActivities',),
    'CapitalExpenditures': ('PaymentsToAcquirePropertyPlantAndEquipment',),
    'Assets': ('Assets',),
    'Liabilities': ('Liabilities',),
    'StockholdersEquity': ('StockholdersEquity',),
    'CashAndEquivalents': ('CashAndCashEquivalentsAtCarryingValue',),
    'LongTermDebt': ('LongTermDebtNoncurrent', 'LongTermDebt'),
    'InventoryNet': ('InventoryNet',),
    'SharesOutstanding': ('CommonStockSharesOutstanding', 'dei:EntityCommonStockSharesOutstanding'),
}

TAG_TO_CONCEPT: dict[str, str] = {
    tag: concept for concept, tags in CONCEPTS.items() for tag in tags
}

# Position within a concept's tuple is its priority: the first tag listed is the one to
# keep when a filer reports the same period under two synonymous tags.
TAG_PRIORITY: dict[str, int] = {
    tag: rank for tags in CONCEPTS.values() for rank, tag in enumerate(tags)
}

WANTED_TAGS: frozenset[str] = frozenset(TAG_TO_CONCEPT)


@dataclass(frozen=True, slots=True)
class NormalizedFact:
    """One XBRL fact, deduplicated to a single economic period."""

    cik: str
    ticker: str
    taxonomy: str
    tag: str
    concept: str
    unit: str
    value: float
    fiscal_year: int
    fiscal_period: str
    form: str
    start_date: date | None
    end_date: date
    accession: str
    frame: str | None


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def fiscal_period_for(
    start_date: date | None,
    end_date: date,
    reported_fp: str,
    fy_end_month: int = 12,
) -> str:
    """Classify a fact by the length of the period it covers.

    The SEC's `fp` field describes the filing, so every fact in a 10-K arrives as `FY` --
    including the quarterly breakdowns that a 10-K also contains. Taking it at face value
    labels NVIDIA's $2.9bn Q4 as a full year alongside its real $9.7bn annual figure, and
    any "annual revenue" query then returns both.

    Duration is unambiguous: a year is a year regardless of which form disclosed it.

    `fy_end_month` matters for quarter numbering. Quarters are fiscal, not calendar, so a
    company whose year ends in January has its Q4 end in January -- calendar-quarter
    arithmetic would call that Q1.
    """
    if start_date is None:
        # Balance-sheet facts are instantaneous, with no duration to measure.
        return 'PIT'
    days = (end_date - start_date).days
    if days >= 300:
        return 'FY'
    if days >= 150:
        # Six- and nine-month cumulative figures from 10-Qs. Not a quarter and not a
        # year; naming them YTD keeps them from being mistaken for either.
        return 'YTD'
    if reported_fp in ('Q1', 'Q2', 'Q3', 'Q4'):
        return reported_fp
    months_into_year = (end_date.month - fy_end_month - 1) % 12
    return f'Q{months_into_year // 3 + 1}'


def fiscal_year_for(end_date: date) -> int:
    """Label a period by the calendar year its fiscal year ends in.

    Deriving this from the period rather than trusting the SEC's `fy` field is what makes
    comparative figures come out right. `fy` describes the filing, so NVIDIA's FY2020 and
    FY2021 revenue -- restated as comparatives in the FY2022 10-K under a tag the company
    only started using then -- both arrive labelled fy=2022.

    The convention here (fiscal year named for the calendar year it ends in) holds for
    every filer in the covered universe. It does not hold universally: some retailers
    with a January year-end name it for the prior year.
    """
    return end_date.year


def _detect_fiscal_year_end_month(
    grouped: dict[tuple[str, str, str, str | None, str], list[dict]],
) -> int:
    """Infer which month the company's fiscal year ends in.

    Read off the annual periods themselves rather than assuming December: roughly a third
    of large filers are on a non-calendar year, and the fiscal year end is what makes
    quarter numbering come out right.
    """
    counts: Counter[int] = Counter()
    for _taxonomy, _tag, _unit, start, end in grouped:
        start_date, end_date = _parse_date(start), _parse_date(end)
        if start_date is None or end_date is None:
            continue
        if (end_date - start_date).days >= 300:
            counts[end_date.month] += 1
    return counts.most_common(1)[0][0] if counts else 12


def normalize_company_facts(
    payload: dict,
    *,
    cik: str,
    ticker: str,
    min_fiscal_year: int = 2018,
) -> list[NormalizedFact]:
    """Flatten `companyfacts` into one row per (concept, unit, period).

    Three collapses happen here, in order:

    1. Across filings -- a period reported in several filings keeps the value and
       accession from the most recent one, so restatements win.
    2. Across synonymous tags -- when a filer reports the same period as both `Revenues`
       and `RevenueFromContractWithCustomerExcludingAssessedTax`, only the higher-priority
       tag survives. Without this, summing or averaging by concept double-counts.
    3. Fiscal labelling -- derived from the period end date, not the filing's `fy`.
    """
    grouped: dict[tuple[str, str, str, str | None, str], list[dict]] = defaultdict(list)

    for taxonomy, tags in payload.get('facts', {}).items():
        for tag, detail in tags.items():
            if tag not in WANTED_TAGS:
                continue
            for unit, entries in detail.get('units', {}).items():
                for entry in entries:
                    if 'val' not in entry or 'end' not in entry:
                        continue
                    key = (taxonomy, tag, unit, entry.get('start'), entry['end'])
                    grouped[key].append(entry)

    fy_end_month = _detect_fiscal_year_end_month(grouped)

    candidates: list[NormalizedFact] = []
    for (taxonomy, tag, unit, start, end), entries in grouped.items():
        newest = max(entries, key=lambda e: (e.get('filed', ''), e.get('accn', '')))

        end_date = _parse_date(end)
        if end_date is None:
            continue

        fiscal_year = fiscal_year_for(end_date)
        if fiscal_year < min_fiscal_year:
            continue

        start_date = _parse_date(start)
        fiscal_period = fiscal_period_for(
            start_date, end_date, str(newest.get('fp') or ''), fy_end_month
        )

        candidates.append(
            NormalizedFact(
                cik=cik,
                ticker=ticker,
                taxonomy=taxonomy,
                tag=tag,
                concept=TAG_TO_CONCEPT[tag],
                unit=unit,
                value=float(newest['val']),
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                form=str(newest.get('form', '')),
                start_date=start_date,
                end_date=end_date,
                accession=str(newest.get('accn', '')),
                frame=newest.get('frame'),
            )
        )

    # Collapse synonymous tags: one row per concept per period.
    best: dict[tuple[str, str, date | None, date], NormalizedFact] = {}
    for fact in candidates:
        key = (fact.concept, fact.unit, fact.start_date, fact.end_date)
        incumbent = best.get(key)
        if incumbent is None or TAG_PRIORITY[fact.tag] < TAG_PRIORITY[incumbent.tag]:
            best[key] = fact

    return list(best.values())


def annual_facts(facts: list[NormalizedFact]) -> list[NormalizedFact]:
    """Just the full-year figures, which is what most comparisons want."""
    return [f for f in facts if f.fiscal_period == 'FY']
