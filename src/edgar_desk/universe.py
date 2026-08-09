"""The seed universe.

Twenty large-cap technology filers, with CIKs taken from the SEC's own
`company_tickers.json`. Deliberately narrow: the point is a corpus small enough to
re-ingest in minutes, but real enough that answers are checkable.
"""

from __future__ import annotations

from edgar_desk.schemas import CompanyRef

SEED_COMPANIES: tuple[CompanyRef, ...] = (
    CompanyRef(ticker='AAPL', name='Apple Inc.', cik='0000320193'),
    CompanyRef(ticker='ADBE', name='Adobe Inc.', cik='0000796343'),
    CompanyRef(ticker='AMD', name='Advanced Micro Devices, Inc.', cik='0000002488'),
    CompanyRef(ticker='AMZN', name='Amazon.com, Inc.', cik='0001018724'),
    CompanyRef(ticker='AVGO', name='Broadcom Inc.', cik='0001730168'),
    CompanyRef(ticker='CRM', name='Salesforce, Inc.', cik='0001108524'),
    CompanyRef(ticker='CSCO', name='Cisco Systems, Inc.', cik='0000858877'),
    CompanyRef(ticker='GOOGL', name='Alphabet Inc.', cik='0001652044'),
    CompanyRef(ticker='INTC', name='Intel Corporation', cik='0000050863'),
    CompanyRef(ticker='META', name='Meta Platforms, Inc.', cik='0001326801'),
    CompanyRef(ticker='MSFT', name='Microsoft Corporation', cik='0000789019'),
    CompanyRef(ticker='MU', name='Micron Technology, Inc.', cik='0000723125'),
    CompanyRef(ticker='NFLX', name='Netflix, Inc.', cik='0001065280'),
    CompanyRef(ticker='NVDA', name='NVIDIA Corporation', cik='0001045810'),
    CompanyRef(ticker='ORCL', name='Oracle Corporation', cik='0001341439'),
    CompanyRef(ticker='PLTR', name='Palantir Technologies Inc.', cik='0001321655'),
    CompanyRef(ticker='QCOM', name='QUALCOMM Incorporated', cik='0000804328'),
    CompanyRef(ticker='SNOW', name='Snowflake Inc.', cik='0001640147'),
    CompanyRef(ticker='TSLA', name='Tesla, Inc.', cik='0001318605'),
    CompanyRef(ticker='TXN', name='Texas Instruments Incorporated', cik='0000097476'),
)

BY_TICKER: dict[str, CompanyRef] = {c.ticker: c for c in SEED_COMPANIES}

_BY_NAME_WORD: dict[str, CompanyRef] = {
    c.name.split()[0].lower().rstrip(',.'): c for c in SEED_COMPANIES
}


def resolve(symbol_or_name: str) -> CompanyRef | None:
    """Look up a company by ticker or leading name word, case-insensitively."""
    key = symbol_or_name.strip().upper()
    if key in BY_TICKER:
        return BY_TICKER[key]
    return _BY_NAME_WORD.get(symbol_or_name.strip().split()[0].lower().rstrip(',.'))
