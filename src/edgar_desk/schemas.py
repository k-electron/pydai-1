"""Typed domain model.

These types are the contract between every agent, tool, and eval in the project.
Structured output is the whole point of the framework, so the shapes live in one place.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class CompanyRef(BaseModel):
    """A company the agent has committed to analyzing."""

    ticker: str = Field(description='Exchange ticker symbol, uppercase, e.g. NVDA')
    name: str = Field(description='Company name as it appears in filings')
    cik: str | None = Field(
        default=None,
        description='SEC Central Index Key, zero-padded to 10 digits. Null if unresolved.',
    )


class Angle(StrEnum):
    """Which retrieval path a sub-question needs.

    This drives routing: `financial` questions go to SQL over XBRL facts,
    `narrative` questions go to vector search over filing prose.
    """

    FINANCIAL = 'financial'
    NARRATIVE = 'narrative'
    BOTH = 'both'


class SubQuestion(BaseModel):
    """One decomposed unit of research, answerable independently."""

    question: str = Field(description='A single self-contained question')
    angle: Angle = Field(description='Which evidence source can answer this')
    companies: list[str] = Field(
        default_factory=list, description='Tickers this sub-question concerns'
    )


class ResearchPlan(BaseModel):
    """The triage agent's output: what to research and where to look."""

    companies: list[CompanyRef] = Field(description='Companies in scope')
    sub_questions: list[SubQuestion] = Field(
        description='Between 1 and 6 sub-questions that together answer the request'
    )
    reasoning: str = Field(description='One sentence on why this decomposition')


class Citation(BaseModel):
    """Provenance for a single claim. Every finding must carry at least one."""

    source: str = Field(
        description="Either an XBRL tag like 'us-gaap:Revenues' or a filing section"
    )
    ticker: str
    fiscal_period: str | None = Field(
        default=None, description="Fiscal period, e.g. 'FY2024' or 'Q3 2025'"
    )
    accession: str | None = Field(default=None, description='SEC accession number of the filing')
    excerpt: str | None = Field(default=None, description='Supporting quote, if narrative')


class Finding(BaseModel):
    """One evidence-backed claim."""

    claim: str = Field(description='A single factual statement')
    citations: list[Citation] = Field(description='At least one source backing the claim')
    confidence: float = Field(ge=0.0, le=1.0, description='0-1 confidence in this claim')


class Brief(BaseModel):
    """The final research artifact a user reviews and approves."""

    question: str
    summary: str = Field(description='Two to four sentences answering the question directly')
    findings: list[Finding]
    caveats: list[str] = Field(default_factory=list, description='Limitations of this analysis')
    generated_on: date | None = None


class FactRow(BaseModel):
    """A single XBRL numeric fact, as stored and as returned by the SQL toolset."""

    ticker: str
    tag: str = Field(description="XBRL concept, e.g. 'Revenues'")
    unit: str
    value: float
    fiscal_year: int
    fiscal_period: str
    form: str = Field(description="Filing form type, e.g. '10-K'")
    end_date: date
    accession: str | None = None


class Passage(BaseModel):
    """A retrieved chunk of filing narrative."""

    ticker: str
    section: str = Field(description="e.g. 'Item 1A. Risk Factors'")
    fiscal_year: int
    accession: str
    text: str
    score: float = Field(description='Relevance score after reranking')
