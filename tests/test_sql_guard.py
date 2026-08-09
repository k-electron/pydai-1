"""SQL guard tests.

This code decides what model-written SQL reaches the database, so it gets tested harder
than the rest. The first test exists because the guard originally returned its own
scanning copy, which silently rewrote every string literal to empty.
"""

from __future__ import annotations

import pytest

from edgar_desk.retrieval.sql_guard import UnsafeQuery, validate_select


def test_string_literals_survive_validation() -> None:
    """The scanning copy blanks string literals; the executed statement must not."""
    sql = "SELECT value FROM xbrl_facts WHERE ticker = 'AAPL' AND concept = 'Revenue'"
    assert validate_select(sql) == sql


def test_trailing_semicolon_is_trimmed() -> None:
    assert validate_select('SELECT 1 FROM companies;') == 'SELECT 1 FROM companies'


def test_cte_is_allowed() -> None:
    sql = (
        "WITH annual AS (SELECT * FROM xbrl_facts WHERE fiscal_period = 'FY') SELECT * FROM annual"
    )
    assert validate_select(sql) == sql


@pytest.mark.parametrize(
    'sql',
    [
        'DELETE FROM xbrl_facts',
        'DROP TABLE companies',
        'UPDATE companies SET ticker = 1',
        'INSERT INTO companies VALUES (1)',
        'TRUNCATE chunks',
        'GRANT ALL ON companies TO public',
    ],
)
def test_writes_are_rejected(sql: str) -> None:
    with pytest.raises(UnsafeQuery):
        validate_select(sql)


def test_stacked_statement_is_rejected() -> None:
    with pytest.raises(UnsafeQuery, match='one statement'):
        validate_select('SELECT 1 FROM companies; DROP TABLE companies')


def test_write_hidden_behind_a_comment_is_rejected() -> None:
    """Comments are stripped before scanning, so a statement cannot hide inside one."""
    with pytest.raises(UnsafeQuery):
        validate_select('SELECT 1 FROM companies /* x */ ; DELETE FROM companies')


def test_keyword_inside_a_string_literal_is_not_a_write() -> None:
    """A company legitimately named in a filter must not trip the keyword scan."""
    sql = "SELECT * FROM xbrl_facts WHERE concept = 'delete this concept'"
    assert validate_select(sql) == sql


def test_system_catalogs_are_rejected() -> None:
    with pytest.raises(UnsafeQuery, match='System catalogs'):
        validate_select('SELECT * FROM pg_shadow')
    with pytest.raises(UnsafeQuery, match='System catalogs'):
        validate_select('SELECT * FROM information_schema.tables')


def test_unknown_table_is_rejected_with_a_helpful_message() -> None:
    with pytest.raises(UnsafeQuery, match='secrets'):
        validate_select('SELECT * FROM secrets')


def test_query_reading_no_table_is_rejected() -> None:
    with pytest.raises(UnsafeQuery):
        validate_select('SELECT 1')


def test_non_select_is_rejected() -> None:
    with pytest.raises(UnsafeQuery, match='Only SELECT'):
        validate_select('EXPLAIN ANALYZE SELECT * FROM companies')


def test_empty_and_oversized_queries_are_rejected() -> None:
    with pytest.raises(UnsafeQuery):
        validate_select('   ')
    with pytest.raises(UnsafeQuery, match='longer than'):
        validate_select('SELECT * FROM companies WHERE ticker = ' + "'x'" * 3000)


def test_join_across_allowed_tables_is_accepted() -> None:
    sql = (
        'SELECT f.ticker, c.name FROM xbrl_facts f '
        'JOIN companies c ON c.cik = f.cik WHERE f.fiscal_year = 2024'
    )
    assert validate_select(sql) == sql
