"""Validation for model-written SQL.

Letting a model write SQL against a real database is the kind of thing that needs both a
belt and braces. The braces are Postgres itself: queries run inside a `READ ONLY`
transaction as a role with no write grants, so even a successful injection cannot modify
anything. The belt is this module, which rejects obviously wrong statements early and
gives the model a specific error it can correct, rather than a Postgres permission error
it will not understand.
"""

from __future__ import annotations

import re

ALLOWED_TABLES = frozenset({'xbrl_facts', 'chunks', 'filings', 'companies'})

FORBIDDEN = (
    'insert',
    'update',
    'delete',
    'drop',
    'alter',
    'create',
    'truncate',
    'grant',
    'revoke',
    'copy',
    'vacuum',
    'reindex',
    'call',
    'do',
    'execute',
    'listen',
    'notify',
)

_COMMENT = re.compile(r'(--[^\n]*)|(/\*.*?\*/)', re.DOTALL)
_STRING = re.compile(r"'(?:[^']|'')*'")
_WORD = re.compile(r'\b[a-z_][a-z0-9_]*\b')


class UnsafeQuery(ValueError):
    """Raised when a query is rejected before it reaches the database."""


def _strip_noise(sql: str) -> str:
    """Remove comments and string literals before keyword scanning.

    Without this, a perfectly innocent `WHERE ticker = 'DELETE'` trips the keyword check,
    and a comment could hide a second statement from a naive scan.
    """
    return _STRING.sub("''", _COMMENT.sub(' ', sql))


def validate_select(sql: str, *, max_length: int = 4000) -> str:
    """Check a single read-only SELECT and return the executable statement.

    Scanning happens on a copy with comments and string literals removed, but what comes
    back is the *original* text with only whitespace and a trailing semicolon trimmed.
    Returning the stripped copy would execute `WHERE ticker = ''` for a query that asked
    for `WHERE ticker = 'AAPL'`, and silently return nothing.

    Raises `UnsafeQuery` with a message written for the model to act on.
    """
    if not sql or not sql.strip():
        raise UnsafeQuery('Query is empty.')
    if len(sql) > max_length:
        raise UnsafeQuery(f'Query is longer than {max_length} characters.')

    statement = sql.strip().rstrip(';').strip()
    scannable = _strip_noise(statement).strip().rstrip(';').strip()
    if not scannable:
        raise UnsafeQuery('Query contains no statement.')

    if ';' in scannable:
        raise UnsafeQuery('Send exactly one statement; remove the semicolon and anything after it.')

    lowered = scannable.lower()
    if not (lowered.startswith('select') or lowered.startswith('with')):
        raise UnsafeQuery('Only SELECT queries are allowed. Start with SELECT or WITH.')

    words = set(_WORD.findall(lowered))
    banned = words & set(FORBIDDEN)
    if banned:
        raise UnsafeQuery(f'These keywords are not allowed: {", ".join(sorted(banned))}.')

    if 'pg_' in lowered or 'information_schema' in lowered:
        raise UnsafeQuery('System catalogs are not available. Query the data tables instead.')

    referenced = _referenced_tables(lowered)
    unknown = referenced - ALLOWED_TABLES
    if unknown:
        raise UnsafeQuery(
            f'Unknown table(s): {", ".join(sorted(unknown))}. '
            f'Available tables: {", ".join(sorted(ALLOWED_TABLES))}.'
        )
    if not referenced:
        raise UnsafeQuery(
            f'Query must read from at least one of: {", ".join(sorted(ALLOWED_TABLES))}.'
        )

    return statement


def _referenced_tables(lowered_sql: str) -> set[str]:
    """Collect names appearing after FROM or JOIN.

    Common table expressions are subtracted, since a `WITH x AS (...)` name is defined by
    the query itself rather than being a table it reads.
    """
    names = set(re.findall(r'\b(?:from|join)\s+([a-z_][a-z0-9_]*)', lowered_sql))
    ctes = set(re.findall(r'\b([a-z_][a-z0-9_]*)\s+as\s*\(', lowered_sql))
    return names - ctes
