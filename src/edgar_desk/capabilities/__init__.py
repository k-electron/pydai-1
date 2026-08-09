"""EDGAR Desk capabilities.

A capability bundles tools, instructions and hooks into one object you drop into an
agent's `capabilities=[...]`. That is what lets "can analyze financials" be a thing an
agent *has*, rather than a pile of constructor arguments to keep in sync across agents.
"""

from edgar_desk.capabilities.financials import financials_capability
from edgar_desk.capabilities.narrative import narrative_capability

__all__ = ['financials_capability', 'narrative_capability']
