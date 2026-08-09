"""Tracing that stays on this machine.

Pydantic AI emits OpenTelemetry, so pointing it at a local collector gives the full
Logfire-style view of agent runs without an account or a network egress.
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import logfire

from edgar_desk.settings import get_settings

_configured = False


def collector_reachable(endpoint: str, timeout: float = 0.3) -> bool:
    """Cheap TCP probe so we can skip the exporter when nothing is listening."""
    parsed = urlparse(endpoint)
    host, port = parsed.hostname or 'localhost', parsed.port or 4318
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def configure(service_name: str = 'edgar-desk', console: bool = False) -> bool:
    """Set up tracing that never leaves this machine.

    Always configures Logfire, because `pydantic-evals` needs a tracer provider before
    an evaluator can read `ctx.span_tree` -- span-based evaluators silently report false
    without one. Only the OTLP *export* is conditional: if no collector is listening,
    spans stay in-process rather than filling the console with retry errors.

    Returns whether traces are being exported.
    """
    global _configured
    settings = get_settings()
    exporting = collector_reachable(settings.otlp_endpoint)

    if _configured:
        return exporting

    if exporting:
        os.environ.setdefault('OTEL_EXPORTER_OTLP_ENDPOINT', settings.otlp_endpoint)
    else:
        # Logfire falls back to a no-op exporter when no endpoint is configured.
        os.environ.pop('OTEL_EXPORTER_OTLP_ENDPOINT', None)

    logfire.configure(
        service_name=service_name,
        send_to_logfire=False,
        console=logfire.ConsoleOptions(min_log_level='info') if console else False,
        # Jaeger ingests OTLP traces but has no /v1/metrics endpoint, so leaving metrics
        # on produces a 404 on every flush. Traces are what this project reads anyway.
        metrics=False,
    )
    logfire.instrument_pydantic_ai()
    if exporting:
        logfire.instrument_httpx(capture_all=False)

    _configured = True
    return exporting
