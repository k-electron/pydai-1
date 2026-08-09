"""An agent that consumes an external MCP server.

The other half of the protocol. `edgar_desk.mcp_server` publishes this project's tools;
this shows the same agent picking up tools from somewhere else, with no code change
beyond one more entry in `capabilities=[...]`.

`native=False` forces the MCP server to run locally in this process rather than being
handed to the model provider to execute server-side. That is the right default here:
local models have no native MCP support, and a locally-run server keeps its credentials
and its traffic on this machine.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP, PrefixTools

from edgar_desk.capabilities import financials_capability, narrative_capability
from edgar_desk.deps import EdgarDeps
from edgar_desk.models import DEFAULT_MODEL_SETTINGS, primary_model
from edgar_desk.schemas import Brief

INSTRUCTIONS = """\
You are a financial research analyst working from SEC filings.

You also have tools from external MCP servers. Treat filings as the authoritative source
for anything a company reported; use external tools only for context filings cannot
provide, and say which source each claim came from.

Every finding needs a claim, at least one citation, and a confidence between 0 and 1.
Never state a figure you did not retrieve with a tool.
"""


def build_connected_agent(
    mcp_url: str,
    *,
    allowed_tools: list[str] | None = None,
    prefix: str | None = 'ext',
    local_capabilities: bool = True,
):
    """Build an analyst that also carries tools from an external MCP server.

    Args:
        mcp_url: URL of the MCP server. Transport is inferred from the URL.
        allowed_tools: Restrict to these tool names. Worth setting for an unfamiliar
            server, since every extra tool costs context on every request.
        prefix: Namespace for the remote tools. A server with no name overlap can pass
            `None`, but a collision is a hard error rather than a silent shadowing, so
            prefixing by default is the safer choice.
        local_capabilities: Whether to also carry this project's own tools.
    """
    remote = MCP(mcp_url, native=False, allowed_tools=allowed_tools)

    capabilities: list[object] = []
    if local_capabilities:
        capabilities.extend([financials_capability, narrative_capability])
    # Tool names live in one flat namespace per agent, and two capabilities offering the
    # same name raises rather than silently shadowing. Any server whose vocabulary
    # overlaps yours -- including, as it happens, this project's own MCP server -- needs
    # the remote side namespaced.
    capabilities.append(PrefixTools(remote, prefix) if prefix else remote)

    return Agent(
        primary_model(),
        deps_type=EdgarDeps,
        output_type=Brief,
        instructions=INSTRUCTIONS,
        name='connected-analyst',
        retries=2,
        model_settings=DEFAULT_MODEL_SETTINGS,
        capabilities=capabilities,  # type: ignore[arg-type]
    )
