from contextlib import contextmanager

import httpx
from openinference.instrumentation.langchain import LangChainInstrumentor, get_current_span
from opentelemetry import trace
from phoenix.otel import register

from . import config


@contextmanager
def node_span():
    """Make the running LangGraph node's span the current OTel span.

    The LangChain instrumentor does not do this inside async node bodies, so without it
    non-LangChain spans (our crawl spans) would start new traces instead of nesting under
    their node. LangChain calls such as ChatOpenAI nest on their own and don't need it."""
    span = get_current_span()
    if span is None:  # tracing is not set up (tests)
        yield trace.INVALID_SPAN
        return
    with trace.use_span(span, end_on_exit=False):
        yield span


def setup_tracing():
    """Send spans to the Phoenix collector (PHOENIX_COLLECTOR_ENDPOINT or localhost:6006).
    Call shutdown() on the returned provider before exit, or the last batch of spans is lost.

    Only the LangChain instrumentor: it covers the graph, the nodes and ChatOpenAI.
    Adding OpenAIInstrumentor would duplicate every LLM call as an extra span."""
    tracer_provider = register(project_name=config.PHOENIX_PROJECT, batch=True, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    return tracer_provider


def phoenix_reachable() -> bool:
    try:
        httpx.get(config.PHOENIX_URL, timeout=2)
        return True
    except httpx.HTTPError:
        return False
