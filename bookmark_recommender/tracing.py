from contextlib import contextmanager

from openinference.instrumentation.langchain import get_current_span
from opentelemetry import trace


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
