import asyncio

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry import trace

from . import config
from .models import PageContent

# Created at import time; it resolves to Phoenix's tracer provider once setup_tracing() has run.
tracer = OITracer(trace.get_tracer(__name__), TraceConfig())

RUN_CONFIG = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, page_timeout=config.CRAWL_TIMEOUT_S * 1000, verbose=False)


def to_page(url: str, result, limit: int) -> PageContent:
    if not result.success:
        return PageContent(url=url, ok=False, error=result.error_message or "fetch failed")
    markdown = result.markdown.raw_markdown if result.markdown else ""
    title = (result.metadata or {}).get("title") or ""
    if not markdown.strip():
        return PageContent(url=url, title=title, ok=False, error="page has no text")
    return PageContent(url=url, title=title, markdown=markdown[:limit], ok=True)


class Crawler:
    """One headless browser shared by every fetch in the CLI run."""

    def __init__(self):
        self._crawler = AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False))

    async def __aenter__(self):
        await self._crawler.start()
        return self

    async def __aexit__(self, *exc):
        await self._crawler.close()

    async def fetch_page(self, url: str, limit: int) -> PageContent:
        with tracer.start_as_current_span("crawl_page", openinference_span_kind="tool") as span:
            span.set_input(url)
            try:
                page = to_page(url, await self._crawler.arun(url, config=RUN_CONFIG), limit)
            except Exception as e:  # a broken page must not end the turn; it is recorded on the span
                span.record_exception(e)
                page = PageContent(url=url, ok=False, error=str(e))
            span.set_output({"title": page.title, "chars": len(page.markdown), "ok": page.ok, "error": page.error})
            return page

    async def fetch_many(self, urls: list[str], limit: int) -> list[PageContent]:
        return list(await asyncio.gather(*(self.fetch_page(u, limit) for u in urls)))
