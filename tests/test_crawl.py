from types import SimpleNamespace

from bookmark_recommender.crawl import to_page


def result(success=True, markdown="# Title\nBody text", title="A page", error=None):
    return SimpleNamespace(
        success=success,
        markdown=SimpleNamespace(raw_markdown=markdown) if markdown is not None else None,
        metadata={"title": title} if title is not None else None,
        error_message=error,
    )


def test_to_page_success_truncates_to_limit():
    page = to_page("https://a.com", result(markdown="x" * 50), limit=10)
    assert page.ok and page.markdown == "x" * 10 and page.title == "A page" and page.error is None


def test_to_page_failed_crawl_is_not_ok():
    page = to_page("https://a.com", result(success=False, markdown=None, error="net::ERR_NAME_NOT_RESOLVED"), limit=10)
    assert not page.ok and page.error == "net::ERR_NAME_NOT_RESOLVED"


def test_to_page_empty_markdown_is_not_ok():
    page = to_page("https://a.com", result(markdown="  \n "), limit=10)
    assert not page.ok and page.error == "page has no text"


def test_to_page_missing_metadata_gives_empty_title():
    assert to_page("https://a.com", result(title=None), limit=100).title == ""
