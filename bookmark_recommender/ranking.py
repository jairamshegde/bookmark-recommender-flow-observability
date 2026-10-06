from urllib.parse import urlsplit, urlunsplit

from .models import Judgment


def normalize_url(url: str) -> str:
    """Comparable form of a URL: lowercase scheme and host, no trailing slash, no fragment."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


def domain_of(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def _passes_gate(j: Judgment) -> bool:
    return j.relevance >= 3 and j.usefulness >= 3 and j.novelty >= 2


def select_top3(judgments: list[Judgment]) -> list[Judgment]:
    ranked = sorted(
        (j for j in judgments if _passes_gate(j)),
        key=lambda j: (-(j.relevance + j.usefulness + j.novelty), -j.novelty, -j.usefulness, j.url),
    )
    picks: list[Judgment] = []
    seen_domains: set[str] = set()
    for j in ranked:
        domain = domain_of(j.url)
        if domain in seen_domains:
            continue
        seen_domains.add(domain)
        picks.append(j)
        if len(picks) == 3:
            break
    return picks
