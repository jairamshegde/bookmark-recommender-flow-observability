from bookmark_recommender.models import Judgment
from bookmark_recommender.ranking import domain_of, normalize_url, select_top3


def j(url, r=4, u=4, n=4):
    return Judgment(
        url=url, title=url, content_source="page",
        relevance=r, usefulness=u, novelty=n,
        relevance_rationale="", usefulness_rationale="", novelty_rationale="",
    )


def urls(judgments):
    return [x.url for x in judgments]


def test_normalize_url_lowercases_host_and_drops_trailing_slash_and_fragment():
    assert normalize_url("HTTPS://Example.COM/a/b/#frag") == "https://example.com/a/b"
    assert normalize_url("https://example.com/") == "https://example.com"
    assert normalize_url("https://example.com/a?q=1") == "https://example.com/a?q=1"


def test_domain_of_strips_www_and_case():
    assert domain_of("https://www.Example.com/x") == "example.com"
    assert domain_of("https://docs.example.com/x") == "docs.example.com"


def test_gate_drops_low_scores():
    judged = [j("https://a.com", r=2), j("https://b.com", u=2), j("https://c.com", n=1), j("https://d.com", 3, 3, 2)]
    assert urls(select_top3(judged)) == ["https://d.com"]


def test_ranks_by_total_score():
    judged = [j("https://a.com", 3, 3, 3), j("https://b.com", 5, 5, 5), j("https://c.com", 4, 4, 4)]
    assert urls(select_top3(judged)) == ["https://b.com", "https://c.com", "https://a.com"]


def test_ties_prefer_novelty_then_usefulness_then_url():
    judged = [
        j("https://a.com", r=5, u=4, n=3),
        j("https://b.com", r=3, u=4, n=5),
        j("https://c.com", r=4, u=5, n=3),
    ]
    assert urls(select_top3(judged)) == ["https://b.com", "https://c.com", "https://a.com"]
    assert urls(select_top3([j("https://z.com"), j("https://y.com")])) == ["https://y.com", "https://z.com"]


def test_one_pick_per_domain():
    judged = [j("https://www.x.com/1", 5, 5, 5), j("https://x.com/2", 4, 4, 4), j("https://y.com", 3, 3, 3)]
    assert urls(select_top3(judged)) == ["https://www.x.com/1", "https://y.com"]


def test_returns_at_most_three():
    judged = [j(f"https://{c}.com") for c in "abcde"]
    assert len(select_top3(judged)) == 3


def test_empty_input_gives_no_picks():
    assert select_top3([]) == []
