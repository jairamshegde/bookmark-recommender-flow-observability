import asyncio

import pytest

from bookmark_recommender.graph import Deps, TurnError, build_graph
from bookmark_recommender.llm import LLMOutputError
from bookmark_recommender.models import Candidate, CandidateList, PageContent, SeedProfile, Verdict

SEED = "https://seed.com/post"
PROFILE = SeedProfile(topic="t", covers=["x"], does_not_cover=["y"], likely_intent="learn")
SEARCH_ANSWER = "Here are pages I found: ..."
A, B, C, D = "https://a.com/1", "https://b.com/2", "https://c.com/3", "https://d.com/4"


def cand(url):
    return Candidate(url=url, title=f"title {url}", snippet=f"snippet {url}", why_found="gap")


def verdict(score):
    return Verdict(
        relevance_rationale="r", relevance=score,
        usefulness_rationale="u", usefulness=score,
        novelty_rationale="n", novelty=score,
    )


class Fakes:
    def __init__(self, candidates, scores, seed_ok=True, failed_fetches=(), failing_judges=(), search_answer=SEARCH_ANSWER,
                 seed_error="timeout"):
        self.candidates, self.scores, self.seed_ok, self.seed_error = candidates, scores, seed_ok, seed_error
        self.failed_fetches, self.failing_judges = set(failed_fetches), set(failing_judges)
        self.search_answer = search_answer
        self.extract_inputs = []
        self.judge_inputs = []

    async def fetch_page(self, url, limit):
        if url == SEED:
            return PageContent(url=url, title="Seed", markdown="seed text", ok=self.seed_ok,
                               error=None if self.seed_ok else self.seed_error)
        if url in self.failed_fetches:
            return PageContent(url=url, ok=False, error="403")
        return PageContent(url=url, title="page", markdown=f"content of {url}", ok=True)

    async def fetch_many(self, urls, limit):
        return [await self.fetch_page(u, limit) for u in urls]

    async def search_text(self, instructions, input):
        return self.search_answer

    async def call_json(self, instructions, input, schema):
        if schema is SeedProfile:
            return PROFILE
        if schema is CandidateList:
            self.extract_inputs.append(input)
            return CandidateList(candidates=self.candidates)
        self.judge_inputs.append(input)
        url = next(u for u in self.scores if u in input)
        if url in self.failing_judges:
            raise LLMOutputError("bad json")
        return verdict(self.scores[url])


def run(fakes):
    deps = Deps(call_json=fakes.call_json, search_text=fakes.search_text,
                fetch_page=fakes.fetch_page, fetch_many=fakes.fetch_many)
    return asyncio.run(build_graph(deps).ainvoke({"seed_url": SEED, "reason": "because"}))


def picked(state):
    return [j.url for j in state["picks"]]


SCORES = {A: 5, B: 4, C: 3, D: 2}


def test_happy_path_picks_top_three():
    state = run(Fakes([cand(u) for u in (A, B, C, D)], SCORES))
    assert picked(state) == [A, B, C]
    assert state["picks"][0].content_source == "page"


def test_filters_seed_duplicates_and_non_http_candidates():
    weird = [cand("HTTPS://SEED.com/post/"), cand(A + "#section"), cand("ftp://e.com/x"), cand("e.com/no-scheme")]
    fakes = Fakes([cand(u) for u in (A, B, C, D)] + weird, SCORES)
    run(fakes)
    assert len(fakes.judge_inputs) == 4


def test_search_answer_is_structured_by_a_separate_call():
    fakes = Fakes([cand(u) for u in (A, B, C)], SCORES)
    run(fakes)
    assert fakes.extract_inputs == [SEARCH_ANSWER]


def test_empty_search_answer_ends_turn():
    fakes = Fakes([cand(A)], SCORES, search_answer="  ")
    with pytest.raises(TurnError, match="no answer"):
        run(fakes)
    assert fakes.extract_inputs == []


def test_seed_fetch_failure_ends_turn():
    fakes = Fakes([cand(A)], SCORES, seed_ok=False)
    with pytest.raises(TurnError, match="timeout"):
        run(fakes)
    assert fakes.judge_inputs == []


CRAWL4AI_ERROR = (
    "Unexpected error in _crawl_web at line 778 in _crawl_web (../site-packages/crawl4ai/async_crawler_strategy.py):\n"
    "Error: Failed on navigating ACS-GOTO:\n"
    "Page.goto: net::ERR_NAME_NOT_RESOLVED at https://nonexistent.invalid-tld-xyz/\n"
    "Code context:\n 778 →     raise RuntimeError(...)"
)


def test_seed_fetch_error_shows_only_the_network_code():
    with pytest.raises(TurnError) as err:
        run(Fakes([cand(A)], SCORES, seed_ok=False, seed_error=CRAWL4AI_ERROR))
    assert str(err.value) == "Could not read the seed page: net::ERR_NAME_NOT_RESOLVED"


def test_seed_fetch_error_without_code_shows_first_line():
    with pytest.raises(TurnError) as err:
        run(Fakes([cand(A)], SCORES, seed_ok=False, seed_error="page has no text\nmore detail"))
    assert str(err.value) == "Could not read the seed page: page has no text"


def test_no_candidates_ends_turn():
    with pytest.raises(TurnError):
        run(Fakes([cand(SEED + "/")], SCORES))


def test_failed_candidate_fetch_falls_back_to_snippet():
    fakes = Fakes([cand(u) for u in (A, B, C)], SCORES, failed_fetches={A})
    state = run(fakes)
    assert state["picks"][0].url == A and state["picks"][0].content_source == "snippet"
    assert any(f"snippet {A}" in i for i in fakes.judge_inputs)


def test_one_failing_judge_is_dropped():
    state = run(Fakes([cand(u) for u in (A, B, C)], SCORES, failing_judges={A}))
    assert picked(state) == [B, C]


def test_all_judges_failing_gives_no_picks():
    state = run(Fakes([cand(u) for u in (A, B)], SCORES, failing_judges={A, B}))
    assert state["picks"] == []
