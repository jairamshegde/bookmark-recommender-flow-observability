import operator
from dataclasses import dataclass
from typing import Annotated, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from openai import OpenAIError

from . import config, prompts
from .llm import LLMOutputError
from .models import Candidate, CandidateList, Judgment, PageContent, SeedProfile, Verdict
from .ranking import normalize_url, select_top3
from .tracing import node_span


class TurnError(Exception):
    """Ends the current turn with a message for the user."""


@dataclass
class Deps:
    call_json: Callable[..., Awaitable]
    fetch_page: Callable[[str, int], Awaitable[PageContent]]
    fetch_many: Callable[[list[str], int], Awaitable[list[PageContent]]]


class State(TypedDict, total=False):
    seed_url: str
    reason: str
    seed_page: PageContent
    seed_profile: SeedProfile
    candidates: list[Candidate]
    candidate_pages: dict[str, PageContent]
    judgments: Annotated[list[Judgment], operator.add]  # merged across the parallel judge branches
    picks: list[Judgment]


class ScoreInput(TypedDict):
    seed_profile: SeedProfile
    candidate: Candidate
    page: PageContent


def build_graph(deps: Deps):
    async def fetch_seed(state: State):
        with node_span():
            page = await deps.fetch_page(state["seed_url"], config.SEED_CHAR_LIMIT)
        if not page.ok:
            raise TurnError(f"Could not read the seed page: {page.error}")
        return {"seed_page": page}

    async def understand_seed(state: State):
        profile = await deps.call_json(
            prompts.UNDERSTAND_SEED, prompts.understand_seed_input(state["seed_page"], state["reason"]), SeedProfile
        )
        return {"seed_profile": profile}

    async def search_candidates(state: State):
        found = await deps.call_json(
            prompts.SEARCH, prompts.search_input(state["seed_url"], state["seed_profile"]), CandidateList,
            web_search=True,
        )
        seen = {normalize_url(state["seed_url"])}
        candidates = []
        for c in found.candidates:
            key = normalize_url(c.url)
            if not c.url.startswith(("http://", "https://")) or key in seen:
                continue
            seen.add(key)
            candidates.append(c)
        if not candidates:
            raise TurnError("Search found no candidate pages other than the seed.")
        return {"candidates": candidates[: config.MAX_CANDIDATES]}

    async def fetch_candidates(state: State):
        urls = [c.url for c in state["candidates"]]
        with node_span():
            pages = await deps.fetch_many(urls, config.CANDIDATE_CHAR_LIMIT)
        return {"candidate_pages": dict(zip(urls, pages))}

    def dispatch_scoring(state: State):
        return [
            Send("score_candidate", {"seed_profile": state["seed_profile"], "candidate": c, "page": state["candidate_pages"][c.url]})
            for c in state["candidates"]
        ]

    async def score_candidate(state: ScoreInput):
        candidate, page = state["candidate"], state["page"]
        source = "page" if page.ok else "snippet"
        content = page.markdown if page.ok else candidate.snippet
        with node_span() as span:  # used here only to get the node span for the judgment.* attributes
            try:
                verdict = await deps.call_json(
                    prompts.JUDGE, prompts.judge_input(state["seed_profile"], candidate, content, source), Verdict
                )
            except (LLMOutputError, OpenAIError) as e:  # one bad judge call must not sink the turn
                span.record_exception(e)
                span.set_attribute("judgment.error", str(e))
                return {"judgments": []}
            judgment = Judgment(url=candidate.url, title=candidate.title, content_source=source, **verdict.model_dump())
            span.set_attributes({
                "judgment.url": judgment.url,
                "judgment.relevance": judgment.relevance,
                "judgment.usefulness": judgment.usefulness,
                "judgment.novelty": judgment.novelty,
                "judgment.content_source": source,
            })
        return {"judgments": [judgment]}

    async def select_top3_node(state: State):
        return {"picks": select_top3(state.get("judgments", []))}

    graph = StateGraph(State)
    graph.add_node("fetch_seed", fetch_seed)
    graph.add_node("understand_seed", understand_seed)
    graph.add_node("search_candidates", search_candidates)
    graph.add_node("fetch_candidates", fetch_candidates)
    graph.add_node("score_candidate", score_candidate)
    graph.add_node("select_top3", select_top3_node)
    graph.add_edge(START, "fetch_seed")
    graph.add_edge("fetch_seed", "understand_seed")
    graph.add_edge("understand_seed", "search_candidates")
    graph.add_edge("search_candidates", "fetch_candidates")
    graph.add_conditional_edges("fetch_candidates", dispatch_scoring, ["score_candidate"])
    graph.add_edge("score_candidate", "select_top3")
    graph.add_edge("select_top3", END)
    return graph.compile()
