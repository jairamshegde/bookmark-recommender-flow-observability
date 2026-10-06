from typing import Literal

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    url: str
    title: str = ""
    markdown: str = ""
    ok: bool
    error: str | None = None


class SeedProfile(BaseModel):
    topic: str
    covers: list[str]
    does_not_cover: list[str]
    likely_intent: str


class Candidate(BaseModel):
    url: str
    title: str
    snippet: str = ""
    why_found: str = ""


class CandidateList(BaseModel):
    candidates: list[Candidate]


class Verdict(BaseModel):
    """What the judge returns. Rationale fields come before scores on purpose:
    the JSON schema keeps this order, so the model reasons before it scores."""

    relevance_rationale: str
    relevance: int = Field(ge=1, le=5)
    usefulness_rationale: str
    usefulness: int = Field(ge=1, le=5)
    novelty_rationale: str
    novelty: int = Field(ge=1, le=5)


class Judgment(Verdict):
    url: str
    title: str
    content_source: Literal["page", "snippet"]
