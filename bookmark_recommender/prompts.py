from .models import Candidate, PageContent, SeedProfile

UNDERSTAND_SEED = """You analyse a web page that a user bookmarked, together with their reason for saving it.
Fill in:
- topic: the page's subject in one line.
- covers: the main points the page covers.
- does_not_cover: gaps, deeper topics or next steps a reader with this intent would want that the page does NOT cover.
- likely_intent: what the user is most likely trying to achieve, given their reason."""


def understand_seed_input(page: PageContent, reason: str) -> str:
    return (
        f"Bookmarked URL: {page.url}\nTitle: {page.title}\nUser's reason: {reason}\n\n"
        f"Page content (markdown, may be truncated):\n{page.markdown}"
    )


SEARCH = """You are a web search specialist. Find web pages that add incremental value for this user beyond the page they bookmarked.
Use the web_search tool more than once, with different phrasings. Prioritise the topics in does_not_cover.
Return up to 8 candidates. Only include URLs you actually saw in search results, and never the bookmarked URL itself.
For each candidate give: url, title, snippet (what the page contains, one or two sentences), why_found (which gap or intent it serves)."""


def search_input(seed_url: str, profile: SeedProfile) -> str:
    return f"Bookmarked URL: {seed_url}\n\nProfile of the bookmark:\n{profile.model_dump_json(indent=2)}"


EXTRACT_CANDIDATES = """Below is a web researcher's answer listing web pages it found.
Extract every listed page as a candidate: url, title, snippet (what the page contains), why_found (which gap or intent it serves).
Only use URLs that appear in the text. Do not add, guess or change URLs."""


JUDGE = """You judge whether a candidate web page is worth recommending to a user, relative to a page they already bookmarked.
Score three criteria from 1 to 5. For each criterion write the rationale (one sentence) first, then the score.

Relevance: How closely is this bookmark related to the given bookmark?
  5 = same topic, directly builds on the seed. 3 = same broad area, different subtopic. 1 = unrelated.
Usefulness: How valuable is this bookmark to the user's likely intent?
  5 = directly advances the intent: substantive, credible, actionable. 3 = somewhat helpful, but shallow or only partly on-intent. 1 = no real value (SEO filler, a stub, a dead page).
Novelty: Does it cover something the seed page does not cover?
  5 = mostly new: a different angle, more depth, or a next step from does_not_cover. 3 = partly overlaps the seed, with some new material. 1 = restates the seed.

Use 2 and 4 for cases between the anchors. Judge only from the content provided."""


def judge_input(profile: SeedProfile, candidate: Candidate, content: str, source: str) -> str:
    return (
        f"Profile of the bookmarked page:\n{profile.model_dump_json(indent=2)}\n\n"
        f"Candidate URL: {candidate.url}\nCandidate title: {candidate.title}\n"
        f"Candidate content ({'full page, may be truncated' if source == 'page' else 'search snippet only'}):\n{content}"
    )
