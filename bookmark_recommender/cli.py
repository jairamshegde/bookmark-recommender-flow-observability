import asyncio
import os
import sys
import uuid
from functools import partial

from dotenv import load_dotenv
from openai import OpenAIError
from openinference.instrumentation import using_session
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt

from . import config
from .crawl import Crawler
from .graph import Deps, TurnError, build_graph
from .llm import LLMOutputError, call_json, make_llm, search_text
from .models import Judgment
from .tracing import phoenix_reachable, setup_tracing

console = Console()

STEP_DONE = {
    "fetch_seed": "Read the seed page",
    "understand_seed": "Understood what it covers",
    "search_candidates": "Found candidate pages",
    "fetch_candidates": "Read candidate pages",
    "score_candidate": "Scored a candidate",
    "select_top3": "Picked the top 3",
}
NEXT_STEP = {
    "fetch_seed": "Understanding the seed…",
    "understand_seed": "Searching the web…",
    "search_candidates": "Reading candidate pages…",
    "fetch_candidates": "Scoring candidates…",
}


def render_pick(rank: int, j: Judgment) -> Panel:
    body = (
        f"[link={j.url}]{escape(j.url)}[/link]\n\n"
        f"[b]Relevance {j.relevance}[/b]  {escape(j.relevance_rationale)}\n"
        f"[b]Usefulness {j.usefulness}[/b]  {escape(j.usefulness_rationale)}\n"
        f"[b]Novelty {j.novelty}[/b]  {escape(j.novelty_rationale)}"
    )
    if j.content_source == "snippet":
        body += "\n\n[dim]Judged from the search snippet: the page could not be fetched.[/dim]"
    return Panel(body, title=f"#{rank} {escape(j.title)}", subtitle=f"R{j.relevance} U{j.usefulness} N{j.novelty}")


async def run_turn(app, seed_url: str, reason: str, session_id: str) -> list[Judgment]:
    picks: list[Judgment] = []
    with using_session(session_id), console.status("Reading the seed page…") as status:
        async for update in app.astream({"seed_url": seed_url, "reason": reason}, stream_mode="updates"):
            for node, output in update.items():
                console.print(f"  [green]✓[/green] {STEP_DONE.get(node, node)}")
                if node in NEXT_STEP:
                    status.update(NEXT_STEP[node])
                if node == "select_top3":
                    picks = output["picks"]
    return picks


async def main() -> int:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        console.print("[red]DEEPSEEK_API_KEY is not set in .env[/red]")
        return 1

    tracer_provider = setup_tracing()
    session_id = str(uuid.uuid4())
    phoenix = f"{config.PHOENIX_URL}" if phoenix_reachable() else f"[yellow]{config.PHOENIX_URL} not reachable: run `phoenix serve`; traces will be lost[/yellow]"
    console.print(Panel(
        f"Model: {config.MODEL} (reasoning effort: {config.REASONING_EFFORT})\nSession: {session_id}\nPhoenix: {phoenix}",
        title="Bookmark recommender",
    ))

    llm = make_llm(api_key)
    try:
        async with Crawler() as crawler:
            app = build_graph(Deps(
                call_json=partial(call_json, llm), search_text=partial(search_text, llm),
                fetch_page=crawler.fetch_page, fetch_many=crawler.fetch_many,
            ))
            while True:
                seed_url = Prompt.ask("\n[b]Seed URL[/b] (empty or q to quit)", default="", show_default=False).strip()
                if seed_url in ("", "q"):
                    return 0
                if not seed_url.startswith(("http://", "https://")):
                    console.print("[red]The URL must start with http:// or https://[/red]")
                    continue
                reason = ""
                while not reason:
                    reason = Prompt.ask("[b]Why is it important to you?[/b]").strip()
                try:
                    picks = await run_turn(app, seed_url, reason, session_id)
                except (TurnError, LLMOutputError, OpenAIError) as e:
                    console.print(f"[red]{escape(str(e))}[/red]")
                    continue
                for rank, j in enumerate(picks, 1):
                    console.print(render_pick(rank, j))
                if len(picks) < 3:
                    console.print(f"[yellow]Only {len(picks)} candidate(s) passed the relevance, usefulness and novelty bar.[/yellow]")
    finally:
        tracer_provider.shutdown()


def run() -> None:
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        console.print("\nBye.")
