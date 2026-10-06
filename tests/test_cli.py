import asyncio

from rich.console import Console

from bookmark_recommender import cli
from bookmark_recommender.cli import read_turn, render_pick, show_turn
from bookmark_recommender.models import Judgment


def test_render_pick_shows_model_text_literally():
    j = Judgment(
        url="https://a.com/x", title="[bold]Odd[/bold] title", content_source="snippet",
        relevance=5, usefulness=4, novelty=3,
        relevance_rationale="uses [x] notation", usefulness_rationale="u [/] text", novelty_rationale="n",
    )
    console = Console(record=True, width=120)
    console.print(render_pick(1, j))
    text = console.export_text()
    assert "[bold]Odd[/bold] title" in text
    assert "uses [x] notation" in text and "u [/] text" in text
    assert "https://a.com/x" in text
    assert "R5 U4 N3" in text
    assert "snippet" in text


def test_render_pick_handles_brackets_in_url():
    j = Judgment(
        url="https://example.com/a?f[tag]=x]", title="t", content_source="page",
        relevance=5, usefulness=4, novelty=3,
        relevance_rationale="r", usefulness_rationale="u", novelty_rationale="n",
    )
    console = Console(record=True, width=120)
    console.print(render_pick(1, j))
    assert "https://example.com/a?f[tag]=x]" in console.export_text()


def answers(monkeypatch, *replies):
    queue = list(replies)

    def ask(*args, **kwargs):
        reply = queue.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr(cli.Prompt, "ask", ask)


def test_read_turn_quits_on_ctrl_d(monkeypatch):
    answers(monkeypatch, EOFError())
    assert read_turn() is None


def test_read_turn_reasks_until_url_and_reason_are_valid(monkeypatch):
    answers(monkeypatch, "ftp://x.com", "https://a.com", "", "because")
    assert read_turn() == ("https://a.com", "because")


class BrokenApp:
    async def astream(self, *args, **kwargs):
        raise RuntimeError("boom")
        yield


def test_show_turn_reports_unexpected_errors_and_keeps_session(monkeypatch):
    console = Console(record=True, width=120)
    monkeypatch.setattr(cli, "console", console)
    asyncio.run(show_turn(BrokenApp(), "https://a.com", "because", "session-1"))
    assert "Unexpected error: boom" in console.export_text()
