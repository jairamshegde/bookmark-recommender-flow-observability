from rich.console import Console

from bookmark_recommender.cli import render_pick
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
