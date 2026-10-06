import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from bookmark_recommender.llm import LLMOutputError, call_json, extract_json, make_llm, search_text


class Thing(BaseModel):
    name: str
    size: int


class FakeLLM:
    """Stands in for ChatOpenAI: returns the queued reply texts in order."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.tools = None

    def bind_tools(self, tools):
        self.tools = tools
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(text=self.outputs.pop(0))


def call(llm, **kwargs):
    return asyncio.run(call_json(llm, "Describe a thing.", "the input", Thing, **kwargs))


def test_make_llm_targets_deepseek_responses_api():
    llm = make_llm("test-key")
    assert llm.model_name == "deepseek-v4-pro"
    assert llm.openai_api_base == "https://api.deepseek.com"
    assert llm.use_responses_api is True
    assert llm.output_version == "responses/v1"
    assert llm.reasoning == {"effort": "low"}


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Sure! Here it is: {"a": {"b": 2}} Hope that helps.') == '{"a": {"b": 2}}'


def test_extract_json_raises_without_object():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_call_json_returns_validated_model():
    llm = FakeLLM('{"name": "box", "size": 3}')
    assert call(llm) == Thing(name="box", size=3)
    (system_role, system), (human_role, human) = llm.calls[0]
    assert (system_role, human_role, human) == ("system", "human", "the input")
    assert system.startswith("Describe a thing.")
    assert '"size"' in system  # schema is appended to the instructions
    assert llm.tools is None


def test_search_text_binds_web_search_and_returns_plain_text():
    llm = FakeLLM("Here are some pages: https://a.com")
    text = asyncio.run(search_text(llm, "Find pages.", "the input"))
    assert text == "Here are some pages: https://a.com"
    assert llm.tools == [{"type": "web_search"}]
    assert llm.calls == [[("system", "Find pages."), ("human", "the input")]]  # no JSON schema, no retry


def test_call_json_retries_once_with_the_error():
    llm = FakeLLM('{"name": "box"}', '{"name": "box", "size": 3}')
    assert call(llm) == Thing(name="box", size=3)
    assert len(llm.calls) == 2
    retry_prompt = llm.calls[1][1][1]
    assert retry_prompt.startswith("the input")
    assert "size" in retry_prompt  # the validation error is fed back


def test_call_json_raises_after_second_failure():
    llm = FakeLLM("nope", "still nope")
    with pytest.raises(LLMOutputError):
        call(llm)
    assert len(llm.calls) == 2
