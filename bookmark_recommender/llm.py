import json
from typing import TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(Exception):
    """The model did not return valid JSON for the schema, even after one retry."""


def make_llm(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.MODEL,
        api_key=api_key,
        base_url=config.DEEPSEEK_BASE_URL,
        use_responses_api=True,
        # Keep raw Responses items in AIMessage.content: DeepSeek puts its reasoning in
        # `reasoning_text` items, which the standard "v1" blocks drop.
        output_version="responses/v1",
        reasoning={"effort": config.REASONING_EFFORT},
        timeout=config.LLM_TIMEOUT_S,
    )


def extract_json(text: str) -> str:
    """Return the outermost {...} block, tolerating code fences and surrounding prose."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in model output")
    return text[start : end + 1]


async def call_json(llm, instructions: str, input: str, schema: type[T], *, web_search: bool = False) -> T:
    system = (
        f"{instructions}\n\nRespond with only a JSON object matching this JSON schema:\n"
        f"{json.dumps(schema.model_json_schema())}"
    )
    runnable = llm.bind_tools([{"type": "web_search"}]) if web_search else llm

    prompt = input
    for _ in range(2):
        message = await runnable.ainvoke([("system", system), ("human", prompt)])
        try:
            return schema.model_validate_json(extract_json(message.text))
        except ValueError as e:  # pydantic's ValidationError is a ValueError too
            error = e
            prompt = f"{input}\n\nYour previous reply was invalid: {e}\nReply again with only the JSON object."
    raise LLMOutputError(f"Model returned invalid JSON twice: {error}")
