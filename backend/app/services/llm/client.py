"""One small client for any OpenAI-compatible chat API (OpenAI, Gemini, Groq, Ollama).

Every call asks for a JSON object, validates it against a Pydantic model, and
gives the model one chance to fix invalid JSON. Calls are counted against an
optional budget and spaced out for free-tier rate limits.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LlmError(RuntimeError):
    pass


class CallCounter(Protocol):
    def charge_llm(self) -> None: ...


class JsonLlm(Protocol):
    model: str

    def parse(self, system: str, user: str, schema: type[T], *, counter: CallCounter | None = None) -> T: ...


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    return text


class OpenAICompatibleLlm:
    def __init__(self, *, api_key: str, base_url: str, model: str, timeout: float = 120.0,
                 min_interval: float = 0.0, max_retries: int = 2, client=None) -> None:
        self.model = model
        self.min_interval = min_interval
        self._last = 0.0
        if client is None:
            import openai

            client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self.client = client

    def _wait(self) -> None:
        gap = time.monotonic() - self._last
        if self._last and gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()

    def parse(self, system: str, user: str, schema: type[T], *, counter: CallCounter | None = None) -> T:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{user}\n\nReply with ONE JSON object only, matching this JSON schema:\n{schema_json}"},
        ]
        for attempt in range(2):
            if counter is not None:
                counter.charge_llm()
            self._wait()
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0, response_format={"type": "json_object"},
            )
            content = _strip_fences(response.choices[0].message.content or "")
            try:
                return schema.model_validate_json(content)
            except (ValidationError, ValueError) as exc:
                if attempt:
                    raise LlmError(f"LLM reply does not fit {schema.__name__}: {exc}") from exc
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"That JSON is invalid: {exc}. Return the corrected JSON object only."},
                ]
        raise LlmError("unreachable")


def llm_from_settings() -> OpenAICompatibleLlm | None:
    s = get_settings()
    if not s.effective_llm_key:
        return None
    return OpenAICompatibleLlm(
        api_key=s.effective_llm_key,
        base_url=s.llm_base_url,
        model=s.llm_model,
        timeout=s.llm_timeout_seconds,
        min_interval=s.llm_min_seconds_between_calls,
        max_retries=s.research_max_retries_per_call,
    )
