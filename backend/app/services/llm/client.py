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


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Reply:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class GeminiNativeClient:
    """Calls Gemini's own generateContent endpoint with an x-goog-api-key header.

    Newer "AQ." Gemini keys are not always accepted by the OpenAI-compatible
    endpoint (Bearer auth), but they work on the native one. This class looks
    like ``openai.OpenAI`` from the outside (``client.chat.completions.create``)
    so OpenAICompatibleLlm can use it unchanged. Enable with LLM_PROVIDER=gemini.
    """

    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, *, api_key: str, timeout: float = 120.0, max_retries: int = 2, http=None) -> None:
        import httpx

        self.api_key = api_key
        self.max_retries = max_retries
        self.http = http or httpx.Client(timeout=timeout)
        self.chat = self
        self.completions = self

    def create(self, *, model: str, messages: list[dict], temperature: float = 0, response_format=None, **_) -> _Reply:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        body: dict = {"contents": contents, "generationConfig": {"temperature": temperature}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if response_format and response_format.get("type") == "json_object":
            body["generationConfig"]["responseMimeType"] = "application/json"
        name = model.removeprefix("models/")
        url = f"{self.BASE}/models/{name}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        for attempt in range(self.max_retries + 1):
            r = self.http.post(url, json=body, headers=headers)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            if r.status_code >= 400:
                raise LlmError(f"Gemini API error {r.status_code}: {r.text[:500]}")
            data = r.json()
            try:
                parts = data["candidates"][0]["content"]["parts"]
            except (KeyError, IndexError) as exc:
                raise LlmError(f"Gemini returned no text: {json.dumps(data)[:500]}") from exc
            return _Reply("".join(p.get("text", "") for p in parts))
        raise LlmError("Gemini API kept failing")


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
        client=(
            GeminiNativeClient(api_key=s.effective_llm_key, timeout=s.llm_timeout_seconds,
                               max_retries=s.research_max_retries_per_call)
            if s.llm_provider.lower() == "gemini" else None
        ),
        model=s.llm_model,
        timeout=s.llm_timeout_seconds,
        min_interval=s.llm_min_seconds_between_calls,
        max_retries=s.research_max_retries_per_call,
    )
