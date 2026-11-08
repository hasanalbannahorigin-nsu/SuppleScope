import json

import httpx
from pydantic import BaseModel

from app.services.llm.client import GeminiNativeClient, OpenAICompatibleLlm


class Out(BaseModel):
    name: str


def test_gemini_native_client_sends_key_header_and_parses_reply():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"name": "Dibedex"}'}]}}]})

    client = GeminiNativeClient(api_key="AQ.test", http=httpx.Client(transport=httpx.MockTransport(handler)))
    llm = OpenAICompatibleLlm(api_key="", base_url="", model="gemini-flash-lite-latest", client=client)
    assert llm.parse("sys", "user", Out).name == "Dibedex"
    assert seen["url"].endswith("/models/gemini-flash-lite-latest:generateContent")
    assert seen["key"] == "AQ.test" and seen["auth"] is None
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"


class _Busy(Exception):
    status_code = 503


class _FlakyClient:
    def __init__(self, fails: int) -> None:
        self.fails = fails
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **_):
        self.calls += 1
        if self.calls <= self.fails:
            raise _Busy("high demand")
        from app.services.llm.client import _Reply
        return _Reply('{"name": "ok"}')


def test_parse_waits_and_retries_when_model_busy(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleLlm, "BUSY_WAITS", (0, 0))
    client = _FlakyClient(fails=2)
    llm = OpenAICompatibleLlm(api_key="", base_url="", model="m", client=client)
    assert llm.parse("s", "u", Out).name == "ok"
    assert client.calls == 3


def test_parse_gives_up_after_busy_waits(monkeypatch):
    import pytest

    monkeypatch.setattr(OpenAICompatibleLlm, "BUSY_WAITS", (0,))
    llm = OpenAICompatibleLlm(api_key="", base_url="", model="m", client=_FlakyClient(fails=5))
    with pytest.raises(_Busy):
        llm.parse("s", "u", Out)
