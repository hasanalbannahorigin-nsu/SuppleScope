"""A used-up Gemini daily quota stops at once with a short message (laptop, 2026-11-28: four minutes of
15/30/60/120 s retries, then a 2 KB error dump in the Analysis status card)."""
import httpx
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.services import pipeline_v2
from app.services.llm import client as llm_client
from app.services.llm.client import GeminiNativeClient, OpenAICompatibleLlm, QuotaExhausted, daily_quota_message
from app.services.ocr.document import ocr_document_from_text
from app.services.research_agent.runner import run_research

from tests.test_research_agent_v2 import AD_TEXT, FAKE_LITERATURE, FakeLlm, FakeSearch, _ad, _budget, fake_fetch

DAILY = ("Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan "
         "and billing details.\\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_"
         "tier_requests, limit: 500, model: gemini-3.5-flash-lite, description: Generate content requests per day limit "
         "exceeded on Free Tier.\\nPlease retry in 9h28m15.947288347s.', 'status': 'RESOURCE_EXHAUSTED', 'details': "
         "[{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaId': "
         "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', "
         "'retryDelay': '34095s'}]}}]")
PER_MINUTE = ("Error code: 429 - Quota exceeded for metric: generate_content_free_tier_requests, limit: 15, "
              "quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier, retryDelay: '21s'")


class Out(BaseModel):
    name: str


class _RateLimited(Exception):
    status_code = 429


class _Client:
    def __init__(self, text: str, fails: int = 99) -> None:
        self.text, self.fails, self.calls = text, fails, 0
        self.chat = self.completions = self

    def create(self, **_):
        self.calls += 1
        if self.calls <= self.fails:
            raise _RateLimited(self.text)
        return llm_client._Reply('{"name": "ok"}')


def test_daily_quota_message_is_short_and_says_when():
    msg = daily_quota_message(DAILY)
    assert msg == ("Gemini daily request limit used up for gemini-3.5-flash-lite (500 requests a day); it resets in "
                   "about 9h28m. Try again then, or set another LLM_MODEL in .env.")
    assert daily_quota_message(PER_MINUTE) is None


def test_daily_quota_fails_at_once(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: pytest.fail(f"waited {s}s for a daily limit"))
    client = _Client(DAILY)
    with pytest.raises(QuotaExhausted, match="9h28m"):
        OpenAICompatibleLlm(api_key="", base_url="", model="m", client=client).parse("s", "u", Out)
    assert client.calls == 1


def test_per_minute_limit_still_waits_and_retries(monkeypatch):
    waits = []
    monkeypatch.setattr(llm_client.time, "sleep", waits.append)
    client = _Client(PER_MINUTE, fails=1)
    assert OpenAICompatibleLlm(api_key="", base_url="", model="m", client=client).parse("s", "u", Out).name == "ok"
    assert waits == [15] and client.calls == 2


def test_native_gemini_client_fails_at_once_too():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, text=DAILY)

    native = GeminiNativeClient(api_key="AQ.test", http=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(QuotaExhausted):
        OpenAICompatibleLlm(api_key="", base_url="", model="gemini-flash-lite-latest", client=native).parse("s", "u", Out)
    assert len(calls) == 1


class _QuotaAfter(FakeLlm):
    """The scripted LLM, but the daily limit runs out after `ok` calls."""

    def __init__(self, ok: int) -> None:
        super().__init__()
        self.ok, self.n = ok, 0

    def parse(self, system, user, schema, *, counter=None):
        self.n += 1
        if self.n > self.ok:
            raise QuotaExhausted(daily_quota_message(DAILY))
        return super().parse(system, user, schema, counter=counter)


def test_research_stops_and_says_why_when_quota_runs_out():
    llm = _QuotaAfter(ok=3)  # product identified, then the limit runs out before the evidence plan
    d = run_research(_ad(), llm=llm, search=FakeSearch(), fetcher=fake_fetch, literature=FAKE_LITERATURE,
                     budget=_budget(), use_cache=False)
    assert d.product.identity.status.value == "VERIFIED"
    assert d.budget.exhausted and d.budget.exhausted_reason == "llm_daily_quota"
    assert any("daily request limit" in u and "9h28m" in u for u in d.uncertainties)
    assert llm.n == 4  # no further calls once the limit is known
    assert d.ingredients and all(i.status.value == "NOT_SEARCHED" for i in d.ingredients)
    assert all(c.status.value == "NOT_SEARCHED" for c in d.claims)


def test_extraction_note_is_short_when_quota_is_gone(monkeypatch):
    settings = Settings(_env_file=None, extraction_backend="grounded_llm", llm_api_key="fake")
    monkeypatch.setattr(pipeline_v2, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline_v2, "llm_from_settings", lambda: _QuotaAfter(ok=0))
    from app.services.extraction.baseline import extract_from_text

    doc = ocr_document_from_text("ad_q", AD_TEXT)
    ex, note = pipeline_v2.extract_ad(doc, extract_from_text(ocr_text=AD_TEXT))
    assert ex.extractor.name != "grounded_llm"
    assert note.startswith("grounded_llm failed (Gemini daily request limit used up") and len(note) < 250
