"""Hard limits on searches, page fetches, LLM calls and wall time for one research run."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import get_settings
from app.schemas.research.dossier import ResearchBudget


class BudgetExhausted(RuntimeError):
    def __init__(self, kind: str) -> None:
        super().__init__(f"research budget exhausted: {kind}")
        self.kind = kind


@dataclass
class Budget:
    max_search_calls: int
    max_fetches: int
    max_llm_calls: int
    max_retries_per_call: int
    max_seconds: int
    max_rounds: int = 3
    max_literature_calls: int = 40
    used_search_calls: int = 0
    used_literature_calls: int = 0
    used_fetches: int = 0
    used_llm_calls: int = 0
    exhausted_reason: str | None = None
    started: float = field(default_factory=time.monotonic)

    @classmethod
    def from_settings(cls) -> "Budget":
        s = get_settings()
        return cls(
            max_search_calls=s.research_max_search_calls,
            max_fetches=s.research_max_fetches,
            max_llm_calls=s.research_max_llm_calls,
            max_retries_per_call=s.research_max_retries_per_call,
            max_seconds=s.research_max_seconds,
            max_rounds=s.research_max_rounds,
            max_literature_calls=s.research_max_literature_calls,
        )

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def _check_time(self) -> None:
        if self.elapsed > self.max_seconds:
            self.exhausted_reason = "time"
            raise BudgetExhausted("time")

    def _charge(self, kind: str, used: str, limit: str) -> None:
        self._check_time()
        if getattr(self, used) >= getattr(self, limit):
            self.exhausted_reason = kind
            raise BudgetExhausted(kind)
        setattr(self, used, getattr(self, used) + 1)

    def charge_search(self) -> None:
        self._charge("search", "used_search_calls", "max_search_calls")

    def charge_fetch(self) -> None:
        self._charge("fetch", "used_fetches", "max_fetches")

    def charge_llm(self) -> None:
        self._charge("llm", "used_llm_calls", "max_llm_calls")

    def charge_literature(self) -> None:
        self._charge("literature", "used_literature_calls", "max_literature_calls")

    def remaining(self, kind: str) -> int:
        return {
            "search": self.max_search_calls - self.used_search_calls,
            "fetch": self.max_fetches - self.used_fetches,
            "llm": self.max_llm_calls - self.used_llm_calls,
            "literature": self.max_literature_calls - self.used_literature_calls,
        }[kind]

    def snapshot(self) -> ResearchBudget:
        return ResearchBudget(
            max_search_calls=self.max_search_calls, max_fetches=self.max_fetches, max_llm_calls=self.max_llm_calls,
            max_retries_per_call=self.max_retries_per_call, max_seconds=self.max_seconds,
            max_literature_calls=self.max_literature_calls,
            used_search_calls=self.used_search_calls, used_fetches=self.used_fetches, used_llm_calls=self.used_llm_calls,
            used_literature_calls=self.used_literature_calls,
            elapsed_seconds=round(self.elapsed, 1), exhausted=self.exhausted_reason is not None,
            exhausted_reason=self.exhausted_reason,
        )
