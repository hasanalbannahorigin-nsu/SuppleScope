from __future__ import annotations

import json
import logging
import re

from app.agents.prompts import SYSTEM_PROMPT
from app.agents.tools import RESEARCH_TOOLS
from app.config import get_settings
from app.schemas.dossier import ResearchDossier

logger = logging.getLogger(__name__)


def llm_configured() -> bool:
    settings = get_settings()
    return bool(settings.effective_llm_key)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def run_research_agent(brief: dict) -> ResearchDossier:
    if not llm_configured():
        raise RuntimeError(
            "LLM_API_KEY or OPENAI_API_KEY is required for the agentic research loop. "
            "A non-LLM literature fallback can still run separately."
        )
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    settings = get_settings()
    model = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.effective_llm_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )
    agent = create_react_agent(model, RESEARCH_TOOLS)
    user_message = (
        "Research this advertised supplement. Investigate identity, composition not "
        "shown in the ad, ingredient properties, real literature, interactions, and "
        "disease risks. Use multiple tools if needed.\n\n"
        + json.dumps(brief, ensure_ascii=False, indent=2)
    )
    result = agent.invoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        }
    )
    messages = result.get("messages") or []
    content = messages[-1].content if messages else "{}"
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    payload = _extract_json(str(content))
    return ResearchDossier.model_validate(payload)
