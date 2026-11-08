"""Fetch a web page or PDF and turn it into plain text the agent can quote from."""
from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser

import httpx

from app.services.research.http import USER_AGENT

logger = logging.getLogger(__name__)

_SKIP = {"script", "style", "noscript", "svg", "template", "iframe", "head"}
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table", "ul", "ol", "dt", "dd"}


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK or tag in {"td", "th"}:
            self.parts.append("\n" if tag not in {"td", "th"} else " | ")

    def handle_endtag(self, tag):
        if tag in _SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextParser()
    parser.feed(html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return parser.title.strip(), text.strip()


def pdf_to_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages[:30]).strip()


@dataclass
class FetchedDoc:
    url: str
    final_url: str
    title: str
    text: str
    content_type: str
    sha256: str
    status: int


def fetch_document(url: str, *, max_chars: int = 60000, timeout: float = 25.0) -> FetchedDoc:
    with httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(url)
        response.raise_for_status()
    ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype == "application/pdf" or url.lower().endswith(".pdf"):
        title, text = "", pdf_to_text(response.content)
    else:
        title, text = html_to_text(response.text)
    text = text[:max_chars]
    return FetchedDoc(url=url, final_url=str(response.url), title=title, text=text, content_type=ctype,
                      sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), status=response.status_code)


def doc_to_json(doc: FetchedDoc) -> dict:
    return asdict(doc)


def doc_from_json(row: dict) -> FetchedDoc:
    return FetchedDoc(**row)
