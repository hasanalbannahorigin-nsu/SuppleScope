"""Top-k evidence sentences per claim, from the abstracts and records the research agent retrieved.

Follows the two best-performing designs among the related studies:
- MelAnalyze (Karkera et al. 2026): split abstracts into sentences and keep the top 5 most similar to the
  claim (SBERT) as the premise for the NLI verifier.
- CliVER (Liu et al. 2024): a first-stage retriever followed by a cross-encoder / T5 re-ranker.

Stage 1 is a hybrid score: BM25 over sentences (always available) plus cosine similarity from a
biomedical bi-encoder (EVIDENCE_EMBEDDING_MODEL, default pritamdeka/S-PubMedBert-MS-MARCO) when
sentence-transformers is installed. Stage 2 re-ranks the best 20 with a cross-encoder
(EVIDENCE_RERANKER_MODEL, default cross-encoder/ms-marco-MiniLM-L-6-v2; ncbi/MedCPT-Cross-Encoder is a
biomedical alternative). Without the ML extras (requirements-ml.txt) BM25 alone is used and recorded as
the retriever, so results stay reproducible on a laptop with no GPU.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

STOP = set("""a an the of in on for to and or with without by from as at is are was were be been being this that these
those it its into than then there their they we our you your not no nor but if so such can could may might will would
should do does did done have has had having also more most less least very over under between among per each both all
any some study studies patients participants group groups results result conclusion conclusions background methods
objective objectives aim aims""".split())
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n+")
_LABEL = re.compile(r"^(?:BACKGROUND|OBJECTIVES?|AIMS?|METHODS?|RESULTS?|CONCLUSIONS?|INTRODUCTION|DISCUSSION|"
                    r"FINDINGS|INTERPRETATION|PURPOSE|DESIGN|SETTING|PARTICIPANTS|INTERVENTIONS?|MEASUREMENTS?)\s*:\s*", re.I)


@dataclass
class Doc:
    doc_id: str  # evidence_id
    source_id: str
    text: str
    tag: str = ""  # e.g. "INGREDIENT|RCT"


@dataclass
class Selected:
    doc_id: str
    source_id: str
    sentence: str
    score: float
    rank: int
    retriever: str
    tag: str = ""


def sentences(text: str, min_chars: int = 25, max_chars: int = 600) -> list[str]:
    out = []
    for part in _SPLIT.split(text or ""):
        s = _LABEL.sub("", " ".join(part.split())).strip()
        if min_chars <= len(s):
            out.append(s[:max_chars])
    return out


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower()) if t not in STOP and len(t) > 1]


class Bm25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus, self.k1, self.b = corpus, k1, b
        self.n = len(corpus)
        self.avgdl = (sum(map(len, corpus)) / self.n) if self.n else 0.0
        df = Counter(t for doc in corpus for t in set(doc))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(doc) for doc in corpus]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for doc, tf in zip(self.corpus, self.tf):
            dl = len(doc) or 1
            s = 0.0
            for q in query:
                if q in tf:
                    f = tf[q]
                    s += self.idf.get(q, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out.append(s)
        return out


@lru_cache(maxsize=2)
def _bi_encoder(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


@lru_cache(maxsize=2)
def _cross_encoder(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


def _minmax(xs: list[float]) -> list[float]:
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    return [0.0 if hi == lo else (x - lo) / (hi - lo) for x in xs]


class SentenceSelector:
    def __init__(self, mode: str = "auto", embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
                 reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 5,
                 first_stage: int = 20, encoder=None, reranker=None) -> None:
        self.mode, self.top_k, self.first_stage = mode, top_k, first_stage
        self.embedding_model, self.reranker_model = embedding_model, reranker_model
        self._encoder, self._reranker = encoder, reranker  # injectable for tests
        self._neural_failed = False

    @classmethod
    def from_settings(cls) -> SentenceSelector:
        from app.config import get_settings

        s = get_settings()
        return cls(mode=s.evidence_selector, embedding_model=s.evidence_embedding_model,
                   reranker_model=s.evidence_reranker_model or None, top_k=s.evidence_top_k)

    def _neural(self):
        if self.mode == "bm25" or self._neural_failed:
            return None, None
        try:
            enc = self._encoder or _bi_encoder(self.embedding_model)
            rr = self._reranker if self._reranker is not None else (
                _cross_encoder(self.reranker_model) if self.reranker_model else None)
            return enc, rr
        except Exception as exc:  # not installed, no internet for the first download, out of memory ...
            if self.mode == "embedding":
                raise
            logger.warning("Neural evidence selection unavailable (%s); using BM25.", exc)
            self._neural_failed = True
            return None, None

    def select(self, query: str, docs: list[Doc], k: int | None = None) -> list[Selected]:
        k = k or self.top_k
        pool = [(d, s) for d in docs for s in sentences(d.text)]
        # de-duplicate identical sentences (the same abstract can back several records)
        seen, uniq = set(), []
        for d, s in pool:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                uniq.append((d, s))
        if not uniq or not query.strip():
            return []
        bm = Bm25([tokens(s) for _, s in uniq]).scores(tokens(query))
        enc, rr = self._neural()
        retriever = "bm25"
        score = _minmax(bm)
        if enc is not None:
            import numpy as np

            vecs = enc.encode([query] + [s for _, s in uniq], normalize_embeddings=True)
            vecs = np.asarray(vecs)
            cos = (vecs[1:] @ vecs[0]).tolist()
            score = [0.3 * b + 0.7 * c for b, c in zip(_minmax(bm), _minmax(cos))]
            retriever = "bm25+embedding"
        order = sorted(range(len(uniq)), key=lambda i: -score[i])[: max(k, self.first_stage)]
        if rr is not None and enc is not None:
            ce = [float(x) for x in rr.predict([(query, uniq[i][1]) for i in order])]
            pairs = sorted(zip(order, ce), key=lambda x: -x[1])
            order, final = [i for i, _ in pairs], dict(pairs)
            retriever = "bm25+embedding+cross-encoder"
        else:
            final = {i: score[i] for i in order}
        out = []
        for rank, i in enumerate(order[:k], start=1):
            d, s = uniq[i]
            if retriever == "bm25" and bm[i] <= 0:
                continue  # no word in common with the claim
            out.append(Selected(doc_id=d.doc_id, source_id=d.source_id, sentence=s, score=round(final[i], 4), rank=rank,
                                retriever=retriever, tag=d.tag))
        for n, x in enumerate(out, start=1):
            x.rank = n
        return out
