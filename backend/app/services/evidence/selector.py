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

Relevance alone favours sentences on the same topic: titles, background ("interest in herbal remedies has
surged"), aims ("To investigate whether ...") and advice ("trials should be conducted"). The verifier needs the
sentences that report findings, as in SciFact rationales. So each sentence gets a rhetorical role, from the
abstract's own section label when it has one (BACKGROUND/OBJECTIVE/METHODS/RESULTS/CONCLUSIONS, the PubMed
200k RCT scheme) and from wording cues otherwise. Relevance is turned into a rank scale (1 = most relevant of
the re-ranked pool, 0 = least); result and conclusion sentences gain EVIDENCE_ROLE_BONUS (default 0.35) and
titles, background, aims and advice lose it, so a finding in the top 70% of the pool outranks the most
relevant background line (0 = relevance only). At most two sentences per study are taken before others are added, so
the premise covers several studies.
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
_CAPS_LABEL = re.compile(r"^([A-Z][A-Z &/,-]{2,60}):\s+")  # any upper-case PubMed label, e.g. "AIM OF THE STUDY:"
_HTML_HEADING = re.compile(r"<h\d[^>]*>\s*([^<]{2,60}?)\s*</h\d>", re.I)  # Europe PMC: "<h4>Results</h4>"
_HTML_TAG = re.compile(r"<[^>]+>")

# Rhetorical role of a sentence and its weight in the ranking: +1 reports what a study found, -1 cannot
# (title, background, aim, call for more research), 0 unknown.
ROLE_WEIGHT = {"result": 1.0, "conclusion": 1.0, "other": 0.0, "methods": -0.5,
               "background": -1.0, "aim": -1.0, "advice": -1.0, "title": -1.0}
_SECTION_WORDS = (("result", ("RESULT", "FINDING")), ("conclusion", ("CONCLUSION", "INTERPRETATION", "DISCUSSION")),
                  ("methods", ("METHOD", "DESIGN", "SETTING", "PARTICIPANT", "PATIENT", "SUBJECT", "INTERVENTION",
                               "MEASURE", "MATERIAL", "DATA SOURCE", "SEARCH", "SELECTION")),
                  ("aim", ("OBJECTIVE", "AIM", "PURPOSE", "GOAL")),
                  ("background", ("BACKGROUND", "INTRODUCTION", "CONTEXT", "RELEVANCE", "RATIONALE")),
                  ("other", ("REGISTRATION", "FUNDING", "KEYWORD", "COPYRIGHT", "ABBREVIATION")))
_STRONG_RESULT = re.compile(
    r"\bp\s*[<=>\u2264\u2265]\s*0?\.\d|\b95\s*%\s*ci\b|confidence interval|\b(?:w?md|smd|rr|hr|or)\s*[=:]\s*-?\d"
    r"|mean difference|odds ratio|risk ratio|hazard ratio|\d\s*(?:mg/dl|mmol/l)"
    r"|\bno (?:statistically )?(?:significant )?(?:differences?|effects?|changes?|improvements?|reductions?|benefits?)\b"
    r"|\bdid not (?:significantly )?(?:differ|change|improve|reduce|lower|affect|alter|show)|\bstatistically significant"
    r"|\bsignificantly (?:\w+ )?(?:reduc|decreas|lower|improv|increas|elevat|restor|attenuat|inhibit|alter|differ|associat|better|worse|higher|great|less|more)"
    r"|\bsignificant (?:\w+ )?(?:reductions?|decreases?|improvements?|increases?|differences?|effects?|changes?|associations?|declines?)\b",
    re.I)
_ADVICE = re.compile(
    r"\bshould be (?:conducted|performed|undertaken|carried out|explored|investigated|confirmed|evaluated)"
    r"|\b(?:further|more|future|larger|additional) (?:[\w-]+ ){0,2}(?:research|studies|trials|investigations?)\b"
    r"|\b(?:are|is|remain) (?:still )?(?:needed|warranted|required)\b", re.I)
_AIM = re.compile(
    r"^(?:to (?:investigate|evaluate|assess|determine|examine|compare|explore|test|review|summari[sz]e)\b"
    r"|(?:this|the present|the current|our) (?:[\w-]+ ){0,4}?(?:study|review|trial|meta-analysis|analysis|work|paper) "
    r"(?:aim(?:s|ed)?|was designed|is designed|sought|evaluat(?:ed|es)|assess(?:ed|es)|investigat(?:ed|es)|examin(?:ed|es)"
    r"|explor(?:ed|es)|summari[sz](?:ed|es)|compar(?:ed|es))\b|(?:we|the authors) (?:aim(?:ed)?|sought|set out)\b"
    r"|the (?:aim|objective|purpose|goal)s? (?:of|was|is)\b)", re.I)
_CONCLUSION = re.compile(
    r"\bin conclusion\b|\bwe conclude\b|\b(?:these|our|the) (?:results|findings|data) (?:suggest|indicate|show|demonstrate|support|confirm)"
    r"|\b(?:suggest|indicate|demonstrate|show|find|conclude)(?:s|d|ed)? that\b|\b(?:shown|found) that\b"
    r"|\b(?:is|was|were|are|appears?|seems?) (?:not )?(?:effective|efficacious|beneficial|useful|helpful)\b"
    r"|\b(?:had|has|have|showed|produced) (?:no|an?|significant|beneficial|modest|small|little) (?:[\w-]+ )?(?:effect|benefit|impact|improvement)s?\b",
    re.I)
_BACKGROUND = re.compile(
    r"\b(?:prevalence|incidence|worldwide|globally|global burden|epidemic)\b|\bglobal (?:incidence|prevalence)"
    r"|\bis an? (?:major|common|growing|serious|chronic|leading)\b|\b(?:has|have) (?:surged|grown|gained|attracted)\b"
    r"|\binterest in\b|\b(?:is|are|has been|have been) (?:widely|commonly|traditionally|frequently|extensively) used\b"
    r"|\btraditional(?:ly)? (?:medicine|use|used)\b|\blittle is known\b|\bremains? (?:unclear|unknown|controversial)\b"
    r"|\bsince ancient times\b|\bfor (?:centuries|thousands of years)\b|\b(?:has|have) been in use\b|\balso known as\b"
    r"|\b(?:increasing|growing) (?:amount|number|body) of\b", re.I)
_METHODS = re.compile(
    r"\bwere randomi[sz]ed\b|\brandomly (?:assigned|allocated)\b|\bwere (?:enrolled|recruited|included|allocated|assigned|divided)\b"
    r"|\b(?:we|were) searched\b|\bdatabases? (?:were|was) searched\b|\bdata were (?:extracted|analy[sz]ed|pooled)\b"
    r"|\bparticipants (?:received|took|were given)\b|\b(?:was|were) measured\b", re.I)
_WEAK_RESULT = re.compile(
    r"\b(?:reduc(?:ed|tion)|decreas(?:ed|e)|lowered|improv(?:ed|ement)|increased|normali[sz]ed|restor(?:ed|es)|attenuat(?:ed|es)"
    r"|compared (?:with|to)|versus|than (?:placebo|control))\b|\d(?:\.\d+)?\s*%", re.I)


@dataclass
class Doc:
    doc_id: str  # evidence_id
    source_id: str
    text: str
    tag: str = ""  # e.g. "INGREDIENT|RCT"
    title: str = ""  # paper title, so the title line is not taken for a finding


@dataclass
class Selected:
    doc_id: str
    source_id: str
    sentence: str
    score: float
    rank: int
    retriever: str
    tag: str = ""
    role: str = "other"


def _section_role(label: str) -> str | None:
    up = label.upper()
    return next((role for role, words in _SECTION_WORDS if any(w in up for w in words)), None)


def labelled_sentences(text: str, min_chars: int = 25, max_chars: int = 600) -> list[tuple[str, str | None]]:
    """(sentence, section role) pairs; the role comes from the abstract's section label, or None."""
    text = _HTML_TAG.sub(" ", _HTML_HEADING.sub(lambda m: f"\n{m.group(1).upper()}: ", text or ""))
    out, section = [], None
    for part in _SPLIT.split(text):
        s = " ".join(part.split())
        m = _CAPS_LABEL.match(s) or _LABEL.match(s)
        if m:
            section = _section_role(m.group(0).rstrip(": ")) or section
            s = s[m.end():].strip()
        if min_chars <= len(s):
            out.append((s[:max_chars], section))
    return out


def sentences(text: str, min_chars: int = 25, max_chars: int = 600) -> list[str]:
    return [s for s, _ in labelled_sentences(text, min_chars, max_chars)]


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def sentence_role(sentence: str, section: str | None = None, title: str = "") -> str:
    """result, conclusion, methods, background, aim, advice, title or other."""
    n = _norm(sentence)
    if title and n and n in _norm(title):
        return "title"
    strong = bool(_STRONG_RESULT.search(sentence))
    if section == "conclusion" and not strong and _ADVICE.search(sentence):
        return "advice"
    if section:
        return section
    if strong:
        return "result"
    if _ADVICE.search(sentence):
        return "advice"
    if _AIM.search(sentence):
        return "aim"
    if _CONCLUSION.search(sentence):
        return "conclusion"
    if _BACKGROUND.search(sentence):
        return "background"
    if _METHODS.search(sentence):
        return "methods"
    if _WEAK_RESULT.search(sentence):
        return "result"
    return "other"


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


def _rank_scale(values: dict[int, float]) -> dict[int, float]:
    """1.0 for the most relevant candidate, falling evenly to 0.0 for the least; ties share a value.
    Unlike min-max, one very high cross-encoder logit does not squeeze every other candidate towards 0."""
    n = len(values)
    if n <= 1:
        return {i: 1.0 for i in values}
    return {i: 1.0 - sum(w > v for w in values.values()) / (n - 1) for i, v in values.items()}


def _minmax(xs: list[float]) -> list[float]:
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    return [0.0 if hi == lo else (x - lo) / (hi - lo) for x in xs]


class SentenceSelector:
    def __init__(self, mode: str = "auto", embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
                 reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 5,
                 first_stage: int = 20, encoder=None, reranker=None, role_bonus: float = 0.35, per_doc: int = 2) -> None:
        self.mode, self.top_k, self.first_stage = mode, top_k, first_stage
        self.role_bonus, self.per_doc = role_bonus, per_doc
        self.embedding_model, self.reranker_model = embedding_model, reranker_model
        self._encoder, self._reranker = encoder, reranker  # injectable for tests
        self._neural_failed = False

    @classmethod
    def from_settings(cls) -> SentenceSelector:
        from app.config import get_settings

        s = get_settings()
        return cls(mode=s.evidence_selector, embedding_model=s.evidence_embedding_model,
                   reranker_model=s.evidence_reranker_model or None, top_k=s.evidence_top_k,
                   role_bonus=s.evidence_role_bonus)

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
        pool = [(d, s, sentence_role(s, sec, d.title)) for d in docs for s, sec in labelled_sentences(d.text)]
        # de-duplicate identical sentences (the same abstract can back several records)
        seen, uniq, roles = set(), [], []
        for d, s, role in pool:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                uniq.append((d, s))
                roles.append(role)
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
        if retriever == "bm25":
            order = [i for i in order if bm[i] > 0]  # no word in common with the claim
        if rr is not None and enc is not None and order:
            ce = [float(x) for x in rr.predict([(query, uniq[i][1]) for i in order])]
            relevance = _rank_scale(dict(zip(order, ce)))
            retriever = "bm25+embedding+cross-encoder"
        else:
            relevance = _rank_scale({i: score[i] for i in order})
        # findings first: relevance plus a bonus for result / conclusion sentences
        final = {i: relevance[i] + self.role_bonus * ROLE_WEIGHT.get(roles[i], 0.0) for i in order}
        order = sorted(order, key=lambda i: -final[i])
        picked, spare, per_doc = [], [], Counter()
        for i in order:  # at most per_doc sentences of one study first, then fill up with the rest
            doc = uniq[i][0].doc_id
            (picked if not self.per_doc or per_doc[doc] < self.per_doc else spare).append(i)
            per_doc[doc] += 1
        out = []
        for i in (picked + spare)[:k]:
            d, s = uniq[i]
            out.append(Selected(doc_id=d.doc_id, source_id=d.source_id, sentence=s, score=round(final[i], 4), rank=0,
                                retriever=retriever, tag=d.tag, role=roles[i]))
        for n, x in enumerate(out, start=1):
            x.rank = n
        return out
