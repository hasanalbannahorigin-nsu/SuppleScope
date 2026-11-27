"""Ingredient name normalisation: Bangla or transliterated names -> English and Latin (scientific) names.

A curated dictionary (data/ingredient_dictionary.csv, editable; INGREDIENT_DICTIONARY_PATH for your own)
is matched exactly first, then with RapidFuzz (WRatio) above INGREDIENT_MATCH_THRESHOLD. Form words such as
"extract", "powder" or "seed" are stripped before matching and kept as `form`. iDISK2.0 ids are attached
by `link_idisk` when the local iDISK2.0 database has been built (scripts/load_idisk.py), following the
concept-normalisation step of iDISK2.0-RAG (Hou et al. 2025), which used QuickUMLS for the same purpose.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.services.text.bangla import normalize

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DICTIONARY = DATA_DIR / "ingredient_dictionary.csv"

# Words that describe the form, not the ingredient. Order matters: longer phrases first.
FORM_WORDS = [
    "এক্সট্র্যাক্ট", "এক্সট্রাক্ট", "নির্যাস", "চূর্ণ", "গুঁড়া", "গুড়া", "পাউডার", "বীজ", "পাতা", "মূল", "ছাল", "ফল",
    "extract", "powder", "seed", "seeds", "leaf", "leaves", "root", "bark", "fruit", "dried", "standardized",
]
_FORM_RE = re.compile(r"(?:^|\s)(" + "|".join(map(re.escape, FORM_WORDS)) + r")(?=\s|$)", re.I)
_BN_GENITIVE = re.compile(r"(ের|এর|র)$")


def name_key(text: str) -> str:
    """Comparison key that keeps Bangla vowel signs (unlike text.bangla.match_key)."""
    t = normalize(text or "").lower()
    return re.sub(r"[\s\-_.,;:()\[\]/'\"’‘`]+", "", t)


@dataclass(frozen=True)
class DictEntry:
    entry_id: str
    bn: str
    en: str
    scientific: str | None
    aliases: tuple[str, ...]
    category: str


@dataclass
class Normalised:
    input: str
    entry_id: str | None
    english: str | None
    scientific: str | None
    category: str | None
    matched: str | None
    score: float  # 0-100; 100 = exact
    method: str  # exact | fuzzy | none
    form: str | None = None

    @property
    def search_names(self) -> list[str]:
        return [n for n in dict.fromkeys([self.scientific or "", self.english or ""]) if n]


def load_dictionary(path: str | Path | None = None) -> list[DictEntry]:
    path = Path(path) if path else DEFAULT_DICTIONARY
    out = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if not (row.get("en") or "").strip():
                continue
            out.append(DictEntry(
                entry_id=row["id"].strip(), bn=row.get("bn", "").strip(), en=row["en"].strip(),
                scientific=(row.get("scientific") or "").strip() or None,
                aliases=tuple(a.strip() for a in (row.get("aliases") or "").split("|") if a.strip()),
                category=(row.get("category") or "").strip(),
            ))
    return out


def split_form(name: str) -> tuple[str, str | None]:
    forms = [m.group(1) for m in _FORM_RE.finditer(" " + name + " ")]
    core = _FORM_RE.sub(" ", " " + name + " ")
    core = " ".join(core.split())
    return (core or name.strip()), (" ".join(forms) or None)


class IngredientNormaliser:
    def __init__(self, entries: list[DictEntry] | None = None, threshold: float = 86.0) -> None:
        self.entries = entries if entries is not None else load_dictionary()
        self.threshold = threshold
        self.index: dict[str, DictEntry] = {}
        for e in self.entries:
            for n in (e.bn, e.en, e.scientific or "", *e.aliases):
                k = name_key(n)
                if k and k not in self.index:
                    self.index[k] = e
        self._keys = list(self.index)

    def normalise(self, name: str | None) -> Normalised:
        name = (name or "").strip()
        empty = Normalised(input=name, entry_id=None, english=None, scientific=None, category=None, matched=None,
                           score=0.0, method="none")
        if not name:
            return empty
        core, form = split_form(name)
        tries = [name, core]
        if re.search(r"[ঀ-৿]", core):
            tries.append(_BN_GENITIVE.sub("", core))
        for t in tries:
            e = self.index.get(name_key(t))
            if e:
                return self._out(name, e, t, 100.0, "exact", form)
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            return empty
        best = None
        for t in tries[1:] or tries:
            k = name_key(t)
            if len(k) < 3:
                continue
            hit = process.extractOne(k, self._keys, scorer=fuzz.WRatio, score_cutoff=self.threshold)
            # Short keys need a near-exact match: "জাম" must not match "জামরুল".
            if hit and (len(k) >= 5 or hit[1] >= 95) and (best is None or hit[1] > best[1]):
                best = hit
        if best:
            return self._out(name, self.index[best[0]], best[0], float(best[1]), "fuzzy", form)
        return empty

    @staticmethod
    def _out(name: str, e: DictEntry, matched: str, score: float, method: str, form: str | None) -> Normalised:
        return Normalised(input=name, entry_id=e.entry_id, english=e.en, scientific=e.scientific, category=e.category,
                          matched=matched, score=round(score, 1), method=method, form=form)

    def same_ingredient(self, a: str | None, b: str | None) -> bool:
        """True when two names (any script) normalise to the same dictionary entry, or have the same key."""
        if not a or not b:
            return False
        if name_key(split_form(a)[0]) == name_key(split_form(b)[0]):
            return True
        na, nb = self.normalise(a), self.normalise(b)
        if na.entry_id and na.entry_id == nb.entry_id:
            return True
        sa, sb = (na.scientific or "").lower(), (nb.scientific or "").lower()
        return bool(sa and sa == sb)


@lru_cache(maxsize=1)
def default_normaliser() -> IngredientNormaliser:
    from app.config import get_settings

    s = get_settings()
    entries = load_dictionary(s.ingredient_dictionary_path or None)
    return IngredientNormaliser(entries, threshold=s.ingredient_match_threshold)
