"""Patient-specific supplement risk: product ingredients x the patient's medicines and conditions.

Python rules, as recommended for the stack (iDISK2.0-RAG answered patient questions from one-hop graph
facts; Behnamian et al. flagged "dangerous" claims from FDA and NIH sources):

1. Medicines: Bangladeshi brand -> generic (data/bd_brand_generic.csv), else the generic name itself,
   else RxNorm approximate match (online, optional). Generic -> drug classes (data/drug_classes.csv).
   RxNav's drug-interaction API was retired by NLM in 2024, so RxNorm is used for names only.
2. Conditions: English and Bangla terms -> canonical conditions (pregnancy, diabetes, kidney disease ...).
3. Curated rules (data/interaction_rules.csv, NCCIH / NIH ODS / LiverTox citations) -> LOW/MEDIUM/HIGH.
4. Dossier safety findings that name one of the patient's medicines, classes or conditions (literature,
   iDISK2.0, fact sheets, FDA drug labels) -> MEDIUM, with the quoted finding ids.
5. Anything that could not be checked -> UNKNOWN, never LOW.

The rule table is a starting point that a pharmacist must review before the tool is used with patients.
This is not a diagnosis or a prescription.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from app.schemas.research.dossier import ResearchDossierV2
from app.schemas.research.patient_risk import (
    PatientFactor,
    PatientProfile,
    PatientRiskReport,
    RiskItem,
    RiskLevel,
)
from app.services.knowledge.factsheets import states_concern
from app.services.knowledge.normalise import default_normaliser, not_an_ingredient

DATA = Path(__file__).resolve().parent / "data"
ENGINE_VERSION = "1.0"
ORDER = {RiskLevel.LOW: 1, RiskLevel.UNKNOWN: 2, RiskLevel.MEDIUM: 3, RiskLevel.HIGH: 4}
DISCLAIMER = ("This is decision support, not a diagnosis or a prescription. The rules are a starting list that must be "
              "reviewed by a pharmacist; always ask a doctor or pharmacist before combining supplements with medicines. "
              "এটি রোগনির্ণয় বা প্রেসক্রিপশন নয়; ওষুধের সাথে সাপ্লিমেন্ট খাওয়ার আগে ডাক্তার বা ফার্মাসিস্টের পরামর্শ নিন।")

CONDITION_TERMS = {
    "diabetes": ["diabetes", "diabetic", "type 2 diabetes", "type 1 diabetes", "high sugar", "blood sugar", "ডায়াবেটিস",
                 "ডায়বেটিস", "ডায়াবেটিক", "সুগার", "বহুমূত্র"],
    "high blood pressure": ["hypertension", "high blood pressure", "high bp", "উচ্চ রক্তচাপ", "হাই প্রেশার", "হাই ব্লাড প্রেশার"],
    "low blood pressure": ["hypotension", "low blood pressure", "low bp", "নিম্ন রক্তচাপ", "লো প্রেশার"],
    "pregnancy": ["pregnant", "pregnancy", "expecting", "গর্ভবতী", "গর্ভাবস্থা", "অন্তঃসত্ত্বা"],
    "breastfeeding": ["breastfeeding", "breast feeding", "lactating", "nursing mother", "বুকের দুধ", "স্তন্যদান"],
    "kidney disease": ["kidney", "renal", "ckd", "dialysis", "কিডনি", "বৃক্ক"],
    "liver disease": ["liver", "hepatitis", "cirrhosis", "fatty liver", "লিভার", "যকৃত"],
    "thyroid disease": ["thyroid", "hypothyroid", "hyperthyroid", "থাইরয়েড"],
    "heart disease": ["heart disease", "heart failure", "cardiac", "angina", "হৃদরোগ", "হার্টের"],
    "bleeding disorder": ["bleeding disorder", "haemophilia", "hemophilia", "রক্তক্ষরণ"],
    "surgery planned": ["surgery", "operation", "অপারেশন", "সার্জারি"],
    "autoimmune disease": ["autoimmune", "lupus", "rheumatoid arthritis"],
}


# How sources usually write a drug class in plain words.
CLASS_TERMS = {
    "antidiabetic": ["diabetes medicine", "diabetes medication", "diabetes drug", "antidiabetic", "hypoglycemic agent",
                     "blood sugar medicine", "blood glucose-lowering", "insulin", "sulfonylurea", "metformin"],
    "anticoagulant": ["anticoagulant", "blood thinner", "warfarin", "blood-thinning"],
    "antiplatelet": ["antiplatelet", "aspirin", "clopidogrel", "blood thinner"],
    "antihypertensive": ["blood pressure medicine", "blood pressure medication", "antihypertensive"],
    "thyroid hormone": ["thyroid hormone", "levothyroxine", "thyroid medicine"],
    "sedative": ["sedative", "benzodiazepine", "sleeping pill"],
    "immunosuppressant": ["immunosuppress"],
    "antidepressant": ["antidepressant", "ssri", "serotonin"],
    "diuretic": ["diuretic", "water pill"],
    "oral contraceptive": ["contraceptive", "birth control"],
}


def _key(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s\-ঀ-৿]+", " ", (s or "").lower()).split())


@dataclass
class Rule:
    rule_id: str
    ingredients: list[str]
    factor_type: str
    factors: list[str]
    level: RiskLevel
    mechanism: str
    advice: str
    source_name: str
    source_url: str


@lru_cache(maxsize=1)
def tables() -> tuple[dict[str, str], dict[str, list[str]], list[Rule]]:
    with (DATA / "bd_brand_generic.csv").open(encoding="utf-8") as fh:
        brands = {_key(r["brand"]): r["generic"].strip().lower() for r in csv.DictReader(fh)}
    with (DATA / "drug_classes.csv").open(encoding="utf-8") as fh:
        classes = {r["generic"].strip().lower(): [c.strip() for c in r["classes"].split("|") if c.strip()]
                   for r in csv.DictReader(fh)}
    with (DATA / "interaction_rules.csv").open(encoding="utf-8") as fh:
        rules = [Rule(rule_id=r["rule_id"], ingredients=[x.strip().lower() for x in r["ingredients"].split("|")],
                      factor_type=r["factor_type"], factors=[x.strip().lower() for x in r["factors"].split("|")],
                      level=RiskLevel(r["level"]), mechanism=r["mechanism"], advice=r["advice"],
                      source_name=r["source_name"], source_url=r["source_url"]) for r in csv.DictReader(fh)]
    return brands, classes, rules


def split_list(text: str | None) -> list[str]:
    return [x.strip() for x in re.split(r"[,;\n/।]+| and | ও ", text or "") if x.strip()]


def _strip_strength(name: str) -> str:
    """'Comet 500 mg tablet' -> 'comet'."""
    k = _key(name)
    k = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?)?\b", " ", k)
    k = re.sub(r"\b(tab|tablet|tablets|cap|capsule|capsules|syrup|inj|injection|xr|sr|er|plus|forte|daily)\b", " ", k)
    return " ".join(k.split())


def normalise_medicine(typed: str, rxnorm=None) -> PatientFactor:
    brands, classes, _ = tables()
    core = _strip_strength(typed)
    if core in brands:
        g = brands[core]
        return PatientFactor(kind="drug", name=g, typed_as=typed, classes=classes.get(g, []), normalised_by="brand_table")
    first = core.split(" ")[0] if core else ""
    if first in brands:
        g = brands[first]
        return PatientFactor(kind="drug", name=g, typed_as=typed, classes=classes.get(g, []), normalised_by="brand_table")
    for g in sorted(classes, key=len, reverse=True):
        if re.search(rf"\b{re.escape(g)}\b", core):
            return PatientFactor(kind="drug", name=g, typed_as=typed, classes=classes[g], normalised_by="generic_name")
    if rxnorm is not None and core:
        try:
            g = rxnorm(core)
        except Exception:
            g = None
        if g:
            g = g.lower()
            return PatientFactor(kind="drug", name=g, typed_as=typed, classes=classes.get(g, []), normalised_by="rxnorm")
    return PatientFactor(kind="drug", name=core or typed, typed_as=typed, classes=[], normalised_by="none")


def normalise_conditions(texts: list[str]) -> list[PatientFactor]:
    out, seen = [], set()
    for t in texts:
        k = _key(t)
        for canon, terms in CONDITION_TERMS.items():
            if canon in seen:
                continue
            if any(re.search(rf"(?<![\wঀ-৿]){re.escape(term)}", k) for term in terms):
                seen.add(canon)
                out.append(PatientFactor(kind="condition", name=canon, typed_as=t, normalised_by="condition_terms"))
    return out


def profile_from_context(age: int | None, diseases: str | None, medications: str | None,
                         health_context: str | None) -> PatientProfile:
    conds = split_list(diseases) + ([health_context] if health_context else [])
    return PatientProfile(age=age, medications=split_list(medications), conditions=conds, free_text=health_context)


@dataclass
class _Ingredient:
    ingredient_id: str | None
    label: str
    names: list[str]  # lower-case names to match rules


def _ingredients(d: ResearchDossierV2) -> list[_Ingredient]:
    norm = default_normaliser()
    out = []
    for ing in d.ingredients:
        names = [ing.name, ing.standardized_name or "", ing.scientific_name or "", *ing.synonyms]
        nz = norm.normalise(ing.name)
        if nz.entry_id:
            names += [nz.english or "", nz.scientific or ""]
        out.append(_Ingredient(ing.ingredient_id, ing.standardized_name or ing.name,
                               [n.lower() for n in dict.fromkeys(names) if n]))
    known = {i.ingredient_id for i in out}
    for ci in d.product.composition.ingredients:  # label ingredients not researched (over the limit)
        if ci.ingredient_id not in known:
            names = [ci.name, ci.standardized_name or ""]
            out.append(_Ingredient(ci.ingredient_id, ci.standardized_name or ci.name, [n.lower() for n in names if n]))
    return out


def _names_match(rule_names: list[str], names: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(r)}\b", n) for r in rule_names for n in names)


def assess(d: ResearchDossierV2, patient: PatientProfile, *, rxnorm=None) -> PatientRiskReport:
    _, _, rules = tables()
    drugs = [normalise_medicine(m, rxnorm) for m in patient.medications]
    conds = normalise_conditions(patient.conditions + ([patient.free_text] if patient.free_text and
                                                       patient.free_text not in patient.conditions else []))
    if patient.pregnant and not any(c.name == "pregnancy" for c in conds):
        conds.append(PatientFactor(kind="condition", name="pregnancy", typed_as="pregnant", normalised_by="condition_terms"))
    if patient.breastfeeding and not any(c.name == "breastfeeding" for c in conds):
        conds.append(PatientFactor(kind="condition", name="breastfeeding", typed_as="breastfeeding",
                                   normalised_by="condition_terms"))
    factors = drugs + conds
    if patient.age is not None and patient.age < 18:
        factors.append(PatientFactor(kind="population", name="children", typed_as=str(patient.age), normalised_by="age"))
    ings = _ingredients(d)
    items: list[RiskItem] = []
    unknowns: list[str] = []
    for f in drugs:
        if f.normalised_by == "none":
            unknowns.append(f"Medicine '{f.typed_as}' was not recognised, so its interactions could not be checked.")

    # 1) curated rules
    for ing in ings:
        for r in rules:
            if not _names_match(r.ingredients, ing.names):
                continue
            for f in factors:
                targets = ([f.name] + f.classes) if f.kind == "drug" else [f.name]
                if r.factor_type == "drug" and f.kind != "drug":
                    continue
                if r.factor_type == "condition" and f.kind not in ("condition", "population"):
                    continue
                hit = next((t for t in targets if t in r.factors), None)
                if hit:
                    items.append(RiskItem(ingredient_id=ing.ingredient_id, ingredient=ing.label,
                                          factor=f.name if hit == f.name else f"{f.name} ({hit})", factor_kind=f.kind,
                                          level=r.level, mechanism=r.mechanism, advice=r.advice, basis="curated_rule",
                                          rule_id=r.rule_id, sources=[r.source_name, r.source_url]))

    # 2) dossier safety findings naming the patient's medicines / classes / conditions
    by_id = {i.ingredient_id: i for i in d.ingredients}
    srcs = {s.source_id: s for s in d.sources}
    finds = {f.finding_id: f for f in d.findings}
    evs = {e.evidence_id: e for e in d.evidence}
    covered = {(i.ingredient_id, i.factor.split(" (")[0]) for i in items}
    for ing in ings:
        r = by_id.get(ing.ingredient_id)
        if r is None:
            continue
        groups = [("drug", x.drug_or_class, x.description, x.finding_ids, x.evidence_ids) for x in r.safety.drug_interactions]
        groups += [("condition", x.condition, x.description, x.finding_ids, x.evidence_ids) for x in r.safety.disease_cautions]
        for kind, target, desc, fids, eids in groups:
            if kind == "condition" and not states_concern(desc):
                continue  # e.g. "fenugreek is promoted for diabetes": names the disease, states no concern
            text = _key(f"{target} {desc}")
            for f in factors:
                if (kind == "drug") != (f.kind == "drug"):
                    continue
                terms = ([f.name, f.typed_as.lower(), *f.classes, *(t for c in f.classes for t in CLASS_TERMS.get(c, []))]
                         if f.kind == "drug" else [f.name, *CONDITION_TERMS.get(f.name, [])])
                if not any(t and re.search(rf"\b{re.escape(_key(t))}", text) for t in terms):
                    continue
                if (ing.ingredient_id, f.name) in covered:
                    continue
                covered.add((ing.ingredient_id, f.name))
                urls = [srcs[finds[x].source_id].url or srcs[finds[x].source_id].title or "" for x in fids
                        if x in finds and finds[x].source_id in srcs]
                urls += [srcs[evs[x].source_id].url or "" for x in eids if x in evs and evs[x].source_id in srcs]
                items.append(RiskItem(ingredient_id=ing.ingredient_id, ingredient=ing.label, factor=f.name, factor_kind=f.kind,
                                      level=RiskLevel.MEDIUM, mechanism=desc[:500],
                                      advice="A cited source reports this concern; ask a doctor or pharmacist before use.",
                                      basis="dossier_finding", finding_ids=list(fids), evidence_ids=list(eids),
                                      sources=[u for u in dict.fromkeys(urls) if u]))

    # 3) what could not be checked
    if not ings:
        unknowns.append("The product's ingredients are unknown (no confirmed label and none listed in the ad).")
    for ing in d.ingredients:
        if not_an_ingredient(ing.standardized_name, ing.name):
            continue  # e.g. "pancreatic beta cell" taken from a claim by an older research run
        if ing.safety.status.value == "NOT_SEARCHED" and not any(i.ingredient_id == ing.ingredient_id for i in items):
            unknowns.append(f"Safety of '{ing.standardized_name or ing.name}' was not researched (limit reached).")
    if d.product.identity.status.value not in ("VERIFIED", "PROBABLE"):
        unknowns.append("The exact product could not be confirmed, so the label may list other ingredients.")

    items.sort(key=lambda i: (-ORDER[i.level], i.ingredient, i.factor))
    if items:
        overall = max((i.level for i in items), key=lambda x: ORDER[x])
        if overall == RiskLevel.LOW and unknowns:
            overall = RiskLevel.UNKNOWN
    elif not factors:
        overall = RiskLevel.UNKNOWN
        unknowns.append("No medicines or conditions were given, so no patient-specific check was possible.")
    else:
        overall = RiskLevel.UNKNOWN if unknowns else RiskLevel.LOW
    return PatientRiskReport(dossier_id=d.dossier_id, ad_id=d.ad_id, created_at=datetime.now(timezone.utc),
                             engine_version=ENGINE_VERSION, overall=overall, items=items, unknowns=list(dict.fromkeys(unknowns)),
                             factors=factors, ingredients_considered=[i.label for i in ings], disclaimer=DISCLAIMER)
