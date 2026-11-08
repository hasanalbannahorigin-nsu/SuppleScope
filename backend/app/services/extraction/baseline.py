from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.ocr.normalize import clean_ocr_text, normalize_unicode

CLAIM_PATTERNS = [
    r"(?i)\b(boosts?|improves?|supports?|enhances?|increases?|reduces?|prevents?|cures?|treats?|immunity|energy|stamina|weight\s*loss|muscle|testosterone|hair\s*growth|skin\s*glow|joint\s*pain)\b.{0,80}",
    r"(ইমিউনিটি|রোগ প্রতিরোধ|শক্তি|স্তামিনা|ওজন কমা|চুল গজায়|ত্বক|জয়েন্ট|ব্যথা|যৌন|টেস্টোস্টেরন|ভিটামিন|খাদ্য সম্পূরক).{0,80}",
]

INGREDIENT_PATTERNS = [
    r"(?i)\b(vitamin\s*[a-k]\b|zinc|magnesium|calcium|iron|ashwagandha|ginseng|moringa|spirulina|collagen|omega[\s-]?3|fish oil|protein|creatine|whey|biotin|folic acid|probiotic)s?\b",
    r"(ভিটামিন|জিংক|ম্যাগনেশিয়াম|ক্যালসিয়াম|আয়রন|অশ্বগন্ধা|জিনসেং|মরিঙ্গা|স্পাইরুলিনা|কোলাজেন|প্রোটিন|বায়োটিন)",
]

DOSAGE_PATTERN = r"(?i)(\d+\s*(mg|mcg|g|ml|iu|মিলিগ্রাম|গ্রাম).{0,40})"
WARNING_PATTERN = r"(?i)(warning|caution|consult.{0,40}doctor|পার্শ্বপ্রতিক্রিয়া|সতর্কতা|ডাক্তার.{0,30})"
DISEASE_PATTERN = r"(?i)\b(diabetes|diabetic|cancer|heart|kidney|liver|hypertension|asthma|ডায়াবেটিস|ক্যান্সার|হৃদরোগ|কিডনি|লিভার|রক্তচাপ|হাঁপানি)\b"
DRUG_PATTERN = r"(?i)\b(warfarin|insulin|statin|aspirin|metformin|ibuprofen|paracetamol|warfarin|ইনসুলিন|অ্যাসপিরিন)\b"


@dataclass
class ExtractionResult:
    product_name: str | None = None
    brand: str | None = None
    manufacturer: str | None = None
    product_category: str | None = None
    claims: list[dict] = field(default_factory=list)
    ingredients: list[dict] = field(default_factory=list)
    dosage: list[str] = field(default_factory=list)
    usage: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diseases: list[str] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = clean_ocr_text(item).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(clean_ocr_text(item))
    return out


def _guess_product_name(text: str) -> str | None:
    lines = [clean_ocr_text(line) for line in text.splitlines() if clean_ocr_text(line)]
    if not lines:
        return None
    # Prefer a short prominent line that is not a generic claim.
    ranked = sorted(lines[:12], key=lambda line: (-(2 if len(line) < 48 else 0), len(line)))
    for line in ranked:
        if len(line) < 3 or len(line) > 80:
            continue
        if re.search(CLAIM_PATTERNS[0], line):
            continue
        return line
    return lines[0][:80]


def extract_from_text(*, ocr_text: str, speech_text: str = "") -> ExtractionResult:
    ocr_text = normalize_unicode(ocr_text or "")
    speech_text = normalize_unicode(speech_text or "")
    combined = "\n".join(part for part in [ocr_text, speech_text] if part)
    result = ExtractionResult()
    result.product_name = _guess_product_name(ocr_text or speech_text)
    result.product_category = "dietary supplement"

    for pattern in CLAIM_PATTERNS:
        for match in re.finditer(pattern, combined):
            span = clean_ocr_text(match.group(0))
            source = "speech" if speech_text and span in speech_text else "ocr"
            result.claims.append(
                {
                    "claim_text_original": span,
                    "claim_text_en": None,
                    "claim_type": "health_claim",
                    "source": source,
                }
            )

    seen_claims = set()
    deduped = []
    for claim in result.claims:
        key = claim["claim_text_original"].lower()
        if key in seen_claims:
            continue
        seen_claims.add(key)
        deduped.append(claim)
    result.claims = deduped[:40]

    for pattern in INGREDIENT_PATTERNS:
        for match in re.finditer(pattern, combined):
            name = clean_ocr_text(match.group(0))
            result.ingredients.append({"name": name, "original_text": name, "source": "ocr"})
    # unique ingredients
    uniq = []
    seen_i = set()
    for item in result.ingredients:
        key = item["name"].lower()
        if key in seen_i:
            continue
        seen_i.add(key)
        uniq.append(item)
    result.ingredients = uniq

    result.dosage = _unique([m.group(0) for m in re.finditer(DOSAGE_PATTERN, combined)])
    result.warnings = _unique([m.group(0) for m in re.finditer(WARNING_PATTERN, combined)])
    result.diseases = _unique([m.group(0) for m in re.finditer(DISEASE_PATTERN, combined)])
    result.drugs = _unique([m.group(0) for m in re.finditer(DRUG_PATTERN, combined)])
    usage_hits = re.findall(r"(?i)(take .{0,60}|সেবন.{0,60}|দিনে .{0,40})", combined)
    result.usage = _unique(usage_hits)
    result.extra = {
        "backend": "baseline",
        "note": (
            "Rule-based Bangla/English extractor. BanglaBERT adapter is available "
            "but no custom-trained model is claimed."
        ),
    }
    return result
