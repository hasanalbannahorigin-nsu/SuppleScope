"""Word-level grounding of extracted claims (D CARE DIA CONTROL ad, Tesseract output as seen on the laptop).

The extractor turned "ইনসুলিন উৎপাদন করবে" into an extra claim "বিটা সেল উৎপাদন করবে". The old check compared
match_key strings, which drop Bangla vowel signs, so "সেল" looked like the "সুল" of ইনসুলিন and the claim passed.
"""
from app.services.extraction.grounded import _Claim, _Cleaned, _Extraction, check_grounding, keep_ocr_where_words_invented
from app.services.text.bangla import letters_key, missing_words, normalize

RAW = normalize('" ৭ দিনে @e স্থ্য়ীসমাধান _ শাল & ইনসুলিন উৎপাদন করবে & দীর্ঘমেয়াদি রোগ দুর করবে কে খা\n'
                "D CARE & সুগার নিয়স্ত্রণকরবে & রোগ প্রতিরোধ ক্ষমতা বাড়াবে প্র) CARE ¢ শতভাগ কার্যকরী পার\n"
                "DIA CONTROL ও পার্শপ্রতিক্রিয়া মুক্ত মাজে")
CLEANED = normalize("৭ দিনে স্থায়ী সমাধান\nইনসুলিন উৎপাদন করবে\nদীর্ঘমেয়াদি রোগ দুর করবে\nD CARE\nসুগার নিয়ন্ত্রণ করবে\n"
                    "রোগ প্রতিরোধ ক্ষমতা বাড়াবে\nশতভাগ কার্যকরী\nDIA CONTROL ও পার্শ্বপ্রতিক্রিয়া মুক্ত")


def _claim(text: str, evidence: str) -> _Claim:
    return _Claim(text=text, claim_type="TREATMENT", evidence=evidence)


def test_letters_key_keeps_vowel_signs():
    assert letters_key("বিটা সেল!") == "বিটা সেল"
    assert letters_key("সেল") != letters_key("সুল")


def test_invented_claim_words_are_flagged():
    ex = _Extraction(claims=[
        _claim("ইনসুলিন উৎপাদন করবে", "ইনসুলিন উৎপাদন করবে"),
        _claim("বিটা সেল উৎপাদন করবে", "ইনসুলিন উৎপাদন করবে"),  # words not in its evidence
        _claim("বিটা সেল উৎপাদন করবে", "বিটা সেল উৎপাদন করবে"),  # evidence itself not in the ad
        _claim("সুগার নিয়ন্ত্রণ করবে", "সুগার নিয়ন্ত্রণ করবে"),
        _claim("স্থায়ী সমাধান করবে", "৭ দিনে স্থায়ী সমাধান"),  # an added auxiliary verb is fine
        _claim("দীর্ঘমেয়াদি রোগ দূর করবে", "দীর্ঘমেয়াদি রোগ দুর করবে"),  # one vowel sign corrected is fine
        _claim("পার্শ্বপ্রতিক্রিয়ামুক্ত", "ও পার্শ্বপ্রতিক্রিয়া মুক্ত"),
    ])
    flagged = {i.path for i in check_grounding(ex, CLEANED, RAW)}
    assert flagged == {"claims[1]", "claims[2]"}
    assert "বিটা" in missing_words("বিটা সেল উৎপাদন করবে", "ইনসুলিন উৎপাদন করবে")


def test_ocr_joined_words_still_count():
    # evidence copied from raw OCR where two words ran together
    ex = _Extraction(claims=[_claim("সুগার নিয়ন্ত্রণ করবে", "সুগার নিয়স্ত্রণকরবে")])
    assert check_grounding(ex, RAW, RAW) == []


def test_added_target_word_is_flagged():
    # second run on the laptop: "৭ দিনে স্থায়ী সমাধান ডায়াবেটিস" although the ad never says "diabetes"
    ex = _Extraction(claims=[_claim("৭ দিনে স্থায়ী সমাধান ডায়াবেটিস", "৭ দিনে স্থায়ী সমাধান")])
    assert [i.problem for i in check_grounding(ex, CLEANED, RAW)] == [f"words not in its evidence: {letters_key('ডায়াবেটিস')}"]


def test_cleaning_cannot_add_words():
    cleaned = _Cleaned(text="\n".join([
        "৭ দিনে স্থায়ী সমাধান ডায়াবেটিস & ইনসুলিন উৎপাদন করবে & দীর্ঘমেয়াদি রোগ দূর করবে",  # adds ডায়াবেটিস
        "D CARE & সুগার নিয়ন্ত্রণ করবে & রোগ প্রতিরোধ ক্ষমতা বাড়াবে CARE শতভাগ কার্যকরী",  # real fixes only
        "DIA CONTROL ও পার্শ্বপ্রতিক্রিয়ামুক্ত সমাজ",  # garbage "মাজে" made into "সমাজ"
    ]), edits=["নিয়স্ত্রণকরবে -> নিয়ন্ত্রণ করবে"])
    cleaned.text = normalize(cleaned.text)  # as GroundedExtractor._clean does
    out = keep_ocr_where_words_invented(cleaned, RAW)
    lines = out.text.split("\n")
    # only the invented words go; the spelling fixes on the same lines stay (laptop 2026-11-28: reverting the
    # whole line left claims like "স্থ্য়ীসমাধান" and "নিয়স্ত্রণকরবে")
    assert lines[0] == normalize("৭ দিনে স্থায়ী সমাধান & ইনসুলিন উৎপাদন করবে & দীর্ঘমেয়াদি রোগ দূর করবে")
    assert lines[1].startswith(normalize("D CARE & সুগার নিয়ন্ত্রণ করবে"))
    assert lines[2] == normalize("DIA CONTROL ও পার্শ্বপ্রতিক্রিয়ামুক্ত")
    assert any(letters_key("ডায়াবেটিস") in e for e in out.edits) and any(letters_key("সমাজ") in e for e in out.edits)
    assert all(not missing_words(line, RAW, min_coverage=1.0) for line in lines)
    # a different line count no longer throws the cleaning away: the invented word still goes
    merged = keep_ocr_where_words_invented(_Cleaned(text=normalize("৭ দিনে স্থায়ী সমাধান ডায়াবেটিস")), RAW)
    assert merged.text == normalize("৭ দিনে স্থায়ী সমাধান")


def test_other_ending_of_the_ads_verb_is_not_invented():
    # 7-ad eval 2026-11-28: r04 says "ওজন কমান প্রাকৃতিক উপায়ে!" and the claim restates it as "... ওজন কমায়"
    ad = normalize("ওজন কমান\nপ্রাকৃতিক উপায়ে!\nরোগ প্রতিরোধ ক্ষমতা বাড়ান")
    assert missing_words(normalize("প্রাকৃতিক উপায়ে ওজন কমায়"), ad, min_coverage=1.0) == []
    assert missing_words(normalize("রোগ প্রতিরোধ ক্ষমতা বাড়াবে"), ad, min_coverage=1.0) == []
    # a new word is still a new word, whatever its ending
    assert missing_words(normalize("ওজন ও ডায়াবেটিস কমায়"), ad, min_coverage=1.0) == [letters_key("ডায়াবেটিস")]
    assert missing_words("সমাজ", RAW, min_coverage=1.0) == ["সমাজ"]
