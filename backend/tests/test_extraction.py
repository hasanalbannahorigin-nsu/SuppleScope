from app.services.extraction.baseline import extract_from_text


def test_extracts_bangla_and_english_claims_without_translation():
    ocr = "হেলথ প্লাস ইমিউনিটি বুস্টার\nভিটামিন C এবং Zinc\nBoosts immunity and energy\nTake 1 tablet daily"
    result = extract_from_text(ocr_text=ocr)
    originals = [c["claim_text_original"] for c in result.claims]
    assert any("ইমিউনিটি" in text or "Boosts" in text for text in originals)
    names = [i["name"].lower() for i in result.ingredients]
    assert any("zinc" in n or "ভিটামিন" in n or "vitamin" in n for n in names)
    assert result.product_name is not None


def test_does_not_invent_unmentioned_ingredients():
    result = extract_from_text(ocr_text="সুপার টনিক শক্তি বাড়ায়")
    assert all("ashwagandha" not in i["name"].lower() for i in result.ingredients)
