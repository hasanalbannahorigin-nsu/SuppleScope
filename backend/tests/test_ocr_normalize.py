from app.services.ocr.normalize import clean_ocr_text, normalize_unicode


def test_preserves_bangla():
    text = "ভিটামিন সি  ৫০০ mg"
    cleaned = clean_ocr_text(text)
    assert "ভিটামিন" in cleaned
    assert "৫০০" in cleaned
    assert normalize_unicode(text)
