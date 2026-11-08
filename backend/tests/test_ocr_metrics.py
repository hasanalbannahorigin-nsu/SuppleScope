import pytest

from app.services.ocr.metrics import character_error_rate, evaluate_transcription, word_error_rate


def test_error_rates_match_simple_ground_truth():
    assert character_error_rate("কথা", "কথ") == pytest.approx(1 / 3)
    assert word_error_rate("বাংলা বিজ্ঞাপন", "বাংলা") == pytest.approx(0.5)


def test_error_rates_normalize_unicode_and_handle_empty_ground_truth():
    assert evaluate_transcription("বাংলা", "বাংলা") == {"cer": 0.0, "wer": 0.0}
    assert character_error_rate("", "বাংলা") == 1.0
    assert word_error_rate("", "") == 0.0
