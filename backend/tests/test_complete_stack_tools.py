"""Completed stack, tooling: training data preparation, evaluation ablation, Label Studio round trip,
EasyOCR line grouping, faster-whisper settings. Offline."""
import io
import json
import tarfile

from app.schemas.research.verifier import VerifierInput
from app.services.ocr.easyocr_provider import EasyOcrProvider, group_lines
from app.services.ocr.pipeline import PROVIDERS
from app.services.verification.handoff import build_verifier_inputs
from labelling.export_tasks import to_task
from labelling.import_annotations import agreement, gold_rows
from verifier_training import prepare_data as PD
from verifier_training.evaluate import evaluate, table

from tests.test_research_agent_v2 import _full_run


def _inputs():
    return build_verifier_inputs(_full_run())


def test_scifact_parsing_from_release_tarball(tmp_path, monkeypatch):
    corpus = [{"doc_id": 1, "title": "T", "abstract": ["Fenugreek lowered fasting glucose in adults.", "No harms were seen."]},
              {"doc_id": 2, "title": "U", "abstract": ["Vitamin C did not shorten colds in this trial of adults."]}]
    claims = [{"id": 1, "claim": "Fenugreek lowers blood glucose.", "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}]},
               "cited_doc_ids": [1]},
              {"id": 2, "claim": "Vitamin C cures colds.", "evidence": {"2": [{"sentences": [0], "label": "CONTRADICT"}]},
               "cited_doc_ids": [2]},
              {"id": 3, "claim": "Vitamin C prevents cancer.", "evidence": {}, "cited_doc_ids": [2]}]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, rows in (("data/corpus.jsonl", corpus), ("data/claims_train.jsonl", claims), ("data/claims_dev.jsonl", [])):
            raw = "\n".join(json.dumps(r) for r in rows).encode()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    (tmp_path / "scifact.tar.gz").write_bytes(buf.getvalue())
    rows = PD.scifact(tmp_path)
    assert [r["label"] for r in rows] == ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    assert rows[0]["text_b"] == "[STUDY] Fenugreek lowered fasting glucose in adults."
    assert rows[0]["text_a"] == "Fenugreek lowers blood glucose."


def test_supplescope_gold_rows_and_split(tmp_path):
    ins = _inputs()
    gold = tmp_path / "gold.jsonl"
    gold.write_text("\n".join(json.dumps({**vi.model_dump(mode="json"), "gold_label": lab})
                              for vi, lab in zip(ins, ["PARTIALLY_SUPPORTED", "NOT_ASSESSABLE"])), encoding="utf-8")
    rows = PD.supplescope(gold)
    assert len(rows) == 1 and rows[0]["label"] == "PARTIALLY_SUPPORTED"  # NOT_ASSESSABLE is rules-only
    parts = PD.split_rows(rows + [{"id": "healthver-train-1", "source": "healthver", "split_hint": "train",
                                   "text_a": "a", "text_b": "b", "label": "SUPPORTED"}])
    assert sum(map(len, parts.values())) == 2 and parts["train"][0]["source"] in ("healthver", "supplescope")


def test_evaluate_ablation_runs_rules_and_skips_missing_models():
    ins = _inputs()
    res = evaluate(ins, ["PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE"], ["rules", "ensemble"], models=[])
    assert res["rules"]["accuracy"] == 1.0 and res["rules"]["macro_f1"] == 1.0
    assert "skipped" in res["ensemble"]
    assert "rules" in table(res)


def test_label_studio_round_trip():
    ins = _inputs()
    tasks = [to_task(vi) for vi in ins]
    assert tasks[0]["data"]["claim_en"] == "helps control sugar level" and "[INGREDIENT|" in tasks[0]["data"]["evidence"]

    def ann(who, label):
        return {"completed_by": who, "result": [{"from_name": "verdict", "value": {"choices": [label]}}]}

    tasks[0]["annotations"] = [ann(1, "PARTIALLY_SUPPORTED"), ann(2, "PARTIALLY_SUPPORTED"), ann(3, "SUPPORTED")]
    tasks[1]["annotations"] = [ann(1, "INSUFFICIENT_EVIDENCE"), ann(2, "NOT_ASSESSABLE")]
    rows, ties = gold_rows(tasks)
    assert [r["gold_label"] for r in rows] == ["PARTIALLY_SUPPORTED"] and len(ties) == 1
    rows, ties = gold_rows(tasks, adjudicator="1")
    assert [r["gold_label"] for r in rows] == ["PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE"] and not ties
    VerifierInput.model_validate({k: v for k, v in rows[0].items() if k != "gold_label"})
    ag = agreement(tasks)
    assert ag["annotators"] == ["1", "2", "3"] and "1 vs 2" in ag["pairwise_cohen"]


def test_easyocr_lines_and_registration():
    results = [([[10, 52], [80, 52], [80, 70], [10, 70]], "সুগার", 0.9),
               ([[90, 50], [160, 50], [160, 72], [90, 72]], "নিয়ন্ত্রণে", 0.8),
               ([[10, 10], [100, 10], [100, 30], [10, 30]], "ডায়বিডেক্স", 0.95)]
    lines, boxes, confs = group_lines(results)
    assert lines == ["ডায়বিডেক্স", "সুগার নিয়ন্ত্রণে"] and len(boxes) == 3 and len(confs) == 3
    assert PROVIDERS["easyocr"] is EasyOcrProvider
    assert EasyOcrProvider().unavailable_reason() in (None, "EasyOCR is not installed (pip install -r requirements-ml.txt).")


def test_faster_whisper_uses_cached_model_and_language(monkeypatch, tmp_path):
    from app.services import speech

    made = []

    class FakeModel:
        def __init__(self, size, device, compute_type):
            made.append((size, device, compute_type))

        def transcribe(self, path, **kw):
            self.kw = kw
            seg = type("S", (), {"start": 0.0, "end": 1.0, "text": "সুগার কমে"})()
            return iter([seg]), type("I", (), {"language": kw.get("language")})()

    import sys
    import types
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel))
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "faster_whisper_model", "medium")  # a local .env may say otherwise
    monkeypatch.setattr(get_settings(), "stt_language", "bn")
    speech._whisper_model.cache_clear()
    p = speech.FasterWhisperProvider()
    r1 = p.transcribe(tmp_path / "a.wav")
    p.transcribe(tmp_path / "b.wav")
    assert made == [("medium", "cpu", "int8")]  # loaded once
    assert r1.language == "bn" and r1.full_text == "সুগার কমে"
    speech._whisper_model.cache_clear()
