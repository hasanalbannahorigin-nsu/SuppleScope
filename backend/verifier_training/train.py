"""Fine-tune one sequence-pair classifier for the claim verifier (run once per model, then vote).

Recipe from the related studies:
  models    roberta-base (MelAnalyze best, F1 0.91) and PubMedBERT
            microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext (CliVER ensemble member)
  settings  5 epochs, batch 8, learning rate 1e-5 (MelAnalyze BERT-family), max length 256 (5 sentences),
            class-weighted loss (labels are imbalanced), best checkpoint by dev macro-F1
Free GPU: Google Colab or Kaggle T4 (see verifier_training/SuppleScope_verifier_training.ipynb).

  python -m verifier_training.train --model roberta-base --data data/verifier --out models/verifier-roberta
  python -m verifier_training.train --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
      --data data/verifier --out models/verifier-pubmedbert
Then set VERIFIER_MODEL_DIRS=models/verifier-roberta,models/verifier-pubmedbert in .env.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from verifier_training.prepare_data import LABELS


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--data", default="data/verifier")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--sources", default="", help="comma list to keep, e.g. 'scifact,healthver,supplescope'")
    args = ap.parse_args()

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from app.services.evaluation.metrics import report

    set_seed(args.seed)
    data = Path(args.data)
    keep = {s.strip() for s in args.sources.split(",") if s.strip()}
    splits = {k: [r for r in load(data / f"{k}.jsonl") if not keep or r["source"] in keep] for k in ("train", "dev", "test")}
    label2id = {lab: i for i, lab in enumerate(LABELS)}
    tok = AutoTokenizer.from_pretrained(args.model)

    def to_ds(rows):
        ds = Dataset.from_list([{"a": r["text_a"], "b": r["text_b"], "label": label2id[r["label"]]} for r in rows])
        return ds.map(lambda x: tok(x["a"], x["b"], truncation="only_second", max_length=args.max_length), batched=True)

    train, dev = to_ds(splits["train"]), to_ds(splits["dev"])
    counts = Counter(r["label"] for r in splits["train"])
    total = sum(counts.values())
    weights = torch.tensor([total / (len(LABELS) * max(counts.get(lab, 0), 1)) for lab in LABELS], dtype=torch.float)
    print("train label counts:", dict(counts), "class weights:", [round(w, 2) for w in weights.tolist()])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label={i: lab for lab, i in label2id.items()}, label2id=label2id)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=weights.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def metrics(pred):
        y = [LABELS[i] for i in pred.label_ids]
        p = [LABELS[i] for i in np.argmax(pred.predictions, axis=-1)]
        r = report(y, p, LABELS)
        return {"macro_f1": r["macro_f1"], "accuracy": r["accuracy"], "kappa": r["cohen_kappa"]}

    targs = TrainingArguments(
        output_dir=str(Path(args.out) / "checkpoints"), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=32, weight_decay=0.01, warmup_ratio=0.06,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1",
        save_total_limit=1, fp16=torch.cuda.is_available(), report_to=[], seed=args.seed, logging_steps=50)
    import inspect
    tok_kw = ({"processing_class": tok} if "processing_class" in inspect.signature(Trainer.__init__).parameters
              else {"tokenizer": tok})
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=train, eval_dataset=dev,
                              data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics, **tok_kw)
    trainer.train()
    out = Path(args.out)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    result = {"model": args.model, "dev": trainer.evaluate(dev)}
    if splits["test"]:
        result["test"] = trainer.evaluate(to_ds(splits["test"]), metric_key_prefix="test")
    result["args"] = vars(args)
    result["train_counts"] = dict(counts)
    (out / "training_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
