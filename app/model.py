"""Portable next-action baseline and optional pretrained-transformer continuation."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path

class NGramActionModel:
    """Smoothed variable-context next-action model; transparent local baseline."""
    def __init__(self, max_context: int = 3):
        self.max_context = max_context
        self.counts = defaultdict(Counter)
        self.vocabulary = set()

    def fit(self, sequences: list[list[str]]):
        for sequence in sequences:
            for index, action in enumerate(sequence):
                self.vocabulary.add(action)
                for width in range(self.max_context + 1):
                    context = tuple(sequence[max(0, index-width):index]) if width else ()
                    self.counts[context][action] += 1
        return self

    def predict(self, partial_sequence: list[str], top_k: int = 5) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        for width in range(min(self.max_context, len(partial_sequence)), -1, -1):
            context = tuple(partial_sequence[-width:]) if width else ()
            counts = self.counts.get(context)
            if counts:
                total = sum(counts.values())
                return [{"action": action, "probability": count / total}
                        for action, count in counts.most_common(top_k)]
        return []

    def save(self, path):
        payload = {
            "max_context": self.max_context,
            "counts": [{"context": list(ctx), "actions": dict(counts)}
                       for ctx, counts in self.counts.items()],
            "vocabulary": sorted(self.vocabulary)
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(payload["max_context"])
        for row in payload["counts"]:
            model.counts[tuple(row["context"])] = Counter(row["actions"])
        model.vocabulary = set(payload["vocabulary"])
        return model

def continue_pretrain_transformer(train_sequences, output_dir: str, model_name: str = "distilgpt2",
                                  epochs: int = 1):
    """Continue causal-LM pretraining on serialized possession sequences (optional extra)."""
    try:
        from datasets import Dataset
        from transformers import (AutoTokenizer, AutoModelForCausalLM,
                                  DataCollatorForLanguageModeling, Trainer, TrainingArguments)
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies: pip install -r requirements-transformer.txt") from exc
    texts = [" events: " + " -> ".join(seq) for seq in train_sequences if seq]
    if len(texts) < 2:
        raise ValueError("Need at least two non-empty training sequences")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_dict({"text": texts})
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(model_name)
    args = TrainingArguments(output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=4, learning_rate=5e-5, logging_steps=10,
        save_strategy="epoch", report_to="none", fp16=False)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


class TransformerActionModel:
    """Score known action labels with a continued-pretrained causal language model."""
    def __init__(self, model_dir: str, vocabulary: list[str]):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("Install optional dependencies: pip install -r requirements-transformer.txt") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.eval()
        self.vocabulary = sorted(set(vocabulary))

    def predict(self, partial_sequence: list[str], top_k: int = 5) -> list[dict]:
        torch = self.torch
        prompt = " events: " + " -> ".join(partial_sequence) + " ->"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        scored = []
        with torch.no_grad():
            for action in self.vocabulary:
                candidate = " " + action
                target = self.tokenizer(candidate, add_special_tokens=False,
                                        return_tensors="pt")["input_ids"]
                if target.numel() == 0:
                    continue
                full_ids = torch.cat([inputs["input_ids"], target], dim=1)
                logits = self.model(full_ids).logits
                start = inputs["input_ids"].shape[1] - 1
                log_probs = torch.log_softmax(logits[0, start:start + target.shape[1]], dim=-1)
                token_scores = log_probs.gather(1, target[0].unsqueeze(1)).sum().item()
                scored.append((action, token_scores))
        if not scored:
            return []
        maximum = max(score for _, score in scored)
        weights = [(action, float(__import__("math").exp(score - maximum)))
                   for action, score in scored]
        total = sum(weight for _, weight in weights)
        weights.sort(key=lambda item: item[1], reverse=True)
        return [{"action": action, "probability": weight / total}
                for action, weight in weights[:top_k]]


def evaluate(model, sequences: list[list[str]], top_k: int = 3) -> dict:
    total = hits = 0
    for sequence in sequences:
        for index in range(1, len(sequence)):
            predictions = model.predict(sequence[:index], top_k=top_k)
            total += 1
            if predictions and sequence[index] in {p["action"] for p in predictions}:
                hits += 1
    return {"examples": total, "hits_at_k": hits / total if total else 0.0,
            "top_k": top_k}
