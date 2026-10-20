"""Training/evaluation loops shared by CLI entry points."""
from __future__ import annotations

import json
from pathlib import Path
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import TweetSpanDataset, collate_batch
from .metrics import token_metrics, span_metrics, tweet_presence_metrics
from .spans import ID2LABEL, bio_to_char_spans


def resolve_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {requested!r} but CUDA is unavailable")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset: TweetSpanDataset, batch_size: int, shuffle: bool, num_workers: int = 0,
                pin_memory: bool = False, pad_token_id: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        collate_fn=partial(collate_batch, pad_token_id=pad_token_id),
    )


def _to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["labels"].to(device),
    )


@torch.no_grad()
def run_validation(model, loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in loader:
        input_ids, attention_mask, labels = _to_device(batch, device)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        batch_n = input_ids.shape[0]
        total_loss += float(output["loss"].item()) * batch_n
        count += batch_n
    if not count:
        raise ValueError("Validation loader is empty")
    return total_loss / count



@torch.no_grad()
def evaluate_model(model, loader, device: torch.device) -> tuple[dict, pd.DataFrame, dict]:
    model.eval()
    token_gold_content: list[list[str]] = []
    token_pred_content: list[list[str]] = []
    tweet_labels: list[int] = []
    gold_for_span_sequences: list[list[str]] = []
    pred_for_span_sequences: list[list[str]] = []
    pred_for_tweet_presence: list[list[str]] = []
    rows: list[dict[str, Any]] = []
    token_sarcasm_scores: list[float] = []
    token_gold_binary: list[int] = []
    score_interpretation = (
        "CRF forward-backward marginal P(B-SARC) + P(I-SARC)"
        if getattr(model, "crf", None) is not None
        else "independent emission softmax P(B-SARC) + P(I-SARC)"
    )

    for batch in loader:
        input_ids, attention_mask, labels = _to_device(batch, device)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        emissions = output["emissions"]
        pred_ids = model.decode(emissions, attention_mask.bool())
        if getattr(model, "crf", None) is not None:
            # Correct sequence-aware probabilities from the CRF forward-backward pass.
            tag_probabilities = model.crf.marginals(emissions, attention_mask.bool())
        else:
            # Ablation variants without CRF have independent token softmax probabilities.
            tag_probabilities = torch.softmax(emissions, dim=-1)
        prob_sarc = (tag_probabilities[..., 1] + tag_probabilities[..., 2]).detach().cpu()
        labels_cpu = labels.detach().cpu()
        masks_cpu = attention_mask.detach().cpu().bool()
        for b in range(input_ids.shape[0]):
            seq_len = int(masks_cpu[b].sum().item())
            gold_tags = [ID2LABEL[int(x)] for x in labels_cpu[b, :seq_len].tolist()]
            pred_tags = [ID2LABEL[int(x)] for x in pred_ids[b]]
            if len(gold_tags) != len(pred_tags):
                raise RuntimeError("Internal evaluation error: gold and prediction sequence lengths differ")
            tweet_label = int(batch["sarcastic"][b])
            tweet_labels.append(tweet_label)
            text = batch["text"][b]
            offsets = [tuple(map(int, p)) for p in batch["offset_mapping"][b][:seq_len]]
            # Special tokens have (0, 0) offsets. Exclude them from token metrics and
            # force them to O for sequence/span metrics, so [CLS]/[SEP] cannot count
            # as predicted sarcasm tokens or spans. Calculate this mask once per tweet.
            content_mask = [0 <= start < end <= len(text) for start, end in offsets]
            content_positions = [j for j, is_content in enumerate(content_mask) if is_content]
            token_gold_content.append([gold_tags[j] for j in content_positions])
            token_pred_content.append([pred_tags[j] for j in content_positions])
            gold_for_span = [tag if content_mask[j] else "O" for j, tag in enumerate(gold_tags)]
            pred_for_span = [tag if content_mask[j] else "O" for j, tag in enumerate(pred_tags)]
            gold_for_span_sequences.append(gold_for_span)
            pred_for_span_sequences.append(pred_for_span)
            pred_for_tweet_presence.append(pred_for_span)
            char_gold = bio_to_char_spans(gold_for_span, offsets, text)
            char_pred = bio_to_char_spans(pred_for_span, offsets, text)
            rows.append({
                "row_index": int(batch["row_index"][b]),
                "sarcastic": tweet_label,
                "tweet": batch["raw_text"][b],
                "model_text": text,
                "gold_bio_sequence": " | ".join(gold_tags),
                "predicted_bio_sequence": " | ".join(pred_tags),
                "annotated_sarcastic_span": batch["annotated_span"][b],
                "annotated_character_span": batch["annotated_char_span"][b],
                "gold_char_spans_from_bio": json.dumps(char_gold, ensure_ascii=False),
                "predicted_char_spans": json.dumps(char_pred, ensure_ascii=False),
                "predicted_span_texts": json.dumps([s["text"] for s in char_pred], ensure_ascii=False),
            })
            # Ignore [CLS], [SEP], padding, and zero-width offsets for token ROC/PR.
            # Scores are CRF marginals for BBiC/CRF variants, or emission softmax for non-CRF ablations.
            for j, (start, end) in enumerate(offsets):
                if 0 <= start < end <= len(text):
                    token_gold_binary.append(int(gold_tags[j] in ("B-SARC", "I-SARC")))
                    token_sarcasm_scores.append(float(prob_sarc[b, j].item()))

    metrics = token_metrics(token_gold_content, token_pred_content)
    metrics.update(span_metrics(gold_for_span_sequences, pred_for_span_sequences))
    metrics.update(tweet_presence_metrics(tweet_labels, pred_for_tweet_presence))
    score_data = {
        "token_gold_binary": token_gold_binary,
        "token_sarcasm_score": token_sarcasm_scores,
        "score_interpretation": score_interpretation,
    }
    return metrics, pd.DataFrame(rows), score_data


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup followed by linear decay, without importing HF scheduler internals."""
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))

    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = total_steps - step
        return max(0.0, float(remaining) / float(max(1, total_steps - warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
