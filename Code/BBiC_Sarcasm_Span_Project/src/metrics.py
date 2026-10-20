"""Token- and span-level evaluation metrics for sarcasm-span extraction."""
from __future__ import annotations

from typing import Sequence

from .spans import LABELS, bio_to_token_spans


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def token_metrics(gold_sequences: Sequence[Sequence[str]], pred_sequences: Sequence[Sequence[str]]) -> dict:
    if len(gold_sequences) != len(pred_sequences):
        raise ValueError("Gold and predicted sequence counts differ")
    confusion = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    gold_flat: list[str] = []
    pred_flat: list[str] = []
    for row_idx, (gold, pred) in enumerate(zip(gold_sequences, pred_sequences)):
        if len(gold) != len(pred):
            raise ValueError(f"Row {row_idx}: gold/pred token lengths differ ({len(gold)} vs {len(pred)})")
        for g, p in zip(gold, pred):
            if g not in LABELS or p not in LABELS:
                raise ValueError(f"Unknown label in evaluation: gold={g!r}, pred={p!r}")
            confusion[g][p] += 1
            gold_flat.append(g)
            pred_flat.append(p)
    per_label = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        per_label[label] = _prf(tp, fp, fn)
    # Main token metrics treat B-SARC and I-SARC as the positive sarcasm-token class.
    tp = sum(1 for g, p in zip(gold_flat, pred_flat) if g in ("B-SARC", "I-SARC") and p in ("B-SARC", "I-SARC"))
    fp = sum(1 for g, p in zip(gold_flat, pred_flat) if g == "O" and p in ("B-SARC", "I-SARC"))
    fn = sum(1 for g, p in zip(gold_flat, pred_flat) if g in ("B-SARC", "I-SARC") and p == "O")
    pos = _prf(tp, fp, fn)
    total = len(gold_flat)
    correct = sum(1 for g, p in zip(gold_flat, pred_flat) if g == p)
    macro_f1 = sum(per_label[x]["f1"] for x in LABELS) / len(LABELS)
    return {
        "token_sarcasm_precision": pos["precision"],
        "token_sarcasm_recall": pos["recall"],
        "token_sarcasm_f1": pos["f1"],
        "token_accuracy": correct / total if total else 0.0,
        "token_macro_f1": macro_f1,
        "per_label": per_label,
        "confusion_matrix": [[confusion[g][p] for p in LABELS] for g in LABELS],
        "confusion_labels": list(LABELS),
        "token_count": total,
    }


def _maximum_overlap_matching_count(pred_spans: list[tuple[int, int]], gold_spans: list[tuple[int, int]]) -> int:
    """Maximum one-to-one matching of predicted/gold spans with any token overlap."""
    adjacency = []
    for ps, pe in pred_spans:
        adjacency.append([j for j, (gs, ge) in enumerate(gold_spans) if max(ps, gs) < min(pe, ge)])
    matched_gold: dict[int, int] = {}

    def augment(pred_idx: int, seen: set[int]) -> bool:
        for gold_idx in adjacency[pred_idx]:
            if gold_idx in seen:
                continue
            seen.add(gold_idx)
            if gold_idx not in matched_gold or augment(matched_gold[gold_idx], seen):
                matched_gold[gold_idx] = pred_idx
                return True
        return False

    count = 0
    for pi in range(len(pred_spans)):
        if augment(pi, set()):
            count += 1
    return count


def span_metrics(gold_sequences: Sequence[Sequence[str]], pred_sequences: Sequence[Sequence[str]]) -> dict:
    if len(gold_sequences) != len(pred_sequences):
        raise ValueError("Gold and predicted sequence counts differ")
    exact_tp = exact_pred = exact_gold = 0
    partial_tp = partial_pred = partial_gold = 0
    for row_idx, (gold, pred) in enumerate(zip(gold_sequences, pred_sequences)):
        if len(gold) != len(pred):
            raise ValueError(f"Row {row_idx}: gold/pred token lengths differ")
        gs = bio_to_token_spans(gold)
        ps = bio_to_token_spans(pred)
        gs_set, ps_set = set(gs), set(ps)
        exact_tp += len(gs_set & ps_set)
        exact_gold += len(gs_set)
        exact_pred += len(ps_set)
        partial_tp += _maximum_overlap_matching_count(ps, gs)
        partial_gold += len(gs)
        partial_pred += len(ps)
    exact = _prf(exact_tp, exact_pred - exact_tp, exact_gold - exact_tp)
    partial = _prf(partial_tp, partial_pred - partial_tp, partial_gold - partial_tp)
    return {
        "exact_span_precision": exact["precision"],
        "exact_span_recall": exact["recall"],
        "exact_span_f1": exact["f1"],
        "partial_span_precision": partial["precision"],
        "partial_span_recall": partial["recall"],
        "partial_span_f1": partial["f1"],
        "gold_span_count": exact_gold,
        "predicted_span_count": exact_pred,
        "exact_matched_spans": exact_tp,
        "partial_matched_spans": partial_tp,
        "partial_match_definition": "one-to-one maximum matching; any shared token counts as overlap",
    }


def tweet_presence_metrics(gold_tweet_labels: Sequence[int], pred_sequences: Sequence[Sequence[str]]) -> dict:
    """Auxiliary tweet-level view: positive if any predicted sarcasm token exists."""
    if len(gold_tweet_labels) != len(pred_sequences):
        raise ValueError("Tweet labels and predicted sequence counts differ")
    tp = fp = tn = fn = 0
    for gold, pred in zip(gold_tweet_labels, pred_sequences):
        predicted_positive = any(tag in ("B-SARC", "I-SARC") for tag in pred)
        if int(gold) == 1 and predicted_positive:
            tp += 1
        elif int(gold) == 0 and predicted_positive:
            fp += 1
        elif int(gold) == 0 and not predicted_positive:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn - fp * fn) / denom**0.5) if denom else 0.0
    return {
        "tweet_presence_accuracy": (tp + tn) / total if total else 0.0,
        "tweet_presence_precision": tp / (tp + fp) if tp + fp else 0.0,
        "tweet_presence_recall": tp / (tp + fn) if tp + fn else 0.0,
        "tweet_presence_f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "tweet_presence_confusion_matrix": [[tn, fp], [fn, tp]],
        "tweet_presence_mcc": mcc,
    }
