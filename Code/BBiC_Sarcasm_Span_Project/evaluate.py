#!/usr/bin/env python3
"""Evaluate a saved BBiC checkpoint on the held-out test set."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.data import DatasetColumns, TweetSpanDataset, read_dataset
from src.engine import evaluate_model, make_loader, resolve_device, write_json
from src.model import BBiCModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved BBiC checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to best_checkpoint.pt")
    p.add_argument("--test-file", default=None, help="Defaults to test_file in the saved run configuration")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def _save_confusion_matrix(matrix: list[list[int]], labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(np.asarray(matrix), interpolation="nearest")
    ax.figure.colorbar(image, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           ylabel="Gold BIO label", xlabel="Predicted BIO label",
           title="Token-level BIO confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    threshold = max(max(row) for row in matrix) / 2 if matrix and max(map(max, matrix)) else 0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:,}", ha="center", va="center",
                    color="white" if value > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _save_roc_pr(score_data: dict, output_dir: Path) -> dict:
    y = np.asarray(score_data["token_gold_binary"], dtype=np.int32)
    scores = np.asarray(score_data["token_sarcasm_score"], dtype=np.float64)
    interpretation = score_data["score_interpretation"]
    result = {"score_interpretation": interpretation, "tokens": int(len(y))}
    if len(y) == 0 or len(np.unique(y)) < 2:
        result["status"] = "Skipped: token labels contain fewer than two classes."
        return result
    fpr, tpr, _ = roc_curve(y, scores)
    auc_value = float(roc_auc_score(y, scores))
    precision, recall, _ = precision_recall_curve(y, scores)
    ap_value = float(average_precision_score(y, scores))
    result.update({"roc_auc": auc_value, "average_precision": ap_value})

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Token ROC using model tag probabilities")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "token_roc_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(recall, precision, label=f"Average precision = {ap_value:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Token precision-recall using model tag probabilities")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "token_pr_curve.png", dpi=300)
    plt.close(fig)
    return result


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    run_cfg = checkpoint["run_config"]
    model_cfg = run_cfg["model"]
    training_cfg = run_cfg["training"]
    output_dir = Path(args.output_dir or checkpoint_path.parent / "test_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency: install requirements.txt first.") from exc
    tokenizer_dir = checkpoint_path.parent / "tokenizer"
    if not tokenizer_dir.is_dir():
        raise SystemExit(f"Tokenizer assets not found next to checkpoint: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, local_files_only=True)
    encoder_config_dir = checkpoint_path.parent / "encoder_config"
    model = BBiCModel(
        base_model=model_cfg["base_model"],
        bilstm_hidden_size=int(model_cfg["bilstm_hidden_size"]),
        dropout=float(model_cfg["dropout"]),
        variant=run_cfg["variant"],
        encoder_config_path=str(encoder_config_dir),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    data_cfg = run_cfg["data"]
    test_file = Path(args.test_file) if args.test_file else Path(data_cfg["test_file"])
    if not test_file.is_absolute():
        # Resolve from the project root recorded at training time, regardless of cwd.
        root = Path(run_cfg.get("project_root", checkpoint_path.parent.parent.parent))
        test_file = root / test_file
    test_file = str(test_file.resolve())
    df = read_dataset(test_file)
    columns = DatasetColumns(
        text=data_cfg.get("text_column", "tweet"),
        label=data_cfg.get("label_column", "sarcastic"),
        processed_text=data_cfg.get("processed_text_column", "Preprocessed Tweet"),
        span=data_cfg.get("span_column", "Sarcastic Span"),
        char_span=data_cfg.get("char_span_column", "Character Span"),
        bio=data_cfg.get("bio_column", "BIO Label Sequence"),
    )
    dataset = TweetSpanDataset(df, tokenizer, int(model_cfg["max_length"]), columns, "test")
    batch_size = int(args.batch_size or training_cfg["batch_size"])
    loader = make_loader(dataset, batch_size, False, 0, device.type == "cuda", tokenizer.pad_token_id)

    metrics, predictions, score_data = evaluate_model(model, loader, device)
    write_json(output_dir / "metrics.json", metrics)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    _save_confusion_matrix(metrics["confusion_matrix"], metrics["confusion_labels"], output_dir / "token_confusion_matrix.png")
    scores_result = _save_roc_pr(score_data, output_dir)
    write_json(output_dir / "token_curve_metrics.json", scores_result)
    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation artifacts to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
