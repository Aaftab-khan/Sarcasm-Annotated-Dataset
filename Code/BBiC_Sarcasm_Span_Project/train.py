#!/usr/bin/env python3
"""Train BBiC or one of the paper's architecture ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.configuration import load_config
from src.data import DatasetColumns, TweetSpanDataset, read_dataset
from src.engine import build_scheduler, make_loader, resolve_device, run_validation, write_json
from src.model import BBiCModel, VARIANTS
from src.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the BERT-BiLSTM-CRF sarcasm-span model")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--train-file", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--variant", choices=sorted(VARIANTS), default="bbic")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    return p.parse_args()


def _optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "LayerNorm.weight" in name or "layer_norm.weight" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def _save_loss_plot(history: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / "training_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(frame["epoch"], frame["train_loss"], label="Training loss")
    ax.plot(frame["epoch"], frame["validation_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CRF negative log-likelihood / token cross-entropy")
    ax.set_title("BBiC training and validation loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    cols_cfg = cfg["data"]
    columns = DatasetColumns(
        text=cols_cfg.get("text_column", "tweet"),
        label=cols_cfg.get("label_column", "sarcastic"),
        processed_text=cols_cfg.get("processed_text_column", "Preprocessed Tweet"),
        span=cols_cfg.get("span_column", "Sarcastic Span"),
        char_span=cols_cfg.get("char_span_column", "Character Span"),
        bio=cols_cfg.get("bio_column", "BIO Label Sequence"),
    )
    train_path = Path(args.train_file) if args.train_file else Path(cols_cfg["train_file"])
    if not train_path.is_absolute():
        train_path = project_root / train_path
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output"]["root"]) / args.variant
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 42))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg["batch_size"])
    epochs = int(args.epochs if args.epochs is not None else train_cfg["epochs"])
    learning_rate = float(args.learning_rate if args.learning_rate is not None else train_cfg["learning_rate"])
    if batch_size < 1 or epochs < 1 or learning_rate <= 0:
        raise ValueError("batch-size, epochs, and learning-rate must be positive")
    set_seed(seed)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency: install packages with `pip install -r requirements.txt`.") from exc

    model_name = model_cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required to use reliable character offsets")
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad_token_id")

    full_df = read_dataset(train_path)
    required = [columns.text, columns.label, columns.processed_text, columns.span, columns.char_span, columns.bio]
    missing = [c for c in required if c not in full_df.columns]
    if missing:
        raise ValueError(f"Training data missing columns: {missing}")
    labels = full_df[columns.label].astype(int).to_numpy()
    indices = list(range(len(full_df)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=float(train_cfg["validation_fraction"]),
        random_state=seed,
        stratify=labels,
    )
    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)

    max_length = int(model_cfg["max_length"])
    train_dataset = TweetSpanDataset(train_df, tokenizer, max_length, columns, "train")
    val_dataset = TweetSpanDataset(val_df, tokenizer, max_length, columns, "validation")
    device = resolve_device(args.device)
    use_pin_memory = bool(train_cfg.get("pin_memory", True)) and device.type == "cuda"
    train_loader = make_loader(train_dataset, batch_size, True, int(train_cfg.get("num_workers", 0)),
                               use_pin_memory, tokenizer.pad_token_id)
    val_loader = make_loader(val_dataset, batch_size, False, int(train_cfg.get("num_workers", 0)),
                             use_pin_memory, tokenizer.pad_token_id)

    model = BBiCModel(
        base_model=model_name,
        bilstm_hidden_size=int(model_cfg["bilstm_hidden_size"]),
        dropout=float(model_cfg["dropout"]),
        variant=args.variant,
        pretrained=True,
    ).to(device)
    optimizer = _optimizer(model, learning_rate, float(train_cfg["weight_decay"]))
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = int(total_steps * float(train_cfg["warmup_ratio"]))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    patience = int(train_cfg["early_stopping_patience"])
    grad_clip = float(train_cfg.get("gradient_clip_norm", 1.0))

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    encoder_config_dir = output_dir / "encoder_config"
    encoder_config_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(encoder_config_dir)

    run_config = {
        "model": dict(model_cfg),
        "training": {**train_cfg, "batch_size": batch_size, "epochs": epochs,
                     "learning_rate": learning_rate, "seed": seed},
        "data": dict(cols_cfg),
        "variant": args.variant,
        "train_file": str(Path(train_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "project_root": str(project_root),
        "device": str(device),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "train_class_counts": {str(k): int(v) for k, v in train_df[columns.label].astype(int).value_counts().items()},
        "validation_class_counts": {str(k): int(v) for k, v in val_df[columns.label].astype(int).value_counts().items()},
        "parameter_note": "Batch size 32 is the confirmed setting. "
                          "Warmup ratio, validation fraction, patience, seed and gradient clipping are explicit run settings.",
    }
    write_json(output_dir / "run_config.json", run_config)
    split_rows = []
    for original_index in train_idx:
        split_rows.append({"source_row_index": int(original_index), "split": "train", "sarcastic": int(full_df.iloc[original_index][columns.label])})
    for original_index in val_idx:
        split_rows.append({"source_row_index": int(original_index), "split": "validation", "sarcastic": int(full_df.iloc[original_index][columns.label])})
    pd.DataFrame(split_rows).sort_values("source_row_index").to_csv(output_dir / "train_validation_split.csv", index=False)

    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    ckpt_path = output_dir / "best_checkpoint.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        example_count = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = out["loss"]
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            size = input_ids.shape[0]
            loss_sum += float(loss.detach().item()) * size
            example_count += size
        train_loss = loss_sum / max(1, example_count)
        val_loss = run_validation(model, val_loader, device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                        "learning_rate": current_lr})
        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | lr={current_lr:.3e}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({
                "model_state_dict": state,
                "run_config": run_config,
                "best_epoch": best_epoch,
                "best_validation_loss": best_val,
            }, ckpt_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping: validation loss did not improve for {patience} consecutive epochs.")
                break

    _save_loss_plot(history, output_dir)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "epochs_completed": len(history),
        "checkpoint": str(ckpt_path.resolve()),
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
