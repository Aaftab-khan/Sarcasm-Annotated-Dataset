#!/usr/bin/env python3
"""Run the four Table 10 architectural variants with the same split and settings."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import json
import pandas as pd

VARIANTS = ("bert_only", "bert_bilstm", "bert_crf", "bbic")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate BBiC ablations")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--output-root", default="outputs/ablations")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = str(config_path.resolve())
    train_script = root / "train.py"
    eval_script = root / "evaluate.py"
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for variant in VARIANTS:
        out = output_root / variant
        train_cmd = [sys.executable, str(train_script), "--config", config, "--variant", variant,
                     "--output-dir", str(out), "--batch-size", str(args.batch_size), "--device", args.device]
        if args.train_file:
            train_cmd.extend(["--train-file", str((root / args.train_file).resolve())])
        if args.epochs is not None:
            train_cmd.extend(["--epochs", str(args.epochs)])
        print("\nTRAIN:", " ".join(train_cmd), flush=True)
        subprocess.run(train_cmd, cwd=root, check=True)
        eval_cmd = [sys.executable, str(eval_script), "--checkpoint", str(out / "best_checkpoint.pt"),
                    "--output-dir", str(out / "test_evaluation"), "--device", args.device]
        if args.test_file:
            eval_cmd.extend(["--test-file", str((root / args.test_file).resolve())])
        print("EVALUATE:", " ".join(eval_cmd), flush=True)
        subprocess.run(eval_cmd, cwd=root, check=True)
        metrics_path = out / "test_evaluation" / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        summary_rows.append({
            "variant": variant,
            "token_precision": metrics["token_sarcasm_precision"],
            "token_recall": metrics["token_sarcasm_recall"],
            "token_f1": metrics["token_sarcasm_f1"],
            "exact_span_precision": metrics["exact_span_precision"],
            "exact_span_recall": metrics["exact_span_recall"],
            "exact_span_f1": metrics["exact_span_f1"],
            "partial_span_precision": metrics["partial_span_precision"],
            "partial_span_recall": metrics["partial_span_recall"],
            "partial_span_f1": metrics["partial_span_f1"],
        })
    pd.DataFrame(summary_rows).to_csv(output_root / "ablation_summary.csv", index=False)
    print(f"All variants completed. Outputs: {output_root}")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
