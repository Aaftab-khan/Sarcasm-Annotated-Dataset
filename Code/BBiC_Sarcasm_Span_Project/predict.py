#!/usr/bin/env python3
"""Extract predicted sarcasm-bearing spans from one raw tweet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.model import BBiCModel
from src.engine import resolve_device
from src.preprocessing import preprocess_tweet
from src.spans import ID2LABEL, bio_to_char_spans


def parse_args():
    p = argparse.ArgumentParser(description="Predict sarcasm-bearing spans with a trained BBiC checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", required=True, help="Raw tweet text")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    run_cfg = checkpoint["run_config"]
    model_cfg = run_cfg["model"]
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency: install requirements.txt first.") from exc
    tokenizer_dir = checkpoint_path.parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, local_files_only=True)
    text = preprocess_tweet(args.text)
    encoded = tokenizer(text, add_special_tokens=True, truncation=True,
                        max_length=int(model_cfg["max_length"]), return_offsets_mapping=True,
                        return_tensors="pt")
    offsets = [tuple(map(int, x)) for x in encoded.pop("offset_mapping")[0].tolist()]
    device = resolve_device(args.device)
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
    model.to(device).eval()
    inputs = {k: v.to(device) for k, v in encoded.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
    with torch.no_grad():
        output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        path = model.decode(output["emissions"], inputs["attention_mask"].bool())[0]
    tags = [ID2LABEL[int(x)] for x in path]
    spans = bio_to_char_spans(tags, offsets, text)
    print(json.dumps({
        "input_text": args.text,
        "model_text": text,
        "predicted_bio_sequence": " | ".join(tags),
        "predicted_spans": spans,
        "has_predicted_sarcasm_span": bool(spans),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
