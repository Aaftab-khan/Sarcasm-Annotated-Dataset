#!/usr/bin/env python3
"""Audit data counts, untouched annotations, offsets, and (optionally) BERT alignment."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data import DatasetColumns, read_dataset
from src.spans import parse_bio_sequence, parse_char_span, validate_bio_sequence


def audit_file(path: str, split_name: str, check_tokenizer: bool, model_name: str) -> dict:
    df = read_dataset(path)
    cols = DatasetColumns()
    missing = [c for c in (cols.text, cols.label, cols.processed_text, cols.span, cols.char_span, cols.bio) if c not in df]
    if missing:
        raise ValueError(f"{split_name}: missing required columns: {missing}")
    labels = df[cols.label].astype(int)
    counts = {int(k): int(v) for k, v in labels.value_counts().items()}
    expected = {"train": {1: 867, 0: 2600}, "test": {1: 200, 0: 1200}}
    if split_name in expected and counts != expected[split_name]:
        raise ValueError(f"{split_name}: expected class counts {expected[split_name]}, found {counts}")
    if df[cols.text].astype(str).str.strip().eq("").any():
        empty = df.index[df[cols.text].astype(str).str.strip().eq("")].tolist()[:10]
        raise ValueError(f"{split_name}: empty original tweet text at rows {empty}")

    offset_errors = []
    substring_errors = []
    case_only_mismatches = []
    bio_errors = []
    negative_field_errors = []
    for idx, row in df.iterrows():
        y = int(row[cols.label])
        raw = str(row[cols.text])
        processed = str(row[cols.processed_text]).strip()
        span = str(row[cols.span]).strip()
        char_span = str(row[cols.char_span]).strip()
        bio = str(row[cols.bio]).strip()
        if y == 0:
            if any(value and value.lower() != "nan" for value in (span, char_span, bio)):
                negative_field_errors.append((idx, "span/offset/BIO must be blank"))
            continue
        if not processed:
            bio_errors.append((idx, "missing Preprocessed Tweet"))
            continue
        if not span or not char_span or not bio:
            bio_errors.append((idx, "missing span, character offsets, or BIO sequence"))
            continue
        try:
            offsets = parse_char_span(char_span)
            if offsets is None:
                raise ValueError("empty character span")
            start, end = offsets
            if end > len(processed):
                raise ValueError(f"offset {char_span} exceeds processed text length {len(processed)}")
            fragment = processed[start:end]
            if fragment != span:
                if fragment.casefold() == span.casefold():
                    case_only_mismatches.append((idx, fragment, span, char_span))
                else:
                    substring_errors.append((idx, fragment, span, char_span))
            tags = parse_bio_sequence(bio)
            validate_bio_sequence(tags, require_one_span=True)
        except Exception as exc:
            offset_errors.append((idx, str(exc)))

    if offset_errors or substring_errors or bio_errors or negative_field_errors:
        raise ValueError(
            f"{split_name} annotation audit failed: offset/BIO errors={len(offset_errors)}, "
            f"substring mismatches={len(substring_errors)}, sarcastic-field errors={len(bio_errors)}, "
            f"non-sarcastic-field errors={len(negative_field_errors)}. "
            f"Examples: offsets={offset_errors[:3]}, substrings={substring_errors[:3]}, "
            f"BIO={bio_errors[:3]}, negative={negative_field_errors[:3]}"
        )

    result = {
        "split": split_name,
        "rows": len(df),
        "class_counts": {"sarcastic": counts.get(1, 0), "non_sarcastic": counts.get(0, 0)},
        "empty_tweets": 0,
        "offset_range_errors": len(offset_errors),
        "case_insensitive_substring_mismatches": len(substring_errors),
        "case_only_span_differences": len(case_only_mismatches),
        "invalid_bio_sequences": len(bio_errors),
        "non_sarcastic_span_field_errors": len(negative_field_errors),
        "tokenizer_check": "not requested",
    }
    if check_tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt to run the WordPiece-tokenizer alignment audit") from exc
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        mismatches = []
        for idx, row in df[labels.eq(1)].iterrows():
            text = str(row[cols.processed_text])
            tags = parse_bio_sequence(row[cols.bio])
            tokenized = tokenizer(text, add_special_tokens=True, truncation=False)
            if len(tags) != len(tokenized["input_ids"]):
                mismatches.append((idx, len(tags), len(tokenized["input_ids"])))
        result["tokenizer_check"] = "passed" if not mismatches else "failed"
        result["wordpiece_alignment_mismatches"] = len(mismatches)
        if mismatches:
            raise ValueError(
                f"{split_name}: {len(mismatches)} saved BIO sequences do not match {model_name} token counts; "
                f"examples {mismatches[:10]}. Do not auto-regenerate labels; investigate the tokenizer/version."
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BBiC span datasets")
    parser.add_argument("--train-file", default="data/processed/TrainAnnotatedSpansBIO.csv")
    parser.add_argument("--test-file", default="data/processed/TestAnnotatedSpansBIO.csv")
    parser.add_argument("--check-tokenizer", action="store_true", help="Also validate saved BIO lengths against full BERT WordPiece sequences")
    parser.add_argument("--model-name", default="bert-base-uncased")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    train_path = Path(args.train_file)
    test_path = Path(args.test_file)
    if not train_path.is_absolute():
        train_path = root / train_path
    if not test_path.is_absolute():
        test_path = root / test_path
    results = [
        audit_file(str(train_path), "train", args.check_tokenizer, args.model_name),
        audit_file(str(test_path), "test", args.check_tokenizer, args.model_name),
    ]
    for result in results:
        print(result)
    print("Structural dataset audit passed.")
    if args.check_tokenizer:
        print("BERT WordPiece alignment audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
