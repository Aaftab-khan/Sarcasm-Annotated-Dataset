#!/usr/bin/env python3
"""Build the model-ready CSVs from the supplied annotated Excel workbooks.

The Excel workbooks are authoritative for row order, labels, and all saved
annotation fields. The misspelled input header ``Proprocessed Tweet`` is
standardized to ``Preprocessed Tweet`` in the CSV only. A blank-text negative
row is excluded from the model-ready CSV because it cannot be tokenized; the
original workbook is copied byte-for-byte and retained in data/source/.

Only the raw ``tweet`` cell is repaired for encoding. Positive raw text is
restored in annotation order from the original raw source after an alignment
check. Negative raw text is decoded in place, preserving the new workbook's
negative-row order. No annotation/span/BIO values are regenerated or edited.
"""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
import unicodedata

import pandas as pd

from src.encoding_repair import repair_mojibake
from src.preprocessing import preprocess_tweet

ROOT = Path(__file__).resolve().parent
PROCESSED_TEXT_ALIASES = ("Preprocessed Tweet", "Proprocessed Tweet")
PROTECTED_COLUMNS = [
    "sarcastic",
    "Preprocessed Tweet",
    "Sarcastic Span",
    "Character Span",
    "BIO Label Sequence",
]


def _norm(value: object) -> str:
    return unicodedata.normalize("NFC", str(value)).casefold().strip()


def _sim(a: object, b: object) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b), autojunk=False).ratio()


def _read_text_source(path: str, text_column: str) -> pd.DataFrame:
    source = pd.read_csv(ROOT / path, keep_default_na=False, encoding="utf-8-sig")
    if text_column not in source.columns or "sarcastic" not in source.columns:
        raise ValueError(f"{path}: expected columns {text_column!r} and 'sarcastic'")
    source["sarcastic"] = source["sarcastic"].astype(int)
    return source


def _standardize_processed_text_header(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    found = [name for name in PROCESSED_TEXT_ALIASES if name in frame.columns]
    if len(found) != 1:
        raise ValueError(
            f"{split}: expected exactly one of {PROCESSED_TEXT_ALIASES}; found {found}"
        )
    if found[0] != "Preprocessed Tweet":
        frame = frame.rename(columns={found[0]: "Preprocessed Tweet"})
    return frame


def repair_split(
    name: str,
    source_csv: str,
    annotation_xlsx: str,
    processed_csv: str,
    source_text_column: str,
    expected_rows: int,
    expected_positive: int,
    minimum_alignment_similarity: float = 0.85,
) -> dict:
    source = _read_text_source(source_csv, source_text_column)
    workbook = pd.read_excel(ROOT / annotation_xlsx, keep_default_na=False)
    workbook = _standardize_processed_text_header(workbook, name)

    required = ["tweet", *PROTECTED_COLUMNS]
    missing = [column for column in required if column not in workbook.columns]
    if missing:
        raise ValueError(f"{name}: workbook lacks required columns: {missing}")

    # Make sure labels are well-formed before filtering any unusable rows.
    try:
        workbook["sarcastic"] = workbook["sarcastic"].astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: sarcastic labels must be integers 0 or 1") from exc
    if not workbook["sarcastic"].isin([0, 1]).all():
        raise ValueError(f"{name}: sarcastic labels must be 0 or 1")

    blank_text = workbook["tweet"].astype(str).str.strip().eq("")
    blank_positive = blank_text & workbook["sarcastic"].eq(1)
    if blank_positive.any():
        rows = workbook.index[blank_positive].tolist()[:10]
        raise ValueError(f"{name}: cannot exclude blank-text positive rows: {rows}")

    # Empty negative examples may be omitted from model input only if all gold
    # annotation cells are also blank. The source workbook itself is never edited.
    blank_negative = blank_text & workbook["sarcastic"].eq(0)
    annotation_cells = ["Preprocessed Tweet", "Sarcastic Span", "Character Span", "BIO Label Sequence"]
    for column in annotation_cells:
        dirty = blank_negative & workbook[column].astype(str).str.strip().ne("")
        if dirty.any():
            raise ValueError(f"{name}: blank-text negatives contain {column!r} values")
    dropped_blank_rows = int(blank_text.sum())
    merged = workbook.loc[~blank_text].copy().reset_index(drop=True)

    positives = merged.loc[merged["sarcastic"].eq(1)].copy().reset_index(drop=True)
    negatives = merged.loc[merged["sarcastic"].eq(0)].copy().reset_index(drop=True)
    source_positive = source.loc[source["sarcastic"].eq(1)].copy().reset_index(drop=True)
    source_negative = source.loc[
        source["sarcastic"].eq(0) & source[source_text_column].astype(str).str.strip().ne("")
    ].copy().reset_index(drop=True)

    if len(merged) != expected_rows:
        raise ValueError(f"{name}: expected {expected_rows} usable rows, found {len(merged)}")
    if len(positives) != expected_positive or len(source_positive) != expected_positive:
        raise ValueError(
            f"{name}: expected {expected_positive} positives, workbook has {len(positives)} "
            f"and source has {len(source_positive)}"
        )
    if len(negatives) != expected_rows - expected_positive:
        raise ValueError(f"{name}: unexpected non-sarcastic count {len(negatives)}")

    # Validate positive source order against the provided annotation-aligned text.
    similarities = [
        _sim(positives.iloc[i]["Preprocessed Tweet"], preprocess_tweet(source_positive.iloc[i][source_text_column]))
        for i in range(expected_positive)
    ]
    low = [i for i, score in enumerate(similarities) if score < minimum_alignment_similarity]
    if low:
        examples = [(i, round(similarities[i], 4)) for i in low[:10]]
        raise ValueError(f"{name}: positive source order/alignment check failed: {examples}")

    # Restore each workbook raw cell with a reversible decode, then verify the
    # text exactly matches the source collection. Positive rows must match in
    # order; negative examples must match as a multiset while retaining workbook
    # order in the output.
    decoded_positive = [repair_mojibake(value) for value in positives["tweet"].astype(str).tolist()]
    canonical_positive = source_positive[source_text_column].astype(str).tolist()
    if decoded_positive != canonical_positive:
        mismatches = [
            (i, decoded_positive[i][:100], canonical_positive[i][:100])
            for i in range(min(len(decoded_positive), len(canonical_positive)))
            if decoded_positive[i] != canonical_positive[i]
        ][:5]
        raise ValueError(f"{name}: raw positive text does not decode to source text in order: {mismatches}")

    decoded_negative = [repair_mojibake(value) for value in negatives["tweet"].astype(str).tolist()]
    canonical_negative = source_negative[source_text_column].astype(str).tolist()
    if Counter(decoded_negative) != Counter(canonical_negative):
        extra = list((Counter(decoded_negative) - Counter(canonical_negative)).items())[:3]
        missing_values = list((Counter(canonical_negative) - Counter(decoded_negative)).items())[:3]
        raise ValueError(
            f"{name}: decoded negative rows do not match the source collection. "
            f"Extra={extra}; missing={missing_values}"
        )

    # Preserve workbook order; repair only the raw tweet column. The exact
    # positive source text is used after passing order/alignment checks.
    repaired = merged.copy()
    repaired.loc[repaired["sarcastic"].eq(1), "tweet"] = canonical_positive
    repaired.loc[repaired["sarcastic"].eq(0), "tweet"] = decoded_negative

    # Annotation cells must remain equal to the source workbook for every kept row.
    original_kept = workbook.loc[~blank_text].copy().reset_index(drop=True)
    original_kept = _standardize_processed_text_header(original_kept, name)
    for column in PROTECTED_COLUMNS:
        before = original_kept[column].astype(str).tolist()
        after = repaired[column].astype(str).tolist()
        if before != after:
            raise AssertionError(f"{name}: protected field {column!r} changed during processing")

    if repaired["tweet"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{name}: empty tweet remains in processed output")
    if repaired["sarcastic"].value_counts().to_dict() != {
        1: expected_positive, 0: expected_rows - expected_positive
    }:
        raise ValueError(f"{name}: unexpected final class distribution")

    output_path = ROOT / processed_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "split": name,
        "source_workbook_rows": len(workbook),
        "processed_rows": len(repaired),
        "blank_text_rows_excluded": dropped_blank_rows,
        "positive_rows": expected_positive,
        "negative_rows": expected_rows - expected_positive,
        "raw_tweet_cells_changed_for_encoding_or_source_restoration": int(
            sum(a != b for a, b in zip(merged["tweet"].astype(str).tolist(), repaired["tweet"].astype(str).tolist()))
        ),
        "protected_annotation_fields_changed": 0,
        "positive_order_matches_original_source": True,
        "negative_text_multiset_matches_original_source": True,
        "negative_row_order_preserved_from_uploaded_workbook": True,
        "minimum_positive_alignment_similarity": round(min(similarities), 4),
    }


def main() -> None:
    reports = [
        repair_split(
            "train", "data/source/train_original.csv",
            "data/source/TrainAnnotatedSpansBIO.xlsx",
            "data/processed/TrainAnnotatedSpansBIO.csv", "tweet", 3467, 867,
        ),
        repair_split(
            "test", "data/source/test_original.csv",
            "data/source/TestAnnotatedSpansBIO.xlsx",
            "data/processed/TestAnnotatedSpansBIO.csv", "text", 1400, 200,
        ),
    ]
    for report in reports:
        print(report)
    print("Dataset build complete. Only raw tweet encoding was repaired; saved gold annotations were preserved.")


if __name__ == "__main__":
    main()
