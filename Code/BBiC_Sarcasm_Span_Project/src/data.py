"""Dataset preparation and batch collation for BBiC."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_tweet
from .spans import LABEL2ID, parse_bio_sequence, validate_bio_sequence


@dataclass
class DatasetColumns:
    text: str = "tweet"
    label: str = "sarcastic"
    processed_text: str = "Preprocessed Tweet"
    span: str = "Sarcastic Span"
    char_span: str = "Character Span"
    bio: str = "BIO Label Sequence"


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Read CSV or Excel while preserving blank annotation cells as empty strings."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, keep_default_na=False)
    else:
        raise ValueError(f"Unsupported dataset format {suffix!r}; use CSV or XLSX")
    return df


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def select_model_text(row: pd.Series | dict[str, Any], columns: DatasetColumns) -> str:
    """Use annotation-aligned processed text when provided; preprocess raw negatives."""
    raw_text = _clean_cell(row.get(columns.text, ""))
    processed_text = _clean_cell(row.get(columns.processed_text, ""))
    try:
        is_sarcastic = int(row.get(columns.label, 0)) == 1
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid sarcasm label: {row.get(columns.label)!r}") from exc

    if is_sarcastic:
        if not processed_text.strip():
            raise ValueError("Sarcastic examples require their existing Preprocessed Tweet text")
        # Do not alter annotated text: its BIO labels were created against this string.
        return processed_text
    # Non-sarcastic rows have no span-derived labels. Apply the documented conservative
    # pipeline to their raw text at runtime; no annotation cells are modified.
    text = processed_text if processed_text.strip() else preprocess_tweet(raw_text)
    if not text.strip():
        raise ValueError("Empty tweet text is not a valid training example")
    return text


class TweetSpanDataset(Dataset):
    """Tokenizer-aligned dataset; uses saved BIO labels for positives and all-O for negatives."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: Any,
        max_length: int = 128,
        columns: DatasetColumns | None = None,
        split_name: str = "dataset",
    ) -> None:
        if max_length < 3:
            raise ValueError("max_length must be at least 3 to hold [CLS], content, and [SEP]")
        self.df = dataframe.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.columns = columns or DatasetColumns()
        self.split_name = split_name
        if not getattr(tokenizer, "is_fast", True):
            raise ValueError("A fast Hugging Face tokenizer is required for offset mappings")
        required = [self.columns.text, self.columns.label, self.columns.processed_text,
                    self.columns.span, self.columns.char_span, self.columns.bio]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required dataset column(s): {missing}")
        # Tokenize and validate once, rather than repeating tokenization every epoch.
        self.features = [self._build_item(i) for i in range(len(self.df))]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]

    def _build_item(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        try:
            sarcastic = int(row[self.columns.label])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{self.split_name} row {index}: invalid sarcastic label") from exc
        if sarcastic not in (0, 1):
            raise ValueError(f"{self.split_name} row {index}: label must be 0 or 1")
        raw_text = _clean_cell(row[self.columns.text])
        text = select_model_text(row, self.columns)
        if not text.strip():
            raise ValueError(f"{self.split_name} row {index}: empty model text")

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        full_input_ids = list(encoding["input_ids"])
        full_offsets = [tuple(map(int, x)) for x in encoding["offset_mapping"]]
        if len(full_input_ids) != len(full_offsets):
            raise ValueError(f"{self.split_name} row {index}: tokenizer IDs/offsets length mismatch")
        if not full_input_ids:
            raise ValueError(f"{self.split_name} row {index}: tokenizer produced an empty sequence")

        if sarcastic == 1:
            tags = parse_bio_sequence(row[self.columns.bio])
            if not tags:
                raise ValueError(f"{self.split_name} row {index}: sarcastic row has blank BIO labels")
            if len(tags) != len(full_input_ids):
                raise ValueError(
                    f"{self.split_name} row {index}: existing BIO length={len(tags)} but tokenizer "
                    f"length={len(full_input_ids)} for the existing Preprocessed Tweet. Do not auto-fix "
                    "or regenerate the annotation; inspect tokenizer/model version and annotation alignment."
                )
            validate_bio_sequence(tags, require_one_span=True)
            label_ids = [LABEL2ID[tag] for tag in tags]
        else:
            # Non-sarcastic tweets have no gold sarcasm span in the spreadsheet.
            # For sequence training, every real token, including [CLS]/[SEP], is O.
            label_ids = [LABEL2ID["O"]] * len(full_input_ids)

        if len(full_input_ids) > self.max_length:
            # Keep [CLS] and the first max_length-2 content tokens, but always retain
            # final [SEP]. Never silently drop an annotated positive token: that would
            # turn a gold span into an incomplete or missing training target.
            keep = list(range(self.max_length - 1)) + [len(full_input_ids) - 1]
            dropped_content = range(self.max_length - 1, len(full_input_ids) - 1)
            if sarcastic == 1 and any(
                label_ids[position] in (LABEL2ID["B-SARC"], LABEL2ID["I-SARC"])
                for position in dropped_content
            ):
                raise ValueError(
                    f"{self.split_name} row {index}: max_length={self.max_length} would truncate "
                    "an annotated sarcasm span. Increase max_length or explicitly review this example; "
                    "the saved annotation will not be changed."
                )
            input_ids = [full_input_ids[i] for i in keep]
            offsets = [full_offsets[i] for i in keep]
            label_ids = [label_ids[i] for i in keep]
        else:
            input_ids, offsets = full_input_ids, full_offsets

        return {
            "input_ids": input_ids,
            "labels": label_ids,
            "offset_mapping": offsets,
            "text": text,
            "raw_text": raw_text,
            "row_index": int(index),
            "sarcastic": sarcastic,
            "annotated_span": _clean_cell(row[self.columns.span]),
            "annotated_char_span": _clean_cell(row[self.columns.char_span]),
        }


def collate_batch(features: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    if not features:
        raise ValueError("Cannot collate an empty batch")
    max_len = max(len(f["input_ids"]) for f in features)
    batch_size = len(features)
    input_ids = torch.full((batch_size, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    labels = torch.full((batch_size, max_len), LABEL2ID["O"], dtype=torch.long)
    offsets: list[list[tuple[int, int]]] = []
    for b, feature in enumerate(features):
        n = len(feature["input_ids"])
        if len(feature["labels"]) != n or len(feature["offset_mapping"]) != n:
            raise ValueError("input_ids, labels, and offset_mapping lengths must match")
        input_ids[b, :n] = torch.tensor(feature["input_ids"], dtype=torch.long)
        labels[b, :n] = torch.tensor(feature["labels"], dtype=torch.long)
        attention_mask[b, :n] = True
        offsets.append(list(feature["offset_mapping"]) + [(0, 0)] * (max_len - n))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "offset_mapping": offsets,
        "text": [f["text"] for f in features],
        "raw_text": [f["raw_text"] for f in features],
        "row_index": [f["row_index"] for f in features],
        "sarcastic": [f["sarcastic"] for f in features],
        "annotated_span": [f["annotated_span"] for f in features],
        "annotated_char_span": [f["annotated_char_span"] for f in features],
    }
