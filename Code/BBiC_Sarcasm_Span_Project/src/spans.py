"""BIO sequence and character-span utilities."""
from __future__ import annotations

import re
from typing import Iterable

LABELS = ("O", "B-SARC", "I-SARC")
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
_OFFSET_RE = re.compile(r"^\s*\[\s*(\d+)\s*,\s*(\d+)\s*\)\s*$")


def parse_bio_sequence(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    tags = [part.strip() for part in text.split("|")]
    invalid = sorted(set(tags) - set(LABEL2ID))
    if invalid:
        raise ValueError(f"Unknown BIO tag(s): {invalid}")
    return tags


def parse_char_span(value: object) -> tuple[int, int] | None:
    if value is None or not str(value).strip() or str(value).strip().lower() == "nan":
        return None
    match = _OFFSET_RE.fullmatch(str(value))
    if not match:
        raise ValueError(f"Invalid character span syntax: {value!r}; expected [start, end)")
    start, end = map(int, match.groups())
    if start < 0 or end <= start:
        raise ValueError(f"Invalid character span bounds: {value!r}")
    return start, end


def validate_bio_sequence(tags: list[str], require_one_span: bool = False) -> None:
    if not tags:
        raise ValueError("BIO sequence is empty")
    previous = "O"
    for position, tag in enumerate(tags):
        if tag == "I-SARC" and previous not in ("B-SARC", "I-SARC"):
            raise ValueError(f"Illegal I-SARC at position {position}; preceding tag is {previous}")
        previous = tag
    if require_one_span and not any(tag in ("B-SARC", "I-SARC") for tag in tags):
        raise ValueError("Sarcastic row has no positive BIO tags")


def bio_to_token_spans(tags: Iterable[str]) -> list[tuple[int, int]]:
    """Convert BIO tags to half-open token-index spans. Invalid I starts a span defensively."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag == "B-SARC":
            if start is not None:
                spans.append((start, i))
            start = i
        elif tag == "I-SARC":
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    return spans


def bio_to_char_spans(tags: list[str], offsets: list[tuple[int, int]], text: str) -> list[dict[str, object]]:
    """Convert predicted BIO WordPiece labels to character spans using fast-tokenizer offsets."""
    if len(tags) != len(offsets):
        raise ValueError(f"Tag/offset mismatch: {len(tags)} tags vs {len(offsets)} offsets")
    spans: list[dict[str, object]] = []
    active_start: int | None = None
    active_end: int | None = None

    def close_span() -> None:
        nonlocal active_start, active_end
        if active_start is not None and active_end is not None and active_end > active_start:
            spans.append({
                "start": active_start,
                "end": active_end,
                "text": text[active_start:active_end],
            })
        active_start, active_end = None, None

    for tag, pair in zip(tags, offsets):
        start, end = int(pair[0]), int(pair[1])
        valid_offset = 0 <= start < end <= len(text)
        if not valid_offset:
            close_span()  # [CLS], [SEP], and padding have (0, 0) offsets.
            continue
        if tag == "B-SARC":
            close_span()
            active_start, active_end = start, end
        elif tag == "I-SARC":
            if active_start is None:
                active_start, active_end = start, end
            else:
                active_start = min(active_start, start)
                active_end = max(active_end if active_end is not None else end, end)
        else:
            close_span()
    close_span()
    return spans
