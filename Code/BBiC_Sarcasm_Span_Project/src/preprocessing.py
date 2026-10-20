"""Conservative preprocessing for raw, non-annotated tweets.

Annotated sarcastic rows use their existing ``Preprocessed Tweet`` field because
BIO positions were created against that text. Do not re-normalize annotated text
here: URL/mention removal or character normalization could invalidate its labels.
"""
from __future__ import annotations

import html
import re

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<![\w])@[A-Za-z0-9_]+")
_LONG_CHAR_RE = re.compile(r"([A-Za-z])\1{3,}")
_SPACE_RE = re.compile(r"\s+")


def _normalize_excessive_repetition(text: str) -> str:
    """Reduce runs of 4+ repeated ASCII letters to 3, retaining emphasis."""
    return _LONG_CHAR_RE.sub(lambda m: m.group(1) * 3, text)


def preprocess_tweet(text: object) -> str:
    """Prepare raw tweets that have no preprocessed annotation field.

    Keeps hashtags, emojis, punctuation, and capitalization of emphatic tokens.
    URLs and user mentions are removed, HTML entities decoded, excessively long
    repeated-letter runs normalized, and whitespace standardized. The rule is
    deliberately conservative because sarcastic intent can depend on these cues.
    """
    if text is None:
        return ""
    value = html.unescape(str(text))

    def remove_url(match: re.Match[str]) -> str:
        found = match.group(0)
        url = found.rstrip(".,!?;:)]}")
        trailing_punctuation = found[len(url):]
        return " " + trailing_punctuation

    value = _URL_RE.sub(remove_url, value)
    value = _MENTION_RE.sub(" ", value)
    value = _normalize_excessive_repetition(value)
    value = _SPACE_RE.sub(" ", value).strip()

    # Preserve all-caps words as emphasis; lowercase other tokens. BERT-base-uncased
    # is case-insensitive; this preserves the project normalization convention.
    normalized = []
    for token in value.split(" "):
        has_letters = any(ch.isalpha() for ch in token)
        is_emphatic_caps = has_letters and token.upper() == token and token.lower() != token
        normalized.append(token if is_emphatic_caps else token.lower())
    return " ".join(normalized)
