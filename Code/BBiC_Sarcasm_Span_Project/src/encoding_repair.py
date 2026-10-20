"""Conservatively reverse common UTF-8-as-CP1252/Latin-1 mojibake.

This helper is intended only for the raw ``tweet`` field. It must never be
applied to ``Preprocessed Tweet`` or any gold span/BIO annotation field.
"""
from __future__ import annotations

_MOJIBAKE_MARKERS = (
    "Ã", "Â", "â€", "âš", "â¬", "â˜", "âœ", "â™", "â†", "â‡",
    "âž", "â„", "â­", "â‰", "â–", "â¸", "â\x9d", "ðŸ", "ï¸", "ï¿",
    "Ð", "Ñ", "ãƒ", "Ãƒ", "Ã‚",
)


def _visible_mojibake_to_bytes(text: str) -> bytes | None:
    """Map one visible mojibake layer back to bytes, when unambiguous."""
    output = bytearray()
    for char in text:
        codepoint = ord(char)
        if codepoint <= 0xFF:
            # Includes Latin-1 control code points sometimes produced by CP1252.
            output.append(codepoint)
            continue
        try:
            encoded = char.encode("cp1252")
        except UnicodeEncodeError:
            return None
        if len(encoded) != 1:
            return None
        output.extend(encoded)
    return bytes(output)


def repair_mojibake(text: object, max_rounds: int = 3) -> str:
    """Reverse only clearly marked, reversible mojibake layers in a raw tweet.

    The function accepts a change only when the input contains a known mojibake
    marker and a full byte-round-trip decodes as valid UTF-8. Unrecoverable
    replacement characters (``�``) are retained; their original byte is unknown.
    """
    current = "" if text is None else str(text)
    for _ in range(max_rounds):
        if not any(marker in current for marker in _MOJIBAKE_MARKERS):
            break
        encoded = _visible_mojibake_to_bytes(current)
        if encoded is None:
            break
        try:
            candidate = encoded.decode("utf-8")
        except UnicodeDecodeError:
            break
        if candidate == current:
            break
        before = sum(current.count(marker) for marker in _MOJIBAKE_MARKERS)
        after = sum(candidate.count(marker) for marker in _MOJIBAKE_MARKERS)
        if after >= before:
            break
        current = candidate
    return current
