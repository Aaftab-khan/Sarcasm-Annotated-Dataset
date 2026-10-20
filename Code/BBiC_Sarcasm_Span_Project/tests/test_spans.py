import unittest
from src.spans import parse_char_span, validate_bio_sequence, bio_to_token_spans, bio_to_char_spans


class TestSpanUtilities(unittest.TestCase):
    def test_half_open_character_offsets(self):
        self.assertEqual(parse_char_span("[22, 30)"), (22, 30))
        with self.assertRaises(ValueError):
            parse_char_span("[124, 30)")

    def test_bio_spans_use_half_open_token_indices(self):
        self.assertEqual(bio_to_token_spans(["O", "B-SARC", "I-SARC", "O"]), [(1, 3)])

    def test_invalid_i_transition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Illegal I-SARC"):
            validate_bio_sequence(["O", "I-SARC"])

    def test_character_span_from_wordpiece_offsets(self):
        text = "thanks #blessed 😊"
        tags = ["O", "O", "B-SARC", "I-SARC", "O"]
        offsets = [(0, 0), (0, 6), (7, 8), (8, 15), (16, 17)]
        spans = bio_to_char_spans(tags, offsets, text)
        self.assertEqual(spans, [{"start": 7, "end": 15, "text": "#blessed"}])


if __name__ == "__main__":
    unittest.main()
