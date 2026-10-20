import unittest

from src.encoding_repair import repair_mojibake


class TestRawTweetEncodingRepair(unittest.TestCase):
    def test_curly_apostrophe(self):
        self.assertEqual(repair_mojibake("Iâ€™m"), "I’m")

    def test_emoji_and_variant_selector(self):
        self.assertEqual(repair_mojibake("ðŸ¤¡"), "🤡")
        self.assertEqual(repair_mojibake("âš½ï¸\x8f"), "⚽️")

    def test_symbols_and_mixed_sentence(self):
        self.assertEqual(repair_mojibake("grÃ¢ce"), "grâce")
        self.assertEqual(repair_mojibake("â‰\xa0"), "≠")

    def test_unrecoverable_replacement_character_is_not_guessed(self):
        self.assertEqual(repair_mojibake("Temp: 5.7ï¿½C"), "Temp: 5.7�C")

    def test_already_clean_text_is_unchanged(self):
        original = "This is normal text with an emoji 😊 and a curly apostrophe ’."
        self.assertEqual(repair_mojibake(original), original)


if __name__ == "__main__":
    unittest.main()
