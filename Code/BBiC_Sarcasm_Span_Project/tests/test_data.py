import re
import unittest
import pandas as pd

from src.data import TweetSpanDataset, collate_batch, DatasetColumns, read_dataset
from src.spans import LABEL2ID
from src.preprocessing import preprocess_tweet


class SimpleFastTokenizer:
    is_fast = True
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, truncation=False, return_offsets_mapping=False, **kwargs):
        pieces = [(m.group(0), (m.start(), m.end())) for m in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE)]
        ids = [101] + [1000 + i for i in range(len(pieces))] + [102]
        offsets = [(0, 0)] + [p[1] for p in pieces] + [(0, 0)]
        if truncation and kwargs.get("max_length") and len(ids) > kwargs["max_length"]:
            n = kwargs["max_length"]
            ids = ids[:n - 1] + [ids[-1]]
            offsets = offsets[:n - 1] + [offsets[-1]]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {"tweet": "Great, this works", "sarcastic": 1,
             "Preprocessed Tweet": "Great, this works", "Sarcastic Span": "Great",
             "Character Span": "[0, 5)", "BIO Label Sequence": "O | B-SARC | O | O | O | O"},
            {"tweet": "WOW!!! @someone see https://example.com #blessed 😊", "sarcastic": 0,
             "Preprocessed Tweet": "", "Sarcastic Span": "", "Character Span": "", "BIO Label Sequence": ""},
        ])

    def test_positive_existing_bio_preserved_and_negative_all_o(self):
        ds = TweetSpanDataset(self.df, SimpleFastTokenizer(), max_length=32)
        positive = ds[0]
        negative = ds[1]
        self.assertEqual(positive["labels"], [0, 1, 0, 0, 0, 0])
        self.assertEqual(negative["sarcastic"], 0)
        self.assertTrue(all(x == LABEL2ID["O"] for x in negative["labels"]))
        self.assertEqual(negative["annotated_span"], "")
        batch = collate_batch([positive, negative], pad_token_id=0)
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        self.assertEqual(batch["attention_mask"].dtype.name if hasattr(batch["attention_mask"].dtype, "name") else str(batch["attention_mask"].dtype), "torch.bool")

    def test_positive_bio_length_mismatch_raises_instead_of_rewriting(self):
        bad = self.df.iloc[[0]].copy()
        bad.loc[bad.index[0], "BIO Label Sequence"] = "O | B-SARC"
        with self.assertRaisesRegex(ValueError, "Do not auto-fix"):
            TweetSpanDataset(bad, SimpleFastTokenizer(), max_length=32)

    def test_truncation_keeps_final_sep_and_label_alignment(self):
        long_df = self.df.iloc[[1]].copy()
        long_df.loc[long_df.index[0], "tweet"] = "one two three four five six seven"
        tokenizer = SimpleFastTokenizer()
        ds = TweetSpanDataset(long_df, tokenizer, max_length=5)
        item = ds[0]
        self.assertEqual(len(item["input_ids"]), 5)
        self.assertEqual(item["input_ids"][-1], 102)
        self.assertEqual(len(item["input_ids"]), len(item["labels"]))

    def test_truncation_that_would_drop_annotated_span_fails_loudly(self):
        long_positive = pd.DataFrame([{
            "tweet": "one two three four five six seven",
            "sarcastic": 1,
            "Preprocessed Tweet": "one two three four five six seven",
            "Sarcastic Span": "six seven",
            "Character Span": "[24, 33)",
            "BIO Label Sequence": "O | O | O | O | O | O | B-SARC | I-SARC | O",
        }])
        with self.assertRaisesRegex(ValueError, "would truncate an annotated sarcasm span"):
            TweetSpanDataset(long_positive, SimpleFastTokenizer(), max_length=5)

    def test_preprocessing_preserves_hashtag_emoji_punctuation_and_removes_mentions_urls(self):
        text = preprocess_tweet("WOW!!! @person https://example.com #blessed 😊")
        self.assertIn("#blessed", text)
        self.assertIn("😊", text)
        self.assertIn("!!!", text)
        self.assertNotIn("@person", text)
        self.assertNotIn("https://", text)
        self.assertEqual(preprocess_tweet("nice https://example.com!!!"), "nice !!!")


if __name__ == "__main__":
    unittest.main()
