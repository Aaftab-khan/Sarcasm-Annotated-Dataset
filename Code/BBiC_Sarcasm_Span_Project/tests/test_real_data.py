import unittest
from collections import Counter
from pathlib import Path
import pandas as pd

from src.encoding_repair import repair_mojibake

ROOT = Path(__file__).resolve().parents[1]


class TestSuppliedDatasets(unittest.TestCase):
    def setUp(self):
        self.train = pd.read_csv(ROOT / "data/processed/TrainAnnotatedSpansBIO.csv", keep_default_na=False, encoding="utf-8-sig")
        self.test = pd.read_csv(ROOT / "data/processed/TestAnnotatedSpansBIO.csv", keep_default_na=False, encoding="utf-8-sig")

    @staticmethod
    def read_supplied_workbook(split):
        original = pd.read_excel(ROOT / f"data/source/{split}AnnotatedSpansBIO.xlsx", keep_default_na=False)
        if "Proprocessed Tweet" in original.columns:
            original = original.rename(columns={"Proprocessed Tweet": "Preprocessed Tweet"})
        return original

    def test_split_counts(self):
        self.assertEqual(len(self.train), 3467)
        self.assertEqual(self.train["sarcastic"].astype(int).value_counts().to_dict(), {0: 2600, 1: 867})
        self.assertEqual(len(self.test), 1400)
        self.assertEqual(self.test["sarcastic"].astype(int).value_counts().to_dict(), {0: 1200, 1: 200})
        self.assertFalse(self.train["tweet"].astype(str).str.strip().eq("").any())
        self.assertFalse(self.test["tweet"].astype(str).str.strip().eq("").any())

    def test_saved_annotation_fields_are_unchanged_for_every_retained_row(self):
        protected = [
            "sarcastic", "Preprocessed Tweet", "Sarcastic Span",
            "Character Span", "BIO Label Sequence",
        ]
        for split, processed in (("Train", self.train), ("Test", self.test)):
            original = self.read_supplied_workbook(split)
            # The training source workbook has one empty-text negative record.
            # It is retained in the workbook but excluded from model-ready CSV.
            original = original.loc[original["tweet"].astype(str).str.strip().ne("")].reset_index(drop=True)
            self.assertEqual(len(original), len(processed))
            for column in protected:
                self.assertEqual(
                    processed[column].astype(str).tolist(),
                    original[column].astype(str).tolist(),
                    msg=f"{split}: protected field changed: {column}",
                )

    def test_positive_raw_tweets_match_original_source_text_in_order(self):
        source_train = pd.read_csv(ROOT / "data/source/train_original.csv", keep_default_na=False, encoding="utf-8-sig")
        source_test = pd.read_csv(ROOT / "data/source/test_original.csv", keep_default_na=False, encoding="utf-8-sig")
        train_positive = source_train.loc[source_train["sarcastic"].astype(int).eq(1), "tweet"].astype(str).tolist()
        test_positive = source_test.loc[source_test["sarcastic"].astype(int).eq(1), "text"].astype(str).tolist()
        self.assertEqual(self.train.loc[self.train["sarcastic"].astype(int).eq(1), "tweet"].tolist(), train_positive)
        self.assertEqual(self.test.loc[self.test["sarcastic"].astype(int).eq(1), "tweet"].tolist(), test_positive)

    def test_negative_text_order_follows_the_uploaded_workbooks(self):
        for split, processed in (("Train", self.train), ("Test", self.test)):
            original = self.read_supplied_workbook(split)
            negatives = original.loc[
                original["sarcastic"].astype(int).eq(0) & original["tweet"].astype(str).str.strip().ne(""),
                "tweet",
            ].astype(str).tolist()
            expected = [repair_mojibake(value) for value in negatives]
            actual = processed.loc[processed["sarcastic"].astype(int).eq(0), "tweet"].astype(str).tolist()
            self.assertEqual(actual, expected, msg=f"{split}: negative order or raw text changed")

    def test_negative_text_collection_matches_original_raw_source(self):
        source_train = pd.read_csv(ROOT / "data/source/train_original.csv", keep_default_na=False, encoding="utf-8-sig")
        source_test = pd.read_csv(ROOT / "data/source/test_original.csv", keep_default_na=False, encoding="utf-8-sig")
        for processed, source, source_text_column in (
            (self.train, source_train, "tweet"), (self.test, source_test, "text"),
        ):
            expected = source.loc[
                source["sarcastic"].astype(int).eq(0) & source[source_text_column].astype(str).str.strip().ne(""),
                source_text_column,
            ].astype(str).tolist()
            actual = processed.loc[processed["sarcastic"].astype(int).eq(0), "tweet"].astype(str).tolist()
            self.assertEqual(Counter(actual), Counter(expected))

    def test_non_sarcastic_annotation_fields_are_blank(self):
        fields = ["Preprocessed Tweet", "Sarcastic Span", "Character Span", "BIO Label Sequence"]
        for df in (self.train, self.test):
            negatives = df.loc[df["sarcastic"].astype(int).eq(0)]
            for field in fields:
                self.assertTrue(negatives[field].astype(str).str.strip().eq("").all(), msg=field)


if __name__ == "__main__":
    unittest.main()
