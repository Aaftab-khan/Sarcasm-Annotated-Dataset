import unittest
from src.metrics import token_metrics, span_metrics, tweet_presence_metrics


class TestMetrics(unittest.TestCase):
    def test_perfect_token_and_exact_span_scores(self):
        gold = [["O", "B-SARC", "I-SARC", "O"]]
        pred = [["O", "B-SARC", "I-SARC", "O"]]
        token = token_metrics(gold, pred)
        span = span_metrics(gold, pred)
        self.assertAlmostEqual(token["token_sarcasm_f1"], 1.0)
        self.assertAlmostEqual(span["exact_span_f1"], 1.0)
        self.assertAlmostEqual(span["partial_span_f1"], 1.0)

    def test_partial_match_is_overlap_based(self):
        gold = [["O", "B-SARC", "I-SARC", "O"]]
        pred = [["O", "O", "B-SARC", "O"]]
        span = span_metrics(gold, pred)
        self.assertAlmostEqual(span["exact_span_f1"], 0.0)
        self.assertAlmostEqual(span["partial_span_f1"], 1.0)

    def test_tweet_presence_confusion_matrix(self):
        result = tweet_presence_metrics([1, 0, 1, 0], [
            ["B-SARC"], ["O"], ["O"], ["I-SARC"]
        ])
        self.assertEqual(result["tweet_presence_confusion_matrix"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(result["tweet_presence_accuracy"], 0.5)

    def test_mismatched_lengths_fail(self):
        with self.assertRaises(ValueError):
            token_metrics([["O"]], [["O", "O"]])


if __name__ == "__main__":
    unittest.main()
