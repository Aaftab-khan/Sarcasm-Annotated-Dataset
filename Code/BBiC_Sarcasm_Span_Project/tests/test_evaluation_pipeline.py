import re
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn

from src.data import TweetSpanDataset, collate_batch
from src.engine import evaluate_model, make_loader
from src.model import BBiCModel


class TinyTokenizer:
    is_fast = True
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, truncation=False, return_offsets_mapping=False, **kwargs):
        matches = [(m.group(0), (m.start(), m.end())) for m in re.finditer(r"\w+|[^\w\s]", text)]
        ids = ([101] if add_special_tokens else []) + [10 + i for i in range(len(matches))] + ([102] if add_special_tokens else [])
        offsets = ([(0, 0)] if add_special_tokens else []) + [m[1] for m in matches] + ([(0, 0)] if add_special_tokens else [])
        return {"input_ids": ids, "offset_mapping": offsets} if return_offsets_mapping else {"input_ids": ids}


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embeddings = nn.Embedding(200, 8)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))


class TinyAutoModel:
    @staticmethod
    def from_pretrained(name):
        return TinyEncoder()


class TinyAutoConfig:
    @staticmethod
    def from_pretrained(path, local_files_only=True):
        return SimpleNamespace(hidden_size=8)


class TinyTransformers(types.ModuleType):
    def __init__(self):
        super().__init__("transformers")
        self.AutoModel = TinyAutoModel
        self.AutoConfig = TinyAutoConfig


class TestEvaluationPipeline(unittest.TestCase):
    def test_end_to_end_evaluation_outputs_metrics_and_spans(self):
        frame = pd.DataFrame([
            {"tweet": "Great, this works", "sarcastic": 1,
             "Preprocessed Tweet": "Great, this works", "Sarcastic Span": "Great",
             "Character Span": "[0, 5)", "BIO Label Sequence": "O | B-SARC | O | O | O | O"},
            {"tweet": "Plain text 😊", "sarcastic": 0, "Preprocessed Tweet": "",
             "Sarcastic Span": "", "Character Span": "", "BIO Label Sequence": ""},
        ])
        with patch.dict(sys.modules, {"transformers": TinyTransformers()}):
            model = BBiCModel(base_model="tiny", bilstm_hidden_size=4, dropout=0.0, variant="bbic")
        tokenizer = TinyTokenizer()
        dataset = TweetSpanDataset(frame, tokenizer, max_length=32)
        loader = make_loader(dataset, batch_size=2, shuffle=False, pad_token_id=tokenizer.pad_token_id)
        metrics, predictions, curve_scores = evaluate_model(model, loader, torch.device("cpu"))
        self.assertEqual(len(predictions), 2)
        self.assertIn("token_sarcasm_f1", metrics)
        self.assertIn("exact_span_f1", metrics)
        self.assertEqual(len(curve_scores["token_gold_binary"]), len(curve_scores["token_sarcasm_score"]))
        self.assertIn("CRF forward-backward marginal", curve_scores["score_interpretation"])
        self.assertIn("predicted_char_spans", predictions.columns)


if __name__ == "__main__":
    unittest.main()
