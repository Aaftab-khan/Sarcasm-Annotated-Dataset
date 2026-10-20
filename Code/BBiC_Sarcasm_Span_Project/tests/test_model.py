import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.model import BBiCModel


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(64, 8)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class DummyAutoModel:
    @staticmethod
    def from_pretrained(name):
        return DummyEncoder()


class DummyAutoConfig:
    @staticmethod
    def from_pretrained(path, local_files_only=True):
        return SimpleNamespace(hidden_size=8)


class DummyTransformers(types.ModuleType):
    def __init__(self):
        super().__init__("transformers")
        self.AutoModel = DummyAutoModel
        self.AutoConfig = DummyAutoConfig


class TestBBiCModel(unittest.TestCase):
    def setUp(self):
        self.patch = patch.dict(sys.modules, {"transformers": DummyTransformers()})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_bbic_forward_loss_backward_and_decode(self):
        model = BBiCModel(base_model="dummy", bilstm_hidden_size=4, dropout=0.1, variant="bbic")
        ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.bool)
        labels = torch.tensor([[0, 1, 2, 0, 0], [0, 0, 1, 0, 0]])
        output = model(input_ids=ids, attention_mask=mask, labels=labels)
        self.assertEqual(tuple(output["emissions"].shape), (2, 5, 3))
        self.assertTrue(torch.isfinite(output["loss"]).item())
        output["loss"].backward()
        paths = model.decode(output["emissions"].detach(), mask)
        self.assertEqual([len(path) for path in paths], [5, 4])
        self.assertNotEqual(paths[0][0], 2)  # No I-SARC start.

    def test_all_ablation_variants_have_finite_loss(self):
        for variant in ("bert_only", "bert_bilstm", "bert_crf"):
            with self.subTest(variant=variant):
                model = BBiCModel(base_model="dummy", bilstm_hidden_size=4, dropout=0.1, variant=variant)
                ids = torch.tensor([[1, 2, 3, 4]])
                mask = torch.ones_like(ids, dtype=torch.bool)
                labels = torch.tensor([[0, 1, 2, 0]])
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                self.assertTrue(torch.isfinite(out["loss"]).item())
                out["loss"].backward()
                self.assertEqual(len(model.decode(out["emissions"].detach(), mask)[0]), 4)


if __name__ == "__main__":
    unittest.main()
