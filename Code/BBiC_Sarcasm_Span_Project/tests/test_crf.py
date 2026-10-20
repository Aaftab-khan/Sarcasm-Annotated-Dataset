import unittest
import torch

from src.crf import LinearChainCRF
from src.spans import LABELS


class TestLinearChainCRF(unittest.TestCase):
    def test_loss_is_finite_and_backpropagates(self):
        torch.manual_seed(7)
        crf = LinearChainCRF(3, list(LABELS))
        emissions = torch.randn(2, 5, 3, requires_grad=True)
        tags = torch.tensor([[0, 1, 2, 0, 0], [0, 0, 1, 2, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.bool)
        loss = crf.neg_log_likelihood(emissions, tags, mask)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(emissions.grad)
        self.assertTrue(torch.isfinite(emissions.grad).all().item())

    def test_viterbi_respects_bio_start_and_transitions(self):
        crf = LinearChainCRF(3, list(LABELS))
        emissions = torch.tensor([[[0.0, 0.0, 50.0], [0.0, 0.0, 50.0], [0.0, 50.0, 0.0]]])
        mask = torch.ones((1, 3), dtype=torch.bool)
        path = crf.decode(emissions, mask)[0]
        self.assertNotEqual(path[0], 2)  # I-SARC cannot begin a sequence.
        for prev, current in zip(path, path[1:]):
            if current == 2:
                self.assertIn(prev, (1, 2))  # I must follow B or I.

    def test_padding_is_ignored_by_decode(self):
        crf = LinearChainCRF(3, list(LABELS))
        emissions = torch.randn(2, 4, 3)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        paths = crf.decode(emissions, mask)
        self.assertEqual([len(x) for x in paths], [4, 2])

    def test_log_likelihood_matches_brute_force_enumeration(self):
        import itertools
        torch.manual_seed(13)
        crf = LinearChainCRF(3, list(LABELS))
        emissions = torch.randn(1, 2, 3)
        tags = torch.tensor([[0, 1]], dtype=torch.long)
        mask = torch.ones((1, 2), dtype=torch.bool)
        loss = crf.neg_log_likelihood(emissions, tags, mask)
        start, transitions, end = crf._constrained_parameters()
        all_scores = []
        for path in itertools.product(range(3), repeat=2):
            # Skip paths with a disallowed start/transition (their score has a -10000 penalty).
            score = start[path[0]] + emissions[0, 0, path[0]]
            score = score + transitions[path[0], path[1]] + emissions[0, 1, path[1]] + end[path[1]]
            all_scores.append(score)
        partition = torch.logsumexp(torch.stack(all_scores), dim=0)
        gold = start[0] + emissions[0, 0, 0] + transitions[0, 1] + emissions[0, 1, 1] + end[1]
        expected = partition - gold
        self.assertTrue(torch.allclose(loss, expected, atol=1e-5))

    def test_forward_backward_marginals_normalize_and_mask_padding(self):
        torch.manual_seed(23)
        crf = LinearChainCRF(3, list(LABELS))
        emissions = torch.randn(2, 4, 3)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        marginals = crf.marginals(emissions, mask)
        self.assertEqual(tuple(marginals.shape), (2, 4, 3))
        sums = marginals.sum(dim=-1)
        self.assertTrue(torch.allclose(sums[mask], torch.ones_like(sums[mask]), atol=1e-5))
        self.assertTrue(torch.equal(marginals[~mask], torch.zeros_like(marginals[~mask])))
        self.assertTrue(torch.isfinite(marginals).all().item())
        self.assertTrue(torch.all(marginals >= 0).item())

    def test_forward_backward_marginals_match_brute_force(self):
        import itertools
        torch.manual_seed(29)
        crf = LinearChainCRF(3, list(LABELS))
        emissions = torch.randn(1, 2, 3)
        mask = torch.ones((1, 2), dtype=torch.bool)
        marginals = crf.marginals(emissions, mask)[0]
        start, transitions, end = crf._constrained_parameters()
        paths = list(itertools.product(range(3), repeat=2))
        scores = []
        for path in paths:
            score = start[path[0]] + emissions[0, 0, path[0]]
            score = score + transitions[path[0], path[1]] + emissions[0, 1, path[1]] + end[path[1]]
            scores.append(score)
        log_z = torch.logsumexp(torch.stack(scores), dim=0)
        expected = torch.zeros_like(marginals)
        for path, score in zip(paths, scores):
            probability = torch.exp(score - log_z)
            expected[0, path[0]] += probability
            expected[1, path[1]] += probability
        self.assertTrue(torch.allclose(marginals, expected, atol=1e-5))

    def test_invalid_mask_rejected(self):
        crf = LinearChainCRF(3, list(LABELS))
        with self.assertRaises(ValueError):
            crf.neg_log_likelihood(torch.randn(1, 2, 3), torch.tensor([[0, 0]]),
                                   torch.tensor([[0, 1]], dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
