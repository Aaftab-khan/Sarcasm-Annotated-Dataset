"""Linear-chain CRF with BIO transition constraints, implemented in PyTorch.

Transition convention: transitions[previous_tag, next_tag]. The CRF scores a
whole sequence and uses Viterbi decoding, matching the BBiC methodology.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int, labels: list[str] | None = None) -> None:
        super().__init__()
        if num_tags < 2:
            raise ValueError("A CRF requires at least two tags.")
        self.num_tags = int(num_tags)
        self.labels = labels or [str(i) for i in range(num_tags)]
        if len(self.labels) != self.num_tags:
            raise ValueError("labels length must equal num_tags")
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.register_buffer("allowed_start", torch.ones(num_tags, dtype=torch.bool))
        self.register_buffer("allowed_transitions", torch.ones(num_tags, num_tags, dtype=torch.bool))
        self._apply_bio_constraints()
        self.reset_parameters()

    def _apply_bio_constraints(self) -> None:
        """Constrain I-SARC from starting a sequence or following O."""
        if not all(x in self.labels for x in ("O", "B-SARC", "I-SARC")):
            return
        o, b, i = (self.labels.index(x) for x in ("O", "B-SARC", "I-SARC"))
        self.allowed_start[i] = False
        # An I-SARC can follow only B-SARC or I-SARC.
        self.allowed_transitions[:, i] = False
        self.allowed_transitions[b, i] = True
        self.allowed_transitions[i, i] = True
        # All transitions to O or B-SARC are valid.
        self.allowed_transitions[:, o] = True
        self.allowed_transitions[:, b] = True

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _constrained_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # A large finite penalty avoids NaNs in mixed precision and is effectively -inf.
        penalty = torch.tensor(-10000.0, device=self.transitions.device, dtype=self.transitions.dtype)
        start = self.start_transitions.masked_fill(~self.allowed_start, penalty)
        transitions = self.transitions.masked_fill(~self.allowed_transitions, penalty)
        return start, transitions, self.end_transitions

    @staticmethod
    def _validate_inputs(emissions: torch.Tensor, tags: torch.Tensor | None, mask: torch.Tensor) -> None:
        if emissions.ndim != 3:
            raise ValueError("emissions must have shape [batch, sequence, num_tags]")
        if mask.shape != emissions.shape[:2]:
            raise ValueError("mask shape must match [batch, sequence]")
        if mask.dtype != torch.bool:
            raise ValueError("mask must be bool")
        if emissions.shape[1] == 0 or not torch.all(mask[:, 0]):
            raise ValueError("Each sequence must have a valid first token")
        if mask.shape[1] > 1 and torch.any(mask[:, 1:] & ~mask[:, :-1]):
            raise ValueError("mask must be contiguous: valid tokens first, followed only by padding")
        if tags is not None and tags.shape != emissions.shape[:2]:
            raise ValueError("tags shape must match [batch, sequence]")

    def _gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        start_trans, transitions, end_trans = self._constrained_parameters()
        batch_size, seq_len, _ = emissions.shape
        batch_idx = torch.arange(batch_size, device=emissions.device)
        score = start_trans[tags[:, 0]] + emissions[batch_idx, 0, tags[:, 0]]
        lengths = mask.long().sum(dim=1)
        for t in range(1, seq_len):
            active = mask[:, t]
            step = transitions[tags[:, t - 1], tags[:, t]] + emissions[batch_idx, t, tags[:, t]]
            score = score + step * active.to(step.dtype)
        last_tags = tags[batch_idx, lengths - 1]
        score = score + end_trans[last_tags]
        return score

    def _log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        start_trans, transitions, end_trans = self._constrained_parameters()
        alpha = start_trans.unsqueeze(0) + emissions[:, 0]
        for t in range(1, emissions.shape[1]):
            # [batch, previous_tag, next_tag]
            candidate = alpha.unsqueeze(2) + transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            next_alpha = torch.logsumexp(candidate, dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), next_alpha, alpha)
        return torch.logsumexp(alpha + end_trans.unsqueeze(0), dim=1)

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(emissions, tags, mask)
        if tags.min().item() < 0 or tags.max().item() >= self.num_tags:
            raise ValueError("tags contain IDs outside the CRF label space")
        log_z = self._log_partition(emissions, mask)
        gold = self._gold_score(emissions, tags, mask)
        return (log_z - gold).mean()

    def marginals(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return per-token CRF marginal probabilities using forward-backward.

        Args:
            emissions: Tensor shaped [batch, sequence, num_tags].
            mask: Boolean tensor shaped [batch, sequence], with contiguous valid
                tokens before any padding.

        Returns:
            Tensor shaped [batch, sequence, num_tags]. Marginals for valid tokens
            sum to one over tags. Padded positions are all zeros.
        """
        self._validate_inputs(emissions, None, mask)
        start_trans, transitions, end_trans = self._constrained_parameters()
        batch_size, max_len, _ = emissions.shape
        batch_marginals: list[torch.Tensor] = []

        for batch_index in range(batch_size):
            length = int(mask[batch_index].long().sum().item())
            sequence_emissions = emissions[batch_index, :length]

            # Forward log-scores: alpha[t, j] scores all prefixes ending in tag j.
            alpha: list[torch.Tensor] = [start_trans + sequence_emissions[0]]
            for position in range(1, length):
                candidates = alpha[-1].unsqueeze(1) + transitions
                alpha.append(torch.logsumexp(candidates, dim=0) + sequence_emissions[position])

            log_partition = torch.logsumexp(alpha[-1] + end_trans, dim=0)

            # Backward log-scores: beta[t, i] scores every suffix after tag i at t.
            beta: list[torch.Tensor] = [torch.empty_like(start_trans) for _ in range(length)]
            beta[-1] = end_trans
            for position in range(length - 2, -1, -1):
                candidates = (
                    transitions
                    + sequence_emissions[position + 1].unsqueeze(0)
                    + beta[position + 1].unsqueeze(0)
                )
                beta[position] = torch.logsumexp(candidates, dim=1)

            sequence_marginals = torch.stack([
                torch.exp(alpha[position] + beta[position] - log_partition)
                for position in range(length)
            ], dim=0)
            if length < max_len:
                padding = emissions.new_zeros((max_len - length, self.num_tags))
                sequence_marginals = torch.cat((sequence_marginals, padding), dim=0)
            batch_marginals.append(sequence_marginals)

        return torch.stack(batch_marginals, dim=0)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        self._validate_inputs(emissions, None, mask)
        start_trans, transitions, end_trans = self._constrained_parameters()
        decoded: List[List[int]] = []
        for b in range(emissions.shape[0]):
            length = int(mask[b].sum().item())
            score = start_trans + emissions[b, 0]
            backpointers: list[torch.Tensor] = []
            for t in range(1, length):
                candidate = score.unsqueeze(1) + transitions
                best_score, best_prev = candidate.max(dim=0)
                score = best_score + emissions[b, t]
                backpointers.append(best_prev)
            last = int((score + end_trans).argmax().item())
            path = [last]
            for pointer in reversed(backpointers):
                last = int(pointer[last].item())
                path.append(last)
            path.reverse()
            decoded.append(path)
        return decoded
