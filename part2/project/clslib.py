"""
Sentence encoders and pair classifier for PAWS experiments.

Reuses part2.clslib EmbedAndPoolEncoder and LSTMEncoder.
Adds BiLSTMEncoder, GRUEncoder, and PairStackDiffProdClassifier
(stack_diff_prod pair combine, same idea as part1 DualLSTM).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from part2 import clslib as part2_clslib


def _apply_word_dropout(embedded, valid, word_dropout, training):
    """Zero non-padding token embeddings with probability word_dropout (train only)."""
    if training and word_dropout > 0:
        drop_mask = (torch.rand(valid.shape, device=embedded.device) < word_dropout) & valid
        embedded = embedded.masked_fill(drop_mask.unsqueeze(-1), 0.0)
    return embedded


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM sentence encoder.
    Final representation is concat of last forward and last backward states (size 2H).
    """

    def __init__(
        self,
        hidden_size,
        embedding_matrix,
        padding_idx=0,
        freeze_embeddings=True,
        word_dropout=0.0,
    ):
        super().__init__()
        _, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=padding_idx,
        )
        self.hidden_size = hidden_size
        self.word_dropout = word_dropout
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    @property
    def enc_dim(self):
        return 2 * self.hidden_size

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):
        embedded = self.embedding(x)
        valid = x.ne(self.embedding.padding_idx)
        embedded = _apply_word_dropout(embedded, valid, self.word_dropout, self.training)
        lengths = torch.sum(valid, dim=-1)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _cell) = self.lstm(packed)
        # hidden: (2, batch, H) with 1 layer — [forward; backward]
        return torch.cat([hidden[0], hidden[1]], dim=-1)


class GRUEncoder(nn.Module):
    """Unidirectional GRU sentence encoder (same interface as part2 LSTMEncoder)."""

    def __init__(
        self,
        hidden_size,
        embedding_matrix,
        padding_idx=0,
        freeze_embeddings=True,
        word_dropout=0.0,
    ):
        super().__init__()
        _, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=padding_idx,
        )
        self.hidden_size = hidden_size
        self.word_dropout = word_dropout
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    @property
    def enc_dim(self):
        return self.hidden_size

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):
        embedded = self.embedding(x)
        valid = x.ne(self.embedding.padding_idx)
        embedded = _apply_word_dropout(embedded, valid, self.word_dropout, self.training)
        lengths = torch.sum(valid, dim=-1)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden[-1]


class PairStackDiffProdClassifier(nn.Module):
    """
    Encode each sentence with a shared encoder, combine with stack_diff_prod:
        [h1; h2; |h1-h2|; h1*h2]
    then MLP + log-softmax (same style as part2 SoftmaxRegressor).
    """

    def __init__(self, encoder: nn.Module, hidden_size1, hidden_size2, num_classes, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        pair_dim = 4 * encoder.enc_dim
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(pair_dim, hidden_size1),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size2, num_classes),
            nn.LogSoftmax(dim=-1),
        )

    @property
    def pad_idx(self):
        return self.encoder.pad_idx

    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        h = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2], dim=-1)
        return self.output_layer(h)


def make_sentence_encoder(enc_type: str, embedding_matrix: torch.Tensor, enc_kwargs: dict):
    """
    Build a sentence encoder by config name.
    Reuses part2 EmbedAndPoolEncoder / LSTMEncoder; local BiLSTM / GRU.
    """
    kwargs = dict(enc_kwargs)
    if enc_type == "embed-and-pool":
        return part2_clslib.EmbedAndPoolEncoder(
            embedding_matrix=embedding_matrix,
            **kwargs,
        )
    if enc_type == "lstm":
        return part2_clslib.LSTMEncoder(
            embedding_matrix=embedding_matrix,
            **kwargs,
        )
    if enc_type == "bilstm":
        return BiLSTMEncoder(
            embedding_matrix=embedding_matrix,
            **kwargs,
        )
    if enc_type == "gru":
        return GRUEncoder(
            embedding_matrix=embedding_matrix,
            **kwargs,
        )
    raise ValueError(f"Unknown enc_type {enc_type!r}")
