from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


RNNType = Literal['lstm', 'gru']


@dataclass(frozen=True)
class RNNConfig:
    input_size: int
    output_size: int
    rnn_type: RNNType = 'lstm'
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0


class RecurrentForecaster(nn.Module):
    """
    Sequence-to-one recurrent forecaster.

    Input:
        x: (batch, seq_len, input_size)

    Output:
        y_hat: (batch, output_size)
    """

    def __init__(self, cfg: RNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        rnn_dropout = cfg.dropout if cfg.num_layers > 1 else 0.0

        if cfg.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=cfg.input_size,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif cfg.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=cfg.input_size,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError(f'Unsupported rnn_type={cfg.rnn_type!r}. Use "lstm" or "gru".')

        self.head = nn.Linear(cfg.hidden_size, cfg.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, input_size)

        Returns
        -------
        torch.Tensor
            Shape (batch, output_size)
        """
        if self.cfg.rnn_type == 'lstm':
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)

        last_hidden = h_n[-1]
        y_hat = self.head(last_hidden)
        return y_hat
