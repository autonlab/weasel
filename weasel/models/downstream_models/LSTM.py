from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
from weasel.models.downstream_models.MLP import MLPNet
from weasel.models.downstream_models.base_model import DownstreamBaseModel


class LSTM(DownstreamBaseModel):
    def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int,
            output_dim: int,
            vocab_size: int = 40_000,  # num_embeddings
            n_layers: int = 1,
            bidirectional: bool = False,
            dropout: float = 0.0,
            activation_func: str = 'Gelu',
            readout_hidden_dims: list = None,
            readout_dropout: Optional[float] = None,
            readout_net_norm: str = 'none',
            adjust_thresh: bool = True
    ):
        super().__init__()
        if readout_hidden_dims is None:
            readout_hidden_dims = [64, 64]

        self.example_input_array = torch.ones((7, 77)).long()

        self.embedder = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = output_dim

        self.LSTM_net = nn.LSTM(
            input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=self.n_layers,
            dropout=dropout, bidirectional=bidirectional, batch_first=True
        )
        if bidirectional:
            self.n_layers *= 2
        self.weight_size = (self.n_layers, embedding_dim, self.hidden_dim)

        self.readout = MLPNet(**{
            'input_dim': self.hidden_dim,
            'hidden_dims': readout_hidden_dims,
            'output_dim': self.out_dim,
            'dropout': readout_dropout or dropout,
            'activation_func': activation_func,
            'net_norm': readout_net_norm,
            'adjust_thresh': adjust_thresh
        })

    def init_hidden(self, batch_size):
        return (torch.autograd.Variable(torch.randn(self.n_layers, batch_size, self.hidden_dim)),
                torch.autograd.Variable(torch.randn(self.n_layers, batch_size, self.hidden_dim)))

    def embed_features(self, X: Union[torch.LongTensor, torch.IntTensor]):
        return self.embedder(X)

    def get_encoder_features(self, X, *args, **kwargs):
        feats_for_enc = self.embed_features(X)
        return feats_for_enc

    def forward(self, X, readout=True):
        X = self.embed_features(X)
        print(X.shape)
        hidden = self.init_hidden(X.shape[0])
        b_output, (hidden, bcell) = self.LSTM_net(X, hidden)
        print(b_output.shape, hidden.shape, bcell.shape)
        if readout:
            logits = self.readout(hidden.reshape(-1, self.hidden_dim))
            return logits
        else:
            return hidden

    def get_decision_thresh(self) -> Optional[float]:
        """
        Get the model's decision threshold for hard predictions in binary classification.
        """
        return self.readout.get_decision_thresh()

    def set_decision_thresh(self, decision_thresh) -> None:
        """
        Set the model's decision threshold for hard predictions in binary classification.
        """
        self.readout.set_decision_thresh(decision_thresh)

    def __str__(self):
        return 'LSTM'
