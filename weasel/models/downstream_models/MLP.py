from typing import Any, List

import torch
import torch.nn as nn
from weasel.models.downstream_models.base_model import DownstreamBaseModel
from weasel.utils.utils import get_activation_function, stem_word


class MLPNet(DownstreamBaseModel):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            output_dim: int,
            dropout: float = 0.0,
            activation_func: str = 'Gelu',
            net_norm: str = 'batch_norm',
            adjust_thresh: bool = True,
            save_hparams: bool = False
    ):
        r"""
        Args:
            input_dim (int): Number of features in the input layer
            hidden_dims (List[int]): A list that defines the hidden layer dimension/#neurons
            output_dim (int): Output dimensionality/#neurons, usually equal to the number of classes, C
            dropout (float): Dropout rate. Default = 0
            activation_func (str): Activation function to use for all hidden layers. Default: GeLU
            net_norm (str): One of {'none', 'layer_norm', 'batch_norm'} & defines the hidden layer normalization scheme.
                'none' Means that no normalization is employed.
            adjust_thresh (bool): Whether to adjust the decision threshold for prediction, based on the validation set.
                Note: Only supported and used for binary classification.
        """
        super().__init__()
        if save_hparams:
            self.save_hyperparameters()
        net_norm = stem_word(net_norm)
        self.out_dim = output_dim
        self.adjust_thresh = adjust_thresh
        self.example_input_array = torch.randn((1, input_dim))

        mlp_modules = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(1, len(dims)):
            mlp_modules += [nn.Linear(dims[i - 1], dims[i], bias=True)]

            if net_norm == 'layer_norm':
                mlp_modules += [nn.LayerNorm(dims[i])]
            elif net_norm == 'batch_norm':
                mlp_modules += [nn.BatchNorm1d(dims[i])]

            mlp_modules += [get_activation_function(activation_func, functional=False)]

            if dropout > 0:
                mlp_modules += [nn.Dropout(dropout)]

        out_layer = nn.Linear(dims[-1], self.out_dim, bias=True)
        mlp_modules += [out_layer]

        self.network = nn.Sequential(*mlp_modules)

        # decision threshold tunable for binary classification
        if output_dim <= 2:
            self.register_buffer("_decision_thresh", torch.tensor(0.5))

    def forward(self, X):
        X = self.network(X)
        return X.squeeze(1)

    def get_encoder_features(self, X):
        return X

    def __str__(self):
        return 'MLP'


