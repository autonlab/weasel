import logging
from typing import Union, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from weasel.models.downstream_models.MLP import MLPNet
from weasel.models.encoder_models.base_encoder import BaseEncoder
from weasel.utils.utils import get_logger

log = logging.getLogger(__name__)


class MLPEncoder(BaseEncoder):
    """
    This is an encoder that supports multi-class classification between C classes {1, 2, ..., C}.
    """

    def __init__(
            self,
            num_LFs: int,
            example_extra_input: Optional[torch.Tensor],
            output_dim: int,
            hidden_dims: Optional[List[int]] = None,
            dropout: float = 0.0,
            activation_func: str = 'Gelu',
            net_norm: str = 'batch_norm',
            *args, **kwargs
    ):
        super().__init__(num_LFs, example_extra_input, output_dim)
        mlp_in_dim = num_LFs

        if example_extra_input is not None:
            if len(example_extra_input.shape) != 2:
                raise ValueError("When using auxiliary input, MLP Encoder expects it to have shape (batch-size, d),"
                                 " where d is an arbitrary dimension.")
            mlp_in_dim += example_extra_input.shape[1]

        if hidden_dims is None:
            hdim2 = int(self.output_dim * 2 / 3)
            hdim1 = max(int(mlp_in_dim / 5), hdim2)
            hidden_dims = [hdim1, hdim2]
            log.warning(
                f"Hidden dimensions of the MLP encoder have been automatically set to {hidden_dims}."
                f" A manual definition is recommended!"
            )

        self.network = MLPNet(
            input_dim=mlp_in_dim,
            output_dim=self.output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation_func=activation_func,
            net_norm=net_norm,
            *args, **kwargs
        )

    def forward(
            self,
            label_matrix: torch.Tensor,
            aux_input: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """

        Args:
            label_matrix (Tensor): a (n, m) tensor of the m LF votes
            aux_input (Tensor): A (n, d) tensor, where d is an arbitrary integer, and given by
                your_downstream_model.get_encoder_features().
                Note: Will be None if Weasel.use_aux_input_for_encoder flag is False
        Returns:
            A Tensor of shape (n, self.output_dim) that defines the raw accuracy scores.
        """
        input_tensor = label_matrix if aux_input is None else torch.cat((label_matrix, aux_input), dim=1)
        raw_accuracies = self.network(input_tensor)
        return raw_accuracies
