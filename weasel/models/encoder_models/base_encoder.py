from typing import Optional, Any, Union, Tuple, List

import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(
            self, num_LFs: int,
            example_extra_input: Optional[torch.Tensor],
            output_dim: int):
        super().__init__()
        self.num_LFs = num_LFs
        self.example_extra_input = example_extra_input
        self.output_dim = output_dim

    def forward(
            self,
            label_matrix: torch.Tensor,
            features: Optional[Any]
    ) -> torch.Tensor:
        """

        Args:
            label_matrix (Tensor): a (n, m) tensor of the m LF votes
            features (Any): Any auxiliary input, e.g. the data features X, used for prediction, and given by
                your_downstream_model.get_encoder_features().
                Note: Will be None if Weasel.use_aux_input_for_encoder flag is False
        Returns:
            A Tensor of shape (n, self.output_dim) that defines the raw accuracy scores.
        """
        pass
