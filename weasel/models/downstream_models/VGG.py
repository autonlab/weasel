from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from weasel.utils.utils import get_activation_function
from weasel.models.downstream_models.MLP import MLPNet
from weasel.models.downstream_models.base_model import DownstreamBaseModel


class VGG(DownstreamBaseModel):
    def __init__(
            self,
            in_channels,
            output_dim,
            vgg: Union[str, int] = "11",
            activation_func: str = 'Gelu',
            readout_hidden_dims: list = None,
            readout_dropout: float = 0.0,
            readout_net_norm: str = 'none',
            adjust_thresh: bool = True
    ):
        super().__init__()

        self.vgg_config = {
            '11':
                [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            '13':
                [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            '16':
                [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            '19':
                [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        self.vgg = str(vgg)
        self.in_channels = in_channels
        self.example_input_array = torch.randn(7, self.in_channels, 32, 32)
        self.activation = activation_func
        self.out_dim = output_dim

        self.features = self._make_layers(self.vgg_config[self.vgg])

        if readout_hidden_dims is None:
            readout_hidden_dims = [64, 64]

        self.classifier = MLPNet(**{
            'input_dim': 512,
            'hidden_dims': readout_hidden_dims,
            'output_dim': output_dim,
            'dropout': readout_dropout,
            'activation_func': self.activation,
            'net_norm': readout_net_norm,
            'adjust_thresh': adjust_thresh
        })

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def get_decision_thresh(self) -> Optional[float]:
        """
        Get the model's decision threshold for hard predictions in binary classification.
        """
        return self.classifier.get_decision_thresh()

    def set_decision_thresh(self, decision_thresh) -> None:
        """
        Set the model's decision threshold for hard predictions in binary classification.
        """
        self.classifier.set_decision_thresh(decision_thresh)

    def __str__(self):
        return 'VGG'

    def _make_layers(self, cfg):
        layers = []
        channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           get_activation_function(self.activation, functional=False)
                           ]
                channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

