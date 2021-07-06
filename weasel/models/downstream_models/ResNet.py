from typing import Union, Optional

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from weasel.models.downstream_models.MLP import MLPNet
from weasel.models.downstream_models.base_model import DownstreamBaseModel


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


class Resnet(DownstreamBaseModel):
    def __init__(
            self,
            output_dim: int,
            resnet: Union[str, int] = '18',
            pretrained: bool = True,
            freeze_resnet: bool = True,
            activation_func: str = 'Gelu',
            readout_hidden_dims: list = None,
            readout_dropout: float = 0.0,
            readout_net_norm: str = 'none',
            adjust_thresh: bool = True

    ):
        r"""
        Args:
            output_dim (int): Output dimensionality, e.g. the number of classes
            resnet (str, int): One of {18, 34, 50}
            pretrained (bool): Whether to use a pre-trained ResNet
            freeze_resnet (bool): Whether to freeze the parameters of the backbone ResNet if using a pre-trained one.
                That is, when freeze_resnet AND pretrained are True, only the readout MLP parameters will be updated.
            activation_func (str): String name of an activation function for the readout MLP
            readout_hidden_dims (List[int]): Hidden dimensions of the readout MLP
            readout_dropout (float): Dropout rate of the readout MLP
            readout_net_norm (str): Net Normalization of the readout MLP
            adjust_thresh (bool): Whether to adjust the decision threshold for prediction, based on the validation set.
                    Note: Only supported and used for binary classification.
        """
        super().__init__()
        resnet = str(resnet)
        if resnet in '18':
            self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        elif resnet == '34':
            self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        elif resnet == '50':
            self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        else:
            raise ValueError('Unknown Resnet, please provide 18, 34, or 50.')

        if not pretrained:
            freeze_resnet = False

        if freeze_resnet:
            freeze_params(self.resnet)

        if readout_hidden_dims is None:
            readout_hidden_dims = [64, 64]

        n_features = self.resnet.fc.in_features
        self.out_dim = output_dim

        self.resnet.fc = MLPNet(**{
            'input_dim': n_features,
            'hidden_dims': readout_hidden_dims,
            'output_dim': output_dim,
            'dropout': readout_dropout,
            'activation_func': activation_func,
            'net_norm': readout_net_norm,
            'adjust_thresh': adjust_thresh
        })

    def forward(self, x, get_features=False):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        if get_features:
            return x
        x = self.resnet.fc(x)
        return x.squeeze(1)

    def __str__(self):
        return 'ResNet'

    def get_encoder_features(self, X) -> torch.Tensor:
        return self.forward(X, get_features=True).detach()

    def get_decision_thresh(self) -> Optional[float]:
        """
        Get the model's decision threshold for hard predictions in binary classification.
        """
        return self.resnet.fc.get_decision_thresh()

    def set_decision_thresh(self, decision_thresh) -> None:
        """
        Set the model's decision threshold for hard predictions in binary classification.
        """
        self.resnet.fc.set_decision_thresh(decision_thresh)
