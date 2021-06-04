import numpy as np
import torch
import torch.nn as nn
from downstream_models.base_model import DownstreamBaseModel, DownstreamBaseTrainer
from utils.utils import get_activation_function


class MLPNet(DownstreamBaseModel):
    def __init__(self, mlp_params):
        super().__init__(mlp_params)
        hidden_dim = mlp_params['hidden_dim']
        dropout = mlp_params['dropout']
        n_layers = mlp_params['L']
        act_func = 'relu' if 'activation' not in mlp_params else mlp_params['activation']
        self.out_size = mlp_params["out_dim"]

        feat_mlp_modules = [
            nn.Linear(mlp_params["input_dim"], hidden_dim, bias=True),
            get_activation_function(act_func, functional=False),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layers - 1):
            feat_mlp_modules.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            feat_mlp_modules.append(get_activation_function(act_func, functional=False))
            feat_mlp_modules.append(nn.Dropout(dropout))
        self.feat_mlp = nn.Sequential(*feat_mlp_modules)

        readout_layers = 1
        readout_mlp_modules = []
        for l in range(readout_layers):
            readout_mlp_modules.append(nn.Linear(hidden_dim // 2 ** l, hidden_dim // 2 ** (l + 1), bias=True))
            readout_mlp_modules.append(get_activation_function(act_func, functional=False))
        readout_mlp_modules.append(nn.Linear(hidden_dim // 2 ** readout_layers, self.out_size, bias=True))
        self.readout_mlp = nn.Sequential(*readout_mlp_modules)

    def forward(self, X, device='cuda'):
        X = X.to(device)
        X = self.feat_mlp(X)
        X = self.readout_mlp(X)
        return X.squeeze(1)

    def get_encoder_features(self, X, device='cuda'):
        X = X.to(device)  # + torch.normal(mean=0, std=0.01, size=X.size()).to(device)
        return X

    def __str__(self):
        return 'MLP'


class MLP_Trainer(DownstreamBaseTrainer):
    def __init__(
            self, downstream_params, name='MLP', seed=None, verbose=False, model_dir="out/MLP",
            notebook_mode=False, model=None
    ):
        super().__init__(downstream_params, name=name, seed=seed, verbose=verbose,
                         model_dir=model_dir, notebook_mode=notebook_mode, model=model)
        self.model_class = MLPNet
        self.name = name


class LabelMeMLP(DownstreamBaseModel):
    def __init__(self, params=None, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.out_size = 8
        self.linear1 = nn.Linear(8192, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(128, self.out_size)
        self.relu = nn.ReLU()

    def forward(self, X, device='cuda'):
        x = X.to(device)
        x = x.reshape(x.shape[0], 8192)
        x = self.dropout1(self.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

    def get_encoder_features(self, X, device='cuda'):
        return X.to(device).reshape(X.shape[0], 8192)

    def predict_proba(self, X, device='cuda'):
        X = self.forward(X, device=device)
        return nn.Softmax(dim=1)(X)

    def predict(self, X=None, probs=None, cutoff=0.5, device='cuda'):
        """
        If no soft labels are given, please provide features to get the corresponding predictions from the end moodel.
        :param X: Either None, or a (n, d) feature tensor that will be used for prediction.
        :param probs: Probabilistic labels (n, 1), or None
        :param cutoff: The model's decision threshold for hard predictions
        :param device: E.g. 'cuda' or 'cpu'
        :return: A tuple (Y_soft, Y_hard) of probabilistic labels with corresponding hard predictions
        """
        if probs is None:
            assert X is not None, 'Please provide soft labels, or features to generate them'
            probs = self.predict_proba(X, device=device)
        preds = torch.argmax(probs, dim=1) if isinstance(probs, torch.Tensor) else np.argmax(probs, axis=1)
        return probs, preds

    def __str__(self):
        return 'MLP'


class LabelMeMLP_Trainer(DownstreamBaseTrainer):
    def __init__(
            self, downstream_params, name='MLP', seed=None, verbose=False, model_dir="out/MLP",
            notebook_mode=False, model=None
    ):
        super().__init__(downstream_params, name=name, seed=seed, verbose=verbose,
                         model_dir=model_dir, notebook_mode=notebook_mode, model=model)
        self.model_class = LabelMeMLP
        self.name = name
