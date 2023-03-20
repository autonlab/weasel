import warnings

warnings.filterwarnings("ignore")

# %%

import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

seed_everything(seed=7)

# Full pipeline for a new WeaSEL problem
# %% md

# %%

n, n_evaluation = 10_000, 1_000  # number of training and test samples
n_channels = 3  # e.g. could be RGB
height = width = 28  # grid resolution

X_train = np.random.randn(n, n_channels, height, width)
X_test = np.random.randn(n_evaluation, n_channels, height, width)


# %%

C = 3
possible_labels = list(range(C))
Y_test = np.random.choice(possible_labels, size=n_evaluation)

# %% md


# %%

m = 10
ABSTAIN = -1

possible_LF_outputs = [ABSTAIN] + list(range(C))
label_matrix = np.empty((n, m))
for LF in range(m):
    label_matrix[:, LF] = np.random.choice(
        possible_LF_outputs, size=n, p=[0.85] + [(1 - 0.85) * 1 / C for _ in range(C)]
    )


# %% md

# From data to DataModule
# %%

from weasel.datamodules.base_datamodule import BasicWeaselDataModule

weasel_datamodule = BasicWeaselDataModule(
    label_matrix=label_matrix,
    X_train=X_train,
    X_test=X_test,
    Y_test=Y_test,
    batch_size=256,
    val_test_split=(200, 800)  # 200 validation, 800 test points will be split from (X_test, Y_test)
)

# %% md

## Defining an End-model

# %%

from weasel.models.downstream_models.base_model import DownstreamBaseModel


class MyCNN(DownstreamBaseModel):
    def __init__(self, in_channels,
                 hidden_dim,
                 conv_layers: int,
                 n_classes: int,
                 kernel_size=(3, 3),
                 *args, **kwargs):
        super().__init__()
        # Good practice:
        self.out_dim = n_classes
        self.example_input_array = torch.randn((1, in_channels, height, width))

        cnn_modules = []

        in_dim = in_channels
        for layer in range(conv_layers):
            cnn_modules += [
                nn.Conv2d(in_dim, hidden_dim, kernel_size),
                nn.GELU(),
                nn.MaxPool2d(2, 2)
            ]
            in_dim = hidden_dim

        self.convs = nn.Sequential(*cnn_modules)

        self.flattened_dim = torch.flatten(
            self.convs(self.example_input_array), start_dim=1
        ).shape[1]

        mlp_modules = [
            nn.Linear(self.flattened_dim, int(self.flattened_dim / 2)),
            nn.GELU()
        ]
        mlp_modules += [nn.Linear(int(self.flattened_dim / 2), n_classes)]
        self.readout = nn.Sequential(*mlp_modules)

    def forward(self, X: torch.Tensor, readout=True):
        conv_out = self.convs(X)
        flattened = torch.flatten(conv_out, start_dim=1)
        if not readout:
            return flattened
        logits = self.readout(flattened)
        return logits  # We predict the raw logits in forward!


# %%

cnn_end_model = MyCNN(in_channels=n_channels, hidden_dim=16, conv_layers=2, n_classes=C)

# %% md

# Coupling end-model into Weasel
#%%

from weasel.models import Weasel

weasel = Weasel(
    end_model=cnn_end_model,
    num_LFs=m,
    n_classes=C,
    encoder={'hidden_dims': [32, 10]},
    optim_encoder={'name': 'adam', 'lr': 1e-4},
    optim_end_model={'name': 'adam', 'lr': 1e-4}  # different way of getting the same optim with Hydra
)

# %% md

## Training Weasel and end-model

# %%

from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(monitor="Val/f1_macro", mode="max")

trainer = pl.Trainer(
    devices="auto",  # CPUs or GPUs
    accelerator="auto",  # DDP, 'gpu', 'cpu', 'tpu' ...
    max_epochs=3,  # since just for illustratory purposes
    logger=False,
    deterministic=True,
    callbacks=[checkpoint_callback]
)

trainer.fit(model=weasel, datamodule=weasel_datamodule)

## Evaluation

# The below will give the same test results
# test_stats = trainer.test(datamodule=weasel_datamodule, ckpt_path='best')

final_cnn_model = weasel.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path
).end_model
# Test the stand-alone, fully-trained CNN model:
pl.Trainer().test(model=final_cnn_model, dataloaders=weasel_datamodule.test_dataloader())

