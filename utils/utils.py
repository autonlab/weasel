import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class dSiLU(nn.Module):
    def forward(self, input):
        return torch.sigmoid(input.float()) * (1 + input * (1 - torch.sigmoid(input.float())))


class Mish(nn.Module):
    def __call__(self, x):
        return x * torch.tanh(F.softplus(x))


def get_activation_function(name, functional=False):
    name = name.lower()

    funcs = {"softmax": F.softmax, "relu": F.relu, "tanh": F.tanh, "sigmoid": torch.sigmoid, "identity": None,
             None: None}
    nn_funcs = {"softmax": nn.Softmax(dim=-1), "relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
                "identity": nn.Identity(), 'silu': nn.SiLU(), 'elu': nn.ELU(), 'prelu': nn.PReLU(),
                'swish': nn.SiLU(), 'mish': Mish(), 'dsilu': dSiLU(), 'gelu': nn.GELU()}
    if functional:
        return funcs[name]
    else:
        return nn_funcs[name]


def change_labels(*args, new_label=-1, old_label=0):
    lst = []
    for arg in args:
        A = arg.copy()
        new_old = (A == new_label)
        A[A == old_label] = new_label
        A[new_old] = old_label
        if len(args) == 1:
            return A
        lst.append(A)

    return tuple(lst)


def load_model(model_name):
    if "E2E" in model_name:
        from end_to_end_ws_model import E2E
        return E2E
    else:
        raise ValueError("Unknown model", model_name)


def set_seed(seed, device='cuda'):
    import random, torch
    # setting seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
