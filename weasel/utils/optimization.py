import warnings
from logging import getLogger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from weasel.utils.utils import stem_word

log = getLogger(__name__)


def get_scheduler(optimizer, name, *args, **kwargs):
    """
    This utility function is only needed if you do *not* use Hydra.
    """
    name = stem_word(name)
    if name is None or name in ['no', 'none']:
        return None
    elif name in ['lron', 'reducelron', 'lronplateau', 'reducelronplateau']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, *args, **kwargs)
    elif name in ['cosine', 'cosineannealing']:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, *args, **kwargs)
    elif name in ['steplr', 'step_lr']:
        scheduler = optim.lr_scheduler.StepLR(optimizer, *args, **kwargs)
    else:
        raise ValueError(f'Unknown scheduler {name}.')
    return scheduler


def get_optimizer(model, name=None, **kwargs):
    """
    This utility function is only needed if you do *not* use Hydra.
    :param model: a nn.Module
    :param name: Name of an optimizer, defaults to Adam if None
    :param kwargs: Optimizer kwargs
    :return: the torch.optim you requested
    """
    name = name.lower().strip() if isinstance(name, str) else 'adam'

    parameters = get_trainable_params(model)
    for key, value in kwargs.items():
        if isinstance(value, str) and 'e-' in value:
            kwargs[key] = float(value)  # fix for when parsers parse scientific notation to string (e.g. '1e-4')
    if name == 'adam':
        return optim.Adam(parameters, **kwargs)
    elif name == 'sgd':
        return optim.SGD(parameters, **kwargs)
    else:
        raise ValueError("Unknown optimizer", name)


def get_trainable_params(model):
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    return trainable_params


def mig_loss_function(yhat, output2, p=None):
    # From Max-MIG crowdsourcing paper
    import numpy as np
    yhat = F.softmax(yhat, dim=1)
    output2 = F.softmax(output2, dim=1)
    batch_size, num_classes = yhat.shape
    I = torch.from_numpy(np.eye(batch_size), )
    E = torch.from_numpy(np.ones((batch_size, batch_size)))
    yhat, output2 = yhat.cpu().float(), output2.cpu().float()
    if p is None:
        p = torch.tensor([1 / num_classes for _ in range(num_classes)]).to(yhat.device)
    new_output = yhat / p
    m = (new_output @ output2.transpose(1, 0))
    noise = torch.rand(1) * 0.0001
    m1 = torch.log(m * I + I * noise + E - I)
    m2 = m * (E - I)
    return -(m1.sum() + batch_size) / batch_size + m2.sum() / (batch_size ** 2 - batch_size)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def squared_hellinger(yhat, yhat2):
    P_hat = F.softmax(yhat, dim=1)
    P_target = F.softmax(yhat2, dim=1)
    return torch.sum((torch.sqrt(P_hat) - torch.sqrt(P_target)) ** 2, dim=1).mean()


def probabilistic_CE_Weasel(yhat, target):
    log_P_hat = F.log_softmax(yhat, dim=1)
    P_target = F.softmax(target, dim=1)
    return -1 * torch.sum(P_target * log_P_hat, dim=1).mean()


def probabilistic_CE(yhat, target):
    log_P_hat = F.log_softmax(yhat, dim=1)
    return -1 * torch.sum(target * log_P_hat, dim=1).mean()


def hard_targets_CE(yhat, target):
    return F.cross_entropy(yhat, target.long())


def get_loss(name, logit_targets=True, probabilistic=True, reduction='mean'):
    # Specify loss function
    name = stem_word(name)
    if name in ['ce', 'crossentropy', 'cross_entropy']:
        if probabilistic:
            loss = probabilistic_CE_Weasel if logit_targets else probabilistic_CE
        else:
            assert not logit_targets, "Standard CE loss expects hard labels, if not a mistake, pass logit_targets=False"
            loss = hard_targets_CE
    elif name in ['maxmig', 'max-mig', 'mig']:
        log.info('Using the MaxMIG loss...')
        return mig_loss_function
    elif name in ['squaredhellinger', 'hellinger']:
        return squared_hellinger
    else:
        # DISTANCES
        if name in ['l1', 'mae']:
            loss = nn.L1Loss(reduction=reduction)
        else:
            warnings.warn(f'{name} not supported. Defaulting to WeaSEL cross-entropy loss.')
            loss = probabilistic_CE_Weasel
    return loss


def is_symmetric_loss(name):
    name = name.lower()
    if name in ['maxmig', 'max-mig', 'mig', 'l1', 'mae', 'mse', 'l2'] or \
            name in ['squaredhellinger', 'hellinger'] or \
            name in ['ce-asym-rev', 'ce_asym_rev', 'ce-asym', 'ce_asym', 'ce_sym_no_stop-grad']:
        return True
    return False
