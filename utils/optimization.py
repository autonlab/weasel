import warnings

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC


class Scheduler(ABC):
    def __init__(self, model, optimizer, total_epochs, *args, **kwargs):
        self.model = model
        self.n_epochs = total_epochs
        pass

    def step(self, model, epoch, metric):
        pass


class ReduceLROnPlateau(Scheduler):
    def __init__(self, model, optimizer, total_epochs, *args, **kwargs):
        super().__init__(model, optimizer, total_epochs, *args, **kwargs)
        patience = 15 if total_epochs <= 100 else 25 if total_epochs <= 250 else 50
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=patience,
                                                              verbose=True)
        self.start = int(total_epochs / 10)

    def step(self, model, epoch, metric):
        if epoch > self.start:
            self.scheduler.step(metric)


class Cosine(Scheduler):
    def __init__(self, model, optimizer, total_epochs, *args, **kwargs):
        super().__init__(model, optimizer, total_epochs, *args, **kwargs)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    def step(self, model, epoch, metric):
        self.scheduler.step()


class StepLR(Scheduler):
    def __init__(self, model, optimizer, total_epochs, *args, **kwargs):
        super().__init__(model, optimizer, total_epochs, *args, **kwargs)
        step_size = kwargs['step_size'] if 'step_size' in kwargs else 30
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

    def step(self, model, epoch, metric):
        self.scheduler.step()


def get_scheduler(name, model, optimizer, total_epochs, *args, **kwargs):
    name = name.lower().strip()
    if name is None or name in ['no', 'none']:
        return Scheduler(model, optimizer, total_epochs)
    elif name in ['lron', 'reducelron', 'lronplateau', 'reducelronplateau']:
        return ReduceLROnPlateau(model, optimizer, total_epochs, *args, **kwargs)
    elif name in ['cosine', 'cosineannealing']:
        return Cosine(model, optimizer, total_epochs, *args, **kwargs)
    elif name in ['steplr', 'step-lr']:
        return StepLR(model, optimizer, total_epochs, *args, **kwargs)
    else:
        raise ValueError('Unknown scheduler!', name, ' (#epochs=', total_epochs, ')')


def get_optimizer(name, model, **kwargs):
    name = name.lower().strip()
    parameters = get_trainable_params(model)
    if name == 'adam':
        lr = kwargs['lr'] if 'lr' in kwargs else 1e-4
        wd = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0
        return optim.Adam(parameters, lr=lr, weight_decay=wd)
    elif name == 'sgd':
        print('Using SGD optimizer')
        lr = 0.01  # kwargs['lr'] if 'lr' in kwargs else 0.01
        momentum = 0.9
        wd = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=wd)
    else:
        raise ValueError("Unknown optimizer", name)


def get_trainable_params(model):
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    return trainable_params


#################################################################################################################
def KLDivergence(yhat, target):
    return F.kl_div(torch.log(yhat), target, reduction='batchmean')


def KLDivergenceWithLogits(yhat, target):
    return F.kl_div(F.log_softmax(yhat, dim=1), F.log_softmax(target, dim=1), reduction='batchmean', log_target=True)


def KLDivergenceWithLogitsBinary(yhat, target):
    return F.kl_div(F.logsigmoid(yhat), F.logsigmoid(target), reduction='batchmean', log_target=True)

def mig_loss_function(yhat, output2, p=None, logit_targets=True):
    # From Max-MIG crowdsourcing paper
    import numpy as np
    yhat = F.softmax(yhat, dim=1)
    output2 = F.softmax(output2, dim=1) if logit_targets else output2
    batch_size, num_classes = yhat.shape
    I = torch.FloatTensor(np.eye(batch_size), )
    E = torch.FloatTensor(np.ones((batch_size, batch_size)))
    yhat, output2 = yhat.cpu().float(), output2.cpu().float()
    if p is None:
        p = torch.tensor([1 / num_classes for _ in range(num_classes)]).to(yhat.device)
    new_output = yhat / p
    m = (new_output @ output2.transpose(1, 0))
    noise = torch.rand(1) * 0.0001
    m1 = torch.log(m * I + I * noise + E - I)
    m2 = m * (E - I)
    return -(m1.sum() + batch_size) / batch_size + m2.sum()/ (batch_size ** 2 - batch_size)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def BCEWithLogits(yhat, target):
    # return F.binary_cross_entropy_with_logits(yhat, torch.round(torch.sigmoid(target)))
    return F.binary_cross_entropy_with_logits(yhat, torch.sigmoid(target))


def squared_hellinger(yhat, yhat2, logit_targets=True):
    P_hat = F.softmax(yhat, dim=1)
    P_target = F.softmax(yhat2, dim=1) if logit_targets else yhat2
    return torch.sum((torch.sqrt(P_hat) - torch.sqrt(P_target)) ** 2, dim=1).mean()

def CEWithLogits(yhat, target, logit_targets=True):
    log_P_hat = F.log_softmax(yhat, dim=1)
    P_target = F.softmax(target, dim=1) if logit_targets else target
    return -1 * torch.sum(P_target * log_P_hat, dim=1).mean()


def CEWithLogits_AsymAblations(yhat, target, logit_targets=True):
    # Asymmetric cross-entropy for ablations, using the encoder/LF-based based labels as targets
    return CEWithLogits(yhat, target, logit_targets)


def CEWithLogits_AsymRevAblations(yhat, target, logit_targets=True):
    # Asymmetric cross-entropy for ablations, using the end-model predictions as targets
    return CEWithLogits(target, yhat, logit_targets)


def CEWithLogits_SymNoStopGrad(yhat, target, logit_targets=True):
    # Same loss as used in the main experiments, but without stop-grad on targets, for the ablations
    return CEWithLogits(yhat, target, logit_targets) / 2 + \
           CEWithLogits(target, yhat, logit_targets) / 2


def MSE(yhat, target, *args, **kwargs):
    return nn.MSELoss()(yhat, target)


class DistanceWithLogitsFunction:
    def __init__(self, distance, logit_to_prob_func):
        self.d = distance
        self.to_probs = logit_to_prob_func

    def __call__(self, yhat, target):
        return self.d(self.to_probs(yhat), self.to_probs(target))


def get_loss(name, reduction='mean', with_logits=False, logits_len=2):
    # Specify loss function
    if logits_len < 1:
        raise ValueError(f'Logits must have a length >=1, but {logits_len} was given. Did you forget to update the'
                         f' out_size field of your downstream model? Please set YourModel.out_size = #outputs.')
    name = name.lower().strip()
    if name in ['bce', 'binary-ce', 'binaryce', 'ce', 'crossentropy', 'cross-entropy']:
        if logits_len == 1:
            if with_logits:
                loss = BCEWithLogits
            else:
                loss = nn.BCELoss(reduction=reduction)
        else:
            if with_logits:
                loss = CEWithLogits  # single_CE_loss # CEWithLogits
            else:
                loss = nn.CrossEntropyLoss(reduction=reduction)
    elif name in ['ce-asym', 'ce_asym']:
        return CEWithLogits_AsymAblations
    elif name in ['ce-asym-rev', 'ce_asym_rev']:
        return CEWithLogits_AsymRevAblations
    elif name in ['ce_sym_no_stop-grad']:
        return CEWithLogits_SymNoStopGrad
    elif name in ['maxmig', 'max-mig',  'mig']:
        print('Using the MaxMIG loss...')
        return mig_loss_function if logits_len >= 2 else mig_loss_function
    elif name in ['squaredhellinger', 'hellinger']:
        return squared_hellinger
    elif name in ['kl', 'kl-div', 'kldiv', 'kl-divergence', 'kldivergence']:
        if with_logits:
            loss = KLDivergenceWithLogits if logits_len >= 2 else KLDivergenceWithLogitsBinary
        else:
            loss = KLDivergence
    else:
        # DISTANCES
        if name in ['l1', 'mae']:
            loss = nn.L1Loss(reduction=reduction)
        elif name in ['l2', 'mse']:
            loss = MSE
        else:
            warnings.warn(f'WARNING: {name} not supported. Defaulting to BCE loss.')
            return nn.BCELoss(reduction=reduction) if not with_logits else BCEWithLogits  # default

        if with_logits:
            prob_func = nn.Sigmoid() if logits_len == 1 else nn.Softmax(dim=1)
            loss = DistanceWithLogitsFunction(loss, logit_to_prob_func=prob_func)
    return loss


def is_symmetric_loss(name):
    name = name.lower()
    if name in ['maxmig', 'max-mig', 'mig', 'l1', 'mae', 'mse', 'l2'] or \
            name in ['squaredhellinger', 'hellinger'] or \
            name in ['ce-asym-rev', 'ce_asym_rev', 'ce-asym', 'ce_asym', 'ce_sym_no_stop-grad']:
        return True
    return False
