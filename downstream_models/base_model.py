import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from utils.model_logging import update_tqdm
from utils.optimization import get_loss, get_optimizer, get_scheduler
from utils.prediction_and_evaluation import get_optimal_decision_threshold, eval_final_predictions
from utils.utils import set_seed


class DownstreamBaseModel(nn.Module):
    """
    This is a template class, that should be inherited by all downstream models.
    It defines all necessary operations, although __init__ and forward will definitely need to be implemented
    by the concrete downstream model.
    """

    def __init__(self, params, *args, **kwargs):
        super().__init__()
        self.out_size = -1
        # raise NotImplementedError("Please create the end model's architecture here")

    def forward(self, X, device='cuda'):
        """
        Downstream model forward pass
        :param X: A feature tensor
        :param device: E.g. 'cuda' or 'cpu'
        :return: The logits (not probabilities!)
        """
        X = X.to(device)
        # raise NotImplementedError("Please implement the downstream model's forward pass here")

    def get_encoder_features(self, X, device='cuda'):
        """
        This method returns features that may be used by the encoder network for label generation.
        Usually, they can just be the same features X, though they may be processed by the network too.
        :param X: the same features that are provided to the main model in self.forward(.)
        :param device: E.g. 'cuda' or 'cpu'
        :return: A (n, d) tensor that is usable by the encoder network (usually just the same features)
        """
        return X.to(device)

    def predict_proba(self, X, device='cuda'):
        """
        This is a wrapper method, and will only differ to the snippet below when the forward method does not return
        probabilities (e.g. in cases where the loss computes them by itself, e.g. in BCEwithLogits)
        :param X: A feature tensor
        :param device: E.g. 'cuda' or 'cpu'
        :return: The probabilities P(Y = 1| X) assigned by the downstream model.
        """
        return self.logits_to_probs(self.forward(X, device=device))

    def logits_to_probs(self, logits):
        if self.out_size == 1:
            return torch.sigmoid(logits)
        else:
            return nn.Softmax(dim=1)(logits)

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
        elif 1 <= self.out_size <= 2:
            cutoff = cutoff or 0.5
            preds = probs.clone() if isinstance(probs, torch.Tensor) else probs.copy()
            if self.out_size == 2:
                preds = preds[:, 1]
            preds[preds >= cutoff] = 1
            preds[preds < cutoff] = 0
        else:
            preds = torch.argmax(probs, dim=1) if isinstance(probs, torch.Tensor) else np.argmax(probs, axis=1)
        return probs, preds

    def __str__(self):
        return 'EndModel'


class DownstreamBaseTrainer:
    """
    A template class to be wrapped around for training & evaluating a
     specific downstream model that inherits from DownstreamBaseModel.

    To wrap around it you just need to inherit from it and overwrite as follows (and replace xyz with an appropriate name):
    >>
        class YourModelTrainer(DownstreamBaseTrainer):
            def __init__(self, model_params, name='xyz', model_dir="out/xyz"):
                super().__init__(downstream_params, name=name, model_dir=model_dir)
                self.model_class = YourModelClass
                self.name = name
    where
         YourModelClass is a pointer to your model class, e.g. ResNet, MLPNet,...
         name  is a name ID for your model, e.g. 'ResNet', 'MLP',...
    """

    def __init__(self, downstream_params, name='_xx_', seed=None, verbose=False, model_dir=None,
                 notebook_mode=False, model=None):
        if seed is not None:
            set_seed(seed)
        self.model_class = DownstreamBaseModel
        self.model = model
        self.downstream_params = downstream_params
        self.verbose = verbose
        self.model_dir = model_dir
        self.cutoff = 0.5
        self.name = name
        if notebook_mode:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        self.tqdm = tqdm
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _get_model_dir(self):
        return f"out/{self.name}_model" if self.model_dir is None else self.model_dir

    def _get_best_model_path(self):
        return f'{self._get_model_dir()}best_{self.name}_model.pkl'

    def _get_model(self):
        if self.model is None:
            return self.model_class(self.downstream_params)
        return self.model

    def fit(self, train_data, hyper_params, valloader=None, testloader=None, device="cuda", trainloader=None, use_wandb=True):
        self.batch_size = hyper_params["batch_size"]
        if trainloader is None:
            trainloader = DataLoader(train_data, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        n_epochs = hyper_params["epochs"]
        adjust_thresh = hyper_params['adjust_thresh'] if 'adjust_thresh' in hyper_params else False
        self.model = self._get_model().to(device)
        self.tqdm_update = update_tqdm
        # Specify loss function
        criterion = get_loss(hyper_params['loss'], with_logits=True, logits_len=self.model.out_size)

        # Initialize Optimizer and Scheduler
        optim_name = hyper_params['optim']  # 'adam'
        scheduler_name = hyper_params['scheduler']
        lr, wd = hyper_params["lr"], hyper_params['mlp_weight_decay']
        optimizer = get_optimizer(optim_name, self.model, lr=lr, weight_decay=wd)
        scheduler = get_scheduler(scheduler_name, self.model, optimizer, total_epochs=n_epochs)

        # Cycle through epochs
        val_metric = hyper_params['val_metric'].lower()
        val_stats = test_stats = None
        example_count = 0
        valid_metric_val = best_valid_val = 1e5 if lower_is_better(val_metric) else -1e5
        with self.tqdm(range(1, n_epochs + 1)) as t:
            for epoch in t:
                t.set_description(f'{self.name}')
                start_t = time.time()
                self.model.train()
                epoch_loss = 0.0
                # Cycle through batches
                for i, (batch_features, batch_y) in enumerate(trainloader, 1):
                    y = batch_y.to(device)
                    example_count += batch_y.shape[0]
                    optimizer.zero_grad()

                    # Forward pass through
                    yhat = self.model.forward(batch_features, device=device)
                    loss = criterion(yhat, y, logit_targets=False)
                    loss.backward()  # Compute the gradients
                    optimizer.step()  # Step the optimizer to update the model weights
                    epoch_loss += loss.detach().item()

                epoch_loss, duration = epoch_loss / i, time.time() - start_t
                logging_dict = {"epoch": epoch, "Train loss f": epoch_loss}
                if valloader is not None:
                    _, _, val_stats = self.evaluate(valloader, device=device, adjust_thresh=adjust_thresh)
                    valid_metric_val = val_stats[val_metric]
                    logging_dict = {**logging_dict, **{'Val F1': val_stats['f1'], 'Val AUC': val_stats['auc'],
                                                       'Val Acc.': val_stats['accuracy']}}
                scheduler.step(self.model, epoch, valid_metric_val)

                if testloader is not None:
                    _, _, test_stats = self.evaluate(testloader, device=device)
                    logging_dict = {**logging_dict,
                                    **{'Test F1': test_stats['f1'], 'Test AUC': test_stats['auc'],
                                       'Test Acc.': test_stats['accuracy'], 'Test Recall': test_stats['recall'],
                                       'Test Precision': test_stats['precision']}
                                    }
                self.tqdm_update(t, train_loss=epoch_loss, time=duration, val_stats=val_stats, test_stats=test_stats)
                best_valid_val = self._save_best_model(val_stats, best_valid_val, hyper_params, metric_name=val_metric)
                if use_wandb:
                    wandb.log(logging_dict, step=example_count)

        if valloader is None:
            self._save_model(hyper_params)  # save last checkpoint if no validation loader was given
        return best_valid_val

    def evaluate(self, dataloader, device="cuda", print_prefix=None, adjust_thresh=False, use_best_model=False):
        if use_best_model:
            if self.model_dir is not None:
                saved_model = torch.load(self._get_best_model_path())
                model = self.model_class(self.downstream_params).to(device)
                try:
                    model.load_state_dict(state_dict=saved_model['model'])
                    self.cutoff, adjust_thresh = saved_model['cutoff'], False
                except FileNotFoundError:
                    print('No model was saved, since no validation set was given, using the last one.')
                    model = self.model
            else:
                print('No models were saved, using the current one.')
                model = self.model
        else:
            model = self.model
        verbose = True
        if print_prefix is None:
            print_prefix = self.name
            verbose = False
        probs, Y = self.get_preds(dataloader, labels_too=True, model=model, device=device)

        try:
            if adjust_thresh:
                if verbose:
                    print("Adjusting decision threshold, this should be done on a validation set.")
                self.cutoff = get_optimal_decision_threshold(probs, Y)
            _, hard_preds = model.predict(probs=probs, cutoff=self.cutoff, device=device)
            stats = eval_final_predictions(Y, hard_preds, probs=probs, abstention=0.5, verbose=verbose,
                                           neg_label=0, model_name=print_prefix, is_binary=self.model.out_size <= 2)
            stats['cutoff'] = self.cutoff
        except ValueError as e:
            print(e)
            stats = {'accuracy': -1, 'recall': -1, 'precision': -1, 'f1': -1, 'auc': -1}
        return probs, Y, stats

    def get_preds(self, dataset, labels_too=False, model=None, device='cuda'):
        if model is None:
            model = self.model
        if not isinstance(dataset, torch.utils.data.DataLoader):
            dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False)
        else:
            dataloader = dataset
        model.eval()
        with torch.no_grad():
            for i, (batch_features, batch_y) in enumerate(dataloader):
                batch_y = batch_y.numpy()

                yhat = model.predict_proba(batch_features)
                yhat = yhat.data.cpu().numpy()
                if i == 0:
                    Y, probs = batch_y, yhat
                else:
                    Y, probs = np.concatenate((Y, batch_y), axis=0), np.concatenate((probs, yhat), axis=0)
        if labels_too:
            return probs, Y
        return probs

    def _save_best_model(self, new_model_stats, old_best, hyperparams, metric_name='auc', optimizer=None):
        r"""
        :param new_model_stats: a metric dictionary containing a key metric_name
        :param old_best: Old best metric_name value
        :param metric_name: A metric, e.g. F1  (note that the code only supports metrics, where higher is better)
        :return: The best new metric, i.e. old_best if it is better than the newer model's performance and vice versa.
        """
        if new_model_stats is None:
            return old_best
        if (lower_is_better(metric_name) and new_model_stats[metric_name] < old_best) or \
                (not lower_is_better(metric_name) and new_model_stats[
                    metric_name] > old_best):  # save best model (wrt validation data)
            best = new_model_stats[metric_name]
            print(f"Best model so far with validation {metric_name} =", '{:.3f}'.format(best))
            self._save_model(hyperparams, new_model_stats['cutoff'] if 'cutoff' in new_model_stats else -1)
        else:
            best = old_best
        return best

    def _save_model(self, hyper_params, cutoff=0.5):
        checkpoint_dict = {
            'model': self.model.state_dict(),
            'cutoff': cutoff,
            # 'optimizer': optimizer.state_dict(),
            'metadata': {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'hyper_params': hyper_params,
                'model_params': self.downstream_params
            }
        }
        # In case a model dir was given --> save best model (wrt validation data)
        if self.model_dir is not None:
            torch.save(checkpoint_dict, self._get_best_model_path())


def lower_is_better(metric_name):
    metric_name = metric_name.lower().strip()
    if metric_name in ['mse', 'l2', 'mae', 'l1']:
        return True
    return False
