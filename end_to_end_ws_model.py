from datetime import datetime
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.tensorboard import SummaryWriter

from utils.optimization import get_scheduler, get_optimizer, get_loss, is_symmetric_loss
from utils.utils import set_seed, get_activation_function
from utils.model_logging import log_epoch_vals, update_tqdm
from utils.prediction_and_evaluation import eval_final_predictions, get_optimal_decision_threshold

from encoder_network import Encoder, MulticlassEncoder


class E2E(nn.Module):
    """
    This is the end-to-end model class that glues the encoder and downstream network for learning both of them
    in parallel, by maximizing their agreements.
    We write
     n - the number of samples or data points in a batch
     m - the number of labeling functions (LF)
     d - feature dimensionality, i.e. number of features used by the downstream/end model

    The get_labels and evaluate functions are helper functions to easily evaluate a reloaded E2E model.
    For training it, use the E2ETrainer class.
    """

    def __init__(self, encoder_params, endmodel, Xdummy, input_len=None, encoder_net=Encoder, verbose=True, *args):
        """
        :param encoder_params: Dictionary of encoder network prameters, see our provided config file for details.
        :param endmodel: An instantiated PyTorch downstream model
                            that inherits from downstream_models.DownstreamBaseModel
        """
        super().__init__()
        assert encoder_params["neg_label"] != encoder_params[
            "abstention"], "Negative class label mustn't equal abstention vote"
        assert encoder_params['neg_label'] == 0, 'Please make sure that the target labels are in {0, 1}'
        self.p = encoder_params['class_balance'] if 'class_balance' in encoder_params else [0.5, 0.5]
        self.features_for_encoder = encoder_params["use_features_for_enc"]
        accuracy_func = get_activation_function(encoder_params["accuracy_func"])
        activation_func = encoder_params["act_func"] if 'act_func' in encoder_params else 'relu'
        enc_dims = encoder_params['encoder_dims']
        dropout = encoder_params["dropout"]
        class_conditional = encoder_params["class_conditional"] if 'class_conditional' in encoder_params else False
        self.num_LFs = encoder_params["num_LFs"]
        self.cutoff = 0.5  # decision threshold for the downstream classifier
        self.cardinality = encoder_params["cardinality"] if 'cardinality' in encoder_params else 2  # number of classes
        self.downstream_model = endmodel
        T = encoder_params["temperature"] if 'temperature' in encoder_params else 1

        self.input_len = self.num_LFs if input_len is None else input_len
        print(self)
        if self.features_for_encoder:
            if Xdummy is None:
                self.input_len += encoder_params['n_features']
                print('WARNING, INFERRED THAT', encoder_params['n_features'], 'FEATURES ARE FED TO THE ENCODER')
            else:
                X_for_encoder_dummy = self.downstream_model.cuda().get_encoder_features(Xdummy, device='cuda')
                self.input_len += X_for_encoder_dummy.shape[-1]
                print('Inferred an encoder input size of', self.input_len, f"(of which {self.num_LFs} are LFs).")
                out_size_down = self.downstream_model.cuda().forward(Xdummy, device='cuda').shape
                if len(out_size_down) == 1:
                    print(out_size_down)
                    assert self.cardinality == 2 and encoder_net == Encoder, 'Encoder requires a single output! or use MulticlassEncoder.'
                else:
                    assert out_size_down[
                               1] == self.cardinality, f'Out size is {out_size_down}, but should be #classes = {self.cardinality}'
        if self.cardinality == 2:
            self.encoder = encoder_net(self.input_len, enc_dims, self.num_LFs, drop_prob=dropout, temperature=T,
                                       accuracy_func=accuracy_func, batch_norm=encoder_params["batch_norm"],
                                       act_func=activation_func,
                                       acc_scaler=encoder_params["accuracy_scaler"],
                                       class_conditional_accs=class_conditional)
        else:  # feel free to use this one for all cardinalities
            self.encoder = MulticlassEncoder(self.input_len, enc_dims, self.num_LFs, drop_prob=dropout,
                                             act_func=activation_func,
                                             cardinality=self.cardinality, class_conditional_accs=class_conditional,
                                             accuracy_func=accuracy_func, batch_norm=encoder_params["batch_norm"],
                                             acc_scaler=encoder_params["accuracy_scaler"], temperature=T)
        if verbose:
            pass  # print('ENCODER architecture:', self.encoder.network)

    def forward(self, LF_outputs, features, L_tensor=None, return_certain_mask=True, device='cuda', *args):
        """
        :param LF_outputs: A (n, m) label matrix, where n = #samples, m = #labeling functions (LF)
        :param features: The input for the downstream model, a (n, d') tensor
        :param args: Additional arguments for the downstream model
        :param return_certain_mask: Whether to return a mask m of shape (n, 1),
                                        where m_i = True if Y_encoder_i != 0.5 (i.e. when at least one LF has voted)
                                        and m_i = False when all LFs abstained for sample i.
        :return: a tuple (Y_endmodel, Y_encoder), each of shape (n,)
        """
        # Generate probabilistic labels from our encoder
        label_logits, accuracies = self.encode(LF_outputs, features, L_tensor)
        # Produce the predictions of the downstream model
        ylogits = self.downstream_model.forward(features, *args, device=device)
        if return_certain_mask:
            # return a mask that should be used to avoid using samples for training where all LFs abstained
            encoder_labels = self.encoder.logits_to_probs(label_logits)
            if self.cardinality == 2 and len(encoder_labels.shape) == 1:
                certains = (encoder_labels != 0.5)  # no need for branching, but it's more clear this way
            else:
                certains = torch.any(encoder_labels != 1 / self.cardinality, dim=1)
            return ylogits, label_logits, certains, accuracies
        return ylogits, label_logits, accuracies

    def get_attention_scores(self, LF_outputs, feats_for_enc):
        """
        :param LF_outputs: a (n, m) tensor, where n = #samples, m = #LFs
        :param feats_for_enc: Either None, or a (n, d) tensor that will be concatenated to the label_matrix
        :return: a (n, m) tensor that represents the sample-dependent accuracies of the LFs
        """
        return self.encoder.get_attention_scores(LF_outputs, feats_for_enc)

    def encode(self, LF_outputs, features=None, L_tensor=None, device='cuda'):
        if self.features_for_encoder:
            features = self.downstream_model.get_encoder_features(features, device=device)
        else:
            features = None
        labels, accs = self.encoder.forward(
            LF_outputs, extra_input=features, label_matrix=L_tensor, get_accuracies=True
        )  # Generate latent labels
        return labels, accs

    def predict_proba(self, x, *args, device='cuda'):
        r""" Final prediction \in (0, 1) of the downstream model, where x are the features used by it """
        return self.downstream_model.predict_proba(x, *args, device=device)

    def predict(self, features=None, soft_labels=None, device='cuda'):
        """
        If no soft labels are given, please provide features to get the corresponding predictions from the end moodel.
        :param features: Either None, or a (n, d) tensor that will be used for prediction by the downstream model
        :param soft_labels: Either None or a (n, 1) tensor of probabilistic labels
        :return:  A tuple (Y_soft, Y_hard), where Y_soft are the class probabilities
                                    (interpreted as P(Y = 1| LFs, X) if binary),
                                    Y_hard labels are the corresponding class membership predictions.
        """
        return self.downstream_model.predict(features, probs=soft_labels, cutoff=self.cutoff, device=device)

    def get_labels(self, dataloader, device='cuda'):
        """
        :param dataloader: A torch DataLoader instance, that returns batches (L, X), where
                            L is the label matrix for the batch (n, m), and X the corresponding features (n, d)
        :param device: where to compute, e.g. 'cuda', 'cpu'
        :return: A tuple (Y_encoder, Y_endmodel), each of shape (n,) where n = #samples and
                Y_encoder are the labels generated by the encoder network, and
                Y_endmodel are the final predictions of the downstream model.
        """
        self.eval()
        Y_encoder, Y_preds = None, None
        with torch.no_grad():
            for i, (batch_L, batch_features) in enumerate(dataloader):
                batch_L = batch_L.to(device)

                yhat, y_enc, _ = self.forward(batch_L, batch_features, return_certain_mask=False, device=device)
                yhat = self.downstream_model.logits_to_probs(yhat)
                y_enc = self.encoder.logits_to_probs(y_enc)
                yhat = yhat.detach().cpu().numpy()
                y_enc = y_enc.detach().cpu().numpy()
                if i == 0:
                    Y_preds, Y_encoder = yhat, y_enc
                else:
                    Y_preds, Y_encoder = np.concatenate((Y_preds, yhat), axis=0), np.concatenate((Y_encoder, y_enc),
                                                                                                 axis=0)
        return Y_encoder, Y_preds

    def evaluate(self, dataloader, device="cuda", prefix='E2E', verbose=True, adjust_thresh=False, cutoff=None):
        """
        This function evaluates the downstream model based on gold labels, in case some are known.
        :param dataloader: A torch DataLoader instance, that returns batches (X, Y), where
                            Y (n,) are true labels corresponding to the features X (n, d)
        :param device: where to compute, e.g. 'cuda', 'cpu'
        :param adjust_thresh: Whether to fine-tune the downstream model's decision threshold (which by default is 0.5)
        :return: A tuple (Y_endmodel, Y_true, stats_dict), where
                Y_endmodel are the final predictions of the downstream model
                Y_true are the corresponding gold labels provided by the dataloader, and
                stats_dict is a dictionary of metric_name --> metric_value pairs, including
                'accuracy', 'f1', 'auc', 'recall', 'precision'
        """
        self.eval()
        Y, preds = None, None
        with torch.no_grad():
            for i, (batch_features, batch_y) in enumerate(dataloader):
                batch_y = batch_y.numpy()

                yhat = self.predict_proba(batch_features, device=device)
                yhat = yhat.detach().cpu().numpy()
                if i == 0:
                    Y, preds = batch_y, yhat
                else:
                    Y, preds = np.concatenate((Y, batch_y), axis=0), np.concatenate((preds, yhat), axis=0)

        try:
            if adjust_thresh:
                if verbose:
                    print("Adjusting decision threshold, this should be done on a validation set.")
                self.cutoff = get_optimal_decision_threshold(preds, Y)
            elif cutoff is not None:
                self.cutoff = cutoff

            # get the hard class predictions for the metrics that require them (i.e. F1, Accuracy, etc.)
            _, hard_preds = self.predict(soft_labels=preds, device=device)
            # Evaluate Y_true, vs Y_endmodel (where we use the soft labels for all metrics that support them)
            stats = eval_final_predictions(Y, hard_preds, probs=preds, abstention=0.5, verbose=verbose,
                                           neg_label=0, only_on_labeled=False, model_name=prefix,
                                           is_binary=self.cardinality == 2)
            stats['cutoff'] = self.cutoff
        except ValueError as e:
            print(e)
            stats = {'accuracy': -1, 'recall': -1, 'precision': -1, 'f1': -1,  'auc': -1, 'cutoff': 0.5}
        return preds, Y, stats


class E2ETrainer:
    """
    This is a wrapper class around the end-to-end multi-source weak supervision model, that allows to train it.
    We write
     n - the number of samples or data points in a batch
     m - the number of labeling functions (LF)
     d - feature dimensionality, i.e. number of features used by the downstream/end model
    """

    def __init__(self, encoder_params, downstream_model, dirs, seed, notebook_mode=False, encoder_net=Encoder):
        """

        :param encoder_params: Dictionary of necessary encoder network parameters, see our provided config files for details.
        :param downstream_model: The instantiated downstream model.
        :param dirs: None, or a Dictionary containing directory path-entries for 'logging', and 'checkpoints', where
                        - Losses, timings, metrics will be logged in the 'logging' directory with tensorboard
                        - Model checkpoints will be saved in the 'checkpoints' directory (including the best model w.r.t
                            validation set performance)
        :param seed: A random seed for reproducibility
        :param notebook_mode: Only relevant for displaying a training bar with tqdm
                    (better to be true if training in a Jupyter notebook)
        """
        self.num_datapoints_seen = 0
        set_seed(seed)
        print('Encoder net params:', ', '.join([f'{k.upper()}: {i}' for k, i in encoder_params.items()]))
        self.model = None
        self.encoder_params = encoder_params
        self.encoder_net = encoder_net
        self.endmodel = downstream_model
        self.trainloader = None
        self.total_epochs = -1
        self.p = encoder_params['class_balance'] if 'class_balance' in encoder_params else [0.5, 0.5]
        self.scheduler1, self.scheduler2 = None, None
        if dirs is not None:
            self.log_dir = dirs['logging'] if 'logging' in dirs else None
            self.model_dir = dirs['checkpoints']
        else:
            self.log_dir, self.model_dir = (None, None)
        if notebook_mode:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        self.tqdm = tqdm

    def _train_epoch(self, train_loader, encoder_optimizer, downstream_optimizer, criterion, loss_encoder, epoch,
                     device="cuda", single_loss=False):
        """
        This method trains the E2E model for a single epoch (i.e. iterates over all batches) and return the resulting
        training loss.
        :param train_loader: A torch DataLoader that returns the training batches (L, X), where
                                L is the label matrix of the batch (n, m)
                                X are the features of the batch (n, d)
        :param encoder_optimizer: An optimizer to use for the encoder network (e.g. SGD or Adam)
        :param downstream_optimizer: An optimizer to use for the downstream model (e.g. SGD or Adam)
        :param criterion: The loss function, e.g. L1Loss, or BCELoss
        :param device: E.g. 'cuda', or 'cpu'
        :return: The training loss of the epoch
        """
        self.model.train()
        n_certains = 0
        epoch_loss, enc_loss, end_loss = 0.0, 0.0, 0.0

        # Cycle through batches
        for i, (batch_L, batch_features) in enumerate(train_loader, 1):
            lf_votes_tensor = batch_L.to(device)
            self.num_datapoints_seen += batch_L.shape[0]

            # clear the gradients in the optimizers
            encoder_optimizer.zero_grad()
            downstream_optimizer.zero_grad()

            # Forward pass through E2E means generating labels from the encoder, and downstream predictions yhat
            ylogits, label_logits, certains, accuracies1 = self.model.forward(
                lf_votes_tensor, batch_features, return_certain_mask=True, device=device
            )
            # mask samples out where all LFs abstained with [certains]
            if single_loss:
                loss = criterion(ylogits[certains], label_logits[certains])
                loss.backward()
                enc_loss += loss.detach().item() / 2
                end_loss += loss.detach().item() / 2
            else:
                endmodel_loss = criterion(ylogits[certains], label_logits.detach()[certains])
                encoder_loss = loss_encoder(label_logits[certains], ylogits.detach()[certains])

                encoder_loss.backward()
                endmodel_loss.backward()
                enc_loss += encoder_loss.detach().item()
                end_loss += endmodel_loss.detach().item()

            # Step the optimizers to update the model weights
            encoder_optimizer.step()
            downstream_optimizer.step()

        enc_loss, end_loss = enc_loss / i, end_loss / i
        epoch_loss = (enc_loss + end_loss) / 2

        return (epoch_loss, enc_loss, end_loss), n_certains  # Return the loss value to track training progress

    def fit(self, train_loader, valloader, hyper_params, testloader=None, device='cuda', model=None, use_wandb=True):
        r"""
        :param train_loader: A torch DataLoader that returns the training batches (L, X), where
                                L is the label matrix of the batch (n, m)
                                X are the features of the batch (n, d)
        :param valloader: A torch DataLoader that returns the validation set batches (Y, X), where
                                Y are gold/true labels of the batch (n, 1)
                                X are the features of the batch (n, d).
                            This set is used for early-stopping/saving a checkpoint of the model that best performs
                            on this validation set (as well as decision threshold tuning, if adjust_thresh is True).
        :param hyper_params: A dictionary of all necessary hyperparameters, please check our provided config files to
                                to see all required parameters (includes optimizer 'lr' and 'weight_decay',
                                    the loss function name 'loss', the number of epochs 'epochs' etc.)
                                If you use custom ones, we recommend to build upon our config files.
        :param testloader: None, or a torch DataLoader that returns the test set batches (Y, X), where
                                Y are gold/true labels of the batch (n, 1)
                                X are the features of the batch (n, d).
                            If given, this set is only used for printing out model test performances.
        :param device: E.g. 'cuda' or 'cpu'
        :param model: None, or an instantiated E2E model to use for training/evaluating
        - adjust_thresh: Whether to fine-tune the downstream model's decision threshold (which by default is 0.5)
        :return: The strongest validation set performance (can be reloaded in the evaluate function).
        """
        # A logger for training loss plots, validation performances plots etc., in tensorboard
        writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir is not None else None
        print("Train End-to-End Weak Supervision Model, with following hyperparameters:")
        print('\t' + ', '.join([f'{k.upper()}: {i}' for k, i in hyper_params.items()]))
        if model is None:
            self.model = E2E(self.encoder_params, Xdummy=next(iter(train_loader))[1],
                             endmodel=self.endmodel, encoder_net=self.encoder_net).to(device)
        else:
            self.model = model

        # Instantiate the two optimizers for the encoder and downstream network
        if use_wandb:
            wandb.watch(self.model)
        self.total_epochs = hyper_params["epochs"]
        adjust_thresh = hyper_params['adjust_thresh']
        self.trainloader = train_loader

        optim_name = hyper_params['optim']  # 'adam'
        lr, wd1, wd2 = hyper_params["lr"], hyper_params["weight_decay"], hyper_params['mlp_weight_decay']
        encoder_optimizer = get_optimizer(optim_name, self.model.encoder, lr=lr, weight_decay=wd1)
        end_optimizer = get_optimizer(optim_name, self.model.downstream_model, lr=lr, weight_decay=wd2)
        # Reduce learning rate, when the validation performance plateaus
        scheduler_name = hyper_params['scheduler']  # 'None', 'LRoN'
        self.scheduler1 = get_scheduler(scheduler_name, self.model.encoder, encoder_optimizer,
                                        total_epochs=self.total_epochs)
        self.scheduler2 = get_scheduler(scheduler_name, self.model.downstream_model, end_optimizer,
                                        total_epochs=self.total_epochs)

        loss_endmodel = get_loss(hyper_params['loss'], with_logits=True, logits_len=self.model.encoder.logits_len())
        loss_enc = get_loss(hyper_params['EncoderLoss'], with_logits=True, logits_len=self.model.encoder.logits_len())

        # Cycle through epochs
        test_stats = val_stats = None
        best_valid_val = cur_valid_val = -10  # save model with best validation F1 performance
        val_metric = hyper_params['val_metric'].lower()
        is_single_loss = hyper_params['loss'] == hyper_params['EncoderLoss'] and is_symmetric_loss(hyper_params['loss'])
        # Use tqdm for a time/loss/val. performance bar while training
        self.num_datapoints_seen = 0
        with self.tqdm(range(1, self.total_epochs + 1)) as t:
            for epoch in t:
                t.set_description('E2E')
                start_t = time.time()
                # Train the networks E2E for one epoch
                (loss, e_loss, f_loss), n_certains = self._train_epoch(
                    train_loader, encoder_optimizer, end_optimizer,
                    loss_endmodel,loss_enc, epoch, single_loss=is_single_loss
                )
                print('DOne')
                logging_dict = {'Train loss e': e_loss, 'Train loss f': f_loss, 'epoch': epoch}
                if writer is not None:
                    writer.add_scalar(f'train/_loss', loss, epoch)

                # Evaluate the current model's validation performance
                if valloader is not None:
                    _, _, val_stats = self.evaluate(valloader, device=device, prefix='Val E2E', verbose=False,
                                                    adjust_thresh=adjust_thresh)
                    log_epoch_vals(writer, val_stats, epoch, dataset='val')  # tensorboard logging
                    cur_valid_val = val_stats[val_metric]
                    logging_dict = {**logging_dict, **{'Val AUC': val_stats['auc'], 'Val F1': val_stats['f1']}}

                self._scheduler_steps(epoch, metric_val=cur_valid_val)

                # Evaluate the current model's test performance, if given
                if testloader is not None:
                    _, _, test_stats = self.evaluate(testloader, device=device, prefix='Test E2E', verbose=False)
                    log_epoch_vals(writer, test_stats, epoch, dataset="test")  # tensorboard logging
                    logging_dict = {
                        **logging_dict,
                        **{'Test AUC': test_stats['auc'], 'Test F1': test_stats['f1'],
                           'Test Acc.': test_stats['accuracy'],
                            'Test Precision': test_stats['precision'], 'Test Recall': test_stats['recall'] }
                    }

                if use_wandb:
                    wandb.log(logging_dict, step=self.num_datapoints_seen)

                # refresh the training bar
                update_tqdm(t, train_loss=loss, n_cer=n_certains, time=time.time() - start_t, val_stats=val_stats,
                            test_stats=test_stats)
                # update the best validation score, in case the new model is the best
                best_valid_val = self._save_best_model(val_stats, best_valid_val, hyper_params, metric_name=val_metric)

        if valloader is None:
            self._save_model(hyper_params)  # save last checkpoint if no validation loader was given
        if writer is not None:
            writer.flush()
            writer.close()
        return best_valid_val

    def _scheduler_steps(self, epoch, metric_val):
        self.scheduler1.step(self.model.encoder, epoch, metric_val)
        self.scheduler2.step(self.model.downstream_model, epoch, metric_val)

    def _save_best_model(self, new_model_stats, old_best, hyperparams, metric_name='auc', optimizer=None):
        r"""
        :param new_model_stats: a metric dictionary containing a key metric_name
        :param old_best: Old best metric_name value
        :param metric_name: A metric, e.g. F1  (note that the code only supports metrics, where higher is better)
        :return: The best new metric, i.e. old_best if it is better than the newer model's performance and vice versa.
        """
        if new_model_stats is None:
            return old_best
        if new_model_stats[metric_name] > old_best:  # save best model (wrt validation data)
            best = new_model_stats[metric_name]
            print(f"Best model so far with validation {metric_name} =", '{:.3f}'.format(best))
            self._save_model(hyperparams, new_model_stats['cutoff'])
        else:
            best = old_best
        return best

    def _save_model(self, hyper_params, cutoff=0.5):
        import platform
        checkpoint_dict = {
            'model': self.model.state_dict(),
            'cutoff': cutoff,
            # 'optimizer': optimizer.state_dict(),
            'metadata': {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'directory': str(Path('.').resolve()),
                'hostname': platform.node(),
                'hyper_params': hyper_params,
                'model_params': self.encoder_params
            }
        }
        # In case a model dir was given --> save best model (wrt validation data)
        if self.model_dir is not None:
            torch.save(checkpoint_dict, f'{self.model_dir}best_model.pkl')

    def evaluate(self, dataloader, device="cuda", prefix='E2E', verbose=True, use_best_model=False,
                 adjust_thresh=False, cutoff=None):
        r"""
        This function evaluates the downstream model based on gold labels, in case they are known.
        :param dataloader: A torch DataLoader instance, that returns batches (X, Y), where
                            Y (n,) are true labels corresponding to the features X (n, d)
        :param device: Where to compute, e.g. 'cuda', 'cpu'
        :param prefix: A prefix to be printed with the metrics, if verbose is True
        :param verbose: Whether to print the metrics or not
        :param use_best_model: If false, the model at the state the function is called is used for prediction
                               if true, the saved model is reloaded and used for prediction (only possible if models
                                  are being saved, i.e. model_dir is not None).
        :param adjust_thresh: Whether to fine-tune the downstream model's decision threshold (which by default is 0.5)
        :return: A tuple (Y_endmodel, Y_true, stats_dict), where
                Y_endmodel are the final predictions of the downstream model
                Y_true are the corresponding gold labels provided by the dataloader, and
                stats_dict is a dictionary of metric_name --> metric_value pairs, including
                'accuracy', 'f1', 'auc', 'recall', 'precision'
        """
        if use_best_model:
            if self.model_dir is not None:
                model = E2E(self.encoder_params, Xdummy=next(iter(dataloader))[0], verbose=False,
                            endmodel=self.endmodel, encoder_net=self.encoder_net).to(device)
                saved_model = torch.load(f'{self.model_dir}best_model.pkl')
                model.load_state_dict(state_dict=saved_model['model'])
                cutoff, adjust_thresh = saved_model['cutoff'], False
            else:
                print('No models were saved, using the current one.')
                model = self.model
        else:
            model = self.model
        preds, Y, stats = model.evaluate(dataloader, device, prefix=prefix, verbose=verbose,
                                         adjust_thresh=adjust_thresh, cutoff=cutoff)
        return preds, Y, stats
