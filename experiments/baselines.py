import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from downstream_models.LSTM import LSTM_Trainer
from downstream_models.MLP import MLP_Trainer, LabelMeMLP_Trainer
from utils.prediction_and_evaluation import get_majority_vote
from utils.utils import change_labels, set_seed
from utils.data_loading import DownstreamDP, split_dataset, DownstreamSpouses


def train_snorkel(label_matrix, X, hyper_params, dataset, Ytrain=None, end_valloader=None, end_testloader=None,
                  n_epoch=100, lr=0.01, num_valid_samples=250, wandb_run=None,
                  mlp_params=None, model_dir=None, class_balance=(0.5, 0.5), cardinality=2):
    from snorkel.labeling.model import LabelModel
    # LABEL MODEL
    seed = hyper_params['seed']
    print("-+-" * 20, f"SNORKEL START seed = {seed}", "-+-" * 20)

    # Train latent label model from Snorkel
    def get_preds(preds):
        return preds if cardinality == 2 else preds

    def score_lm(L, Y):
        return roc_auc_score(Y, lm.predict_proba(L)[:, 1]) if cardinality == 2 else accuracy_score(Y, lm.predict(L))

    if Ytrain is not None:
        set_seed(3)  # to always get the same valid-test split
        (L_dev, Y_dev), (L_train, _) = split_dataset(num_valid_samples, None, label_matrix, Ytrain,
                                                     filter_uncertains=False)
        set_seed(seed)  # reset the seed

        best_val_snorkel, best_stats_snorkel, best_probs = -1, None, None
        for epochs, lr, reg in [(100, 0.01, 0.0), (200, 0.05, 0.0), (2000, 0.0003, 0.0),
                                (500, 0.003, 0.0), (1000, 0.003, 0.1), (5000, 0.01, 0.0)]:
            lm = LabelModel(cardinality=cardinality)
            try:
                lm.fit(L_train, seed=seed, n_epochs=epochs, lr=lr, class_balance=class_balance, l2=reg)
            except Exception as e:
                print(e, 'Snorkel failed for this config. Skipping it...')
                continue
            cur_val = score_lm(L_dev, Y_dev)
            if cur_val > best_val_snorkel:  # search for best generative label model
                best_val_snorkel = cur_val
                best_probs = get_preds(lm.predict_proba(label_matrix))
        print('Best generative Snorkel performance on dev set2 = {:.3f}'.format(cur_val))
    else:
        lm = LabelModel(cardinality=cardinality)
        lm.fit(label_matrix, seed=seed, n_epochs=n_epoch, lr=lr, class_balance=class_balance)
        best_probs = get_preds(lm.predict_proba(label_matrix))

    stats = train_downstream(mlp_params, X, best_probs, end_valloader, end_testloader, seed, model_dir, hyper_params,
                             prefix='Snorkel', dataset=dataset, wandb_run=wandb_run)

    print("-+-" * 20, "SNORKEL END", "-+-" * 20)
    return lm, best_probs, stats


def train_triplets(label_matrix, X, hyper_params, dataset, end_valloader=None, end_testloader=None, method="mean",
                   mlp_params=None, model_dir=None, class_balance=None, wandb_run=None):
    from flyingsquid.label_model import LabelModel
    if class_balance is None:
        class_balance = [0.5, 0.5]
    # triplets has 0 = abstain, while we use in the binary case -1
    label_matrix = change_labels(label_matrix.copy(), new_label=-1, old_label=0)
    seed = hyper_params['seed']
    print("-+-" * 20, f"TRIPLETS {method} START seed = {seed}", "-+-" * 20)

    # Train latent label model from triplets
    solve_with = "triplet" if method == '' else 'triplet_' + method
    lm = LabelModel(label_matrix.shape[1], triplet_seed=seed)
    lm.fit(label_matrix, solve_method=solve_with, class_balance=np.array(class_balance))
    # Evaluate the learned generative model
    probs = lm.predict_proba(label_matrix)  # [:, 1]
    stats = train_downstream(mlp_params, X, probs, end_valloader, end_testloader, seed, model_dir, hyper_params,
                             prefix=solve_with, dataset=dataset, wandb_run=wandb_run)
    return lm, probs, stats




def majority_vote(label_matrix, dataset, X=None, mlp_params=None, hyper_params=None, end_valloader=None,
                  end_testloader=None, model_dir=None, wandb_run=None, **kwargs):
    """
    Only use for binary classification!
    """
    label_matrix = change_labels(label_matrix.copy().astype(np.int), new_label=-1, old_label=0)  # MV has 0=abstain
    preds_MMVV = get_majority_vote(label_matrix, probs=True, abstention_policy='Stay', abstention=0)
    hard_preds = preds_MMVV  # [:, 1]
    hard_preds[hard_preds[:, 0] != 0.5, :] = np.round(hard_preds[hard_preds[:, 0] != 0.5, :])
    seed = hyper_params['seed']
    stats = train_downstream(mlp_params, X, hard_preds, end_valloader, end_testloader, seed, model_dir,
                             hyper_params, prefix='MV', dataset=dataset, wandb_run=wandb_run)
    return stats


def train_downstream(mlp_params, features, soft_labels, valloader, testloader, seed, model_dir, hyper_params, dataset,
                     prefix='', wandb_run=None):
    import wandb
    downstream_dataset, trainer = get_dataset_and_trainer(dataset, mlp_params)  # DownstreamDP
    train_data = downstream_dataset(Y=soft_labels, features=features, filter_uncertains=True, uncertain=0.5)
    # Train MLP
    print("Models are saved at:", model_dir)
    trainer = trainer(mlp_params, seed=seed, model_dir=model_dir, notebook_mode=hyper_params['notebook_mode'])
    best_valid = trainer.fit(train_data, hyper_params, valloader=valloader, testloader=testloader, device="cuda")
    # Test MLP
    _, _, end_stats = trainer.evaluate(testloader, device="cuda", print_prefix=f'{prefix} MLP END:\n',
                                           use_best_model=False)
    _, _, best_stats = trainer.evaluate(testloader, device="cuda", print_prefix=f'{prefix} MLP BEST:\n',
                                            use_best_model=True)
    if wandb_run is not None:
        wandb.log({'Final Test F1': best_stats['f1'], 'Final Test AUC': best_stats['auc'],
                   'Final Test Prec.': best_stats['precision'], 'Final Test Rec.': best_stats['recall'],
                   f"Best Val {hyper_params['val_metric'].upper()}": best_valid}
                  )
        wandb.run.summary["f1"] = best_stats['f1']
        wandb_run.finish()
    best_stats['validation_val'] = best_valid
    stats = {"MLP_best": best_stats, 'MLP_end': end_stats}
    return stats


def train_supervised(train_data, mlp_params, hyper_params, dataset, model_dir=None, valloader=None, testloader=None,
                     prefix='Superv.', trainloader=None):
    # Data handling
    seed = hyper_params['seed']
    _, trainer = get_dataset_and_trainer(dataset, mlp_params)  # DownstreamDP
    print("~+~" * 15, f"SUPERVISED START WITH GROUND TRUTH LABELS seed = {seed}", "~+~" * 15)
    if train_data is not None:
        print('With', len(train_data), 'labels.')
    trainer = trainer(mlp_params, seed=seed, model_dir=model_dir, notebook_mode=hyper_params['notebook_mode'])
    best_valid = trainer.fit(train_data, hyper_params, valloader=valloader, testloader=testloader, device="cuda",
                             trainloader=trainloader)
    # Test MLP
    _, _, end_stats = trainer.evaluate(testloader, device="cuda", print_prefix=f'{prefix} MLP END:\n')
    _, _, mlp_stats = trainer.evaluate(testloader, device="cuda", print_prefix=f'{prefix} MLP BEST:\n',
                                       use_best_model=True)
    mlp_stats['validation_val'] = best_valid
    stats = {"MLP_best": mlp_stats, 'MLP_end': end_stats}
    print("~+~" * 20, "SUPERVISED END", "~+~" * 20)
    return stats


def get_dataset_and_trainer(dataset, params=None):
    dataset = dataset.lower()
    if dataset == 'spouses':
        return DownstreamSpouses, LSTM_Trainer
    elif dataset == 'labelme':
        return DownstreamDP, LabelMeMLP_Trainer
    else:
        return DownstreamDP, MLP_Trainer
