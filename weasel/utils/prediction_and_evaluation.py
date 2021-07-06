from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve
from weasel.utils.utils import change_labels


def eval_final_predictions(Y, preds, probs, only_on_labeled=False, model_name="",
                           verbose=False, add_prefix="", parent_stats: Optional[dict] = None, is_binary=None):
    predsC = preds.copy()
    Yc = Y.copy()
    probsC = predsC if probs is None else probs.copy()
    is_binary = is_binary or len(probs.shape) <= 1 or set(np.unique(Y)) == {0, 1}

    if is_binary:
        stats = eval_binary(Yc, predsC, probsC, only_on_labeled, model_name, verbose)
    else:
        stats = eval_multiclass(Yc, predsC, probsC, only_on_labeled, model_name, verbose)

    stats = {f"{add_prefix}/{key}".lstrip("/"): value for key, value in stats.items()}
    if parent_stats is None:
        return stats
    else:
        return {**parent_stats, **stats}


def eval_binary(Y, preds, probs=None, only_on_labeled=False, model_name="", compute_confusion_matrix=False,
                verbose=False):
    if len(probs.shape) == 2:
        probs = probs[:, 1]
    neg_label = 0
    abstention = 0.5
    labeled = (preds != abstention)
    if only_on_labeled:
        print(f"Evaluating on {np.count_nonzero(labeled)} samples only.")
        preds = preds[labeled]
        Y = Y[labeled]
        probs = probs[labeled]
    acc = accuracy_score(Y, preds)
    recall = recall_score(Y, preds)
    precision = precision_score(Y, preds)
    f1 = f1_score(Y, preds)
    auc = roc_auc_score(Y, probs)
    # coverage = np.count_nonzero(labeled) / Y.shape[0]
    # MSE = mean_squared_error(Y, probs)
    # MAE = mean_absolute_error(Y, probs)
    stats = {'accuracy': acc,  'recall': recall, 'precision': precision,
             'f1': f1,
             'auc': auc,
             } #"MSE": MSE, "MAE": MAE,'coverage': coverage}
    if compute_confusion_matrix:
        t = preds == Y
        tp = np.count_nonzero(np.logical_and(t, Y == 1))
        tn = np.count_nonzero(np.logical_and(t, Y == neg_label))
        fp = np.count_nonzero(np.logical_and(np.logical_not(t), preds == 1))
        fn = np.count_nonzero(np.logical_and(np.logical_not(t), preds == neg_label))
        stats = {**stats, "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    if verbose:
        print(model_name, 'Accuracy:{:.3f} | Precision:{:.3f} | Recall:{:.3f} | F1 score:{:.3f} | AUC:{:.3f}'
              .format(acc, precision, recall, f1, auc))
    return stats

def to_categorical(hard_preds, num_classes):
    if num_classes is None:
        num_classes = len(np.unique(hard_preds))
    return np.eye(num_classes)[hard_preds.astype(np.int)]

def eval_multiclass(Y, preds, probs=None, only_on_labeled=False, model_name="", verbose=False):
    n_samples, num_classes = probs.shape
    labeled = np.any(probs != 1 / num_classes, axis=1)
    if only_on_labeled:
        print(f"Evaluating on {np.count_nonzero(labeled)} samples only.")
        preds = preds[labeled]
        Y = Y[labeled]
        probs = probs[labeled, :]
    acc = accuracy_score(Y, preds)
    f1_micro = f1_score(Y, preds, average='micro')
    f1_macro = f1_score(Y, preds, average='macro')
    Y_cat = to_categorical(Y, num_classes)
    brier = np.sum((Y_cat - probs)**2)/(2*n_samples)
    # auc = roc_auc_score(Y, probs)
    auc = -1
    coverage = np.count_nonzero(labeled) / Y.shape[0]
    stats = {'accuracy': acc, 'f1_micro': f1_micro, "f1_macro": f1_macro, "brier": brier} #, 'auc': auc, 'coverage': coverage}
    if verbose:
        print(model_name, 'Accuracy:{:.3f} | F1-micro score:{:.3f} | F1-macro score:{:.3f} | Brier: {:.3f} | AUC:{:.3f}'
                          ' | Coverage:{:.3f}'
              .format(acc, f1_micro, f1_macro, brier, auc, coverage))
    return stats


def get_majority_vote(label_matrix, probs=False, n_classes=2, abstention_policy='drop', abstention=-1, metal=False):
    def majority_vote(row, abst=0):
        tmp = np.zeros(n_classes)
        if metal:
            for i in row:
                tmp[i - 1] += 1 if i != abst else 0
        else:
            for i in row:
                tmp[i] += 1 if i != abst else 0

        if not tmp.any():
            if probs:
                return np.ones(n_classes) / n_classes  # return uniform probs
            else:
                return abstention
        elif probs:
            res = np.zeros(n_classes)
            res[np.argmax(tmp)] = 1
            return res
            return (tmp / len(row)).reshape(1, -1)
        else:
            pred = np.argmax(tmp)
            if abstention == 0 and pred == 0:
                pred = -1
            return pred

    '''
    Equivalent to:
    maj_voter = MajorityLabelVoter()
    majority_preds = maj_voter.predict_proba(label_matrix)
    majority_preds = np.argmax(majority_preds[idxs], axis=1)
    accMV, precMV, recMV, f1MV = iws.utils.eval_final_predictions(majority_preds, Ytrain)
    '''
    label_mat = label_matrix.copy()
    if abstention == 0:  # better map 0 -> -1 and vice versa probably
        label_mat = change_labels(label_mat, old_label=0, new_label=-1)
        votes = np.apply_along_axis(majority_vote, 1, label_mat, -1)
    else:
        votes = np.apply_along_axis(majority_vote, 1, label_mat, abstention)
    if probs:
        votes = votes.reshape(votes.shape[0], n_classes)

    if abstention_policy == 'drop':
        votes = votes[votes != abstention]
    elif abstention_policy == 'random' and not probs:
        votes[votes == abstention] = np.random.choice([-1, 1], np.count_nonzero(votes == abstention))
    return votes


def lf_coverages(L, abstention=0):
    return np.ravel((L != abstention).sum(axis=0)) / L.shape[0]


def get_optimal_decision_threshold(preds, true_labels):
    """
    :param preds: Model predictions
    :param true_labels: Ground truth labels
    :return: The optimal decision threshold w.r.t to having the highest F1 score on the predictions
    """
    if len(preds.shape) == 2:
        preds = preds[:, 1]
    precision, recall, threshs = precision_recall_curve(true_labels, preds)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-6)  # +1e-6 to avoid NaNs
    best_thresh = threshs[np.argmax(f1_scores)]
    return best_thresh  # print('Best F1-Score: ', np.max(f1_scores))
