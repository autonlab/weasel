import itertools
import os
import pickle
import time
from typing import Tuple, Callable, Optional, Any

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils.prediction_and_evaluation import to_categorical
from utils.utils import change_labels


class DP(torch.utils.data.Dataset):
    """
    Torch Dataset that expects a label matrix (N, m) and features (N, d), and can be used to create a torch DataLoader.
    If labels are binary the expected mapping is:
        -1 --> Abstain
        0 --> Negative
        +1 --> Positive
    If labels are multi-class with C classes, the values of L should be in {0, 1, ..., C}, where 0 means abstains.
    """

    def __init__(self, L, features, feats_to_torch=True, binary=False):
        if binary or len(np.unique(L)) == 3:
            print('Binary classification: internally mapping abstains to 0, negative label to -1.')
            L = change_labels(L.copy(), old_label=0, new_label=-1)  # we want abstain to be 0, negative label = -1
        self.L = torch.from_numpy(L).float()
        self.X = torch.from_numpy(features).float() if feats_to_torch else features

    def __getitem__(self, item):
        return self.L[item, :], self.X[item, :]

    def __len__(self):
        return self.L.shape[0]



class SpousesDP(torch.utils.data.Dataset):
    def __init__(self, L, features):
        self.L = torch.from_numpy(L).float()
        self.X = torch.tensor(features[0]).long(), torch.tensor(features[1]).long(), torch.tensor(features[2]).long()

    def __getitem__(self, item):
        return self.L[item, :], (self.X[0][item], self.X[1][item], self.X[2][item]),

    def __len__(self):
        return self.L.shape[0]


def split_dataset(val_size, dataset_class, *args, sequentally=False, **kwargs):
    assert len(args) > 0
    if sequentally:
        indices = np.arange(args[0].shape[0])
    else:
        indices = np.random.permutation(args[0].shape[0])
    val_idx, test_idx = indices[:val_size], indices[val_size:]
    valset = [arg[val_idx] for arg in args]
    testset = [arg[test_idx] for arg in args]
    if dataset_class is not None:
        return dataset_class(*valset, **kwargs), dataset_class(*testset, **kwargs)
    else:
        return valset, testset


class DownstreamDP(torch.utils.data.Dataset):
    """
    This a dataset creator to be used for training an end/downstream model on soft/probabilistic labels
    produced by our weak supervision framework (e.g. by the encoder of the VAE).
    N -- Number of samples
    d -- feature dimensionality
    """

    def __init__(self, features, Y, Y_to_torch=True, X_to_torch=True, filter_uncertains=True, uncertain=0.5,
                 categorical=False,  num_classes=None):
        r"""

        :param Y: (N, ) array -- soft labels to be used as targets by the downstream model
        :param features: (N, d) array of features
        :param Y_to_torch: Whether to use a torch tensor for Y
        :param filter_uncertains: Whether to filter soft labels that are uncertain (i.e. = 0.5), which usually happens
                                   when all LFs abstained on this sample. This bool is recommended to be set to true,
                                   otherwise the downstream model will just be fed noise.
        :param uncertain: Uncertainty value for Y, usually 0.5, for soft labels \in [0, 1]
        """
        self.X = torch.from_numpy(features).float() if X_to_torch else features
        if categorical:
            Y = to_categorical(Y, num_classes=num_classes)
        self.Y = torch.from_numpy(Y).float() if Y_to_torch else Y.copy()
        if filter_uncertains:
            total_samples = self.__len__()
            certains = (self.Y != uncertain) if len(self.Y.size()) <= 1 else torch.any(self.Y != 1 / len(Y.shape), dim=1)

            self.Y, self.X = self.Y[certains], self.X[certains]
            print(f"Eliminated noisy samples, {total_samples - self.__len__()} removed.")

    def __getitem__(self, item):
        return self.X[item, :], self.Y[item]

    def __len__(self):
        return len(self.Y)

class DownstreamSpouses(DownstreamDP):
    def __init__(self, features, Y, filter_uncertains=True, uncertain=0.5, **kwargs):
        super().__init__(features, Y, X_to_torch=False, filter_uncertains=False, **kwargs)
        self.X = torch.tensor(self.X[0]).long(), torch.tensor(self.X[1]).long(), torch.tensor(self.X[2]).long()
        if filter_uncertains:
            total_samples = self.__len__()
            certains = (self.Y != uncertain) if len(self.Y.size()) <= 1 else torch.any(self.Y != 1 / len(Y.shape), dim=1)
            self.Y, self.X = self.Y[certains], (self.X[0][certains], self.X[1][certains], self.X[2][certains])
            print(f"Eliminated noisy samples, {total_samples - self.__len__()} removed.")

    def __getitem__(self, item):
        return (self.X[0][item], self.X[1][item], self.X[2][item]), self.Y[item]


def get_spouses_data(batch_size=64, dataloader=True):
    import collections
    from torchtext.vocab import Vocab
    arrs = np.load("data/spouses/L.npz", allow_pickle=True)
    L_train = arrs['train']
    L_dev = arrs['dev']
    L_train, L_dev = change_labels(L_train, L_dev, old_label=0, new_label=-1)

    with open("data/spouses/dev_data.pkl", "rb") as f:
        df_dev = pickle.load(f)
        Y_dev = pickle.load(f)

    with open("data/spouses/train_data.pkl", "rb") as f:
        df_train = pickle.load(f)

    with open("data/spouses/test_data.pkl", "rb") as f:
        df_test = pickle.load(f)
        Y_test = pickle.load(f)

    # Convert labels to {0, 1} format from {-1, 1} format.
    Y_dev = (1 + Y_dev) // 2
    Y_test = (1 + Y_test) // 2
    Xtrain = get_feature_arrays(df_train)
    Xdev = get_feature_arrays(df_dev)
    Xtest = get_feature_arrays(df_test)

    all_tokens = np.concatenate(
        (Xtrain[0].reshape(-1), Xtrain[1].reshape(-1), Xtrain[2].reshape(-1),
         Xdev[0].reshape(-1), Xdev[1].reshape(-1), Xdev[2].reshape(-1),
         Xtest[0].reshape(-1), Xtest[1].reshape(-1), Xtest[2].reshape(-1),
         ), axis=0)
    counter = collections.Counter(all_tokens)
    vocab = Vocab(counter, max_size=40000, min_freq=20)

    def numericalize(X):
        X = list(X)
        X[0] = [[vocab.stoi[word] for word in row] for row in X[0]]
        X[1] = [[vocab.stoi[word] for word in row] for row in X[1]]
        X[2] = [[vocab.stoi[word] for word in row] for row in X[2]]
        return [np.array(sub_x) for sub_x in X]

    Xtrain, Xdev, Xtest = numericalize(Xtrain), numericalize(Xdev), numericalize(Xtest)

    if dataloader:
        train = SpousesDP(L_train, features=Xtrain)
        val_LSTM = DownstreamSpouses(features=Xdev, Y=Y_dev, filter_uncertains=False)
        test_LSTM = DownstreamSpouses(features=Xtest, Y=Y_test, filter_uncertains=False)
        trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
        valmlp_loader = DataLoader(val_LSTM, batch_size=batch_size, shuffle=False)
        testmlp_loader = DataLoader(test_LSTM, batch_size=batch_size, shuffle=False)
        return trainloader, valmlp_loader, testmlp_loader
    else:
        return L_train, L_dev, Xtrain, Xdev, Xtest, Y_dev, Y_test


def get_feature_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Taken from Snorkel tutorials
        Get np arrays of upto max_length tokens and person idxs."""
    bet = df.between_tokens
    left = df.apply(lambda c: c.tokens[: c.person1_word_idx[0]][-4:-1], axis=1)
    right = df.person2_right_tokens

    def pad_or_truncate(l, max_length=40):
        return l[:max_length] + [""] * (max_length - len(l))

    left_tokens = np.array(list(map(pad_or_truncate, left)))
    bet_tokens = np.array(list(map(pad_or_truncate, bet)))
    right_tokens = np.array(list(map(pad_or_truncate, right)))
    return left_tokens, bet_tokens, right_tokens


def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets

def get_LabelMe_data():
    """
    Load the dataset
    :param train: training or not
    :return:
    """

    def load_data(filename):
        with open(filename, 'rb') as f:
            data = np.load(f)

        f.close()
        return data
    N_CLASSES = 8

    path = 'data/LabelMe/prepared/'

    data_train_vgg16 = load_data(path + "data_train_vgg16.npy").transpose(0, 3, 1, 2)
    print("Training data shape:", data_train_vgg16.shape)

    # print("\nLoading AMT data...")
    answers = load_data(path + "answers.npy")
    label_train = load_data(path + "labels_train.npy")
    print("Crowdsourced label shape:", answers.shape)
    print("label shape:", label_train.shape)
    N_ANNOT = answers.shape[1]

    print("N_ANNOT:", N_ANNOT)

    answers_bin_missings = []

    for i in range(len(answers)):
        row = []
        for r in range(N_ANNOT):
            if answers[i, r] == -1:
                row.append(0 * np.ones(N_CLASSES))
            else:
                # print(answers[i,r])
                row.append(one_hot(answers[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)

    answers_bin_missings = np.array(answers_bin_missings)
    print(answers_bin_missings.shape)
    # label_trainN = one_hot(label_train, N_CLASSES)

    # print(answers_bin_missings[0])
    data_test_vgg16 = load_data(path + "data_test_vgg16.npy").transpose(0, 3, 1, 2)

    labels_test = load_data(path + "labels_test.npy")
    return data_train_vgg16, label_train, data_test_vgg16, labels_test, answers #answers_bin_missings


