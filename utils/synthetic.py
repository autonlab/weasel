import numpy as np


def synthetic_ablation(labels, num_LFs=11, abstain_too=False):
    n_samples = labels.shape[0]
    L = np.empty((n_samples, num_LFs))
    L[:, 0] = labels.copy()
    classes = list(np.unique(labels))

    random_vote = random_LF(n_samples, classes, abstain_too=abstain_too)
    for i in range(1, num_LFs):
        L[:, i] = random_vote.copy()
        # L[:, i] = np.random.choice(classes, size=n_samples)
    L = L[:, np.random.permutation(num_LFs)]   # just to shuffle the LF columns
    return L


def random_LF(n_samples: int, classes: list, abstain_too=False):
    if abstain_too:
        classes = classes + [-1]
    random_vote = np.random.choice(classes, size=n_samples)
    return random_vote
