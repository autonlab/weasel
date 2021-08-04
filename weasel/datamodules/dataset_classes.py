from typing import Union, Any, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader


class AbstractWeaselDataset(torch.utils.data.Dataset):
    """
    An abstract torch dataset whose subclasses/implementations (see e.g. BasicWeaselDataset) can
    be used to train Weasel (by creating a torch DataLoader or pl.LightningDataModule).
    """

    def __init__(self,
                 L: Union[np.ndarray, torch.Tensor],
                 X: Any):
        """
        Args:
            L (np.ndarray, torch.Tensor): Label matrix of shape (n, m)
                        With C classes, LF labels should have values in {-1, 0, 1, ..., C-1}, where -1 means abstains.
            X (Any): Whatever input/features your end-model takes as input, usually just a Tensor (see BasicWeaselDataset).

        n -- Number of (training) samples
        m -- Number of labeling heuristics
        C -- Number of classes
        """
        if isinstance(L, np.ndarray):
            L = torch.from_numpy(L)
        if torch.is_tensor(L):
            if len(L.shape) != 2:
                raise ValueError(f"Label matrix must have shape (n, m) = (#samples, #LFs), but {L.shape} was found!")
            self.L = L.float()
        else:
            raise ValueError(f"Label matrix arg 'L' must be a torch tensor or numpy ndarray, but {type(L)} was found!")

    def __getitem__(self, item) -> Tuple[torch.Tensor, Any]:
        """
        Args:
         item: Key for fetching a data example, e.g. the i-th sample.

        Returns:
            The corresponding (L, X) example to 'item'.
        """
        raise NotImplementedError("Please return here (L, X) pairs, where L is the label matrix"
                                  " and X arbitrary features that are passed to your end-model")

    def __len__(self) -> int:
        return self.L.shape[0]


class BasicWeaselDataset(AbstractWeaselDataset):
    """
    A basic torch dataset that be used to train Weasel (by creating a torch DataLoader or pl.LightningDataModule),
    provided that the end-model takes a torch tensor as input (which applies to most models).
    """

    def __init__(self,
                 L: Union[np.ndarray, torch.Tensor],
                 X: Union[np.ndarray, torch.Tensor]):
        """
        Args:
            L (np.ndarray, torch.Tensor): Label matrix of shape (n, m)
                        With C classes, LF labels should have values in {-1, 0, 1, ..., C-1}, where -1 means abstains.
            X (np.ndarray, torch.Tensor): Tensor input features for your end-model.

        n -- Number of (training) samples
        m -- Number of labeling heuristics
        C -- Number of classes
        """
        super().__init__(L, X)
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if torch.is_tensor(X):
            self.X = X.float()
        else:
            raise ValueError(f"Features X must be a torch tensor or numpy ndarray for the simple 'BasicWeasel_dataset',"
                             f" but {type(X)} was found! If your end-model inputs are not plain tensors, you'll have to"
                             f" implement your own subclass of 'Abstract_Weasel_dataset'.")

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.L[item, :], self.X[item]


# --------------------------------------------------------------------------
# --------------------------------- Datasets for the end-model
class AbstractDownstreamDataset(torch.utils.data.Dataset):
    """
    This an abstract dataset creator to be used for the datasets that
        1) evaluate/test the end/downstream model (within Weasel) on gold labels;
     AND/OR training datasets for baselines, where the fixed targets are either
        2a) soft/probabilistic labels generated by 2-step baseline WS approaches like Snorkel, or
        2b) hard (ground truth) labels (e.g. for training on a validation set or majority vote).
    """

    def __init__(self,
                 X: Union[np.ndarray, torch.Tensor],
                 Y: Union[np.ndarray, torch.Tensor]
                 ):
        r"""
        Args:
            X (np.ndarray, torch.Tensor): Input feature array/tensor for forward pass through your end-model
            Y (np.ndarray, torch.Tensor): Target labels \in {0, .., C-1}^N

        N -- Number of dataset samples
        C -- Number of classes
        """
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y)
        if torch.is_tensor(Y):
            self.Y = Y.float()
        else:
            raise ValueError(f"Labels Y must be a torch tensor or numpy ndarray for the simple 'BasicWeasel_dataset',"
                             f" but {type(Y)} was found!")

    def __getitem__(self, item) -> Tuple[Any, torch.Tensor]:
        """
        Args:
         item: Key for fetching a data example, e.g. the i-th sample.

       Returns:
        The corresponding (X, Y) example to 'item'.
            """
        raise NotImplementedError("Please return here (X, Y) pairs, where X are your end-model's inputs"
                                  " and Y the target labels.")

    def __len__(self) -> int:
        return self.Y.shape[0]


class BasicDownstreamDataset(AbstractDownstreamDataset):
    """
    This a basic -- but most likely all that you need -- dataset creator to be used for the datasets that
        1) evaluate/test the end/downstream model (within Weasel) on gold labels;
     AND/OR training datasets for baselines, where the fixed targets are either
        2a) soft/probabilistic labels generated by 2-step baseline WS approaches like Snorkel, or
        2b) hard (ground truth) labels (e.g. for training on a validation set or majority vote).
    """

    def __init__(self,
                 X: Union[np.ndarray, torch.Tensor],
                 Y: Union[np.ndarray, torch.Tensor],
                 filter_uncertains: bool = False):
        r"""
        Args:
            X: (N, ..) array of features (either numpy or already a torch tensor)
            Y: (N, C) array -- labels to be treated as targets for the downstream model
            filter_uncertains (bool): Only important for case 2a), i.e. when the labels Y are soft labels generated by
                        a weak supervision framework and you will want to filter out those samples all LFs abstained.
                            (i.e. P(y = c| x) = 1/num_classes for all classes c).
                        This flag is recommended to be set to true for this case,
                          otherwise the downstream model will just be fed noise.

        N -- Number of dataset samples
        C -- Number of classes
        """
        super().__init__(X, Y)
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if torch.is_tensor(X):
            self.X = X.float()
        else:
            raise ValueError(f"Features X must be a torch tensor or numpy ndarray for the simple 'BasicWeasel_dataset',"
                             f" but {type(X)} was found! If your end-model inputs are not plain tensors, you'll have to"
                             f" implement your own subclass of 'Abstract_Weasel_dataset'.")

        self.Y = self.Y.clone()
        if filter_uncertains:
            total_samples = self.__len__()
            certains = torch.any(self.Y != 1 / len(Y.shape), dim=1)

            self.Y, self.X = self.Y[certains], self.X[certains]
            print(f"Eliminated noisy samples from BasicDownstreamDataset, {total_samples - self.__len__()} removed.")

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[item], self.Y[item]

