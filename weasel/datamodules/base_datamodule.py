import logging
from typing import Optional, Tuple, Callable, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule

from weasel.datamodules.dataset_classes import BasicWeaselDataset, BasicDownstreamDataset, \
    AbstractDownstreamDataset, AbstractWeaselDataset

log = logging.getLogger(__name__)


class AbstractWeaselDataModule(LightningDataModule):
    """
    Abstract LightningDataModule for End-to-End Weak Supervision data.

    --> YOUR TASK:
    For your own data and application you will have to at least override:
        - get_train_data(), to return a torch Dataset that contains (L, X) batches for training of Weasel
        - get_test_data(), to return a torch Dataset that contains (X, Y) batches for evaluation of your end-model
                            and to return None if you do not want to test your model.

    You may optionally override
        - get_val_data(), to return a torch Dataset that contains (X, Y) batches for validation.


    L - the label matrix, a (#data_points, #labeling_functions) matrix
    X - the input features of your end-model. Usually a tensor, but can be anything.
    Y - Ground truth labels of shape (#data_points, #classes)

    ----------------------------------------------------------------------------------------------------------
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            val_test_split: Union[Tuple[float, float], Tuple[int, int], Tuple[float, int]] = (0.1, 0.9),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            seed: int = 3,
            *args,
            **kwargs,
    ):
        """
        Args:
            val_test_split (Tuple): Defines how to split the test/evaluation set into test and validation sets.
                        Only used if get_val_data() returns None, which is the default, and get_test_data() is not None.
                        Floats in the tuple define the fraction to take for the validation & test sets, while
                        integers directly define how many data points each set will contain.
                        If the value corresponding to the test split is -1 (or negative), the test set will
                        automatically consist of all evaluation data points not used for the validation set.
                        If you do not want to use a validation set you may set this arg to (0, -1).
                        Default: 10% of get_test_data() samples are randomly split apart for validation, while
                            the rest 90% remain for the held-out testing.
            batch_size (int): Batch size for the Dataloaders
            num_workers (int): Dataloader arg for higher efficiency
            pin_memory (bool): Dataloaders arg for higher efficiency
            seed (int): Used to seed the validation-test set split, such that the split will always be the same.
        """
        super().__init__()
        self.val_test_split = val_test_split

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.seed = seed

        self._data_train: Optional[AbstractWeaselDataset] = None
        self._data_val: Optional[AbstractDownstreamDataset] = None
        self._data_test: Optional[AbstractDownstreamDataset] = None

    def get_train_data(self, *args, **kwargs) -> AbstractWeaselDataset:
        r"""
        Returns a subclass of AbstractWeaselDataset (which is a torch.Dataset) to be used for training Weasel.
        That is, it is expected to return tuples (lf_votes, x),
         where lf_votes \in {-1, 0, .., C-1}^m are the m outputs of the LFs for this data point with features, x.

         m - Number of labeling functions (LF)
         C - Number of classes
         Note: -1 means that the LF abstained from labeling x
        """
        raise NotImplementedError('Please override WeaselDataModule.get_train_data() to return the (L, X) train data.')

    def get_test_data(self, *args, **kwargs) -> AbstractDownstreamDataset:
        r"""
        Returns a subclass of AbstractDownstreamDataset (a subclass of torch.Dataset) to be used for evaluating Weasel/your end-model.
        That is, it is expected to return tuples (x, y),
         where x are arbitrary features of a data point with ground truth label y \in {0, .., C-1\}.

         C - Number of classes
        """
        raise NotImplementedError('Please override WeaselDataModule.get_test_data() to return the (X, Y) test data.')

    def get_val_data(self, *args, **kwargs) -> Optional[AbstractDownstreamDataset]:
        r"""
        Optional function that may or may not be overridden.
            If not overridden, a validation set will be split from the set returned by get_test_data() based on the
                val-test split defined by self.val_test_split.
            If overridden, it should return a subclass of AbstractDownstreamDataset (which is a torch.Dataset)
                to be used for validating/early-stopping/tuning Weasel.
                Exact same interface like the test set, i.e. is expected to return tuples (x, y),
                where x are the input features of a data point with ground truth labels y \in {0, .., C-1\}.

            C - Number of classes
        """
        return None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        # if stage in (None, 'fit'):
        self._data_train = self.get_train_data()

        testset = self.get_test_data()
        valset = self.get_val_data()
        if valset is None and testset is not None and self.val_test_split[0] > 0:
            evaluation_set_size = len(testset)
            if isinstance(self.val_test_split[0], float):
                val_size = int(self.val_test_split[0] * evaluation_set_size)
            elif isinstance(self.val_test_split[0], int):
                val_size = self.val_test_split[0]
            else:
                raise ValueError("Validation split must be float or int.")

            if isinstance(self.val_test_split[1], float) and self.val_test_split[1] >= 0:
                test_size = int(self.val_test_split[1] * evaluation_set_size)
                test_size += evaluation_set_size - val_size - test_size
            elif isinstance(self.val_test_split[1], int) and self.val_test_split[1] >= 0:
                test_size = self.val_test_split[1]
            else:
                test_size = evaluation_set_size - val_size

            valset, testset = random_split(testset, [val_size, test_size], torch.Generator().manual_seed(self.seed))
        if stage == 'fit':
            val_sz, test_sz = len(valset) if valset else 0, len(testset) if testset else 0
            log.info(f"Data split sizes for training, validation, testing: {len(self._data_train)}, {val_sz}, {test_sz}")

        self._data_val, self._data_test = valset, testset

        # if stage in (None, 'test'):

    def train_dataloader(self):
        return DataLoader(
            dataset=self._data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self._data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self._data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


class BasicWeaselDataModule(AbstractWeaselDataModule):
    """
    A convenient class to easily create the correct LightningDataModule for training Weasel based on
        - label_matrix, L
        - training features, X_train
        - test features, X_test
        - ground truth labels Y_test

    Alternatively, you may override AbstractWeaselDataModule yourself as in the ProfTeacher_DataModule.

    Optionally: Pass X_validation, Y_validation for creating the validation set, otherwise a validation set will be
        split from the test set, based on val_test_split.

    Note: Data features must be Tensor-like, if your end-model requires more complex inputs, e.g. multiple tensors,
        you will have to override the appropriate abstract Dataset and DataModule classes.
    """

    def __init__(
            self,
            label_matrix: Union[np.ndarray, torch.Tensor],
            X_train: Union[np.ndarray, torch.Tensor],
            X_test: Union[np.ndarray, torch.Tensor],
            Y_test: Union[np.ndarray, torch.Tensor],
            X_validation: Optional[Union[np.ndarray, torch.Tensor]] = None,
            Y_validation: Optional[Union[np.ndarray, torch.Tensor]] = None,
            val_test_split: Union[Tuple[float, float], Tuple[int, int], Tuple[float, int]] = (0.1, -1),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            seed: int = 3,
            *args,
            **kwargs,
    ):
        """
        Args:
            label_matrix
            X_train: training features
            X_test: test features
            Y_test: ground truth labels
            val_test_split (Tuple): Defines how to split the test/evaluation set into test and validation sets.
                                Only used if get_val_data() returns None, which is the default.
                                Floats in the tuple define the fraction to take for the validation & test sets, while
                                integers directly define how many data points each set will contain.
                                If the value corresponding to the test split is -1 (or negative), the test set will
                                automatically consist of all evaluation data points not used for the validation set.
            batch_size (int): Batch size for the Dataloaders
            num_workers (int): Dataloader arg for higher efficiency
            pin_memory (bool): Dataloaders arg for higher efficiency
            seed (int): Used to seed the validation-test set split, such that the split will always be the same.
        """
        super().__init__(val_test_split, batch_size, num_workers, pin_memory, seed)
        self._train_set = BasicWeaselDataset(L=label_matrix, X=X_train)
        self._test_set = BasicDownstreamDataset(X=X_test, Y=Y_test, *args, **kwargs)
        if X_validation is not None and Y_validation is not None:
            self._val_set = BasicDownstreamDataset(X=X_validation, Y=Y_validation, *args, **kwargs)
        else:
            self._val_set = None

    def get_train_data(self) -> BasicWeaselDataset:
        return self._train_set

    def get_test_data(self) -> BasicDownstreamDataset:
        return self._test_set

    def get_val_data(self) -> BasicDownstreamDataset:
        return self._val_set
