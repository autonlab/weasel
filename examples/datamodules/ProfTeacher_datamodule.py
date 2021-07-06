import os
from typing import Union, Tuple

from weasel.datamodules.dataset_classes import BasicWeaselDataset, BasicDownstreamDataset
from weasel.datamodules.base_datamodule import AbstractWeaselDataModule
import numpy as np

from weasel.utils.utils import get_filepath


class ProfTeacher_DataModule(AbstractWeaselDataModule):
    def __init__(
            self,
            val_test_split: Union[Tuple[float, float], Tuple[int, int]] = (250, -1),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            seed: int = 3,
            *args,
            **kwargs
    ):
        """
        Args:
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
        filepath = get_filepath('data/professor_teacher_99LFs.npz', parent_dir='examples')
        data = np.load(filepath)

        label_matrix = data['L']  # weak source votes
        Xtrain = data['Xtrain']  # features for training on soft labels

        Xtest = data['Xtest']  # features for evaluating the model
        Ytest = data['Ytest']  # gold labels for evaluating the model

        self.ws_train_set = BasicWeaselDataset(label_matrix, X=Xtrain)  # Multi-source Weak Supervision training set
        self.ws_test_set = BasicDownstreamDataset(Xtest, Ytest)  # Normal test set

    def get_train_data(self) -> BasicWeaselDataset:
        return self.ws_train_set

    def get_test_data(self) -> BasicDownstreamDataset:
        return self.ws_test_set
