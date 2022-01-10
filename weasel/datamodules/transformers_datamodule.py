from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from weasel.datamodules.base_datamodule import AbstractWeaselDataModule
from weasel.datamodules.dataset_classes import (
    AbstractWeaselDataset,
    AbstractDownstreamDataset,
)


class TransformersTrainDataset(AbstractWeaselDataset):
    """Train dataset for the TransformersDataModule.

    Args:
        L: The weak label matrix.
        inputs: A list of output dictionaries of a `transformers.PreTrainedTokenizerBase`.
    """

    def __init__(self, L: Union[np.ndarray, torch.Tensor], inputs: List[Dict]):
        super().__init__(L, None)
        self.inputs = inputs

        if self.L.shape[0] != len(self.inputs):
            raise ValueError("L and inputs have different number of samples")

    def __getitem__(self, item) -> Tuple[torch.Tensor, Dict]:
        return self.L[item], self.inputs[item]


class TransformersTestDataset(AbstractDownstreamDataset):
    """Test/Validation dataset for the TransformersDataModule.

    Args:
        inputs: A list of output dictionaries of a `transformers.PreTrainedTokenizerBase`.
        Y: Target labels \in {0, .., C-1}^N
    """

    def __init__(self, inputs: List[Dict], Y: Union[np.ndarray, torch.Tensor]):
        super().__init__(None, Y)
        self.inputs = inputs

        if len(self.Y) != len(self.inputs):
            raise ValueError("inputs and Y have different number of samples")

    def __getitem__(self, item) -> Tuple[Dict, torch.Tensor]:
        return self.inputs[item], self.Y[item]


class TransformersCollator:
    """Collator for the TransformersDataModule.

    Args:
        tokenizer: The tokenizer instance used to dynamically pad the samples in the batch.
        **kwargs: Passed on to the `tokenizer.pad` method.
    """

    def __init__(
        self,
        tokenizer: "transformers.PreTrainedTokenizerBase",
        **kwargs,
    ):
        self._tokenizer = tokenizer
        self._kwargs = {"return_tensors": "pt"}
        self._kwargs.update(kwargs)

    def train_collate(
        self, batch: List[Tuple[torch.Tensor, Dict]]
    ) -> Tuple[torch.Tensor, Dict]:
        L, inputs = self._collate(
            L_or_Y=[sample[0] for sample in batch],
            inputs=[sample[1] for sample in batch],
        )
        return L, inputs

    def test_collate(
        self, batch: List[Tuple[Dict, torch.Tensor]]
    ) -> Tuple[Dict, torch.Tensor]:
        Y, inputs = self._collate(
            L_or_Y=[sample[1] for sample in batch],
            inputs=[sample[0] for sample in batch],
        )
        return inputs, Y

    def _collate(
        self, L_or_Y: List[torch.Tensor], inputs: List[Dict]
    ) -> Tuple[torch.Tensor, Dict]:
        return torch.stack(L_or_Y), self._tokenizer.pad(inputs, **self._kwargs)


class TransformersDataModule(AbstractWeaselDataModule):
    """LightningDataModule for the Transformers downstream model.

    Args:
        label_matrix: The weak label matrix.
        X_train: Outputs of a `transformers.PreTrainedTokenizerBase`.
        collator: Needed to dynamically pad the input samples.
        X_test: Outputs of a `transformers.PreTrainedTokenizerBase`.
        Y_test: Ground truth labels for the test set.
        X_validation: Outputs of a `transformers.PreTrainedTokenizerBase`.
        Y_validation: Ground truth labels for the validation set.
        **kwargs: Passed on to the init of the parent class.

    Examples:
        Minimal example
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> X_train = [tokenizer(text) for text in ["First example", "Second example"]]
        >>> label_matrix = np.random.randint(-1, 2, (len(X_train), 10))  # mock weak label matrix
        >>> data_module = TransformersDataModule(
        ...     label_matrix = label_matrix,
        ...     X_train = X_train,
        ...     collator=TransformersCollator(tokenizer),
        ... )

        Using Hugging Face's datasets
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> ds = load_dataset("tweet_eval", "sentiment")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> ds_train = ds["train"].map(tokenizer, input_columns="text", remove_columns=ds["train"].column_names)
        >>> ds_test = ds["test"].map(tokenizer, input_columns="text", remove_columns=ds["test"].column_names)
        >>> label_matrix = np.random.randint(-1, 4, (len(ds_train), 10))  # mock weak label matrix
        >>> data_module = TransformersDataModule(
        ...     label_matrix = label_matrix,
        ...     X_train = list(ds_train),
        ...     collator=TransformersCollator(tokenizer),
        ...     X_test = list(ds_test),
        ...     Y_test = np.array(ds["test"]["label"])
        ... )
    """

    def __init__(
        self,
        label_matrix: Union[np.ndarray, torch.Tensor],
        X_train: List[Dict],
        collator: TransformersCollator,
        X_test: Optional[List[Dict]] = None,
        Y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
        X_validation: Optional[List[Dict]] = None,
        Y_validation: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._train_set = TransformersTrainDataset(L=label_matrix, inputs=X_train)
        self._collator = collator

        self._test_set = None
        if X_test is not None and Y_test is not None:
            self._test_set = TransformersTestDataset(inputs=X_test, Y=Y_test)

        self._val_set = None
        if X_validation is not None and Y_validation is not None:
            self._val_set = TransformersTestDataset(inputs=X_validation, Y=Y_validation)

    def get_train_data(self) -> TransformersTrainDataset:
        return self._train_set

    def get_test_data(self) -> TransformersTestDataset:
        return self._test_set

    def get_val_data(self) -> TransformersTestDataset:
        return self._val_set

    def train_dataloader(self):
        return DataLoader(
            dataset=self._data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self._collator.train_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self._data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self._collator.test_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self._data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self._collator.test_collate,
        )
