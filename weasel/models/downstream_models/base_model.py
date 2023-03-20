from typing import Optional, Union, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from weasel.utils.optimization import get_loss
from weasel.utils.prediction_and_evaluation import eval_final_predictions, get_optimal_decision_threshold


class DownstreamBaseModel(LightningModule):
    """
    This is a template class, that should be inherited by all downstream models (== end-models).
    Methods that need to be implemented by your concrete end model (just as if you would define a torch.nn.Module):
        - __init__(.)
        - forward(.)

    The other methods may be overridden as needed.
    It is recommended to define the attributes:
        - self.example_input_array = torch.randn(<YourModelInputShape>)  # batch dimension can be anything, e.g. 7
        - self.out_dim OR self.n_classes = <#classes>

    ------------
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.out_dim = -1  # set to the out dimension so that multi-class predictions can be easily recognized

    def forward(self, X: Any) -> torch.Tensor:
        r"""
        Downstream model forward pass
        Args:
            X: Input(s) for a forward-pass.
               Usually a feature tensor, but can be anything that your application calls for, e.g. a list of tensors.
               X will be the identical to what your torch.Datasets that subclass AbstractWeaselDataset or
               AbstractDownstreamDataset return as features, X.

        Returns:
            The predicted logits = end_model(X).

        Important: Weasel expects the end-model to return logits!, i.e. *not* the final probabilistic predictions.
        """
        raise NotImplementedError("Please implement your downstream model's forward pass in the inherited method"
                                  " of DownstreamBaseModel.")

    def get_encoder_features(self, X: Any, *args, **kwargs) -> Any:
        """
        This method returns features that will be used by Weasel's encoder network when Weasel.use_aux_input_for_encoder is True.
        Usually, they can just be the same features X, though they may (need to) be processed by the network or manually.

        args:
            X (Any): the same features that are provided to the main model in self.forward(.)
        Returns:
            Anything that is usable by the encoder network (usually just the same features/a tensor)
                as auxiliary input, besides the label matrix. E.g. a (n, d) tensor for a default MLP encoder.
        """
        return X

    def logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (Tensor): The output of the model's forward method, i.e. logits (unnormalized class probabilities)
        Returns:
            Predicted class probabilities
        """
        if self.out_dim == 1:
            return torch.sigmoid(logits)
        else:
            return nn.Softmax(dim=1)(logits)

    def predict_proba(self, X: Any, *args, **kwargs) -> torch.Tensor:
        r"""
        This is a wrapper method, and will only differ to the snippet below when the forward method does not return
        probabilities (e.g. in cases where the loss computes them by itself, e.g. in BCEwithLogits)
        Args:
            X: the same features that are provided to the main model in self.forward(.), for a batch of size n
        Returns:
            The probabilities P(Y | X) \in [0, 1]^{n \times C} assigned by your downstream model.
        """
        return self.logits_to_probs(self(X))

    def get_decision_thresh(self) -> Optional[float]:
        """
        Get the model's decision threshold for hard predictions in binary classification.
        Override this function when the decision_threshold is stored by a sub-module, e.g. a MLP readout head.
        """
        return float(self._decision_thresh) if hasattr(self, '_decision_thresh') else None

    def set_decision_thresh(self, decision_thresh: Union[float, torch.Tensor]) -> None:
        """
        Set the model's decision threshold for hard predictions in binary classification.
        Override this function when the decision_threshold is stored by a sub-module, e.g. a MLP readout head.
        """
        self._decision_thresh = decision_thresh if torch.is_tensor(decision_thresh) else torch.tensor(decision_thresh)

    def predict(self,
                X: Optional[Any] = None,
                probs: Optional[Union[torch.Tensor, np.ndarray]] = None,
                *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            X: Either None, or features/object(s) that will be used for prediction by self.predict_proba(X).
            probs: Probabilistic labels (n, C), or None

        Note: If no probabilistic/soft labels are given, X must not be None in order to forward-pass through the model.

        Returns:
             A tuple (Y_soft, Y_hard), where
              - Y_soft are probabilistic labels \in [0, 1]^{n \times C}, and
              - Y_hard are the corresponding hard predictions \in {0, .., C-1}^n
        """
        if probs is None:
            assert X is not None, 'Please provide soft labels, or features to generate them'
            probs = self.predict_proba(X)

        if 1 <= self.out_dim <= 2 or probs.shape[1] <= 2:
            decision_thresh = self.get_decision_thresh() or 0.5
            preds = probs.clone() if torch.is_tensor(probs) else probs.copy()
            if self.out_dim == 2:
                preds = preds[:, 1]
            preds[preds >= decision_thresh] = 1
            preds[preds < decision_thresh] = 0
        else:
            preds = torch.argmax(probs, dim=1) if torch.is_tensor(probs) else np.argmax(probs, axis=1)
        return probs, preds

    # --------------------- support for evaluation with PyTorch Lightning (directly used as-is by Weasel)
    def _evaluation_step(self, batch: Any, batch_idx: int):
        X, Y = batch
        preds = self.predict_proba(X)
        return {'targets': Y, 'preds': preds}

    def _evaluation_get_preds(self, outputs: List[Any]):
        Y = torch.cat([batch['targets'] for batch in outputs], dim=0).cpu().numpy()
        preds = torch.cat([batch['preds'] for batch in outputs], dim=0).detach().cpu().numpy()
        return Y, preds

    def _evaluation_log(self, Y: np.ndarray, preds: np.ndarray, split: str, verbose: bool = False) -> dict:
        _, hard_preds = self.predict(probs=preds)
        stats = eval_final_predictions(Y, hard_preds, probs=preds, verbose=verbose,
                                       only_on_labeled=False, add_prefix=split.capitalize(),
                                       model_name=self.__str__(),
                                       is_binary=1 <= self.out_dim <= 2 or preds.shape[1] <= 2)
        if self.get_decision_thresh() is not None:
            stats['decision_thresh'] = self.get_decision_thresh()
        if self._trainer is not None:
            self.log_dict(stats, prog_bar=True)
        return stats

    def validation_step(self, batch: Any, batch_idx: int):
        return self._evaluation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs: List[Any]):
        Y, preds = self._evaluation_get_preds(outputs)
        if self.get_decision_thresh() and (not hasattr(self, 'adjust_thresh') or self.adjust_thresh):
            self.set_decision_thresh(
                get_optimal_decision_threshold(preds, Y)
            )

        return self._evaluation_log(Y, preds, split='Val')

    def test_step(self, batch: Any, batch_idx: int):
        return self._evaluation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: List[Any]):
        Y, preds = self._evaluation_get_preds(outputs)
        return self._evaluation_log(Y, preds, split='Test')

    # -------------------- And for training, e.g. for baselines. This is for your convenience, Weasel doesnt use this.
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def set_criterion(self, criterion):
        self.criterion = criterion

    def on_train_start(self) -> None:
        if hasattr(self, 'criterion') and self.criterion is not None:
            return
        else:
            # Training with (X, Y) pairs, where Y is probabilistic
            self.criterion = get_loss('cross_entropy', logit_targets=False, probabilistic=True)

    def training_step(self, batch: Any, batch_idx: int):
        X, Y = batch
        preds = self(X)
        loss = self.criterion(preds, target=Y)
        return loss
