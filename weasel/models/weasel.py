import importlib
import logging
from typing import Any, List, Optional, Union, Dict, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule

from weasel.utils.optimization import get_loss, get_scheduler, get_optimizer
from weasel.models.downstream_models.base_model import DownstreamBaseModel
from weasel.utils.utils import to_dict, to_DictConfig, get_logger

log = logging.getLogger(__name__)
if importlib.util.find_spec("hydra"):
    USE_HYDRA = True
    from hydra.utils import instantiate as hydra_instantiate
    from omegaconf import DictConfig, OmegaConf

    HIERARCHICAL_ARG = Union[DictConfig, SimpleNamespace, Dict, List]

else:
    USE_HYDRA = False
    HIERARCHICAL_ARG = Union[SimpleNamespace, Dict]


class Weasel(LightningModule):
    def __init__(
            self,
            end_model: DownstreamBaseModel,
            num_LFs: int,
            n_classes: Optional[int] = None,
            encoder: Optional[HIERARCHICAL_ARG] = None,
            optim_end_model: Optional[HIERARCHICAL_ARG] = None,
            optim_encoder: Optional[HIERARCHICAL_ARG] = None,
            scheduler: Optional[HIERARCHICAL_ARG] = None,
            monitor: Optional[str] = None,
            loss_function: str = 'cross_entropy',
            use_aux_input_for_encoder: bool = False,
            accuracy_scaler: Union[float, str] = "sqrt",
            temperature: float = 2.0,
            class_balance: Optional[List[float]] = None,
            class_conditional_accuracies: bool = True,
            verbose: bool = True,
            end_model_example_input: Optional[Any] = None,
    ):
        r"""
        m - Number of labeling functions (LF)
        C - Number of classes

        Args:
            end_model (DownstreamBaseModel): Your instantiated downstream model that we aim to train based on our
                end-to-end weakly supervised approach based on only multiple noisy labeling heuristics for training.

            num_LFs (int): The number of labeling functions, m

            n_classes (int): jThe number of classes, C (e.g. 2 for binary classification).

            encoder (optional dict, Namespace, DictConfig, List): A configuration that defines the encoder model's
                    architecture, see models/encoder_models/encoder_MLP.py for a simple, but well performing, example of
                     an encoder model that you can simply re-use (default).
                     When None, a MLP encoder is used, however its (hyper-)parameters may not be the best option,
                     or it might not even work if 'use_aux_input_for_encoder' is True AND
                     end_model.get_encoder_features() has not been defined appropriately to return (n, d)-shaped tensors
                     for the MLP encoder.

            optim_end_model (optional dict, Namespace, DictConfig, List): Configuration that defines a torch.optim that
                    will optimize your end-model. If None, it defaults to Adam with 1e-4 learning-rate.

            optim_encoder (optional dict, Namespace, DictConfig, List): Configuration that defines a torch.optim that
                    will optimize the encoder. If None, it defaults to Adam with 1e-4 learning-rate.

            scheduler (optional dict, Namespace, DictConfig, List): Configuration that defines a torch scheduler that
                    will step the learning-rates of both end- and encoder-models. If None, no scheduler is used.

            monitor (optional str): Only used for LRonPlateau scheduler.
                    Defaults to AUC in binary, F1_micro in multi-class case.

            loss_function (str): The loss function to use. Default (and recommended): Cross-entropy.
                    Internally, a symmetric version with stop-gradient applied to the targets is used.

            use_aux_input_for_encoder (bool): Flag that controls whether the encoder model will process the output of
                    end_model.get_encoder_features(), e.g. the input features X, as auxiliary input to the label matrix.

            accuracy_scaler (float, str): How to scale the accuracies, either a float or 'sqrt'.
                    If neither of, the scaler becomes num_LFs.

            temperature (float): The softmax temperature, \tau_1, to use before generating the accuracy scores.
                    A higher temperature will lead to more uniform accuracies,
                        as \tau_1 -> infinity, the scores are thus similar to the (equally-weighted) majority vote.
                        as \tau_2 -> 0, the softmax will behave like a max(.) & the scores will concentrate around
                         one LF for each sample. This can make Weasel more prone to overfitting. Default: 2.0

            class_balance (None or list): The prior probability of the classes, P(y) \in [0, 1]^C that sums up to 1.
                    Default: None, i.e. an uniform class balance is assumed.

            class_conditional_accuracies (bool): Whether to use class-conditional accuracies, i.e. learn different
                    accuracy scores for each class.
                    That is, when set (default) the accuracy scores will have shape (m, C), when False (m, 1).

            end_model_example_input: Only used when use_aux_input_for_encoder is True. It is optional in that case, and
                used to infer the form/shape of the auxiliary input.
                If None (and use_aux_input_for_encoder is True), one of the following must hold_
                    - YOUR_END_MODEL.example_input_array is not None (has the attribute set correctly), recommended!
                    - 'extra_input_shape' is defined in the encoder DictConfig
        """
        super().__init__()
        if num_LFs <= 1:
            raise ValueError(f'Weasel expects a minimum of 2 LFs, but num_LFs={num_LFs} was given to Weasel(...).')
        if n_classes is None:
            if hasattr(end_model, 'n_classes') and end_model.n_classes >= 2:
                n_classes = end_model.n_classes
            elif hasattr(end_model, 'out_dim') and end_model.out_dim >= 1:
                n_classes = 2 if 1 <= end_model.out_dim <= 2 else end_model.out_dim
            else:
                raise ValueError("Weasel arg 'n_classes' is None, but end_model doesn't have a 'n_classes' or 'out_dim'"
                                 " attribute. Pass the number of classes explicitly or define them in your end-model.")
            log.info(f"Weasel inferred #classes = {n_classes} from your end-model, "
                         f"since n_classes was passed as None in Weasel's constructor.")

        elif n_classes <= 1:
            raise ValueError(
                f'Weasel expects a minimum of 2 classes, but n_classes={n_classes} was given to Weasel(...).'
            )

        # Automatically saves all arguments :)
        self.save_hyperparameters(ignore=['end_model_example_input'])
        self.end_model = end_model

        if not use_aux_input_for_encoder:
            encoder_aux_input = None
        elif end_model_example_input is not None:
            encoder_aux_input = self.end_model.get_encoder_features(end_model_example_input)
        elif self.end_model.example_input_array is not None:
            encoder_aux_input = self.end_model.get_encoder_features(self.end_model.example_input_array)
        elif encoder.get('extra_input_shape') is not None:
            encoder_aux_input = eval(encoder['extra_input_shape'])
        else:
            raise ValueError("When using auxiliary input for the encoder model, other than the label matrix,"
                             " *one of the following must hold*:\n"
                             "  - Weasel argument 'end_model_example_input' is not None\n"
                             "  - YOUR_END_MODEL.example_input_array is not None\n"
                             "  - 'extra_input_shape' is defined in the encoder Config")
        if use_aux_input_for_encoder and not torch.is_tensor(encoder_aux_input):
            encoder_aux_input = torch.randn(size=encoder_aux_input)

        encoder_out_size = num_LFs
        if class_conditional_accuracies:
            encoder_out_size *= n_classes

        if USE_HYDRA and '_target_' in to_DictConfig(encoder).keys():  # more flexible and recommended
            self.encoder = hydra_instantiate(to_DictConfig(encoder),
                                             num_LFs=num_LFs,
                                             example_extra_input=encoder_aux_input,
                                             output_dim=encoder_out_size)
        else:
            from weasel.models.encoder_models.encoder_MLP import MLPEncoder as DefaultEncoder
            self.encoder = DefaultEncoder(**to_dict(encoder),
                                          num_LFs=num_LFs,
                                          example_extra_input=encoder_aux_input,
                                          output_dim=encoder_out_size)

        self.accuracy_func = nn.Softmax(dim=1)

        # Class prior
        if class_balance is None:
            class_balance = [1 / n_classes] * n_classes
        self.register_buffer('class_balance', torch.tensor(class_balance))

        # loss function
        self.criterion = get_loss(loss_function)

        if isinstance(accuracy_scaler, float) or isinstance(accuracy_scaler, int):
            assert accuracy_scaler > 0, 'Accuracy scaler must be positive'
            self.acc_scaler = accuracy_scaler
        else:
            self.acc_scaler = num_LFs
            if accuracy_scaler.lower() in ["sqrt", "root"]:
                self.acc_scaler = np.sqrt(self.acc_scaler)
            if class_conditional_accuracies:
                self.acc_scaler *= n_classes

    def on_fit_start(self) -> None:
        self.end_model.trainer = None  # so that the end-model doesn't try to log anything if wrapper around by Weasel

    def on_test_start(self) -> None:
        self.end_model.trainer = None  # so that the end-model doesn't try to log anything if wrapper around by Weasel

    def forward(self, X: Any):
        return self.end_model(X)

    def encode(self, label_matrix: torch.Tensor, extra_input: Any = None):
        r"""
        Args:
            label_matrix: a (n, m) tensor L, where
                            L_ij = -1 if the j-th LF abstained on i, and
                            L_ij = c if the j-th LF voted for class c for the sample i, c \in {0, ..., C-1}

            extra_input: Either None, or an auxiliary input for the encoder net

        Returns:
            A (n, C) tensor, with logits of the class probabilities predicted by the encoder network
        """
        # get the LF sample-dependent accuracies
        accuracies = self.get_accuracy_scores(label_matrix, extra_input)

        # Make label matrix (n, m) a one-hot/indicator matrix (n, m, C)
        L = self._create_L_ind(label_matrix)

        if self.hparams.class_conditional_accuracies:
            aggregation = (L * accuracies).sum(dim=1)
            # The following snippet of code is equivalent to the (vectorized) line above:
            # aggregationLONG = torch.zeros((accuracies.shape[0], self.cardinality)).to(device)  # (n, C)
            # for i, (lf_votes, accs) in enumerate(zip(L, accuracies)):  # iterate through all n batches
            # Multiply the class-conditional accuracies against each class indicator column
            #    aggregationLONG[i, :] = torch.sum(accs * lf_votes, dim=0)  # (m, C) * (m, C) --> (m, C) --> (1, C)
            # assert torch.allclose(aggregationLONG, aggregation),

        else:
            aggregation = (accuracies.unsqueeze(1) @ L).squeeze(1)
            # The following snippet of code is equivalent to the (vectorized) line above:
            # aggregationLONG = torch.zeros((accuracies.shape[0], self.cardinality)).to(device)  # (n, C)
            # for i, (lf_votes, accs) in enumerate(zip(L, accuracies)):  # iterate through all n batches
            # Multiply the accuracies against each class indicator column
            #     aggregationLONG[i, :] = accs @ lf_votes  # (1, m) x (m, C) --> (1, C)
            # assert torch.allclose(aggregationLONG, aggregation)

        aggregation += torch.log(self.class_balance)  # add class prior
        return aggregation, accuracies

    def get_accuracy_scores(self, label_matrix: torch.Tensor, X: Any) -> torch.Tensor:
        enc_features = self.end_model.get_encoder_features(X) if self.hparams.use_aux_input_for_encoder else None
        # Forward pass through the encoder network
        raw_accuracies = self.encoder(label_matrix, enc_features)
        # shape: (batch-size, #LFs) or (batch-size, #LFs * #classes) if class conditional accuracies

        if self.hparams.class_conditional_accuracies:  # make accuracies have shape (batch_size, #LFs, #classes)
            raw_accuracies = raw_accuracies.reshape(-1, self.hparams.num_LFs, self.hparams.n_classes)
        raw_accuracies = raw_accuracies / self.hparams.temperature
        # Produce attention scores from the encoder output
        accuracies = self.acc_scaler * self.accuracy_func(raw_accuracies)

        return accuracies

    def _create_L_ind(self, label_matrix):
        """ Adapted from Snorkel v0.9
        Convert a label matrix to a one-hot format.

        Args:
            label_matrix: A (n,m) label matrix with values in {-1, 0, 1,...,C-1}, where -1 means abstains

        Returns:
            A (n, m, C) torch tensor with values in {0, 1}
        """
        n, m = label_matrix.shape
        label_matrix = label_matrix + 1  # mapping abstain to 0 and the classes to {1, ..., C}
        L_ind = torch.zeros((n, m, self.hparams.n_classes), requires_grad=False)
        for class_y in range(1, self.hparams.n_classes + 1):
            # go through Y == 1 (negative), Y == 2 (positive)...
            # A[x::y] slices A starting at x at intervals of y
            # e.g., np.arange(9)[0::3] == np.array([0,3,6])
            L_ind[:, :, class_y - 1] = torch.where(label_matrix == class_y, 1, 0)
        return L_ind.to(label_matrix.device)

    def predict(self, X: Optional[Any] = None,
                probs: Optional[Union[torch.Tensor, np.ndarray]] = None,
                *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Final hard prediction \in {0, 1,..., C-1} of the downstream model for each batch element,
            where X are the features used by it and C are the number of classes

        Returns:
             A tuple (Y_soft, Y_hard), where
              - Y_soft are probabilistic labels \in [0, 1]^{n \times C}, and
              - Y_hard are the corresponding hard predictions \in {0, .., C-1}^n
            """
        return self.end_model.predict(X, probs, *args, **kwargs)

    def predict_proba(self, X: Any, *args, **kwargs) -> torch.Tensor:
        r""" Final probabilistic prediction \in [0, 1]^C of the downstream model for each batch element,
            where X are the features used by it and C are the number of classes """
        return self.end_model.predict_proba(X, *args, **kwargs)

    # -------------------------------------------------------------------- TRAINING

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        L, X = batch

        logits_endmodel = self(X)
        logits_encoder, accuracies = self.encode(L, X)

        # A mask that should be used to avoid using samples for training where all LFs abstained
        encoder_labels = self.end_model.logits_to_probs(logits_encoder)
        certains = torch.any(encoder_labels != 1 / self.hparams.n_classes, dim=1)

        loss = None
        # Train Encoder
        if optimizer_idx == 0:
            loss = self.criterion(
                logits_encoder[certains], logits_endmodel.detach()[certains]
            )
            self.log("Train/loss_encoder", loss, on_step=False, on_epoch=True, prog_bar=False)

        # Train Downstream model
        if optimizer_idx == 1:
            loss = self.criterion(
                logits_endmodel[certains], logits_encoder.detach()[certains]
            )
            self.log("Train/loss_endmodel", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in on_training_epoch_end()
        # remember to always return loss from training_step, or else backpropagation will fail!
        # return {"loss": loss, "preds_enc": logits_encoder, "targets": targets}

        return loss
    # -------------------------------------------------------------------- Evaluation is on the downstream model simply!
    def validation_step(self, batch: Any, batch_idx: int):
        return self.end_model.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs: List[Any]):
        stats = self.end_model.validation_epoch_end(outputs)
        self.log_dict(stats, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        return self.end_model.test_step(batch, batch_idx)

    def test_epoch_end(self, outputs: List[Any]):
        stats = self.end_model.test_epoch_end(outputs)
        self.log_dict(stats, prog_bar=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self.end_model.predict_step(batch, batch_idx, dataloader_idx)

    # ---------------------------------------------------------------------- Optimizers and scheduler(s)
    def configure_optimizers(self):
        if self.hparams.optim_encoder is None:
            encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        elif USE_HYDRA and '_target_' in to_DictConfig(self.hparams.optim_encoder).keys():
            encoder_optimizer = hydra_instantiate(to_DictConfig(self.hparams.optim_encoder), self.encoder.parameters())
        else:
            encoder_optimizer = get_optimizer(self.encoder, **to_dict(self.hparams.optim_encoder))

        if self.hparams.optim_end_model is None:
            end_optimizer = torch.optim.Adam(self.end_model.parameters(), lr=1e-4)
        elif USE_HYDRA and '_target_' in to_DictConfig(self.hparams.optim_end_model).keys():
            end_optimizer = hydra_instantiate(to_DictConfig(self.hparams.optim_end_model), self.end_model.parameters())
        else:
            end_optimizer = get_optimizer(self.end_model, **to_dict(self.hparams.optim_end_model))

        if self.hparams.scheduler is None:
            return [encoder_optimizer, end_optimizer], []
        elif USE_HYDRA and '_target_' in to_DictConfig(self.hparams.scheduler).keys():
            scheduler_params = to_DictConfig(self.hparams.scheduler)
            scheduler_enc = hydra_instantiate(scheduler_params, optimizer=encoder_optimizer)
            scheduler_end = hydra_instantiate(scheduler_params, optimizer=end_optimizer)
        else:
            scheduler_enc = get_scheduler(encoder_optimizer, **to_dict(self.hparams.scheduler))
            scheduler_end = get_scheduler(end_optimizer, **to_dict(self.hparams.scheduler))

        if self.hparams.monitor is None:
            self.hparams.monitor = 'Val/auc' if self.hparams.n_classes == 2 else 'Val/f1_micro'

        lr_dict_kwargs = {'monitor': self.hparams.monitor}
        lr_dicts = [{'scheduler': scheduler_enc, **lr_dict_kwargs}, {'scheduler': scheduler_end, **lr_dict_kwargs}]
        return [encoder_optimizer, end_optimizer], lr_dicts

    def get_progress_bar_dict(self):
        return dict()
