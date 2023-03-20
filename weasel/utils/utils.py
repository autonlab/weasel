import json
import logging
import os
import sys
import warnings
from types import SimpleNamespace
from typing import List, Sequence, Union, Optional, Dict

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_filepath(file, parent_dir=None):
    try:
        import hydra
        original_cwd = hydra.utils.get_original_cwd()
    except (ImportError, ValueError):
        original_cwd = os.getcwd()

    if os.path.isfile(os.path.join(original_cwd, file)):
        return os.path.join(original_cwd, file)
    elif os.path.isfile(os.path.join(original_cwd, os.path.join(parent_dir, file))):
        return os.path.join(original_cwd, os.path.join(parent_dir, file))
    else:
        raise FileNotFoundError(f"Neither {os.path.join(original_cwd, file)} nor {os.path.join(original_cwd, os.path.join(parent_dir, file))}"
                                f" are files!")


def get_namespace_from_json_file(filepath):
    with open(filepath) as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    return config


def to_dict(obj: Optional[Union[dict, SimpleNamespace]]):
    if obj is None:
        return dict()
    elif isinstance(obj, dict):
        return obj
    else:
        return vars(obj)


def to_DictConfig(obj: Optional[Union[List, Dict]]):
    from omegaconf import OmegaConf, DictConfig

    if isinstance(obj, DictConfig):
        return obj

    if isinstance(obj, list):
        try:
            dict_config = OmegaConf.from_dotlist(obj)
        except ValueError as e:
            dict_config = OmegaConf.create(obj)

    elif isinstance(obj, dict):
        dict_config = OmegaConf.create(obj)

    else:
        dict_config = OmegaConf.create()  # empty

    return dict_config


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def stem_word(word):
    return word.lower().strip().replace('-', '_').replace('&', '').replace('+', '')


def get_ML_logger(name, *args, **kwargs):
    from pytorch_lightning import loggers as pl_loggers
    name_stem = stem_word(name)
    if name_stem == "wandb":
        logger = pl_loggers.WandbLogger
    elif name_stem == "comet":
        logger = pl_loggers.CometLogger
    elif name_stem == "tensorboard":
        logger = pl_loggers.TensorBoardLogger
    else:
        # No logging
        return False

    return logger(*args, **kwargs)


def get_all_instantiable_hydra_modules(config, module_name: str):
    from hydra.utils import instantiate as hydra_instantiate
    modules = []
    if module_name in config:
        for _, module_config in config[module_name].items():
            if "_target_" in module_config:
                modules.append(
                    hydra_instantiate(module_config)
                )
    return modules


def extras(config) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Modifies DictConfig in place.
    """

    log = get_logger()

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    accelerator = config.trainer.get("accelerator")
    if accelerator in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info(f"Forcing ddp friendly configuration! <config.trainer.accelerator={accelerator}>")
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False


@rank_zero_only
def print_config(
        config,
        fields: Union[str, Sequence[str]] = (
                "datamodule",
                "end_model",
                "Weasel",
                "trainer",
                # "callbacks",
                # "logger",
                "seed",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Args:
        config (ConfigDict): Configuration
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    import importlib
    if not importlib.util.find_spec("rich") or not importlib.util.find_spec("omegaconf"):
        # no pretty printing
        return
    from omegaconf import DictConfig, OmegaConf
    import rich.syntax
    import rich.tree

    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)
    if isinstance(fields, str):
        if fields.lower() == 'all':
            fields = config.keys()
        else:
            fields = [fields]

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


def no_op(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
        config,
        model: pl.LightningModule,
        data_module: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Additionally saves:
        - number of {total, trainable, non-trainable} model parameters
    """

    def copy_and_ignore_keys(dictionary, *keys_to_ignore):
        new_dict = dict()
        for k in dictionary.keys():
            if k not in keys_to_ignore:
                new_dict[k] = dictionary[k]
        return new_dict

    params = dict()
    if 'seed' in config:
        params['seed'] = config['seed']

    # Remove redundant keys or those that are not important to know after training -- feel free to edit this!
    params["datamodule"] = copy_and_ignore_keys(config["datamodule"], 'pin_memory', 'num_workers')
    params["trainer"] = copy_and_ignore_keys(config["trainer"])
    params["Weasel"] = copy_and_ignore_keys(config["Weasel"],
                                            'num_LFs', 'n_classes', 'class_balance', 'verbose',
                                            'encoder', 'optim_end_model', 'optim_encoder', 'scheduler')
    # encoder, optims, and scheduler as separate top-level key
    params['encoder'] = config['Weasel']['encoder']
    params['optim_end_model'] = config['Weasel']['optim_end_model']
    params['optim_encoder'] = config['Weasel']['optim_encoder']
    params['scheduler'] = config['Weasel']['scheduler'] if 'scheduler' in config['Weasel'] else None

    if "callbacks" in config:
        if 'model_checkpoint' in config['callbacks']:
            params["model_checkpoint"] = copy_and_ignore_keys(
                config["callbacks"]['model_checkpoint'], 'save_top_k'
            )

    # save number of model parameters
    params["model/params_total"] = sum(p.numel() for p in model.parameters())
    params["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    params["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(params)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = no_op


def change_labels(*args, new_label=-1, old_label=0):
    lst = []
    for arg in args:
        A = arg.copy()
        new_old = (A == new_label)
        A[A == old_label] = new_label
        A[new_old] = old_label
        if len(args) == 1:
            return A
        lst.append(A)

    return tuple(lst)


def get_activation_function(name, functional=False, num=1):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    name = name.lower().strip()

    funcs = {"softmax": F.softmax, "relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid, "identity": None,
             None: None, 'swish': F.silu, 'silu': F.silu, 'elu': F.elu, 'gelu': F.gelu,
             'prelu': nn.PReLU()}

    nn_funcs = {"softmax": nn.Softmax(dim=1), "relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
                "identity": nn.Identity(), 'silu': nn.SiLU(), 'elu': nn.ELU(), 'prelu': nn.PReLU(),
                'swish': nn.SiLU(), 'gelu': nn.GELU()}
    if num == 1:
        return funcs[name] if functional else nn_funcs[name]
    else:
        return [nn_funcs[name] for _ in range(num)]


def lower_is_better(metric_name):
    metric_name = metric_name.lower().strip()
    if metric_name in ['mse', 'l2', 'mae', 'l1']:
        return True
    return False
