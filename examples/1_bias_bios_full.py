"""
This script is basically the same as the notebook 1_bias_bios, but with small extensions/changes.
It is meant as a script that you would realistically run on your own dataset and application.
This includes
    - logging to Weights&Biases
    - callbacks
    - how to retrieve your stand-alone end-model after training it with Weasel.
    - differently to the notebook, we will also be adjusting the model's decision threshold on the small validation set.
        This will have small positive effects in this example dataset, but can help quite a bit for unbalanced classes.
It excludes the Snorkel baseline at the bottom of the notebook which was for illustrative purposes.
Note:
    This script will log to Weights&Biases (wandb), which is a great choice for logging,
    but will require you to set up an account to run this if you do not have one yet (and have it pip installed).
"""
import os
import hydra
import torch
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig

import wandb
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import weasel.utils.utils as utils


@hydra.main(config_path="configs/", config_name="profTeacher_full.yaml", version_base=None)
def run(config: DictConfig):
    utils.print_config(config, fields='all')
    seed_everything(config.seed, workers=True)
    log = utils.get_logger(__name__)
    utils.extras(config)
    uses_wandb = config.get('logger') and config.logger.get("wandb") is not None

    # First we instantiate our end-model, in this case a simple 2-layer MLP, but you
    # can easily replace it with *any* neural net model, see the instructions in the Readme.
    end_model = hydra_instantiate(config.end_model)

    data_module = hydra_instantiate(config.datamodule)

    # We now simply need to pass this end-model to the wrapping Weasel model.
    weasel_model = hydra_instantiate(config.Weasel, end_model=end_model, _recursive_=False)

    # Then, with all the convenience and ease of PyTorch Lightning,
    # we can train our model on the DataModule from above (checkpointing the best model w.r.t. a small validation set),
    # and passing any callbacks you fancy to the Trainer.
    # Init Lightning callbacks and loggers
    callbacks = utils.get_all_instantiable_hydra_modules(config, 'callbacks')
    loggers = utils.get_all_instantiable_hydra_modules(config, 'logger')

    # Init Lightning trainer
    trainer = hydra_instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial", deterministic=True
    )
    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters to the PyTorch Lightning loggers.")
    utils.log_hyperparameters(config=config, model=weasel_model, data_module=data_module, trainer=trainer,
                              callbacks=callbacks)

    trainer.fit(model=weasel_model, datamodule=data_module)

    # Optional Testing:
    trainer.test(datamodule=data_module, ckpt_path='best')

    # After training Weasel, we can simply load its best state based on validation set, and
    #  extract the downstream model that will now be able to make predictions based on only the features, X.
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
    final_model = weasel_model.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    ).end_model
    # Test the final model module directly (same as test above, but removing Weasel and training data from the code):
    pl.Trainer().test(model=final_model, dataloaders=data_module.test_dataloader())

    # Finishing up:
    if uses_wandb:
        wandb.finish()


if __name__ == "__main__":
    os.environ['HYDRA_FULL_ERROR'] = '1'
    run()
