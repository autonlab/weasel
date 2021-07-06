# Pre-defined downstream models
In this directory you can find some pre-defined downstream models that you can directly use
for training together with Weasel's encoder network.

Specifically, you can find

- A Multi-layer Perceptron ([MLP](MLP.py), or feed-forward network)
- A (pre-trained) [ResNet](ResNet.py) and [VGG](VGG.py) for vision/image tasks
- A [LSTM](LSTM.py) for sequential tasks

You simply need to instantiate it & pass it to Weasel. Given data, you're then ready to go
to train/fit your downstream model based on the labeling function votes/outputs:
    
    my_DataModule = ...
    
    from weasel.models.downstream_models.LSTM import LSTM
    my_LSTM = LSTM(<architecture_arg1>, ..., <architecture_argN>)

    from weasel.models import Weasel
    weasel = Weasel(
         end_model=my_LSTM,
         num_LFs=m,
         n_classes=C,
         <weasel_arg1>, ..., <weasel_argN>
    )
                            
    # Train your downstream model by maximizing its agreement with the structured encoder network within Weasel
    trainer = pl.Trainer()
    trainer.fit(model=weasel, datamodule=my_DataModule)
                   
    # Evaluate the model's performance on data with ground truth labels, if available
    trainer.test(ckpt_path='best', datamodule=my_DataModule)
    
If you have any questions about these models, how to use them or would like a different one, feel free to reach out 
or open a pull request.


# Using your own custom model

If you want to use your own custom downstream model, this should be easy :)
To use a custom model, just inherit the class
 [DownstreamBaseModel](base_model.py) and implement your model in the standard 
 \__init\__(.) and forward(.) methods (and, if needed, override other methods).
 
For this, feel free to take inspiration from the existing downstream models that build upon this base class, and/or check out
[this notebook](../../../examples/0_full_pipeline.ipynb) that creates a custom CNN and shows the full Weasel
pipeline.

Then you should be good to go to just train our end-to-end model as above:
  
    # Instantiate the downstream model
    your_model = YourModelClass(**model_parameters)
    
    weasel = Weasel(
         end_model=your_model,
         num_LFs=m,
         n_classes=C,
         <weasel_arg1>, ..., <weasel_argN>
    )
        
    # ... continue as above :)



# Baselines, Snorkel & Standard supervised training

For convenience, ``DownstreamBaseModel`` is a ``LightningModule`` with all necessary methods
defined with appropriate defaults, including those that Weasel does not use (mostly training methods).
 
Feel free to use this to train baselines.
For instance:
- train on a small validation set, like below
- or train on Snorkel labels like in [this notebook](../../../examples/1_bias_bios.ipynb) 

If you want to train an end-model in a supervised way, i.e. on *hard labels* y \in {0,..,C-1}, where
the hard-labels can e.g. be your validation set or the majority vote of the LFs, you will
have to adjust the loss function to be the standard cross-entropy (or whatever else you fancy):

    from weasel.utils.optimization import get_loss
    your_model.set_criterion(
        get_loss('CE', logit_targets=False, probabilistic=False)
    )
    
    # To e.g. train on the validation set do:
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        limit_val_batches=0
    )
    trainer.fit(
        model=your_model,  #  instead of weasel
        train_dataloader=my_DataModule.val_dataloader()
     )

 