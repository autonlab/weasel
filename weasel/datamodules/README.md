# Custom data and/or LFs

## Required components
Recall that WeaSEL is applicable whenever you have:
- Training features, *X_train*
- Label matrix, *L*
- Optionally: Test features, *X_test*, with corresponding ground truth test labels, *Y_test*

Here, *L* is a (*n, m*) array/tensor, where
- L_ij = -1 if the j-th LF abstained on i, and
- L_ij = *c* if the j-th LF voted for class *c* \in {0, ..., *C*-1} for the sample i,
- and, *n* are the number of training samples, *m* the number of labeling functions (LF), *C* the number of classes.

<details><p>
<summary><b>Note on probabilistic LFs:</b></summary>

WeaSEL's algorithm in theory is flexible enough to support continuous LF votes, probabilistic LF are such a case, where 
each L_ij \in [0, 1]^C sums up to 1, and the label matrix becomes a (*n, m, C*) tensor 
(as is internally used anyways to compute the encoder labels). The encoder network would however need to be adapted, 
let us know if such an option interests you, you have an use-case or even want to develop it.
</p></details>

The other components are like in a standard machine learning setting:
*X_train* and *X_test* are arbitrary features that your end-model processes in its forward-pass for prediction, while
*Y_test* \in {0, .., C-1}^*n_test* are the *n_test* ground truth labels.
                 
## Making the components ready for WeaSEL training
We recommend that you create a
 [``pl.LightningDataModule``](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html)
  for your dataset.
                    
We provide two convenient options to create such classes that can then be directly used to train WeaSEL:
1. If your features are simple tensors (of arbitrary shape), you can just pass all components to a
[``base_datamodule.BasicWeaselDataModule``](base_datamodule.py) like so (done in [this notebook](../../examples/0_full_pipeline.ipynb) too):
        
        from weasel.datamodules.base_datamodule import BasicWeaselDataModule
        weasel_datamodule = BasicWeaselDataModule(
            label_matrix=L,
            X_train=X_train,
            X_test=X_test,
            Y_test=Y_test,
            batch_size=256,
            val_test_split=(200, -1)  # 200 validation test points will be split from (X_test, Y_test) for validation
        )
 
2. Alternatively, if your features are more complex (e.g. tensor sequences/lists), you can subclass the 
    [``base_datamodule.AbstractWeaselDataModule``](base_datamodule.py).
    You will have to override
     - ``get_train_data()``, to return a torch Dataset that contains (L, X_train) batches for training of Weasel
     - ``get_test_data()``, to return a torch Dataset that contains (X_test, Y_test) batches for evaluation of your end-model
                            and to return None if you do not want to test your model.
     - Optionally: ``get_val_data()``, to return a torch Dataset that contains (X_val, Y_val) batches for validation.
     
     See the [code that defines ProfTeacher_DataModule](../examples/datamodules/ProfTeacher_datamodule.py) for an example.
     

### Note on test/validation sets:
A small validation set with ground truth labels is strongly recommended, but not strictly needed, and the same goes for the test set:
<details><p>
<summary><b>Neither test nor validation sets:</b></summary>
Just make ``get_test_data()`` return ``None`` and only a training set will exist.
</p></details>
<details><p>
<summary><b>Test but no validation set:</b></summary>
Override ``get_test_data()`` appropriately to return your test set and make sure
 to set the arg ``val_test_split=(0, -1)`` so that no validation set is split off your test set.
</p></details>


***If you run Weasel without a validation set, we recommend to run for fewer epochs (e.g. 50), and set a higher temperature 
parameter (e.g. 3.0) in [Weasel's constructor](../models/weasel.py).***
Note that without validation set you won't be able to do early-stopping, decision threshold tuning,
 or checkpointing based on validation performance.

**Important:** without validation set you will have to train the pl.Trainer the following way:
    
    my_DataModule = ...
    from weasel.models import Weasel
    weasel_model = Weasel(...)
    
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        limit_val_batches=0
    )
    trainer.fit(
        model=weasel_model, 
        train_dataloader=my_DataModule.train_dataloader()
     )