# End-to-End Weak Supervision

This is the source code supporting our [*End-to-End Weak Supervision* paper](https://arxiv.org/abs/2107.02233).
Our core encoder network is implemented [here](encoder_network.py), while the whole end-to-end model 
 can be found [here](end_to_end_ws_model.py).

## Environment & Data Setup

To get started with the code, please create a new conda environment as follows (it runs on Python 3.7):

    conda env create -f env_gpu.yml 
    conda activate E2E  # E2E is the name of the environment
    
This will install all required dependencies, and requires a GPU. If no GPU is available, please 
remove line 7 (*cudatoolkit*) from the [environment file](env_gpu.yml), and proceed as above.

The smallest dataset, Bias in Bios is already included in [the data subdirectory](data/).
All the rest can be downloaded from this [Google drive link](https://drive.google.com/drive/folders/1v7IzA3Ab5zDEsRpLBWmJnXo5841tSOlh?usp=sharing).
Please put the downloaded data into the [data/](data) directory.

## Reproducibility

### Main Benchmark and Crowdsourcing
To reproduce our main reported results, please run the [benchmarking script](run_benchmark.py) for a specific dataset, by setting the prepared lines

        dataset_name, num_LFs =  <dataset>, <number-of-labeling-functions>      

to the experiment you wish to reproduce.
You can then run the script as follows

    python run_benchmark.py --<argument1> <argument-value1> .... --<argumentN> <argument-valueN>
E.g.

    python run_benchmark.py --gpu_id 2  # runs the benchmark on the 2nd GPU

This will run the label model benchmark for 7 different seeds, and save the resulting statistics
in ``benchmark_runs/<experiment-ID>/results/benchmark1,2,3,4,5,6,7.pkl``, where ``<experiment-ID>`` is an
 automatically generated unique ID for the experiment that starts with the dataset name and number of LFs. 
The Spouse benchmark needs to be run from this [script](run_benchmark_spouses.py).
All best performing models w.r.t validation AUC will be saved at a checkpoints subdir.
Note that to fully reproduce our experiments, you will need to run the script
for the different hyperparameter configurations (i.e. using a learning rate of 3e-5 and 1e-4 (default) for all datasets)
 
**All statistics and metrics are also saved to Weights&Biases (wandb), so please enter your wandb username
if you want to use it too.**

### Synthetic robustness experiments
 
To reproduce our synthetic experiments results and the LF set generation, where one LF is 100% accurate (i.e. the ground truth labels
of ProfTeacher) and the rest are not better than a coin flip, please run [this script](run_robustness_exps.py).


## Using your own downstream model

Our framework supports arbitrary downstream models, we implemented a MLP and a LSTM already in the
 [downstream_models](downstream_models) directory. To use a custom model, just inherit the class
 [DownstreamBaseModel](downstream_models/base_model.py) and implement your model in the standard 
 \__init\__(.) and forward(.) methods (and, if needed, the other methods too).
  Then you should be good to go to just train our end-to-end model as follows:
  
    # Instantiate the downstream model
    your_model = YourModelClass(model_parameters)
    # Instantiate the E2E trainer class, in dirs you may pass directories for 'checkpoints', 'logging' (tensorboard)                 
    trainer = E2ETrainer(encoder_params, downstream_model=your_model, dirs=dirs)
    # Train the model by maximizing its agreement with the structured encoder network
    trainer.fit(trainloader, valloader, hyper_params, testloader=testloader)
    # Evaluate the model's performance on true data, if available
    trainer.evaluate(testloader, use_best_model=True)

The model will be expected to process the features inside the torch DataLoaders on its own 
(e.g. including moving the tensor(s) to the GPU: ``x.to('cuda')``).


## Using your own dataset or weak supervision sources
To do this, you just need to create torch DataLoader instances for your training, validation sets
(and, possibly, test set), where the training dataset should have the form of the ``DP`` torch Dataset
class within [the data loading script](utils/data_loading.py) (or just pass your data to ``DP``).
That is, it is expected to return ``(label_matrix, features)`` pairs. <br>
The validation set should be (analogous) to ``DownstreamDP``, and return ``(features, label)``
pairs (which will be solely processed for downstream model evaluation).

