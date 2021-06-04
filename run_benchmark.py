from experiments.benchmark import Benchmarker as MainBenchmarker
from experiments.benchmark_LabelMe import BenchmarkerLabelMe
from get_params_and_args_benchmark import setup_and_get_params

if __name__ == '__main__':
    seeds = [1, 2, 3, 4, 5, 6, 7]

    ######################################################################################
    #      PLEASE EDIT THE FOLLOWING LINES FOR THE EXPERIMENT THAT YOU WANT TO RUN       #
    dataset_name, num_LFs = 'professor_teacher', 99                                      #
    # dataset_name, num_LFs = 'Amazon', 175                                              #
    # dataset_name, num_LFs = 'IMDB', 136                                                #
    # dataset_name, num_LFs = 'IMDB', 12                                                 #
    # dataset_name, num_LFs = 'LabelMe', 59                                              #
    ######################################################################################

    data_dict, param_dict, dirs = setup_and_get_params(dataset_name, prefix='', num_LFs=num_LFs)

    L_arr, Xtrain, Xtest, Ytest \
        = data_dict['L'], data_dict['Xtrain'], data_dict['Xtest'], data_dict['Ytest']
    Ytrain = data_dict['Ytrain_gold']   # only used for creating a validation set for Snorkel generative training.
    model_params, mlp_params, hyper_params = param_dict['model_params'], param_dict['end_params'], param_dict[
        'hyper_params']

    # RUN BENCHMARK
    if param_dict['dataset'].lower() == 'labelme':
        benchmarker = BenchmarkerLabelMe(seeds=seeds, dirs=dirs, dataset=param_dict['dataset'])
    else:
        benchmarker = MainBenchmarker(seeds=seeds, dirs=dirs, dataset=param_dict['dataset'])
    benchmarker.run(L_arr, Xtrain, Ytrain, Ytest, Xtest, model_params, hyper_params, end_params=mlp_params)

    # ANALYZE THE RESULTS
    endmodel = 'MLP'
    result_dir = "benchmark_runs/" + dirs['ID'] + "/results/"
    # Process the statistics...  (also, check out the wandb online outputs)
    benchmarker.analyze(direc=result_dir, endmodel=endmodel)
    # The following will print out all statistics for E2E and baselines (same config for the end model for all models).
    # You can also run this code snippet in the separate script analyze_benchmark.py
    metric = 'f1' if param_dict['dataset'].lower() != 'labelme' else 'accuracy'
    benchmarker.print_latex(endmodel=endmodel, metrics=metric)  # can also print AUC, Accuracy, ...
