from experiments.duplicated_LFs import BenchmarkerDeps
from experiments.independent_random_LFs import BenchmarkerRandomLFs
from get_params_and_args_benchmark import setup_and_get_params
if __name__ == '__main__':
    ######################################################################################
    #        PLEASE EDIT THE FOLLOWING LINES FOR THE SYNTHETIC                           #
    #          ROBUSTNESS EXPERIMENT THAT YOU WANT TO RUN                                #
    EXPERIMENT_TYPE = 'SYNTH_DUPLICATED_RANDOM_LF'    # either duplications of the same random LF
    # EXPERIMENT_TYPE = 'SYNTH_INDEPENDENT_RANDOM_LFs'  # or multiple, independent random LFs
    dataset_name, num_LFs = 'professor_teacher', 99                                      #
    ######################################################################################
    data_dict, param_dict, dirs = setup_and_get_params(dataset_name, prefix='', num_LFs=num_LFs)

    _, Xtrain, Xtest, Ytest = data_dict['L'], data_dict['Xtrain'], data_dict['Xtest'], data_dict['Ytest']

    model_params, mlp_params, hyper_params = param_dict['model_params'], param_dict['end_params'], param_dict[
        'hyper_params']
    Ytrain = data_dict['Ytrain_gold']  # will become one LF, that is corrupted by the other random LFs

    if EXPERIMENT_TYPE == 'SYNTH_DUPLICATED_RANDOM_LF':
        benchmarker = BenchmarkerDeps(dirs=dirs,  dataset=param_dict['dataset'])
    elif EXPERIMENT_TYPE == 'SYNTH_INDEPENDENT_RANDOM_LFs':
        benchmarker = BenchmarkerRandomLFs(dirs=dirs, dataset=param_dict['dataset'])
    else:
        raise ValueError
    # no label matrix is needed, since it is generated from the labels themselves
    benchmarker.run_synth(Xtrain, Ytrain, Ytest, Xtest, model_params, hyper_params, end_params=mlp_params)

    benchmarker.analyze_seeded_runs()
    # benchmarker.plot_seeded_stats(metric='f1')  # PLOTTING

