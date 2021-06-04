import json
import os
import pickle
import time

import numpy as np
from flyingsquid.label_model import LabelModel as LM_Triplets
from snorkel.labeling.model import LabelModel as LM_Snorkel

from experiments.benchmark import Benchmarker
from encoder_network import MulticlassEncoder
from end_to_end_ws_model import E2ETrainer
from utils.data_loading import get_spouses_data, DownstreamSpouses
from utils.prediction_and_evaluation import eval_final_predictions, get_majority_vote
from utils.utils import change_labels, set_seed
from downstream_models.LSTM import LSTM_Trainer, LSTMModel
from get_params_and_args_benchmark import get_argparser, set_gpu, set_args, setup_and_get_params


def main(dirs, args, seeds, mode='all'):
    with open('configs/spouses9.json') as f:
        config = json.load(f)
    model_params, hyper_params = config["model_params"], config["hyper_params"]
    lstm_params = config['end_params']
    trainloader, valloader, testloader = get_spouses_data(batch_size=hyper_params['batch_size'])
    mode = mode.lower()
    set_gpu(args.gpu_id)
    class_balance = [0.9, 0.1]
    model_dir = dirs['checkpoints']
    benchmark_stats = {seed: dict() for seed in seeds}

    L_train, L_dev, Xtrain, _, _, Y_dev, _ = get_spouses_data(dataloader=False)
    L_snorkel, L_dev_snorkel = change_labels(L_train, L_dev, old_label=0, new_label=-1)

    for seed in seeds:
        seed_id = str(seed) + 'seed_'
        set_seed(seed)
        print(dirs['ID'], 'Seed:', seed)
        hyper_params['seed'] = seed
        if mode in ['all', 'baselines']:
            kwargs = {
                'features': Xtrain, 'valloader': valloader, 'testloader': testloader,
                'hyper_params': hyper_params, 'lstm_params': lstm_params
            }


            # Majority Vote
            preds_MMVV = get_majority_vote(L_train, probs=True, abstention_policy='Stay', abstention=0)
            hard_preds = preds_MMVV
            hard_preds[hard_preds[:, 0] != 0.5, :] = np.round(hard_preds[hard_preds[:, 0] != 0.5, :])
            mv_stats = train_lstm_with_soft_labels(hard_preds, **kwargs, prefix='MV',
                                                   model_dir=model_dir + f'majority/{seed_id}')
            benchmark_stats[seed]['MV'] = mv_stats

            # Snorkel
            best_val_snorkel, snorkel_probs = -1, None
            for epochs, lr, reg in [(100, 0.01, 0.0), (200, 0.05, 0.0), (2000, 0.0003, 0.0),
                                    (500, 0.003, 0.0), (1000, 0.003, 0.1), (5000, 0.01, 0.0)]:
                lm = LM_Snorkel(cardinality=2)
                lm.fit(L_snorkel, class_balance=class_balance, n_epochs=epochs, lr=lr, seed=seed, l2=reg)
                probs_snorkel_dev = lm.predict_proba(L_dev_snorkel)[:, 1]
                gen_stats = eval_final_predictions(Y_dev, np.round(probs_snorkel_dev), probs=probs_snorkel_dev,
                                                   abstention=-1, verbose=True, neg_label=0, only_on_labeled=False,
                                                   model_name=f'Snorkel GEN\n')
                if gen_stats['auc'] > best_val_snorkel:  # search for best generative label model
                    best_val_snorkel = gen_stats['auc']
                    snorkel_probs = lm.predict_proba(L_snorkel)
            sn_stats = train_lstm_with_soft_labels(snorkel_probs, **kwargs, prefix='Snorkel',
                                                   model_dir=model_dir + f'snorkel/{seed_id}')
            benchmark_stats[seed]['snorkel'] = sn_stats

            for method in ['triplet', 'triplet_mean', 'triplet_median']:
                lm = LM_Triplets(L_train.shape[1], triplet_seed=seed)
                lm.fit(L_train, solve_method=method, class_balance=np.array(class_balance))
                probs = lm.predict_proba(L_train)  # [:, 1]
                tr_stats = train_lstm_with_soft_labels(probs, **kwargs, prefix=method,
                                                       model_dir=model_dir + f'_{method}/{seed_id}')
                benchmark_stats[seed][method] = tr_stats

        if mode in ['all', 'e2e']:
            print('***' * 5, dirs['ID'], ID)
            endmodel = LSTMModel(lstm_params)
            encoder = MulticlassEncoder
            trainer = E2ETrainer(model_params, downstream_model=endmodel, seed=seed, dirs=dirs, encoder_net=encoder)
            valid_f1 = trainer.fit(trainloader, valloader, hyper_params, testloader=testloader, use_wandb=False)
            _, _, end_stats = trainer.evaluate(testloader, prefix=f'E2E Test END:\n', verbose=True)
            _, _, best_stats = trainer.evaluate(testloader, prefix=f'E2E Test BEST:\n',
                                                adjust_thresh=False, use_best_model=True, verbose=True)
            best_stats[f'validation_val'] = valid_f1
            benchmark_stats[seed][f'E2E'] = {"LSTM_best": best_stats, "LSTM_end": end_stats}

        with open(dirs['results'] + "benchmark" + ",".join([str(s) for s in seeds]) + '.pkl', 'wb') as f:
            pickle.dump(benchmark_stats, f, pickle.HIGHEST_PROTOCOL)


def train_lstm_with_soft_labels(softlabels, features, hyper_params, valloader, testloader, lstm_params=None, prefix='', model_dir=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    traindata = DownstreamSpouses(Y=softlabels, features=features, filter_uncertains=True, uncertain=0.5)
    lstm = LSTM_Trainer(lstm_params, model_dir=model_dir)
    best_valid = lstm.fit(traindata, hyper_params=hyper_params, valloader=valloader, testloader=testloader,
                          device="cuda", use_wandb=False)
    _, _, end_stats = lstm.evaluate(testloader, device="cuda", print_prefix=f'{prefix} LSTM END:\n',
                                    adjust_thresh=False, use_best_model=False)
    _, _, best_stats = lstm.evaluate(testloader, device="cuda", print_prefix=f'{prefix} LSTM BEST:\n',
                                     adjust_thresh=False, use_best_model=True)
    best_stats['validation_val'] = best_valid
    stats = {"LSTM_best": best_stats, 'LSTM_end': end_stats}
    return stats

if __name__ == '__main__':
    mode = ID = 'all'
    ID = ''
    setup_and_get_params('Spouses', prefix=mode, num_LFs=None, notebook_mode=False, reload_mode=True)
    parser = get_argparser()
    args = parser.parse_args()
    seeds = [1, 2, 3, 4, 5, 6, 7]
    ID += str(len(seeds)) + 'seeds'
    out_dir = 'benchmark_runs/'
    suffix = ID + "_GPU" + str(args.gpu_id) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    pref = 'Spouses_' + suffix
    out_dir = out_dir + pref + '/'
    log_dir = out_dir + 'logs/'
    ckpt_dir = out_dir + 'checkpoints/'
    stats_dir = out_dir + 'results/'
    dirs = {"logging": log_dir, "checkpoints": ckpt_dir, "results": stats_dir, "out": out_dir, 'ID': pref}

    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    with open(f"configs/spouses9.json") as f_from:
        with open(dirs['out'] + 'config.json', 'w') as ff:
            json.dump(json.load(f_from), ff)

    # RUN THE EXPERIMENTS
    main(dirs, args, seeds, mode=mode)

    # ANALYZE
    print(pref)
    result_dir = "benchmark_runs/" + dirs['ID'] + "/results/"
    benchmarker = Benchmarker(seeds=seeds, dirs=None, dataset='Spouses')
    benchmarker.analyze(direc=result_dir, endmodel='LSTM')
    # The following will print out all statistics,
    # note that models with ending '_STO' refer to the stochastic/regularized variant of the end model for the baselines
    # and regularized encoder and end model for our own models (same config for the end model for all models).
    benchmarker.print_latex(endmodel='LSTM', metrics=['f1'])
