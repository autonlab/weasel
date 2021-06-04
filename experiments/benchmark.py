import copy
import json
import os
import pickle
import re
import time
from typing import Iterable

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from experiments.baselines import train_supervised, train_triplets, train_snorkel, majority_vote
from downstream_models.LSTM import LSTMModel, LSTM_Trainer
from downstream_models.MLP import MLPNet, MLP_Trainer, LabelMeMLP
from encoder_network import MulticlassEncoder, Encoder
from end_to_end_ws_model import E2ETrainer
from utils.data_loading import split_dataset, DP, DownstreamDP, get_spouses_data
from utils.utils import set_seed



class Benchmarker:
    def __init__(self, seeds, dirs, dataset):
        self.seeds = seeds
        self.dataset = dataset
        if dirs is not None:
            self.model_dir = dirs['checkpoints']
            self.dirs = dirs
            self.save_to = self.dirs['results']
        self.result_file = "benchmark" + ",".join([str(s) for s in self.seeds]) + '.pkl'

    def get_loaders(self, hyperparams, L, Xtrain, Xtest, Ytest, num_workers=0):
        set_seed(3)  # to always get the same valid-test split
        batch_size = hyperparams['batch_size']
        if self.dataset.lower() == 'spouses':
            trainloader, valloader, testloader = get_spouses_data(batch_size=batch_size, dataloader=True)
        else:
            dp_train = DP(L, features=Xtrain)
            kwargs = {'batch_size': batch_size, 'pin_memory': True}
            trainloader = DataLoader(dp_train, shuffle=True, num_workers=num_workers, **kwargs)
            if hyperparams["valset_size"] > 0:
                mlp_val, mlp_test = split_dataset(hyperparams["valset_size"], DownstreamDP, Xtest, Ytest,
                                                  filter_uncertains=False)
                valloader = DataLoader(mlp_val, shuffle=False, **kwargs)
                print('Validation, Test set sizes = ', len(mlp_val), len(mlp_test))
            else:
                mlp_test = DownstreamDP(Xtest, Ytest)
                valloader = None
            testloader = DataLoader(mlp_test, shuffle=False, **kwargs)
        set_seed(hyperparams['seed'])  # reset to old seed
        return trainloader, valloader, testloader

    def get_models(self, end_params):
        if self.dataset.lower() == 'spouses':
            return Encoder, LSTMModel(end_params)
        if self.dataset.lower() == 'labelme':
            return MulticlassEncoder, LabelMeMLP()
        else:
            encModel = MulticlassEncoder if end_params['out_dim'] > 1 else Encoder
            return encModel, MLPNet(end_params)

    def run(self, L, Xtrain, Ytrain, Ytest, Xtest, encoder_params, hyper_params, end_params):
        benchmark_stats = {
            seed: dict() for seed in self.seeds
        }
        for seed in self.seeds:
            seed_id = str(seed) + 'seed_'
            set_seed(seed)
            hyper_params['seed'] = seed
            trainloader, end_valloader, end_testloader = self.get_loaders(hyper_params, L, Xtrain, Xtest, Ytest)
            kwargs = {
                'label_matrix': L, 'X': Xtrain, 'class_balance': encoder_params['class_balance'],
                'end_valloader': end_valloader, 'end_testloader': end_testloader,
                'mlp_params': end_params, 'hyper_params': hyper_params, 'dataset': self.dataset
            }

            wandb_init_dict = {
                'project': f"E2E_{self.dataset}_{encoder_params['num_LFs']}LFs",
                'name': f"{seed}seed" + '___' + time.strftime('%Hh%Mm_on_%b_%d'),
                'config': {**hyper_params, **end_params},
                'reinit': True
            }

            run = wandb.init(**wandb_init_dict, group='Fully Supervised')
            fully_sup_trainset = DownstreamDP(Xtrain, Ytrain, categorical=end_params['out_dim'] > 1)
            stats_sup = train_supervised(train_data=fully_sup_trainset, dataset=self.dataset, wandb_run=run,
                                         valloader=end_valloader, testloader=end_testloader, hyper_params=hyper_params,
                                         mlp_params=end_params, model_dir=self.model_dir + f'supervised/{seed_id}')
            benchmark_stats[seed]['Supervised'] = stats_sup

            run = wandb.init(**wandb_init_dict, group='Majority Vote')
            mv_stats = majority_vote(**kwargs, model_dir=self.model_dir + f'MV/{seed_id}', wandb_run=run)
            benchmark_stats[seed]['majority'] = mv_stats

            run = wandb.init(**wandb_init_dict, group='Snorkel')
            _, sn_probs, sn_stats = train_snorkel(**kwargs, Ytrain=Ytrain, cardinality=end_params['out_dim'],
                                                  model_dir=self.model_dir + f'snorkel/{seed_id}', wandb_run=run)
            benchmark_stats[seed]['snorkel'] = sn_stats

            try:
                run = wandb.init(**wandb_init_dict, group='Triplets')
                _, _, tr_stats = train_triplets(**kwargs, method='', model_dir=self.model_dir + f'triplet/{seed_id}', wandb_run=run)
            except Exception as e:
                print(e, '\nTriplets did not converge!')
                _, tr_stats = get_dummies(sn_probs.shape[0])

            try:
                run = wandb.init(**wandb_init_dict, group='Triplets-Median')
                _, _, tr_med_stats = train_triplets(**kwargs, method='median', wandb_run=run,
                                                    model_dir=self.model_dir + f'tripletMedian/{seed_id}')
            except Exception as e:
                print(e, '\nTriplets-Median did not converge!')
                _, tr_med_stats = get_dummies(sn_probs.shape[0])

            try:
                run = wandb.init(**wandb_init_dict, group='Triplets-Mean')
                _, _, tr_mean_stats = train_triplets(**kwargs, method='mean', wandb_run=run,
                                                     model_dir=self.model_dir + f'tripletMean/{seed_id}')
            except Exception as e:
                print(e, '\nTriplets-Mean did not converge!')
                _, tr_mean_stats = get_dummies(sn_probs.shape[0])

            benchmark_stats[seed]['snorkel'] = sn_stats
            benchmark_stats[seed]['triplet'] = tr_stats
            benchmark_stats[seed]['triplet_median'] = tr_med_stats
            benchmark_stats[seed]['triplet_mean'] = tr_mean_stats

            ''' OURS '''
            run = wandb.init(**wandb_init_dict, group='E2E')
            print('***' * 5, self.dirs['ID'])
            encoder, endmodel = self.get_models(end_params)
            endmodel_name = str(endmodel)
            e2e_dirs = {k: direc + f'End2EndDP{seed_id}seed/' for k, direc in self.dirs.items()}
            e2e_dirs['checkpoints'] = self.model_dir + f'End2EndDP/{seed_id}_'
            trainer = E2ETrainer(encoder_params, encoder_net=encoder, downstream_model=endmodel, seed=seed,
                                 dirs=e2e_dirs)
            valid_f1 = trainer.fit(trainloader, end_valloader, hyper_params, testloader=end_testloader)
            _, _, end_stats = trainer.evaluate(end_testloader, prefix=f'E2E{seed_id} Test End:\n',
                                               use_best_model=False, adjust_thresh=False)
            _, _, best_stats = trainer.evaluate(end_testloader, prefix=f'E2E{seed_id} Test Best:\n',
                                                use_best_model=True, adjust_thresh=False)
            wandb.log({'Final Test F1': best_stats['f1'], 'Final Test AUC': best_stats['auc'],
                       'Final Test Prec.': best_stats['precision'], 'Final Test Rec.': best_stats['recall'],
                       f"Best Val {hyper_params['val_metric'].upper()}": valid_f1}
                      )
            wandb.run.summary["f1"] = best_stats['f1']
            run.finish()
            best_stats['validation_val'] = valid_f1
            print(f"Best valid. {hyper_params['val_metric']}=", valid_f1)
            benchmark_stats[seed][f'E2E'] = {f'{endmodel_name}_end': end_stats,
                                             f'{endmodel_name}_best': best_stats}

            with open(self.save_to + self.result_file, 'wb') as f:
                pickle.dump(benchmark_stats, f, pickle.HIGHEST_PROTOCOL)

        print("---------------------------->         Saving all stats to ", self.save_to)
        with open(self.save_to + self.result_file, 'wb') as f:
            pickle.dump(benchmark_stats, f, pickle.HIGHEST_PROTOCOL)

        return benchmark_stats

    def analyze(self, direc=None, endmodel='MLP'):
        filename = self.save_to + self.result_file if direc is None else direc + self.result_file
        with open(filename, 'rb') as f:
            benchmark_stats = pickle.load(f)
        print(f"Performances on {filename.split('/')[-3]}:")

        MODELS = [re.sub('[0-9]seed_', '', model) for model in benchmark_stats[self.seeds[0]].keys()]
        print(f'Stats over {len(benchmark_stats.keys())} seeds.')
        stat_types_dict = {stat_type: dict() for stat_type in [f'{endmodel}_best']}
        all_stats = {model: copy.deepcopy(stat_types_dict) for model in MODELS}
        mean_stats = copy.deepcopy(all_stats)
        st_dev_stats = copy.deepcopy(all_stats)
        for i, (seed, seed_stats) in enumerate(benchmark_stats.items()):
            for model_name, model_stats in seed_stats.items():
                try:
                    mlp_best_stats = model_stats[f"{endmodel}_best"]
                except:
                    try:
                        mlp_best_stats = model_stats["EndModel_best"]
                    except TypeError:
                        mlp_best_stats = model_stats[2][f"{endmodel}_best"]

                model_name = re.sub('[0-9]seed_', '', model_name)
                if i == 0:
                    all_stats[model_name][f'{endmodel}_best'] = {k: [v] for k, v in mlp_best_stats.items()}
                else:
                    for (m2, v2) in mlp_best_stats.items():
                        if m2 not in all_stats[model_name][f'{endmodel}_best']:
                            print('HM:', m2, model_name)
                            continue
                        all_stats[model_name][f'{endmodel}_best'][m2].append(v2)

        for model_name, models in all_stats.items():
            for model_type, metrics in models.items():
                for metr, stats in metrics.items():
                    stats_np = np.array(stats)
                    mean_stats[model_name][model_type][metr] = np.mean(stats_np)
                    st_dev_stats[model_name][model_type][metr] = np.std(stats_np).copy()

        self.stats = {"All": all_stats, "Mean": mean_stats, "Std": st_dev_stats}
        return self.stats

    def print_latex(self, endmodel='MLP', metrics=None):
        if metrics is None:
            metrics = ['f1']
        if not isinstance(metrics, list):
            metrics = [metrics]
        metrics = [m.lower() for m in metrics]
        model_type = f'{endmodel}_best'
        MODELS = [model for model in self.stats['All'].keys()]

        for model_name in MODELS:
            s = 'Test '
            avgs = self.stats['Mean'][model_name][model_type]
            stds = self.stats['Std'][model_name][model_type]
            if 'accuracy' in avgs:
                metric_avg, metric_std = avgs['accuracy'], stds['accuracy']
            if 'f1' in avgs:
                metric_avg_f1, metric_std_f1 = avgs['f1'], stds['f1']
            if 'auc' in avgs:
                metric_avg_auc, metric_std_auc = avgs['auc'], stds['auc']
            try:
                val_avg, val_std = avgs['validation_f1'] * 100, stds['validation_f1'] * 100
                name = 'F1'
            except KeyError:
                try:
                    val_avg, val_std = avgs['validation_val'], stds['validation_val']
                    name = 'AUC'
                except KeyError:
                    val_avg, val_std, name = -1, 0, '--'
            if 'f1' in metrics:
                s += r'F1: ${:.2f} \pm {:.2f}$ '.format(metric_avg_f1 * 100, metric_std_f1 * 100)
            if 'auc' in metrics:
                s += r'AUC:{:.3f} $\pm$ {:.2f} '.format(metric_avg_auc, metric_std_auc)
            if 'accuracy' in metrics:
                s += r'Acc:{:.2f} $\pm$ {:.2f} '.format(metric_avg * 100, metric_std * 100)
            # s_end += model_type + " "
            if val_avg != -1 and val_avg != -100:
                val_str = ":{:.2f}".format(val_avg) if name == 'F1' else ":{:.3f}".format(val_avg)
                s += f"| Validation {name}" + val_str
            print(s, f'<-- by {model_name}')


def get_dummies(Y_len):
    """
    For models that did not converge
    """
    probs = np.ones(Y_len) * 0.5
    xxx = {'accuracy': 0.5, 'f1': 0.5, 'recall': 0.5, 'precision': 0.5, "auc": 0.5, "MSE": 1.0, "MAE": 1.0,
           "tp": Y_len / 4, "tn": Y_len / 4, "fp": Y_len / 4, "fn": Y_len / 4, 'brier': 1.0, 'coverage': 1.0,
           'validation_f1': 0.5
           }
    stats = {"Gen": xxx.copy(), "MLP_end": xxx.copy(), 'MLP_best': xxx.copy()}
    return probs, stats
