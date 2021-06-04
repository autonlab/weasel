import copy
import json
import pickle
import time

import numpy as np
import wandb

from end_to_end_ws_model import E2ETrainer
from experiments.baselines import train_triplets, train_snorkel
from experiments.benchmark import Benchmarker, get_dummies
from utils.synthetic import synthetic_ablation
from utils.utils import set_seed


class BenchmarkerDeps(Benchmarker):
    def __init__(self, dirs, dataset, *args):
        super().__init__(seeds=[1, 2, 3, 4, 5], dirs=dirs, dataset=dataset)
        self.result_file = 'benchmarkSyntheticExp.pkl'
        self.synth_LFs = [2, 5, 25, 100, 500, 2000]

    def run_synth(self, Xtrain, Ytrain, Ytest, Xtest, encoder_params, hyper_params, end_params):
        benchmark_stats = {
            seed: {
                synthetic_LF_num: dict() for synthetic_LF_num in self.synth_LFs
            } for seed in self.seeds
        }

        for seed in self.seeds:
            set_seed(seed)
            hyper_params['seed'] = seed
            seed_id = str(seed) + 'seed_'
            for synthetic_LF_num in self.synth_LFs:
                run_id = str(synthetic_LF_num)
                L_run = synthetic_ablation(Ytrain, num_LFs=synthetic_LF_num + 1)  # synthetic LFs
                encoder_params['num_LFs'] = L_run.shape[1]
                trainloader, end_valloader, end_testloader = self.get_loaders(hyper_params, L_run, Xtrain, Xtest, Ytest)
                kwargs = {
                    'label_matrix': L_run, 'X': Xtrain, 'class_balance': encoder_params['class_balance'],
                    'end_valloader': end_valloader, 'end_testloader': end_testloader,
                    'mlp_params': end_params, 'hyper_params': hyper_params, 'dataset': self.dataset
                }
                print("L SHAPE", L_run.shape)
                print("NOW WITH", L_run.shape[1], 'LFs')

                wandb_init_dict = {
                    'project': f"E2E_{self.dataset}_DUPLICATION_OF_RANDOM_LFs",
                    'name': f"{L_run.shape[1]}LFs_{seed}seed" + '___' + time.strftime('%Hh%Mm_on_%b_%d'),
                    'config': {**hyper_params, **end_params, **encoder_params},
                    'reinit': True
                }

                # Snorkel and Triplets
                run = wandb.init(**wandb_init_dict, group='Snorkel')
                wandb.log({"num_LFs": encoder_params['num_LFs']})
                _, sn_probs, sn_stats = train_snorkel(**kwargs, Ytrain=Ytrain, wandb_run=run,
                                                      model_dir=self.model_dir + f'snorkel/{seed_id}{run_id}')
                benchmark_stats[seed][synthetic_LF_num]['snorkel'] = sn_stats


                try:
                    run = wandb.init(**wandb_init_dict, group='Triplets-Mean')
                    wandb.log({"num_LFs": encoder_params['num_LFs']})
                    _, _, tr_mean_stats = train_triplets(**kwargs, method='mean', wandb_run=run,
                                                         model_dir=self.model_dir + f'tripletMean/{seed_id}{run_id}')
                except Exception as e:
                    print(e, '\nTriplets-Mean did not converge!')
                    _, tr_mean_stats = get_dummies(sn_probs.shape[0])
                benchmark_stats[seed][synthetic_LF_num]['triplet_mean'] = tr_mean_stats

                ''' OURS '''
                ID = run_id
                run = wandb.init(**wandb_init_dict, group='E2E')
                print('***' * 5, self.dirs['ID'], ID)
                encoder, endmodel = self.get_models(end_params)
                endmodel_name = str(endmodel)
                e2e_dirs = {k: direc + f'End2EndDP{seed_id}seed{ID}/' for k, direc in self.dirs.items()}
                e2e_dirs['checkpoints'] = self.model_dir + f'End2EndDP/{seed_id}{ID}'
                trainer = E2ETrainer(encoder_params, encoder_net=encoder, downstream_model=endmodel, seed=seed,
                                     dirs=e2e_dirs)
                valid_f1 = trainer.fit(trainloader, end_valloader, hyper_params, testloader=end_testloader)
                _, _, end_stats = trainer.evaluate(end_testloader, prefix=f'E2E{seed_id} Test End {ID}:\n',
                                                   use_best_model=False, adjust_thresh=False)
                _, _, best_stats = trainer.evaluate(end_testloader, prefix=f'E2E{seed_id} Test Best {ID}:\n',
                                                    use_best_model=True, adjust_thresh=False)
                wandb.log({'Final Test F1': best_stats['f1'], 'Final Test AUC': best_stats['auc'],
                           'Final Test Prec.': best_stats['precision'], 'Final Test Rec.': best_stats['recall'],
                           f"Best Val {hyper_params['val_metric'].upper()}": valid_f1,
                           "num_LFs": encoder_params['num_LFs']}
                          )
                wandb.run.summary["f1"] = best_stats['f1']
                run.finish()
                best_stats['validation_val'] = valid_f1
                print(f"Best valid. {hyper_params['val_metric']}=", valid_f1)
                benchmark_stats[seed][synthetic_LF_num]['E2E'] = {f'{endmodel_name}_end': end_stats,
                                                                  f'{endmodel_name}_best': best_stats}

                with open(self.save_to + self.result_file, 'wb') as f:
                    pickle.dump(benchmark_stats, f, pickle.HIGHEST_PROTOCOL)

        print("---------------------------->         Saving all stats to ", self.save_to)
        with open(self.save_to + self.result_file, 'wb') as f:
            pickle.dump(benchmark_stats, f, pickle.HIGHEST_PROTOCOL)

        return benchmark_stats

    def analyze_seeded_runs(self, direc=None, metric='f1', plot=False):
        filename = self.save_to + self.result_file if direc is None else direc + self.result_file
        with open(filename, 'rb') as f:
            benchmark_stats = pickle.load(f)
        random_seed = list(benchmark_stats.keys())[0]
        MODELS = [model for model in benchmark_stats[random_seed][self.synth_LFs[0]].keys()]
        num_runs = max(benchmark_stats[random_seed].keys()) + 1  # +1 since we count from 0 on
        s = list(benchmark_stats[random_seed][0].keys())[0]
        stat_types_dict = {
            stat_type: {
                metric: [] for metric in benchmark_stats[random_seed][0][s][stat_type]
            }
            for stat_type in ['MLP_end', 'MLP_best']  # ['Gen', 'MLP_end', 'MLP_best']
        }
        all_stats = {model: copy.deepcopy(stat_types_dict) for model in MODELS}
        mean_stats = copy.deepcopy(all_stats)
        st_dev_stats = copy.deepcopy(all_stats)
        for i, (seed, seed_stats) in enumerate(benchmark_stats.items()):
            if len(seed_stats[0].keys()) == 0:
                continue
            if i == 0:
                for model in MODELS:
                    all_stats[model]['MLP_end'] = {metric: [] for metric in seed_stats[0][MODELS[0]]['MLP_end']}
                    all_stats[model]['MLP_best'] = {metric: [] for metric in seed_stats[0][MODELS[0]]['MLP_best']}
            for model in MODELS:
                # New row for each seed

                for metric in seed_stats[0][model]['MLP_end']:
                    all_stats[model]['MLP_end'][metric].append(np.zeros(num_runs))
                for metric in seed_stats[0][model]['MLP_best']:
                    all_stats[model]['MLP_best'][metric].append(np.zeros(num_runs))

            for j, (run, run_stats) in enumerate(seed_stats.items()):
                # print(all_stats['MV']['MLP_end']['auc'])
                for model_name, model_stats in run_stats.items():
                    mlp_end_stats = model_stats["MLP_end"]
                    mlp_best_stats = model_stats["MLP_best"]
                    for k, v in mlp_end_stats.items():
                        all_stats[model_name]['MLP_end'][k][i][j] = v
                    for k, v in mlp_best_stats.items():
                        all_stats[model_name]['MLP_best'][k][i][j] = v

        for model_name, models in all_stats.items():
            for model_type, model_type_stats in models.items():
                for metric_name, metrics in model_type_stats.items():
                    all_seeds_matrix = np.array(all_stats[model_name][model_type][metric_name])
                    all_stats[model_name][model_type][metric_name] = all_seeds_matrix
                    if model_type == 'Gen':  # and model_name not in self.GEN_MODELS:
                        continue
                    mean_stats[model_name][model_type][metric_name] = np.mean(all_seeds_matrix, axis=0)
                    st_dev_stats[model_name][model_type][metric_name] = np.std(all_seeds_matrix, axis=0)

        self.stats = {"All": all_stats, "Mean": mean_stats, "Std": st_dev_stats}
        if plot:
            self.plot_seeded_stats(metric=metric)
        return self.stats

    def plot_seeded_stats(self, metric='f1', show=True, model_types=None, relative=False):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        if model_types is None:
            model_types = ['MLP_best']
        MODELS = [model for model in self.stats['All'].keys()]
        fig, axs = plt.subplots(1)

        if not (isinstance(axs, list) or isinstance(axs, np.ndarray)):
            axs = [axs]

        colors = list(mcolors.BASE_COLORS)[:-1] + list(mcolors.TABLEAU_COLORS)
        for i, (ax, model_type) in enumerate(zip(axs, model_types)):
            for color, model_name in zip(colors, MODELS):

                avgs = self.stats['Mean'][model_name][model_type][metric]
                stds = self.stats['Std'][model_name][model_type][metric]

                label = model_name
                if metric == 'f1':
                    avgs *= 100
                    stds *= 100

                start_val = avgs[0]
                if relative:
                    avgs -= start_val

                if 'snorkel' in model_name.lower():
                    label = 'Snorkel'
                    color = 'darkturquoise'
                if model_name.strip() == 'triplet_mean':
                    label = 'Triplet-Mean'
                    color = 'orange'
                if model_name.strip() == 'E2E _STO' or 'e2e' in model_name.lower():
                    label = "E2E"
                    color = 'blue'
                else:
                    label = label.capitalize()

                xaxis = range(len(avgs))
                ax.errorbar(xaxis, avgs, color=color, yerr=stds, label=label)
                if 'E2E' in model_name:
                    ax.fill_between(xaxis, avgs - stds, avgs + stds, color=color, alpha=0.25)
                    ax.grid(axis='y')
                else:
                    ax.fill_between(xaxis, avgs - stds, avgs + stds, color=color, alpha=0.08)

            ax.set_ylabel(f'Downstream {metric.upper()} score')
            ax.set_xticks(xaxis)  # xaxis
            ax.xaxis.set_ticklabels(self.synth_LFs)  # change the ticks' names to x
            if relative:
                ax.set_ylim([-0.3, 0.05])  # np.max(to_plot['Stds'])])
            ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.1))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # print(to_plot['Means'])
        if show:
            plt.show()
        return ax
