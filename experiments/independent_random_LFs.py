import copy
import json
import pickle
import time

import numpy as np
import wandb

from experiments.benchmark import get_dummies
from experiments.duplicated_LFs import BenchmarkerDeps
from end_to_end_ws_model import E2ETrainer
from experiments.baselines import train_triplets, train_snorkel
from utils.synthetic import random_LF
from utils.utils import set_seed


class BenchmarkerRandomLFs(BenchmarkerDeps):
    def __init__(self, dirs, dataset, *args):
        super().__init__(dirs, dataset, *args)
        #  number of random LFs
        self.synth_LFs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 75, 100]

    def run_synth(self, Xtrain, Ytrain, Ytest, Xtest, encoder_params, hyper_params, end_params):
        benchmark_stats = {
            seed: {
                n_random_LFs: dict() for n_random_LFs in self.synth_LFs
            } for seed in self.seeds
        }
        n_samples, classes = Ytrain.shape[0], list(np.unique(Ytrain))
        for seed in self.seeds:
            set_seed(seed)
            hyper_params['seed'] = seed
            seed_id = str(seed) + 'seed_'
            initial_random_LF = random_LF(n_samples, classes, abstain_too=False)  # synthetic LF
            L_run = np.concatenate((Ytrain.copy().reshape((-1, 1)),
                                    initial_random_LF.reshape((-1, 1))), axis=1)
            for n_random_LFs in self.synth_LFs:
                run_id = n_random_LFs
                while True:
                    cur_random_LFs = L_run.shape[1] - 1
                    if cur_random_LFs >= n_random_LFs:
                        break
                    new_random_LF = random_LF(n_samples, classes, abstain_too=False)  # synthetic LF
                    L_run = np.concatenate((L_run, new_random_LF.reshape((-1, 1))), axis=1)
                    # print(n_random_LFs, np.count_nonzero(initial_random_LF == new_random_LF))
                encoder_params['num_LFs'] = L_run.shape[1]
                hyper_params['epochs'] = 100

                trainloader, end_valloader, end_testloader = self.get_loaders(hyper_params, L_run, Xtrain, Xtest, Ytest)
                kwargs = {
                    'label_matrix': L_run, 'X': Xtrain, 'class_balance': encoder_params['class_balance'],
                    'end_valloader': end_valloader, 'end_testloader': end_testloader,
                    'mlp_params': end_params, 'hyper_params': hyper_params, 'dataset': self.dataset
                }
                print("L SHAPE", L_run.shape)
                print("NOW WITH", L_run.shape[1], 'LFs')

                wandb_init_dict = {
                    'project': f"E2E_{self.dataset}Y*_plus_INDEPENDENT_RANDOM_LFs",
                    'name': f"{L_run.shape[1]}LFs_{seed}seed" + '___' + time.strftime('%Hh%Mm_on_%b_%d'),
                    'config': {**hyper_params, **end_params, **encoder_params},
                    'reinit': True
                }

                # Snorkel and Triplets

                run = wandb.init(**wandb_init_dict, group='Snorkel')
                wandb.log({"num_LFs": encoder_params['num_LFs'], "num_random_LFs": n_random_LFs})
                _, sn_probs, sn_stats = train_snorkel(**kwargs, Ytrain=Ytrain, wandb_run=run,
                                                      model_dir=self.model_dir + f'snorkel/{seed_id}{run_id}')
                benchmark_stats[seed][run_id]['snorkel'] = sn_stats

                try:
                    run = wandb.init(**wandb_init_dict, group='Triplets-Mean')
                    wandb.log({"num_LFs": encoder_params['num_LFs'], "num_random_LFs": n_random_LFs})
                    _, _, tr_mean_stats = train_triplets(**kwargs, method='mean', wandb_run=run,
                                                         model_dir=self.model_dir + f'tripletMean/{seed_id}{run_id}')
                except Exception as e:
                    print(e, '\nTriplets-Mean did not converge!')
                    _, tr_mean_stats = get_dummies(L_run.shape[0])

                # benchmark_stats[seed][synthetic_LF_num]['triplet'] = tr_stats
                benchmark_stats[seed][run_id]['triplet_mean'] = tr_mean_stats

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
                           "num_LFs": encoder_params['num_LFs'], "num_random_LFs": n_random_LFs}
                          )
                wandb.run.summary["f1"] = best_stats['f1']
                run.finish()
                best_stats['validation_val'] = valid_f1
                print(f"Best valid. {hyper_params['val_metric']}=", valid_f1)
                benchmark_stats[seed][run_id][f'E2E'] = {f'{endmodel_name}_end': end_stats,
                                                                        f'{endmodel_name}_best': best_stats}

                with open(self.save_to + self.result_file, 'wb') as f:
                    pickle.dump(benchmark_stats, f, pickle.HIGHEST_PROTOCOL)

        print("---------------------------->         Saving all stats to ", self.save_to)
        with open(self.save_to + self.result_file, 'wb') as f:
            pickle.dump(benchmark_stats, f, pickle.HIGHEST_PROTOCOL)

        return benchmark_stats