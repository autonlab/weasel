import pickle
from experiments.baselines import train_snorkel
from experiments.benchmark import Benchmarker
from end_to_end_ws_model import E2ETrainer
from utils.utils import set_seed


class BenchmarkerLabelMe(Benchmarker):
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

            for loss in ['MIG', 'CE']:
                ID = loss + 'loss'
                hyper_params['loss'] = loss
                hyper_params['EncoderLoss'] = loss

                # Snorkel
                _, sn_probs, sn_stats = train_snorkel(**kwargs, Ytrain=Ytrain, cardinality=end_params['out_dim'],
                                                      num_valid_samples=200,
                                                      model_dir=self.model_dir + f'snorkel/{seed_id}{ID}')
                benchmark_stats[seed][f'snorkel_{ID}'] = sn_stats

                ''' OURS '''
                print('***' * 5, self.dirs['ID'], ID)
                encoder, endmodel = self.get_models(end_params)
                endmodel_name = str(endmodel)
                e2e_dirs = {k: direc + f'End2EndDP{seed_id}seed{ID}/' for k, direc in self.dirs.items()}
                e2e_dirs['checkpoints'] = self.model_dir + f'End2EndDP/{seed_id}run_{ID}'
                trainer = E2ETrainer(encoder_params, encoder_net=encoder, downstream_model=endmodel, seed=seed,
                                     dirs=e2e_dirs)
                valid_f1 = trainer.fit(trainloader, end_valloader, hyper_params, testloader=end_testloader)
                _, _, end_stats = trainer.evaluate(end_testloader, prefix=f'E2E{seed_id} Test End {ID}:\n',
                                                   use_best_model=False, adjust_thresh=False)
                _, _, best_stats = trainer.evaluate(end_testloader,
                                                    prefix=f'E2E{seed_id} Test Best {ID}:\n',
                                                    use_best_model=True, adjust_thresh=False)
                best_stats['validation_val'] = valid_f1
                print(f"Best valid. {hyper_params['val_metric']}=", valid_f1)
                benchmark_stats[seed][f'E2E_{ID}'] = {f'{endmodel_name}_end': end_stats,
                                                      f'{endmodel_name}_best': best_stats}

            with open(self.save_to + self.result_file, 'wb') as f:
                pickle.dump(benchmark_stats, f, pickle.HIGHEST_PROTOCOL)

        print("---------------------------->         Saving all stats to ", self.save_to)
        with open(self.save_to + self.result_file, 'wb') as f:
            pickle.dump(benchmark_stats, f, pickle.HIGHEST_PROTOCOL)

        return benchmark_stats
