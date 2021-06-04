import argparse
import json
import os
import time
import numpy as np
import torch
from torchsummary import summary

from utils.data_loading import get_LabelMe_data
from utils.utils import load_model, set_seed

datasets = ['amazon', 'imdb', 'labelme', 'professor_teacher', 'spouses']


def get_dataset_name_and_nlfs(dataset_name, num_LFs=None):
    dname = dataset_name.lower()
    assert dname in datasets, f'Please provide a valid dataset name like {datasets}'
    if dname == 'amazon':
        return 'Amazon', dname, 175
    if dname == 'imdb':
        if num_LFs is None or num_LFs not in [12, 136]:
            print('Number of LFs was not specified or invalid for IMDB, defaulting to 12.')
            num_LFs = 12
        return 'IMDB', dname, num_LFs
    if dname == 'professor_teacher':
        return 'professor_teacher', dname, 99
    if dname == 'spouses':
        return 'Spouses', dname, 9
    if dname == 'labelme':
        return 'LabelMe', dname, 59


def setup_and_get_params(dataset_name=None, num_LFs=None, notebook_mode=False, reload_mode=False, verbose=True,
                         prefix=''):
    parser = get_argparser()
    args = parser.parse_args(args=[]) if notebook_mode else parser.parse_args()

    if args.dataset is not None:
        dataset_name = args.dataset
    if args.num_LFs is not None:
        num_LFs = args.num_LFs
    dataset_name, dname, num_LFs = get_dataset_name_and_nlfs(dataset_name, num_LFs)

    # print(os.getcwd())
    with open(f"configs/{dname}{str(num_LFs)}.json") as f:
        config = json.load(f)
        if verbose:
            print("Dataset:", config["dataset"])
    model_params, down_params, hyper_params = config["model_params"], config["end_params"], config["hyper_params"]
    MODEL_NAME = config["model_name"]
    device = gpu_setup(config, args.gpu_id, verbose=verbose)
    out_dir = 'benchmark_runs/'
    set_seed(hyper_params['seed'])
    hyper_params['notebook_mode'] = notebook_mode

    if dname in ['spouses']:
        data_dict = {'L': None, 'Xtrain': None, 'Xtest': None, 'Ytest': None}
    elif dname == 'labelme':
        Xtrain, Ytrain, Xtest, Ytest, L_arr = get_LabelMe_data()
        data_dict = {'L': L_arr, 'Xtrain': Xtrain, 'Xtest': Xtest, 'Ytest': Ytest, 'Ytrain_gold': Ytrain}
    else:
        data_dict = np.load(f'data/{dataset_name}_{num_LFs}LFs.npz')
        label_matrix = data_dict['L']  # weak source votes
        assert label_matrix.shape[1] == num_LFs
        num_samples, num_LFs = label_matrix.shape
        model_params['num_LFs'] = num_LFs

    process_parsed_args(args, model_params, down_params, hyper_params)
    print(model_params)
    dirs = set_logger_paths(out_dir, model_params, hyper_params, gpu_id=config['gpu']['id'], prefix=prefix,
                            MODEL_NAME=MODEL_NAME, DATASET_NAME=dataset_name, reload_mode=reload_mode)

    # total_params = view_model_param(MODEL_NAME, model_params, hyper_params['batch_size'], device, verbose, down_params)

    param_dict = {
        "model_params": model_params, "end_params": down_params, "hyper_params": hyper_params,
        "device": device, "model_name": config['model_name'], "dataset": dataset_name, "total_params": 0
    }
    if not reload_mode:
        with open(dirs['out'] + 'config.json', 'w') as f:  # save the params for reproducibility
            json.dump(param_dict, f)

    return data_dict, param_dict, dirs


def get_argparser():
    parser = argparse.ArgumentParser(description=f'Neural Weak Supervision')
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', type=str, help="Please give a value for out_dir")
    parser.add_argument('--stochastic', help="Please give a value for out_dir")
    parser.add_argument('--seed', type=int, help="Please give a value for seed")
    parser.add_argument('--loss', type=str, help="Please give a value for loss, e.g. BCE or L1")
    parser.add_argument('--encLoss', type=str, help="Please give a value for loss, e.g. BCE or L1")
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--scheduler', type=str)
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--dropout', type=float, help='dropout rate')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--temperature', type=float, default=1.0, help='batch size')
    parser.add_argument('--weight_decay', type=float, help='weight decay rate')
    parser.add_argument('--mlp_weight_decay', type=float, help='weight decay rate')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, help="Please give a value for hidden_dim")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--accuracy_func', type=str, help="Please give a value for Accuracy function")
    parser.add_argument('--label_func', type=str, help="Please give a value for label function")
    parser.add_argument('--accuracy_scaler', help="Please give a value for scaling the accuracy params")
    parser.add_argument('--neg_label', type=int, help="Please give a value for negative label")
    parser.add_argument('--num_LFs', type=int, help='Number of LFs (to subsample)')
    parser.add_argument('--n_features', type=int, help='Number of training features')
    parser.add_argument('--valset_size', type=int, help='Number of training features')
    parser.add_argument('--val_metric', type=str, help='Which metric to use for early-stopping (higher->better needed)')

    parser.add_argument('--use_features_for_enc', help='Number of training features')
    parser.add_argument('--adjust_thresh', help='whether to adjust the decision threshold based on validation set')

    parser.add_argument('--mlp_L', type=int, help="Please give a value for #MLP layers")
    parser.add_argument('--mlp_dropout', type=float, help='Number of MLP training epochs')
    parser.add_argument('--mlp_hidden_dim', type=int, help='Number of MLP training epochs')
    parser.add_argument('--mlp_out_dim', type=int, help='Number of MLP training epochs')
    parser.add_argument('--mlp_out_func', type=str, help='Number of MLP training epochs')
    return parser


def process_parsed_args(args, model_params, mlp_params, hyper_params):
    set_args(args.mlp_weight_decay, hyper_params, float, "mlp_weight_decay")
    set_args(args.adjust_thresh, hyper_params, bool, "adjust_thresh")
    set_args(args.weight_decay, hyper_params, float, "weight_decay")
    set_args(args.valset_size, hyper_params, int, "valset_size")
    set_args(args.val_metric, hyper_params, str, "val_metric")
    set_args(args.batch_size, hyper_params, int, "batch_size")
    set_args(args.scheduler, hyper_params, str, "scheduler")
    set_args(args.encLoss, hyper_params, str, "EncoderLoss")
    set_args(args.epochs, hyper_params, int, "epochs")
    set_args(args.optim, hyper_params, str, "optim")
    set_args(args.seed, hyper_params, int, "seed")
    set_args(args.loss, hyper_params, str, "loss")
    set_args(args.lr, hyper_params, float, "lr")

    set_args(args.accuracy_scaler, model_params, [float, str], "accuracy_scaler")
    set_args(args.use_features_for_enc, model_params, bool, "use_features_for_enc")
    set_args(args.accuracy_func, model_params, str, "accuracy_func")
    set_args(args.temperature, model_params, float, "temperature")
    set_args(args.batch_norm, model_params, bool, "batch_norm")
    set_args(args.label_func, model_params, str, "label_func")
    set_args(args.neg_label, model_params, int, "neg_label")
    set_args(args.dropout, model_params, float, "dropout")
    set_args(args.num_LFs, model_params, int, "num_LFs")

    set_args(args.mlp_hidden_dim, mlp_params, int, "hidden_dim")
    set_args(args.mlp_dropout, mlp_params, float, "dropout")
    set_args(args.mlp_out_func, mlp_params, str, "out_func")
    set_args(args.n_features, mlp_params, int, "input_dim")
    set_args(args.mlp_out_dim, mlp_params, int, "out_dim")
    set_args(args.mlp_L, mlp_params, int, "L")


def gpu_setup(config, args_gpu_id, verbose=True):
    if args_gpu_id is not None:  # device
        config['gpu']['id'] = int(args_gpu_id)
        config['gpu']['use'] = True
    gpu_id = config['gpu']['id']
    set_gpu(gpu_id)

    if torch.cuda.is_available() and config['gpu']['use']:
        if verbose:
            print('cuda available with GPU:', torch.cuda.get_device_name(0), "ID =", args_gpu_id)
        device = "cuda"  # torch.device("cuda")
    else:
        if verbose:
            print('cuda not available')
        device = "cpu"  # torch.device("cpu")
    return device


def set_gpu(gpu_id):
    gpu_id = int(gpu_id) if gpu_id is not None else 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def set_logger_paths(out_dir, model_params, hyperparams, gpu_id, MODEL_NAME, DATASET_NAME, reload_mode=False,
                     prefix=''):
    def prefix_args():
        s = f"{model_params['num_LFs']}nLF_"
        s += f"{model_params['accuracy_func'].upper()}accFunc_"
        s += f"{model_params['temperature']}temp_"
        s += f"{hyperparams['lr']}lr_"
        s += f"{hyperparams['batch_size']}bs_"
        s += f"{hyperparams['epochs']}eps_"
        s += f"{str(model_params['accuracy_scaler'])}accsc_"
        s += f"{len(model_params['encoder_dims'])}encL_"
        s += f"ADJcutoff_" if hyperparams['adjust_thresh'] else "FIXEDcutoff_"
        s += f"{hyperparams['scheduler']}_"
        s += f"{hyperparams['val_metric']}metric_"
        if hyperparams['weight_decay'] > 0:
            s += f"{hyperparams['weight_decay']}wd_"
        if model_params['dropout'] > 0:
            s += f"{model_params['dropout']}dout_"
        s += f"{hyperparams['loss']}Floss_{hyperparams['EncoderLoss']}Eloss_"
        # if not use_scheduler:
        #    s += "NoSchedule_"
        s += f"{int(hyperparams['seed'])}init-seed_"
        return s

    suffix = "_GPU" + str(gpu_id) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    pref = DATASET_NAME + prefix + prefix_args() + suffix
    out_dir = out_dir + pref + '/'
    log_dir = out_dir + 'logs/'
    ckpt_dir = out_dir + 'checkpoints/'
    write_file_name = out_dir + 'results/'
    dirs = {"logging": log_dir, "checkpoints": ckpt_dir, "results": write_file_name, "out": out_dir, 'ID': pref}

    if reload_mode:  # do not create directories
        return dirs
    if not os.path.exists(write_file_name):
        os.makedirs(write_file_name)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    models = ['snorkel', 'majority', 'triplet', 'tripletMean', "tripletMedian", 'End2EndDP', 'supervised', 'MaxMIG']
    for m in models:
        if not os.path.exists(ckpt_dir + m + '/'):
            os.makedirs(ckpt_dir + m + '/')

    return dirs


"""
    VIEWING MODEL CONFIG AND PARAMS
"""


def view_model_param(MODEL_NAME, net_params, batch_size, device='cuda', verbose=True, *args):
    model = load_model(MODEL_NAME)(net_params, *args).to(device)
    total_param = 0
    if verbose:
        print("MODEL DETAILS:")
        try:
            summary(model.encoder, (model.input_len,), batch_size)
        except RuntimeError:
            pass
    # for param in model.parameters():
    # print(param.data.size())
    #   total_param += np.prod(list(param.data.train_size()))
    if verbose:
        print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return int(total_param)


def set_args(value, dictio, valType, name):
    if value is None:
        return dictio[name] if name in dictio else None

    if valType == bool:
        dictio[name] = True if value in ['True', True] else False
    elif isinstance(valType, list) or isinstance(valType, tuple):
        try:
            dictio[name] = valType[0](value)
        except:
            dictio[name] = valType[1](value)
    else:
        dictio[name] = valType(value)

    return dictio[name]
