def log_epoch_vals(writer, stats, epoch, dataset):
    if writer is None:
        return
    writer.add_scalar(f'{dataset}/_accuracy', stats['accuracy'], epoch)
    if 'precision' in stats and 'recall' in stats:
        writer.add_scalar(f'{dataset}/_precision', stats['precision'], epoch)
        writer.add_scalar(f'{dataset}/_recall', stats['recall'], epoch)
        writer.add_scalar(f'{dataset}/_f1', stats['f1'], epoch)
        writer.add_scalar(f'{dataset}/_auc', stats['auc'], epoch)
    elif 'f1_micro' in stats:  # multi-class
        writer.add_scalar(f'{dataset}/_f1_micro', stats['f1_micro'], epoch)
        writer.add_scalar(f'{dataset}/_f1_macro', stats['f1_macro'], epoch)
    if 'brier' in stats:
        writer.add_scalar(f'{dataset}/_brier', stats['brier'], epoch)
    if 'MSE' in stats:
        writer.add_scalar(f'{dataset}/_mse', stats['MSE'], epoch)
    if 'MAE' in stats:
        writer.add_scalar(f'{dataset}/_mae', stats['MAE'], epoch)

    # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

def set_if_exists(dictio_from, dictio_to, key, prefix):
    if key in dictio_from:
        dictio_to[f'{prefix}_{key}'.lstrip('_')] = dictio_from[key]

def update_tqdm(tq, train_loss, val_stats=None, test_stats=None, **kwargs):
    def get_stat_dict(dictio, prefix, all=False):
        d = dict()
        set_if_exists(dictio, d, 'accuracy', prefix)
        set_if_exists(dictio, d, 'f1', prefix)
        set_if_exists(dictio, d, 'auc', prefix)
        set_if_exists(dictio, d, 'brier', prefix)
        set_if_exists(dictio, d, 'f1_micro', prefix)
        set_if_exists(dictio, d, 'f1_macro', prefix)

        if all:
            set_if_exists(dictio, d, 'precision', prefix)
            set_if_exists(dictio, d, 'recall', prefix)
        return d

    if val_stats is None:
        if test_stats is None:
            tq.set_postfix(train_loss=train_loss, **kwargs)
        else:
            test_print = get_stat_dict(test_stats, 'test')
            tq.set_postfix(train_loss=train_loss, **test_print, **kwargs)
    else:
        val_print = get_stat_dict(val_stats, 'val', all=True)
        if test_stats is None:
            tq.set_postfix(train_loss=train_loss, **val_print, **kwargs)
        else:
            test_print = get_stat_dict(test_stats, 'test')
            tq.set_postfix(train_loss=train_loss, **val_print, **test_print, **kwargs)
