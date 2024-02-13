from dgl.data.utils import save_graphs
from input_data.load import lab_kdata
import argparse
import os
import torch
import pickle
from pathlib import Path
from utils.d_util import print_arguments
from utils.util import init_seed


def set_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ckm")
    parser.add_argument('--dst', type=str, default='./data/prep_nets', help='Destination folder where preprocessed data are saved.')
    parser.add_argument('--src', type=str, default='./data/nets/', help='Folder where input data are stored.')
    parser.add_argument('--seed', type=int, default=72,  help='Random seed.')
    parser.add_argument('--fold', type=int, default=10, help='Number of folds for K-fold cross validation.')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':

    args = set_params()
    dataset = args.dataset
    src_d = args.src
    dst_d = args.dst
    seed = int(args.seed)
    init_seed(seed)

    print_arguments(args)
    print()

    sp = os.path.join(dst_d, dataset)
    sp2 = os.path.join(dst_d, dataset)
    Path(sp).mkdir(parents=True, exist_ok=True)
    lf = ['g_supra', 'g_train', 'g_train_pos', 'g_test_pos', 'g_test_neg', 'g_val_pos', 'g_val_neg', 'split_edges']

    for f in lf:
        Path(os.path.join(sp, f)).mkdir(parents=True, exist_ok=True)
    k = args.fold
    g_supra, g_train, g_train_pos, g_test_pos, g_test_neg, g_val_pos, g_val_neg, feats, \
        folds, n_info = \
        lab_kdata(dataset, no_supra=False, src_dir=src_d, kfold=k)

    n, n_layers, directed, mpx, _, p = n_info
    dim = n

    with open(os.path.join(sp2, 'n_info.pkl'), 'wb') as f:
        pickle.dump(n_info, f)

    for kf in range(k):
        gss = [g_supra[kf], g_train[kf], g_train_pos[kf], g_test_pos[kf], g_test_neg[kf],
               g_val_pos[kf], g_val_neg[kf]]

        split_edge = folds[kf]
        r_train = split_edge['train']['edge'].shape[0]
        r_valid = split_edge['valid']['edge'].shape[0]
        r_test = split_edge['test']['edge'].shape[0]

        train_edges = sum([g.number_of_edges() for g in g_train_pos[kf]])
        test_edges = sum([g.number_of_edges() for g in g_test_pos[kf]])
        val_edges = sum([g.number_of_edges() for g in g_val_pos[kf]])
        test_neg_edges = sum([g.number_of_edges() for g in g_test_neg[kf]])
        val_neg_edges = sum([g.number_of_edges() for g in g_val_neg[kf]])
        assert test_edges == test_neg_edges and val_edges == val_neg_edges

        for idx, f in enumerate(lf[:-1]):
            save_graphs(os.path.join(sp, f, f'{f}_{kf}.bin'), gss[idx])

        with open(os.path.join(sp, lf[-1], f'split_edges_{kf}.pkl'), 'wb') as f:
            pickle.dump(folds[kf], f)

    print('Preprocessed data saved in ', sp)

    name_f = f'features.pt'
    if feats is not None:
        print(f'Saving features in {os.path.join(sp2, name_f)}; Size : {feats.shape}')
        torch.save(feats, os.path.join(sp2, name_f))






