import torch
import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import dgl
from scipy.sparse import bmat
from dgl import AddReverse
from dgl.data.utils import load_graphs
import pickle
from dgl import ToSimple
import os
from utils.util import identity_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


_DIRECTED = 'DIRECTED'
_UNDIRECTED = 'UNDIRECTED'


def load_meta_information(dataset, meta_info='meta_info.txt', l_info='l_info.txt', src_dir='./data/nets'):
    path = os.path.join(src_dir, dataset, meta_info)
    minfo = pd.read_csv(filepath_or_buffer=path, sep=' ')
    minfo.columns = minfo.columns.str.strip().str.upper()
    n_entity = int(minfo['N'][0])
    etype = minfo['E'][0]
    assert etype == _DIRECTED \
           or etype == _UNDIRECTED, f'Assertion Error: Unrecognized edge type.'
    num_layers = int(minfo['L'][0])

    path = os.path.join(src_dir, dataset, l_info)
    if os.path.isfile(path):
        linfo = pd.read_csv(filepath_or_buffer=path, sep=' ').to_numpy()
    else:
        linfo = build_p(num_layers)

    layers_id = []
    p = [None] * num_layers
    for i in range(num_layers):
        r = linfo[i]
        r = r[~np.isnan(r)].astype(np.int32).tolist()
        layers_id.append(r[0])
        p[r[0]-1] = r[1:]
    p = np.array(p)-1
    directed = True if etype == _DIRECTED else False
    mpx = True
    return n_entity, num_layers, directed, mpx, layers_id, p


def build_p(num_layers):
    lrs = [np.arange(1, num_layers + 1)]
    pxs = []
    for l_id in range(1, num_layers + 1):
        px = np.arange(1, num_layers + 1)
        px = px[px != l_id]
        pxs.append(px)
    pxs = np.vstack(pxs)
    linfo = np.vstack((lrs, pxs.T)).T
    return linfo


def load_netf(dataset, edges_name='net.edges', src_dir='./data/nets'):
    n_entity, n_el, directed, mpx, layers_id, p = load_meta_information(dataset, src_dir=src_dir)
    path = os.path.join(src_dir, dataset, edges_name)
    edges_mat = pd.read_csv(filepath_or_buffer=path, sep=' ', header=None).to_numpy(dtype=np.int32)
    edges_mat = edges_mat[:, 0:3].astype(np.int32)
    return edges_mat, n_entity, n_el, directed, mpx, layers_id, p


def load_features(path_dir='./data/nets', features='features.pt'):
    path = os.path.join(path_dir, features)
    if os.path.isfile(path):
        features = torch.load(path)
        return features
    return None


def get_random_negs(adj_sp, val_n, test_n, split_edge, directed=False, rei=True):
    if type(adj_sp) == dgl.DGLGraph:
        adj_sp = adj_sp.adj_external(scipy_fmt='csr')
    symmetrization = True
    adj = adj_sp
    if not directed:
        adj = adj_sp + adj_sp.T - sp.diags(adj_sp.diagonal())
        symmetrization = not symmetrization

    adj_neg = 1 - adj.todense() - sp.eye(adj.shape[0])
    if rei:  # remove negative samples including isolated nodes
        dim = adj_sp.shape[0]
        if type(split_edge) == dict:
            rows = split_edge['train']['edge'][:, 0]
            cols = split_edge['train']['edge'][:, 1]
        else:
            rows = split_edge[:, 0]
            cols = split_edge[:, 1]
        values = np.ones(rows.shape[0])
        degs = sp.csr_matrix((values, (rows, cols)), shape=(dim, dim)) #adj.copy()

        degs = sp.csgraph.laplacian(degs, return_diag=True, use_out_degree=True, symmetrized=symmetrization)[1]
        zd = np.where(degs == 0)[0]
        adj_negg = adj_neg.copy()
        adj_negg[zd] = .0
        adj_negg[:, zd] = .0

        nz = np.count_nonzero(adj_negg)
        if directed and nz >= (val_n + test_n):
            adj_neg = adj_negg
        elif not directed and nz/2 >= (val_n + test_n):
            adj_neg = adj_negg
        else:
            print(f'We are taking {test_n+val_n} from {np.count_nonzero(adj_neg)} possible samples')

    if not directed:
        adj_neg = np.triu(adj_neg)

    neg_u, neg_v = np.where(adj_neg != 0)
    pop = min(test_n+val_n, np.count_nonzero(adj_neg))
    neg_eids = np.random.choice(len(neg_u), pop, replace=False)
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_n]], neg_v[neg_eids[:test_n]]
    val_neg_u, val_neg_v = neg_u[neg_eids[test_n:]], neg_v[neg_eids[test_n:]]
    test_neg_edges = np.vstack((test_neg_u, test_neg_v)).T
    val_neg_edges = np.vstack((val_neg_u, val_neg_v)).T
    if split_edge is not None and type(split_edge) == dict:
        split_edge['test']['edge_neg'] = test_neg_edges

        split_edge['valid']['edge_neg'] = val_neg_edges

        return split_edge
    else:
        return test_neg_edges, val_neg_edges


def get_samplesk(g, directed, k=10, seed=72, negatives=True):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    x = torch.vstack(g.edges()).T.numpy()
    split = kf.split(x)
    fold = []
    for i, (train_index, test_index) in enumerate(split):

        split_edge = {'train': {}, 'test': {}, 'valid': {}}
        train = x[train_index]
        test = x[test_index]
        test, val = train_test_split(test, train_size=0.5, shuffle=True, random_state=seed,
                                     )

        split_edge['train']['edge'] = torch.tensor(train, dtype=torch.long)
        split_edge['test']['edge'] = torch.tensor(test, dtype=torch.long)
        split_edge['valid']['edge'] = torch.tensor(val, dtype=torch.long)

        if negatives:
            split_edge = get_random_negs(g.adj_external(scipy_fmt='csr'), val_n=len(val), test_n=len(test), split_edge=split_edge, directed=directed)
            split_edge['test']['edge_neg'] = torch.tensor(split_edge['test']['edge_neg'], dtype=torch.long)
            split_edge['valid']['edge_neg'] = torch.tensor(split_edge['valid']['edge_neg'], dtype=torch.long)

        fold.append(split_edge)
    return fold



def build_supra_adj(a, p, directed):
    values = []
    n_layers = len(a)
    for i, ai in enumerate(a):
        degs = sp.csgraph.laplacian(ai, return_diag=True, use_out_degree=False, symmetrized=True)[1]
        if not directed:
            degs = degs / 2.0
        zx = np.where(degs == 0)[0]
        dim = ai.shape[0]
        value = torch.ones(dim)
        value[zx] = .0
        values.append(value)

    blocks = [[None for _ in range(n_layers)] for _ in range(n_layers)]
    for i in range(n_layers):
        blocks[i][i] = a[i]
        for j in p[i]:  # solo strati confrontabili
            if j == -1 or j is None:
                continue
            vi = values[i]
            vj = values[j]
            vij = (vi*vj).numpy()
            inter = sp.diags(vij)
            blocks[i][j] = inter

    supra_adj = bmat(blocks)
    return supra_adj


def build_identity_matrix(n, n_layers):
    features = identity_matrix(n, sparse=False).unsqueeze(0)
    features = list(features.expand(n_layers, n, n))
    features = torch.vstack(features)
    return features


def load_data(dataset, no_supra=False, prep_dir='./data/prep_nets/', run=0):
    prep_path = os.path.join(prep_dir, dataset)

    if os.path.isfile(os.path.join(prep_path, 'g_train', f'g_train_{run}.bin')):
        print(f'Loading dataset from {prep_path} for fold {run}')
        g_supra = load_graphs(os.path.join(prep_path, 'g_supra', f'g_supra_{run}.bin'))[0][0]
        g_train = load_graphs(os.path.join(prep_path, 'g_train', f'g_train_{run}.bin'))[0]
        g_train_pos = load_graphs(os.path.join(prep_path, 'g_train_pos', f'g_train_pos_{run}.bin'))[0]
        g_test_pos = load_graphs(os.path.join(prep_path, 'g_test_pos', f'g_test_pos_{run}.bin'))[0]
        g_test_neg = load_graphs(os.path.join(prep_path, 'g_test_neg',  f'g_test_neg_{run}.bin'))[0]
        g_val_pos = load_graphs(os.path.join(prep_path, 'g_val_pos', f'g_val_pos_{run}.bin'))[0]
        g_val_neg = load_graphs(os.path.join(prep_path, 'g_val_neg', f'g_val_neg_{run}.bin'))[0]

        print(f'Loading meta information from ', os.path.join(prep_path,  'n_info.pkl'))
        info_name = 'n_info.pkl'
        with open(os.path.join(prep_path, info_name), 'rb') as f:
            n_info = pickle.load(f)
        n, n_layers, directed, _, _, _ = n_info
        print(f'Loading training-test-validation split information from ', os.path.join(prep_path, 'split_edges',
                                                                                        f'split_edges_{run}.pkl'))
        with open(os.path.join(prep_path, 'split_edges', f'split_edges_{run}.pkl'), 'rb') as f:
            split_edges = pickle.load(f)
        features = None
        if not no_supra:

            name_f = 'features.pt'
            print(f'Loading features from {prep_path}')
            features = load_features(prep_path, features=name_f)
            if features is None:
                print(f'Real features not found. Using identity features!')
                features = build_identity_matrix(n_info[0], n_info[1])
            else:
                print(f'Real features loaded successfully!')

        return g_supra, g_train, g_train_pos, g_test_pos, g_test_neg, g_val_pos, g_val_neg, features, split_edges, \
               n_info
    else:
         raise ValueError('Data not found!')


def lab_kdata(dataset, no_supra=False,
              edges_name='net.edges', src_dir='./data/nets', weight_norm=False,
              add_self_loop=False, kfold=10):

    net, n, n_layers, directed, mpx, layers_id, p = load_netf(dataset, edges_name, src_dir=src_dir)

    edges = []
    for li in range(n_layers):
        edges.append(net[net[:, 0] == li + 1])
        edges[-1] = edges[-1][:, 1:].T

    net_mono = net[:, 1:3]  # all edges

    aggregator = 'sum'

    transform = ToSimple(aggregator=aggregator, return_counts='w')
    if not directed:
        g_rep = nx.from_edgelist(net_mono, create_using=nx.Graph)
        mono_edges = np.array(g_rep.edges())
        g_rep = dgl.graph((mono_edges[:, 0], mono_edges[:, 1]))

    else:
        rows = torch.tensor(net_mono[:, 0])
        cols = torch.tensor(net_mono[:, 1])

        g_rep = dgl.graph((rows, cols), idtype=torch.int64)

    g_mono = transform(g_rep)

    g_mono.edata['w'] = torch.ones(g_mono.edata['w'].shape)

    folds = get_samplesk(g_mono, directed=directed, k=kfold, negatives=True)

    g_supra_k, g_train_k, g_train_pos_k, g_test_pos_k, g_test_neg_k,\
        g_val_pos_k, g_val_neg_k = [], [], [], [], [], [], []
    features = None
    for kf, split_edge in enumerate(folds):
        g_train = []
        g_train_pos = []
        g_test_pos = []
        g_test_neg = []
        g_val_pos = []
        g_val_neg = []
        adjs = []
        dim = n
        train_edges_k = split_edge['train']['edge']  # train edges
        test_edges_k = split_edge['test']['edge']  # test edges
        valid_edges_k = split_edge['valid']['edge']  # validation edges

        train_adj = sp.csr_matrix((np.ones(train_edges_k.shape[0]), (train_edges_k[:, 0], train_edges_k[:, 1])),
                                 shape=(dim, dim))
        test_adj = sp.csr_matrix((np.ones(test_edges_k.shape[0]), (test_edges_k[:, 0], test_edges_k[:, 1])),
                                 shape=(dim, dim))
        valid_adj = sp.csr_matrix((np.ones(valid_edges_k.shape[0]), (valid_edges_k[:, 0], valid_edges_k[:, 1])),
                                 shape=(dim, dim))

        if not directed:
            train_adj = train_adj + train_adj.T - sp.diags(train_adj.diagonal())
            test_adj = test_adj + test_adj.T - sp.diags(test_adj.diagonal())
            valid_adj = valid_adj + valid_adj.T - sp.diags(valid_adj.diagonal())
            train_adj = sp.triu(train_adj)
            test_adj = sp.triu(test_adj)
            valid_adj = sp.triu(valid_adj)

        for li in range(n_layers):
            rows = edges[li][0]
            cols = edges[li][1]
            values = np.ones(rows.shape[0])

            adj_sp = sp.csr_matrix((values, (rows, cols)), shape=(dim, dim))
            # projection
            if not directed:
                adj_sp = adj_sp + adj_sp.T - sp.diags(adj_sp.diagonal())
                adj_sp = sp.triu(adj_sp)
            g = dgl.from_scipy(adj_sp, eweight_name='w')
            adj_sp = g.adj_external(scipy_fmt='csr')
            train_edges = np.vstack(adj_sp.multiply(train_adj).nonzero()).T
            temp_g = dgl.graph((train_edges[:, 0], train_edges[:, 1]), num_nodes=n)
            candidates = temp_g.in_degrees() + temp_g.out_degrees()
            candidates = torch.where(candidates > 0)[0]

            def filtering(edg, can):
                return edg[0] in can and edg[1] in can
            test_edges_c = np.vstack(adj_sp.multiply(test_adj).nonzero()).T
            valid_edges_c = np.vstack(adj_sp.multiply(valid_adj).nonzero()).T
            if test_edges_c.size == 0:
                test_edges = test_edges_c
            else:
                test_idx = np.array([filtering(row, candidates) for row in test_edges_c])
                test_edges = test_edges_c[test_idx]
            if valid_edges_c.size == 0:
                valid_edges = valid_edges_c
            else:
                valid_idx = np.array([filtering(row, candidates) for row in valid_edges_c])
                valid_edges = valid_edges_c[valid_idx]
            test_neg_edges, valid_neg_edges = get_random_negs(adj_sp, val_n=valid_edges.shape[0],
                                                              test_n=test_edges.shape[0], split_edge=train_edges,
                                                              directed=directed)

            assert test_edges.shape[0] == test_neg_edges.shape[0]
            assert valid_edges.shape[0] == valid_neg_edges.shape[0]

            train_g = dgl.graph((train_edges[:, 0], train_edges[:, 1]), num_nodes=dim)

            if not directed:
                transform = ToSimple(aggregator='arbitrary', return_counts='w')
                transform_rev = AddReverse(copy_edata=True)
                train_g = transform(transform_rev(train_g))
                train_g.edata.pop(dgl.NID)
            else:
                train_g.edata['w'] = torch.ones(train_g.number_of_edges(), dtype=torch.float32)

            train_pos_g = dgl.graph((train_edges[:, 0], train_edges[:, 1]), num_nodes=dim)

            adjs.append(train_g.adj_external(scipy_fmt='csr'))
            train_g.edata['w'] = train_g.edata['w'].float()
            if weight_norm:
                norm = dgl.nn.EdgeWeightNorm(norm='both')
                norm_edge_weight = norm(train_g, train_g.edata['w'])
                train_g.edata['w'] = norm_edge_weight
            if add_self_loop:
                train_g = dgl.add_self_loop(train_g, edge_feat_names='w', fill_data=1.0)

            g_train.append(train_g)

            g_train_pos.append(train_pos_g)

            test_pos_g = dgl.graph((test_edges[:, 0], test_edges[:, 1]), num_nodes=dim)
            test_neg_g = dgl.graph((test_neg_edges[:, 0], test_neg_edges[:, 1]), num_nodes=dim)
            g_test_pos.append(test_pos_g)
            g_test_neg.append(test_neg_g)

            val_pos_g = dgl.graph((valid_edges[:, 0], valid_edges[:, 1]), num_nodes=dim)
            val_neg_g = dgl.graph((valid_neg_edges[:, 0], valid_neg_edges[:, 1]), num_nodes=dim)
            g_val_pos.append(val_pos_g)
            g_val_neg.append(val_neg_g)

        if not no_supra:
            supra_adj = build_supra_adj(adjs, p, directed)
            g_supra = dgl.from_scipy(supra_adj)
            g_supra = dgl.add_self_loop(g_supra)

        else:
            g_supra = None

        if features is None:

            f_name = 'features.pt'
            features = load_features(os.path.join(src_dir, dataset), features=f_name)
        g_supra_k.append(g_supra)
        g_train_k.append(g_train)
        g_train_pos_k.append(g_train_pos)
        g_test_pos_k.append(g_test_pos)
        g_test_neg_k.append(g_test_neg)
        g_val_pos_k.append(g_val_pos)
        g_val_neg_k.append(g_val_neg)

    if features is None:
        print('Real features not found!')
    n_info = (n, n_layers, directed, mpx, layers_id, p)
    return g_supra_k, \
        g_train_k, g_train_pos_k, g_test_pos_k, g_test_neg_k, g_val_pos_k, g_val_neg_k, \
        features, folds, n_info
