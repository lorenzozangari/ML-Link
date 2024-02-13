import random, os
import numpy as np
import pandas as pd
import networkx as nx
import argparse
import dgl
from pathlib import Path


def set_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--nodes', type=int, default=500, help='Number of nodes per layer.')
    parser.add_argument('--beta', type=float, default=0.1, help='Rewiring probability.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')

    args, _ = parser.parse_known_args()
    return args


def init_seed(seed=72):
    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def make_watts_strogatz(n, n_layers, beta):
    graphs = []
    k = int(n*0.5/100)
    k_step = 5
    ks = []
    for lid in range(n_layers):
      ks.append(k + k_step*lid)
    for lid in range(n_layers):
      g = nx.watts_strogatz_graph(n, ks[lid], beta)
      graphs.append(g)

    return graphs


def get_edges(graphs):
    n_layers = len(graphs)
    edges = []
    for lid in range(n_layers):
      g = graphs[lid]
      edges.append(np.array(g.edges()))
    return edges


def save_graph(graphs, n_layers, n, beta, name):
    edges = get_edges(graphs)
    nets = []

    for lid in range(n_layers):
      e = edges[lid]
      layer = np.full((e.shape[0], 1), lid + 1)
      net = np.hstack((layer, e))
      nets.append(net)

    nets = np.vstack(nets)
    path = os.path.join('../data/nets', f'{name}_{n}_{n_layers}_{beta}')
    Path(path).mkdir(parents=True, exist_ok=True)

    for lid in range(n_layers):
      e = edges[lid]
      np.savetxt(os.path.join(path, f'l{lid+1}.edges'), e, delimiter=' ', fmt='%d')
    meta_info = {'N': [str(n)], 'L': [str(n_layers)], 'E': ['UNDIRECTED'], 'TYPE': ['MPX']}
    meta_info = pd.DataFrame(data=meta_info)
    np.savetxt(os.path.join(path, f'net.edges'), nets, delimiter=' ', fmt='%d')
    meta_info.to_csv(os.path.join(path, 'meta_info.txt'), index=False, sep=' ')
    return nets


if __name__ == '__main__':
    args = set_params()
    n = args.nodes
    seed = args.seed
    n_layers = args.layers
    beta = args.beta
    init_seed(seed)
    directed = False

    print('Creating random multilayer graph : '
          f' Number of layers : {n_layers}\n'
          f' Number of nodes per layer: {n}\n'
          f' Rewiring probability: {beta}\n')
    graphs = make_watts_strogatz(n, n_layers, beta)

    save_graph(graphs, n_layers, n, beta, name='rn')
    print('Network saved successfully!')

