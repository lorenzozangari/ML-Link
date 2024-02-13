import dgl

import torch.nn as nn
import torch.nn.functional as F


class MLPPredictor(nn.Module):
    def __init__(self, h_feats, dropout=0.2):
        super().__init__()
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(h_feats, h_feats, bias=True))
        self.linear.append(nn.Linear(h_feats, 1, bias=True))
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def apply_edges(self, edges):

        h = edges.src['h'] * edges.dst['h']

        for lin in self.linear[:-1]:
            h = lin(h)
            h = F.relu(h)
            h = self.dropout(h)
        h = self.linear[-1](h)
        return{'score': h}

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for lin in self.linear:
            nn.init.xavier_normal_(lin.weight, gain=gain)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)

            return g.edata['score']


class LinkPredictor(nn.Module):

    def __init__(self, op):
        super().__init__()
        self.predictor = dgl.nn.EdgePredictor(op)

    def forward(self, g, h):
        src, dst = g.edges(order='srcdst')
        h_src = h[src]
        h_dst = h[dst]
        out = self.predictor(h_src, h_dst)
        return out
