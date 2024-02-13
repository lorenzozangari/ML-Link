from models.sfg import MLStruct
import torch.nn as nn
from models.mgat import MGAT
import torch
from models.sfg import MAA, Glob
import utils.const as C


class Mm(nn.Module):

    def __init__(self, n_layers, dropout, eps=1e-8, no_struct=False, no_gnn=False, **args):
        super(Mm, self).__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.eps = eps
        self.struct = not no_struct
        self.gnn = not no_gnn
        self.edge_w = False
        self.beta = 1.0
        if not no_struct:
            self.psi = args['psi']
            self.edge_dim = args['edge_dim']
            self.node_dim = args['node_dim']
            self.phi_dim = args['phi_dim']
            self.attn_dropout = args['attn_dropout']
            self.f_dropout = args['f_dropout']
            self.n_heads = args['heads']

            self.struct_model = MLStruct(self.edge_dim, self.node_dim, self.phi_dim, self.n_layers, beta=self.beta,
                                         dropout=self.dropout, eps=self.eps, f_dropout=self.f_dropout)
            self.tacl = MAA(self.n_layers, self.beta, self.eps, phi_dim=self.phi_dim,
                            dropout=self.dropout, f_dropout=self.f_dropout)
            self.gacl = Glob(self.n_layers, self.beta, self.eps, phi_dim=self.phi_dim,
                             dropout=self.dropout, f_dropout=self.f_dropout)
            self.p2att = Attention(self.n_layers, self.phi_dim, self.attn_dropout, num_heads=1)

        if not no_gnn:
            self.input_dim = args['input_dim']
            self.hidden_dim = args['hidden_dim']
            self.num_hidden = args['num_hidden']
            self.heads = [args['heads']] * self.num_hidden
            self.activation = args['activation']
            self.attn_dropout = .0
            if 'attn_dropout' in args:
                self.attn_dropout = args['attn_dropout']
            self.residual = False
            if 'residual' in args:
                self.residual = args['residual']
            self.aggregation = 'avg'
            if 'aggregation' in args:
                self.aggregation = args['aggregation']

            self.gnn_model = MGAT(self.num_hidden, self.input_dim, self.hidden_dim, self.heads, self.activation, self.aggregation, self.dropout, self.attn_dropout, residual=self.residual)

        if self.gnn and self.struct:
            par = torch.FloatTensor([0, 0]).repeat(self.n_layers, 1)
            self.alpha = nn.Parameter(par)

        self.reset_parameters()

    def reset_parameters(self):
        if self.gnn and self.struct:
            nn.init.constant_(self.alpha, val=0.0)

    def combine_ps(self, out_structs1, acl):
        out_structs2 = []
        for lid in range(len(out_structs1)):
            out_structs2.append(
                torch.sigmoid((1 - self.psi) *
                              out_structs1[lid] + self.psi * acl[lid])
            )
        return out_structs2

    def forward(self, g_supra, g, p, g_edges, features, predictor, inter_layer=None):
        if self.struct:
            out_structs1, node_struct_feats = self.struct_model(g, g_edges, edge_w=self.edge_w)
            out_structs2 = []
            if inter_layer:
                for mode in inter_layer:
                    if mode == C.MAAN:
                        tacl = self.tacl(g, g_edges, out_structs1, node_struct_feats, p, edge_w=self.edge_w)
                        out_structs2.append(tacl)
                    elif mode == C.OAN:
                        gacl = self.gacl(g, g_edges, out_structs1, node_struct_feats, p, edge_w=self.edge_w)
                        out_structs2.append(gacl)

                out_structs1 = list(filter(lambda x: x is not None, out_structs1))
                if len(out_structs2) > 1:
                    out_structs3 = []
                    for i in range(len(out_structs2)):
                        out_structs3.append(list(filter(lambda x: x is not None, out_structs2[i])))
                    out_structs2 = self.p2att(out_structs3)
                else:
                    out_structs2 = out_structs2[0]

                out_structs2 = self.combine_ps(out_structs1, out_structs2)

            else:
                out_structs1 = list(filter(lambda x: x is not None, out_structs1))
                out_structs2 = [torch.sigmoid(o) for o in out_structs1]

        if self.gnn:
            out_gnn = self.gnn_model(g_supra, features)
            out_gnns_l = list(out_gnn.view(self.n_layers, -1, out_gnn.shape[1]))
            out_gnns = []
            for l_id in range(self.n_layers):
                if g_edges[l_id].number_of_edges() <= 0:
                    continue
                if isinstance(predictor, list):
                    score = predictor[l_id](g_edges[l_id], out_gnns_l[l_id])
                else:
                    score = predictor(g_edges[l_id], out_gnns_l[l_id])
                out_gnns.append(torch.sigmoid(score))

        if self.struct and self.gnn:
            alpha = torch.softmax(self.alpha, dim=-1)
            outs = []
            for l_id in range(len(out_structs2)):
                out = alpha[l_id][0] * out_structs2[l_id] + alpha[l_id][1] * out_gnns[l_id]
                outs.append(out)

            return outs, out_structs2, out_gnns
        elif self.struct:
            return out_structs2, None, None
        else:
            return out_gnns, None, None


class Attention(nn.Module):
    def __init__(self, n_layers, hidden_dim, attn_drop, num_heads=1):
        super(Attention, self).__init__()

        self.fc = torch.nn.Sequential(torch.nn.Linear(1, hidden_dim, bias=True),
                                      nn.PReLU(), nn.Linear(hidden_dim, hidden_dim, bias=True))
        self.tanh = nn.Tanh()

        self.att = nn.Parameter(torch.empty(size=(1, num_heads, hidden_dim)), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.n_layers = n_layers
        self.bb = None

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        self.fc.apply(self.weight_reset)
        nn.init.xavier_normal_(self.att.data, gain=gain)

    def weight_reset(self, m):

        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=gain)

    def forward(self, embeds):
        beta = []
        attn_curr = self.att

        for lid in range(len(embeds[0])):
            for i in range(len(embeds)):

                sp = self.tanh(self.fc(embeds[i][lid])).mean(dim=0)

                beta.append(attn_curr.matmul(sp.t()).mean(dim=-1))

        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        self.bb = beta
        k = 0
        z_mps = []
        for lid in range(len(embeds[0])):
            z_mp = .0
            for i in range(len(embeds)):
                z_mp += embeds[i][lid]*beta[k]
                k += 1
            z_mps.append(z_mp)

        return z_mps

