import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import SparseTensor
import dgl


class MLStruct(torch.nn.Module):

    def __init__(self, edge_dim, node_dim, phi_dim, n_layers, beta=1.0,  dropout=0.2,
                 f_dropout=0.7, eps=1e-8):
        super(MLStruct, self).__init__()

        self.n_layers = n_layers
        self.beta = beta
        self.eps = eps
        self.f_dropout = f_dropout
        self.cns = nn.ModuleList()

        for _ in range(n_layers):
            f_edge = torch.nn.Sequential(torch.nn.Linear(1, edge_dim), nn.Dropout(dropout),
                                         nn.ReLU(), nn.Linear(edge_dim, 1))

            f_node = torch.nn.Sequential(torch.nn.Linear(1, node_dim),
                                         nn.ReLU(),  nn.Linear(node_dim, 1))

            g_phi = torch.nn.Sequential(torch.nn.Linear(1, phi_dim),
                                        nn.ReLU(), nn.Linear(phi_dim, 1))

            cns_l = nn.ModuleList()
            cns_l.append(f_edge)
            cns_l.append(f_node)
            cns_l.append(g_phi)
            self.cns.append(cns_l)

        self.reset_parameters()

    def reset_parameters(self):
        for cnl in self.cns:
            for l in cnl:
                l.apply(self.weight_reset)

    def weight_reset(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def random_fill(self, mat, dtype=torch.float32):
        device = mat.device
        return mat + \
            F.dropout(torch.full((mat.shape[0], mat.shape[1]), self.eps, dtype=dtype).to(device), p=self.f_dropout)

    def forward(self, g, g_edges, edge_w=False):
        out_structs = []
        node_struct_feats = []
        for l_id in range(self.n_layers):
            batch_size = g_edges[l_id].number_of_edges()
            device = g[l_id].device
            num_nodes = g[l_id].number_of_nodes()
            f_edge = self.cns[l_id][0]
            f_node = self.cns[l_id][1]
            g_phi = self.cns[l_id][2]

            rows, cols, eids = g[l_id].edges(order='srcdst', form='all')

            edge_weights = g[l_id].edata['w'][eids].to(device)
            edge_weight_A = f_edge(edge_weights.unsqueeze(-1))
            node_struct_feat = scatter_add(edge_weight_A, cols, dim=0, dim_size=g[l_id].number_of_nodes())
            node_struct_feat = f_node(node_struct_feat)

            node_struct_feats.append(node_struct_feat)
            if batch_size <= 0:
                out_structs.append(None)
                continue

            A = g[l_id].adj_external(scipy_fmt="csr")
            if edge_w:
                A.data = edge_weights.cpu().numpy()

            edge = g_edges[l_id].edges(order='srcdst')

            indexes_src = edge[0].cpu().numpy()
            row_src, col_src = A[indexes_src].nonzero()
            edge_index_src = torch.stack([torch.from_numpy(row_src), torch.from_numpy(col_src)]).type(
                torch.LongTensor).to(device)
            edge_weight_src = torch.from_numpy(A[indexes_src].data).to(device)
            edge_weight_src = edge_weight_src * node_struct_feat[col_src].squeeze()

            indexes_dst = edge[1].cpu().numpy()
            row_dst, col_dst = A[indexes_dst].nonzero()
            edge_index_dst = torch.stack([torch.from_numpy(row_dst), torch.from_numpy(col_dst)]).type(
                torch.LongTensor).to(device)
            edge_weight_dst = torch.from_numpy(A[indexes_dst].data).to(device)
            edge_weight_dst = edge_weight_dst * node_struct_feat[col_dst].squeeze()

            mat_src = SparseTensor.from_edge_index(edge_index_src, edge_weight_src, [batch_size, num_nodes])
            mat_dst = SparseTensor.from_edge_index(edge_index_dst, edge_weight_dst, [batch_size, num_nodes])

            out_struct = (mat_src @ mat_dst.to_dense().t()).diag()

            dtype = out_struct.dtype

            mat_src_dense = mat_src.to_dense()
            mat_dst_dense = mat_dst.to_dense()
            mat_src_dense = self.random_fill(mat_src_dense, dtype=dtype)
            mat_dst_dense = self.random_fill(mat_dst_dense, dtype=dtype)
            d_src = torch.norm(mat_src_dense, dim=-1)
            d_dst = torch.norm(mat_dst_dense, dim=-1)
            D = d_src * d_dst
            out_struct_n = (out_struct / D).unsqueeze(-1)

            out_struct_n = g_phi(out_struct_n)

            out_structs.append(out_struct_n)
        del edge_weight_src, edge_weight_dst, mat_src, mat_dst, edge_index_src, edge_index_dst

        torch.cuda.empty_cache()
        return out_structs, node_struct_feats


class MAA(torch.nn.Module):

    def __init__(self, n_layers, beta, eps=1e-16, **kwargs):
        super(MAA, self).__init__()
        self.n_layers = n_layers
        self.beta = beta
        self.eps = eps
        self.phi_dim = kwargs['phi_dim']
        self.dropout = kwargs['dropout']
        self.f_dropout = kwargs['f_dropout']

        self.g_phi1 = torch.nn.ModuleList()
        self.g_phi2 = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.g_phi1.append(torch.nn.Sequential(torch.nn.Linear(1, self.phi_dim),
                                                    nn.ReLU(), nn.Linear(self.phi_dim, 1)))
            self.g_phi2.append(torch.nn.Sequential(torch.nn.Linear(1, self.phi_dim),
                                                    nn.ReLU(), nn.Linear(self.phi_dim, 1)))
        self.reset_parameters()

    def reset_parameters(self):
        for lm in self.g_phi1:
                lm.apply(self.weight_reset)
        for lm in self.g_phi2:
                lm.apply(self.weight_reset)

    def weight_reset(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def random_fill(self, mat, dtype=torch.float32):
        device = mat.device
        return mat + \
            F.dropout(torch.full((mat.shape[0], mat.shape[1]), self.eps, dtype=dtype).to(device), p=self.f_dropout)

    def forward(self, g, g_edges, out_structs, node_struct_feats, p, edge_w=False):
        device = g[0].device
        out_structs2 = []

        for lid1 in range(self.n_layers):
            batch_size = g_edges[lid1].number_of_edges()
            if batch_size <= 0:
                out_structs2.append(None)
                continue

            pol = p[lid1]
            num_nodes = g[lid1].number_of_nodes()
            A = g[lid1].adj_external(scipy_fmt="csr")  # lid1 is the current target layer
            if edge_w:
                eids = g[lid1].edges(order='srcdst', form='eid')
                edge_weights = g[lid1].edata['w'][eids].to(device)
                A.data = edge_weights.cpu().numpy()

            edge = g_edges[lid1].edges(order='srcdst')
            indexes_src = edge[0].cpu().numpy()
            row_src, col_src = A[indexes_src].nonzero()
            edge_index_src = torch.stack([torch.from_numpy(row_src), torch.from_numpy(col_src)]).type(
                torch.LongTensor).to(device)
            edge_weight_src = torch.from_numpy(A[indexes_src].data).to(device)
            edge_weight_src = edge_weight_src * node_struct_feats[lid1][
                col_src].squeeze()

            mat_src = SparseTensor.from_edge_index(edge_index_src, edge_weight_src, [batch_size, num_nodes])
            dtype = node_struct_feats[lid1].dtype

            lm = len(pol)
            sl1 = torch.zeros_like(out_structs[lid1])
            for i, lid2 in enumerate(pol):
                if lid2 == -1 or lid2 is None:
                    continue
                if lid1 != lid2:
                    indexes_dst = edge[1].cpu().numpy()
                    A2 = g[lid2].adj_external(scipy_fmt="csr")

                    if edge_w:
                        eids2 = g[lid2].edges(order='srcdst', form='eid')
                        edge_weights2 = g[lid2].edata['w'][eids2].to(device)
                        A2.data = edge_weights2.cpu().numpy()

                    row_dst, col_dst = A2[indexes_dst].nonzero()
                    edge_index_dst = torch.stack([torch.from_numpy(row_dst), torch.from_numpy(col_dst)]).type(
                        torch.LongTensor).to(device)

                    edge_weight_dst = torch.from_numpy(A2[indexes_dst].data).to(device)
                    edge_weight_dst = edge_weight_dst * node_struct_feats[lid2][col_dst].squeeze()

                    mat_dst = SparseTensor.from_edge_index(edge_index_dst, edge_weight_dst, [batch_size, num_nodes])

                    # Asymmetric case
                    row_src_rev, col_src_rev = A2[indexes_src].nonzero()
                    edge_index_src_rev = torch.stack([torch.from_numpy(row_src_rev), torch.from_numpy(col_src_rev)]).type(
                        torch.LongTensor).to(device)
                    edge_weight_src_rev = torch.from_numpy(A2[indexes_src].data).to(device)
                    edge_weight_src_rev = edge_weight_src_rev * node_struct_feats[lid2][
                        col_src_rev].squeeze()

                    mat_src_rev = SparseTensor.from_edge_index(edge_index_src_rev, edge_weight_src_rev, [batch_size, num_nodes])

                    row_dst_rev, col_dst_rev = A[indexes_dst].nonzero()
                    edge_index_dst_rev = torch.stack([torch.from_numpy(row_dst_rev),
                                                      torch.from_numpy(col_dst_rev)]).type(
                        torch.LongTensor).to(device)
                    edge_weight_dst_rev = torch.from_numpy(A[indexes_dst].data).to(device)
                    edge_weight_dst_rev = edge_weight_dst_rev * node_struct_feats[lid1][col_dst_rev].squeeze()

                    mat_dst_rev = SparseTensor.from_edge_index(edge_index_dst_rev, edge_weight_dst_rev, [batch_size, num_nodes])

                    out_struct = (mat_src @ mat_dst.to_dense().t()).diag()
                    out_struct_rev = (mat_src_rev @ mat_dst_rev.to_dense().t()).diag()

                    mat_src_dense = mat_src.to_dense()
                    mat_dst_dense = mat_dst.to_dense()
                    mat_src_rev_dense = mat_src_rev.to_dense()
                    mat_dst_rev_dense = mat_dst_rev.to_dense()
                    mat_src_dense = self.random_fill(mat_src_dense, dtype)
                    mat_dst_dense = self.random_fill(mat_dst_dense, dtype)
                    mat_src_rev_dense = self.random_fill(mat_src_rev_dense, dtype)
                    mat_dst_rev_dense = self.random_fill(mat_dst_rev_dense, dtype)
                    d_src = torch.norm(mat_src_dense, dim=-1)
                    d_dst = torch.norm(mat_dst_dense, dim=-1)
                    d_src_rev = torch.norm(mat_src_rev_dense, dim=-1)
                    d_dst_rev = torch.norm(mat_dst_rev_dense, dim=-1)
                    D = d_src * d_dst
                    D_rev = d_src_rev * d_dst_rev
                    out_struct_n = ((out_struct / D) + (out_struct_rev / D_rev)).unsqueeze(-1)
                    out_struct_n = self.g_phi1[lid1](out_struct_n)
                    sl1 = sl1 + self.beta * (out_struct_n / lm)

            sl1 = self.g_phi2[lid1](sl1)
            out_structs2.append(sl1)
        out_structs2 = list(filter(lambda x: x is not None, out_structs2))
        del edge_weight_src, edge_weight_dst, edge_index_src, edge_index_dst, mat_src, mat_dst
        return out_structs2


class Glob(torch.nn.Module):
    def __init__(self, n_layers, beta,  eps=1e-8, **kwargs):
        super(Glob, self).__init__()
        self.n_layers = n_layers
        self.eps = eps
        self.beta = beta
        self.phi_dim = kwargs['phi_dim']
        self.dropout = kwargs['dropout']
        self.f_dropout = kwargs['f_dropout']
        self.g_phi1 = torch.nn.ModuleList()
        self.g_phi2 = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.g_phi1.append(torch.nn.Sequential(torch.nn.Linear(1, self.phi_dim),
                                                   nn.ReLU(), nn.Linear(self.phi_dim, 1)))
            self.g_phi2.append(torch.nn.Sequential(torch.nn.Linear(1, self.phi_dim),
                                                       nn.ReLU(), nn.Linear(self.phi_dim, 1)))

        self.reset_parameters()

    def reset_parameters(self):
        for lm in self.g_phi1:
                lm.apply(self.weight_reset)
        for lm in self.g_phi2:
                lm.apply(self.weight_reset)

    def weight_reset(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def random_fill(self, mat, dtype=torch.float32):
        device = mat.device
        return mat + \
            F.dropout(torch.full((mat.shape[0], mat.shape[1]), self.eps, dtype=dtype).to(device), p=self.f_dropout)

    def forward(self, g, g_edges, out_structs, node_struct_feats, p, edge_w=False):
        device = g[0].device
        out_structs2 = []
        cache = {}
        for lid1 in range(self.n_layers):
            batch_size = g_edges[lid1].number_of_edges()
            if batch_size <= 0:
                out_structs2.append(None)
                continue
            pol = p[lid1]
            num_nodes = g[lid1].number_of_nodes()

            edge = g_edges[lid1].edges(order='srcdst')

            lm = len(pol)
            sl1 = torch.zeros_like(out_structs[lid1])
            dtype = node_struct_feats[lid1].dtype
            for i, lid2 in enumerate(pol):
                if lid2 == -1 or lid2 is None:
                    continue
                if lid1 != lid2:
                    if (lid1, lid2) not in cache:
                        if 'x' in g[lid1].ndata:
                            x1, x2 = g[lid1].ndata.pop('x'), g[lid2].ndata.pop('x')
                            g_merged = dgl.to_simple(dgl.merge([g[lid1].cpu(), g[lid2].cpu()]))
                            g[lid1].ndata['x'] = x1
                            g[lid2].ndata['x'] = x2
                        else:
                            g_merged = dgl.to_simple(dgl.merge([g[lid1].cpu(), g[lid2].cpu()]))

                        A = g_merged.adj_external(scipy_fmt="csr")  # lid1 is the current target layer
                        if lid2 > lid1:
                            cache[(lid2, lid1)] = A
                        if edge_w:
                            eids = g_merged.edges(order='srcdst', form='eid')
                            edge_weights = g_merged.edata['w'][eids].to(device)
                            A.data = edge_weights.cpu().numpy()
                    else:
                        A = cache[(lid1, lid2)]

                    node_struct_feat = torch.mean(torch.stack(
                        [node_struct_feats[lid1], node_struct_feats[lid2]], dim=0), dim=0)

                    indexes_src = edge[0].cpu().numpy()
                    row_src, col_src = A[indexes_src].nonzero()
                    edge_index_src = torch.stack([torch.from_numpy(row_src), torch.from_numpy(col_src)]).type(
                        torch.LongTensor).to(device)
                    edge_weight_src = torch.from_numpy(A[indexes_src].data).to(device)
                    edge_weight_src = edge_weight_src * node_struct_feat[
                        col_src].squeeze()

                    indexes_dst = edge[1].cpu().numpy()
                    row_dst, col_dst = A[indexes_dst].nonzero()
                    edge_index_dst = torch.stack([torch.from_numpy(row_dst), torch.from_numpy(col_dst)]).type(
                        torch.LongTensor).to(device)
                    edge_weight_dst = torch.from_numpy(A[indexes_dst].data).to(device)
                    edge_weight_dst = edge_weight_dst * node_struct_feat[col_dst].squeeze()

                    mat_src = SparseTensor.from_edge_index(edge_index_src, edge_weight_src, [batch_size, num_nodes])
                    mat_dst = SparseTensor.from_edge_index(edge_index_dst, edge_weight_dst, [batch_size, num_nodes])
                    out_struct = (mat_src @ mat_dst.to_dense().t()).diag()

                    mat_src_dense = mat_src.to_dense()
                    mat_dst_dense = mat_dst.to_dense()
                    mat_src_dense = self.random_fill(mat_src_dense, dtype)
                    mat_dst_dense = self.random_fill(mat_dst_dense, dtype)
                    d_src = torch.norm(mat_src_dense, dim=-1)
                    d_dst = torch.norm(mat_dst_dense, dim=-1)
                    D = d_src * d_dst
                    out_struct_n = (out_struct / D).unsqueeze(-1)

                    out_struct_n = self.g_phi1[lid1](out_struct_n)

                    sl1 = sl1 + self.beta*(out_struct_n / lm)

            sl1 = self.g_phi2[lid1](sl1)
            out_structs2.append(sl1)
        del edge_weight_src, edge_weight_dst, edge_index_src, edge_index_dst, mat_src, mat_dst, cache
        out_structs2 = list(filter(lambda x: x is not None, out_structs2))
        return out_structs2


