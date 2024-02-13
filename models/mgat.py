import torch.nn as nn
from dgl.nn.pytorch import GATv2Conv


class MGAT(nn.Module):
    _multi_head = {'concat', 'avg'}

    def __init__(self, num_hidden, in_dim, hidden_dim, heads, activation, aggregation='concat',
                 drop=.0, attn_drop=.0, negative_slope=0.2, residual=False):

        super(MGAT, self).__init__()
        aggregation = aggregation.lower()
        if aggregation not in self._multi_head:
            raise ValueError("Unrecognized  aggregation mode for attention heads : {} ".format(aggregation))
        self.aggregation = aggregation
        self.num_layers = num_hidden
        self.activation = activation
        self.layers = nn.ModuleList()
        self.layers.append(GATv2Conv(
            in_dim, hidden_dim, heads[0],
            drop, attn_drop, negative_slope, residual=residual, activation=self.activation, allow_zero_in_degree=True,
            bias=True))
        for l in range(1, num_hidden):
            if self.aggregation == 'concat':
                inf = hidden_dim * heads[l - 1]
            else:
                inf = hidden_dim
            if l == num_hidden - 1:
                drop = .0
            self.layers.append(GATv2Conv(
                inf, hidden_dim, heads[l],
                drop, attn_drop, negative_slope, residual, self.activation, allow_zero_in_degree=True, bias=True))

        self.reset_parameters()

    def reset_parameters(self):
        for gnn in self.layers:
            gnn.reset_parameters()

    def forward(self, g, inputs):
        h = inputs
        for i in range(self.num_layers):
            if self.aggregation == 'concat':
                h = self.layers[i](g, h).flatten(1)
            else:
                h = self.layers[i](g, h).mean(1)
        return h

