import numpy as np
import torch
import random
import dgl


def init_seed(seed=72):
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def identity_matrix(n, sparse=False):
    if sparse:
        values = torch.ones(n, dtype=torch.float32)
        indices = np.vstack((torch.arange(n), torch.arange(n)))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = (n, n)
        identity = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        return identity

    return torch.eye(n)

