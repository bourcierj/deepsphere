from scipy import sparse
import torch

def sparse2tensor(m):
    """
    Convert sparse matrix (scipy.sparse) to tensor (torch.sparse)
    """
    assert(isinstance(m, sparse.coo.coo_matrix))
    i = torch.LongTensor([m.row, m.col])
    v = torch.FloatTensor(m.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(m.shape))
