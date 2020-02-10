
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.inits import glorot, zeros

class CachedChebConv(MessagePassing):
    r"""
    The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper, with some modifications
    from the original PyTorch Geometric implementation.
    Original:
    https://github.com/rusty1s/pytorch_geometric/blob/751555b15c04e836d1ff77c96a3212aea20cc42d/torch_geometric/nn/conv/cheb_conv.py
    Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
    This is intended to be used with the same graph structure and computes the
    Laplacian only once.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size, *i.e.* number of hops :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a scalar when
            operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, graph_data, K, x_size,
                 dtype=torch.float32, normalization='sym', bias=True,
                 batch=None, lambda_max=None, **kwargs):
        # x_size = x.size(self.node_dim)
        # TODO: can be computed from in_channels?

        super(CachedChebConv, self).__init__(aggr='add', **kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')
        lambda_max = 2.0 if lambda_max is None else lambda_max

        self.edge_index, self.norm_value = self.norm(
            graph_data.edge_index, x_size, graph_data.edge_attr,
            self.normalization, lambda_max, dtype=dtype, batch=batch
        )
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, normalization, lambda_max,
             dtype=None, batch=None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        if batch is not None and torch.is_tensor(lambda_max):
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight[edge_weight == float('inf')] = 0

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1,
                                                 num_nodes=num_nodes)

        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        edge_index, norm = self.edge_index, self.norm_value

        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])

        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)


def max_pool(x, p, sampling='healpix', ratio=1.):
    """Maximum pooling of a spherical signal with hierarchichal sampling.
    Args:
        x (tensor): tensor of shape (batch, n_pixels, n_features)
        p (int): the pooling size or down-sampling factor (power of 2)
        sampling (str): the sampling scheme
    """
    if p > 1:
        #@TODO: WORKS ONLY ON TENSOR BATCHES? And PyTorch Geometric?
        if sampling is 'equiangular':
            assert(len(x.size()) == 3)
            x = x.transpose(1, 2)  # channels first for compatibility with F.max_pool*
            batch_size, n_feats, npix = x.size()
            x = x.view(batch_size, n_feats, int((npix/ratio)**0.5),
                       int((npix/ratio)**0.5))
            #@TODO: apply 'same' padding
            x = F.max_pool2d(x, kernel_size=(p**0.5, p**0.5), stride=(p**0.5,p**0.5))
            x = x.view(batch_size, n_feats, -1)
            x = x.transpose(2, 1)

        if sampling is 'healpix':
            x = x.transpose(1, 2) # channels first for compatibility with F.max_pool*
            #@TODO: apply 'same' padding
            x = F.max_pool1d(x, kernel_size=p, stride=p)
            x = x.transpose(2, 1)

        if sampling is 'icosahedral':
            return x[:, :p, :]
    return x


class MaxPool(nn.Module):
    """Maximum pooling module wrapper"""

    def __init__(self, p, sampling='healpix', ratio=1.):
        super(MaxPool, self).__init__()
        self.p = p
        self.sampling = sampling
        self.ratio = ratio

    def forward(self, x):
        return max_pool(x, self.p, self.sampling, self.ratio)


def global_avg_pool(x):
    """Global average pooling (GAP).
    Args:
        x (tensor): tensor of shape (batch, n_pixels, n_features)
    """
    x = x.transpose(1, 2)
    x = F.adaptive_avg_pool1d(x, output_size=1)
    return x.transpose(2, 1)


class GlobalAvgPool(nn.Module):
    """Global average pooling module wrapper"""

    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return global_avg_pool(x)


def get_pooling_sizes(sparams, sampling='healpix'):
    pool_sizes = []
    sp_last = -1
    if sampling != 'icosahedral':
        for i, (sp, index) in enumerate(zip(sparams, indexes)):
                if isinstance(sp, tuple):
                    sp = sp[0]
                if i > 0:
                    pool_sizes.append((sp_last // sp)**2)
                sp_last = sp
    if sampling == 'icosahedron':
        for nv in sparams[1:]:  # levels
            pool_sizes.append(10 * 4**order + 2)  # number of vertices for level
    else:
        raise ValueError('Unsupported sampling: {}'.format(sampling))


if __name__ == '__main__':

    # Test max pooling on five sampling levels: ok
    print('Test Max pool')
    nsides = [32, 16, 8, 4, 2]
    pooling_sizes = [(nsides[i] // nsides[i+1])**2 for i in range(len(nsides)-1)]
    nside = 32
    npix = 12 * nside**2
    input_x = torch.randn(16, npix, 6)  # random examples minimatch
    out = input_x
    for nside, p in zip(nsides, pooling_sizes):
        assert(p == 4)
        out = max_pool(out, p, sampling='healpix')
        print('p: {}, output shape: {}'.format(p, tuple(out.shape)))
        new_npix = 12 * (nside//2) **2
        assert(out.shape == (16, new_npix, 6))

    # Test global avg pooling: ok
    print('Test Global avg pool')
    npix = 48
    input_x = torch.randn(16, npix, 6)  # random examples minimatch
    out = global_avg_pool(input_x)
    print('output shape: {}'.format(tuple(out.shape)))
