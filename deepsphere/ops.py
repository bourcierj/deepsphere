
import torch
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

        if self.normalization != get_laplacian'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')
        lambda_max = 2.0 if lambda_max is None else lambda_max

        self.edge_index, self.norm_value = self.norm(
            graph_data.edge_index, x_size, graph_data.edge_weight,
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


def pool_max(self, x, p):
    """Max pooling of size p. Should be a power of 2."""
    if p > 1:
        if self.sampling is 'equiangular':
            N, M, F = x.get_shape()
            N, M, F = int(N), int(M), int(F)
            x = tf.reshape(x,[N,int((M/self.ratio)**0.5), int((M*self.ratio)**0.5), F])
            x = tf.nn.max_pool(x, ksize=[1,p**0.5,p**0.5,1], strides=[1,p**0.5,p**0.5,1], padding='SAME')
            return tf.reshape(x, [N, -1, F])
        elif self.sampling  is 'icosahedron':
            return x[:, :p, :]
        else:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
    else:
        return x


def max_pool(x, p, sampling='healpix', ratio=1.):
    """Maximum pooling of a spherical signal with hierarchichal sampling.
    Args:
        x (tensor): tensor of shape (batch?, n_samples, n_features)
        p (int): the pooling size or down-sampling factor (power of 2)
        sampling (str): the sampling scheme
    """
    if p > 1:
        #@TODO: WORKS ONLY ON TENSOR BATCHES? And PyTorch Geometric?
        if sampling is 'equiangular':
            assert(len(x.size()) == 3)
            x = x.transpose(1, 2)  # channels first for compatibility with F.max_pool*
            batch_size, n_feats, n_points = x.size()
            x = x.view(batch_size, n_feats, int((n_points/ratio)**0.5),
                       int((n_points/ratio)**0.5))
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
        self.p = p
        self.sampling = sampling
        self.ratio = ratio

    def forward(self, x):
        return max_pool(x, self.p, self.sampling, self.ratio)


if __name__ == '__main__':

    nside = 32
    npix = 12 * nside**2
    x = torch.randn(4, npix, 6)
    # p = ?? # pooling size
    output = max_pool(x, p, sampling='healpix')
    print(output.shape)
