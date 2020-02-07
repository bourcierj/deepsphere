import sys
sys.path.append('./deepsphere')
import torch
from torch_geometric.data import Data

from sampling import *

def _test():
    graph = SphereHealpix(nside=16)
    print(graph.edge_index)
    print(graph.edge_weight)

    data = Data(edge_index=graph.edge_index, edge_attr=graph.edge_weight)
    print(data)

    graph = SphereEquiangular(bw=16)
    print(graph.edge_index)
    print(graph.edge_weight)

    data = Data(edge_index=graph.edge_index, edge_attr=graph.edge_weight)
    print(data)

    graph = SphereIcosahedral(level=1)
    print(graph.edge_index)
    print(graph.edge_weight)

    data = Data(edge_index=graph.edge_index, edge_attr=graph.edge_weight)
    print(data)


if __name__ == '__main__':
    _test()
