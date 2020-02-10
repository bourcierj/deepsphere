import sys
import os
sys.path.append('./deepsphere')
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import DataLoader

from dataset import ModelNet40Dataset
from sampling import SphereHealpix


class ModelNet40GraphDataset(Dataset):
    """The ModelNet40 Dataset: https://modelnet.cs.princeton.edu/
    PyTorch Geometric Dataset wrapper for Data objects loading.
    """
    def __init__(self, root='./data/ModelNet40', role='train', nside=32,
                 nfeat=6, nfile=None, experiment='deepsphere_healpix_nside_32',
                 transform=None,
                 indexes=None, similarity='perraudin'):

        self.dataset = ModelNet40Dataset(root, role, nside, nfeat, nfile, experiment,
                                         fix=False, cache=True, verbose=False)

        sphere = SphereHealpix(nside=nside, indexes=indexes, similarity=similarity)

        self.coords = sphere.coords
        self.edge_index = sphere.edge_index
        self.edge_weight = sphere.edge_weight

        self.experiment = experiment
        self.nside = nside
        self.root = os.path.expanduser(root)
        self.role = role
        self.classes = self.dataset.classes
        self.nclasses = self.dataset.nclasses

        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)

        data_obj = Data(img, self.edge_index, self.edge_weight, target, self.coords)
        return data_obj

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':

    root = './data/ModelNet40'
    dataset = ModelNet40GraphDataset(root, 'train', nside=32, nfeat=6)
    # for idx in tqdm(range(len(dataset))):
    #     data = graph_dataset[idx]
    #     print(data)

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for batch_idx, data in enumerate(loader):
        print(data)
        if batch_idx > 10:
            break
