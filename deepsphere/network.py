import torch
from torch import nn
import torch.nn.functional as F
from .cheb_conv import CachedChebConv

class DeepSphereNet(nn.Module):
    def __init__(self, in_features, out_features, graph_data, K, x_size):
        super().__init__()
        
        self.cnn_layers = []
        for in_ch, out_ch in [(in_features, 16), (16, 32), (32, 64), (64, 64), (64, 64)]:
            layer = nn.Sequential(
                CachedChebConv(in_ch, out_ch, graph_data, K, x_size),
                nn.ReLU()
                # TODO: MaxPooling
            )
            self.cnn_layers.append(layer)

        in_lin_features = x_size # TODO???
        self.fully_connected = nn.Sequential([
            nn.Linear(in_lin_features, out_features),
            nn.LogSoftmax()
        ])
    
    def forward(self, x):
        for layer in self.cnn_layers:
            x = layer(x)

        return self.fully_connected(x)
