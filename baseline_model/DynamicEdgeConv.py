from tkinter import W
from matplotlib.backends.backend_pdf import PdfPages
from torch.nn import Sequential, Linear, MultiheadAttention, Sigmoid, ReLU, ELU, LeakyReLU, Softmax
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.utils import to_undirected, batched_negative_sampling, remove_self_loops
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv,EdgeConv, GAE, VGAE, InnerProductDecoder
from torch_scatter import scatter_mean
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from matplotlib.backends.backend_pdf import PdfPages
import math
import torch
import numpy
import pandas
import random
import numpy as np


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Sequential(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super(DynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k
    def forward(self, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super(DynamicEdgeConv, self).forward(x, edge_index)



class DEEncoder(torch.nn.Module): 
    def __init__(self, in_channels, out_channels, cis_span = 9):
        super().__init__()
        self.conv0 = DynamicEdgeConv(in_channels, out_channels)
        self.conv1 = DynamicEdgeConv(out_channels, out_channels // 2)
        self.conv2 = DynamicEdgeConv(out_channels // 2, out_channels)
        self.cis = torch.tensor(cis_span)
    def forward(self, x, batch):
        x0 = self.conv0(x, batch)
        # print(f'x0: {x0.size()}')
        x0 = x0.relu()
        x = self.conv1(x0, batch)
        # print(f'x: {x.size()}')
        x = x.relu()       
        x = self.conv2(x, batch)
        # print(f'x: {x.size()}')
        x = x + x0  
        x = x.relu()
        return x

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cis_span = 9):
        super().__init__()
        self.encoder = DEEncoder(in_channels, out_channels, cis_span)
        self.decoder = InnerProductDecoder()
    def forward(self, x,batch, edge_index):
        z = self.encoder(x, batch)
        pred = self.decoder(z, edge_index, sigmoid=False)
        return(pred)
