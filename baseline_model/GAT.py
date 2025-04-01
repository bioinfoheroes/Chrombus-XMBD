from tkinter import W
from matplotlib.backends.backend_pdf import PdfPages
from torch.nn import Sequential, Linear, MultiheadAttention, Sigmoid, ReLU, ELU, LeakyReLU, Softmax
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.utils import to_undirected, batched_negative_sampling, remove_self_loops
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE, InnerProductDecoder
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


class GATEncoder(torch.nn.Module): 
    def __init__(self, in_channels, out_channels, cis_span = 9, n_head = 4):
        super().__init__()
        self.conv0 = GATConv(in_channels, out_channels, dropout=0.5, heads=n_head)
        self.linear0 = Linear(out_channels * n_head, out_channels)
        self.conv1 = GATConv(out_channels, out_channels // 2, dropout=0.5, heads=n_head)
        self.linear1 = Linear(out_channels // 2 * n_head, out_channels // 2)
        self.conv2 = GATConv(out_channels // 2, out_channels, dropout=0.5, heads=n_head)
        self.linear2 = Linear(out_channels * n_head, out_channels)
        self.cis = torch.tensor(cis_span)
    def forward(self, x, edge_index):
        x0 = self.linear0(self.conv0(x, edge_index))
        # print(f'x0: {x0.size()}')
        x0 = x0.relu()
        x = self.linear1(self.conv1(x0, edge_index))
        # print(f'x: {x.size()}')
        x = x.relu()       
        x = self.linear2(self.conv2(x, edge_index))
        # print(f'x: {x.size()}')
        x = x + x0  
        x = x.relu()
        return x

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cis_span = 9):
        super().__init__()
        self.encoder = GATEncoder(in_channels, out_channels, cis_span)
        self.decoder = InnerProductDecoder()
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        pred = self.decoder(z, edge_index, sigmoid=False)
        return(pred)
