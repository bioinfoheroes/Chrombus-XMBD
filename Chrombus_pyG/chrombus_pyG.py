# %%
import math
import torch
import random
from torch.nn import Linear,Softmax
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.nn import MessagePassing

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')


class EdgeConv(MessagePassing):
    def __init__(self, n_heads, in_channels, out_channels):
        super().__init__(aggr='mean') #aggregation.
        self.n_heads = n_heads
        self.head_size = out_channels // n_heads
        self.all_heads = out_channels   
        self.softmax = Softmax(dim=-1)
        self.q = Linear(in_channels * 2, out_channels)
        self.k = Linear(in_channels * 2, out_channels)
        self.v = Linear(in_channels * 2, out_channels)
    def reshape(self, e):
        new_shape = e.size()[:-1] + (self.n_heads, self.head_size)
        e = e.view(*new_shape)
        return e.permute(1, 0, 2)  
    def forward(self, x, edge_index, edge_weight, batch=None):
        #print(f'number of edges: {edge_index.size(1)}')
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
    def message(self, x_i, x_j, edge_weight):
        query = self.q(torch.cat([x_i, x_j - x_i], dim=1))
        key = self.k(torch.cat([x_i, x_j - x_i], dim=1))
        value = self.v(torch.cat([x_i, x_j - x_i], dim=1))
        # print(f'values: {value.size()}')
        query = self.reshape(query)
        key = self.reshape(key)
        value = self.reshape(value)
        value = value * edge_weight[:,None]
        # print(f'keys: {key.size()}')
        scores = torch.mul(query, key)
        scores = scores / math.sqrt(self.head_size)
        probs = self.softmax(scores).to(device)  
        context = torch.mul(probs, value)
        # print(f'context: {context.size()}')
        context = context.permute(1, 0, 2).contiguous()
        # print(f'context2: {context.size()}')
        context_shape = context.size()[:-2] + (self.all_heads, )
        context = context.view(*context_shape)
        # print(f'context3: {context.size()}')
        #print(f'context: {context.size()}')
        return context

class DynamicEdgeConv(EdgeConv):
    def __init__(self, n_heads, in_channels, out_channels, thres=0.5, K=500, cis=8, n_neighbor=6):
        super().__init__(n_heads, in_channels, out_channels)
        self.K = K
        self.thres = torch.tensor(thres).to(device)
        self.cis =  torch.tensor(cis).to(device)
        self.n_neighbor = n_neighbor
    def forward(self, x, edge_index, batch):
        edge_kept = random.sample(range(0,edge_index.shape[1]), int(edge_index.shape[1] * 0.5))
        edge_index = edge_index[:,edge_kept]
        d = (edge_index[1] - edge_index[0]).abs()
        edge_weight = (d.log() - self.cis.log()).sign()
        return super().forward(x, edge_index, edge_weight=edge_weight)

class EdgeConvEncoder(torch.nn.Module): 
    def __init__(self, n_heads, in_channels, out_channels, cis_span):
        super().__init__()
        self.conv0 = DynamicEdgeConv(n_heads, in_channels, out_channels, thres=-.5, K=10000, cis=cis_span)
        self.conv1 = DynamicEdgeConv(n_heads, out_channels, out_channels // 2, thres=-.5, K=10000, cis=cis_span)
        self.conv2 = DynamicEdgeConv(n_heads, out_channels // 2, out_channels, thres=-.5, K=10000, cis=cis_span)
    def forward(self, x, edge_index, batch):
        x0 = self.conv0(x, edge_index, batch)
        # print(f'x0: {x0.size()}')
        x0 = x0.relu()
        x = self.conv1(x0, edge_index, batch)
        x = x.relu()       
        x = self.conv2(x, edge_index, batch)
        x = x + x0  
        x = x.relu()      
        return x
    
class EdgeConvEncoder_PE(torch.nn.Module): 
    def __init__(self, n_heads, in_channels, out_channels, cis_span):
        super().__init__()
        self.pe = torch.nn.Embedding(128,14)
        self.lin = torch.nn.Linear(12,14)
        self.conv0 = DynamicEdgeConv(n_heads, in_channels, out_channels, thres=-.5, K=10000, cis=cis_span)
        self.conv1 = DynamicEdgeConv(n_heads, out_channels, out_channels // 2, thres=-.5, K=10000, cis=cis_span)
        self.conv2 = DynamicEdgeConv(n_heads, out_channels // 2, out_channels, thres=-.5, K=10000, cis=cis_span)
    def forward(self, x, edge_index, batch):
        x = self.lin(x[:,2:-1]) + self.pe(x[:,-1].long())
        x0 = self.conv0(x, edge_index, batch)
        # print(f'x0: {x0.size()}')
        x0 = x0.relu()
        x = self.conv1(x0, edge_index, batch)
        x = x.relu()       
        x = self.conv2(x, edge_index, batch)
        x = x + x0  
        x = x.relu()      
        return x

class EdgeConvDecoder(torch.nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1):
        super().__init__()
        self.decoder = InnerProductDecoder()
        self.linear = Linear(in_channels,out_channels)
    def forward(self, x, edge_index):
        pred = self.decoder(x, edge_index, sigmoid=False)
        #pred = self.linear(pred.unsqueeze(1))
        return pred




