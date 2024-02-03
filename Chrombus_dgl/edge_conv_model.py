import pandas as pd
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import torch
import torch as th
from dgl.data import DGLDataset
import dgl.function as fn
from torch import Tensor
from load_chrombusdata import ChrombusDatset
from dgl.dataloading import GraphDataLoader
from dgl.nn import EdgeConv
import torch.nn.functional as F
from torch import nn
import dgl.function as fn
from dgl.base import DGLError
import random
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')


class EdgeConvGAT(nn.Module):
    def __init__(self, in_feat, out_feat, feat_drop = 0.5, attn_drop = 0.5, edge_drop = 0.5, batch_norm=False, allow_zero_in_degree=False, cis = torch.tensor(9), n_heads = 4, max_span = 128):
        super(EdgeConvGAT,self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.cis = cis
        self.heads = n_heads
        self.max_span = max_span
        ###
        # self.theta = nn.Linear(in_feat, out_feat)
        self.q = nn.Linear(in_feat * 2, out_feat)
        self.k = nn.Linear(in_feat * 2, out_feat)
        self.v = nn.Linear(in_feat * 2, out_feat)
        # self.attn_fc = nn.Linear(2 * out_feat, 1, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)
        self.reset_parameters()
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.k.weight, gain=gain)
        nn.init.xavier_normal_(self.q.weight, gain=gain)
        nn.init.xavier_normal_(self.v.weight, gain=gain)
    def edge_qkv(self, edges):
        #print(edges.src["x"].shape)
        q = self.q(torch.cat([edges.src["x"] - edges.dst["x"], edges.dst["x"]], dim=1))
        k = self.k(torch.cat([edges.src["x"] - edges.dst["x"], edges.dst["x"]], dim=1))
        v = self.v(torch.cat([edges.src["x"] - edges.dst["x"], edges.dst["x"]], dim=1))
        # a = self.attn_fc(z2)
        return {"q": q, "k": k, "v": v}
    def message_func(self, edges):
        d = edges.data["d"] 
        rdn_drop = torch.tensor(random.sample(range((d<=self.max_span).sum()), int((d<=self.max_span).sum() * self.edge_drop))).to(device)
        index =torch.tensor(range(len(d))).to(device)
        index = index[d<=self.max_span][rdn_drop]
        drop_v = torch.zeros(len(d)).to(device)
        drop_v[index] = 1
        drop_v = drop_v.view(-1,1)
        q = edges.data["q"] * drop_v
        k = edges.data["k"] * drop_v
        v = edges.data["v"] * drop_v
        return {"q": q, "k": k, "v": v, "ew":edges.data["ew"]}
    # def reduce_func(self, nodes):
    #     # alpha = self.attn_drop(F.softmax(nodes.mailbox["e"], dim=1))
    #     alpha = F.softmax(nodes.mailbox["e"], dim=1)
    #     h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
    #     return {"h": h}
    def reduce_func(self,nodes):
        # print(nodes.mailbox["e"].shape)
        q = nodes.mailbox["q"] #[batch,nodes,feature]
        k = nodes.mailbox["k"]
        v = nodes.mailbox["v"]
        B,N,C = q.shape
        q = q.contiguous().view(B, N, self.heads,C // self.heads).transpose(1,2)
        k = k.contiguous().view(B, N, self.heads,C // self.heads).transpose(1,2)
        v = v.contiguous().view(B, N, self.heads,C // self.heads).transpose(1,2)
        edge_weight = nodes.mailbox["ew"]
        # print(edge_weight.shape)
        alpha = torch.matmul(q, k.permute(0,1,3,2))
        alpha = alpha / th.sqrt(th.tensor(C))
        alpha = F.softmax(alpha, dim = -1) #[B,H,N,N]
        # print(alpha.shape)
        alpha = alpha * edge_weight.view(B,1,1,N)
        context = torch.matmul(alpha, v)
        context = context.permute(0,2,1,3).contiguous()
        context = context.view(B,N,C)
        # print(context.shape)
        h = torch.mean(context, dim = 1)
        # print(alpha.shape)
        return {"h": h}
    def forward(self, g, h):
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise DGLError("There are 0-in-degree nodes in the graph")
            # h_src, h_dst = expand_as_pair(feat, g)
            # g.srcdata["h"] = self.feat_drop(h)
            # g.dstdata["h"] = self.feat_drop(h)
            d = g.edges()[0] - g.edges()[1]
            g.edata['d'] = d.abs()
            g.ndata["x"] = h
            d = g.edges()[1]-g.edges()[0].abs()
            edge_weight = (d.log() - self.cis.log()).sign()
            g.edata["ew"] = edge_weight
            g.apply_edges(self.edge_qkv)
            g.update_all(self.message_func, self.reduce_func)
            return g.ndata.pop("h")


class EdgeConvGAT_noWeight(nn.Module):
    def __init__(self, in_feat, out_feat, feat_drop = 0.5, attn_drop = 0.5, edge_drop = 0.5, batch_norm=False, allow_zero_in_degree=False, cis = torch.tensor(9), n_heads = 4, max_span = 128):
        super(EdgeConvGAT_noWeight,self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.cis = cis
        self.heads = n_heads
        self.max_span = max_span
        ###
        # self.theta = nn.Linear(in_feat, out_feat)
        self.q = nn.Linear(in_feat * 2, out_feat)
        self.k = nn.Linear(in_feat * 2, out_feat)
        self.v = nn.Linear(in_feat * 2, out_feat)
        # self.attn_fc = nn.Linear(2 * out_feat, 1, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)
        self.reset_parameters()
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.k.weight, gain=gain)
        nn.init.xavier_normal_(self.q.weight, gain=gain)
        nn.init.xavier_normal_(self.v.weight, gain=gain)
    def edge_qkv(self, edges):
        #print(edges.src["x"].shape)
        q = self.q(torch.cat([edges.src["x"] - edges.dst["x"], edges.dst["x"]], dim=1))
        k = self.k(torch.cat([edges.src["x"] - edges.dst["x"], edges.dst["x"]], dim=1))
        v = self.v(torch.cat([edges.src["x"] - edges.dst["x"], edges.dst["x"]], dim=1))
        # a = self.attn_fc(z2)
        return {"q": q, "k": k, "v": v}
    def message_func(self, edges):
        d = edges.data["d"] 
        rdn_drop = torch.tensor(random.sample(range((d<=self.max_span).sum()), int((d<=self.max_span).sum() * self.edge_drop))).to(device)
        index =torch.tensor(range(len(d))).to(device)
        index = index[d<=self.max_span][rdn_drop]
        drop_v = torch.zeros(len(d)).to(device)
        drop_v[index] = 1
        drop_v = drop_v.view(-1,1)
        q = edges.data["q"] * drop_v
        k = edges.data["k"] * drop_v
        v = edges.data["v"] * drop_v
        return {"q": q, "k": k, "v": v}
    # def reduce_func(self, nodes):
    #     # alpha = self.attn_drop(F.softmax(nodes.mailbox["e"], dim=1))
    #     alpha = F.softmax(nodes.mailbox["e"], dim=1)
    #     h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
    #     return {"h": h}
    def reduce_func(self,nodes):
        # print(nodes.mailbox["e"].shape)
        q = nodes.mailbox["q"] #[batch,nodes,feature]
        k = nodes.mailbox["k"]
        v = nodes.mailbox["v"]
        B,N,C = q.shape
        q = q.contiguous().view(B, N, self.heads,C // self.heads).transpose(1,2)
        k = k.contiguous().view(B, N, self.heads,C // self.heads).transpose(1,2)
        v = v.contiguous().view(B, N, self.heads,C // self.heads).transpose(1,2)
        # edge_weight = nodes.mailbox["ew"]
        # print(edge_weight.shape)
        alpha = torch.matmul(q, k.permute(0,1,3,2))
        alpha = alpha / th.sqrt(th.tensor(C))
        alpha = F.softmax(alpha, dim = -1) #[B,H,N,N]
        # print(alpha.shape)
        # alpha = alpha * edge_weight.view(B,1,1,N)
        context = torch.matmul(alpha, v)
        context = context.permute(0,2,1,3).contiguous()
        context = context.view(B,N,C)
        # print(context.shape)
        h = torch.mean(context, dim = 1)
        # print(alpha.shape)
        return {"h": h}
    def forward(self, g, h):
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise DGLError("There are 0-in-degree nodes in the graph")
            # h_src, h_dst = expand_as_pair(feat, g)
            # g.srcdata["h"] = self.feat_drop(h)
            # g.dstdata["h"] = self.feat_drop(h)
            d = g.edges()[0] - g.edges()[1]
            g.edata['d'] = d.abs()
            g.ndata["x"] = h
            # d = g.edges()[1]-g.edges()[0].abs()
            # edge_weight = (d.log() - self.cis.log()).sign()
            # g.edata["ew"] = edge_weight
            g.apply_edges(self.edge_qkv)
            g.update_all(self.message_func, self.reduce_func)
            return g.ndata.pop("h")


class EdgeConvGAT_noWeight_noPos(nn.Module):
    def __init__(self, in_feat, out_feat, feat_drop = 0.5, attn_drop = 0.5, edge_drop = 0.5, batch_norm=False, allow_zero_in_degree=False, cis = torch.tensor(9), n_heads = 4, max_span = 128):
        super(EdgeConvGAT_noWeight_noPos,self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.cis = cis
        self.heads = n_heads
        self.max_span = max_span
        ###
        # self.theta = nn.Linear(in_feat, out_feat)
        self.q = nn.Linear(in_feat * 2, out_feat)
        self.k = nn.Linear(in_feat * 2, out_feat)
        self.v = nn.Linear(in_feat * 2, out_feat)
        # self.attn_fc = nn.Linear(2 * out_feat, 1, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)
        self.reset_parameters()
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.k.weight, gain=gain)
        nn.init.xavier_normal_(self.q.weight, gain=gain)
        nn.init.xavier_normal_(self.v.weight, gain=gain)
    def edge_qkv(self, edges):
        #print(edges.src["x"].shape)
        q = self.q(torch.cat([edges.src["x"] - edges.dst["x"], edges.dst["x"]], dim=1))
        k = self.k(torch.cat([edges.src["x"] - edges.dst["x"], edges.dst["x"]], dim=1))
        v = self.v(torch.cat([edges.src["x"] - edges.dst["x"], edges.dst["x"]], dim=1))
        # a = self.attn_fc(z2)
        return {"q": q, "k": k, "v": v}
    def message_func(self, edges):
        d = edges.data["d"] 
        rdn_drop = torch.tensor(random.sample(range((d<=self.max_span).sum()), int((d<=self.max_span).sum() * self.edge_drop))).to(device)
        index =torch.tensor(range(len(d))).to(device)
        index = index[d<=self.max_span][rdn_drop]
        drop_v = torch.zeros(len(d)).to(device)
        drop_v[index] = 1
        drop_v = drop_v.view(-1,1)
        q = edges.data["q"] * drop_v
        k = edges.data["k"] * drop_v
        v = edges.data["v"] * drop_v
        return {"q": q, "k": k, "v": v}
    # def reduce_func(self, nodes):
    #     # alpha = self.attn_drop(F.softmax(nodes.mailbox["e"], dim=1))
    #     alpha = F.softmax(nodes.mailbox["e"], dim=1)
    #     h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
    #     return {"h": h}
    def reduce_func(self,nodes):
        # print(nodes.mailbox["e"].shape)
        q = nodes.mailbox["q"] #[batch,nodes,feature]
        k = nodes.mailbox["k"]
        v = nodes.mailbox["v"]
        B,N,C = q.shape
        q = q.contiguous().view(B, N, self.heads,C // self.heads).transpose(1,2)
        k = k.contiguous().view(B, N, self.heads,C // self.heads).transpose(1,2)
        v = v.contiguous().view(B, N, self.heads,C // self.heads).transpose(1,2)
        # edge_weight = nodes.mailbox["ew"]
        # print(edge_weight.shape)
        alpha = torch.matmul(q, k.permute(0,1,3,2))
        alpha = alpha / th.sqrt(th.tensor(C))
        alpha = F.softmax(alpha, dim = -1) #[B,H,N,N]
        # print(alpha.shape)
        # alpha = alpha * edge_weight.view(B,1,1,N)
        context = torch.matmul(alpha, v)
        context = context.permute(0,2,1,3).contiguous()
        context = context.view(B,N,C)
        # print(context.shape)
        h = torch.mean(context, dim = 1)
        # print(alpha.shape)
        return {"h": h}
    def forward(self, g, h):
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise DGLError("There are 0-in-degree nodes in the graph")
            # h_src, h_dst = expand_as_pair(feat, g)
            # g.srcdata["h"] = self.feat_drop(h)
            # g.dstdata["h"] = self.feat_drop(h)
            d = g.edges()[0] - g.edges()[1]
            g.edata['d'] = d.abs()
            if h.shape[1] == 14:
                h = h[:,2:]
            g.ndata["x"] = h
            # d = g.edges()[1]-g.edges()[0].abs()
            # edge_weight = (d.log() - self.cis.log()).sign()
            # g.edata["ew"] = edge_weight
            g.apply_edges(self.edge_qkv)
            g.update_all(self.message_func, self.reduce_func)
            return g.ndata.pop("h")

class EdgeConvGAT_noWeight_onlyPos(nn.Module):
    def __init__(self, in_feat, out_feat, feat_drop = 0.5, attn_drop = 0.5, edge_drop = 0.5, batch_norm=False, allow_zero_in_degree=False, cis = torch.tensor(9), n_heads = 4, max_span = 128):
        super(EdgeConvGAT_noWeight_onlyPos,self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.cis = cis
        self.heads = n_heads
        self.max_span = max_span
        ###
        # self.theta = nn.Linear(in_feat, out_feat)
        self.q = nn.Linear(in_feat * 2, out_feat)
        self.k = nn.Linear(in_feat * 2, out_feat)
        self.v = nn.Linear(in_feat * 2, out_feat)
        # self.attn_fc = nn.Linear(2 * out_feat, 1, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)
        self.reset_parameters()
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.k.weight, gain=gain)
        nn.init.xavier_normal_(self.q.weight, gain=gain)
        nn.init.xavier_normal_(self.v.weight, gain=gain)
    def edge_qkv(self, edges):
        #print(edges.src["x"].shape)
        q = self.q(torch.cat([edges.src["x"] - edges.dst["x"], edges.dst["x"]], dim=1))
        k = self.k(torch.cat([edges.src["x"] - edges.dst["x"], edges.dst["x"]], dim=1))
        v = self.v(torch.cat([edges.src["x"] - edges.dst["x"], edges.dst["x"]], dim=1))
        # a = self.attn_fc(z2)
        return {"q": q, "k": k, "v": v}
    def message_func(self, edges):
        d = edges.data["d"] 
        rdn_drop = torch.tensor(random.sample(range((d<=self.max_span).sum()), int((d<=self.max_span).sum() * self.edge_drop))).to(device)
        index =torch.tensor(range(len(d))).to(device)
        index = index[d<=self.max_span][rdn_drop]
        drop_v = torch.zeros(len(d)).to(device)
        drop_v[index] = 1
        drop_v = drop_v.view(-1,1)
        q = edges.data["q"] * drop_v
        k = edges.data["k"] * drop_v
        v = edges.data["v"] * drop_v
        return {"q": q, "k": k, "v": v}
    # def reduce_func(self, nodes):
    #     # alpha = self.attn_drop(F.softmax(nodes.mailbox["e"], dim=1))
    #     alpha = F.softmax(nodes.mailbox["e"], dim=1)
    #     h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
    #     return {"h": h}
    def reduce_func(self,nodes):
        # print(nodes.mailbox["e"].shape)
        q = nodes.mailbox["q"] #[batch,nodes,feature]
        k = nodes.mailbox["k"]
        v = nodes.mailbox["v"]
        B,N,C = q.shape
        q = q.contiguous().view(B, N, self.heads,C // self.heads).transpose(1,2)
        k = k.contiguous().view(B, N, self.heads,C // self.heads).transpose(1,2)
        v = v.contiguous().view(B, N, self.heads,C // self.heads).transpose(1,2)
        # edge_weight = nodes.mailbox["ew"]
        # print(edge_weight.shape)
        alpha = torch.matmul(q, k.permute(0,1,3,2))
        alpha = alpha / th.sqrt(th.tensor(C))
        alpha = F.softmax(alpha, dim = -1) #[B,H,N,N]
        # print(alpha.shape)
        # alpha = alpha * edge_weight.view(B,1,1,N)
        context = torch.matmul(alpha, v)
        context = context.permute(0,2,1,3).contiguous()
        context = context.view(B,N,C)
        # print(context.shape)
        h = torch.mean(context, dim = 1)
        # print(alpha.shape)
        return {"h": h}
    def forward(self, g, h):
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise DGLError("There are 0-in-degree nodes in the graph")
            # h_src, h_dst = expand_as_pair(feat, g)
            # g.srcdata["h"] = self.feat_drop(h)
            # g.dstdata["h"] = self.feat_drop(h)
            d = g.edges()[0] - g.edges()[1]
            g.edata['d'] = d.abs()
            if h.shape[1] == 14:
                h = h[:,:2]
            g.ndata["x"] = h
            # d = g.edges()[1]-g.edges()[0].abs()
            # edge_weight = (d.log() - self.cis.log()).sign()
            # g.edata["ew"] = edge_weight
            g.apply_edges(self.edge_qkv)
            g.update_all(self.message_func, self.reduce_func)
            return g.ndata.pop("h")


