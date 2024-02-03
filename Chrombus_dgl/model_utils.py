# %%
import os
os.chdir("/shareN/data9/UserData/yuanyuan/GRN/scripts/")
os.environ["DGLBACKEND"] = "pytorch"
# %%
import dgl
import numpy as np
import torch
import torch as th
from dgl.nn import EdgeConv,GATv2Conv,GraphConv
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import edge_conv_model
import decode_model
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# %%
def get_model(conv_model_name, in_dim, out_dim, num_heads,decoder = "dpp"):
    edgeconv = getattr(edge_conv_model, conv_model_name)
    if decoder == "mlp":
        Predictor = getattr(decode_model, "MLPPredictor")
    else:
        Predictor = getattr(decode_model, "DotProductPredictor")
    class chrombus(nn.Module):
        def __init__(self, in_dim = in_dim, out_dim = out_dim, num_heads = num_heads):
            super(chrombus, self).__init__()
            self.layer1 = edgeconv(in_dim, out_dim, n_heads = num_heads)
            self.layer2 = edgeconv(out_dim, out_dim // 2, n_heads = num_heads)
            self.layer3 = edgeconv(out_dim // 2, out_dim, n_heads = num_heads)
            self.norm1 = nn.BatchNorm1d(out_dim)
            self.norm2 = nn.BatchNorm1d(out_dim // 2)
            self.norm3 = nn.BatchNorm1d(out_dim)
            # self.dropout = nn.Dropout(0.4)
            self.pred = Predictor(out_dim)
        def forward(self, g, h):
            h = self.layer1(g,h)
            h = self.norm1(h)
            h0 = F.elu(h)
            h = self.layer2(g,h0)
            # h = self.dropout(h)
            h = self.norm2(h)
            h = F.elu(h)
            h = self.layer3(g,h)
            h = self.norm3(h)
            h = h + h0
            score = self.pred(g,h)
            return score,h
    model = chrombus(in_dim,out_dim,num_heads)
    return model

def get_baseline(conv_model_name, in_dim, out_dim):
    edgeconv = getattr(dgl.nn, conv_model_name)
    Predictor = getattr(decode_model, "DotProductPredictor")
    class baseline(nn.Module):
        def __init__(self, in_dim = in_dim, out_dim = out_dim):
            super(baseline, self).__init__()
            if conv_model_name == "GATv2Conv":
                self.layer1 = edgeconv(in_dim, out_dim, 1)
                self.layer2 = edgeconv(out_dim, out_dim // 2, 1)
                self.layer3 = edgeconv(out_dim // 2, out_dim, 1)            
            else:
                self.layer1 = edgeconv(in_dim, out_dim)
                self.layer2 = edgeconv(out_dim, out_dim // 2)
                self.layer3 = edgeconv(out_dim // 2, out_dim)
            self.norm1 = nn.BatchNorm1d(out_dim)
            self.norm2 = nn.BatchNorm1d(out_dim // 2)
            self.norm3 = nn.BatchNorm1d(out_dim)
            # self.dropout = nn.Dropout(0.4)
            self.pred = Predictor(out_dim)
        def forward(self, g, h):
            h = self.layer1(g,h)
            # h = self.norm1(h)
            h0 = F.elu(h)
            h = self.layer2(g,h0)
            # h = self.dropout(h)
            # h = self.norm2(h)
            h = F.elu(h)
            h = self.layer3(g,h)
            # h = self.norm3(h)
            h = h + h0
            score = self.pred(g,h)
            return score,h
    model = baseline(in_dim,out_dim)
    return model


# # %%
# model = get_model("EdgeConvGAT",14,32,4)


def train(model,loader,lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for batched_graph in loader:
        batched_graph = batched_graph.to(device)
        #label = torch.tensor(batched_graph.edata["label"], dtype=torch.float32)
        score,_ = model(batched_graph, batched_graph.ndata["x"])
        index = batched_graph.edata["label"] > -2.5
        score = score.view(-1)[index]
        label = batched_graph.edata["label"][index]
        loss = ((score - label) ** 2).mean()
        # loss = ((score.view(-1) - batched_graph.edata["label"]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

def test(model, loader):
    model.eval()
    total_loss = 0
    pred_cor = 0
    for batched_graph in loader:
        batched_graph = batched_graph.to(device)
        score,_ = model(batched_graph, batched_graph.ndata["x"])
        index = batched_graph.edata["label"] > -2.5
        score = score.view(-1)[index]
        label = batched_graph.edata["label"][index]
        loss = ((score - label) ** 2).mean()
        total_loss += loss.cpu().detach()
        df = pd.DataFrame({'hic':label.cpu(), 'pred':score.cpu().detach()})
        pred_cor += df.corr().iloc[0,1]
    return total_loss / len(loader), pred_cor / len(loader)


def train_model(net,trainloader,testloader, epochs = 300,lr = 1e-2, filepath = ''):
    net.to(device)
    #opt = torch.optim.Adam(net.parameters(), lr=lr)
    writer = SummaryWriter(filepath + '/runs/chrombus_experiment_'+'300_epochs')
    for epoch in range(epochs):
        train(net,trainloader,lr = lr)
        train_mse,train_cor = test(net,trainloader)
        test_mse,test_cor = test(net,testloader)
        if (test_cor > 0.7) & (train_cor > 0.7):
            torch.save(net.state_dict(), filepath + '/model_epoch' + str(epoch) + '.chrom' + '.chrombus.pkl')
        print('Epoch: {}, Train Loss:{:.3f}, Train Cor:{:.3f}, Test Loss:{:.3f}, Test Cor:{:3f}'.format(epoch, train_mse.item(),train_cor, test_mse.item(), test_cor))
        writer.add_scalar('Loss train',train_mse,epoch) # new line
        writer.add_scalar('COR train',train_cor,epoch)   # new line
        writer.add_scalar('Loss test',test_mse,epoch) # new line
        writer.add_scalar('COR test',test_cor,epoch)   # new line
    torch.save(net.state_dict(), filepath + '/model_epoch' + str(epoch) + '.chrombus.pkl')



