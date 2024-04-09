# %%
from Chrombus_pyG import chrombus_pyG
from torch_geometric.nn import GAE
import numpy as np
import pandas
import torch
from torch_geometric.utils import to_undirected
import copy
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from Chrombus_pyG.generate_segments_for_sequence_cross import process_data
from Chrombus_pyG.generate_segments_for_sequence_cross import process_data_singlechrom
import os
import math
from matplotlib import pyplot as plt

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

def get_models(n_heads = 8, in_channels = 14, out_channels = 32, cis_span = 9):
    encoder = getattr(chrombus_pyG, "EdgeConvEncoder")
    decoder = getattr(chrombus_pyG, "EdgeConvDecoder")
    model = GAE(encoder=encoder(n_heads, in_channels, out_channels, cis_span), decoder=decoder())
    return model

def getEdgeIndex(n_seg=128, max_span=64, batch=None):
    index_all = np.zeros((2,0),dtype='int32')
    for i in range(max(batch) + 1):
        index = np.vstack((np.arange(n_seg).repeat(n_seg),np.tile(np.arange(n_seg),n_seg)))
        index = index[:,index[0] < index[1]]
        index = index[:,(index[1] - index[0] <= max_span)]
        index = index + (i * n_seg)
        index_all = np.concatenate((index_all, index),axis=1)
    index_all = torch.tensor(index_all)
    return to_undirected(index_all)


def train(model,loader,max_span = 64,lr = 1E-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.train()
    running_loss = 0
    for data in loader: # Iterate in batches over the training dataset.
        data = data.to(device)
        pred_edge_index = getEdgeIndex(128, max_span, batch=data.batch).to(device)
        z = model.encode(data.x, pred_edge_index, data.batch) #encoding output        
        pred = model.decoder(z, data.edge_index) 
        loss = criterion(pred, data.edge_attr.view(-1)) # Compute the loss.
        loss.backward() # Derive gradients.  
        optimizer.step() # Update parameters based on gradients.
        optimizer.zero_grad() # Clear gradients.
        running_loss += loss
    #print(f'total loss: {(running_loss/len(loader)):4f}')

def test(model,loader,max_span = 64):
    criterion = torch.nn.MSELoss()
    model.eval()
    mse = 0
    pred_cor = 0
    for data in loader: # Iterate in batches over the training/test dataset.
        data = data.to(device)
        pred_edge_index = getEdgeIndex(128, max_span, batch=data.batch).to(device)
        z = model.encode(data.x, pred_edge_index, data.batch) 
        pred = model.decoder(z, data.edge_index) 
        loss = criterion(pred, data.edge_attr.view(-1)) # Check against ground-truth labels.
        mse += loss.cpu().detach()
        df = pandas.DataFrame({'hic':data.edge_attr[:,0].cpu(), 'pred':pred.cpu().detach()})
        pred_cor += df.corr().iloc[0,1]
    return mse / len(loader), pred_cor / len(loader) # Derive ratio of correct predictions.


def train_model(model,trainloader,testloader, epochs = 400, max_span = 64,lr = 1e-2, filepath = ''):
    model.to(device)
    for epoch in range(epochs):
        train(model,trainloader, max_span=max_span,lr = lr)
        train_mse,train_cor = test(model,trainloader, max_span=max_span)
        test_mse,test_cor = test(model,testloader, max_span=max_span)
        if (test_cor > 0.7) & (train_cor > 0.7):
            torch.save(model.state_dict(), filepath + '/model_epoch' + str(epoch) + '.chrombus.pkl')
        print('Epoch: {}, Train Loss:{:.3f}, Train Cor:{:.3f}, Test Loss:{:.3f}, Test Cor:{:3f}'.format(epoch, train_mse.item(),train_cor, test_mse.item(), test_cor))
    torch.save(model.state_dict(), filepath + '/model_epoch' + str(epoch) + '.chrombus.pkl')


def load_chrombus_data(datapath,test_chr,outpath,batch_size = 16):
    if os.path.exists(outpath + '/chr' + str(test_chr) + '_epi_traindata.pt'):
        train_dataset = torch.load(outpath + '/chr' + str(test_chr) + '_epi_traindata.pt')
        test_dataset = torch.load(outpath + '/chr' + str(test_chr) + '_epi_testdata.pt')
    else:
        process_data(dataset="test", testchr = test_chr, datapath = datapath, outpath = outpath, n_seg = 128, N_chr = 50)
        process_data(dataset="train", testchr = test_chr, datapath = datapath, outpath = outpath, n_seg = 128, N_chr = 50)
        train_dataset = torch.load(outpath + '/chr' + str(test_chr) + '_epi_traindata.pt')
        test_dataset = torch.load(outpath + '/chr' + str(test_chr) + '_epi_testdata.pt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader

def load_chrombus_data_singlechrom(datapath,chrom,outpath,batch_size = 16):
    if os.path.exists(outpath + '/chr' + str(chrom) + '_epi_data.pt'):
        dataset = torch.load(outpath + '/chr' + str(chrom) + '_epi_data.pt')
    else:
        process_data_singlechrom(chr = chrom, datapath = datapath, outpath = outpath, n_seg = 128, N_chr = 50)
        dataset = torch.load(outpath + '/chr' + str(chrom) + '_epi_data.pt')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def get_region_pred(model_path, chrom, start, end, datapath, outpath, max_span = 64):
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    chr_len = [0,248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622, 133275309,114364328, 107043718, 101991189,  90338345,  83257441,  80373285,  58617616, 64444167, 46709983,50818468]
    model = get_models()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    ndata = pandas.read_csv(datapath + '/V_chr'+str(chrom)+'.csv')
    edata = pandas.read_csv(datapath + '/chr'+str(chrom)+'_edges.csv')
    edata = edata.iloc[:,:3]
    edata.columns = ['from','to','hic']
    #
    idx0 = ndata[(ndata.v_start * chr_len[chrom] <= start) & (ndata.v_end * chr_len[chrom] >= start)].v_index.values[0]
    idx1 = ndata[(ndata.v_start * chr_len[chrom] <= end) & (ndata.v_end * chr_len[chrom] >= end)].v_index.values[0]
    n = int((128 - (idx1 - idx0)) / 2)
    pos0 = idx0 - n
    if pos0 < 0:
        pos0 = 0
    if pos0 + 128 > ndata.v_index.max():
        pos0 = ndata.v_index.max() - 128
    pos1 = pos0 + 128
    #
    ndata1 = copy.deepcopy(ndata)
    ndata1.columns = [*range(ndata1.shape[1])]
    x = ndata1[pos0:pos1]
    x = torch.tensor(x[[*range(2,16)]].values, dtype=torch.float)
    edge_index = torch.tensor(np.vstack((np.arange(128).repeat(128),np.tile(np.arange(128),128))))
    data = Data(x=x, y=torch.tensor([1]), edge_index=edge_index)
    dataloader = DataLoader([data], batch_size=1, shuffle=False)
    data = next(iter(dataloader))
    data = data.to(device)
    #
    pred_edge_index = getEdgeIndex(128, max_span, batch=data.batch).to(device)
    z = model.encode(data.x, pred_edge_index, data.batch)
    pred = model.decoder(z, data.edge_index)
    df = pandas.DataFrame({'from':data.edge_index[0].cpu(), 'to':data.edge_index[1].cpu(), 'pred':pred.detach().cpu()})
    df = df[df['from'] <= df['to']]
    df['from'] = df['from'] + pos0
    df['to'] = df['to'] + pos0
    #
    df.reset_index(inplace=True)
    from_pos = ndata.iloc[df['from'].values,:4]
    from_pos.reset_index(inplace = True)
    to_pos = ndata.iloc[df['to'].values,2:4]
    to_pos.reset_index(inplace=True)
    result = pd.concat([df.iloc[:,1:],from_pos.iloc[:,2:],to_pos.iloc[:,1:]], axis = 1)
    result.columns = ['from', 'to', 'pred', 'v_chr','v_start1','v_end1','v_start2','v_end2']
    result['dist'] = (result['v_start2'] - result['v_end1']) * chr_len[chrom]
    result['chrom'] = chrom
    result = pd.merge(result, edata,how = 'left',on = ['from','to'])
    result.to_csv(outpath +'/chr'+str(chrom)+ ":" + str(start) + "-" + str(end) +'_pred.csv',index=None)
    return result

def get_pred(model_path,chrom,datapath, outpath,max_span = 64):
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    chr_len = [0,248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622, 133275309,114364328, 107043718, 101991189,  90338345,  83257441,  80373285,  58617616, 64444167, 46709983,50818468]
    model = get_models()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    ndata = pd.read_csv(datapath + '/V_chr'+str(chrom)+'.csv')
    edata = pd.read_csv(datapath + '/chr'+str(chrom)+'_edges.csv')
    edata = edata.iloc[:,:3]
    edata.columns = ['from','to','hic']
    chr_df = pd.DataFrame()
    for pos0 in [*range(0, ndata.v_index.max() - 128,9)]:
        if pos0 + 128 >= (ndata.v_index.max() - 128):
            pos0 = ndata.v_index.max() - 128
        pos1 = pos0 + 128
        ndata1 = copy.deepcopy(ndata)
        ndata1.columns = [*range(ndata1.shape[1])]
        x = ndata1[pos0:pos1]
        x = torch.tensor(x[[*range(2,16)]].values, dtype=torch.float)
        # data = next(iter(test_loader))
        # data.x = x
        edge_index = torch.tensor(np.vstack((np.arange(128).repeat(128),np.tile(np.arange(128),128))))
        edge_index = edge_index[:,edge_index[0] <= edge_index[1]]
        data = Data(x=x, y=torch.tensor([1]), edge_index=edge_index)
        dataloader = DataLoader([data], batch_size=1, shuffle=False)
        data = next(iter(dataloader))
        # data.edge_index = edge_index
        data = data.to(device)
        #
        pred_edge_index = getEdgeIndex(128, max_span, batch=data.batch).to(device)
        z = model.encode(data.x, pred_edge_index, data.batch)
        pred = model.decoder(z, data.edge_index)
        df = pd.DataFrame({'from':data.edge_index[0].cpu(), 'to':data.edge_index[1].cpu(), 'pred':pred.detach().cpu()})
        df = df[df['from'] < df['to']]
        df['from'] = df['from'] + pos0
        df['to'] = df['to'] + pos0
        chr_df = pd.concat([chr_df, df])
        #
    chr_df = pd.DataFrame(chr_df.groupby(['from','to'],as_index=False)['pred'].mean())
    chr_df.reset_index(inplace=True)
    from_pos = ndata.iloc[chr_df['from'].values,:4]
    from_pos.reset_index(inplace = True)
    to_pos = ndata.iloc[chr_df['to'].values,2:4]
    to_pos.reset_index(inplace=True)
    result = pd.concat([chr_df.iloc[:,1:],from_pos.iloc[:,2:],to_pos.iloc[:,1:]], axis = 1)
    result.columns = ['from', 'to', 'pred', 'v_chr','v_start1','v_end1','v_start2','v_end2']
    result['dist'] = (result['v_start2'] - result['v_end1']) * chr_len[chrom]
    result['chrom'] = chrom
    result = pd.merge(result, edata,how = 'left',on = ['from','to'])
    result.to_csv(outpath + 'chrombus_pred.chr' + str(chrom) + '.csv',index=None)


def hic_heatmap(result, chrom, outpath):
    #
    chr_len = [0,248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622, 133275309,114364328, 107043718, 101991189,  90338345,  83257441,  80373285,  58617616, 64444167, 46709983,50818468]
    start = math.ceil(result[['v_start1']].values[0][0] * chr_len[chrom] / 5000) * 5000
    end = int(result[['v_end1']].values[-1][0] * chr_len[chrom] / 5000) * 5000
    [start, end]
    nrow = int((end - start) / 5000) + 1
    init_value0 = np.nan
    #
    mat0 = np.array([init_value0] * nrow * nrow).reshape([nrow, nrow])
    mat1 = np.array([init_value0] * nrow * nrow).reshape([nrow, nrow])
    for i, row in result.iterrows():
        x1 = int((row.iloc[4] * chr_len[chrom] - start) / 5000)
        x2 = int((row.iloc[5] * chr_len[chrom] - start) / 5000)
        y1 = int((row.iloc[6] * chr_len[chrom] - start) / 5000)
        y2 = int((row.iloc[7] * chr_len[chrom] - start) / 5000)
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > nrow:
            x2 = nrow
        if y2 > nrow:
            y2 = nrow
        mat0[x1:x2, y1:y2] = row.iloc[10]
        mat0[y1:y2, x1:x2] = row.iloc[10]
        mat1[y1:y2, x1:x2] = row.iloc[2]
        mat1[x1:x2, y1:y2] = row.iloc[2]
    ### plot
    plt.figure(figsize=(9,4))
    plt.subplot(121) 
    im = plt.matshow(mat0, fignum=False, vmax = 5, cmap = 'Reds')
    plt.colorbar(im, fraction=.04, pad = 0.02)
    plt.axis('off')
    plt.title("chr"+str(chrom)+":"+str(int(start/1e6))+"M - "+str(int(end/1e6))+"M")
    plt.subplot(122) 
    im = plt.matshow(mat1, fignum=False, vmax = 5, cmap = 'Reds')
    plt.colorbar(im, fraction=.04, pad = 0.02)
    plt.axis('off')
    plt.title("chr"+str(chrom)+":"+str(int(start/1e6))+"M - "+str(int(end/1e6))+"M")
    plt.savefig(outpath + "/chr" + str(chrom)+":"+str(int(start/1e6))+"M - "+str(int(end/1e6))+"M" + '.png')

