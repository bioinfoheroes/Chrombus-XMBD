from typing import Optional, Callable, List
import os
import os.path as osp
import shutil
import pandas
import torch
import numpy
from torch_geometric.data import Data
#from torch_geometric.data import InMemoryDataset, download_url, extract_zip
import random
from torch_geometric.utils import to_undirected

#generate dataset from whole-genome HiC data
def process_data(dataset, testchr, datapath, outpath, n_seg = 128, N_chr = 50):
    min_interact = -10
    data_list = []
    rand_chr = []
    pos_list = []
    if dataset == "train":
        # rand_chr = [8, 22, 7, 11, 14, 1, 21, 2, 19, 20, 5]
        rand_chr = [i for i in [*range(1,23)] if i != testchr]
        # rand_chr = [8, 22, 7, 11]
    elif dataset == "test":
        # rand_chr = [3, 9, 12, 13, 15, 16,17]
        rand_chr = [testchr]
        # rand_chr = [3, 12]
    else:
        print("error1")
    for i in rand_chr:
        print(i)
        #pos_dict[i] = []     ## for extracting sequence
        n_data_all = f'{datapath}/V_chr{i}.csv'
        e_data_all = f'{datapath}/chr{i}_edges.csv'
        nodes_all = pandas.read_csv(n_data_all, sep=',')
        nodes_all.columns = [*range(0, nodes_all.shape[1])]
        edges_all = pandas.read_csv(e_data_all, sep=',')
        edges_all.columns = [*range(0, edges_all.shape[1])]
        edges_all = edges_all[edges_all[0] != edges_all[1]]
        if edges_all.shape[0] == 0:
            continue
        n_nodes = range(nodes_all[0].max() - n_seg + 1)
        if N_chr < max(n_nodes):
            n_pos = random.sample(n_nodes, N_chr)
        else:
            n_pos = random.sample(n_nodes, int(max(n_nodes) * 0.6))
        # n_pos = pos_dict[i]
        for pos in n_pos:
            pos1 = pos + n_seg
            #get node features
            nodes = nodes_all[pos:pos1]
            nodes = nodes.reset_index(drop = True)
            if nodes.shape[0] < n_seg:
                continue
            #node classification labels ...                
            #y = [TADparser(nodes[16][i]) for i in range(nodes.shape[0])]
            #y = torch.tensor(y, dtype=torch.long)
            offset1 = nodes[0].min()
            nodes[0] = nodes[0] - offset1
            x = torch.tensor(nodes[[*range(2,16)]].values, dtype=torch.float)
            pos_list.append([i, pos])
            #get edges
            edges = edges_all[(edges_all[0] >= pos) & (edges_all[0] < pos1) & (edges_all[1] >= pos) & (edges_all[1] < pos1)]
            edges = edges.reset_index(drop=True)
            v_from = torch.tensor(edges[0].values, dtype=torch.long)
            v_to = torch.tensor(edges[1].values, dtype=torch.long)
            e_attribute = torch.tensor(edges[2].values, dtype=torch.float)
            #offset = v_from.numpy().min()
            v_from = v_from - offset1
            v_to = v_to - offset1
            indx = torch.tensor(edges[2].values > min_interact)
            edge_index = torch.cat([torch.stack([v_from[indx], v_to[indx]]), torch.stack([v_to[indx], v_from[indx]])], axis = 1)
            edge_attr = torch.reshape(torch.cat([e_attribute[indx],e_attribute[indx]]), (-1, 1))
            g1 = Data(x=x, y=torch.tensor([1]), edge_index=edge_index, edge_attr=edge_attr)
            data_list = data_list + [g1]
    torch.save(data_list, outpath + '/chr' + str(testchr) + '_epi_' + str(dataset)+ 'data.pt')
    numpy.save(outpath + '/chr' + str(testchr) + '_' + str(dataset) + '_pos_list.npy', pos_list)


def process_data_singlechrom(chr, datapath, outpath, n_seg = 128, N_chr = 50):
    min_interact = -10
    data_list = []
    pos_list = []
    for i in [chr]:
        print(i)
        #pos_dict[i] = []     ## for extracting sequence
        n_data_all = f'{datapath}/V_chr{i}.csv'
        e_data_all = f'{datapath}/chr{i}_edges.csv'
        nodes_all = pandas.read_csv(n_data_all, sep=',')
        nodes_all.columns = [*range(0, nodes_all.shape[1])]
        edges_all = pandas.read_csv(e_data_all, sep=',')
        edges_all.columns = [*range(0, edges_all.shape[1])]
        edges_all = edges_all[edges_all[0] != edges_all[1]]
        if edges_all.shape[0] == 0:
            continue
        n_nodes = range(nodes_all[0].max() - n_seg + 1)
        if N_chr < max(n_nodes):
            n_pos = random.sample(n_nodes, N_chr)
        else:
            n_pos = random.sample(n_nodes, int(max(n_nodes) * 0.6))
        # n_pos = pos_dict[i]
        for pos in n_pos:
            pos1 = pos + n_seg
            #get node features
            nodes = nodes_all[pos:pos1]
            nodes = nodes.reset_index(drop = True)
            if nodes.shape[0] < n_seg:
                continue
            #node classification labels ...                
            #y = [TADparser(nodes[16][i]) for i in range(nodes.shape[0])]
            #y = torch.tensor(y, dtype=torch.long)
            offset1 = nodes[0].min()
            nodes[0] = nodes[0] - offset1
            x = torch.tensor(nodes[[*range(2,16)]].values, dtype=torch.float)
            pos_list.append([i, pos])
            #get edges
            edges = edges_all[(edges_all[0] >= pos) & (edges_all[0] < pos1) & (edges_all[1] >= pos) & (edges_all[1] < pos1)]
            edges = edges.reset_index(drop=True)
            v_from = torch.tensor(edges[0].values, dtype=torch.long)
            v_to = torch.tensor(edges[1].values, dtype=torch.long)
            e_attribute = torch.tensor(edges[2].values, dtype=torch.float)
            #offset = v_from.numpy().min()
            v_from = v_from - offset1
            v_to = v_to - offset1
            indx = torch.tensor(edges[2].values > min_interact)
            edge_index = torch.cat([torch.stack([v_from[indx], v_to[indx]]), torch.stack([v_to[indx], v_from[indx]])], axis = 1)
            edge_attr = torch.reshape(torch.cat([e_attribute[indx],e_attribute[indx]]), (-1, 1))
            g1 = Data(x=x, y=torch.tensor([1]), edge_index=edge_index, edge_attr=edge_attr)
            data_list = data_list + [g1]
    torch.save(data_list, outpath + '/chr' + str(chr) + '_epi_' + 'data.pt')
    numpy.save(outpath + '/chr' + str(chr) + '_'  + '_pos_list.npy', pos_list)


def process_data_singlechrom_pe(chr, datapath, outpath, n_seg = 128, N_chr = 50):
    min_interact = -10
    data_list = []
    pos_list = []
    for i in [chr]:
        print(i)
        #pos_dict[i] = []     ## for extracting sequence
        n_data_all = f'{datapath}/V_chr{i}.csv'
        e_data_all = f'{datapath}/chr{i}_edges.csv'
        nodes_all = pandas.read_csv(n_data_all, sep=',')
        nodes_all.columns = [*range(0, nodes_all.shape[1])]
        edges_all = pandas.read_csv(e_data_all, sep=',')
        edges_all.columns = [*range(0, edges_all.shape[1])]
        edges_all = edges_all[edges_all[0] != edges_all[1]]
        if edges_all.shape[0] == 0:
            continue
        n_nodes = range(nodes_all[0].max() - n_seg + 1)
        if N_chr < max(n_nodes):
            n_pos = random.sample(n_nodes, N_chr)
        else:
            n_pos = random.sample(n_nodes, int(max(n_nodes) * 0.6))
        # n_pos = pos_dict[i]
        for pos in n_pos:
            pos1 = pos + n_seg
            #get node features
            nodes = nodes_all[pos:pos1]
            nodes = nodes.reset_index(drop = True)
            if nodes.shape[0] < n_seg:
                continue
            #node classification labels ...                
            #y = [TADparser(nodes[16][i]) for i in range(nodes.shape[0])]
            #y = torch.tensor(y, dtype=torch.long)
            offset1 = nodes[0].min()
            nodes[0] = nodes[0] - offset1
            x = torch.tensor(nodes[[*range(2,16)]].values, dtype=torch.float)
            x = torch.cat([x,torch.arange(n_seg)[:,None]],dim=-1)
            pos_list.append([i, pos])
            #get edges
            edges = edges_all[(edges_all[0] >= pos) & (edges_all[0] < pos1) & (edges_all[1] >= pos) & (edges_all[1] < pos1)]
            edges = edges.reset_index(drop=True)
            v_from = torch.tensor(edges[0].values, dtype=torch.long)
            v_to = torch.tensor(edges[1].values, dtype=torch.long)
            e_attribute = torch.tensor(edges[2].values, dtype=torch.float)
            #offset = v_from.numpy().min()
            v_from = v_from - offset1
            v_to = v_to - offset1
            indx = torch.tensor(edges[2].values > min_interact)
            edge_index = torch.cat([torch.stack([v_from[indx], v_to[indx]]), torch.stack([v_to[indx], v_from[indx]])], axis = 1)
            edge_attr = torch.reshape(torch.cat([e_attribute[indx],e_attribute[indx]]), (-1, 1))
            g1 = Data(x=x, y=torch.tensor([1]), edge_index=edge_index, edge_attr=edge_attr)
            data_list = data_list + [g1]
    torch.save(data_list, outpath + '/chr' + str(chr) + '_epi_pe_' + 'data.pt')
    numpy.save(outpath + '/chr' + str(chr) + '_'  + '_pos_list.pe.npy', pos_list)
