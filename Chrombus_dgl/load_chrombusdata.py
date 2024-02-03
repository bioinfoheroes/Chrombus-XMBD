import dgl
from dgl.data import DGLDataset
import pandas
import random
import torch
import os
import numpy as np
os.environ["DGLBACKEND"] = "pytorch"
from dgl.data.utils import save_graphs, load_graphs

def ChrombusDatset(raw_dir, type = 'train'):
    if os.path.exists(f'{raw_dir}/dataset_{type}.bin'):
        graphs,_ = load_graphs(f'{raw_dir}/dataset_{type}.bin')
        return graphs
    graphs = []
    n_seg = 128
    N_chr = 50
    min_interact = -2.5
    if type == 'train':
        rand_chr = [1,3,5,7,8,12,16,21]
    if type == 'test':
        rand_chr = [9,10,14,17]
    for i in rand_chr:
        print(i)
        #pos_dict[i] = []     ## for extracting sequence
        n_data_all = f'{raw_dir}/V_chr{i}.csv'
        e_data_all = f'{raw_dir}/chr{i}_edges.csv'
        # n_data_all = '/shareN/data9/UserData/yuanyuan/gm12878/cross_model/data/raw/V_chr1.csv'
        # e_data_all = '/shareN/data9/UserData/yuanyuan/gm12878/cross_model/data/raw/chr1_edges.csv'
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
            offset1 = nodes[0].min()
            nodes[0] = nodes[0] - offset1
            x = torch.tensor(nodes[[*range(2,16)]].values, dtype=torch.float)
            #full connected-- edges
            edges = edges_all[(edges_all[0] >= pos) & (edges_all[0] < pos1) & (edges_all[1] >= pos) & (edges_all[1] < pos1)]
            edges = edges.reset_index(drop=True)
            edges[0] = edges[0] - offset1
            edges[1] = edges[1] - offset1
            edge_index = np.vstack((np.arange(n_seg).repeat(n_seg),np.tile(np.arange(n_seg),n_seg)))
            edge_index = edge_index[:,edge_index[0] < edge_index[1]]
            edge_df = pandas.DataFrame(edge_index.T)
            edge_df = pandas.merge(edge_df, edges, how='left')
            # edge_df = edge_df.fillna({2:-2.5})
            edge_df = edge_df[edge_df[2] > min_interact]
            ## graph
            edges_src = torch.cat([torch.from_numpy(edge_df[0].values), torch.from_numpy(edge_df[1].values)])
            edges_dst = torch.cat([torch.from_numpy(edge_df[1].values),torch.from_numpy(edge_df[0].values)])
            edge_attr = torch.cat([torch.from_numpy(edge_df[2].values),torch.from_numpy(edge_df[2].values)])
            g = dgl.graph((edges_src, edges_dst), idtype=torch.int32)
            # g = dgl.to_bidirected(g)
            if g.num_nodes() != 128:
                continue
            g.ndata['x'] = x
            g.edata['label'] = edge_attr
            graphs.append(g)
    save_graphs(f'{raw_dir}/dataset_{type}.bin', graphs)
    return(graphs)

# train_data = ChrombusDatset()
# test_data = ChrombusDatset(type = "test")

def ChrombusDatset_singlechrom(raw_dir,chrom):
    if os.path.exists(f'{raw_dir}/dataset_chr{chrom}.bin'):
        graphs,_ = load_graphs(f'{raw_dir}/dataset_chr{chrom}.bin')
        return graphs
    graphs = []
    n_seg = 128
    N_chr = 50
    min_interact = -2.5
    for i in chrom:
        print(i)
        #pos_dict[i] = []     ## for extracting sequence
        n_data_all = f'{raw_dir}/V_chr{i}.csv'
        e_data_all = f'{raw_dir}/chr{i}_edges.csv'
        # n_data_all = '/shareN/data9/UserData/yuanyuan/gm12878/cross_model/data/raw/V_chr1.csv'
        # e_data_all = '/shareN/data9/UserData/yuanyuan/gm12878/cross_model/data/raw/chr1_edges.csv'
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
            offset1 = nodes[0].min()
            nodes[0] = nodes[0] - offset1
            x = torch.tensor(nodes[[*range(2,16)]].values, dtype=torch.float)
            #full connected-- edges
            edges = edges_all[(edges_all[0] >= pos) & (edges_all[0] < pos1) & (edges_all[1] >= pos) & (edges_all[1] < pos1)]
            edges = edges.reset_index(drop=True)
            edges[0] = edges[0] - offset1
            edges[1] = edges[1] - offset1
            edge_index = np.vstack((np.arange(n_seg).repeat(n_seg),np.tile(np.arange(n_seg),n_seg)))
            edge_index = edge_index[:,edge_index[0] < edge_index[1]]
            edge_df = pandas.DataFrame(edge_index.T)
            edge_df = pandas.merge(edge_df, edges, how='left')
            # edge_df = edge_df.fillna({2:-2.5})
            edge_df = edge_df[edge_df[2] > min_interact]
            ## graph
            edges_src = torch.cat([torch.from_numpy(edge_df[0].values), torch.from_numpy(edge_df[1].values)])
            edges_dst = torch.cat([torch.from_numpy(edge_df[1].values),torch.from_numpy(edge_df[0].values)])
            edge_attr = torch.cat([torch.from_numpy(edge_df[2].values),torch.from_numpy(edge_df[2].values)])
            g = dgl.graph((edges_src, edges_dst), idtype=torch.int32)
            # g = dgl.to_bidirected(g)
            if g.num_nodes() != 128:
                continue
            g.ndata['x'] = x
            g.edata['label'] = edge_attr
            graphs.append(g)
    save_graphs(f'{raw_dir}/dataset_chr{chrom}.bin', graphs)
    return(graphs)

