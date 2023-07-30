from typing import Optional, Callable, List
import os
import os.path as osp
import shutil
import pandas
import torch
import numpy
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
import random
from torch_geometric.utils import to_undirected

#generate dataset from whole-genome HiC data
class HiCDataset(InMemoryDataset):
    url = ''
    # N_chr = 300
    min_interact = -2.5
    def __init__(self, root: str, name: str, n_seg: int, dataset:str, testchr: int, N_chr: int, transform=None, pre_transform=None):
        self.name = name
        self.n_seg = n_seg
        self.dataset = dataset
        self.testchr = testchr
        self.N_chr = N_chr
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')
    @property
    def raw_file_names(self) -> List[str]:        
        features = ['V']
        return [f'{feature}_chr{i}.csv' for i in range(1, 23) for feature in features]      
    @property
    def processed_file_names(self) -> str:
        return self.dataset + 'data.pt'
    def download(self):
        url = self.url
        folder = osp.join(self.root, self.name)
        path = download_url(f'{url}/{self.name}.zip', folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)
    def process(self):
        def TADparser(tad):
            t = tad.split('.', -1)
            l = len(t)
            if l <= 2:
                return [-1, -1]
            elif l == 3: 
                return [int(t[1]), -1]
            else:
                return [int(t[1]), int(t[2])]
        data_list = []
        rand_chr = []
        pos_list = []
        # train_chr = random.sample(range(1,23),16)
        # test_chr = list(set([*range(1,23)]) - set(train_chr))
        if self.dataset == "train":
            # rand_chr = [8, 22, 7, 11, 14, 1, 21, 2, 19, 20, 5]
            rand_chr = [i for i in [*range(1,23)] if i != self.testchr]
            # rand_chr = [8, 22, 7, 11]
        elif self.dataset == "test":
            # rand_chr = [3, 9, 12, 13, 15, 16,17]
            rand_chr = [self.testchr]
            # rand_chr = [3, 12]
        elif self.dataset == "singleChr":
            rand_chr = [9]
        else:
            print("error1")
        for i in rand_chr:
            print(i)
            #pos_dict[i] = []     ## for extracting sequence
            n_data_all = f'{self.raw_dir}/V_chr{i}.csv'
            e_data_all = f'{self.raw_dir}/chr{i}_edges.csv'
            nodes_all = pandas.read_csv(n_data_all, sep=',')
            nodes_all.columns = [*range(0, nodes_all.shape[1])]
            edges_all = pandas.read_csv(e_data_all, sep=',')
            edges_all.columns = [*range(0, edges_all.shape[1])]
            edges_all = edges_all[edges_all[0] != edges_all[1]]
            if edges_all.shape[0] == 0:
                continue
            n_nodes = range(nodes_all[0].max() - self.n_seg + 1)
            if self.N_chr < max(n_nodes):
                n_pos = random.sample(n_nodes, self.N_chr)
            else:
                n_pos = random.sample(n_nodes, int(max(n_nodes) * 0.6))
            # n_pos = pos_dict[i]
            for pos in n_pos:
                pos1 = pos + self.n_seg
                #get node features
                nodes = nodes_all[pos:pos1]
                nodes = nodes.reset_index(drop = True)
                if nodes.shape[0] < self.n_seg:
                    continue             
                y = [TADparser(nodes[16][i]) for i in range(nodes.shape[0])]
                y = torch.tensor(y, dtype=torch.long)
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
                v_from = v_from - offset1
                v_to = v_to - offset1
                indx = torch.tensor(edges[2].values > self.min_interact)
                edge_index = torch.cat([torch.stack([v_from[indx], v_to[indx]]), torch.stack([v_to[indx], v_from[indx]])], axis = 1)
                edge_attr = torch.reshape(torch.cat([e_attribute[indx],e_attribute[indx]]), (-1, 1))
                g1 = Data(x=x, y=torch.tensor([1]), edge_index=edge_index, edge_attr=edge_attr)
                data_list = data_list + [g1]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        numpy.save(self.processed_dir + '/' + self.dataset + '_pos_list.npy', pos_list)

