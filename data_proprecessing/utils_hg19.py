import os
import pandas as pd
import numpy as np
import sys
import argparse
from tqdm import tqdm
import pickle

chr_length = {
    'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260, 'chr6': 171115067,
    'chr7': 159138663, 'chr8': 146364022,
    'chr9': 141213431, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878,
    'chr14': 107349540, 'chr15': 102531392,
    'chr16': 90354753, 'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr20': 63025520, 'chr21': 48129895,
    'chr22': 51304566, 'chrX': 155270560
}

def save_dict(file,file_path):
    if os.path.exists(file_path):
        print('file exists, overwriting')
    else:
        print('writing')
    with open(file_path,'wb') as f:
        pickle.dump(file,f,pickle.HIGHEST_PROTOCOL)

def load_dict(file_path):
    try:
        with open(file_path,'rb') as f:
            return pickle.load(f)
    except:
        print('file not exists')

def filter_bed(bed):
    try:
        bed.iloc[0,0].startswith('chr') #判斷是否過濾
        chrs, l, r = [], [], []
        a = ['X', 'Y']
        for x, i, j in zip(bed[0].values, bed[1].values, bed[2].values):
            c = x[3:] #获取没有"chr"的染色体编号
            if c not in a:
                chrs.append(c)
                l.append(i)
                r.append(j)
        bed = pd.DataFrame([chrs, l, r]).T
        bed = bed.sort_values(by=[0, 1, 2])
        bed = bed.astype(int)
    except:
        pass
    return bed

def filter_bed1(bed):
    chrs, l, r = [], [], []
    a = [str(i) for i in range(1,23)]
    for x, i, j in zip(bed[0].values, bed[1].values, bed[2].values):
        if x in a:
            chrs.append(x)
            l.append(i)
            r.append(j)
    bed = pd.DataFrame([chrs, l, r]).T
    # bed = bed.sort_values(by=[0, 1, 2])
    bed = bed.astype(int)
    return bed


def get_mid(bed,add_head_tail=False):
    mid = {}
    for i in range(1,23):
        tmp = bed[bed.values[:,0]==i]
        mid[i] = ((tmp.values[:,1] + tmp.values[:,2])/2).astype(int)
        if add_head_tail:
            mid[i] = np.concatenate([np.zeros([1]),mid[i],np.ones([1])*chr_length[f'chr{i}']]).astype(int)
    return mid




