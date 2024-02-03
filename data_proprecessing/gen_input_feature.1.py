import dgl
from dgl.data.utils import load_graphs,save_graphs
import os
import numpy as np
import pandas as pd
import torch as th
import argparse
from scipy.stats import zscore
from utils_hg19 import *

parser = argparse.ArgumentParser()
parser.add_argument('-d',dest='workdir',help= "workdir",required=True)
#parser.add_argument('-i',dest='bedfile',help= "conservative IDR thresholded peak file format bed",required=True)
#parser.add_argument('-c',dest='ctcfbed',help= "filted_ctcfbed",required=True)
#parser.add_argument('-x',dest='mean_feats',help= "filted_ctcfbed",required=True)
#parser.add_argument('-f',dest='fimo',help='meme prediction algorithm',required=True)
#parser.add_argument('-o',dest='outname',help= "output filename",default='out/ctcf')
#parser.add_argument('-g',dest='genome',help= "hg38 genome",required=True)
# parser.add_argument('-chr',dest='chrom',help= "chromosome",required=True)

args = parser.parse_args()
workdir = args.workdir
# chrom = args.chrom
#ctcfbed = args.ctcfbed
#motif = args.ctcfmotif
#fimo = args.fimo
#output_file = args.outname
#genome =args.genome

def gen_node_feats(chrom,ctcf_ss_chr,chip_chr):
    #
    ctcf_mid = ((ctcf_ss_chr.values[:,1] + ctcf_ss_chr.values[:,2])/2).astype(int)
    ctcf_mid = np.concatenate([np.zeros([1]), ctcf_mid, np.ones([1]) * chr_length[f'chr{chrom}']]).astype(int)
    node_mid = ((ctcf_mid[:-1] +ctcf_mid[1:])/2).astype(int) #得到节点中心位置
    ctcf_chr = pd.concat([ctcf_ss_chr,ctcf_ss_chr])
    ctcf_chr = ctcf_chr.reset_index(drop=True)
    ctcf_chr.loc[ctcf_chr.shape[0]] = [chrom,chr_length[f'chr{chrom}'],chr_length[f'chr{chrom}'],0,0,0]
    ctcf_chr.loc[-1] = [chrom,0,0,0,0,0]
    ctcf_chr = ctcf_chr.sort_index()
    node_feats = np.concatenate([ctcf_chr.values[:ctcf_chr.shape[0]//2,3:],ctcf_chr.values[ctcf_chr.shape[0]//2:,3:],np.array([np.insert(chip_chr[:,0],0,0.0)[:-1], np.insert(chip_chr[:,0],chip_chr.shape[0],0.0)[1:]]).T, chip_chr[:,1:],node_mid[:,np.newaxis]],axis=1)
    node_feats = node_feats.astype('float32')
        # node_feats = np.concatenate([ctcf_chr.values[:ctcf_chr.shape[0]//2,3:],ctcf_chr.values[ctcf_chr.shape[0]//2:,3:],chip_chr,node_mid[:,np.newaxis]],axis=1)    
    return node_feats

def gen_graph(chr,ctcf_ss_chr,chip_chr,counts_chr):
    node_feats = gen_node_feats(chr, ctcf_ss_chr, chip_chr)
    node_mid = node_feats[:,-1]
    graph = dgl.DGLGraph()
    node_size = node_mid.shape[0]
    dist_mtx = np.zeros([node_size, node_size])
    for i in range(node_size):
        for j in range(i,node_size):
            dist_mtx[i,j] = node_mid[i] -node_mid[j]
    dist_mtx = np.abs(dist_mtx)
    dist_mtx += dist_mtx.T - np.diagonal(dist_mtx)
    dist_mtx = np.log10(1+dist_mtx)
    graph.add_nodes(node_size)
    graph.add_edges(np.arange(node_size).repeat(node_size),np.tile(np.arange(node_size),node_size))
    graph.ndata['c'] = th.tensor(node_feats)
    graph.edata['dist'] = th.tensor(dist_mtx.reshape([-1,1]))
    graph.edata['hic'] = th.tensor(counts_chr.reshape([-1,1]))
    return graph

def main(ctcf_ss, ctcf, chip):
    nodes_feat_all = pd.DataFrame()
    for chr in range(1,23):
        print(chr)
        ctcf_ss_chr = ctcf_ss[ctcf_ss.values[:,0]==chr]
        chip_chr = np.concatenate([i[chr] for i in chip]).reshape([len(chip), -1]).T
        # chip_chr = np.concatenate(chip).reshape([len(chip), -1]).T
        nodes_feat = gen_node_feats(chr, ctcf_ss_chr, chip_chr)
        nodes_feat = pd.DataFrame(nodes_feat,columns=['l_dir','l_score','l_cohesin','r_dir','r_score','r_cohesin','l_ctcf_peak','r_ctcf_peak','ATAC_mean','H3K4me3_mean','H3K27ac_mean','Pol2_mean','location'])
        ctcf_mid = get_mid(ctcf,True)
        start,end = ctcf_mid[chr][:-1],ctcf_mid[chr][1:]
        nodes_feat.insert(0,'end',end)
        nodes_feat.insert(0, 'start', start)
        nodes_feat.insert(0, 'node_idx', np.arange(nodes_feat.shape[0]))
        nodes_feat.insert(0, 'chr', chr)
        nodes_feat_all = pd.concat([nodes_feat_all,nodes_feat])
    return nodes_feat_all
    


def gen_csv(G,ctcf,outdir,chrom):
    ctcf_mid = get_mid(ctcf,True)
    nodes_feats = []
    edges_feats = []
    for i in tqdm([int(chrom)]):
        start,end = ctcf_mid[i][:-1],ctcf_mid[i][1:]
        nodes_feat = pd.DataFrame(G[0].ndata['c'].numpy(),columns=['l_dir','l_score','l_cohesin','r_dir','r_score','r_cohesin','l_ctcf_peak','r_ctcf_peak','ATAC_mean','H3K4me3_mean','H3K27ac_mean','Pol2_mean','location'])
        nodes_feat.insert(0,'end',end)
        nodes_feat.insert(0, 'start', start)
        nodes_feat.insert(0, 'node_idx', np.arange(nodes_feat.shape[0]))
        nodes_feat.insert(0, 'chr', i)
        nodes_feats.append(nodes_feat)
        src,dst = G[0].edges()
        edges_feat = np.concatenate([src.numpy()[:,np.newaxis],dst.numpy()[:,np.newaxis],G[0].edata['dist'].numpy(),G[0].edata['hic'].numpy()],axis=-1)
        edges_feat = pd.DataFrame(edges_feat)
        edges_feat.columns = ['src','dst','dist','hic']
        edges_feat.insert(0,'chr',i)
        edges_feats.append(edges_feat)
    nodes_feats = pd.concat(nodes_feats)
    edges_feats = pd.concat(edges_feats)
    nodes_feats.to_csv(outdir + 'nodes_feats.csv',index=None,sep='\t')
    edges_feats.to_csv(outdir + 'edges_feats.csv', index=None, sep='\t')

def get_df(cohesin):
    chrs,c = [],[]
    for i in range(1,23):
        tmp = cohesin[i]
        chrs.append(np.ones(tmp.shape[0])*i)
        c.append(tmp)
    chrs = np.concatenate(chrs)
    c = np.concatenate(c)
    df = pd.DataFrame([chrs,c]).T
    #df.to_csv(r'cohesin.csv',header=None,sep='\t',index=None)
    return df

if __name__ == "__main__":
    # inputdir = '/public/home/yuanyuan/3Dgenome/data_preprocessing/gm12878/v1/'
    # ctcfbed = '/public/home/yuanyuan/3Dgenome/data_preprocessing/gm12878/v1/CTCF.bed'
    # ctcfbed = r'/public/home/yuanyuan/3Dgenome/net2/test1031/inputdata/CTCF_max_value.bed'
    inputdir = workdir
    ctcfbed = inputdir + '/CTCF.bed'
    ctcf = pd.read_csv(ctcfbed,header=None,sep='\t') # filtered and merged ctcf
    ctcf = filter_bed(ctcf)
    # ctcf = merge_peaks(ctcf)
    # for chrom in CTCF.keys():
    #     ctcf.loc[ctcf[0] == chrom, 3] = CTCF[chrom]
    # ctcf[[3]] = np.log10(1 + ctcf[[3]].values)  # ctcf的peak取log10
    ss = pd.read_csv(inputdir + '/strand_score.csv',header=None,sep='\t')
    # ss[[1]] = zscore(ss[[1]].values) # 对ctcf方向的score进行zscore
    #cohesin =  pd.read_csv(r'/public/home/zhiyu/fgraph/out/cohesin.csv',header=None,sep='\t')
    #cohesin = np.load(inputdir + 'cohesin.npy')
    # cohesin = np.load(inputdir + 'cohesin.npy')
    cohesin = load_dict(inputdir + '/cohesin.pkl')
    cohesin = get_df(cohesin)
    # cohesin = get_df(cohesin, chrom)
    ctcf_ss = pd.concat([ctcf,ss],axis=1)
    ctcf_ss.insert(ctcf_ss.shape[1],'cohesin',cohesin.values[:,1])
    # ctcf_ss.insert(ctcf_ss.shape[1],'cohesin',cohesin)
    CTCF = load_dict(inputdir + '/CTCF.pkl')
    ATAC = load_dict(inputdir + '/ATAC.pkl')
    H3K4me3 = load_dict(inputdir + '/H3K4me3.pkl')
    H3K27ac = load_dict(inputdir + '/H3K27ac.pkl')
    Pol2 = load_dict(inputdir + '/Pol2.pkl')
    chip = [CTCF,ATAC,H3K4me3,H3K27ac,Pol2]
    # counts = load_dict(inputdir + '/count_mtx.pkl')
    nodes_feat = main(ctcf_ss, ctcf, chip)
    nodes_feat.to_csv(inputdir + '/nodes_feats.csv',index=None,sep='\t')
