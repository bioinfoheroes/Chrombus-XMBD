'''
由chipseq的bigwig文件生成ctcf每个segment的counts
'''
import os
import pandas as pd
import numpy as np
import sys
import argparse
from tqdm import tqdm
from utils_hg19 import *


parser = argparse.ArgumentParser()
parser.add_argument('-c',dest='ctcf',help= "ctcf bedfile",required=True)
parser.add_argument('-i',dest='bigwig',help= "chip-seq bigwig file",required=True)
parser.add_argument('-b',dest='bigWigToBedGraph',help= "bigWigToBedGraph tools",required=True)
parser.add_argument('-o',dest='outname',help='accept={ATAC,H3K4me3,H3K27ac,Pol2}',required=True)

#exp = r'-c sourcedata/ENCFF631SBB.bed -i xxx -b scripts/bigWigToBedGraph -o xxx'

args = parser.parse_args()
bigwig = args.bigwig
ctcf = args.ctcf
bigWigToBedGraph =args.bigWigToBedGraph
outname = args.outname

def bdgfile_transfer(bdgfile):
    #input bdg file ,transform chrn to n,keep site, keep counts
    L,N = [],[]
    for i in bdgfile[0].values:
        try:
            tmp = int(i[3:])
            L.append(True)
            N.append(tmp)
        except:
            L.append(False)
    bdgfile = bdgfile[L]
    bdgfile[0] = N
    bdgfile = bdgfile.reset_index(drop=True)
    return bdgfile



def bdg2depth(bdg,ctcf_mid,mode='mean'):
    assert mode in ['mean','max']
    if mode =='mean':
        mode = np.mean
    elif mode== 'max':
        mode = np.max
    depth = {}
    for i in tqdm(range(1,23)):
        bdg_chr = bdg[bdg.values[:,0]==i]
        ctcf_chr = ctcf_mid[i]
        depth_bp = np.zeros([chr_length[f'chr{i}']])
        for start,end,counts in zip(bdg_chr.values[:,1].astype(int),bdg_chr.values[:,2].astype(int),bdg_chr.values[:,3]):
            depth_bp[start:end] = counts
        depth[i] = np.zeros([ctcf_chr.shape[0]-1])
        for idx,(start,end) in enumerate(zip(ctcf_chr[:-1],ctcf_chr[1:])):
            depth[i][idx] = mode(depth_bp[start:end])
    return depth


if __name__ == '__main__':
    bigwig_p = os.path.split(outname)[0]
    bigwig_p = bigwig_p + '/' + os.path.splitext(os.path.split(bigwig)[1])[0]
    if not os.path.exists(f'{bigwig_p}' + '.bdg'):
        print('generating bdgfile')
        os.system(rf'{bigWigToBedGraph} {bigwig} {bigwig_p}.bdg')
    print('loading bdgfile')
    bdgfile = pd.read_csv(rf'{bigwig_p}' + '.bdg', header=None, sep='\t')
    bdgfile = bdgfile_transfer(bdgfile)
    print('loading ctcf bedfile')
    ctcf = pd.read_csv(ctcf,header=None,sep='\t')
    ctcf = filter_bed(ctcf)
    ctcf_mid = get_mid(ctcf,True)
    print('generating depthfile')
    depth = bdg2depth(bdgfile,ctcf_mid,'mean')
    save_dict(depth,rf'{outname}.pkl')
    # np.save(rf'{outname}', depth)











