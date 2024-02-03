import os
import pandas as pd
import numpy as np
import sys
import argparse
from tqdm import tqdm
from utils_hg19 import *

parser = argparse.ArgumentParser()
parser.add_argument('-c',dest='ctcf_bedfile',help= "ctcf_bedfile",required=True,default='CTCF.bed')
parser.add_argument('-r',dest='rad21_bedfile',help= "rad21_bedfile",required=True)
parser.add_argument('-t',dest='threshold',help= "threshold to distinguish cohesin",default=500)
parser.add_argument('-od',dest='outdir',help= "output dir", required=True)


args = parser.parse_args()
ctcf = args.ctcf_bedfile
rad21 =args.rad21_bedfile
threshold = args.threshold
outdir=args.outdir


def check_cohesin(ctcf_mid,rad21_mid,threshold):
    cohesin={}
    for i in range(1,23):
        ctcf_size = ctcf_mid[i].shape[0]
        rad21_size = rad21_mid[i].shape[0]
        ctcf_mtx = ctcf_mid[i].repeat(rad21_size).reshape([ctcf_size,rad21_size])
        rad21_mtx = np.tile(rad21_mid[i],ctcf_size).reshape([ctcf_size,rad21_size])
        sub_mtx = np.abs(ctcf_mtx - rad21_mtx) < threshold
        sub_mtx = np.sum(sub_mtx,axis=-1)
        sub_mtx[sub_mtx>=1]=1
        cohesin[i] = sub_mtx
    return cohesin


if __name__ == "__main__":
    root_path,_ = os.path.split(ctcf)
    ctcf = pd.read_csv(rf'{ctcf}',header=None,sep='\t')
    rad21 = pd.read_csv(rf'{rad21}',header=None,sep='\t')
    ctcf = filter_bed(ctcf)
    rad21 = filter_bed(rad21)
    ctcf_mid = get_mid(ctcf)
    rad21_mid = get_mid(rad21)
    cohesin = check_cohesin(ctcf_mid,rad21_mid,threshold)
    save_dict(cohesin, outdir + '/' + 'cohesin.pkl')
    # cohesin_df = get_df(cohesin)














