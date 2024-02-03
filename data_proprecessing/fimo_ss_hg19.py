import os
import pandas as pd
import numpy as np
import sys
import argparse
from tqdm import tqdm
from utils_hg19 import *

parser = argparse.ArgumentParser()
parser.add_argument('-i',dest='bedfile',help= "conservative IDR thresholded peak file format bed",required=True)
parser.add_argument('-c',dest='ctcfmotif',help= "meme file from jasper database",required=True)
parser.add_argument('-f',dest='fimo',help='meme prediction algorithm',required=True)
parser.add_argument('-o',dest='outname',help= "output filename",default='onectcf')
parser.add_argument('-g',dest='genome',help= "hg19 genome",required=True)
parser.add_argument('-od',dest='outdir',help= "output dir",required=True)

#exp = '-c sourcedata/MAO139.1.meme -i sourcedata/ENCFF631SBB.bed -f scripts/fimo -g sourcedata/genome.fa'

args = parser.parse_args()
input_bed = args.bedfile
motif = args.ctcfmotif
fimo = args.fimo
genome =args.genome
outdir = args.outdir
output_file = outdir + '/' + args.outname

def merge_peaks(bed):
    lines = [[0,0,0]]
    for i in range(1,23):
        bed_chr = bed[bed.values[:,0]==i]
        for j in bed_chr.values:
            if j[0] == lines[-1][0]:
                if j[1] <= lines[-1][2]:
                    last = lines.pop()
                    lines.append([j[0],last[1],j[2]])
                else:
                    lines.append(j.tolist())
            else:lines.append(j.tolist())
    lines.pop(0)
    lines = pd.DataFrame(lines)
    return lines

def bed2txt(bed,outdir):
    bed.to_csv(outdir + '/' + 'CTCF.bed',header=None,sep='\t',index=None)
    inputbed = outdir + '/' + 'CTCF.bed'
    os.system(fr"bedtools getfasta -fi {genome} -bed {inputbed} -fo {output_file}.txt")
    txt = pd.read_csv(rf"{output_file}.txt",header=None)
    return txt

def saveline(txt,index,outdir):
    line = txt.iloc[index*2:index*2+2]
    line.to_csv(outdir + '/' + r'oneline.csv',header=None,index=None)

def run_fimo(txt,outdir):
    bar = tqdm(range(txt.shape[0]//2))
    strand_list, score_list= [],[]
    fimo_oc = outdir + "/oneline"
    csv = outdir + '/' + r'oneline.csv'
    for i in bar:
        saveline(txt,i,outdir)
        os.system(fr'{fimo} --oc {fimo_oc} {motif} {csv}')
        prediction_file = pd.read_csv(fimo_oc +'/fimo.tsv',sep='\t')[:-3]
        if prediction_file.shape[0] <1 :
            strand = 0
            score = 0
        else:
            tmp_idx = prediction_file['score'].argmax()
            prediction = prediction_file.iloc[tmp_idx]
            strand = prediction['strand']
            if strand == '+':
                strand = 1
            else:
                strand = -1
            score = prediction['score']
        strand_list.append(strand)
        score_list.append(score)
    return strand_list,score_list

if __name__ == '__main__':
    bed = pd.read_csv(input_bed, header=None, sep='\t')
    if not os.path.exists(output_file+'.txt'):
        print('Generating sequence')
        bed = filter_bed(bed)
        bed = merge_peaks(bed)
        txt = bed2txt(bed,outdir)
    else:
        txt = pd.read_csv(rf"{output_file}.txt",header=None)
    strand_list, score_list = run_fimo(txt,outdir)
    ss = pd.DataFrame([strand_list,score_list]).T
    ss.to_csv(outdir + '/' + 'strand_score.csv', header=None,sep='\t',index=None)
#    os.system(fr'rm -fr {output_file}*')
