##### 0. set path
outdir=yourpath/gm12878_hg19/
datdir=/yourpath/GM12878/hg19/
reference_genome=/yourpath/hg19.fa
hicdir=/yourpath/
##############
mkdir -p $outdir
ctcf_chip=$datdir/ENCFF473RXY.bed  ##先过滤非常染色体
atac_seq=$datdir/ENCFF901GZH.bigWig  
h3k4me3_chip=$datdir/ENCFF818GNV.bigWig
H3K27ac_chip=$datdir/ENCFF180LKW.bigWig
pol2_chip=$datdir/ENCFF368HBX.bigWig
cohesin_chip=$datdir/ENCFF001VFE.bed
ctcf_bw=$datdir/ENCFF886KRA.bigWig
# # #1. fimo motif score, CTCF bed
python ./preprocessing/fimo_ss_hg19.py -i $ctcf_chip -c ./preprocessing/MA0139.1.meme -f ./preprocessing/fimo -g ${reference_genome} -od $outdir
# # #2. compute mean value of histone-markers, cohesin, pol2 and atact-seq in segment
ctcf_bed=$outdir/CTCF.bed

python ./preprocessing/chip_mean_value_hg19.py -c $ctcf_bed -i $ctcf_bw -b ./preprocessing/bigWigToBedGraph -o $outdir/CTCF

python ./preprocessing/chip_mean_value_hg19.py -c $ctcf_bed -i $atac_seq -b ./preprocessing/bigWigToBedGraph -o $outdir/ATAC

python ./preprocessing/chip_mean_value_hg19.py -c $ctcf_bed -i $h3k4me3_chip -b ./preprocessing/bigWigToBedGraph -o $outdir/H3K4me3

python ./preprocessing/chip_mean_value_hg19.py -c $ctcf_bed -i $H3K27ac_chip -b ./preprocessing/bigWigToBedGraph -o $outdir/H3K27ac

python ./preprocessing/chip_mean_value_hg19.py -c $ctcf_bed -i $pol2_chip -b ./preprocessing/bigWigToBedGraph -o $outdir/Pol2

python ./preprocessing/check_rad21_hg19.py -c $ctcf_bed -r $cohesin_chip -od $outdir #rad21

# # # # # # #4. 拼接input feature
python ./preprocessing/gen_input_feature.1.py -d $outdir

# # # # 5. 画input feature的分布，归一化到[0,1]
cd $outdir
#module load  R/v4.0.3
mkdir -p $outdir/inputdata/raw/
mkdir -p $outdir/stat/
### edge feature
Rscript ./preprocessing/edata_hg19_sqvrt.R $hicdir $outdir/inputdata/raw/ 4
### node feature
Rscript ./preprocessing/normalized_nodes_hg19.R $outdir $outdir/nodes_feats.csv ./preprocessing/GSE63525_GM12878_primary_and_replicate_Arrowhead_domainlist.txt