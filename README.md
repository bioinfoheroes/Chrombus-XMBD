# Chrombus-XMBD
  A graph generative model capable of predicting chromatin interactions ab inito based on available chromatin features, including DNA accessibility, CTCF, RAD21, POLR2A, H3K4me3, and H3K27ac.

  This repository contains the source code for the paper [Chrombus-XMBD, a novel graph neural network framework for predictiing 3D-Genome](https://www.biorxiv.org/content/10.1101/2023.08.02.551072v1).

## 1. Construction of graph and model architecture

<img src="https://github.com/bioinfoheroes/Chrombus-XMBD/assets/37092527/8f9135b5-6603-4bd8-8749-55a772f183f6" width="700">

### (1) Construction of graph
  Each graph consists 128 vertices, and each vertex represents a chromatin segment derived from CTCF binding peaks. The node (vertex) attributes consist 14-dimensional chromatin features. The goal of the learning process is to generate the interactions among vertices, of which the labelling is based on Hi-C data (left panel).
### (2) Model architecture
   Chrombus is adopted from GAE architecture. The encoder consists of three edge convolution layers with embedded QKV attention mechanism and outputs embedding of dimensions 32, 16, and 32. The decoder is implemented as a plain inner product (right panel).
## 2. Predicting known chromatin interactions and TAD with Chrombus
  Chrombus's predictions of genes related to chromatin 3D structures in GM12878 cell line, including the TADs located at IRF4, BACH, HOXD family, HLA family, HOXA family and the beta-globin locus.
  
<img src="https://github.com/bioinfoheroes/Chrombus-XMBD/assets/37092527/c2b3b74c-0855-49a4-a6cb-b710a9a348b9" width="900">

## 4. Dependency
python=3.11.5

torch==2.1.0+cu118

torch-geometric==2.4.0

## 5. Tutorial
### 5.0 Preprocessing input features
CTCF, RAD21, H3K27ac, H3K4me3, POLR2A and DNase I signals are processed into 14-dimension features as model input. For each segment, the mean binding strength of CTCF,H3K27ac, H3K4me3, POLR2A and DNase I signals is calculated. The RAD21-binding at a given CTCF-site (“left cohesin” and “right cohesin”) was also processed as a binary status, where “1” indicated that RAD21-binding peak was in vicinity of the CTCF-site (within a 500bp). In additon, we inferred the directionality of CTCF motifs within the CTCF-binding sites using fimo. Each CTCF-site was labeled as “0”, “1” according to the positive and negative strand, and the confidence score. See the pipeline in [data_preprocessing_gm12878_hg19.sh](https://github.com/bioinfoheroes/Chrombus-XMBD/blob/main/data_preprocessing_gm12878_hg19.sh). Here [bedtools](https://bedtools.readthedocs.io/en/latest/) should be installed.
### 5.1 Train across-chromosome model
Each sample for GM12878 model is randomly put-back cropped into 128-segments from the training and testing chromosomes. For model training, 200 samples were generated from each chromosome, and 50 samples for testing purposes. See the [Chrombus_pyG_train.py](https://github.com/bioinfoheroes/Chrombus-XMBD/blob/main/Chrombus_pyG_train.py). The trained model can be load from the directory "trained_model".
### 5.2 Predicting chromatin interaction at specific regions with well-trained Chrombus
We provide test dataset of GM12878 cell line on chromsome 18. See the [Chrombus_pyG_predict.py](https://github.com/bioinfoheroes/Chrombus-XMBD/blob/main/Chrombus_pyG_predict.py) and [Chrombus_tutorial.ipynb](https://github.com/bioinfoheroes/Chrombus-XMBD/blob/main/Chrombus_tutorial.ipynb).

## Contact:
Yuanyuan Zeng: yuanyuanzeng0001@stu.xmu.edu.cn


