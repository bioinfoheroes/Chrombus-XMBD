# Chrombus-XMBD
  A graph generative model capable of predicting chromatin interactions ab inito based on available chromatin features, including DNA accessibility, CTCF, RAD21, POLR2A, H3K4me3, and H3K27ac.

  This repository contains the source code for the paper Chrombus-XMBD is a novel graph neural network framework for predictiing 3D-Genome. (paper link)

## 1. Construction of graph and model architecture
<img src="https://github.com/bioinfoheroes/Chrombus-XMBD/assets/37092527/fb2940e7-157b-4e8c-b521-9b5393865162" width="700">

### (1) Construction of graph
  Each graph consists 128 vertices, and each vertex represents a chromatin segment derived from CTCF binding peaks. The node (vertex) attributes consist 14-dimensional chromatin features. The goal of the learning process is to generate the interactions among vertices, of which the labelling is based on Hi-C data (left panel).
### (2) Model architecture
   Chrombus is adopted from GAE architecture. The encoder consists of three edge convolution layers with embedded QKV attention mechanism and outputs embedding of dimensions 32, 16, and 32. The decoder is implemented as a plain inner product (right panel).
## 2. Predicting known chromatin interactions and TAD with Chrombus
  Chrombus's predictions of genes related to chromatin 3D structures in GM12878 cell line, including the TADs located at IRF4, BACH, HOXD family, HLA family, HOXA family and the beta-globin locus.
  
<img src="https://github.com/bioinfoheroes/Chrombus-XMBD/assets/37092527/c2b3b74c-0855-49a4-a6cb-b710a9a348b9" width="900">

## 4. Dependency
python=3.9.7

torch=1.9.1+cu111

pyg=2.0.2

## 5. Tutorial



