
import os
# os.chdir("/shareN/data9/UserData/yuanyuan/GRN/scripts/")
import dgl
from model_utils import get_model,train_model,get_baseline
from load_chrombusdata import ChrombusDatset,ChrombusDatset_singlechrom
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dp', type=str, help='raw input data path,including chr*_edges.csv,V_chr*.csv')
parser.add_argument('-testchr', type=int, help='chromosome for test')
parser.add_argument('-op', type=str, help='path to save model')
parser.add_argument('-type', type=int, help='1 or 2',help = '1:multiple and random test and train chromsome;2:single chromsome for testing and training', default=2)
parser.add_argument('-trainchr', type=int, help='chromosome for train',required=False)
args = parser.parse_args()

raw_dir = args.dp
test_chrom = args.testchr
train_chrom = args.trainchr
type = args.type
filepath = args.op


if __name__ == '__main__':
    if type == 1:
        train_data = ChrombusDatset(raw_dir=raw_dir,type = "train")
        test_data = ChrombusDatset(raw_dir=raw_dir,type = "test")
    else:
        train_data = ChrombusDatset_singlechrom(raw_dir=raw_dir,chrom=train_chrom)
        test_data = ChrombusDatset_singlechrom(raw_dir=raw_dir,chrom=test_chrom)    
    trainloader = dgl.dataloading.GraphDataLoader(train_data, batch_size=16, shuffle=True, drop_last=False)
    testloader = dgl.dataloading.GraphDataLoader(test_data, batch_size=16, shuffle=False, drop_last=False)
    ####### 1. decoder is innerproduct
    conv_name = "EdgeConvGAT"
    model = get_model(conv_name,14,32,4)
    filepath = '/shareN/data9/UserData/yuanyuan/gm12878/cross_model/model_dgl/' + conv_name
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    #
    train_model(model,trainloader,testloader,filepath= filepath, epochs=400)
    ######## 2. decoder is 1-depth linear
    conv_name = "EdgeConvGAT"
    model = get_model(conv_name,14,32,4,decoder="mlp")
    filepath = '/shareN/data9/UserData/yuanyuan/gm12878/cross_model/model_dgl/' + conv_name + '_mlp'
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    #
    train_model(model,trainloader,testloader,filepath= filepath,epochs=400)
