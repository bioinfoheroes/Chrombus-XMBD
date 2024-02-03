
import torch
from chrombus_pyG.model_utils_pyG import get_models,train_model
from chrombus_pyG.model_utils_pyG import load_chrombus_data,load_chrombus_data_singlechrom
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dp', type=str, help='raw input data path,including chr*_edges.csv,V_chr*.csv')
parser.add_argument('-dataset', type=str, help='dataset path,chr*_epi_data.pt')
parser.add_argument('-testchr', type=int, help='chromosome for test')
parser.add_argument('-op', type=str, help='path to save model')
parser.add_argument('-type', type=int, help='1 or 2', default=2)
parser.add_argument('-trainchr', type=int, help='chromosome for train',required=False)
args = parser.parse_args()


#### 1. load data
datapath = args.dp
outpath = args.dataset
test_chr = args.testchr
modelpath = args.op
type = args.type
if type != 1:
    train_chr = args.trainchr
####
    
if __name__ == '__main__':
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    if type == 1:
        #### 1. train "leve-one-out" model
        train_loader, test_loader = load_chrombus_data(datapath,test_chr,outpath)
        model = get_models()
        train_model(model=model, trainloader=train_loader, testloader=test_loader,filepath=modelpath)
    else:
        #### 2. train "across-chromosome" model
        train_loader = load_chrombus_data_singlechrom(datapath,train_chr,outpath)
        test_loader = load_chrombus_data_singlechrom(datapath,test_chr,outpath)
        model = get_models()
        train_model(model=model, trainloader=train_loader, testloader=test_loader,filepath=modelpath)
