from chrombus_pyG.model_utils_pyG import get_pred
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mp', type=str, help='fine-tuned model path')
parser.add_argument('-chr', type=int, help='chromosome')
parser.add_argument('-dp', type=str, help='input data path,including chr*_edges.csv,V_chr*.csv')
parser.add_argument('-op', type=str, help='output path')
parser.add_argument('-ms', type=int, help='max_span',default=64)
args = parser.parse_args()

if __name__ == '__main__':
    get_pred(model_path=args.mp,chrom=args.chr,datapath=args.dp,outpath=args.op)

