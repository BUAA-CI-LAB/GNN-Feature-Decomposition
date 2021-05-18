import time
import os.path as osp
import argparse
import torch
import sys

from torch.nn import Sequential
from torch.nn import Sequential, Linear, ReLU

sys.path.append("..")

import torch_geometric.transforms as T
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Planetoid
from model.inf_model import GCNNet, GATNet, SAGE_agg1, SAGE_com1, GIN_agg, GIN_com
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--gpu", type=int, default="0", help='0:use cpu  1:use gpu')
parser.add_argument("--hidden", type=int, default=8, help='hidden number')
parser.add_argument("--agg", type=str, default="gas", help='gas=scatter  spmm=sparse mul')
parser.add_argument("--m", type=str, default="GCN", help='model name: GCN,GIN,GAT,SAGE')
parser.add_argument("--layer", type=int, default=0, help='layer represent the number of layers cutting along '
                                                         'feature dimension, None = optpair x , 1 = single x')
parser.add_argument("--order", type=str, default='com', help='agg first or com first')
parser.add_argument("--data", type=str, default='rd', help='cr = cora,pb=pubmed,cs = cisteer, pt=ogbn-proteins, rd=reddit')
args = parser.parse_args()
hidden = args.hidden
layer = args.layer

a = layer.split(',')
a0 = (int)(a[0])
a1 = (int)(a[1])
layer = [a0, a1]
data_use = args.data

path = ''
dataset = ''
if data_use == 'rd':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../dataset', 'Reddit')
    if args.agg == "gas":
        dataset = Reddit(path)
        print('Reddit dataset, gas model!')
    else:
        dataset = Reddit(path, transform=T.ToSparseTensor())
        print('Reddit dataset, spmm model')
elif data_use == 'cr':
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../dataset')
    if args.agg == "gas":
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
        print('Cora dataset, gas model!')
    else:
        dataset = Planetoid(path, dataset, transform=T.ToSparseTensor())
        print('Cora dataset, spmm model')
elif data_use == 'pb':
    dataset = 'pubmed'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../dataset')
    if args.agg == "gas":
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
        print('Pumbed dataset, gas model!')
    else:
        dataset = Planetoid(path, dataset, transform=T.ToSparseTensor())
        print('Pumbed dataset, spmm model')
elif data_use == 'cs':
    dataset = 'CiteSeer'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../dataset')
    if args.agg == "gas":
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
        print('Citeseer dataset, gas model !')
    else:
        dataset = Planetoid(path, dataset, transform=T.ToSparseTensor())
        print('Citeseer dataset, spmm model')
elif data_use == 'pt':
    if args.agg == "gas":
        dataset = PygNodePropPredDataset(name='ogbn-proteins')
        print('Ogbn-proteins dataset, gas model!')
    else:
        dataset = PygNodePropPredDataset(name='ogbn-proteins', transform=T.ToSparseTensor())
        print('Ogbn-proteins dataset, spmm model!')

a = time.time()
data = dataset[0]

b = time.time()
print('Dataset process time : ', b - a)

device = torch.device('cuda' if args.gpu else 'cpu')

model = ''
if args.m == "GCN":
    print('using gcn')
    if data_use == 'pt':
        num_features = 1
    else:
        num_features = dataset.num_features
    model = GCNNet(num_features, hidden, dataset.num_classes)
elif args.m == "GAT":
    print('using gat')
    if data_use == 'pt':
        num_features = 1
    else:
        num_features = dataset.num_features
    model = GATNet(num_features, hidden, dataset.num_classes)
elif args.m == "SAGE":
    if args.order == 'agg':
        print('using 1st agg sage')
        if data_use == 'pt':
            num_features = 1
        else:
            num_features = dataset.num_features
        model = SAGE_agg1(num_features, hidden, dataset.num_classes)
    else:
        print('using 1st com sage')
        if data_use == 'pt':
            num_features = 1
        else:
            num_features = dataset.num_features
        model = SAGE_com1(num_features, hidden, dataset.num_classes)
elif args.m == "GIN":
    if args.order == 'agg':
        print('using 1st agg GIN,hid=', args.hidden)
        if data_use == 'pt':
            num_features = 1
        else:
            num_features = dataset.num_features
        model = GIN_agg(num_features, hidden, dataset.num_classes)
    else:
        print('using 1st com GIN,hid=', args.hidden)
        if data_use == 'pt':
            num_features = 1
        else:
            num_features = dataset.num_features
        model = GIN_com(num_features, hidden, dataset.num_classes)
else:
    print('error,model not exit')

model = model.to(device)


x = data.x.to(device)
y = data.y.squeeze().to(device)

if args.agg == "gas":
    edge_index = data.edge_index.to(device)
else:
    edge_index = data.adj_t.to(device)


print('using ', device)
print('layers : ', args.layer)

@torch.no_grad()
def test():
    model.eval()

    # with torch.autograd.profiler.profile(use_cuda=True if args.gpu else False, profile_memory=True) as prof:
    t0 = time.time()
    out = model(x, edge_index, layer=layer)
    t1 = time.time()
    # print(prof.key_averages().table(sort_by="self_cpu_time_total" if args.gpu == 0 else "self_cuda_time_total"))
    print('Inference time:', t1 - t0)
    print('Total time:', t1 - t0 + b - a)


if __name__ == '__main__':
    test()
