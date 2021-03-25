import time
import os.path as osp
import argparse
import torch
import sys
# from memory_profiler import profile
from torch.nn import Sequential
from torch.nn import Sequential, Linear, ReLU

sys.path.append("..")
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, EdgeConv
from model.gnn_model_1_6 import GATConv_3d, GINConv_3d, GCNConv_3d, SAGEConv_3d, GINConv_3d_com, SAGEConv_3d_com
import torch_geometric.transforms as T
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--gpu", type=int, default="0", help='0:use cpu  1:use gpu')
parser.add_argument("--hidden", type=int, default=8, help='hidden number')
parser.add_argument("--agg", type=str, default="gas", help='gas = scatter  spmm = sparse mul')
parser.add_argument("--m", type=str, default="GCN", help='model name: GCN,GIN,GAT,SAGE')
parser.add_argument("--layer", type=int, default=0, help='layer represent the number of layers cutting along '
                                                         'feature dimension, None = optpair x , 1 = single x')
parser.add_argument("--order", type=str, default='agg', help='agg or com')
parser.add_argument("--data", type=str, default='rd', help='cr = cora,pb=pubmed,cs = cisteer, pt=ogbn-proteins, rd=reddit')

args = parser.parse_args()
layer = args.layer
hidden = args.hidden

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
        print('Citeseer dataset, gas model!')
    else:
        dataset = Planetoid(path, dataset, transform=T.ToSparseTensor())
        print('Citeseer dataset, spmm model')
elif data_use == 'pt':
    if args.agg == "gas":
        dataset = PygNodePropPredDataset(name='ogbn-proteins',root='../dataset')
        print('Ogbn-proteins dataset, gas model!')
    else:
        dataset = PygNodePropPredDataset(name='ogbn-proteins', root='../dataset',transform=T.ToSparseTensor())
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
    model = GCNConv_3d(num_features, hidden)
elif args.m == "GAT":
    print('using gat')
    if data_use == 'pt':
        num_features = 1
    else:
        num_features = dataset.num_features
    model = GATConv_3d(num_features, hidden)
elif args.m == "SAGE":
    if args.order == 'agg':
        print('using 1st agg graphsage')
        if data_use == 'pt':
            num_features = 1
        else:
            num_features = dataset.num_features
        model = SAGEConv_3d(num_features, hidden)
    else:
        print('using 1st com graphsage')
        if data_use == 'pt':
            num_features = 1
        else:
            num_features = dataset.num_features
        model = SAGEConv_3d_com(num_features, hidden)
elif args.m == "GIN":
    if args.order == 'agg':
        if data_use == 'pt':
            num_features = 1
        else:
            num_features = dataset.num_features
        nn1 = Sequential(Linear(num_features, args.hidden), ReLU(), Linear(args.hidden, args.hidden))
        print('using 1st agg GIN , hid=', args.hidden)
        model = GINConv_3d(nn1)
    else:
        if data_use == 'pt':
            num_features = 1
        else:
            num_features = dataset.num_features
        nn1 = Sequential(Linear(num_features, args.hidden), ReLU(), Linear(args.hidden, args.hidden))
        print('using 1st com GIN , hid=', args.hidden)
        model = GINConv_3d_com(nn1)
else:
    print('error,model not exit')

model = model.to(device)

if data_use == 'pt':
    x = torch.Tensor(132534, 1)
else:
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
    #print(prof.key_averages().table(sort_by="self_cpu_time_total" if args.gpu == 0 else "self_cuda_time_total"))
    print('Inference time:', t1 - t0)
    print('Total time:', t1 - t0 + b - a)


if __name__ == '__main__':
    test()
