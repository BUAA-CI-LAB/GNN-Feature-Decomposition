import os.path as osp
import argparse
from torch.nn import Sequential, Linear, ReLU
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import time
from model.gnn_model_1_6 import GATConv_3d, GINConv_3d, GCNConv_3d, SAGEConv_3d, GINConv_3d_com, SAGEConv_3d_com


class GCNNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv_3d(in_channels, out_channels, cached=True,
                                normalize=True)
        self.conv2 = GCNConv_3d(out_channels, num_classes, cached=True,
                                normalize=True)

    def forward(self, x, edge_index, edge_attr=None, layer=(0, 0)):
        x = F.relu(self.conv1(x, edge_index, edge_attr, layer=layer[0]))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr, layer=layer[1])
        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int):
        super(GATNet, self).__init__()
        self.conv1 = GATConv_3d(in_channels, out_channels, heads=1, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv_3d(out_channels, num_classes, heads=1, concat=False,
                                dropout=0.6)

    def forward(self, x, edge_index, layer=(0, 0)):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, layer=layer[0]))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, layer=layer[1])

        return F.log_softmax(x, dim=1)


class SAGE_agg1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes: int):
        super(SAGE_agg1, self).__init__()
        self.conv1 = SAGEConv_3d(in_channels, hidden_channels)
        self.conv1 = SAGEConv_3d(hidden_channels, num_classes)

    def forward(self, x, edge_index, layer=(0, 0)):
        x = self.conv1(x, edge_index, layer=layer[0])
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, layer=layer[1])

        return F.log_softmax(x, dim=1)


class SAGE_com1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes: int):
        super(SAGE_com1, self).__init__()
        self.conv1 = SAGEConv_3d_com(in_channels, hidden_channels)
        self.conv2 = SAGEConv_3d_com(hidden_channels, num_classes)

    def forward(self, x, edge_index, layer=(0, 0)):
        x = self.conv1(x, edge_index, layer=layer[0])
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, layer=layer[1])

        return F.log_softmax(x, dim=1)


class GIN_agg(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes: int):
        super(GIN_agg, self).__init__()

        num_features = in_channels
        dim = hidden_channels

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv_3d(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv_3d(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv_3d(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv_3d(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv_3d(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, edge_index, layer=(0, 0)):
        x = F.relu(self.conv1(x, edge_index, layer=layer[0]))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index, layer=layer[1]))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index, layer=layer[1]))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index, layer=layer[1]))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index, layer=layer[1]))
        x = self.bn5(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class GIN_com(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes: int):
        super(GIN_agg, self).__init__()

        num_features = in_channels
        dim = hidden_channels

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv_3d_com(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv_3d_com(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv_3d_com(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv_3d_com(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv_3d_com(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, edge_index, layer=(0, 0)):
        x = F.relu(self.conv1(x, edge_index, layer=layer[0]))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index, layer=layer[0]))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index, layer=layer[0]))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index, layer=layer[0]))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index, layer=layer[1]))
        x = self.bn5(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
