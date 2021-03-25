# GNN-Feature-Decomposition
#### Overview

This is a repository for our work: GNN Feature Decomposition,
which is accepted by RTAS 2021(Brif Industry Track), named ***"Optimizing Memory Efficiency of Graph NeuralNetworks on Edge Computing Platforms"***

Graph neural networks (GNN) have achieved state-of-the-art performance on various industrial tasks.
However, the poor efficiency of GNN inference and frequent Out-Of-Memory (OOM) problem limit the successful application of GNN on edge computing platforms.
To tackle these problems, a feature decomposition approach is proposed for memory efficiency optimization of GNN inference.
The proposed approach could achieve outstanding optimization on various GNN models, covering a wide range of datasets, which speeds up the inference by up to 3x.
Furthermore, the proposed feature decomposition could significantly reduce the peak memory usage (up to 5x in memory efficiency improvement) and mitigate OOM problems during GNN inference.

#### Requirements

Recent versions of PyTorch, numpy, torch_geometric(1.6.3) are required. 


####Contents
There are two main top-level scripts in this repo:

    1.test_gnn_layer.py: runs a gnn feature decomposition method on single GNN layer.
    2.test_gnn_total.py: runs a gnn feature decomposition method on total gnn models.
    
#### Running the code
##### test single gnn layer by our feature decomposition method.
    cd test
    python test_gnn_layer.py --hidden=32 --agg="gas" --m="GCN" --layer=32 --data="rd"
    
##### test total gnn model by our feature decomposition method.
    cd test
    python test_gnn_total.py --hidden=32 --agg="gas" --m="GCN" --layer="32,41" --data="rd"

- hidden: the hidden layer size of gnn.
- agg: the aggregate model, include spmm and gas. if using feature decomposition, there shoule be "gas".
- m: the gnn model name,include "GCN,GAT,GIN,SAGE".
- layer: the layers of feature decomposition along dimension of feature vector, the basic gnn inference using 1 layer.
if test total gnn model,there should be two parameters.
- data: dataset name.


