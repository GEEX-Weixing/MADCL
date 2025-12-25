import pandas as pd
import networkx as nx
import json
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from torch_geometric.data import Data

def AggTranProbMat(G, step):
    '''aggregated K-step transition probality'''
    G = MyScaleSimMat(G)
    G = csc_matrix.toarray(G)
    A_k = G
    A = G
    for k in np.arange(2, step + 1):
        A_k = np.matmul(A_k, G)
        A = A + A_k / k
    return A

def MyScaleSimMat(W):
    '''L1 row norm of a matrix'''
    rowsum = np.array(np.sum(W, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    W = r_mat_inv.dot(W)
    return W

def ComputePPMI(A):
    '''compute PPMI, given aggregated K-step transition probality matrix as input'''
    np.fill_diagonal(A, 0)
    A = MyScaleSimMat(A)
    (p, q) = np.shape(A)
    col = np.sum(A, axis=0)
    col[col == 0] = 1
    PPMI = np.log((float(p) * A) / col[None, :])
    IdxNan = np.isnan(PPMI)
    PPMI[IdxNan] = 0
    PPMI[PPMI < 0] = 0
    return PPMI

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_graph(graph_path):
    data = pd.read_csv(graph_path)
    edges = data.values.tolist()
    edges = [[int(edge[0]), int(edge[1])] for edge in edges]
    graph = nx.from_edgelist(edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    # 根据节点在g.nodes中出现的先后次序把节点重新编号
    node_id_map = {}
    nodes = list(graph.nodes)
    for i in range(len(nodes)):
        node_id_map[nodes[i]] = i
    adj = nx.to_scipy_sparse_array(graph ,format="coo")
    return adj ,graph.number_of_nodes(), graph.number_of_edges(), node_id_map

def load_features(features_path, node_id_map):
    features = json.load(open(features_path))
    features = {node_id_map[int(k)]: [int(val) for val in v] for k, v in features.items()}
    return features

def create_onehot_embedding(features, n_nodes):
    x = np.zeros((n_nodes, 3170))
    for row_idx, col_idxes in features.items():
        x[row_idx, col_idxes] = np.ones_like(col_idxes)
    return torch.FloatTensor(x)

def load_labels(target_path ,node_id_map):
    data = pd.read_csv(target_path)[["mature" ,"new_id"]]
    data.sort_values(by=['new_id'] ,inplace=True)
    data = data[~data.duplicated(subset="new_id")]  # 去掉重复的行
    pos_idx_ = data[data['mature' ]==True]['new_id'].tolist()
    neg_idx_ = data[data['mature' ]==False]['new_id'].tolist()
    pos_idx = [node_id_map[n] for n in pos_idx_]
    neg_idx = [node_id_map[n] for n in neg_idx_]
    y = np.zeros((len(pos_idx ) +len(neg_idx) ,2))
    y[pos_idx] = np.array([0 ,1])
    y[neg_idx] = np.array([1 ,0])
    return torch.FloatTensor(y)


def get_data(name):
    adj, number_of_nodes, number_of_edges, node_id_map = load_graph('/share/home/u20526/wx/cycle_vgae_gan/data/edge/{}_edges.csv'.format(name))
    features = load_features('/share/home/u20526/wx/cycle_vgae_gan/data/features/{}.json'.format(name), node_id_map)
    features = create_onehot_embedding(features, number_of_nodes)
    labels = load_labels('/share/home/u20526/wx/cycle_vgae_gan/data/target/{}_target.csv'.format(name), node_id_map)
    labels = torch.argmax(labels, dim=1)
    indices = np.vstack([adj.row, adj.col])
    edge_index = torch.tensor(indices, dtype=torch.long)
    # A_k = AggTranProbMat(adj, 3)
    # PPMI_ = ComputePPMI(A_k)
    # n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
    # n_PPMI_mx = lil_matrix(n_PPMI_)
    # a_ppmi = sparse_mx_to_torch_sparse_tensor(n_PPMI_mx)
    data = Data(x=features, edge_index=edge_index, y=labels, num_classes=int(labels.max() + 1))
    return data

def get_data_t(name):
    adj, number_of_nodes, number_of_edges, node_id_map = load_graph('/share/home/u20526/wx/cycle_vgae_gan/data/edge/{}_edges.csv'.format(name))
    features = load_features('/share/home/u20526/wx/cycle_vgae_gan/data/features/{}.json'.format(name), node_id_map)
    features = create_onehot_embedding(features, number_of_nodes)
    labels = load_labels('/share/home/u20526/wx/cycle_vgae_gan/data/target/{}_target.csv'.format(name), node_id_map)
    labels = torch.argmax(labels, dim=1)
    indices = np.vstack([adj.row, adj.col])
    edge_index = torch.tensor(indices, dtype=torch.long)
    A_k = AggTranProbMat(adj, 3)
    PPMI_ = ComputePPMI(A_k)
    n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
    n_PPMI_mx = lil_matrix(n_PPMI_)
    a_ppmi = sparse_mx_to_torch_sparse_tensor(n_PPMI_mx)
    data = Data(x=features, edge_index=edge_index, y=labels, num_classes=int(labels.max() + 1))
    return data, a_ppmi

def get_data_2(name):
    adj, number_of_nodes, number_of_edges, node_id_map = load_graph('/share/home/u20526/wx/cycle_vgae_gan/data/edge/{}_edges.csv'.format(name))
    features = load_features('/share/home/u20526/wx/cycle_vgae_gan/data/features/{}.json'.format(name), node_id_map)
    features = create_onehot_embedding(features, number_of_nodes)
    labels = load_labels('/share/home/u20526/wx/cycle_vgae_gan/data/target/{}_target.csv'.format(name), node_id_map)
    labels = torch.argmax(labels, dim=1)
    indices = np.vstack([adj.row, adj.col])
    edge_index = torch.tensor(indices, dtype=torch.long)
    A_k = AggTranProbMat(adj, 3)
    PPMI_ = ComputePPMI(A_k)
    n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
    n_PPMI_mx = lil_matrix(n_PPMI_)
    a_ppmi = sparse_mx_to_torch_sparse_tensor(n_PPMI_mx)
    data = Data(x=features, edge_index=edge_index, y=labels, num_classes=int(labels.max() + 1))
    return data, a_ppmi

