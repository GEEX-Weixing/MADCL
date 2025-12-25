import torch
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from torch_geometric.data import Data


def edgeidx_to_adj(edge_source, edge_target, num_node):
    data = np.ones(len(edge_source))
    adj = sp.csr_matrix((data, (edge_source, edge_target)),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])
    return adj

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

def get_data_m(name):
    graph = torch.load('/share/home/u20526/wx/cycle_vgae_gan/data/{}_labels_20.pt'.format(name))
    features = graph.x
    adj = edgeidx_to_adj(graph.edge_index[0], graph.edge_index[1], graph.num_nodes)
    graph.adj = adj
    graph.edge_index = torch.from_numpy(np.array([adj.nonzero()[0], adj.nonzero()[1]])).long()
    graph.num_classes = torch.max(graph.y) + 1

    idx = np.arange(graph.num_nodes)
    np.random.shuffle(idx)
    idx_len = idx.shape[0]
    graph.source_training_mask = idx[0:int(0.6 * idx_len)]
    graph.source_validation_mask = idx[int(0.6 * idx_len):int(0.8 * idx_len)]
    graph.source_testing_mask = idx[int(0.8 * idx_len):]
    graph.target_validation_mask = idx[0:int(0.2 * idx_len)]
    graph.target_testing_mask = idx[int(0.2 * idx_len):]
    graph.source_mask = idx
    graph.target_mask = idx

    data = Data(x=features, edge_index=graph.edge_index, y=graph.y, num_classes=graph.num_classes)

    row = graph.edge_index[0].cpu().numpy()  # 起点节点
    col = graph.edge_index[1].cpu().numpy()  # 终点节点

    # 使用scipy的coo格式创建稀疏矩阵，然后转换为csc格式
    adj_matrix = sp.coo_matrix((torch.ones(graph.edge_index.size(1)).cpu().numpy(), (row, col)), shape=(graph.num_nodes, graph.num_nodes))
    # 转换为csc格式
    adj = adj_matrix.tocsc()
    A_k = AggTranProbMat(adj, 3)
    PPMI_ = ComputePPMI(A_k)
    n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
    n_PPMI_mx = lil_matrix(n_PPMI_)
    a_ppmi = sparse_mx_to_torch_sparse_tensor(n_PPMI_mx)
    return data, a_ppmi

def get_data(name):
    graph = torch.load('/share/home/u20526/wx/cycle_vgae_gan/data/{}_labels_20.pt'.format(name))
    features = graph.x
    adj = edgeidx_to_adj(graph.edge_index[0], graph.edge_index[1], graph.num_nodes)
    graph.adj = adj
    graph.edge_index = torch.from_numpy(np.array([adj.nonzero()[0], adj.nonzero()[1]])).long()
    graph.num_classes = torch.max(graph.y) + 1

    idx = np.arange(graph.num_nodes)
    np.random.shuffle(idx)
    idx_len = idx.shape[0]
    graph.source_training_mask = idx[0:int(0.6 * idx_len)]
    graph.source_validation_mask = idx[int(0.6 * idx_len):int(0.8 * idx_len)]
    graph.source_testing_mask = idx[int(0.8 * idx_len):]
    graph.target_validation_mask = idx[0:int(0.2 * idx_len)]
    graph.target_testing_mask = idx[int(0.2 * idx_len):]
    graph.source_mask = idx
    graph.target_mask = idx

    data = Data(x=features, edge_index=graph.edge_index, y=graph.y, num_classes=graph.num_classes)

    # row = graph.edge_index[0].cpu().numpy()  # 起点节点
    # col = graph.edge_index[1].cpu().numpy()  # 终点节点
    #
    # # 使用scipy的coo格式创建稀疏矩阵，然后转换为csc格式
    # adj_matrix = sp.coo_matrix((torch.ones(graph.edge_index.size(1)).cpu().numpy(), (row, col)), shape=(graph.num_nodes, graph.num_nodes))
    # # 转换为csc格式
    # adj = adj_matrix.tocsc()
    # A_k = AggTranProbMat(adj, 3)
    # PPMI_ = ComputePPMI(A_k)
    # n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
    # n_PPMI_mx = lil_matrix(n_PPMI_)
    # a_ppmi = sparse_mx_to_torch_sparse_tensor(n_PPMI_mx)
    return data

def get_data_3(name):
    graph = torch.load('/share/home/u20526/wx/cycle_vgae_gan/data/{}_labels_20.pt'.format(name))
    features = graph.x
    adj = edgeidx_to_adj(graph.edge_index[0], graph.edge_index[1], graph.num_nodes)
    graph.adj = adj
    graph.edge_index = torch.from_numpy(np.array([adj.nonzero()[0], adj.nonzero()[1]])).long()
    graph.num_classes = torch.max(graph.y) + 1

    idx = np.arange(graph.num_nodes)
    np.random.shuffle(idx)
    idx_len = idx.shape[0]
    graph.source_training_mask = idx[0:int(0.6 * idx_len)]
    graph.source_validation_mask = idx[int(0.6 * idx_len):int(0.8 * idx_len)]
    graph.source_testing_mask = idx[int(0.8 * idx_len):]
    graph.target_validation_mask = idx[0:int(0.2 * idx_len)]
    graph.target_testing_mask = idx[int(0.2 * idx_len):]
    graph.source_mask = idx
    graph.target_mask = idx

    data = Data(x=features, edge_index=graph.edge_index, y=graph.y, num_classes=graph.num_classes)

    row = graph.edge_index[0].cpu().numpy()  # 起点节点
    col = graph.edge_index[1].cpu().numpy()  # 终点节点

    # 使用scipy的coo格式创建稀疏矩阵，然后转换为csc格式
    adj_matrix = sp.coo_matrix((torch.ones(graph.edge_index.size(1)).cpu().numpy(), (row, col)), shape=(graph.num_nodes, graph.num_nodes))
    # 转换为csc格式
    adj = adj_matrix.tocsc()
    A_k = AggTranProbMat(adj, 3)
    PPMI_ = ComputePPMI(A_k)
    n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
    n_PPMI_mx = lil_matrix(n_PPMI_)
    a_ppmi = sparse_mx_to_torch_sparse_tensor(n_PPMI_mx)
    return data, a_ppmi


