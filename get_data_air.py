import networkx as nx
import numpy as np
import torch
from collections import defaultdict
from torch_geometric.data import Data
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix

def construct_pyg(graph, labels):
    node_mapping = defaultdict(int)
    relabels = []
    for node in sorted(list(graph.nodes())):
        node_mapping[node] = len(node_mapping)
        relabels.append(labels[node])
    assert len(node_mapping) == len(labels)

    edge_index = []
    for edge in graph.edges():
        edge_index.append([node_mapping[edge[0]], node_mapping[edge[1]]])
        edge_index.append([node_mapping[edge[1]], node_mapping[edge[0]]])

    # Convert to PyG format: edge_index should be a 2xE tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return edge_index, relabels


def read_struct_net(file_path, label_path):
    g = nx.Graph()
    with open(file_path) as IN:
        for line in IN:
            tmp = line.strip().split()
            g.add_edge(int(tmp[0]), int(tmp[1]))
    labels = dict()
    with open(label_path) as IN:
        IN.readline()
        for line in IN:
            tmp = line.strip().split(' ')
            labels[int(tmp[0])] = int(tmp[1])
    return g, labels


def degree_bucketing(graph, max_degree):
    features = torch.zeros([graph.number_of_nodes(), max_degree])
    for i in range(graph.number_of_nodes()):
        try:
            features[i][min(graph.in_degree(i), max_degree - 1)] = 1
        except:
            features[i][0] = 1
    return features

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

def get_data(name, d="s"):
    g, labels = read_struct_net(file_path='data/{}-airports.edgelist'.format(name),
                                label_path='data/labels-{}-airports.txt'.format(name))
    g.remove_edges_from(nx.selfloop_edges(g))

    edge_index, relabels = construct_pyg(g, labels)

    labels = torch.LongTensor(relabels)  # Convert relabels to tensor
    features = degree_bucketing(g, 128)
    num_nodes = labels.shape[0]
    if d == "t":
        features += 0.1

    # Create PyG Data object
    data = Data(x=features, edge_index=edge_index, y=labels, num_nodes=num_nodes, num_classes=int(labels.max() + 1))
    # row = edge_index[0].cpu().numpy()  # 起点节点
    # col = edge_index[1].cpu().numpy()  # 终点节点
    #
    # # 使用scipy的coo格式创建稀疏矩阵，然后转换为csc格式
    # adj_matrix = sp.coo_matrix((torch.ones(edge_index.size(1)).cpu().numpy(), (row, col)), shape=(num_nodes, num_nodes))
    #
    # # 转换为csc格式
    # adj = adj_matrix.tocsc()
    # # print(adj)
    # A_k = AggTranProbMat(adj, 3)
    # PPMI_ = ComputePPMI(A_k)
    # n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
    # n_PPMI_mx = lil_matrix(n_PPMI_)
    # a_ppmi = sparse_mx_to_torch_sparse_tensor(n_PPMI_mx)
    return data

def get_data_air(name, d="s"):
    g, labels = read_struct_net(file_path='data/{}-airports.edgelist'.format(name),
                                label_path='data/labels-{}-airports.txt'.format(name))
    g.remove_edges_from(nx.selfloop_edges(g))

    edge_index, relabels = construct_pyg(g, labels)

    labels = torch.LongTensor(relabels)  # Convert relabels to tensor
    features = degree_bucketing(g, 128)
    num_nodes = labels.shape[0]
    if d == "t":
        features += 0.1

    # Create PyG Data object
    data = Data(x=features, edge_index=edge_index, y=labels, num_nodes=num_nodes, num_classes=int(labels.max() + 1))
    # row = edge_index[0].cpu().numpy()  # 起点节点
    # col = edge_index[1].cpu().numpy()  # 终点节点
    #
    # # 使用scipy的coo格式创建稀疏矩阵，然后转换为csc格式
    # adj_matrix = sp.coo_matrix((torch.ones(edge_index.size(1)).cpu().numpy(), (row, col)), shape=(num_nodes, num_nodes))
    #
    # # 转换为csc格式
    # adj = adj_matrix.tocsc()
    # # print(adj)
    # A_k = AggTranProbMat(adj, 3)
    # PPMI_ = ComputePPMI(A_k)
    # n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
    # n_PPMI_mx = lil_matrix(n_PPMI_)
    # a_ppmi = sparse_mx_to_torch_sparse_tensor(n_PPMI_mx)
    return data

def get_data_a(name, d="s"):
    g, labels = read_struct_net(file_path='data/{}-airports.edgelist'.format(name),
                                label_path='data/labels-{}-airports.txt'.format(name))
    g.remove_edges_from(nx.selfloop_edges(g))

    edge_index, relabels = construct_pyg(g, labels)

    labels = torch.LongTensor(relabels)  # Convert relabels to tensor
    features = degree_bucketing(g, 128)
    num_nodes = labels.shape[0]
    if d == "t":
        features += 0.1

    # Create PyG Data object
    data = Data(x=features, edge_index=edge_index, y=labels, num_nodes=num_nodes, num_classes=int(labels.max() + 1))
    row = edge_index[0].cpu().numpy()  # 起点节点
    col = edge_index[1].cpu().numpy()  # 终点节点

    # 使用scipy的coo格式创建稀疏矩阵，然后转换为csc格式
    adj_matrix = sp.coo_matrix((torch.ones(edge_index.size(1)).cpu().numpy(), (row, col)), shape=(num_nodes, num_nodes))

    # 转换为csc格式
    adj = adj_matrix.tocsc()
    # print(adj)
    A_k = AggTranProbMat(adj, 3)
    PPMI_ = ComputePPMI(A_k)
    n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
    n_PPMI_mx = lil_matrix(n_PPMI_)
    a_ppmi = sparse_mx_to_torch_sparse_tensor(n_PPMI_mx)
    return data, a_ppmi




