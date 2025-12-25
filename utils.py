import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import scipy.sparse as sp
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import random
from torch_geometric.data import Data

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def lossFunction(recon_x, x, mu, log_var):
    # bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    bce = F.mse_loss(recon_x, x)
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kld


def gamma_re_loss(x_orig, x_recon, gamma):
    """
    实现论文公式 (3): L_rec = mean( max( ||x - x_recon||_2 - gamma, 0 ) )
    """
    # 计算每个样本的 L2 范数（对非 batch 维度求和）
    l2_norm = torch.norm(x_orig - x_recon, p=2, dim=tuple(range(1, x_orig.dim())))

    loss = torch.clamp(l2_norm - gamma, min=0.0)

    # 返回平均损失
    return loss.mean()

def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj, device):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized).to(device)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def get_norm(adj_train):
    adj = adj_train
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    # pos_weight =  torch.sparse.FloatTensor(torch.FloatTensor(pos_weight))
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    return adj_label, torch.tensor(pos_weight), norm

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return torch.tensor(features)

def adjust_learning_rate(optimizer, epoch, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def get_pos_weight(adj, device):
#     pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
#     return pos_weight.to(device)

def for_gae(features, adj, device):
    n_nodes, feat_dim = features.shape
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj, device)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    return adj_label.to(device), torch.tensor(norm).to(device), torch.tensor(pos_weight).to(device)

def compute_accuracy_teacher_mask(prediction, label, mask):
    correct = 0
    indices = torch.nonzero(mask)
    for i in indices:
        if prediction[i] == label[i]:
            correct += 1
    accuracy = correct / len(prediction) * 100
    return accuracy

def compute_accuracy_teacher(prediction, label):
    correct = 0
    # label = torch.argmax(label, dim=1)
    for i in range(len(label)):
        if prediction[i] == label[i]:
            correct += 1
    accuracy = correct / len(prediction) * 100
    return accuracy


import numpy as np
import torch
import networkx as nx
from sklearn.metrics import pairwise_distances
import community as community_louvain


def sample_subgraph(adj_matrix, features, sampling_type="community", target_size=200):
    """
    从密集邻接矩阵和特征矩阵采样子图

    参数:
    adj_matrix: 密集邻接矩阵 (numpy或torch.Tensor) [N, N]
    features: 节点特征矩阵 [N, F]
    sampling_type: 采样类型 ("community", "pagerank", "random_walk")
    target_size: 目标子图节点数

    返回:
    sub_adj: 子图邻接矩阵 [target_size, target_size]
    sub_features: 子图特征矩阵 [target_size, F]
    """
    # 转换为NetworkX图
    G = nx.from_numpy_array(adj_matrix.cpu().numpy() if torch.is_tensor(adj_matrix) else adj_matrix)

    if sampling_type == "community":
        return community_based_sampling(G, features, target_size)
    elif sampling_type == "pagerank":
        return pagerank_sampling(G, features, target_size)
    elif sampling_type == "random_walk":
        return random_walk_sampling(G, features, target_size)
    else:
        raise ValueError(f"未知采样类型: {sampling_type}")


def community_based_sampling(G, features, target_size):
    """社区结构感知采样"""
    # 使用Louvain算法检测社区
    partition = community_louvain.best_partition(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    # 按社区大小比例采样
    sampled_nodes = []
    total_nodes = len(G)

    for comm_id, nodes in communities.items():
        ratio = len(nodes) / total_nodes
        n_sample = max(1, int(target_size * ratio))

        # 在社区内均匀采样
        comm_sample = np.random.choice(nodes, size=n_sample, replace=False)
        sampled_nodes.extend(comm_sample)

    # 确保采样节点数符合目标
    if len(sampled_nodes) > target_size:
        sampled_nodes = np.random.choice(sampled_nodes, size=target_size, replace=False)
    elif len(sampled_nodes) < target_size:
        # 补充随机节点
        extra = np.random.choice(list(G.nodes), size=target_size - len(sampled_nodes), replace=False)
        sampled_nodes = np.concatenate([sampled_nodes, extra])

    return extract_subgraph(sampled_nodes, G, features)


def pagerank_sampling(G, features, target_size):
    """PageRank重要性采样"""
    pr = nx.pagerank(G)
    sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)

    # 分层采样
    high = [n for n, _ in sorted_nodes[:int(target_size * 0.5)]]
    mid = [n for n, _ in sorted_nodes[int(target_size * 0.5):int(target_size * 0.8)]]
    low = [n for n, _ in sorted_nodes[-int(target_size * 0.2):]]

    sampled_nodes = np.array(high + mid + low)
    return extract_subgraph(sampled_nodes, G, features)


def random_walk_sampling(G, features, target_size, walk_length=40, restart_prob=0.2):
    """随机游走采样"""
    sampled_nodes = set()
    current = np.random.choice(list(G.nodes))

    while len(sampled_nodes) < target_size:
        if np.random.rand() < restart_prob or not list(G.neighbors(current)):
            current = np.random.choice(list(G.nodes))
        else:
            current = np.random.choice(list(G.neighbors(current)))
        sampled_nodes.add(current)

    return extract_subgraph(list(sampled_nodes), G, features)


def extract_subgraph(sampled_nodes, G, features):
    """
    从原始图中提取子图
    """
    # 对节点索引排序
    sampled_nodes = sorted(sampled_nodes)

    # 创建子图
    subgraph = G.subgraph(sampled_nodes)

    # 获取邻接矩阵
    adj_matrix = nx.to_numpy_array(subgraph)

    # 获取特征矩阵
    if torch.is_tensor(features):
        sub_features = features[sampled_nodes]
    else:
        sub_features = features[sampled_nodes, :]

    return sp.coo_matrix(torch.tensor(adj_matrix)), sub_features


import torch.nn as nn
from torch import Tensor


class SemanticConsistency(nn.Module):
    """
    Semantic consistency loss is introduced by
    `CyCADA: Cycle-Consistent Adversarial Domain Adaptation (ICML 2018) <https://arxiv.org/abs/1711.03213>`_

    This helps to prevent label flipping during image translation.

    Args:
        ignore_index (tuple, optional): Specifies target values that are ignored
            and do not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: ().
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples::

        >>> loss = SemanticConsistency()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, ignore_index=(), reduction='mean'):
        super(SemanticConsistency, self).__init__()
        self.ignore_index = ignore_index
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        for class_idx in self.ignore_index:
            target[target == class_idx] = -1
        return self.loss(input, target)


def entropy_loss_f(logit):
    probs_t = F.softmax(logit, dim=-1)
    probs_t = torch.clamp(probs_t, min=1e-9, max=1.0)
    entropy_loss = torch.mean(torch.sum(-probs_t * torch.log(probs_t), dim=-1))
    return entropy_loss

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


def gamma_constrained_similarity(x, y, gamma):
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # [N, M, D]
    l2_dist = torch.norm(diff, p=2, dim=-1)  # [N, M]
    sim = -torch.clamp(l2_dist - gamma, min=0.0)  # [N, M]
    return sim


def intra_domain_cl_loss(z, recon_z, shuffle_recon_z, a, gamma=0.01, temperature=1.0):
    N, D = z.shape
    device = z.device

    # 1. 正样本相似度: s(z[i], recon_z[i])
    pos_sim = -torch.clamp(torch.norm(z - recon_z, p=2, dim=1) - gamma, min=0.0) / temperature  # [N]

    # 2. 负样本相似度: s(z[i], shuffle_recon_z[j]) for all j
    # 使用广播计算所有 i,j 对
    neg_sim = gamma_constrained_similarity(z, shuffle_recon_z, gamma) / temperature  # [N, N]

    # 3. 构建 logits: [pos_sim[i], neg_sim[i, :]]
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [N, 1+N]

    # 4. 数值稳定：减去每行最大值（注意：相似度 ≤ 0，最大值接近 0）
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # 5. 计算 log prob of positive
    exp_logits = torch.exp(logits)
    log_prob = logits[:, 0] - torch.log(exp_logits.sum(dim=1))  # [N]

    # 6. 加权损失
    weighted_loss = -a * log_prob
    total_weight = a.sum()
    if total_weight == 0:
        return torch.tensor(0.0, device=device)
    loss = weighted_loss.sum() / total_weight

    return loss


import torch
import torch.nn.functional as F


def intra_domain_contrastive_loss(z, recon_z, shuffle_recon_z, temperature=0.05):
    N, D = z.shape
    device = z.device
    z_norm = F.normalize(z, dim=1)  # [N, D]
    recon_z_norm = F.normalize(recon_z, dim=1)  # [N, D]
    shuffle_recon_z_norm = F.normalize(shuffle_recon_z, dim=1)  # [N, D]
    pos_sim = torch.sum(z_norm * recon_z_norm, dim=1) / temperature  # [N]
    neg_sim = torch.mm(z_norm, shuffle_recon_z_norm.t()) / temperature  # [N, N]
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [N, 1 + N]
    labels = torch.zeros(N, dtype=torch.long, device=device)  # [N]
    loss = F.cross_entropy(logits, labels)
    return loss

def intra_domain_contrastive_loss_2(z, z_1, z_2, decoder, num_shuffles=150, shuffle_batch_size=10, temperature=0.05, generator=None):
    N, D_z = z.shape
    device = z.device

    # Normalize anchor once
    z_norm = F.normalize(z, dim=1)  # [N, D_z]

    # Positive reconstruction (only once)
    with torch.no_grad() if shuffle_batch_size > 0 else torch.enable_grad():
        _, pos_recon = decoder(torch.cat((z_1, z_2), 1))
    pos_recon_norm = F.normalize(pos_recon.detach(), dim=1)  # detach to save memory if needed
    pos_sim = torch.sum(z_norm * pos_recon_norm, dim=1) / temperature  # [N]

    total_loss = 0.0
    num_batches = (num_shuffles + shuffle_batch_size - 1) // shuffle_batch_size

    for batch_idx in range(num_batches):
        # Determine how many shuffles in this batch
        start = batch_idx * shuffle_batch_size
        end = min(start + shuffle_batch_size, num_shuffles)
        current_batch_size = end - start

        # Generate current batch of negative reconstructions
        neg_sims_list = []
        for _ in range(current_batch_size):
            # perm = torch.randperm(N, device=device)
            perm = torch.randperm(N, device=device, generator=generator)
            z2_shuffled = z_2[perm]
            _, neg_recon = decoder(torch.cat((z_1, z2_shuffled), 1))  # [N, D_z]
            neg_recon_norm = F.normalize(neg_recon, dim=1)  # [N, D_z]
            neg_sim = torch.sum(z_norm * neg_recon_norm, dim=1) / temperature  # [N]
            neg_sims_list.append(neg_sim)

        # Stack: [N, current_batch_size]
        neg_sims = torch.stack(neg_sims_list, dim=1)  # [N, B]

        # Build logits: [N, 1 + B]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=device)

        # Compute loss for this shuffle batch
        loss = F.cross_entropy(logits, labels)
        total_loss += loss

    # Average over shuffle batches
    avg_loss = total_loss / num_batches
    return avg_loss



# /share/home/u20526/wx/pkdd-gda

def intra_domain_contrastive_loss_3(z, z_1, z_2, decoder, num_shuffles=120, shuffle_batch_size=20,
                                    temperature=0.5, generator=None, domain_predictor=None,
                                    domain_labels=None, weight_min=0.15):
    N, D_z = z.shape
    device = z.device

    # ==================== 自适应权重计算(核心新增) ==========
    if domain_predictor is not None and domain_labels is not None:
        with torch.no_grad():
            # 预测域概率分布
            domain_probs = F.softmax(domain_predictor(z.detach()), dim=-1)  # [N, num_domains]

            # 提取每个样本对应其真实域的置信度
            domain_labels = domain_labels.to(device)
            gt_weights = domain_probs.gather(1, domain_labels.unsqueeze(1)).squeeze(1)  # [N]

            # 应用下限约束: max(weight_min, gt_weight)
            dynamic_weights = torch.clamp(gt_weights, min=weight_min)

            # 转换为1/weight形式（类比原始MAE的1/gt_weight）
            adaptive_weights = 1.0 / dynamic_weights  # [N]
    else:
        # 不使用自适应权重时，所有权重为1.0
        adaptive_weights = torch.ones(N, device=device)

    # ========== 原始逻辑不变 ==========
    z_norm = F.normalize(z, dim=1)

    with torch.no_grad() if shuffle_batch_size > 0 else torch.enable_grad():
        _, pos_recon = decoder(torch.cat((z_1, z_2), 1))
    pos_recon_norm = F.normalize(pos_recon.detach(), dim=1)
    pos_sim = torch.sum(z_norm * pos_recon_norm, dim=1) / temperature

    total_loss = 0.0
    num_batches = (num_shuffles + shuffle_batch_size - 1) // shuffle_batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * shuffle_batch_size
        end = min(start + shuffle_batch_size, num_shuffles)
        current_batch_size = end - start

        # 生成负样本
        neg_sims_list = []
        for _ in range(current_batch_size):
            perm = torch.randperm(N, device=device, generator=generator)
            z2_shuffled = z_2[perm]
            _, neg_recon = decoder(torch.cat((z_1, z2_shuffled), 1))
            neg_recon_norm = F.normalize(neg_recon, dim=1)
            neg_sim = torch.sum(z_norm * neg_recon_norm, dim=1) / temperature
            neg_sims_list.append(neg_sim)

        neg_sims = torch.stack(neg_sims_list, dim=1)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=device)

        # ========== 关键修改：加权损失 ====================
        # 计算每个样本的损失（不使用reduction='mean'）
        loss_per_sample = F.cross_entropy(logits, labels, reduction='none')  # [N]
        # 应用自适应权重
        weighted_loss = (loss_per_sample * adaptive_weights).mean()
        total_loss += weighted_loss

    avg_loss = total_loss / num_batches
    return avg_loss


def intra_domain_contrastive_loss_4(z, z_1, z_2, decoder, num_shuffles=120, shuffle_batch_size=20,
                                    temperature=0.5, generator=None, domain_predictor=None,
                                    domain_labels=None, weight_min=0.15):
    N, D_z = z.shape
    device = z.device

    # ==================== 自适应权重计算(核心新增) ==========
    if domain_predictor is not None and domain_labels is not None:
        with torch.no_grad():
            # 预测域概率分布
            domain_probs = F.softmax(domain_predictor(z.detach()), dim=-1)  # [N, num_domains]

            # 提取每个样本对应其真实域的置信度
            domain_labels = domain_labels.to(device)
            gt_weights = domain_probs.gather(1, domain_labels.unsqueeze(1)).squeeze(1)  # [N]

            # 应用下限约束: max(weight_min, gt_weight)
            dynamic_weights = torch.clamp(gt_weights, min=weight_min)

            # 转换为1/weight形式（类比原始MAE的1/gt_weight）
            adaptive_weights = 1.0 / dynamic_weights  # [N]
    else:
        # 不使用自适应权重时，所有权重为1.0
        adaptive_weights = torch.ones(N, device=device)

    # ========== 原始逻辑不变 ==========
    z_norm = F.normalize(z, dim=1)

    with torch.no_grad() if shuffle_batch_size > 0 else torch.enable_grad():
        _, pos_recon = decoder(torch.cat((z_1, z_2), 1))
    pos_recon_norm = F.normalize(pos_recon.detach(), dim=1)
    pos_sim = torch.sum(z_norm * pos_recon_norm, dim=1) / temperature

    total_loss = 0.0
    num_batches = (num_shuffles + shuffle_batch_size - 1) // shuffle_batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * shuffle_batch_size
        end = min(start + shuffle_batch_size, num_shuffles)
        current_batch_size = end - start

        # 生成负样本
        neg_sims_list = []
        for _ in range(current_batch_size):
            perm = torch.randperm(N, device=device, generator=generator)
            z2_shuffled = z_2[perm]
            _, neg_recon = decoder(torch.cat((z_1, z2_shuffled), 1))
            neg_recon_norm = F.normalize(neg_recon, dim=1)
            neg_sim = torch.sum(z_norm * neg_recon_norm, dim=1) / temperature
            neg_sims_list.append(neg_sim)

        neg_sims = torch.stack(neg_sims_list, dim=1)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=device)

        # ========== 关键修改：加权损失 ====================
        # 计算每个样本的损失（不使用reduction='mean'）
        loss_per_sample = F.cross_entropy(logits, labels, reduction='none')  # [N]
        # 应用自适应权重
        weighted_loss = (loss_per_sample * adaptive_weights).mean()
        total_loss += weighted_loss

    avg_loss = total_loss / num_batches
    return avg_loss

def compute_mmd(x, y, kernel_type='rbf', sigma=1.0):
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    if y.dim() > 2:
        y = y.view(y.size(0), -1)

    batch_size_x = x.size(0)
    batch_size_y = y.size(0)

    # 计算核矩阵
    if kernel_type == 'rbf':
        # 计算X-X对的距离矩阵
        xx = torch.matmul(x, x.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = rx.t()
        dxx = rx + ry - 2 * xx

        # 计算Y-Y对的距离矩阵
        yy = torch.matmul(y, y.t())
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        ryy = ry.t()
        dyy = ry + ryy - 2 * yy

        # 计算X-Y对的距离矩阵
        xy = torch.matmul(x, y.t())
        rx = xx.diag().unsqueeze(1).expand_as(xy)
        ry = yy.diag().unsqueeze(0).expand_as(xy)
        dxy = rx + ry - 2 * xy

        # 应用RBF核函数
        Kxx = torch.exp(-dxx / (2 * sigma ** 2))
        Kyy = torch.exp(-dyy / (2 * sigma ** 2))
        Kxy = torch.exp(-dxy / (2 * sigma ** 2))

    elif kernel_type == 'linear':
        # 线性核 K(x,y) = x^T y
        Kxx = torch.matmul(x, x.t())
        Kyy = torch.matmul(y, y.t())
        Kxy = torch.matmul(x, y.t())
    else:
        raise ValueError(f"不支持的核函数类型: {kernel_type}")

    # 计算MMD
    # MMD^2 = E[K(X,X)] + E[K(Y,Y)] - 2*E[K(X,Y)]
    mmd_xx = Kxx.sum() / (batch_size_x * batch_size_x)
    mmd_yy = Kyy.sum() / (batch_size_y * batch_size_y)
    mmd_xy = Kxy.sum() / (batch_size_x * batch_size_y)

    mmd_value = mmd_xx + mmd_yy - 2 * mmd_xy

    return mmd_value

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    多核或单核高斯核矩阵函数，根据输入样本集x和y，计算返回对应的高斯核矩阵
    Params:
     source: (b1,n)的X分布样本数组
     target:（b2，n)的Y分布样本数组
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
    Return:
      sum(kernel_val): 多个核矩阵之和
    '''
     # 堆叠两组样本，上面是X分布样本，下面是Y分布样本，得到（b1+b2,n）组总样本
    n_samples = int(source.shape[0])+int(target.shape[0])
    total = np.concatenate((source, target), axis=0)
    # 对总样本变换格式为（1,b1+b2,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按行复制
    total0 = np.expand_dims(total,axis=0)
    total0= np.broadcast_to(total0,[int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
    # 对总样本变换格式为（b1+b2,1,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按复制
    total1 = np.expand_dims(total,axis=1)
    total1=np.broadcast_to(total1,[int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
    # total1 - total2 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差
    # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方
    L2_distance_square = np.sum(np.square(total0-total1),axis=2)###ERROR：上版本为np.cumsum，会导致计算出现几何倍数的L2
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = np.sum(L2_distance_square) / (n_samples**2-n_samples)
    # 多核MMD
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # print(bandwidth_list)
    #高斯核函数的数学表达式
    kernel_val = [np.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#多核合并

def MK_MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
     source: (b1,n)的X分布样本数组
     target:（b2，n)的Y分布样本数组
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
 Return:
     loss: MK-MMD loss
    '''
    batch_size = int(source.shape[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # print(kernels)
    # 将核矩阵分成4部分
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    # 这里计算出的n_loss是每个维度上的MK-MMD距离，一般还会做均值化处理
    n_loss= loss / float(batch_size)
    return np.mean(n_loss)

from sklearn.metrics.pairwise import rbf_kernel


def compute_simple_mmd(X_s, X_t, sigma=None):

    if sigma is None:
        # 自动计算sigma
        X_combined = np.vstack([X_s[:1000], X_t[:1000]])  # 限制样本数量以提高效率
        pairwise_dists = np.linalg.norm(X_combined[:, None] - X_combined[None, :], axis=2)
        median_dist = np.median(pairwise_dists[np.triu_indices_from(pairwise_dists, k=1)])
        sigma = median_dist / np.sqrt(2)

    gamma = 1.0 / (2 * sigma ** 2)

    n_s = X_s.shape[0]
    n_t = X_t.shape[0]

    # 计算RBF核矩阵
    K_ss = rbf_kernel(X_s, gamma=gamma)
    K_tt = rbf_kernel(X_t, gamma=gamma)
    K_st = rbf_kernel(X_s, X_t, gamma=gamma)

    # 计算MMD^2
    mmd_sq = (
            (K_ss.sum() - np.trace(K_ss)) / (n_s * (n_s - 1)) +
            (K_tt.sum() - np.trace(K_tt)) / (n_t * (n_t - 1)) -
            2 * K_st.sum() / (n_s * n_t)
    )

    # 确保非负性
    mmd_sq = max(mmd_sq, 0.0)

    # 返回MMD值 (sqrt(MMD^2))
    mmd_value = np.sqrt(mmd_sq)

    return mmd_value