import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch


def main(locations):
    environment = nx.Graph()

    for v1 in locations:
        for v2 in locations:
            environment.add_edge(v1, v2)

    environment =  nx.adjacency_matrix(environment)

    # print(environment)
    adj = sp.csr_matrix(environment)
    # print(adj.toarray())
    adj_norm = preprocess_graph(adj)
    print("HERE")
    print(adj_norm)
    return adj_norm


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    # print(adj_)
    # exit()
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# def hello(locations):
# #     locations = [1,2,3,4]
#     print("LOC", locations)
#     return main(locations)