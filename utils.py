import numpy as np
import scipy.sparse as sp
import torch
import os

#THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
#path = os.path.join(THIS_FOLDER, 'cora.content')

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


#def load_data(path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/cora/'), dataset="cora"):
def load_data(path = os.path.join(os.getcwd(),'pygcn/data/cora/'), dataset="cora"):

    ## Replace path = "../data/cora/" by the absolute path: 
    ## path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/cora/')
    """Load citation network dataset (cora only for now)"""
   
    print('Loading {} dataset...'.format(dataset))
    
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
  
  #  adj1 = adj
  #  adj2 = sp_second_tied_prev(adj)
  #  adj3 = sp_square(adj)
  #  adj4 = sp_second_tied(adj)
  #  adj = adj1
    
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def sp_square(adj):   
    '''Perform square operation on COO-sparsed matrix'''
    '''Adj has no self-loop'''
    adj_csr = sp.csr_matrix(adj)
    tmp = sp.csr_matrix.dot(adj_csr, adj_csr) # square of a matrix
    tmp = tmp - sp.diags(sp.csr_matrix.diagonal(tmp)) # Remove Self-loop 
    adj2 = tmp + adj_csr.multiply(adj_csr) # Add back component-wise squares of all non-self-loop edges of adj 
    adj2 = adj2.tocoo()
    return adj2 #sp.csr_matrix.dot(adj_csr, adj_csr)

def sp_second_tied_prev(adj_coo): # This version contains self-loop and equal weights of 1st-tied and 2nd-tied connections
    '''Input: adj in COO format; Output: adj second_tied in COO format'''
    rows, cols = adj_coo.nonzero()
    N = adj_coo.shape[0]
    adj_list = [set() for i in range(N)] #adjacency list initialization
    for i in range(len(rows)):
        adj_list[rows[i]].add(cols[i])
        adj_list[cols[i]].add(rows[i])
    new_adj_list = adj_list.copy()
    #new adjacency matrix: if (i, j) then new neighbors of i = old neighbors of i union neighbors of j
    for i in range(len(adj_list)):
        for j in adj_list[i]:
            new_adj_list[i] = new_adj_list[i].union(adj_list[j])
    #construct resulting matrix from new adjacency list
    new_rows = []
    new_cols = []
    for i in range(len(new_adj_list)):
        for j in new_adj_list[i]:
            new_rows.append(i)
            new_cols.append(j)
    data = [1 for i in range(len(new_cols))]
    new_adj_coo = sp.coo_matrix((data, (new_rows, new_cols)), shape = (N, N)) 
    return new_adj_coo
    
def sp_second_tied(adj): # This version contains no self-loop, weight = 1 for 1st-tied connection, weight = 0.5 for 2nd-tied connections
    '''Input: adj in COO format; Output: adj second_tied in COO format'''

    adj = sp.csr_matrix(adj) # getting all nonzero entries of a matrix in csr format could be faster than in coo format?
    rows, cols = adj.nonzero()

    # Extract all direct adjacency (neighborhood) of each vertex
    N = adj.shape[0]
    nbd = [set() for i in range(N)] # adjacency(neighbourhood) list initialization, row by row
    for i in range(len(rows)):
        nbd[rows[i]].add(cols[i])
     #   nbhd[cols[i]].add(rows[i]) # If the input matrix is symmetric, we don't need this line

    # Construct all first and second-tied adjacency of each vertex
    nbd2 = nbd.copy()
    #new adjacency matrix: if (i, j) then new neighbors of i = old neighbors union neighbors of j
    for i in range(len(nbd)):
        for j in nbd[i]:
            nbd2[i] = nbd2[i].union(nbd[j])

    # Construct resulting matrix from new adjacency list
    new_rows = []
    new_cols = []
    for i in range(len(nbd2)):
        for j in nbd2[i]:
            new_rows.append(i)
            new_cols.append(j)
    data = [0.5 for i in range(len(new_cols))]
    data0 = [0.5 for i in range(len(cols))]
    b = sp.coo_matrix((data, (new_rows, new_cols)), shape = (N, N)) + sp.coo_matrix((data0, (rows, cols)), shape = (N, N)) 

    b = b - sp.diags(b.diagonal(),0) # Remove Self-loop 

    b = b.tocoo()
    return b

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def main():
    row = [0, 1, 2]
    col = [1, 0, 2]
    data = [1, 1, 1]
    A = sp.coo_matrix((data, (row, col)))
    B = sp_square(A)
    print(B.toarray())
if __name__ == '__main__':
    main()