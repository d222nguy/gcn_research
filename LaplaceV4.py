import numpy as np 
import scipy.sparse as sp 
from collections import deque
global N
def construct_adj_list(rows, cols):
    global N
    adj_list = [[] for i in range(N)]
    # print('N = ', N)
    # print(adj_list)
    for i in range(len(rows)):
        print(rows[i], cols[i])
        adj_list[rows[i]].append(cols[i])
        #adj_list[cols[i]].append(rows[i])
    print(adj_list)
    return adj_list
def run_bfs(adj_list, src):
    global N
    dq = deque()
    dist = [-1] * N
    dq.append(src)
    dist[src] = 0
    while (dq):
        u = dq.popleft()
        for v in adj_list[u]:
            if dist[v] == -1:
                dq.append(v)
                dist[v] = dist[u] + 1
    return dist
def build_graph(adj_coo, K):
    global N
    rows, cols = adj_coo.nonzero()
    N = adj_coo.shape[0]
    #adjacency list
    adj_list = construct_adj_list(rows, cols)
    adj_k_list = [[] for i in range(N)]
    avg = [-1] * N
    #distance matrix
    dist_mtx = []
    for i in range(N):
        dist = run_bfs(adj_list, i)
        dist_mtx.append(dist)
        for u in range(N):
            if dist[u] > 0 and dist[u] <= K:
                adj_k_list[i].append(u)
        avg[i] = sum(dist)/len(adj_k_list[i])
    dist_mtx = np.array(dist_mtx)
    print("adj_k_list", adj_k_list)
    return dist_mtx, avg

def main():
    '''driver code to test the method'''
    #build COO matrix
    rows = [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6]
    cols = [1, 2, 3, 4, 6, 0, 0, 4, 5, 0, 4, 5, 0, 2, 3, 2, 3, 6, 0, 5]
    data = [1] * len(rows)
    A = sp.coo_matrix((data, (rows, cols)), shape = (7, 7))
    k = 2
    dist_mtx, avg = build_graph(A, K = k)
    print(dist_mtx)
    print(avg)
    #A_arr = A.toarray()
if __name__ == "__main__":
    main()