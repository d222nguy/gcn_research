import numpy as np 
import scipy.sparse as sp 
from collections import deque
global N
def construct_adj_list(A_):
    #Construct adjacency list and cost (weight) list from sparse coo matrix A_
    global N
    A = A_.tocsr()
    N = A.shape[0]
    adj_list = [[] for i in range(N)]
    cost_list = [[] for i in range(N)]
    rows, cols = A.nonzero()
    for i in range(len(rows)):
        print(rows[i], cols[i])
        adj_list[rows[i]].append(cols[i])
        cost_list[rows[i]].append(A[rows[i], cols[i]])
    print(adj_list)
    print(cost_list)
    return adj_list, cost_list
def run_ford_bellman(src, adj, cost, K):
    #Run Ford-Bellman with relaxation K times.
    #this way we can guarantee that every shortest paths has at most K edges!
    #Input: source vertex, adj list, cost (weight) list, and K
    #Output: shortest distance from source to every vertex
    #Time complexity: O(K * |E|)
    #Auxilary space: O(|V|)
    global N
    d = [float('inf') for i in range(N)]
    nb = []
    d[src] = 0
    for k in range(K): #modify K
        d_ = d.copy()
        for i in range(N):
            for j in range(len(adj[i])):
                u = adj[i][j]
                if d_[i] > d[u] + cost[i][j]:
                    d_[i] = d[u] + cost[i][j]
        d = d_.copy()
    for i in range(N):
        if d[i] < float('inf'):
            nb.append(i)
    return d, nb
def calculate_dist(adj, cost, K):
    #Calculate distance matrix dist, by calculating d(i) for every vertex v
    #Input: adj list, cost list, K
    #Output: dist matrix
    #Time complexity: O(K * |V| * |E|)
    #Auxilary space: O(|V|^2)
    global N
    dist = []
    N = len(adj)
    print(N)
    neighbors = []
    for i in range(N):
        d, nb = run_ford_bellman(i, adj, cost, K)
        dist.append(d)
        neighbors.append(nb)
    dist = np.array(dist)
    return dist, neighbors

def main():
    '''driver code to test the method'''
    #build COO matrix
    #using example cua co
    rows = [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6]
    cols = [1, 2, 3, 4, 6, 0, 0, 4, 5, 0, 4, 5, 0, 2, 3, 2, 3, 6, 0, 5]
    data = [3, 1, 4, 1, 5, 3, 1, 9, 2, 4, 2, 5, 1, 9, 2, 2, 5, 3, 5, 3]
    A = sp.coo_matrix((data, (rows, cols)), shape = (7, 7))
    #print(A[0, 0])
    adj_list, cost_list = construct_adj_list(A)
    print('=======================K = 1======================')
    K = 1
    dist, neighbors = calculate_dist(adj_list, cost_list, K)
    print('dist = ', dist) #Notice: dist[0, 3] == 4,
    print('neighbors = ', neighbors)
    print('=======================K = 2======================')
    K = 2
    dist, neighbors = calculate_dist(adj_list, cost_list, K)
    print('dist = ', dist) #Notice: dist[0, 3] == 3, dist[3, 2] = 5 (3->0->2)
    print('neighbors = ', neighbors)
    print('=======================K = 3======================')
    K = 3
    dist, neighbors = calculate_dist(adj_list, cost_list, K)
    print('dist = ', dist) #Notice: dist[0, 3] == 3, dist[3, 2] = 4 (3->4->0->2)
    print('neighbors = ', neighbors)
if __name__ == "__main__":
    main()