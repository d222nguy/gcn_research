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
    d[src] = 0
    for k in range(K): #modify K
        for i in range(N):
            for j in range(len(adj[i])):
                u = adj[i][j]
                if d[i] > d[u] + cost[i][j]:
                    d[i] = d[u] + cost[i][j]
    return d
def calculate_dist(adj, cost, K):
    #Calculate distance matrix dist, by calculating d(i) for every vertex v
    #Input: adj list, cost list, K
    #Output: dist matrix
    #Time complexity: O(K * |V| * |E|)
    #Auxilary space: O(|V|^2)
    global N
    dist = []
    for i in range(N):
        d = run_ford_bellman(i, adj, cost, K)
        dist.append(d)
    dist = np.array(dist)
    return dist

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
    K = 1
    dist = calculate_dist(adj_list, cost_list, K)
    print(dist) #Notice: dist[0, 3] == 4
    K = 2
    dist = calculate_dist(adj_list, cost_list, K)
    print(dist) #Notice: dist[0, 3] == 3
if __name__ == "__main__":
    main()