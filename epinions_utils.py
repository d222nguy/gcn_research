import networkx as nx 
import pandas as pd 
import numpy as np
from collections import OrderedDict, defaultdict
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity as cosine
# from libc.math cimport sqrt
def cosine(x,y):
    return np.inner(x,y)/np.sqrt(np.dot(x,x)*np.dot(y,y))
import random
params = {} #global parameters of dataset, shared among funcitons
def load_data(fileName):
    rate, rated = defaultdict(list), defaultdict(list)
    #rate: rate[u] = {v_1, .., v_k} means user u rate product v_1,..,v_k
    #rated: rated[v] = {u_1, .., u_k} means product v is rated by u_1,..,u_k
    path = 'data/' + fileName + '.txt'
    ratings = {}
    user_to_idx, idx_to_user = {}, {}
    i = 0
    with open(path) as f:
        for line in f:
            user, product, rating = list(map(int, line.split()))
            if user not in user_to_idx:
                user_to_idx[user] = i 
                idx_to_user[i] = user
                i += 1
            rate[user_to_idx[user]].append(product)
            rated[product].append(user)
            ratings[(user_to_idx[user], product)] = rating
    params["n_users"] = n_users = len(rate)
    params["n_products"] = n_products = len(rated)
    params["n_ratings"] = n_ratings = len(ratings)
    print(n_users, n_products, n_ratings)
    return rate, rated, ratings, user_to_idx, idx_to_user
def build_adj(rated, user_to_idx, idx_to_user):
    n_users = params["n_users"]
    adj = [{} for _ in range(n_users)]
    k = 0
    for product in rated:
        k += 1
        if k % 200 == 0: 
            print("Product {0} / {1}".format(k, len(rated)))
        for i in range(len(rated[product])):
            u = user_to_idx[rated[product][i]]
            for j in range(i + 1, len(rated[product])):
                v = user_to_idx[rated[product][j]]
                #print(u, v)
                adj[u][v] = 1 
                adj[v][u] = 1
    print(adj[0])

    #save adj matrix to file, to save File I/O time (Take too much time, file size ~1Gb)
    # with open('adj.txt', "w+") as f:
    #     for i in range(len(adj)):
    #         print("User {0}/ {1}".format(i, len(adj)))
    #         for v in adj[i]:
    #             f.writelines(str(i) + " " + str(v) + "\n")
    #     f.close()
    return adj
def build_weight(rate, adj, user_to_idx, idx_to_user, ratings):
    print("user_to_idx[1] = ", user_to_idx[1])
    print("idx_to_user[0] = ", idx_to_user[0])
    print("Rating list of user 1, rate[1] = ", rate[user_to_idx[1]])
    print("Rating list of user 18157, rate[user_to_idx[18157]] = ", rate[user_to_idx[18157]])
    print("Rating list of user 48524, rate[user_to_idx[48524]] = ", rate[user_to_idx[48524]])
    # rate_set = [0 for i in range(len(rate))]
    rate_set = [0 for i in range(len(rate))]
    for i in rate:
        rate_set[i] = set(rate[i])
    print("Rating set of user 1, rate[user_to_idx[0]] = ", rate_set[user_to_idx[1]])
    print("Rating set of user 18157, rate[user_to_idx[18157]] = ", rate_set[user_to_idx[18157]])
    print("Rating set of user 48524, rate[user_to_idx[48524]] = ", rate_set[user_to_idx[48524]])
    weight = [{} for _ in range(len(adj))]
    for u in range(len(adj)):
        if u % 1000 == 0:
            print("User {0}/{1}".format(u, len(adj)))
        for v in adj[u]:
            if v < u:
                continue
            mutual_set = rate_set[u].intersection(rate_set[v])
            vector_u, vector_v = np.zeros(len(mutual_set)), np.zeros(len(mutual_set))
            for i, ele in enumerate(mutual_set):
                # print(u, v, ele)
                vector_u[i] = ratings[(u, ele)]
                vector_v[i] = ratings[(v, ele)]
            weight[u][v] = cosine(vector_u, vector_v)
            weight[v][u] = weight[u][v]
            if u == 1788 and v == 6897:
                print(vector_u, vector_v)
            # if len(mutual_set) > 10 and random.random() < 0.0001:
            #     print(u, v, idx_to_user[u], idx_to_user[v], mutual_set)
    print(adj[1788][6897])
    print(weight[6897][1788])
def main():
    rate, rated, ratings, user_to_idx, idx_to_user = load_data('ratings_data')
    adj = build_adj(rated, user_to_idx, idx_to_user)
    weight = build_weight(rate, adj, user_to_idx, idx_to_user, ratings)
if __name__ == "__main__":
    main()