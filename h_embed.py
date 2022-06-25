import numpy as np
import pandas as pd
from treelib import Tree
import pydot
import re
import networkx as nx
import matplotlib.pyplot as plt
import json
import Levenshtein as lst
import math
import random
import time

epoch_num = []
obj_f = []


def dis(x_a, x_b):
    x_a_2 = np.sum(x_a ** 2)
    x_b_2 = np.sum(x_b ** 2)
    x_a_b_2 = np.sum((x_a - x_b) ** 2)
    return np.arccosh(1 + 2 * x_a_b_2 / ((1 - x_a_2) * (1 - x_b_2)))


def load_graph(path):
    g = nx.Graph()
    with open(path, 'r') as f:
        for line in f:
            s = line.split()
            g.add_edge(s[0], s[2])
    return g


def create_negative_graph(g, N):
    neg_g = nx.Graph()
    for n in g.nodes():
        neg_g.add_node(n)
    for n in neg_g.nodes():
        for _ in range(N):
            neg_node = random.choice(tuple(neg_g.nodes() - g[n] - {n}))
            neg_g.add_edge(n, neg_node)
    return neg_g


def sigma(x):
    return 1 / (1 + np.exp(-x))


def objective_function(rws, emb, neg_nodes):
    theta = 0
    for rw in rws:
        for a in len(rw):
            for b in range(max(0, a - 5), min(a + 5, len(rw) - 1)):
                node_a = rw[a]
                node_b = rw[b]
                for i in range(5):
                    node_n = random.choice(neg_nodes[node_a])
                    theta += np.log(sigma(dis(emb[node_a], emb[node_n]) - dis(emb[node_a], emb[node_b])))
    return theta


def proj(x):
    x_l2 = np.dot(x, x)
    if x_l2 >= 1:
        return x / x_l2 - 1e-5
    else:
        return x


def train_epoch(rws, emb, neg_nodes, lr, epoch_no):
    t = 0
    for rw in rws:
        for a in len(rw):
            for b in range(max(0, a - 5), min(a + 5, len(rw) - 1)):
                node_a = rw[a]
                node_b = rw[b]
                if node_a == node_b:
                    continue
                for i in range(5):
                    node_n = random.choice(neg_nodes[node_a])
                    x_a_2 = np.sum(emb[node_a] ** 2)
                    x_b_2 = np.sum(emb[node_b] ** 2)
                    x_n_2 = np.sum(emb[node_n] ** 2)
                    alpha = 1 - x_a_2
                    beta_b = 1 - x_b_2
                    beta_n = 1 - x_n_2
                    gama_ab = 1 + 2 / (alpha * beta_b) * np.sum((emb[node_a] - emb[node_b]) ** 2)
                    gama_an = 1 + 2 / (alpha * beta_n) * np.sum((emb[node_a] - emb[node_n]) ** 2)
                    dot_ab = np.dot(emb[node_a], emb[node_b])
                    dot_an = np.dot(emb[node_a], emb[node_n])
                    derivative_Dab_xa = 4 / (beta_b * np.sqrt(gama_ab ** 2 - 1)) * (
                        (x_b_2 - 2 * dot_ab + 1) / (alpha ** 2) * emb[node_a] - emb[node_b] / alpha)
                    derivative_Dab_xb = 4 / (alpha * np.sqrt(gama_ab ** 2 - 1)) * (
                        (x_a_2 - 2 * dot_ab + 1) / (beta_b ** 2) * emb[node_b] - emb[node_a] / beta_b)
                    derivative_Dan_xa = 4 / (beta_n * np.sqrt(gama_an ** 2 - 1)) * (
                        (x_n_2 - 2 * dot_an + 1) / (alpha ** 2) * emb[node_a] - emb[node_n] / alpha)
                    derivative_Dan_xn = 4 / (alpha * np.sqrt(gama_an ** 2 - 1)) * (
                        (x_a_2 - 2 * dot_an + 1) / (beta_n ** 2) * emb[node_n] - emb[node_a] / beta_n)
                    coe_der_E = 1 - sigma(dis(emb[node_a], emb[node_n]) - dis(emb[node_a], emb[node_b]))
                    derivative_E_xa = coe_der_E * (derivative_Dan_xa - derivative_Dab_xa)
                    derivative_E_xn = coe_der_E * derivative_Dan_xn
                    derivative_E_xb = coe_der_E * (-derivative_Dab_xb)
                    emb[node_a] = proj(emb[node_a] + lr * (1 - x_a_2) ** 2 / 4 * derivative_E_xa)
                    emb[node_b] = proj(emb[node_b] + lr * (1 - x_b_2) ** 2 / 4 * derivative_E_xb)
                    emb[node_n] = proj(emb[node_n] + lr * (1 - x_n_2) ** 2 / 4 * derivative_E_xn)
        t += 1
        if t % 7000 == 0:
            obj_cur = objective_function(g, emb, neg_nodes)
            print(t / g.number_of_nodes() + epoch_no, obj_cur)
            epoch_num.append(t / g.number_of_nodes() + epoch_no)
            obj_f.append(obj_cur)


def new_part():
    pass


if __name__ == "__main__":

    t0 = time.time()
    # g = create_full_graph(info_path, visits_path)
    # g = create_poi_cbg_graph(visits_path)
    # print(g.number_of_nodes(), g.number_of_edges())
    # y = np.array(nx.degree_histogram(g))
    # # result = np.log(y / np.sum(y))
    # result = y.tolist()
    # x = list(range(len(result)))
    # print(x)
    # print(result)
    # plt.scatter(x[1:], result[1:], label="degree distribution")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.ylim(0, 3000)
    # plt.xlabel("degree")
    # plt.ylabel("frequency")
    # plt.title("digree distribution")
    # plt.legend()
    # plt.show()

    g = load_graph("data/ml_1m.txt")

    emb = dict()
    for node in g.nodes():
        emb[node] = np.random.rand(20) * (np.sqrt(5) / 10)
    
    neg_nodes = dict()
    for n in g.nodes():
        neg_nodes[n] = tuple(g.nodes() - g[n] - {n})
    
    # neg_g = create_negative_graph(g, 5)
    # print(neg_g.number_of_nodes(), neg_g.number_of_edges())
    
    print("================ begin training ================", time.time() - t0)
    obj_cur = objective_function(g, emb, neg_nodes)
    epoch_num.append(0)
    obj_f.append(obj_cur)
    print(0, obj_cur, time.time() - t0)
    learning_rate = 0.001
    for epoch in range(1000):
        train_epoch(g, emb, neg_nodes, learning_rate, epoch)
        if learning_rate >= 0.001:
            learning_rate *= 0.98
        obj_cur = objective_function(g, emb, neg_nodes)
        epoch_num.append(epoch + 1)
        obj_f.append(obj_cur)
        print(epoch + 1, obj_cur, time.time() - t0)
    
        with open("emb.{}".format(epoch), "w+") as f:
            for n in emb.keys():
                f.write("\"{0}\": {1}\n".format(n, emb[n]))
    
        plt.plot(epoch_num, obj_f, label="objective function")
        plt.xlabel("epoch_num")
        plt.ylabel("objective function")
        plt.title("objective function")
        plt.legend()
        plt.show()
    
        print("================ epoch {} ================".format(epoch + 1), time.time() - t0)
