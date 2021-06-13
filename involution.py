# -*- coding:utf-8 -*-
# 1. 给定人数和边数，随机构造社交网络
# 2. 内卷中的投入函数和收益函数
# 所有浮点数保留小数点后两位

import argparse
import math
import networkx as nx
from networkx.generators.random_graphs import fast_gnp_random_graph
import numpy as np
import sympy
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=10, help='number of nodes in the network')
parser.add_argument('--e', type=int, default=30, help='number of edges in the network')
args = parser.parse_args()


def draw(G, title=None):
    plt.figure(figsize=[8, 8])
    node_labels = nx.get_node_attributes(G, 'effort')
    node_max_inv = nx.get_node_attributes(G, 'max_inv')
    colors = []
    for eachnode in node_labels:
        if node_labels[eachnode] / node_max_inv[eachnode] >= 1:
            colors.append((1, 1, 0))
        else:
            colors.append((node_labels[eachnode] / node_max_inv[eachnode], 1 - node_labels[eachnode] / node_max_inv[eachnode], 0))
    for k in node_labels:
        node_labels[k] = f'{k}:{round(node_labels[k], 2)}'
    pos = nx.circular_layout(G)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    # nx.draw_networkx_nodes(G,pos, node_color=colors,cmap=plt.cm.Blues) # 越“卷”的节点颜色越深
    nx.draw_networkx_nodes(G, pos, node_color=colors)  # 越“卷”的节点颜色越红
    nx.draw_networkx_edges(G, pos)
    if title:
        plt.title(title)
    plt.show()


def new_effort(nodeid, node, G):
    # 投入函数，计算并修改这一节点经历这一轮迭代之后的effort
    old_effort = node['effort']
    node['new_effort'] = old_effort + round(marginal_effort(old_effort), 2)
    if node['new_effort'] > node['max_inv']:
        node['new_effort'] = node['max_inv']


def marginal_benefit(eft):
    return -(eft + 0.2) * (eft - 1)


def marginal_effort(eft):
    return math.log(eft)


def update_effort(G):
    # 更新effort
    for nid in G.nodes:
        G.nodes[nid]['effort'] = G.nodes[nid]['new_effort']
    return


def g(E):
    # 每个节点的心理成本，关于E的导数单调递增
    return E ** 2


def f(E, other_eft_avg):
    # 与系统中其他人付出相关的回报，关于other_effort偏导为负数，关于E导数单调递减
    return args.n * E - other_eft_avg * (args.n - 1)


def I(E, other_effort):
    # 收益函数，计算当前effort下每个节点的收益，并更新节点收益
    ret = f(E, other_effort) - g(E)
    # 这里使用sigmoids函数，系数0.1是防止sigmoid在ret绝对值很大的时候返回0
    # 如果一个人像qipeng一样菜，那么努力后得到的收益会很低，因为还是那么菜
    # 如果一个人是avg水准的，那么努力一点收益应该是比较明显的
    # 但如果是teacher Song一样的大佬，再怎么努力也只能让别人salute
    return 1 / (1 + math.exp(-ret / 10))


def build_graph(args):
    # 采用第一次作业友谊悖论的方式构建社交网络
    graph = [[False] * args.n for _ in range(args.n)]
    for d in range(1, math.ceil(args.e / args.n)):
        for i in range(args.n):
            j = (i + d) % args.n
            graph[i][j] = graph[j][i] = True
    relations = [i * args.n + j for i in range(args.n) for j in range(args.n) if j < i and graph[i][j]]
    others = [i * args.n + j for i in range(args.n) for j in range(args.n) if j < i and not graph[i][j]]
    swap_size = int(0.5 * len(relations))
    removed_relation = np.random.choice(relations, size=swap_size, replace=False)
    new_relation = np.random.choice(others, size=swap_size, replace=False)
    for id in removed_relation:
        graph[id // args.n][id % args.n] = graph[id % args.n][id // args.n] = False
    for id in new_relation:
        graph[id // args.n][id % args.n] = graph[id % args.n][id // args.n] = True

    return graph


def part2():
    G = nx.Graph()  # 创建空的简单无向图
    G.add_nodes_from([(i, {'effort': round(1 + np.random.random(), 2), 'max_inv': np.random.random() * 5 + 5}) for i in range(args.n)])
    node_max_inv = nx.get_node_attributes(G, 'max_inv')

    graph = build_graph(args)
    weighted_graph = np.multiply(graph, np.random.random([args.n, args.n]))
    weighted_graph = (weighted_graph + weighted_graph.T)  # 确保邻接矩阵是对称的，并且值域 [0, 2]，邻居里面也有熟悉和不熟悉的
    G.add_weighted_edges_from([(i, j, weighted_graph[i][j]) for i in range(args.n) for j in range(i) if graph[i][j]])

    print(graph)
    rnd = 0
    while True:
        print('第 %d 次迭代' % rnd)
        rnd += 1
        flag = None
        for node in G.nodes:
            new_effort(node, G.nodes[node], G)
            effort = G.nodes[node]['effort']
            # 邻居的影响应该是加权平均的
            neighbor_efforts_avg = float(np.mean([G.nodes[neighbor]['effort'] * weighted_graph[node][neighbor] for neighbor in range(args.n) if graph[node][neighbor]]))
            print('第 %d 个人的实际效用为 %f\t增加努力后的实际效用为 %f' % (node, I(effort, neighbor_efforts_avg), I(G.nodes[node]['new_effort'], neighbor_efforts_avg)))
            # 如果提升自己的努力程度E可以升高实际效用I，则参与者会选择增加E
            if I(G.nodes[node]['new_effort'], neighbor_efforts_avg) <= I(effort, neighbor_efforts_avg):
                G.nodes[node]['new_effort'] = G.nodes[node]['effort']
            else:
                flag = True
        update_effort(G)
        draw(G)
        # 当且仅当所有人无法通过提升自己的卷度E来提高实际效用I时，达到平衡
        if not flag:
            break
    draw(G, 'finish')


def part1():
    G = nx.Graph()  # 创建空的简单无向图
    for i in range(1, args.n + 1):
        G.add_nodes_from([(i, {'effort': round(1 + np.random.random(), 2), 'max_inv': np.random.random() * 5 + 5})])

    edges = np.random.choice(args.n * (args.n - 1) // 2, args.e, replace=False)  # 随机生成不重复的边
    for eachedge in edges:
        node1 = eachedge // args.n + 1
        node2 = eachedge % args.n + 1
        if node1 == node2:
            continue
        # 第三个参数是权值，表示两个人的熟悉程度，是一个0～1之间的浮点数
        G.add_weighted_edges_from([(node1, node2, round(np.random.random(), 2))])

    num = 0

    while True:
        print('第 %d 次迭代' % (num))
        num = num + 1
        flag = 0

        effort_list = []  # 每个节点的卷度
        node_labels = nx.get_node_attributes(G, 'effort')
        print(node_labels)
        for i in range(1, args.n + 1):  # effort_list为每个节点的卷度构成的list
            effort_list.append(node_labels[i])
        sum_of_effort = sum(effort_list)
        print(effort_list, sum_of_effort, type(G.nodes), type(G.nodes[1]))
        for node in G.nodes:
            new_effort(node, G.nodes[node], G)
            effort = G.nodes[node]['effort']
            other_effort_avg = (sum_of_effort - effort) / (args.n - 1)
            print(type(effort), type(other_effort_avg))
            print('debug: effort = %f, other_effort = %f' % (effort, other_effort_avg))
            print('第 %d 个人的实际效用为 %f\t增加努力后的实际效用为 %f' % (node, I(effort, other_effort_avg), I(G.nodes[node]['new_effort'], other_effort_avg)))
            # 如果提升自己的努力程度E可以升高实际效用I，则参与者会选择增加E
            if I(G.nodes[node]['new_effort'], other_effort_avg) <= I(effort, other_effort_avg):
                G.nodes[node]['new_effort'] = G.nodes[node]['effort']
            else:
                # print('not converge')
                flag = 1
        update_effort(G)
        draw(G)
        # 当且仅当所有人无法通过提升自己的卷度E来提高实际效用I时，达到平衡
        if flag == 0:
            break
    draw(G, 'finish')


if __name__ == '__main__':
    part2()
