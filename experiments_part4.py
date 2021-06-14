import argparse
import math
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from involution import draw, new_effort, update_effort, g, f, I, build_graph

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=10, help='number of nodes in the network')
parser.add_argument('--e', type=int, default=20, help='number of edges in the network')
parser.add_argument('--s', type=int, default=0.8, help='swap rate of graph building')

args = parser.parse_args()

def build_graph_ramdomly(args):
    graph = [[False] * args.n for _ in range(args.n)]
    cur_edge = 0
    while cur_edge != args.e:
        edge_num = np.random.choice(args.n * (args.n - 1) // 2)  # 随机生成不重复的边
        node1 = edge_num // args.n
        node2 = edge_num % args.n
        if node1 == node2 or graph[node1][node2] is True:
            continue
        else:
            cur_edge += 1
            graph[node1][node2] = graph[node2][node1] = True
    return graph

# 探究平均卷度与图连通性（边的数量）的关系
def connectivity_involution():
    G = nx.Graph()  # 创建空的简单无向图
    G.add_nodes_from([(i, {'effort': 1 + round(np.random.random(), 2), 'max_inv': np.random.random() * 5 + 5}) for i in
                      range(args.n)])
    node_max_inv = nx.get_node_attributes(G, 'max_inv')
    init_node_info = {
        'effort': dict(nx.get_node_attributes(G, 'effort')),
        'max_inv': dict(nx.get_node_attributes(G, 'max_inv'))
    }

    for num_edges in range(20, 50, 5):
        args.e = num_edges

        # 节点初始化，删除边关系
        nodes = list(G.nodes())
        G.remove_edges_from(G.edges())
        G.remove_nodes_from(nodes)
        G.add_nodes_from([(i, {'effort': init_node_info['effort'][i], 'max_inv': init_node_info['max_inv'][i]}) for i in
                      range(args.n)])

        # 友谊悖论式建图
        graph = build_graph(args)
        weighted_graph = np.multiply(graph, np.random.random([args.n, args.n]))
        weighted_graph = (weighted_graph + weighted_graph.T)  # 确保邻接矩阵是对称的，并且值域 [0, 2]，邻居里面也有熟悉和不熟悉的
        G.add_weighted_edges_from([(i, j, weighted_graph[i][j]) for i in range(args.n) for j in range(i) if graph[i][j]])

        print(graph)
        rnd = 0
        while True:
            # input()
            print('第 %d 次迭代' % rnd)
            print(nx.get_node_attributes(G, 'effort'))
            rnd += 1
            flag = None
            for node in G.nodes:
                new_effort(node, G.nodes[node], G)
                effort = G.nodes[node]['effort']
                # 邻居的影响应该是加权平均的
                neighbor_efforts_avg = float(np.mean(
                    [G.nodes[neighbor]['effort'] * weighted_graph[node][neighbor] for neighbor in range(args.n) if
                     graph[node][neighbor]]))
                # 如果提升自己的努力程度E可以升高实际效用I，则参与者会选择增加E
                if I(args, G.nodes[node]['new_effort'], neighbor_efforts_avg) <= I(args, effort, neighbor_efforts_avg):
                    G.nodes[node]['new_effort'] = G.nodes[node]['effort']
                else:
                    flag = True
            update_effort(G)
            # draw(G)
            # 当且仅当所有人无法通过提升自己的卷度E来提高实际效用I时，达到平衡
            if not flag:
                break
        draw(G, f'e={num_edges} results')
        average = np.mean([nx.get_node_attributes(G, 'effort')[i] for i in range(args.n)])
        for node in G.nodes:
            effort = G.nodes[node]['effort']
            # 邻居的影响应该是加权平均的
            neighbor_efforts_avg = float(np.mean(
                [G.nodes[neighbor]['effort'] * weighted_graph[node][neighbor] for neighbor in range(args.n) if
                 graph[node][neighbor]]))
            print('第 %d 个人努力程度为 %f\t邻居努力程度为 %f\t实际效用为 %f\t增加努力后的实际效用为 %f' % (
                node, effort, neighbor_efforts_avg if neighbor_efforts_avg != None else 0,
                I(args, effort, neighbor_efforts_avg), I(args, G.nodes[node]['new_effort'], neighbor_efforts_avg)))
        print(f'average_involution = {average}')

if __name__ == '__main__':
    connectivity_involution()