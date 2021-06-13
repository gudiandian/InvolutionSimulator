# -*- coding:utf-8 -*-
# 1. 给定人数和边数，随机构造社交网络
# 2. 内卷中的投入函数和收益函数
# 所有浮点数保留小数点后两位

import argparse
import math
import networkx as nx
import numpy as np
import sympy
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=10, help='number of nodes in the network')
parser.add_argument('--e', type=int, default=30, help='number of edges in the network')



def draw(G):
	node_labels = nx.get_node_attributes(G, 'effort')
	pos = nx.spring_layout(G)
	nx.draw_networkx_labels(G, pos, labels=node_labels)
	colors = []
	for eachnode in node_labels:
		if node_labels[eachnode] / 5 > 1:
			colors.append((1, 1, 0))
		else:
			colors.append((node_labels[eachnode] / 5, 1 - node_labels[eachnode] / 5, 0))
	#nx.draw_networkx_nodes(G,pos, node_color=colors,cmap=plt.cm.Blues) # 越“卷”的节点颜色越深
	nx.draw_networkx_nodes(G,pos, node_color=colors) # 越“卷”的节点颜色越红
	nx.draw_networkx_edges(G, pos)
	plt.show()


def new_effort(nodeid, node, G):
	"""
	投入函数，计算并修改这一节点经历这一轮迭代之后的effort
	"""
	old_effort = node['effort']
	node['new_effort'] = old_effort + round(marginal_effort(old_effort), 2)
	if node['new_effort'] > 5:
		node['new_effort'] = 5


def marginal_benefit(effort):
	return -(x + 0.2) * (x - 1)


def marginal_effort(effort):
	return math.log(effort)


def update_effort(G):
	"""
	更新effort
	"""
	for eachnode in G.nodes:
		G.nodes[eachnode]['effort'] = G.nodes[eachnode]['new_effort']
	return


def g(E):
	"""
	每个节点的心理成本，关于E的导数单调递增
	"""
    return E**2


def f(E, other_effort):
	"""
	与系统中其他人付出相关的回报，关于other_effort偏导为负数，关于E导数单调递减
	"""
    return args.n * E - other_effort


def I(E, other_effort):
	"""
	收益函数，计算当前effort下每个节点的收益，并更新节点收益
	"""
    return f(E,other_effort)-g(E)


if __name__ == '__main__':
	args = parser.parse_args()

	G=nx.Graph() # 创建空的简单无向图
	for i in range(1, args.n + 1):
		G.add_nodes_from([(i, {'effort': round(1 + np.random.random((1, 1))[0][0], 2)})])
	
	edges = np.random.choice(args.n * (args.n - 1) // 2, args.e, replace=False) #随机生成不重复的边
	for eachedge in edges:
		node1 = eachedge // args.n + 1
		node2 = eachedge % args.n + 1
		if node1 == node2:
			continue
		# 第三个参数是权值，表示两个人的熟悉程度，是一个0～1之间的浮点数
		G.add_weighted_edges_from([(node1, node2, round(np.random.random((1, 1))[0][0], 2))])

	num = 0
	
	while True:
		print('第 %d 次迭代' % (num))
		num = num + 1
		flag = 0
		index = 0
		effort_list=[]	# 每个节点的卷度
		node_labels = nx.get_node_attributes(G, 'effort')
		for i in range(1, args.n + 1):	# effort_list为每个节点的卷度构成的list
			effort_list.append(node_labels[i])
		sum_of_effort = sum(effort_list)
		for node in G.nodes:
			new_effort(node, G.nodes[node], G)
			print('第 %d 个人的实际效用为 %f' % (index, I(G.nodes[node]['effort'], (sum_of_effort - G.nodes[node]['effort']) / (args.n - 1))))
			print('第 %d 个人增开努力后的实际效用为 %f' % (index, I(G.nodes[node]['new_effort'], (sum_of_effort - G.nodes[node]['effort']) / (args.n - 1))))
			# 如果提升自己的努力程度E可以升高实际效用I，则参与者会选择增加E
			if I(G.nodes[node]['new_effort'], (sum_of_effort - G.nodes[node]['effort']) / (args.n - 1)) <= I(G.nodes[node]['effort'], (sum_of_effort - G.nodes[node]['effort']) / (args.n - 1)): 
				G.nodes[node]['new_effort'] = G.nodes[node]['effort']
			else:
				flag = 1
			index = index + 1
		update_effort(G)
		draw(G)
		# 当且仅当所有人无法通过提升自己的卷度E来提高实际效用I时，达到平衡
		if flag == 0:
			break
