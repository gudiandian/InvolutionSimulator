# -*- coding:utf-8 -*-
# 1. 给定人数和边数，随机构造社交网络
# 2. 内卷中的投入函数和收益函数
# 所有浮点数保留小数点后两位

import argparse
import networkx as nx
import numpy as np
import sympy
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--k', type=float, default=0.1, 
	help='k为一个大于0的可调的参数，k越大，节点越容易在别人比自己努力的时候受到别人影响，从而加剧内卷程度')
parser.add_argument('--n', type=int, default=10, help='number of nodes in the network')
parser.add_argument('--e', type=int, default=30, help='number of edges in the network')


def draw(G):
	node_labels = nx.get_node_attributes(G, 'effort')
	pos = nx.spring_layout(G)
	nx.draw_networkx_labels(G, pos, labels=node_labels)
	colors = []
	for eachnode in node_labels:
		if node_labels[eachnode] > 1:
			colors.append((1, 1, 0))
		else:
			colors.append((node_labels[eachnode], 1 - node_labels[eachnode], 0))
	#nx.draw_networkx_nodes(G,pos, node_color=colors,cmap=plt.cm.Blues) # 越“卷”的节点颜色越深
	nx.draw_networkx_nodes(G,pos, node_color=colors) # 越“卷”的节点颜色越红
	nx.draw_networkx_edges(G, pos)
	plt.show()


def new_effort(nodeid, node, G):
	"""
	投入函数，计算并修改这一节点经历这一轮迭代之后的effort
	"""
	old_effort = node['effort']
	neighbor_list = G.neighbors(nodeid)
	influences = []
	# 如果一个人的neighbor没有ta卷，则这个人不会变得更“卷”，但是要保证自己比他的邻居更“卷”
	# 如果一个人有neighbor比他卷，则他会变得更“卷”，越熟悉的人对他的影响更大，这个人有可能变得比邻居更“卷”，也可能不会
	# 算出来一个加权平均的影响
	for each in neighbor_list:
		if G.nodes[each]['effort'] == old_effort:
			continue # 无影响
		elif G.nodes[each]['effort'] < old_effort:
			influence = (G.nodes[each]['effort'] - old_effort) * G[nodeid][each]['weight']
			influences.append(influence)
		else:
			influence = (G.nodes[each]['effort'] - old_effort) * (G[nodeid][each]['weight'] + args.k)
			influences.append(influence)
	if len(influences) == 0:
		node['new_effort'] = node['effort']
	else:
		#print(sum(influences) / len(influences))
		node['new_effort'] = round(node['effort'] + sum(influences) / len(influences), 2)


def marginal_benefit(effort):
	return -(x + 0.2) * (x - 1)


def benefit(G):
	"""
	收益函数，计算当前effort下每个节点的收益，并更新节点收益
	"""
	knowledge_increasement_dict = {}
	for eachnode in G.nodes:
		effort = G.nodes[eachnode]['effort']
		x = sympy.symbols('x')
		knowledge_increasement = sympy.integrate(x, (x, 0, effort))
		G.nodes[eachnode]['knowledge_increasement'] = knowledge_increasement
		print(knowledge_increasement)
		knowledge_increasement_dict[eachnode] = knowledge_increasement
	knowledge_increasement_dict = sorted(knowledge_increasement_dict.items(), key=lambda d:d[1], reverse=False)
	# 根据卷的程度排名，最终的benefit是相对的，即：knowledge_increasement排名越靠前，benifit越大
	for i in range(len(knowledge_increasement_dict)):
		G.nodes[knowledge_increasement_dict[i][0]]['benefit'] = i



def update_effort(G):
	"""
	更新effort
	"""
	for eachnode in G.nodes:
		G.nodes[eachnode]['effort'] = G.nodes[eachnode]['new_effort']
	return


if __name__ == '__main__':
	args = parser.parse_args()

	G=nx.Graph() # 创建空的简单无向图
	for i in range(1, args.n + 1):
		G.add_nodes_from([(i, {'effort': round(np.random.random((1, 1))[0][0], 2)})])
	
	edges = np.random.choice(args.n * (args.n - 1) // 2, args.e, replace=False) #随机生成不重复的边
	for eachedge in edges:
		node1 = eachedge // args.n + 1
		node2 = eachedge % args.n + 1
		if node1 == node2:
			continue
		# 第三个参数是权值，表示两个人的熟悉程度
		G.add_weighted_edges_from([(node1, node2, round(np.random.random((1, 1))[0][0], 2))])

	draw(G)
	# 一轮迭代
	for eachnode in G.nodes:
		new_effort(eachnode, G.nodes[eachnode], G)
	update_effort(G)
	benefit(G)

	draw(G)
