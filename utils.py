import numpy as np
import os
# from graph import Graph
import networkx as nx
import matplotlib.pyplot as plt
import re
import copy

def read_aligned_info(path):
    pattern_label = re.compile("^G[0-9]+[ \t]*G[0-9]+$")
    pattern_data = re.compile("^[0-9]+[ \t]*[0-9]+$")

    with open(path, 'r', encoding='utf-8') as file:
        align_read = file.read()
        align_read = align_read.split('\n')

    label_idx = [i for i, x in enumerate(align_read) if re.match(pattern_label, x)]
    aligned = [[] for _ in range(len(label_idx)+1)]

    for each_idx in label_idx:
        graph_pair = align_read[each_idx].split('\t')
        graph_pair = [int(x[1:]) for x in graph_pair]
        align_map = [[],[]]
        dyn_i = each_idx + 1

        while (re.match(pattern_data, align_read[dyn_i])):
            mapping = align_read[dyn_i].split('\t')
            align_map[0].append(int(mapping[0]))
            align_map[1].append(int(mapping[1]))
            dyn_i += 1

        aligned[graph_pair[0]] = copy.deepcopy(align_map)

    align_map = [aligned[-2][1].copy(), aligned[-2][1].copy()]

    for link_map in aligned[-2::-1]:
        temp_map = []
        del_idx = []

        for i, x in enumerate(align_map[1]):
            if x in link_map[1]:
                temp_map.append(link_map[0][link_map[1].index(x)])
            else:
                del_idx.append(i)

        if del_idx:
            for idx in sorted(del_idx, reverse=True):
                del align_map[0][idx]

        align_map[1] = temp_map

    aligned[-1] = align_map

    return np.array(aligned)

def read_graph_corpus(path, label_center_path=None):
    graphs = []
    # label_center = open(label_center_path, 'r', encoding='utf-8')
    label_centers = []
    with open(path, 'r', encoding='utf-8') as file:
        nodes = {}
        edges = {}
        for line in file:
            if 't' in line:
                if len(nodes) > 0:
                    graphs.append((nodes, edges))
                    # if len(graphs) > 9:
                        # break
                nodes = {}
                edges = {}
            if 'v' in line:
                data_line = line.split()
                node_id = int(data_line[1])
                node_label = int(data_line[2])
                nodes[node_id] = node_label
            if 'e' in line:
                data_line = line.split()
                source_id = int(data_line[1])
                target_id = int(data_line[2])
                label = int(data_line[3])
                edges[(source_id, target_id)] = label
        if len(nodes) > 0:
            graphs.append((nodes,edges))
    return graphs#[10:]

def readGraphs(path):
    rawGraphs = read_graph_corpus(path)
    graphs = []
    for graph in rawGraphs:
        numVertices = len(graph[0])
        g = np.zeros((numVertices,numVertices),dtype=int)
        for v,l in graph[0].items():
            g[v,v] = l
        for e,l in graph[1].items():
            g[e[0],e[1]] = l
            g[e[1],e[0]] = l
        graphs.append(g)#[:10,:10])
    return np.array(graphs)

def plotGraph(graph : np.ndarray,isShowedID=True):
    edges = []
    edgeLabels = {}
    for i in range(graph.shape[0]):
        indices = np.where(graph[i][i+1:] > 0)[0]
        for id in indices:
            edges.append([i,i+id+1])
            edgeLabels[(i,i+id+1)] = graph[i,i+id+1]
    # print(edges,edgeLabels)
    # exit(0)
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    plt.figure()
    nodeLabels = {node:node for node in G.nodes()} if isShowedID else {node:graph[node,node] for node in G.nodes()}
    nx.draw(G,pos,edge_color='black',width=1,linewidths=1,
        node_size=500,node_color='pink',alpha=0.9,
        labels=nodeLabels)

    nx.draw_networkx_edge_labels(G,pos,edge_labels=edgeLabels,font_color='red')
    plt.axis('off')
    # plt.savefig('./figures/{}.png'.format(np.array2string(graph[0])),format='PNG')
    plt.show()
