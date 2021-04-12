import numpy as np
import copy
from generate_graph import generate
from margin import embedGraph, isGraphConnected, GraphCollection
from algorithm import string2matrix
from ExpansionGraph import ExpansionGraph
from utils import plotGraph, readGraphs, read_aligned_info
import argparse
import time

def ensureConnected(graph):
    index_to_remove = []
    list_of_subgraph = []
    len_graph = len(graph)
    total_visit = [False]*len_graph

    start_idx = 0
    while True:
        visited = [False]*len_graph
        queue = []
        queue.append(start_idx)

        while queue:
            s = queue.pop(0)
            visited[s] = True
            total_visit[s] = True

            edge_list = np.where(graph[s]>0)[0].tolist()
            # Sort edge
            node_list = [graph[x, x] for x in edge_list]

            edge_list = list(sorted(zip(edge_list, node_list), key=lambda x: x[1], reverse=True))
            # print(edge_list)

            for i, _ in edge_list:
                if not visited[i]:
                    if i not in queue:
                        queue.append(i)

        if sum(visited) >= 0.5*len_graph:
            index_to_remove = [i for i, x in enumerate(visited) if x == False]
            break
        else:
            list_of_subgraph.append(visited)
            if sum(total_visit) == len_graph:
                list_num_node = [sum(x) for x in list_of_subgraph]
                max_num_node = max(list_num_node)
                max_idx = list_num_node.index(max_num_node)
                index_to_remove = [i for i, x in enumerate(list_of_subgraph[max_idx]) if x == False]
                break
            else:
                start_idx = total_visit.index(False)

    # for k in range(graph.shape[0]):
    #     if np.sum(graph[k] > 0) <= 1:
    #         index_to_remove.append(k)

    index_to_remove = sorted(index_to_remove, reverse=True)

    for k in index_to_remove:
        # Orphan node =>> need remove
        graph = np.delete(graph, k, axis=0)
        graph = np.delete(graph, k, axis=1)

    return graph, index_to_remove

def getFrequentEdges(graphs, theta, sg_link_visited, missing_chain):
    frequentEdges = {}
    for idGraph, graph in enumerate(graphs):
        edgesSet = set()
        for i in range(graph.shape[0]):
            indicesEdge = np.where(graph[i,i+1:] > 0)
            for des in indicesEdge[0]:
                labelNodes = [graph[i,i], graph[i+des+1,i+des+1]]
                labelNodes = sorted(labelNodes)#,reverse=True)
                encodeEdges = (labelNodes[0],labelNodes[1],graph[i,i+des+1])

                if i in sg_link_visited[idGraph][0] and i+des+1 in sg_link_visited[idGraph][0] and encodeEdges not in missing_chain:
                    continue

                if encodeEdges not in edgesSet:
                    if encodeEdges not in frequentEdges:
                        frequentEdges[encodeEdges] = {}
                        frequentEdges[encodeEdges]['freq'] = 1
                        frequentEdges[encodeEdges]['edges'] = {}
                    else:
                        frequentEdges[encodeEdges]['freq'] += 1

                    edgesSet.add(encodeEdges)
                    frequentEdges[encodeEdges]['edges'][idGraph] = [(i,i+des+1) if graph[i,i] == labelNodes[0] else (des + i + 1,i)]
                else:
                    frequentEdges[encodeEdges]['edges'][idGraph].append((i,i + des + 1) if graph[i,i] == labelNodes[0] else (des + i + 1,i))

    # print(frequentEdges)
    frequents = {}
    for k,v in frequentEdges.items():
        if v['freq'] >= theta:
            frequents[k] = v['edges']
    return frequents

def hasNoExternalAssEdge(graphs, tree, embeddings):
    numEmb = 0
    for k in embeddings.keys():
        numEmb += len(embeddings[k])

    for i in range(tree.shape[0]):
        externalEdges = {}

        for idGraph in embeddings.keys():
            for subGraph in embeddings[idGraph]:
                curNode = subGraph[i]
                indices = np.where(graphs[idGraph][curNode] > 0)[0] # neighbor of curNode
                nodes = list(set(indices) - set(subGraph)) # node not in subgraph of node curNode
                edges = set()

                for node in nodes:
                    if node != curNode:
                        edges.add((graphs[idGraph][curNode,curNode],graphs[idGraph][node,node],graphs[idGraph][curNode,node]))
                for edge in edges:
                    try:
                        externalEdges[edge]  += 1
                    except:
                        externalEdges[edge]  = 1
        for k,v in externalEdges.items():
            if v == numEmb:
                print("Has External Associative Edge",tree)
                return False

    print("Has NO External Associative Edge")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='Graph dataset', type=str, default="mico.outx")
    parser.add_argument('--aligned', help='Aligned info', type=str, default="aligned_info.txt")
    parser.add_argument('--theta', help='Theta', type=float, default=0.5)
    args = parser.parse_args()
    # print("GENERATING GRAPHS...")
    # NUM_GRAPH = 100
    # THETA = 1.0
    # NUMBER_FOR_COMMON = THETA * NUM_GRAPH
    # graph_db, sg_link = generate(num_of_graphs=NUM_GRAPH, min_node=80, max_node=100,
    #                             subgraph_size=70, node_degree=35,
    #                             random_node=True, random_edge=True, plotSG=False)
    print("LOADING GRAPHS...")
    graph_db = readGraphs(args.graph)
    sg_link = read_aligned_info(args.aligned, spliter=" ")

    NUM_GRAPH = graph_db.shape[0]
    THETA = args.theta
    NUMBER_FOR_COMMON = THETA * NUM_GRAPH

    print("Graphs Dataset: ", graph_db.shape)
    print("Aligned Info: ", sg_link.shape)
    time_start = time.time()

    print("COPYING SUBGRAPH...")
    subgraph_db = []
    total_nodes = []
    total_edges = []

    error_occur = False
    for i, sg in enumerate(sg_link[:, 0]):
        length = len(sg)
        subgraph = np.zeros((length,length), dtype=np.int)

        try:
            for k, node_0 in enumerate(sg):
                for l, node_1 in enumerate(sg):
                    subgraph[k][l] = graph_db[i][node_0][node_1]

            total_nodes.append(length)
            total_edges.append((np.sum(subgraph > 0) - length) / 2)

            subgraph_db.append(subgraph)
        except:
            print("Error occurs in graph %d" % i)
            error_occur = True

    if error_occur:
        exit(0)
    # for x in subgraph_db:
    #     print(x)
    #     plotGraph(x, False)

    print("CHECKING SUBGRAPH...")
    # THETE cao thi dung min, thap thi dung max
    max_node = max(total_nodes)
    list_max_node = np.where(np.array(total_nodes) == max_node)[0]
    # print(list_max_node)

    max_edge = max(total_edges)
    list_max_edge = np.where(np.array(total_edges) == max_edge)[0]
    # print(list_max_edge)

    list_candidate_sg = np.intersect1d(list_max_node, list_max_edge)
    # print(list_candidate_sg)
    if len(list_candidate_sg) == 0:
        list_candidate_sg = list_max_node

    candidate_sg = copy.deepcopy(subgraph_db[list_candidate_sg[0]])
    candidate_index = list_candidate_sg[0]
    candidate_length = candidate_sg.shape[0]
    missing_chain = []

    for i in range(candidate_length):
        for k in range(i+1, candidate_length):
            potential_missing = False
            # Check edge
            edge_val = candidate_sg[i][k]
            edge_val_list = {}
            edge_val_list[edge_val] = 1

            vertex_0 = i
            vertex_1 = k
            for cur_idx in range(candidate_index, 0, -1):
                node_link_0 = sg_link[cur_idx][0][vertex_0]
                node_link_1 = sg_link[cur_idx][0][vertex_1]

                anchor_idx_0 = np.where(np.array(sg_link[cur_idx - 1][1]) == node_link_0)[0]
                anchor_idx_1 = np.where(np.array(sg_link[cur_idx - 1][1]) == node_link_1)[0]

                if anchor_idx_0.size > 0 and anchor_idx_1.size > 0:
                    e_val = subgraph_db[cur_idx-1][anchor_idx_0[0]][anchor_idx_1[0]]
                    if e_val not in edge_val_list.keys():
                        edge_val_list[e_val] = 1
                    else:
                        edge_val_list[e_val] += 1

                    vertex_0 = anchor_idx_0[0]
                    vertex_1 = anchor_idx_1[0]
                else:
                    potential_missing = True
                    break

            vertex_0 = i
            vertex_1 = k
            for cur_idx in range(candidate_index, NUM_GRAPH-1):
                node_link_0 = sg_link[cur_idx][1][vertex_0]
                node_link_1 = sg_link[cur_idx][1][vertex_1]

                anchor_idx_0 = np.where(np.array(sg_link[cur_idx + 1][0]) == node_link_0)[0]
                anchor_idx_1 = np.where(np.array(sg_link[cur_idx + 1][0]) == node_link_1)[0]

                if anchor_idx_0.size > 0 and anchor_idx_1.size > 0:
                    e_val = subgraph_db[cur_idx+1][anchor_idx_0[0]][anchor_idx_1[0]]
                    if e_val not in edge_val_list.keys():
                        edge_val_list[e_val] = 1
                    else:
                        edge_val_list[e_val] += 1

                    vertex_0 = anchor_idx_0[0]
                    vertex_1 = anchor_idx_1[0]
                else:
                    potential_missing = True
                    break

            # If count_freq < theta ===> delete edge
            count_freq = max(edge_val_list.values())
            e_val = max(edge_val_list, key=edge_val_list.get)

            if count_freq < NUMBER_FOR_COMMON:
                if potential_missing and e_val != 0:
                    # TODO; MISSING chain to index
                    labelNodes = [candidate_sg[i][i], candidate_sg[k][k]]
                    labelNodes = sorted(labelNodes)#,reverse=True)
                    encodeEdges = (labelNodes[0],labelNodes[1],candidate_sg[i][k])

                    missing_chain.append(encodeEdges)

                candidate_sg[i][k] = 0
                candidate_sg[k][i] = 0
            else:
                candidate_sg[i][k] = e_val
                candidate_sg[k][i] = e_val

    # Check missing chain
    print("FOUND MISSING CHAIN...")
    print(missing_chain)

    # Ensure graph is isGraphConnected
    candidate_sg, removed_idx = ensureConnected(candidate_sg)

    checked_node = copy.deepcopy(sg_link[candidate_index][0])

    # No need to remove isolated node in visited list
    absolute_mapping = copy.deepcopy(sg_link[candidate_index][0])
    for idx in removed_idx:
        absolute_mapping = np.delete(absolute_mapping, idx)

    # Filter visited nodes in all graphs
    sg_link_visited = {}
    sg_link_visited[candidate_index] = [np.array(checked_node)]
    for cur_idx in range(candidate_index, 0, -1):
        refer_node_list = sg_link_visited[cur_idx][0]
        alignment = sg_link[cur_idx-1]
        align_node_list = []

        for node_i in refer_node_list:
            node_align_i = np.where(np.array(alignment[1])  == node_i)[0]
            if node_align_i.size > 0:
                align_node_list.append(alignment[0][node_align_i[0]])

        sg_link_visited[cur_idx-1] = [np.array(align_node_list)]

    for cur_idx in range(candidate_index, NUM_GRAPH-1):
        refer_node_list = sg_link_visited[cur_idx][0]
        alignment = sg_link[cur_idx]
        align_node_list = []

        for node_i in refer_node_list:
            node_align_i = np.where(np.array(alignment[0])  == node_i)[0]
            if node_align_i.size > 0:
                align_node_list.append(alignment[1][node_align_i[0]])

        sg_link_visited[cur_idx+1] = [np.array(align_node_list)]

    # print(sg_link)
    # print(sg_link_visited)

    # Recheck external edge
    # Get list of frequent edges connected with candidate_sg
    # print(frequentEdgeSet)
    # print(candidate_sg)
    print("CHECK EXTERNAL ASSOCIATIVE EDGE...")
    frequentEdgeSet = getFrequentEdges(graph_db, NUMBER_FOR_COMMON, sg_link_visited, missing_chain)
    # print(frequentEdgeSet)
    
    if hasNoExternalAssEdge(graph_db, candidate_sg, sg_link_visited):
        padding_candidate_sg = candidate_sg.copy()
        for key, value in frequentEdgeSet.items():
            link_0 = 0
            link_1 = 0

            # print(value[candidate_index][0][0])
            # print(sg_link_visited[candidate_index][0])

            for v_i in range(len(value[candidate_index])):
                if value[candidate_index][v_i][0] not in absolute_mapping and \
                    value[candidate_index][v_i][0] in sg_link_visited[candidate_index][0]:
                    padding_candidate_sg = np.pad(padding_candidate_sg, [(0,1),(0,1)])
                    padding_candidate_sg[-1][-1] = key[0]
                    link_0 = padding_candidate_sg.shape[0] - 1
                    absolute_mapping = np.append(absolute_mapping, value[candidate_index][v_i][0])

                    # TODO: UPDATE sg_link_visited

                else:
                    link_0 = np.where(np.array(absolute_mapping) == value[candidate_index][v_i][0])[0][0]

                if value[candidate_index][v_i][1] not in absolute_mapping and \
                    value[candidate_index][v_i][1] in sg_link_visited[candidate_index][0]:
                    padding_candidate_sg = np.pad(padding_candidate_sg, [(0,1),(0,1)])
                    padding_candidate_sg[-1][-1] = key[1]
                    link_1 = padding_candidate_sg.shape[0] - 1
                    absolute_mapping = np.append(absolute_mapping, value[candidate_index][v_i][1])
                    # TODO: UPDATE sg_link_visited

                else:
                    link_1 = np.where(np.array(absolute_mapping) == value[candidate_index][v_i][1])[0][0]

                padding_candidate_sg[link_0][link_1] = key[2]
                padding_candidate_sg[link_1][link_0] = key[2]
            # print(absolute_mapping)
            # print(padding_candidate_sg)

        eg = ExpansionGraph(
            padding_candidate_sg,
            sg_link_visited,
            graph_db,
            frequentEdgeSet,
            NUMBER_FOR_COMMON
        )
        expansionX = eg.expand()

        # print("Expansion X",expansionX)
        if len(expansionX.keys()) > 0:
            list_sg = [string2matrix(k) for k,v in expansionX.items()]
            # print(list_sg)
            list_num_edge = [(np.sum(x>0) - x.shape[0])//2 for x in list_sg]
            max_sg_idx = np.argmax(list_num_edge)
            # TODO: HANDLE ENSURE CONNECTED
            candidate_sg = list_sg[max_sg_idx]

    time_end = time.time()

    print("FINAL RESULT:")
    print(candidate_sg)
    print("Subgraph size: ", candidate_sg.shape[1])
    print("Time: %.5f" % (time_end-time_start))
    if candidate_sg.size > 0:
        plotGraph(candidate_sg, False)
