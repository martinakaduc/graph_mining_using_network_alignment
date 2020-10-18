import numpy as np
import copy
from generate_graph import generate
from margin import embedGraph, isGraphConnected, GraphCollection
from algorithm import string2matrix
from ExpansionGraph import ExpansionGraph
from utils import plotGraph

def ensureConnected(graph):
    index_to_remove = []

    for k in range(graph.shape[0]):
        if np.sum(graph[k] > 0) <= 1:
            index_to_remove.append(k)

    index_to_remove.sort(reverse=True)

    for k in index_to_remove:
        # Orphan node =>> need remove
        graph = np.delete(graph, k, axis=0)
        graph = np.delete(graph, k, axis=1)

    return graph, index_to_remove

def getFrequentEdges(graphs, theta, sg_link_visited):
    frequentEdges = {}
    for idGraph, graph in enumerate(graphs):
        edgesSet = set()
        for i in range(graph.shape[0]):
            indicesEdge = np.where(graph[i,i+1:] > 0)
            for des in indicesEdge[0]:
                if i in sg_link_visited[idGraph][0] and i+des+1 in sg_link_visited[idGraph][0]:
                    continue

                labelNodes = [graph[i,i], graph[i+des+1,i+des+1]]
                labelNodes = sorted(labelNodes)#,reverse=True)
                encodeEdges = (labelNodes[0],labelNodes[1],graph[i,i+des+1])
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
    print("GENERATING GRAPHS...")
    NUM_GRAPH = 100
    THETA = 1.0
    NUMBER_FOR_COMMON = THETA * NUM_GRAPH
    graph_db, sg_link = generate(num_of_graphs=NUM_GRAPH, min_node=80, max_node=100,
                                subgraph_size=70, node_degree=35,
                                random_node=True, random_edge=True, plotSG=False)

    print(graph_db)
    print(sg_link)

    print("COPYING SUBGRAPH...")
    subgraph_db = []
    total_nodes = []
    total_edges = []

    for i, sg in enumerate(sg_link[:, 0]):
        length = len(sg)
        subgraph = np.zeros((length,length), dtype=np.int)

        for k, node_0 in enumerate(sg):
            for l, node_1 in enumerate(sg):
                subgraph[k][l] = graph_db[i][node_0][node_1]

        total_nodes.append(length)
        total_edges.append((np.sum(subgraph > 0) - length) / 2)

        subgraph_db.append(subgraph)

    # for x in subgraph_db:
    #     print(x)
    #     plotGraph(x.data, False)

    # print("START MARGIN...")
    # # Ensure all subgraph is explored at the same level
    # min_edge = min([x.min_edge for x in subgraph_db])
    # for i in range(len(subgraph_db)):
    #     subgraph_db[i].generateSpecLattice(min_edge)
    #
    # MARGIN = GraphCollection(subgraph_db, 1.0)
    # result = {"tree": [], "code": []}
    #
    # while len(result["code"]) == 0 and min_edge > 0:
    #     MARGIN.graphs = subgraph_db
    #     result = MARGIN.margin()
    #
    #     for i in range(len(subgraph_db)):
    #         subgraph_db[i].generateNextLattice()
    #
    #     min_edge -= 1

    print("CHECKING SUBGRAPH...")
    # THETE cao thi dung min, thap thi dung max
    min_node = max(total_nodes)
    list_min_node = np.where(np.array(total_nodes) == min_node)[0]

    min_edge = max(total_edges)
    list_min_edge = np.where(np.array(total_edges) == min_edge)[0]

    list_candidate_sg = np.intersect1d(list_min_node, list_min_edge)
    # print(list_candidate_sg)

    candidate_sg = copy.deepcopy(subgraph_db[list_candidate_sg[0]])
    candidate_index = list_candidate_sg[0]

    candidate_length = candidate_sg.shape[0]
    for i in range(candidate_length):
        for k in range(i+1, candidate_length):
            # Check edge
            edge_val = candidate_sg[i][k]
            edge_val_list = {}
            edge_val_list[edge_val] = 1

            vertex_0 = i
            vertex_1 = k
            for cur_idx in range(candidate_index, 0, -1):
                node_link_0 = sg_link[cur_idx][0][vertex_0]
                node_link_1 = sg_link[cur_idx][0][vertex_1]

                anchor_idx_0 = np.where(sg_link[cur_idx - 1][1] == node_link_0)[0]
                anchor_idx_1 = np.where(sg_link[cur_idx - 1][1] == node_link_1)[0]

                if anchor_idx_0.size > 0 and anchor_idx_1.size > 0:
                    e_val = subgraph_db[cur_idx-1][anchor_idx_0[0]][anchor_idx_1[0]]
                    if e_val not in edge_val_list.keys():
                        edge_val_list[e_val] = 1
                    else:
                        edge_val_list[e_val] += 1

                vertex_0 = anchor_idx_0[0]
                vertex_1 = anchor_idx_1[0]

            vertex_0 = i
            vertex_1 = k
            for cur_idx in range(candidate_index, NUM_GRAPH-1):
                node_link_0 = sg_link[cur_idx][1][vertex_0]
                node_link_1 = sg_link[cur_idx][1][vertex_1]

                anchor_idx_0 = np.where(sg_link[cur_idx + 1][0] == node_link_0)[0]
                anchor_idx_1 = np.where(sg_link[cur_idx + 1][0] == node_link_1)[0]

                if anchor_idx_0.size > 0 and anchor_idx_1.size > 0:
                    e_val = subgraph_db[cur_idx+1][anchor_idx_0[0]][anchor_idx_1[0]]
                    if e_val not in edge_val_list.keys():
                        edge_val_list[e_val] = 1
                    else:
                        edge_val_list[e_val] += 1

                vertex_0 = anchor_idx_0[0]
                vertex_1 = anchor_idx_1[0]

            # If count_freq < theta ===> delete edge
            count_freq = max(edge_val_list.values())
            if count_freq < NUMBER_FOR_COMMON:
                candidate_sg[i][k] = 0
                candidate_sg[k][i] = 0
            else:
                e_val = max(edge_val_list, key=edge_val_list.get)
                candidate_sg[i][k] = e_val
                candidate_sg[k][i] = e_val

    # Ensure graph is isGraphConnected
    candidate_sg, removed_idx = ensureConnected(candidate_sg)

    checked_node = copy.deepcopy(sg_link[candidate_index][0])

    for idx in removed_idx:
        del checked_node[idx]

    # Filter visited nodes in all graphs
    sg_link_visited = {}
    sg_link_visited[candidate_index] = [checked_node]
    for cur_idx in range(candidate_index, 0, -1):
        refer_node_list = sg_link_visited[cur_idx][0]
        alignment = sg_link[cur_idx-1]
        align_node_list = []

        for node_i in refer_node_list:
            node_align_i = np.where(alignment[1]  == node_i)[0][0]
            align_node_list.append(alignment[0][node_align_i])

        sg_link_visited[cur_idx-1] = [np.array(align_node_list)]

    for cur_idx in range(candidate_index, NUM_GRAPH-1):
        refer_node_list = sg_link_visited[cur_idx][0]
        alignment = sg_link[cur_idx]
        align_node_list = []

        for node_i in refer_node_list:
            node_align_i = np.where(alignment[0]  == node_i)[0][0]
            align_node_list.append(alignment[1][node_align_i])

        sg_link_visited[cur_idx+1] = [np.array(align_node_list)]

    # print(sg_link)
    # print(sg_link_visited)

    # Recheck external edge
    # Get list of frequent edges connected with candidate_sg
    # print(frequentEdgeSet)
    print("CHECK EXTERNAL ASSOCIATIVE EDGE...")
    if hasNoExternalAssEdge(graph_db, candidate_sg, sg_link_visited):
        frequentEdgeSet = getFrequentEdges(graph_db, NUMBER_FOR_COMMON, sg_link_visited)

        eg = ExpansionGraph(
            candidate_sg.copy(),
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
            candidate_sg = list_sg[max_sg_idx]

    # Truong hop: Graph co label node & canh giong nhau - OK
    # Label node nhiều, không trọng số cạnh - OK
    # directed graph
    # degree tb khi generate - OK
    print("FINAL RESULT:")
    print(candidate_sg)
    print(len(candidate_sg))
    if candidate_sg.size > 0:
        plotGraph(candidate_sg, False)
