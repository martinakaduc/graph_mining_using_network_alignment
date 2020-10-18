import numpy as np
import random
from utils import plotGraph

def generate(num_of_graphs, min_node, max_node, subgraph_size, node_degree=3, label_ratio=0.5,
            random_node=True, random_edge=True, plotSG=True):
    graph_db = []
    graph_index = []
    subgraph = np.zeros((subgraph_size, subgraph_size), dtype=np.int)

    # Random filling index
    for _ in range(num_of_graphs):
        length = random.randint(min_node, max_node)
        graph_index.append(random.sample(range(length), k=length))

    # Random subgraph
    if random_node:
        list_node_val = random.choices(list(range(1, int(subgraph_size*label_ratio))), k=subgraph_size)
    else:
        list_node_val = random.choices([1], k=subgraph_size)

    for i in range(subgraph_size):
        subgraph[i][i] = list_node_val[i]

    subgraph_total_edge = subgraph_size * node_degree
    max_edge_val = int(subgraph_total_edge * label_ratio)

    for _ in range(int(subgraph_total_edge)):
        y, x = np.where(subgraph == 0)
        if len(y) == 0 or len(x) == 0:
            break
        y, x = random.choice(list(zip(y, x)))

        if random_edge:
            edge_val = random.randint(1, max_edge_val)
        else:
            edge_val = 1

        subgraph[x][y] = edge_val
        subgraph[y][x] = edge_val

    if plotSG:
        plotGraph(subgraph, False)

    # Intergrate subgraph and random graph
    for i in range(num_of_graphs):
        print("GRAPH: %d" % i, end='\r')
        # Copy subgraph
        length = len(graph_index[i])
        graph = np.zeros((length,length), dtype=np.int)

        for k in range(subgraph_size):
            graph[graph_index[i][k]][graph_index[i][k]] = subgraph[k][k] # Node
            iter_j = np.where(subgraph[k] > 0)[0]
            iter_j = iter_j[iter_j > k]
            # Edge
            for j in iter_j:
                graph[graph_index[i][k]][graph_index[i][j]] = subgraph[k][j]
                graph[graph_index[i][j]][graph_index[i][k]] = subgraph[j][k]

        # Generate random remaining node & edge
        if random_node:
            list_node_val = random.choices(list(range(1, int(length*label_ratio))), k=length-subgraph_size)
        else:
            list_node_val = random.choices([1], k=length-subgraph_size)

        for enum, k in enumerate(graph_index[i][subgraph_size:]):
            graph[k][k] = list_node_val[enum]

        graph_total_edge = length * node_degree
        max_edge_val = int(graph_total_edge * label_ratio)

        for _ in range(graph_total_edge-subgraph_total_edge):
            y, x = np.where(graph == 0)
            if len(y) == 0 or len(x) == 0:
                break
            y, x = random.choice(list(zip(y, x)))

            if random_edge:
                edge_val = random.randint(1, max_edge_val)
            else:
                edge_val = 1

            graph[x][y] = edge_val
            graph[y][x] = edge_val

            x = x + 1
            if x == length:
                x = subgraph_size

        graph_db.append(graph)

    link = np.asarray([x[:subgraph_size] for x in graph_index])
    return np.asarray(graph_db), np.asarray([[link[i], link[(i+1)%num_of_graphs]] for i in range(link.shape[0])])

if __name__ == '__main__':
    graph_db, _ = generate(num_of_graphs=10, min_node=8, max_node=10, subgraph_size=5)
    for x in graph_db:
        print(x)
    print(_)
