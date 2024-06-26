import networkx as nx
import numpy as np

from common import GraphLayer
from graph_generator import extract_cluster_list_subgraph


def find_path(
        layer: GraphLayer,
        from_node: int,
        to_node: int,
        alg='dijkstra') -> tuple[float, list[int]]:
    from_d = layer.graph.nodes[from_node]
    to_d = layer.graph.nodes[to_node]

    from_cluster = from_d['cluster']
    to_cluster = to_d['cluster']

    def h(a, b):
        # print(a, b)
        da = layer.graph.nodes[a]
        db = layer.graph.nodes[b]
        return ((da['x'] - db['x']) ** 2 + (da['y'] - db['y']) ** 2) ** 0.5 / 360 * 2 * np.pi * 6371.01 * 1000

    if from_cluster == to_cluster:
        try:
            g = extract_cluster_list_subgraph(layer.graph, [to_cluster], layer.communities)
            if alg == 'dijkstra':
                return nx.single_source_dijkstra(g, from_node, to_node, weight='length')
            if alg == 'bidirectional':
                return nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')
            if alg == 'astar':
                return [nx.astar_path_length(g, from_node, to_node, weight='length', heuristic=h)]
        except nx.NetworkXNoPath as e:
            print('No path found in one cluster')
            raise e

    from_center = layer.cluster_to_center[from_cluster]
    to_center = layer.cluster_to_center[to_cluster]

    try:
        g = layer.centroids_graph
        path = []
        if alg == 'dijkstra':
            path = nx.single_source_dijkstra(g, from_center, to_center, weight='length')
        if alg == 'bidirectional':
            path = nx.bidirectional_dijkstra(g, from_center, to_center, weight='length')
        if alg == 'astar':
            path = [nx.astar_path_length(g, from_center, to_center, weight='length', heuristic=h)]
    except nx.NetworkXNoPath as e:
        print('No path found in clusters')
        raise e

    cls = set()
    cls.add(to_cluster)
    for u in path:
        c = layer.graph.nodes[u]['cluster']
        cls.add(c)

    g = extract_cluster_list_subgraph(layer.graph, cls, layer.communities)
    try:
        if alg == 'dijkstra':
            return nx.single_source_dijkstra(g, from_node, to_node, weight='length')
        if alg == 'bidirectional':
            return nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')
        if alg == 'astar':
            return [nx.astar_path_length(g, from_node, to_node, weight='length', heuristic=h)]
    except nx.NetworkXNoPath as e:
        print(nx.is_connected(g))
        print('No path in cluster subgraph')
        raise e



def find_path_length(
        layer: GraphLayer,
        from_node: int,
        to_node: int,
        alg='dijkstra') -> float:
    from_d = layer.graph.nodes[from_node]
    to_d = layer.graph.nodes[to_node]

    from_cluster = from_d['cluster']
    to_cluster = to_d['cluster']

    def h(a, b):
        # print(a, b)
        da = layer.graph.nodes[a]
        db = layer.graph.nodes[b]
        return ((da['x'] - db['x']) ** 2 + (da['y'] - db['y']) ** 2) ** 0.5 / 360 * 2 * np.pi * 6371.01 * 1000

    if from_cluster == to_cluster:
        try:
            g = extract_cluster_list_subgraph(layer.graph, [to_cluster], layer.communities)
            if alg == 'dijkstra':
                return nx.single_source_dijkstra(g, from_node, to_node, weight='length')[0]
            if alg == 'bidirectional':
                return nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')[0]
            if alg == 'astar':
                return nx.astar_path_length(g, from_node, to_node, weight='length', heuristic=h)
        except nx.NetworkXNoPath as e:
            print('No path found in one cluster')
            return -1

    from_center = layer.cluster_to_center[from_cluster]
    to_center = layer.cluster_to_center[to_cluster]

    try:
        g = layer.centroids_graph
        path = []
        if alg == 'dijkstra':
            path = nx.single_source_dijkstra(g, from_center, to_center, weight='length')[1]
        if alg == 'bidirectional':
            path = nx.bidirectional_dijkstra(g, from_center, to_center, weight='length')[1]
        if alg == 'astar':
            path = nx.astar_path(g, from_center, to_center, weight='length', heuristic=h)
    except nx.NetworkXNoPath as e:
        print('No path found in clusters')
        return -1

    cls = set()
    cls.add(to_cluster)
    for u in path:
        c = layer.graph.nodes[u]['cluster']
        cls.add(c)

    g = extract_cluster_list_subgraph(layer.graph, cls, layer.communities)
    try:
        if alg == 'dijkstra':
            return nx.single_source_dijkstra(g, from_node, to_node, weight='length')[0]
        if alg == 'bidirectional':
            return nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')[0]
        if alg == 'astar':
            return nx.astar_path_length(g, from_node, to_node, weight='length', heuristic=h)
    except nx.NetworkXNoPath as e:
        print(nx.is_connected(g))
        print('No path in cluster subgraph')
        return -1
