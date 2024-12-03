import networkx as nx
import numpy as np
from collections import defaultdict


class IHCS:
    """
    Iterated Highly Connected Subgraphs (IHCS) clustering algorithm implementation.
    Based on: "Hartuv, E., & Shamir, R. (2000). A clustering algorithm based on graph connectivity"

    The Iterated HCS version repeatedly applies HCS until convergence.
    """

    def __init__(self, min_cluster_size=3):
        """
        Initialize IHCS clustering algorithm.

        Args:
            min_cluster_size (int): Minimum number of nodes in a cluster (default: 3)
        """
        self.min_cluster_size = min_cluster_size
        self.labels = {}
        self.clusters = {}

    def fit(self, graph):
        """
        Perform IHCS clustering on the input graph.

        Args:
            graph (nx.Graph): Input graph to be clustered

        Returns:
            dict: Cluster labels for each node
        """
        self.graph = graph.copy()

        if self.graph is None or len(self.graph) == 0:
            raise ValueError("No graph provided for clustering")

        # Initial clustering
        clustered_graph = self._ihcs_recursive(self.graph)

        # Assign labels and group clusters
        self._assign_cluster_labels(clustered_graph)
        self._group_clusters()

        return self.labels

    def _is_highly_connected(self, graph):
        """
        Check if a graph is highly connected using edge connectivity.
        A graph is highly connected if its edge connectivity is greater than |V|/2.
        """
        n = len(graph)
        if n <= 1:
            return True

        try:
            edge_connectivity = nx.edge_connectivity(graph)
            return edge_connectivity > n / 2
        except nx.NetworkXError:
            return False

    def _get_minimum_cut(self, graph):
        """Get the minimum cut of a graph"""
        try:
            return nx.minimum_edge_cut(graph)
        except nx.NetworkXError:
            return set()

    def _remove_edges(self, graph, edges):
        """Remove edges from graph"""
        graph_copy = graph.copy()
        graph_copy.remove_edges_from(edges)
        return graph_copy

    def _ihcs_recursive(self, graph):
        """
        Recursive implementation of the Iterated HCS algorithm.

        Args:
            graph (nx.Graph): Input graph to be processed

        Returns:
            nx.Graph: Processed graph with final clustering
        """
        # Base cases
        if len(graph) < self.min_cluster_size:
            return graph

        if not nx.is_connected(graph):
            # Process each component separately
            components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
            result = nx.Graph()
            for component in components:
                if len(component) >= self.min_cluster_size:
                    result = nx.compose(result, self._ihcs_recursive(component))
                else:
                    result = nx.compose(result, component)
            return result

        # Check if current graph is highly connected
        if self._is_highly_connected(graph):
            return graph

        # If not highly connected, find minimum cut and split
        min_cut = self._get_minimum_cut(graph)
        if not min_cut:
            return graph

        # Split graph and recursively process subgraphs
        split_graph = self._remove_edges(graph, min_cut)
        subgraphs = [split_graph.subgraph(c).copy() for c in nx.connected_components(split_graph)]

        # Process each subgraph
        result = nx.Graph()
        for subgraph in subgraphs:
            if len(subgraph) >= self.min_cluster_size:
                processed_subgraph = self._ihcs_recursive(subgraph)
                result = nx.compose(result, processed_subgraph)
            else:
                result = nx.compose(result, subgraph)

        return result

    def _assign_cluster_labels(self, graph):
        """Assign cluster labels to nodes based on connected components"""
        self.labels = {}
        for cluster_idx, component in enumerate(nx.connected_components(graph), 1):
            if len(component) >= self.min_cluster_size:
                for node in component:
                    self.labels[node] = cluster_idx
            else:
                # Assign small components to cluster 0 (outliers)
                for node in component:
                    self.labels[node] = 0

    def _group_clusters(self):
        """
        Group nodes by their cluster labels.

        Returns:
            dict: Dictionary where keys are cluster labels and values are lists of nodes
        """
        self.clusters = defaultdict(list)
        for node, label in self.labels.items():
            if label == 0:
                self.clusters['Singletons'].append(node)
            else:
                self.clusters[f'Subgraph-{label}'].append(node)
        return self.clusters


if __name__ == "__main__":
    # Create example graph
    G = nx.Graph()
    edges = [
        (1, 2), (2, 3), (3, 1),  # triangle
        (4, 5), (5, 6), (6, 4),  # triangle
        (7, 8), (8, 7),  # edge
        (1, 4),  # connection between triangles
    ]
    G.add_edges_from(edges)
    G.add_node(9)  # Add singleton node

    # Run IHCS
    ihcs = IHCS(min_cluster_size=3)
    ihcs.fit(G)

    # Get results
    labels = ihcs.labels
    print("Clusters:", labels)

    clusters = ihcs.clusters
    print("Clusters:", clusters)