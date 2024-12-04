import networkx as nx
from collections import defaultdict
from itertools import combinations
import numpy as np
from typing import List, Set, Dict, Tuple


class LCMARouting:
    """
    Local Clique Merging Algorithm implementation for route analysis.
    Handles route data as dictionary with (origin, destination) tuples as keys.
    """

    def __init__(self, similarity_threshold: float = 0.5, min_clique_size: int = 3):
        """
        Initialize LCMA router

        Args:
            similarity_threshold: Threshold for merging cliques (0.0 to 1.0)
            min_clique_size: Minimum size for considering a clique
        """
        self.similarity_threshold = similarity_threshold
        self.min_clique_size = min_clique_size
        self.graph = None
        self.cliques = []
        self.clusters = []

    def create_route_graph(self, routes: Dict[Tuple[str, str], float]) -> nx.Graph:
        """
        Create a graph from route dictionary

        Args:
            routes: Dictionary with (origin, destination) tuples as keys and weights as values

        Returns:
            NetworkX graph representation
        """
        G = nx.Graph()
        for (origin, dest), weight in routes.items():
            # Add edge with weight. If edge exists, use maximum weight
            if G.has_edge(origin, dest):
                G[origin][dest]['weight'] = max(G[origin][dest]['weight'], weight)
            else:
                G.add_edge(origin, dest, weight=weight)
        return G

    def find_maximal_cliques(self) -> List[Set[str]]:
        """
        Find all maximal cliques in the route graph

        Returns:
            List of sets containing nodes in each clique
        """
        all_cliques = list(nx.find_cliques(self.graph))
        return [set(c) for c in all_cliques if len(c) >= self.min_clique_size]

    def calculate_clique_similarity(self, clique1: Set[str], clique2: Set[str]) -> float:
        """
        Calculate similarity between two cliques using weighted Jaccard coefficient

        Args:
            clique1: First clique
            clique2: Second clique

        Returns:
            Similarity score between 0 and 1
        """
        intersection = len(clique1.intersection(clique2))
        union = len(clique1.union(clique2))

        # Include edge weights in similarity calculation
        if intersection > 0:
            subgraph1 = self.graph.subgraph(clique1)
            subgraph2 = self.graph.subgraph(clique2)
            avg_weight1 = np.mean([d['weight'] for _, _, d in subgraph1.edges(data=True)])
            avg_weight2 = np.mean([d['weight'] for _, _, d in subgraph2.edges(data=True)])
            weight_factor = (avg_weight1 + avg_weight2) / 2 / 20  # Normalize by max weight
            return (intersection / union) * (1 + weight_factor)

        return 0.0

    def merge_similar_cliques(self, cliques: List[Set[str]]) -> List[Set[str]]:
        """
        Merge cliques that have similarity above threshold

        Args:
            cliques: List of cliques to merge

        Returns:
            List of merged cliques
        """
        merged = True
        while merged:
            merged = False
            for i, j in combinations(range(len(cliques)), 2):
                if i < len(cliques) and j < len(cliques):
                    similarity = self.calculate_clique_similarity(cliques[i], cliques[j])
                    if similarity >= self.similarity_threshold:
                        cliques[i] = cliques[i].union(cliques[j])
                        cliques.pop(j)
                        merged = True
                        break
        return cliques

    def analyze_cluster_connectivity(self, cluster: Set[str]) -> Dict:
        """
        Analyze connectivity patterns within a cluster

        Args:
            cluster: Set of nodes in the cluster

        Returns:
            Dict containing connectivity metrics
        """
        subgraph = self.graph.subgraph(cluster)
        edges = subgraph.edges(data=True)
        weights = [d['weight'] for _, _, d in edges]

        return {
            'size': len(cluster),
            'density': nx.density(subgraph),
            'avg_degree': sum(dict(subgraph.degree()).values()) / len(cluster),
            'avg_weight': np.mean(weights) if weights else 0,
            'max_weight': max(weights) if weights else 0,
            'total_flights': sum(weights) if weights else 0
        }

    def find_route_clusters(self, routes: Dict[Tuple[str, str], float]) -> List[Dict]:
        """
        Main method to find and analyze route clusters

        Args:
            routes: Dictionary with (origin, destination) tuples as keys and weights as values

        Returns:
            List of cluster information dictionaries
        """
        self.graph = self.create_route_graph(routes)
        self.cliques = self.find_maximal_cliques()
        self.clusters = self.merge_similar_cliques(self.cliques)

        cluster_info = []
        for i, cluster in enumerate(self.clusters):
            info = {
                'cluster_id': i,
                'nodes': list(cluster),
                'metrics': self.analyze_cluster_connectivity(cluster)
            }
            cluster_info.append(info)

        return cluster_info

    def get_cluster_routes(self, cluster: Set[str]) -> Dict[Tuple[str, str], float]:
        """
        Get all routes within a cluster

        Args:
            cluster: Set of nodes in the cluster

        Returns:
            Dictionary of routes with weights
        """
        subgraph = self.graph.subgraph(cluster)
        routes = {}
        for origin, dest, data in subgraph.edges(data=True):
            routes[(origin, dest)] = data['weight']
        return routes


# Example usage
if __name__ == "__main__":
    # Sample route data as dictionary
    sample_routes = {
        ('ORD', 'ATL'): 20,
        ('ATL', 'ORD'): 19,
        ('ORD', 'MSY'): 13,
        ('HKT', 'BKK'): 13,
        ('HKG', 'BKK'): 12,
        ('CAN', 'HGH'): 12,
        ('DOH', 'BAH'): 12,
        ('ATL', 'MIA'): 12,
        ('AUH', 'MCT'): 12,
        ('BKK', 'HKG'): 12,
        ('JFK', 'LHR'): 12,
        ('MIA', 'ATL'): 12,
        ('LHR', 'JFK'): 12,
        ('LHR', 'LAX'): 11,
        ('ATL', 'DFW'): 11,
        ('KGL', 'EBB'): 11,
        ('MSY', 'JFK'): 11,
        ('MCT', 'AUH'): 11,
        ('CNX', 'BKK'): 11,
        ('CDG', 'JFK'): 11,
        ('SFO', 'ATL'): 11
    }
    sample_routes = {
        ("JFK", "LAX") : 100,
        ("LAX", "SFO") : 80,
        ("SFO", "JFK") : 90,  # Forms a triangle
        ("ORD", "DFW") : 70,
        ("DFW", "MIA") : 60,
        ("MIA", "ORD") : 65,  # Forms another triangle
        ("LHR", "CDG") : 85,
        ("CDG", "AMS") : 75,
        ("AMS", "LHR") : 80,  # European triangle
        ("JFK", "LHR") : 95,  # Connection between clusters
    }

    # Initialize and run LCMA
    lcma = LCMARouting(similarity_threshold=0.3, min_clique_size=3)
    clusters = lcma.find_route_clusters(sample_routes)

    # Print results
    print("\nRoute Clusters Found:")
    for cluster in clusters:
        print(f"\nCluster {cluster['cluster_id']}:")
        print(f"Nodes (Airports): {', '.join(sorted(cluster['nodes']))}")
        print("Metrics:")
        for metric, value in cluster['metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")

        # Get routes within cluster
        routes = lcma.get_cluster_routes(set(cluster['nodes']))
        if routes:
            print("Routes (with frequencies):")
            for (origin, dest), weight in sorted(routes.items()):
                print(f"  {origin} -> {dest}: {weight}")

    try:
        import matplotlib.pyplot as plt

        # Visualize clusters
        plt.figure(figsize=(15, 10))
        G = lcma.graph
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw nodes for each cluster with different colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
        for i, cluster in enumerate(clusters):
            nx.draw_networkx_nodes(G, pos,
                                   nodelist=cluster['nodes'],
                                   node_color=colors[i % len(colors)],
                                   node_size=700,
                                   label=f'Cluster {i}')

        # Draw edges with weights affecting thickness
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights)
        edge_widths = [2 * w / max_weight for w in edge_weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)

        # Draw labels
        nx.draw_networkx_labels(G, pos)

        plt.title('Route Clusters Identified by LCMA\n(Edge thickness represents flight frequency)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nMatplotlib not available for visualization")