import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Tuple

class SSN:
    def __init__(self, snn_routes: Dict[Tuple[str, str], int], method='mixed'):
        '''
        Initialize Jarvis-Patrick clustering.
        Parameters:
        - snn_routes: A dictionary of route counts with keys as (source, destination)
        - method: How similarity is calculated ('source', 'destination', or 'mixed').
            'source': similarity calculate by route counts from source airports - only forth
            'destination': similarity calculate by route counts to destination airports - only back
            'mixed': similarity calculate by aggregate route counts - both back and forth
        '''
        if method not in ('source', 'destination', 'mixed'):
            raise ValueError("Invalid method. Choose from 'source', 'destination', 'mixed'.")
        self.method = method
        self.snn_routes = snn_routes.copy()
        self.similarity_df = None

    def similarity_matrix_calu(self):
        """Calculate the similarity matrix based on the chosen method."""
        # Extract unique airports
        snn_airports = list(set([airport for route in self.snn_routes.keys() for airport in route]))
        airport_index = {airport: idx for idx, airport in enumerate(snn_airports)}

        # Initialize similarity matrix
        similarity_matrix = np.zeros((len(snn_airports), len(snn_airports)), dtype=int)

        # Fill similarity matrix
        for (source, destination), count in self.snn_routes.items():
            i, j = airport_index[source], airport_index[destination]
            if self.method == 'source':
                similarity_matrix[i, j] = count
            elif self.method == 'destination':
                similarity_matrix[j, i] = count
            elif self.method == 'mixed':
                similarity_matrix[i, j] += count
                similarity_matrix[j, i] += count

        # Convert to DataFrame
        self.similarity_df = pd.DataFrame(similarity_matrix, index=snn_airports, columns=snn_airports)

    def non_zero_similarity(self):
        """Generate descriptive statistics for non-zero similarities."""
        if self.similarity_df is None:
            self.similarity_matrix_calu()
        non_zero_counts = (self.similarity_df != 0).sum()
        return non_zero_counts

    def jp_cluster_calu(self, k=3, T1=2, T2=1):
        """
        Perform Jarvis-Patrick clustering based on SNN similarity.

        Parameters:
        - k: Number of nearest neighbors to consider.
        - T1: Minimum shared neighbors for clustering.
        - T2: Minimum similarity score for clustering.

        Returns:
        - clusters: A dictionary with cluster indices and members.
        """
        if self.similarity_df is None:
            self.similarity_matrix_calu()

        # Find k-nearest neighbors for each airport
        neighbors = {
            airport: self.similarity_df.loc[airport]
                     .nlargest(k + 1)  # Include self
                     .iloc[1:]  # Exclude self
            .index.tolist()
            for airport in self.similarity_df.index
        }
        self.neighbors = neighbors

        # Build graph based on shared neighbors and similarity thresholds
        G = nx.Graph()
        for airport, neighbor_list in neighbors.items():
            for neighbor in neighbor_list:
                shared_neighbors = len(set(neighbors[airport]) & set(neighbors[neighbor]))
                if shared_neighbors >= T1 and self.similarity_df.loc[airport, neighbor] >= T2:
                    G.add_edge(airport, neighbor)

        # Find connected components (clusters)
        self.clusters = {i: list(c) for i, c in enumerate(nx.connected_components(G))}
        return self.clusters

if __name__ == '__main__':
    sample_routes = {
        (1, 2): 5,  # Hub 1 to Hub 2
        (2, 1): 1,
        (1, 3): 3,  # Hub 1 to Hub 3
        (3, 1): 2,
        (2, 4): 2,  # Hub 2 to Hub 4
        (4, 2): 5,
        (3, 5): 2,  # Hub 3 to Hub 5
        (5, 3): 1,
        (5, 2): 3,  # Hub 5 to Hub 2
        (2, 5): 4,
        (4, 6): 5,  # Hub 4 to Hub 6
        (6, 4): 5
    }

    # Test with different methods
    for method in ['source', 'destination', 'mixed']:
        print(f"\nMethod: {method}")
        snn = SSN(sample_routes, method)
        snn.similarity_matrix_calu()
        print(snn.similarity_df)
        clusters = snn.jp_cluster_calu(k=3, T1=2, T2=1)
        print(f"Clusters: {clusters}")