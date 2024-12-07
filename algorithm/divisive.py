import networkx as nx
import itertools


def girvan_newman_(G, most_valuable_edge=None):
    """
    Repeatedly remove edges from the graph to detect communities via the Girvan-Newman method.
    This modified version supports both directed and undirected graphs.

    The Girvan–Newman algorithm detects communities by progressively removing edges from the
    original graph. The algorithm removes the "most valuable" edges first. After each removal,
    it measures the number of connected components in the graph.

    Args:
        G: NetworkX graph or directed graph (DiGraph)
        most_valuable_edge: Optional function that takes a graph as input and returns
            an edge to be removed. If not specified, edge betweenness centrality is used.

    Yields:
        tuple: A 2-tuple for each iteration:
            - First element is the set of communities (as frozensets of nodes)
            - Second element contains edges removed in this iteration

    Note:
        This is a modified version of the original NetworkX implementation with:
        1. Support for directed weighted graphs (converts to undirected internally for component analysis), as there are different weights for back and dorth routes
        2. Use Dijkstra's algorithm with the 'weight' edge attribute
        3. Returns information about removed edges for each iteration
    """
    if G.number_of_edges() == 0:
        yield tuple(nx.connected_components(G))
        return
    # If no function is provided for computing the most valuable edge,
    # use the edge betweenness centrality.
    if most_valuable_edge is None:
        def most_valuable_edge(G):
            """Returns the edge with the highest betweenness centrality
            in the graph `G`.

            """
            # We have guaranteed that the graph is non-empty, so this
            # dictionary will never be empty.

            # Calculate weighted betweenness centrality using edge weights
            # weight='weight' tells NetworkX to use Dijkstra's algorithm with the 'weight' edge attribute
            betweenness = nx.edge_betweenness_centrality(G, weight = 'weight')
            return max(betweenness, key=betweenness.get)

    # The copy of G here must include the edge weight data.
    g = G.copy()

    # Remove self-loops as they don't affect community structure
    g.remove_edges_from(nx.selfloop_edges(g))
    while g.number_of_edges() > 0:
        yield _without_most_central_edges(g, most_valuable_edge)


def _without_most_central_edges(G, most_valuable_edge):
    """
    Removes edges from graph until the number of connected components increases.

    This function implements one iteration of the Girvan-Newman algorithm by removing
    edges until the graph splits into more components. It tracks all edges removed
    during the process.

    Args:
        G: NetworkX graph or directed graph (will be modified in-place)
        most_valuable_edge: Function that takes a graph and returns the next edge to remove

    Returns:
        tuple: A 2-tuple containing:
            - The new connected components after edge removal (as frozensets of nodes)
            - List of edges removed during this iteration

    Note:
        For directed graphs, component analysis is performed on the undirected version,
        but edge removal is done on the original directed graph.
    """
    # Get initial number of components (using undirected version for directed graphs)
    original_num_components = nx.number_connected_components(G.to_undirected())
    num_new_components = original_num_components
    edges_removed = []

    # Remove edges until we get more components
    while num_new_components <= original_num_components:
        edge = most_valuable_edge(G)
        edges_removed.append(edge)
        G.remove_edge(*edge)
        new_components = tuple(nx.connected_components(G.to_undirected()))
        num_new_components = len(new_components)
    return new_components, edges_removed


if __name__ == '__main__':
    G = nx.DiGraph()
    edges = {
        (1, 2): 1,
        (2, 3): 2,
        (3, 4): 3,
        (4, 1): 4,
        (1, 4): 3,
        (5, 6): 2,
        (6, 7): 3,
        (7, 8): 4,
        (8, 5): 5,
        (4, 5): 1,
    }
    weighted_edges = [(u, v, weight) for (u, v), weight in edges.items()]
    G.add_weighted_edges_from(weighted_edges)

    # stop when the number of communities is greater than *k*
    k = 5
    comp = girvan_newman_(G)
    limited = itertools.takewhile(lambda c: len(c[0]) <= k, comp)
    for communities in limited:
        print(f'Number of communities: {len(communities[0])}')
        print(tuple(sorted(c) for c in communities[0]))
        print(f'Number of Removed edges: {len(communities[1])}')
        print(communities[1])
        print('-----------------------')
