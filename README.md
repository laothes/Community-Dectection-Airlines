# Community Detection in Airport Networks

This repository implements various community detection algorithms in Python to analyze airport networks. The algorithms aim to identify clusters of airports with high connectivity within the network.

## Data Source

This project utilizes data from OpenFlights: https://openflights.org/data.php.

## Implemented Algorithms

* **Cliques**: Identifies complete subgraphs where every pair of nodes is connected.
* **k-cores**: Finds dense subgraphs where each node has at least k neighbors within the subgraph.
* **Shared Near Neighbor (SNN)**: Groups nodes based on the number of shared neighbors they have.
* **Iterated Highly Connected Subgraphs (IHCS)**: Iteratively identifies and merges highly connected subgraphs.
* **Local Clique Merging Algorithm (LCMA)**: Progressively merges local cliques to form larger communities.

## Dependencies

Core Dependencies:
* networkx (for graph manipulation and algorithms)
* pandas (for data manipulation)
* numpy (for numerical operations)
* pyecharts (for network visualization)

Python Standard Library:
* typing (for type hints)
* itertools (for combinatorial operations)
* collections (for specialized container datatypes)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* OpenFlights.org for providing the airport network dataset
* NetworkX developers for the graph analysis library
* PyeCharts team for the visualization tools
