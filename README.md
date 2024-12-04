# Community Detection in Airport Networks - unfinished

This repository implements various community detection algorithms in Python to analyze airport networks. The algorithms aim to identify clusters of airports with high connectivity within the network.

## Project Structure

```
.
├── data/
│   ├── airports.dat    # Airport information dataset
│   └── routes.dat      # Flight routes dataset
├── echart/            # Dynamic visualization results
│   ├── AirportBased/  # Visualizations for airport-based analysis
│   │   └── *.html     # Interactive network visualizations
│   └── RouteBased/    # Visualizations for route-based analysis
│       └── *.html     # Interactive network visualizations
├── algorithms/
│   ├── ihcs.py        # Iterated Highly Connected Subgraphs implementation
│   └── lcma.py        # Local Clique Merging Algorithm implementation
├── notebooks/
│   ├── community-detection_airportbased.ipynb    # Airport-based analysis
│   └── community-detection_routebased.ipynb      # Route-based analysis
└── requirements.txt
```

## Data Source

This project utilizes data from OpenFlights: https://openflights.org/data.php.
- `airports.dat`: Contains airport information including ID, name, location, etc.
- `routes.dat`: Contains route information including source and destination airports.

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
    - pyecharts.charts
    - pyecharts.options
    - pyecharts.globals

Python Standard Library:
* typing (for type hints)
* itertools (for combinatorial operations)
* collections (for specialized container datatypes)

You can install all required packages using:
```bash
pip install -r requirements.txt
```

## Analysis Notebooks

The project includes two main analysis notebooks with different network construction approaches:

1. `community-detection_airportbased.ipynb`
   - Airport-centric network construction:
     * Filters airports based on minimum flight threshold (n = 300 flights)
     * Only includes routes where both source and destination airports meet this threshold
     * Results in a network focused on major aviation hubs
   - Implements all algorithms using this filtered network structure
   - Visualizes communities and their relationships
   - Suitable for analyzing patterns among major airports

2. `community-detection_routebased.ipynb`
   - Route-centric network construction:
     * Filters routes based on minimum frequency threshold (n = 5 occurrences)
     * Only includes airports that appear in these filtered routes
     * Results in a network focused on frequently used connections
   - Implements all algorithms using this filtered network structure
   - Includes route-specific visualizations and analysis
   - Better for analyzing actual flight traffic patterns

Both approaches allow for adjusting the threshold n to analyze different network densities and focus on different aspects of the air transportation network.

## Independent Algorithm Implementations

The project features two custom algorithm implementations:

1. `ihcs.py` - Iterated Highly Connected Subgraphs
   - Iteratively identifies dense subgraphs
   - Merges subgraphs based on connectivity thresholds

2. `lcma.py` - Local Clique Merging Algorithm
   - Identifies and merges local cliques
   - Progressive community formation

## Visualization Results

The `echart` folder contains interactive HTML visualizations organized by analysis type:

### Airport-Based Analysis (`echart/AirportBased/`)
- Network visualizations based on airport connectivity
- Communities detected using airport-centric approach
- Geographic distribution of major airport hubs

### Route-Based Analysis (`echart/RouteBased/`)
- Network visualizations based on route frequency
- Communities detected using route-centric approach
- Frequent route pattern visualizations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* OpenFlights.org for providing the airport network dataset
* NetworkX developers for the graph analysis library
* PyeCharts team for the visualization tools
  
