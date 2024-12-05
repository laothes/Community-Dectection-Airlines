# Community Detection in Airport Networks

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
│   ├── lcma.py        # Local Clique Merging Algorithm implementation
│   ├── snn.py         # Shared Near Neighbor implementation
│   └── divisive.py    # Modified Girvan-Newman algorithm implementation
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
* **Girvan-Newman Algorithm**: A divisive hierarchical clustering algorithm based on edge betweenness centrality, adapted specifically for airport networks.

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

The project features four custom algorithm implementations:

1. `ihcs.py` - Iterated Highly Connected Subgraphs
   - Iteratively identifies dense subgraphs
   - Merges subgraphs based on connectivity thresholds
   - Optimized for airport network analysis

2. `lcma.py` - Local Clique Merging Algorithm
   - Identifies and merges local cliques
   - Progressive community formation
   - Specifically adapted for airport connectivity patterns

3. `snn.py` - Shared Near Neighbor
   - Groups nodes based on shared neighbor counts
   - Identifies communities through neighbor similarity
   - Effective for finding naturally clustered airports

4. `divisive.py` - Modified Girvan-Newman Algorithm
   - Based on NetworkX's girvan_newman implementation
   - Customized for airport network analysis
   - Uses edge betweenness centrality to identify community boundaries
   - Includes additional geographical considerations for airport communities

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

### Visualization Suggestions

1. Display Dimensions:
   - Width: 1600px
   - Height: 800px
   - **Note**: Best viewed on large screens or displays with at least 1920×1080 resolution

2. Viewing Instructions:
   - Open any `.html` file in the respective folders using a web browser
   - Allow a few moments for large visualizations to fully render
   - Use browser zoom controls to adjust the view if needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* OpenFlights.org for providing the airport network dataset
* NetworkX developers for the graph analysis library
* PyeCharts team for the visualization tools