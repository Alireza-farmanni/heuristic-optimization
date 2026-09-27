# heuristic_opt

A modular, highly scalable Python package implementing 13 distinct heuristic and meta-heuristic optimization algorithms. 

Designed for research and production pipelines, this library provides a unified, object-oriented API that separates continuous space optimization from discrete graph-based routing. All continuous algorithms are fully vectorized using NumPy to eliminate slow nested loops and handle high-dimensional parameter spaces efficiently.

## Supported Algorithms

The package is categorized into three algorithm families based on their search mechanics:

**Swarm Intelligence (`heuristic_opt.algorithms.swarm`)**
*   **PSO:** Particle Swarm Optimization
*   **GWO:** Grey Wolf Optimizer
*   **SSA:** Salp Swarm Algorithm
*   **BA:** Bat Algorithm
*   **CSA:** Crow Search Algorithm
*   **GOA:** Grasshopper Optimization Algorithm
*   **KH:** Krill Herding

**Evolutionary & Nature-Inspired (`heuristic_opt.algorithms.evolutionary`)**
*   **CSO:** Cuckoo Search Optimization
*   **EHO:** Elephant Herding Optimization
*   **DFO:** Dispersive Fly Optimization
*   **HUS:** Hunter Search
*   **MBO:** Monarch Butterfly Optimization
*   **BFO:** Bacterial Foraging Optimization

**Discrete Graph Optimization (`heuristic_opt.algorithms.ant_colony`)**
*   **ACO:** Ant Colony Optimization

## Repository Structure

```text
heuristic_opt/
├── __init__.py
├── base.py                   # Core abstract classes (BaseContinuousOptimizer, BaseDiscreteOptimizer)
├── algorithms/
│   ├── __init__.py
│   ├── ant_colony.py         # Graph-based routing
│   ├── evolutionary.py       # Breeding, flight, and dispersion algorithms
│   └── swarm.py              # Collective behavior algorithms
└── examples/
    ├── __init__.py
    ├── discrete_graph_optimization.py
    ├── single_algorithm_execution.py
    └── multi_algorithm_benchmarking_pipeline.py