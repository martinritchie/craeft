# Generators

Graph generation via config → graph pipelines and legacy generator
dataclasses.

## Graph base classes (new API)

::: craeft.graphs.base
    options:
      members:
        - GraphConfig
        - BaseGraph
        - UndirectedGraph
        - DirectedGraph
        - Subgraph
        - DirectedSubgraph

## Legacy generators (networks/)

::: craeft.networks.generation.generator
    options:
      members:
        - NetworkGenerator
        - ErdosRenyiGenerator
        - ConfigurationModelGenerator
        - PoissonNetworkGenerator
        - BigVRewiringGenerator
        - MotifDecompositionGenerator