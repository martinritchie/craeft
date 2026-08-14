# Configuration Model

The configuration model with optional clustered (CMA) subgraph embedding.
Uses the config → graph pipeline: `ConfigModelConfig` produces
`ConfigModelGraph` via `from_config()`.

## Top-level types

::: craeft.graphs.configuration_model
    options:
      members:
        - ConfigModelConfig
        - ConfigModelGraph

## Graph model

`ConfigModelGraph.from_config` guarantees the realized degree sequence matches
`config.degrees` exactly: it verifies this before returning and raises
`DegreeMismatchError` on any mismatch, unless `config.verify_degrees` is set
to `False`. See [Degree preservation guarantee](../concepts/ccm.md#degree-preservation-guarantee)
for details.

::: craeft.graphs.configuration_model.models
    options:
      members:
        - ConfigModelConfig
        - ConfigModelGraph
        - DegreeMismatchError

## Sequence sampling

Participation sequence sampling, orbit splitting, and greedy allocation.

::: craeft.graphs.configuration_model.sequence.sampling
    options:
      members:
        - sample_degree_sequence

::: craeft.graphs.configuration_model.sequence.subgraph_sequence
    options:
      members:
        - SubgraphSequence

::: craeft.graphs.configuration_model.sequence.allocation
    options:
      members:
        - Allocation
        - AllocationError
        - allocate_subgraphs

## Connection

Edge formation from subgraph allocations and single stub pairing.

::: craeft.graphs.configuration_model.connection
    options:
      members:
        - Connector
        - ConnectionError

## Legacy modules (networks/)

::: craeft.networks.generation.configuration_model
    options:
      members:
        - configuration_model
        - sample_network
        - sample_degree_sequence

::: craeft.networks.generation.configuration_model.clustered
    options:
      members:
        - clustered_configuration_model
        - sample_clustered_network