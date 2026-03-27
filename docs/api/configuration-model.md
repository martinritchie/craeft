# Configuration Model

The configuration model pipeline, including both the standard (unclustered)
and clustered variants.

## Top-level functions

::: craeft.networks.generation.configuration_model
    options:
      members:
        - configuration_model
        - sample_network
        - sample_degree_sequence

## Core utilities

::: craeft.networks.generation.configuration_model.core
    options:
      members:
        - sample_degree_sequence
        - edges_to_csr
        - configuration_model

## Clustered configuration model

::: craeft.networks.generation.configuration_model.clustered
    options:
      members:
        - clustered_configuration_model
        - sample_clustered_network

## Stub partition

::: craeft.networks.generation.configuration_model.partition
    options:
      members:
        - StubPartition
        - MixedCardinalityError
        - partition_stubs

## Connectors

Strategies for forming motif instances from hyperstub arrays.

::: craeft.networks.generation.configuration_model.connectors
    options:
      members:
        - Connector
        - Repeated
        - Refuse
        - Erased
        - get_connector

## Pairing

Strategies for pairing single (non-motif) stubs into edges.

::: craeft.networks.generation.configuration_model.pairing
    options:
      members:
        - Pairer
        - PairingResult
        - Repeated
        - Refuse
        - Erased
        - get_pairer
        - pair_singles

## Assembly

::: craeft.networks.generation.configuration_model.assembly
    options:
      members:
        - AssemblyResult
        - assemble
