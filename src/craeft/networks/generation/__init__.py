"""Network generation algorithms."""

from craeft.networks.generation.configuration_model import (
    SubgraphSpec,
    cma,
    configuration_model,
    sample_degree_sequence,
    sample_network,
)
from craeft.networks.generation.erdos_renyi import random_graph
from craeft.networks.generation.generator import (
    BigVRewiringGenerator,
    CMAGenerator,
    ConfigurationModelGenerator,
    ErdosRenyiGenerator,
    MotifDecompositionGenerator,
    NetworkGenerator,
    PoissonNetworkGenerator,
)

__all__ = [
    "cma",
    "configuration_model",
    "random_graph",
    "sample_degree_sequence",
    "sample_network",
    "SubgraphSpec",
    # Generator protocol + implementations
    "NetworkGenerator",
    "ErdosRenyiGenerator",
    "ConfigurationModelGenerator",
    "PoissonNetworkGenerator",
    "BigVRewiringGenerator",
    "MotifDecompositionGenerator",
    "CMAGenerator",
]
