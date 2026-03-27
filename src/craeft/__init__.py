"""
craeft: Generate random networks with controlled higher-order structure.

Tools for constructing random networks where you control the degree sequence,
clustering coefficient, and local motif structure. Includes a generic Gillespie
simulation engine for running continuous-time Markov chains on the results.

Quick Start:
    >>> from craeft import configuration_model, G2
    >>> from craeft.point_processes.epidemics import SIRConfig, SIRSimulator

Submodules:
    craeft.networks.generation   - Network generators (ER, configuration model, CCM)
    craeft.networks.generation.motifs - Graph motifs (G2, G8, etc.)
    craeft.networks.rewiring     - Rewiring algorithms (Big-V, motif decomposition)
    craeft.networks.metrics      - Network metrics (clustering, connectivity)
    craeft.point_processes       - Gillespie engine and process abstractions
    craeft.point_processes.epidemics - Epidemic models (SIR)
    craeft.utils                 - Plotting utilities
"""

from craeft.experiment import Experiment, ExperimentResult
from craeft.networks.generation import (
    BigVRewiringGenerator,
    ConfigurationModelGenerator,
    ErdosRenyiGenerator,
    MotifDecompositionGenerator,
    NetworkGenerator,
    PoissonNetworkGenerator,
    configuration_model,
    random_graph,
    sample_network,
)
from craeft.networks.generation.motifs import G2, G8, G14, K6, Motif
from craeft.networks.metrics import (
    DisconnectedGraphError,
    global_clustering_coefficient,
    is_connected,
)
from craeft.point_processes.epidemics.simulator import EpidemicSimulator, SIRSimulator

__all__ = [
    # Generation
    "configuration_model",
    "random_graph",
    "sample_network",
    # Generator protocol + implementations
    "NetworkGenerator",
    "ErdosRenyiGenerator",
    "ConfigurationModelGenerator",
    "PoissonNetworkGenerator",
    "BigVRewiringGenerator",
    "MotifDecompositionGenerator",
    # Simulator protocol + implementations
    "EpidemicSimulator",
    "SIRSimulator",
    # Experiment
    "Experiment",
    "ExperimentResult",
    # Motifs
    "G2",
    "G8",
    "G14",
    "K6",
    "Motif",
    # Metrics
    "DisconnectedGraphError",
    "global_clustering_coefficient",
    "is_connected",
]
