# API Reference

Full reference for the craeft public API. All primary symbols are
re-exported from the package root:

```python
from craeft import (
    # Generation
    configuration_model,
    random_graph,
    sample_network,
    # Generator implementations
    ErdosRenyiGenerator,
    ConfigurationModelGenerator,
    PoissonNetworkGenerator,
    BigVRewiringGenerator,
    MotifDecompositionGenerator,
    # Metrics
    global_clustering_coefficient,
    is_connected,
    DisconnectedGraphError,
    # Motifs
    G2, G8, G14, K6, Motif,
    # Orchestration
    Experiment, ExperimentResult,
    # Simulation
    EpidemicSimulator, SIRSimulator,
)
```

## Subpackages

| Package | Description |
|---------|-------------|
| [`craeft.networks.generation`](generators.md) | Network generator protocol and implementations |
| [`craeft.networks.generation.configuration_model`](configuration-model.md) | CCM pipeline (partition, connectors, pairing, assembly) |
| [`craeft.networks.generation.motifs`](motifs.md) | Motif definitions and registry |
| [`craeft.networks.generation.distributions`](distributions.md) | Degree distribution sampling |
| [`craeft.networks.metrics`](metrics.md) | Clustering coefficients, connectivity |
| [`craeft.networks.rewiring`](rewiring.md) | Big-V and motif decomposition |
| [`craeft.point_processes`](point-processes.md) | Gillespie engine and core abstractions |
| [`craeft.point_processes.epidemics`](sir.md) | SIR model |
| [`craeft.experiment`](experiment.md) | Experiment orchestration |
| [`craeft.utils.plotting`](plotting.md) | Visualisation |
