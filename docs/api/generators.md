# Generators

The `NetworkGenerator` protocol and its concrete implementations. Each
generator is a frozen dataclass that produces a `csr_matrix` adjacency
from a NumPy `Generator`.

::: craeft.networks.generation.generator
    options:
      members:
        - NetworkGenerator
        - ErdosRenyiGenerator
        - ConfigurationModelGenerator
        - PoissonNetworkGenerator
        - BigVRewiringGenerator
        - MotifDecompositionGenerator
