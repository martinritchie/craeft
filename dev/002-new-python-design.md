# Plugin API & Configuration Schema Design

This document specifies the plugin architecture and declarative configuration system for the network-simulation library.

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Plugin API Contracts](#plugin-api-contracts)
   - [Motifs](#motifs)
   - [SIR Epidemic Model](#sir-epidemic-model)
   - [Connection Algorithms](#connection-algorithms)
   - [Distributions](#distributions)
3. [Plugin Registry](#plugin-registry)
4. [Configuration Schema](#configuration-schema)
5. [Validation & Error Handling](#validation--error-handling)
6. [Example Workflows](#example-workflows)
7. [Key Design Decisions](#key-design-decisions)

---

## Core Concepts

### Design Principles

1. **Explicit over implicit**: Plugins declare their capabilities upfront
2. **Fail fast**: Invalid configurations error at load time, not runtime
3. **Sensible defaults**: Minimal config should "just work"
4. **Composable**: Mix built-in and custom components freely
5. **Introspectable**: Users can query what's available (`netsubgraph list-motifs`)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  YAML/TOML  │  │  Python API │  │  CLI                    │  │
│  │  Config     │  │  (direct)   │  │  netsubgraph run ...    │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Configuration Layer                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Config Parser + Validator (Pydantic models)            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Plugin Registry                            │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐  │
│  │  Motifs    │ │  Models    │ │ Connectors │ │ Distributions│  │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core Engine                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  CMA / UDA  │  │  PGF-ODE    │  │  Stochastic Simulation  │  │
│  │  Generator  │  │  Generator  │  │  Engine                 │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Plugin API Contracts

### Motifs

Motifs (subgraphs) are the fundamental building blocks for network generation.
A motif is simply an adjacency matrix - all other properties are derived.

**Constraint**: A valid motif of N nodes must contain a Hamiltonian cycle
(a cycle visiting all N nodes exactly once). This ensures the motif is
sufficiently connected for the CMA algorithm.

```python
# netsubgraph/core/motif.py

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


class MotifValidationError(ValueError):
    """Invalid motif structure."""
    pass


def _has_hamiltonian_cycle(adj: NDArray[np.int_]) -> bool:
    """
    Check if graph has a Hamiltonian cycle using backtracking.

    For small motifs (≤10 nodes), brute force is fine.
    """
    n = adj.shape[0]
    if n < 3:
        return n == 2 and adj[0, 1] == 1

    def backtrack(path: list[int], visited: set[int]) -> bool:
        if len(path) == n:
            return adj[path[-1], path[0]] == 1

        current = path[-1]
        for next_node in range(n):
            if next_node not in visited and adj[current, next_node] == 1:
                path.append(next_node)
                visited.add(next_node)
                if backtrack(path, visited):
                    return True
                path.pop()
                visited.remove(next_node)
        return False

    return backtrack([0], {0})


def _validate_adjacency(adj: NDArray[np.int_]) -> None:
    """Validate adjacency matrix structure. Raises MotifValidationError."""
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise MotifValidationError(f"Adjacency must be square, got shape {adj.shape}")

    if not np.allclose(adj, adj.T):
        raise MotifValidationError("Adjacency matrix must be symmetric")

    if not np.all((adj == 0) | (adj == 1)):
        raise MotifValidationError("Adjacency matrix must be binary (0s and 1s)")

    if np.any(np.diag(adj) != 0):
        raise MotifValidationError("Adjacency must have zero diagonal (no self-loops)")

    if not _has_hamiltonian_cycle(adj):
        raise MotifValidationError(
            "Motif must have a Hamiltonian cycle (a cycle visiting all nodes). "
            "This ensures sufficient connectivity for the CMA algorithm."
        )


def _compute_corner_types(degrees: tuple[int, ...]) -> tuple[int, ...]:
    """Derive corner types from degrees. Same degree → same type."""
    unique_degrees = sorted(set(degrees))
    degree_to_type = {d: i + 1 for i, d in enumerate(unique_degrees)}
    return tuple(degree_to_type[d] for d in degrees)


def _compute_cardinalities(
    corner_types: tuple[int, ...], degrees: tuple[int, ...]
) -> Mapping[int, int]:
    """Map each corner type to its degree within the motif."""
    result: dict[int, int] = {}
    for i, t in enumerate(corner_types):
        if t not in result:
            result[t] = degrees[i]
    return result


@dataclass(frozen=True, slots=True)
class Motif:
    """
    A motif (subgraph) defined by its adjacency matrix.

    Immutable value object. All properties are derived from the adjacency
    matrix at construction time.

    Example:
        >>> triangle = Motif.create([[0, 1, 1],
        ...                          [1, 0, 1],
        ...                          [1, 1, 0]])
        >>> triangle.num_nodes
        3
        >>> triangle.corner_types
        (1, 1, 1)
    """

    adjacency: NDArray[np.int_]
    num_nodes: int
    num_edges: int
    degrees: tuple[int, ...]
    corner_types: tuple[int, ...]
    cardinalities: Mapping[int, int]

    def __hash__(self) -> int:
        return hash(self.adjacency.tobytes())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Motif):
            return NotImplemented
        return np.array_equal(self.adjacency, other.adjacency)

    @property
    def is_complete(self) -> bool:
        """True if all nodes have the same degree (complete subgraph)."""
        return len(set(self.degrees)) == 1

    @classmethod
    def create(cls, adjacency: list[list[int]] | NDArray[np.int_]) -> "Motif":
        """
        Create a validated Motif from an adjacency matrix.

        Args:
            adjacency: Square symmetric binary matrix (list or numpy array)

        Returns:
            Validated Motif instance

        Raises:
            MotifValidationError: If adjacency is invalid or lacks Hamiltonian cycle
        """
        adj = np.asarray(adjacency, dtype=np.int_)
        _validate_adjacency(adj)

        # Make immutable
        adj = adj.copy()
        adj.flags.writeable = False

        degrees = tuple(int(d) for d in np.sum(adj, axis=1))
        corner_types = _compute_corner_types(degrees)
        cardinalities = _compute_cardinalities(corner_types, degrees)

        return cls(
            adjacency=adj,
            num_nodes=adj.shape[0],
            num_edges=int(np.sum(adj) // 2),
            degrees=degrees,
            corner_types=corner_types,
            cardinalities=cardinalities,
        )

    def __repr__(self) -> str:
        return f"Motif(nodes={self.num_nodes}, edges={self.num_edges})"
```

#### Built-in Motifs

```python
# netsubgraph/library/builtins.py

from netsubgraph.core.motif import Motif

# Complete subgraphs (all corners equivalent, all Hamiltonian)
EDGE = Motif.create([[0, 1],
                     [1, 0]])

TRIANGLE = Motif.create([[0, 1, 1],
                         [1, 0, 1],
                         [1, 1, 0]])

K4 = Motif.create([[0, 1, 1, 1],
                   [1, 0, 1, 1],
                   [1, 1, 0, 1],
                   [1, 1, 1, 0]])

K5 = Motif.create([[0, 1, 1, 1, 1],
                   [1, 0, 1, 1, 1],
                   [1, 1, 0, 1, 1],
                   [1, 1, 1, 0, 1],
                   [1, 1, 1, 1, 0]])

K6 = Motif.create([[0, 1, 1, 1, 1, 1],
                   [1, 0, 1, 1, 1, 1],
                   [1, 1, 0, 1, 1, 1],
                   [1, 1, 1, 0, 1, 1],
                   [1, 1, 1, 1, 0, 1],
                   [1, 1, 1, 1, 1, 0]])

# Cycles (incomplete subgraphs, but Hamiltonian by definition)
SQUARE_CYCLE = Motif.create([[0, 1, 0, 1],
                             [1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [1, 0, 1, 0]])

# Incomplete subgraphs (multiple corner types)
#
# Toast structure:
#     0 --- 1 --- 3
#     |     |     |
#     +--2--+-----4
#
# Hamiltonian cycle: 0 → 1 → 3 → 4 → 2 → 0
# Degrees: [2, 3, 3, 2, 2] → corner_types: [1, 2, 2, 1, 1]
#
TOAST = Motif.create([[0, 1, 1, 0, 0],
                      [1, 0, 1, 1, 0],
                      [1, 1, 0, 0, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 1, 1, 0]])
```

```python
# netsubgraph/library/registry.py

from netsubgraph.core.motif import Motif

# Built-in motifs (immutable reference)
_BUILTINS: dict[str, Motif] = {}  # Populated on import from builtins

# Custom motifs (registered at runtime)
_CUSTOM: dict[str, Motif] = {}


def register(name: str, motif: Motif) -> None:
    """Register a custom motif."""
    if name in _BUILTINS:
        raise ValueError(f"Cannot override built-in motif '{name}'")
    _CUSTOM[name] = motif


def get(name: str) -> Motif:
    """Get a motif by name."""
    if name in _BUILTINS:
        return _BUILTINS[name]
    if name in _CUSTOM:
        return _CUSTOM[name]
    available = ", ".join(sorted(list_all()))
    raise KeyError(f"Unknown motif '{name}'. Available: {available}")


def list_all() -> list[str]:
    """List all available motif names."""
    return sorted(list(_BUILTINS.keys()) + list(_CUSTOM.keys()))
```

#### Custom Motifs

```python
# User code
from netsubgraph.core.motif import Motif, MotifValidationError
from netsubgraph.library.registry import register

# Valid: House motif (5 nodes, has Hamiltonian cycle)
#
#       0
#      / \
#     1---3
#     |   |
#     2---4
#
# Cycle: 0 → 1 → 2 → 4 → 3 → 0
#
register("house", Motif.create([
    [0, 1, 0, 1, 0],  # 0 connects to 1, 3
    [1, 0, 1, 1, 0],  # 1 connects to 0, 2, 3
    [0, 1, 0, 0, 1],  # 2 connects to 1, 4
    [1, 1, 0, 0, 1],  # 3 connects to 0, 1, 4
    [0, 0, 1, 1, 0],  # 4 connects to 2, 3
]))


# Invalid: Bowtie (no Hamiltonian cycle) - this will raise an error!
#
#     1       4
#    / \     / \
#   0---2---3---5
#
# Nodes 2 and 3 are cut vertices; no cycle visits all 6 nodes.
#
try:
    Motif.create([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ])
except MotifValidationError as e:
    print(e)  # "Motif must have a Hamiltonian cycle..."
```

---

### SIR Epidemic Model

The library uses the SIR (Susceptible-Infected-Recovered) model exclusively.
The domain object is an immutable dataclass; Pydantic is used only in the config layer.

```python
# netsubgraph/core/sir.py

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SIRConfig:
    """
    SIR epidemic model configuration.

    Immutable value object representing simulation parameters.

    Transitions:
        S → I: Infection (rate τ per S-I edge per unit time)
        I → R: Recovery (rate γ per infected per unit time)
    """

    tau: float = 1.0
    gamma: float = 1.0
    initial_infected: float = 0.0001
    t_end: float = 15.0
    method: Literal["ode", "stochastic", "both"] = "both"
    realizations: int = 100

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if not 0 <= self.initial_infected <= 1:
            raise ValueError("initial_infected must be in [0, 1]")
        if self.t_end <= 0:
            raise ValueError("t_end must be positive")
        if self.realizations < 1:
            raise ValueError("realizations must be >= 1")

    @property
    def R0(self) -> float:
        """Basic reproduction number (for homogeneous mixing)."""
        return self.tau / self.gamma
```

#### Usage

```python
from netsubgraph.core.sir import SIRConfig

# Default parameters (τ=1, γ=1)
config = SIRConfig()

# Custom parameters
config = SIRConfig(tau=2.0, gamma=0.5, t_end=20.0)
print(config.R0)  # 4.0

# Immutable - this raises FrozenInstanceError
config.tau = 3.0  # Error!
```

---

### Connection Algorithms

```python
# netsubgraph/plugins/base.py (continued)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import networkx as nx


class ConnectionAlgorithm(ABC):
    """
    Base class for connection algorithm plugins.

    Connection algorithms take allocated hyperstubs and connect them
    to form the final network. Different algorithms trade off speed
    vs. sampling bias.
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""

    # Characteristics for documentation
    is_unbiased: ClassVar[bool] = False
    relative_speed: ClassVar[str] = "medium"  # "fast", "medium", "slow"

    @abstractmethod
    def connect(
        self,
        hyperstub_bins: dict[str, list[int]],
        motifs: list[Motif],
        existing_edges: set[tuple[int, int]],
        rng: np.random.Generator,
    ) -> set[tuple[int, int]]:
        """
        Connect hyperstubs to form edges.

        Args:
            hyperstub_bins: Maps motif corner identifiers to lists of node IDs
            motifs: List of motif definitions being used
            existing_edges: Edges already in the network (to avoid duplicates)
            rng: Random number generator for reproducibility

        Returns:
            Set of (node_i, node_j) tuples representing new edges.
        """
        ...

    def validate(self) -> list[str]:
        return []


def register_connector(name: str):
    """Decorator to register a connection algorithm plugin."""
    def decorator(cls: type[ConnectionAlgorithm]) -> type[ConnectionAlgorithm]:
        cls.name = name
        from netsubgraph.registry import Registry
        Registry.register_connector(cls)
        return cls
    return decorator
```

#### Built-in Connection Algorithms

```python
# netsubgraph/plugins/connectors/builtins.py

from netsubgraph.plugins.base import ConnectionAlgorithm, register_connector


@register_connector("repeated")
class RepeatedConnector(ConnectionAlgorithm):
    """
    Fast connection with resampling on collision.

    When a self-loop or multi-edge is detected, returns nodes to bins
    and resamples. Fast but introduces slight bias toward configurations
    that avoid collisions.
    """

    description = "Fast connection with collision resampling (slight bias)"
    is_unbiased = False
    relative_speed = "fast"

    def connect(self, hyperstub_bins, motifs, existing_edges, rng):
        # Implementation follows MATLAB Connect_repeated.m
        ...


@register_connector("refuse")
class RefuseConnector(ConnectionAlgorithm):
    """
    Unbiased connection via rejection sampling.

    If any collision is detected in a batch, rejects the entire
    configuration and starts over. Guarantees uniform sampling
    but can be slow for dense networks.
    """

    description = "Unbiased rejection sampling (slower)"
    is_unbiased = True
    relative_speed = "slow"

    def connect(self, hyperstub_bins, motifs, existing_edges, rng):
        # Implementation follows MATLAB Connect_refuse.m
        ...


@register_connector("erased")
class ErasedConnector(ConnectionAlgorithm):
    """
    Fastest connection with post-hoc edge removal.

    Connects all hyperstubs, then removes self-loops and duplicate
    edges. Very fast but most biased—use only when bias is acceptable.
    """

    description = "Fastest connection, removes invalid edges after (most bias)"
    is_unbiased = False
    relative_speed = "fast"

    def connect(self, hyperstub_bins, motifs, existing_edges, rng):
        # Implementation follows MATLAB Connect_erased.m
        ...
```

---

### Distribution Plugins

```python
# netsubgraph/plugins/base.py (continued)

class DegreeDistribution(ABC):
    """
    Base class for degree distribution plugins.

    Generates degree sequences for network generation.
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.int_]:
        """
        Sample a degree sequence.

        Args:
            n: Number of nodes
            rng: Random number generator

        Returns:
            Array of non-negative integers of length n.
        """
        ...

    @abstractmethod
    def expected_mean(self) -> float:
        """Expected mean degree."""
        ...

    def validate(self) -> list[str]:
        return []


def register_distribution(name: str):
    """Decorator to register a degree distribution plugin."""
    def decorator(cls: type[DegreeDistribution]) -> type[DegreeDistribution]:
        cls.name = name
        from netsubgraph.registry import Registry
        Registry.register_distribution(cls)
        return cls
    return decorator
```

#### Built-in Distributions

```python
# netsubgraph/plugins/distributions/builtins.py

from netsubgraph.plugins.base import DegreeDistribution, register_distribution


@register_distribution("poisson")
class PoissonDistribution(DegreeDistribution):
    """Poisson degree distribution."""

    description = "Poisson distribution with mean λ"

    def __init__(self, lam: float):
        self.lam = lam

    def sample(self, n, rng):
        return rng.poisson(self.lam, size=n)

    def expected_mean(self):
        return self.lam


@register_distribution("power_law")
class PowerLawDistribution(DegreeDistribution):
    """Power-law (scale-free) degree distribution."""

    description = "Power-law distribution P(k) ~ k^(-α)"

    def __init__(self, alpha: float, k_min: int = 1, k_max: int = 100):
        self.alpha = alpha
        self.k_min = k_min
        self.k_max = k_max

    def sample(self, n, rng):
        # Discrete power-law sampling
        ks = np.arange(self.k_min, self.k_max + 1)
        probs = ks.astype(float) ** (-self.alpha)
        probs /= probs.sum()
        return rng.choice(ks, size=n, p=probs)

    def expected_mean(self):
        ks = np.arange(self.k_min, self.k_max + 1)
        probs = ks.astype(float) ** (-self.alpha)
        probs /= probs.sum()
        return float(np.sum(ks * probs))


@register_distribution("fixed")
class FixedDistribution(DegreeDistribution):
    """Fixed (regular) degree distribution."""

    description = "All nodes have the same degree k"

    def __init__(self, k: int):
        self.k = k

    def sample(self, n, rng):
        return np.full(n, self.k, dtype=int)

    def expected_mean(self):
        return float(self.k)


@register_distribution("empirical")
class EmpiricalDistribution(DegreeDistribution):
    """Empirical distribution from observed data."""

    description = "Sample with replacement from provided degree sequence"

    def __init__(self, degrees: list[int]):
        self.degrees = np.array(degrees)

    def sample(self, n, rng):
        return rng.choice(self.degrees, size=n, replace=True)

    def expected_mean(self):
        return float(np.mean(self.degrees))
```

---

## Plugin Registry

```python
# netsubgraph/registry.py

from typing import TypeVar, Generic
from importlib import import_module
from pathlib import Path
import sys


T = TypeVar("T")


class PluginCollection(Generic[T]):
    """Type-safe collection of registered plugins."""

    def __init__(self, plugin_type: type[T], type_name: str):
        self._plugins: dict[str, type[T]] = {}
        self._plugin_type = plugin_type
        self._type_name = type_name

    def register(self, cls: type[T]) -> None:
        """Register a plugin class."""
        name = cls.name

        # Validate before registration
        instance = cls() if not hasattr(cls, '__init__') or cls.__init__ is object.__init__ else None
        if instance:
            errors = instance.validate()
            if errors:
                raise ValueError(
                    f"Invalid {self._type_name} plugin '{name}':\n" +
                    "\n".join(f"  - {e}" for e in errors)
                )

        if name in self._plugins:
            raise ValueError(
                f"{self._type_name} '{name}' is already registered. "
                f"Choose a different name or unregister the existing plugin."
            )

        self._plugins[name] = cls

    def get(self, name: str) -> type[T]:
        """Get a plugin class by name."""
        if name not in self._plugins:
            available = ", ".join(sorted(self._plugins.keys()))
            raise KeyError(
                f"Unknown {self._type_name} '{name}'. "
                f"Available: {available}"
            )
        return self._plugins[name]

    def list(self) -> list[str]:
        """List all registered plugin names."""
        return sorted(self._plugins.keys())

    def items(self) -> list[tuple[str, type[T]]]:
        """Return all (name, class) pairs."""
        return sorted(self._plugins.items())

    def unregister(self, name: str) -> None:
        """Unregister a plugin (useful for testing)."""
        if name in self._plugins:
            del self._plugins[name]


class Registry:
    """
    Central registry for all plugin types.

    Usage:
        from netsubgraph.registry import Registry

        # Register (usually via decorators)
        Registry.register_motif(MyMotifClass)

        # Retrieve
        TriangleClass = Registry.motifs.get("triangle")

        # List available
        print(Registry.motifs.list())
    """

    motifs: PluginCollection["Motif"] = PluginCollection(None, "motif")  # type: ignore
    models: PluginCollection["EpidemicModel"] = PluginCollection(None, "epidemic model")  # type: ignore
    connectors: PluginCollection["ConnectionAlgorithm"] = PluginCollection(None, "connector")  # type: ignore
    distributions: PluginCollection["DegreeDistribution"] = PluginCollection(None, "distribution")  # type: ignore

    @classmethod
    def register_motif(cls, motif_cls: type) -> None:
        cls.motifs.register(motif_cls)

    @classmethod
    def register_model(cls, model_cls: type) -> None:
        cls.models.register(model_cls)

    @classmethod
    def register_connector(cls, connector_cls: type) -> None:
        cls.connectors.register(connector_cls)

    @classmethod
    def register_distribution(cls, dist_cls: type) -> None:
        cls.distributions.register(dist_cls)

    @classmethod
    def load_plugin_module(cls, module_path: str) -> None:
        """
        Import a module to trigger its @register_* decorators.

        Args:
            module_path: Dotted module path (e.g., "my_research.motifs")
        """
        import_module(module_path)

    @classmethod
    def load_plugin_file(cls, file_path: str | Path) -> None:
        """
        Load plugins from a Python file.

        Args:
            file_path: Path to .py file containing plugin definitions
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Plugin file not found: {path}")

        # Add parent directory to path temporarily
        parent = str(path.parent.absolute())
        sys.path.insert(0, parent)
        try:
            module_name = path.stem
            import_module(module_name)
        finally:
            sys.path.remove(parent)

    @classmethod
    def discover_plugins(cls, package: str = "netsubgraph.plugins") -> None:
        """
        Auto-discover and load all built-in plugins.

        Called automatically on library import.
        """
        import_module(f"{package}.motifs.builtins")
        import_module(f"{package}.models.builtins")
        import_module(f"{package}.connectors.builtins")
        import_module(f"{package}.distributions.builtins")


# Auto-discover built-in plugins on import
def _init_registry():
    Registry.discover_plugins()

_init_registry()
```

---

## Configuration Schema

Configuration files use YAML with Pydantic validation.

### Schema Definition

```python
# netsubgraph/config/schema.py

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Any
from pathlib import Path


class SubgraphSpec(BaseModel):
    """Specification for a subgraph type in the network."""

    type: str = Field(description="Motif name (e.g., 'triangle')")
    count_per_node: float = Field(
        ge=0,
        description="Average number of this subgraph per node"
    )

    @field_validator("type")
    @classmethod
    def validate_motif_exists(cls, v: str) -> str:
        from netsubgraph.registry import Registry
        # Validation happens after plugins loaded
        if v not in Registry.motifs.list():
            available = ", ".join(Registry.motifs.list())
            raise ValueError(f"Unknown motif '{v}'. Available: {available}")
        return v


class DistributionSpec(BaseModel):
    """Specification for a probability distribution."""

    type: str = Field(description="Distribution name")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Distribution parameters"
    )

    @field_validator("type")
    @classmethod
    def validate_distribution_exists(cls, v: str) -> str:
        from netsubgraph.registry import Registry
        if v not in Registry.distributions.list():
            available = ", ".join(Registry.distributions.list())
            raise ValueError(f"Unknown distribution '{v}'. Available: {available}")
        return v


class NetworkConfig(BaseModel):
    """Network generation configuration."""

    nodes: int = Field(ge=1, description="Number of nodes")

    degree_distribution: DistributionSpec = Field(
        description="Degree distribution specification"
    )

    subgraphs: list[SubgraphSpec] = Field(
        default_factory=list,
        description="List of subgraph specifications"
    )

    algorithm: Literal["cma", "uda"] = Field(
        default="cma",
        description="Network generation algorithm"
    )

    connector: str = Field(
        default="repeated",
        description="Connection algorithm for hyperstubs"
    )

    ensure_connected: bool = Field(
        default=False,
        description="Retry generation until network is connected"
    )

    max_retries: int = Field(
        default=100,
        ge=1,
        description="Maximum generation attempts if ensure_connected=True"
    )


class SIRConfigInput(BaseModel):
    """SIR configuration for YAML/TOML parsing. Converts to core.SIRConfig."""

    tau: float = Field(
        default=1.0,
        gt=0,
        description="Transmission rate per S-I edge per unit time"
    )

    gamma: float = Field(
        default=1.0,
        gt=0,
        description="Recovery rate per infected per unit time"
    )

    initial_infected: float = Field(
        default=0.0001,
        ge=0,
        le=1,
        description="Initial fraction of infected nodes"
    )

    t_end: float = Field(
        default=15.0,
        gt=0,
        description="Simulation end time"
    )

    method: Literal["ode", "stochastic", "both"] = Field(
        default="both",
        description="Simulation method"
    )

    realizations: int = Field(
        default=100,
        ge=1,
        description="Number of stochastic simulation runs"
    )

    def to_domain(self) -> "SIRConfig":
        """Convert to immutable domain object."""
        from netsubgraph.core.sir import SIRConfig
        return SIRConfig(
            tau=self.tau,
            gamma=self.gamma,
            initial_infected=self.initial_infected,
            t_end=self.t_end,
            method=self.method,
            realizations=self.realizations,
        )


class OutputConfig(BaseModel):
    """Output and analysis configuration."""

    directory: Path = Field(
        default=Path("./output"),
        description="Output directory for results"
    )

    save_networks: bool = Field(
        default=False,
        description="Save generated networks to files"
    )

    network_format: Literal["edgelist", "graphml", "gexf", "adjlist"] = Field(
        default="edgelist",
        description="Network file format"
    )

    trajectories: bool = Field(
        default=True,
        description="Save S(t), I(t), R(t) trajectories"
    )

    final_size_distribution: bool = Field(
        default=True,
        description="Compute final epidemic size distribution"
    )

    network_metrics: list[str] = Field(
        default_factory=lambda: ["clustering", "degree_distribution"],
        description="Network metrics to compute"
    )

    figures: bool = Field(
        default=True,
        description="Generate publication-ready figures"
    )

    figure_format: Literal["png", "pdf", "svg"] = Field(
        default="pdf",
        description="Figure file format"
    )


class ExperimentConfig(BaseModel):
    """
    Top-level experiment configuration.

    This is the root schema for YAML configuration files.
    """

    name: str = Field(
        default="experiment",
        description="Experiment name (used for output filenames)"
    )

    description: str = Field(
        default="",
        description="Human-readable experiment description"
    )

    plugins: list[str] = Field(
        default_factory=list,
        description="Plugin modules to load before running"
    )

    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility (None = random)"
    )

    network: NetworkConfig

    sir: SIRConfigInput = Field(
        default_factory=SIRConfigInput
    )

    output: OutputConfig = Field(
        default_factory=OutputConfig
    )

    @model_validator(mode="after")
    def load_plugins(self) -> "ExperimentConfig":
        """Load plugin modules specified in config."""
        from netsubgraph.registry import Registry
        for plugin_path in self.plugins:
            Registry.load_plugin_module(plugin_path)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def from_toml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from TOML file."""
        import tomllib
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
```

### Example Configuration Files

#### Minimal Example

```yaml
# minimal.yaml
network:
  nodes: 1000
  degree_distribution:
    type: poisson
    params:
      lam: 6
```

#### Full Example (Paper 1 Reproduction)

```yaml
# paper1_figure3.yaml
name: paper1_fig3_clustering_effect
description: |
  Reproduce Figure 3 from "Higher-order structure and epidemic dynamics
  in clustered networks" (2014). Compares epidemic trajectories across
  networks with varying clustering coefficients.

seed: 42

network:
  nodes: 10000
  degree_distribution:
    type: poisson
    params:
      lam: 6
  subgraphs:
    - type: triangle
      count_per_node: 2.0
  algorithm: cma
  connector: refuse  # Unbiased for publication
  ensure_connected: true

sir:
  tau: 1.0
  gamma: 1.0
  initial_infected: 0.0001
  t_end: 15.0
  method: both
  realizations: 100

output:
  directory: ./results/paper1/fig3
  save_networks: true
  network_format: edgelist
  trajectories: true
  final_size_distribution: true
  network_metrics:
    - clustering
    - degree_distribution
    - triangle_count
  figures: true
  figure_format: pdf
```

#### Custom Motif Example

```yaml
# custom_motif_experiment.yaml
name: hexagonal_cage_dynamics
description: Testing epidemic dynamics on networks with custom hexagonal cage motifs

plugins:
  - my_research.custom_motifs  # Loads custom plugin module

network:
  nodes: 5000
  degree_distribution:
    type: power_law
    params:
      alpha: 2.5
      k_min: 2
      k_max: 50
  subgraphs:
    - type: triangle
      count_per_node: 1.0
    - type: hexagonal_cage  # Custom motif from plugin
      count_per_node: 0.2

sir:
  method: ode  # ODE only for speed
```

#### Parameter Sweep Example

```yaml
# sweep_template.yaml
name: clustering_sweep_{lambda}
description: Parameter sweep over triangle density

network:
  nodes: 5000
  degree_distribution:
    type: poisson
    params:
      lam: 6
  subgraphs:
    - type: triangle
      count_per_node: ${lambda}  # Templated value

sir:
  method: both
  realizations: 50
```

```python
# Running a parameter sweep
from netsubgraph import Experiment

for lam in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    exp = Experiment.from_yaml(
        "sweep_template.yaml",
        overrides={"network.subgraphs.0.count_per_node": lam}
    )
    exp.name = f"clustering_sweep_{lam}"
    exp.run()
```

---

## Validation & Error Handling

### Error Hierarchy

```python
# netsubgraph/exceptions.py

class NetSubgraphError(Exception):
    """Base exception for all library errors."""
    pass


class ConfigurationError(NetSubgraphError):
    """Invalid configuration file or parameters."""
    pass


class PluginError(NetSubgraphError):
    """Error in plugin definition or loading."""
    pass


class GenerationError(NetSubgraphError):
    """Network generation failed."""
    pass


class InfeasibleConfigError(GenerationError):
    """
    Requested configuration is mathematically infeasible.

    Examples:
        - Total subgraph degree exceeds node degree
        - Odd total degree (can't form edges)
    """
    pass


class SimulationError(NetSubgraphError):
    """Epidemic simulation failed."""
    pass
```

### Validation Examples

```python
# Good error messages

# Missing motif
>>> exp = Experiment.from_yaml("config.yaml")
ConfigurationError: Unknown motif 'triangel' in network.subgraphs[0].type
  Did you mean: 'triangle'?
  Available motifs: edge, hexagon, pentagon, square, square_cycle, toast, triangle

# Infeasible configuration
>>> exp.run()
InfeasibleConfigError: Cannot generate network with requested configuration.
  - Mean degree from distribution: 6.0
  - Total degree required by subgraphs: 8.5
    - triangle (count_per_node=2.0): contributes 4.0 edges per node
    - square (count_per_node=1.5): contributes 4.5 edges per node
  Reduce subgraph counts or increase mean degree.

# Plugin validation
>>> @register_motif("bad_motif")
... class BadMotif(Motif):
...     adjacency = [[0, 1], [0, 0]]  # Not symmetric
...     corner_types = [1, 1]
PluginError: Invalid motif plugin 'bad_motif':
  - Adjacency matrix must be symmetric
```

---

## Example Workflows

### Workflow 1: Quick Start (Python API)

```python
from netsubgraph import Network, SIR

# Generate network with triangles
net = Network.generate(
    nodes=1000,
    degree_distribution="poisson(6)",
    subgraphs={"triangle": 2.0}
)

# Run epidemic
result = SIR(net, tau=1, gamma=1).run()

# Plot
result.plot()
```

### Workflow 2: Configuration-Driven

```bash
# Generate and simulate
$ netsubgraph run experiment.yaml

# List available components
$ netsubgraph list-motifs
$ netsubgraph list-models

# Validate config without running
$ netsubgraph validate experiment.yaml

# Parameter sweep
$ netsubgraph sweep experiment.yaml \
    --param network.subgraphs.0.count_per_node \
    --values 0.5,1.0,1.5,2.0 \
    --parallel 4
```

### Workflow 3: Custom Motifs

```python
# my_research/motifs.py

from netsubgraph.core.motif import Motif
from netsubgraph.library.registry import register

# House motif: square with a triangle on top
#
#       0
#      / \
#     1---3
#     |   |
#     2---4
#
# Hamiltonian cycle: 0 → 1 → 2 → 4 → 3 → 0
#
register("house", Motif.create([
    [0, 1, 0, 1, 0],  # 0 connects to 1, 3
    [1, 0, 1, 1, 0],  # 1 connects to 0, 2, 3
    [0, 1, 0, 0, 1],  # 2 connects to 1, 4
    [1, 1, 0, 0, 1],  # 3 connects to 0, 1, 4
    [0, 0, 1, 1, 0],  # 4 connects to 2, 3
]))
```

```yaml
# Using the custom motif
plugins:
  - my_research.motifs

network:
  nodes: 2000
  degree_distribution:
    type: poisson
    params:
      lam: 8
  subgraphs:
    - type: triangle
      count_per_node: 1.0
    - type: bowtie
      count_per_node: 0.3
```

### Workflow 4: Reproduce Paper Figure

```python
# reproduce_paper1_fig3.py

from netsubgraph import Experiment
from netsubgraph.reproduce import Paper1

# Built-in reproduction script
Paper1.figure3(
    output_dir="./results/paper1",
    seed=42
)

# Or customize
exp = Experiment.from_yaml("paper1_fig3.yaml")
exp.network.nodes = 20000  # Larger for smoother curves
results = exp.run()
results.save()
results.plot(style="publication")
```

---

## Key Design Decisions

### Motif = Adjacency Matrix (Frozen Dataclass)

A motif is a validated, immutable dataclass. Creation via factory method:

```python
triangle = Motif.create([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
```

All properties are computed at creation time and frozen:

| Property | Derivation |
|----------|------------|
| `corner_types` | Nodes with the same degree get the same type |
| `cardinalities` | Degree of each corner type within the motif |
| `num_nodes`, `num_edges` | Computed from adjacency dimensions and sum |
| `is_complete` | True if all nodes have the same degree |

**Why this works**: For the CMA algorithm, corner types determine *cardinality* (how much degree a subgraph participation consumes). Nodes with the same degree within the motif have the same cardinality, so they're equivalent for allocation purposes.

**Ordering is preserved**: `corner_types[i]` corresponds to row `i` of the adjacency matrix, ensuring correct wiring during the connection phase.

### Immutable Domain Objects

All domain types use `@dataclass(frozen=True)`:

```python
@dataclass(frozen=True, slots=True)
class Motif: ...

@dataclass(frozen=True, slots=True)
class SIRConfig: ...

@dataclass(frozen=True, slots=True)
class ODEResult:
    t: NDArray[np.float64]
    S: NDArray[np.float64]
    I: NDArray[np.float64]
    R: NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class StochasticResult:
    t: NDArray[np.float64]
    S_mean: NDArray[np.float64]
    I_mean: NDArray[np.float64]
    R_mean: NDArray[np.float64]
    final_sizes: NDArray[np.float64]
```

Pydantic models are used **only** at system boundaries (YAML/TOML parsing) and
convert to frozen dataclasses immediately.

### Parallelism for Monte Carlo

Stochastic simulation is CPU-bound. Use `multiprocessing` for parallel
ensemble runs:

```python
from concurrent.futures import ProcessPoolExecutor
from netsubgraph.simulation import simulate_single

def simulate_ensemble(
    network: nx.Graph,
    config: SIRConfig,
    n_workers: int = 4,
) -> StochasticResult:
    """Run stochastic ensemble in parallel."""
    seeds = [rng.integers(2**32) for _ in range(config.realizations)]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(
            lambda s: simulate_single(network, config, seed=s),
            seeds,
        ))

    return aggregate_results(results)
```

No async is needed — all work is CPU-bound (matrix operations, ODE solving,
Gillespie algorithm steps).

---

## Summary

This design provides:

1. **Immutable domain objects** - Frozen dataclasses for `Motif`, `SIRConfig`, results
2. **Validation at boundaries** - Pydantic parses config, converts to domain types
3. **Derived properties** - Corner types, cardinalities computed at creation
4. **Flexible configuration** via YAML/TOML with schema validation
5. **Progressive disclosure** - Simple things easy, complex things possible
6. **Extensibility** - Register custom motifs with one line
7. **Reproducibility** through declarative configs and seeding
8. **Parallelism** - Multiprocessing for Monte Carlo simulations

The registry handles "what can I use", the config handles "what do I want to do".
