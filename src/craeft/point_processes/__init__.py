"""Point process simulation framework.

Generic continuous-time Markov chain simulation using the Gillespie
algorithm. Process-specific dynamics (SIR epidemics, Hawkes processes,
etc.) implement the ``ContinuousTimeProcess`` interface.

Submodules:
    craeft.point_processes.process    - Core abstractions (ABC, Trajectory)
    craeft.point_processes.gillespie  - Gillespie engine and ensemble
    craeft.point_processes.epidemics  - Epidemic models (SIR, etc.)
"""

from craeft.point_processes.gillespie import (
    ConvergenceInfo,
    ConvergenceMonitor,
    EnsembleResult,
    ProgressCallback,
    run_once,
    simulate,
)
from craeft.point_processes.process import (
    ContinuousTimeProcess,
    ConvergenceConfig,
    ProcessFactory,
    Trajectory,
)

__all__ = [
    # Abstractions
    "ContinuousTimeProcess",
    "ProcessFactory",
    "Trajectory",
    "ConvergenceConfig",
    # Engine
    "ConvergenceInfo",
    "ConvergenceMonitor",
    "EnsembleResult",
    "ProgressCallback",
    "run_once",
    "simulate",
]
