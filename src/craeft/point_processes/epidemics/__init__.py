"""Epidemic models as continuous-time point processes.

Submodules:
    craeft.point_processes.epidemics.sir       - SIR model
    craeft.point_processes.epidemics.simulator  - Simulator protocol
"""

from craeft.point_processes.epidemics.simulator import (
    EpidemicSimulator,
    SIRSimulator,
)
from craeft.point_processes.epidemics.sir import (
    SIRConfig,
    SIRProcess,
    SIRProcessFactory,
)

__all__ = [
    "SIRConfig",
    "SIRProcess",
    "SIRProcessFactory",
    "EpidemicSimulator",
    "SIRSimulator",
]
