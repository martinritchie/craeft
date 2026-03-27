# SIR Epidemics

Susceptible-Infected-Recovered model as a concrete `ContinuousTimeProcess`
implementation.

## Configuration

::: craeft.point_processes.epidemics.sir
    options:
      members:
        - SIRConfig
        - SIRProcess
        - SIRProcessFactory

## Simulator

::: craeft.point_processes.epidemics.simulator
    options:
      members:
        - EpidemicSimulator
        - SIRSimulator
