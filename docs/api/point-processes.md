# Point Processes

The generic simulation framework: core abstractions and the Gillespie engine.

## Core abstractions

::: craeft.point_processes.process
    options:
      members:
        - Trajectory
        - ConvergenceConfig
        - ContinuousTimeProcess
        - ProcessFactory

## Gillespie engine

::: craeft.point_processes.gillespie
    options:
      members:
        - run_once
        - simulate
        - EnsembleResult
        - ConvergenceInfo
        - ConvergenceMonitor
        - ProgressCallback
