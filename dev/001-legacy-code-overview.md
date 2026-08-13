# Codebase Overview: Network Simulation Tools

## Summary
This codebase implements network generation algorithms and epidemic modeling tools developed across three research papers (2014-2016). The code enables generation of complex networks with prescribed degree sequences and subgraph compositions, plus deterministic epidemic modeling using probability generating functions (PGF).

## Research Papers
1. **2014** - "Higher-order structure and epidemic dynamics in clustered networks"
2. **2015** - "Generation and analysis of networks with a prescribed degree sequence and subgraph family"
3. **2016** - "Beyond clustering: mean-field dynamics on networks with arbitrary subgraph composition"

## Code Structure (~2,500 lines)

### 1. Network Generation Algorithms (`./matlab/network_generation_algorithms/`)

**Core Algorithms:**
- **CMA.m** (178 lines) - Cardinality Matching Algorithm
  - Generates networks with prescribed degree and subgraph sequences
  - Handles both complete and incomplete subgraphs (e.g., triangles, squares, "toast" motifs)
  - Uses multinomial distribution to allocate hyperstubs for incomplete subgraphs
  - Intelligent stub matching based on cardinality constraints

- **UDA.m** (88 lines) - Underdetermined Diophantine Algorithm
  - Alternative generation method using Diophantine equations
  - Finds all valid stub configurations for each degree
  - Random selection from solution space ensures proper distribution

- **CMA.py** (237 lines) - Python implementation of CMA
  - NetworkX-based implementation
  - Includes connection process with artifact edge prevention
  - Auto-retry logic for difficult configurations

**Connection Procedures:**
- **Connect_repeated.m** - Fast biased sampling (accepts repeated edges temporarily)
- **Connect_refuse.m** - Unbiased but slower (refuses invalid configurations)
- **Connect_erased.m** - Removes invalid edges post-connection

**Support Functions:**
- **clustering.m** - Computes global clustering coefficient via matrix powers
- **dio_recur.m** - Recursive Diophantine equation solver for UDA
- **subGraph.py** - Python class for subgraph representation

### 2. PGF-Based ODE Generation (`./matlab/pgf_odes/`)

**Core System:**
- **PGF_equation_generator.m** (426 lines) - Main symbolic code generator
  - Takes expected subgraph counts (λ) and subgraph types as input
  - Generates MATLAB functions dynamically using symbolic math toolbox
  - Creates: `x_alpha.m` (initial conditions), `x_equations.m` (state transitions), `func.m` (ODE system)
  - Computes PGF Jacobian and Hessian for excess degree calculations
  - Handles arbitrary subgraph compositions (lines, triangles, custom motifs)
  - SIR epidemic model: S→I (infection rate τ), I→R (recovery rate γ)

- **episolve.m** (17 lines) - ODE45 wrapper for solving generated system
  - Returns S(t), I(t), R(t) trajectories

**Support Functions:**
- **combinator.m** - Generates all possible SIR state combinations for subgraphs
- **inf_neighbors.m** - Counts infectious neighbors for transition rates

### 3. Key Concepts

**Hyperstubs:** Generalization of stubs for higher-order structures
- Each hyperstub represents a node's position in a subgraph
- Cardinality = number of classical edges the hyperstub contributes
- Incomplete subgraphs have multiple corner types (e.g., "toast" has 2 types)

**Subgraph Sequences:**
- Specify how many times each node participates in each subgraph type
- CMA requires sequences as input; UDA generates them from degree sequence
- Must satisfy balance conditions (e.g., triangles need 3× corner count)

**Connection Algorithms:**
- Form hyperstub bins and sample combinations
- Check for self-loops and multi-edges
- Different trade-offs between speed and bias

**PGF Approach:**
- Tracks probability distributions of subgraph states (all S, mixed SI, etc.)
- Survivor functions (θ) represent infection-free probability
- Mean-field closure via excess degree distributions
- Validates against stochastic simulations

## Dependencies
- MATLAB Symbolic Math Toolbox (for PGF code generation)
- Python: NetworkX, NumPy (for Python CMA)
- `subgraphs.mat` - Library of predefined subgraph adjacency matrices

## Typical Workflows

**Generate network with clustering:**
```matlab
D = ones(1,500)*4;  % Degree sequence
[Sd, sg] = CMA(D, 'ones(1,500)', 'C3');  % 1 triangle per node
[A, Sd] = Connect_repeated(Sd, sg);
```

**Epidemic modeling:**
```matlab
PGF_equation_generator([2 2], 'C2', 'C3');  % λ=[2,2] for edges and triangles
[S, I, R, T] = episolve();
```
