"""Tests for stochastic SIR simulation via the generic Gillespie engine."""

import numpy as np
from scipy.sparse import csr_array

from craeft.point_processes.epidemics.sir import (
    I,
    S,
    SIRConfig,
    SIRProcess,
    SIRProcessFactory,
)
from craeft.point_processes.gillespie import (
    ConvergenceInfo,
    ConvergenceMonitor,
    EnsembleResult,
    run_once,
    simulate,
)
from craeft.point_processes.process import ConvergenceConfig, Trajectory


def _make_factory(
    adjacency: csr_array,
    *,
    tau: float = 1.0,
    gamma: float = 1.0,
    t_end: float = 5.0,
    initial_infected: int = 1,
    filter_subcritical: bool = False,
    convergence_threshold: float = 0.02,
    min_realizations: int = 30,
    max_realizations: int = 10000,
    num_workers: int = 1,
) -> SIRProcessFactory:
    """Convenience builder for tests."""
    return SIRProcessFactory(
        config=SIRConfig(
            tau=tau,
            gamma=gamma,
            initial_infected=initial_infected,
            filter_subcritical=filter_subcritical,
        ),
        adjacency=adjacency,
        _convergence_config=ConvergenceConfig(
            t_end=t_end,
            convergence_threshold=convergence_threshold,
            min_realizations=min_realizations,
            max_realizations=max_realizations,
            num_workers=num_workers,
        ),
    )


class TestTrajectory:
    """Tests for generic Trajectory dataclass."""

    def test_getitem_accesses_compartment(self) -> None:
        trajectory = Trajectory(
            t=np.array([0.0, 1.0, 2.0]),
            compartments={
                "s": np.array([9, 8, 8]),
                "i": np.array([1, 2, 1]),
                "r": np.array([0, 0, 1]),
            },
        )
        assert trajectory["r"][-1] == 1

    def test_names_returns_compartment_names(self) -> None:
        trajectory = Trajectory(
            t=np.array([0.0, 1.0]),
            compartments={
                "s": np.array([9, 8]),
                "i": np.array([1, 2]),
                "r": np.array([0, 0]),
            },
        )
        assert trajectory.names == ["s", "i", "r"]

    def test_duration_is_last_time(self) -> None:
        trajectory = Trajectory(
            t=np.array([0.0, 1.5, 3.7]),
            compartments={"s": np.array([9, 8, 8])},
        )
        assert trajectory.t[-1] == 3.7


class TestEnsembleResult:
    """Tests for EnsembleResult dataclass."""

    def test_scalar_output_mean(self) -> None:
        convergence = ConvergenceInfo(
            converged=True, n_realizations=3, relative_sem=0.01, threshold=0.02
        )
        result = EnsembleResult(
            t=np.array([0.0, 1.0]),
            means={"s": np.array([10.0, 5.0]), "i": np.array([0.0, 2.0])},
            stds={"s": np.array([0.0, 1.0]), "i": np.array([0.0, 0.5])},
            scalar_outputs=np.array([5.0, 7.0, 6.0]),
            convergence=convergence,
        )
        assert result.scalar_output_mean == 6.0

    def test_scalar_output_std(self) -> None:
        convergence = ConvergenceInfo(
            converged=True, n_realizations=3, relative_sem=0.0, threshold=0.02
        )
        result = EnsembleResult(
            t=np.array([0.0, 1.0]),
            means={"s": np.array([10.0, 5.0])},
            stds={"s": np.array([0.0, 1.0])},
            scalar_outputs=np.array([10.0, 10.0, 10.0]),
            convergence=convergence,
        )
        assert result.scalar_output_std == 0.0


class TestSIRProcess:
    """Tests for SIRProcess state management."""

    def test_initial_state_counts(self) -> None:
        config = SIRConfig(tau=1.0, gamma=1.0, initial_infected=2)
        adjacency = csr_array(np.zeros((10, 10)))
        process = SIRProcess(config, adjacency)
        # Check via rates — 2 infected means 2 non-zero recovery rates
        rates = process.rates()
        n = 10
        recovery_rates = rates[n:]
        assert np.sum(recovery_rates > 0) == 2

    def test_initial_agent_states(self) -> None:
        config = SIRConfig(tau=1.0, gamma=1.0, initial_infected=2)
        adjacency = csr_array(np.zeros((5, 5)))
        process = SIRProcess(config, adjacency)
        assert process._agent_state[0] == I
        assert process._agent_state[1] == I
        assert process._agent_state[2] == S
        assert process._agent_state[3] == S
        assert process._agent_state[4] == S

    def test_execute_infection_event(self) -> None:
        config = SIRConfig(tau=1.0, gamma=1.0, initial_infected=1)
        adjacency = csr_array(np.zeros((10, 10)))
        process = SIRProcess(config, adjacency)
        # Execute infection event on node 5 (event_index < n)
        process.execute(5, 0.5)
        assert process._agent_state[5] == I
        assert process.time == 0.5

    def test_execute_recovery_event(self) -> None:
        config = SIRConfig(tau=1.0, gamma=1.0, initial_infected=2)
        adjacency = csr_array(np.zeros((10, 10)))
        process = SIRProcess(config, adjacency)
        # Execute recovery event on node 0 (event_index = n + 0)
        process.execute(10, 1.0)
        assert process._agent_state[0] == 2  # R
        assert process.time == 1.0

    def test_pressure_on_chain_graph(self, chain_graph: csr_array) -> None:
        """
        Chain: 0 - 1 - 2 - 3
        If node 1 is infected, nodes 0 and 2 have pressure tau.
        """
        config = SIRConfig(tau=0.5, gamma=1.0, initial_infected=1)
        process = SIRProcess(config, chain_graph)
        # Reset to specific state: only node 1 infected
        process._agent_state[:] = S
        process._agent_state[1] = I

        rates = process.rates()
        n = 4
        infection_rates = rates[:n]

        assert infection_rates[0] == 0.5  # Node 0 neighbours infected node 1
        assert infection_rates[1] == 0.0  # Node 1 is infected, not susceptible
        assert infection_rates[2] == 0.5  # Node 2 neighbours infected node 1
        assert infection_rates[3] == 0.0  # Node 3 has no infected neighbours

    def test_pressure_with_multiple_infected_neighbours(
        self, complete_graph_k4: csr_array
    ) -> None:
        """
        K4: all nodes connected.
        If nodes 1 and 2 are infected, node 0 has pressure 2*tau.
        """
        config = SIRConfig(tau=0.5, gamma=1.0, initial_infected=2)
        process = SIRProcess(config, complete_graph_k4)
        process._agent_state[:] = S
        process._agent_state[1] = I
        process._agent_state[2] = I

        rates = process.rates()
        n = 4
        infection_rates = rates[:n]

        assert infection_rates[0] == 2 * 0.5  # 2 infected neighbours
        assert infection_rates[1] == 0.0  # Infected
        assert infection_rates[2] == 0.0  # Infected
        assert infection_rates[3] == 2 * 0.5  # 2 infected neighbours


class TestConvergenceMonitor:
    """Tests for ConvergenceMonitor running statistics tracker."""

    def test_variance_returns_inf_with_zero_observations(self) -> None:
        monitor = ConvergenceMonitor(threshold=0.05)
        assert monitor.variance == float("inf")

    def test_variance_returns_inf_with_single_observation(self) -> None:
        monitor = ConvergenceMonitor(threshold=0.05)
        monitor.update(10.0)
        assert monitor.variance == float("inf")

    def test_variance_finite_with_two_observations(self) -> None:
        monitor = ConvergenceMonitor(threshold=0.05)
        monitor.update(10.0)
        monitor.update(12.0)
        assert monitor.variance == 2.0  # Sample variance of [10, 12]

    def test_sem_returns_inf_with_single_observation(self) -> None:
        monitor = ConvergenceMonitor(threshold=0.05)
        monitor.update(10.0)
        assert monitor.sem == float("inf")

    def test_relative_sem_returns_inf_with_single_observation(self) -> None:
        monitor = ConvergenceMonitor(threshold=0.05)
        monitor.update(10.0)
        assert monitor.relative_sem == float("inf")

    def test_relative_sem_returns_inf_when_mean_is_zero(self) -> None:
        monitor = ConvergenceMonitor(threshold=0.05)
        monitor.update(5.0)
        monitor.update(-5.0)  # Mean becomes 0
        assert monitor.relative_sem == float("inf")

    def test_is_converged_false_below_min_n(self) -> None:
        monitor = ConvergenceMonitor(threshold=1.0)  # Very loose threshold
        for _ in range(5):
            monitor.update(10.0)  # Identical values = zero variance
        # Would converge if min_n were met, but only 5 samples
        assert not monitor.is_converged(min_n=10)

    def test_is_converged_true_when_criteria_met(self) -> None:
        monitor = ConvergenceMonitor(threshold=0.1)
        # Add many identical values for very low relative SEM
        for _ in range(100):
            monitor.update(10.0)
        assert monitor.is_converged(min_n=10)

    def test_welford_algorithm_numerical_stability(self) -> None:
        """Welford's algorithm should handle large values without overflow."""
        monitor = ConvergenceMonitor(threshold=0.05)
        large_base = 1e10
        for i in range(100):
            monitor.update(large_base + i)
        # Mean should be approximately large_base + 49.5
        assert abs(monitor.mean - (large_base + 49.5)) < 1e-6


class TestRunOnce:
    """Tests for single-realisation Gillespie execution."""

    def test_reproducible_with_same_seed(self, chain_graph: csr_array) -> None:
        factory = _make_factory(chain_graph, tau=1.0, gamma=0.5, t_end=10.0)

        traj1, _, _ = run_once(factory, 10.0, np.random.default_rng(42))
        traj2, _, _ = run_once(factory, 10.0, np.random.default_rng(42))

        np.testing.assert_array_equal(traj1.t, traj2.t)
        np.testing.assert_array_equal(traj1["s"], traj2["s"])
        np.testing.assert_array_equal(traj1["i"], traj2["i"])
        np.testing.assert_array_equal(traj1["r"], traj2["r"])

    def test_different_seeds_different_trajectories(
        self, complete_graph_k4: csr_array
    ) -> None:
        factory = _make_factory(complete_graph_k4, tau=1.0, gamma=0.5, t_end=10.0)

        traj1, _, _ = run_once(factory, 10.0, np.random.default_rng(1))
        traj2, _, _ = run_once(factory, 10.0, np.random.default_rng(2))

        trajectories_differ = len(traj1.t) != len(traj2.t) or not np.allclose(
            traj1.t, traj2.t
        )
        assert trajectories_differ

    def test_terminates_when_no_infected(self, chain_graph: csr_array) -> None:
        factory = _make_factory(chain_graph, tau=0.1, gamma=10.0, t_end=1000.0)

        traj, _, _ = run_once(factory, 1000.0, np.random.default_rng(99))

        assert traj.t[-1] < 1000.0
        assert traj["i"][-1] == 0

    def test_terminates_at_t_end(self, complete_graph_k4: csr_array) -> None:
        factory = _make_factory(complete_graph_k4, tau=10.0, gamma=0.01, t_end=0.1)

        traj, _, _ = run_once(factory, 0.1, np.random.default_rng(123))

        assert traj.t[-1] <= 0.1

    def test_returns_trajectory_with_consistent_lengths(
        self, chain_graph: csr_array
    ) -> None:
        factory = _make_factory(chain_graph, tau=1.0, gamma=1.0, t_end=5.0)

        traj, _, _ = run_once(factory, 5.0, np.random.default_rng(456))

        assert len(traj.t) == len(traj["s"])
        assert len(traj.t) == len(traj["i"])
        assert len(traj.t) == len(traj["r"])

    def test_sir_counts_sum_to_n(self, chain_graph: csr_array) -> None:
        factory = _make_factory(chain_graph, tau=1.0, gamma=1.0, t_end=5.0)

        traj, _, _ = run_once(factory, 5.0, np.random.default_rng(789))
        n = chain_graph.shape[0]

        totals = traj["s"] + traj["i"] + traj["r"]
        assert np.all(totals == n)


class TestSimulate:
    """Tests for simulate() public API."""

    def test_returns_ensemble_result(self, chain_graph: csr_array) -> None:
        factory = _make_factory(chain_graph, min_realizations=10)

        result = simulate(factory, n_points=200)

        assert isinstance(result, EnsembleResult)

    def test_respects_min_realizations(self, chain_graph: csr_array) -> None:
        factory = _make_factory(chain_graph, min_realizations=50, max_realizations=50)

        result = simulate(factory)

        assert len(result.scalar_outputs) >= 50

    def test_stops_at_max_realizations(self, chain_graph: csr_array) -> None:
        factory = _make_factory(
            chain_graph,
            convergence_threshold=0.0001,
            min_realizations=10,
            max_realizations=100,
        )

        result = simulate(factory)

        assert len(result.scalar_outputs) == 100
        assert not result.convergence.converged

    def test_convergence_metadata_populated(self, chain_graph: csr_array) -> None:
        factory = _make_factory(chain_graph, min_realizations=30)

        result = simulate(factory)

        assert result.convergence.n_realizations == len(result.scalar_outputs)
        assert result.convergence.threshold == 0.02
        assert result.convergence.relative_sem >= 0

    def test_reproducible_with_seed(self, chain_graph: csr_array) -> None:
        factory = _make_factory(chain_graph, min_realizations=30, max_realizations=30)

        result1 = simulate(factory, np.random.default_rng(42))
        result2 = simulate(factory, np.random.default_rng(42))

        np.testing.assert_array_equal(result1.scalar_outputs, result2.scalar_outputs)

    def test_time_grid_has_expected_length(self, chain_graph: csr_array) -> None:
        factory = _make_factory(chain_graph, min_realizations=10)

        result = simulate(factory, n_points=100)

        assert len(result.t) == 100

    def test_time_grid_spans_zero_to_t_end(self, chain_graph: csr_array) -> None:
        factory = _make_factory(chain_graph, t_end=10.0, min_realizations=10)

        result = simulate(factory)

        assert result.t[0] == 0.0
        assert result.t[-1] == 10.0

    def test_progress_callback_invoked(self, chain_graph: csr_array) -> None:
        factory = _make_factory(chain_graph, min_realizations=10, max_realizations=10)
        calls: list[tuple[int, int, float, bool]] = []

        def callback(n: int, max_n: int, sem: float, converged: bool) -> None:
            calls.append((n, max_n, sem, converged))

        simulate(factory, progress_callback=callback)

        assert len(calls) == 10
        assert all(c[1] == 10 for c in calls)
        assert [c[0] for c in calls] == list(range(1, 11))

    def test_parallel_execution_produces_valid_result(
        self, complete_graph_k4: csr_array
    ) -> None:
        factory = _make_factory(
            complete_graph_k4,
            min_realizations=10,
            max_realizations=10,
            num_workers=2,
        )

        result = simulate(factory, np.random.default_rng(42))

        assert isinstance(result, EnsembleResult)
        assert len(result.scalar_outputs) == 10
        assert result.convergence.n_realizations == 10

    def test_parallel_reproducible_with_seed(
        self, complete_graph_k4: csr_array
    ) -> None:
        factory = _make_factory(
            complete_graph_k4,
            min_realizations=10,
            max_realizations=10,
            num_workers=2,
        )

        result1 = simulate(factory, np.random.default_rng(123))
        result2 = simulate(factory, np.random.default_rng(123))

        np.testing.assert_array_equal(result1.scalar_outputs, result2.scalar_outputs)

    def test_parallel_progress_callback_invoked(
        self, complete_graph_k4: csr_array
    ) -> None:
        factory = _make_factory(
            complete_graph_k4,
            min_realizations=12,
            max_realizations=12,
            num_workers=2,
        )
        calls: list[tuple[int, int, float, bool]] = []

        def callback(n: int, max_n: int, sem: float, converged: bool) -> None:
            calls.append((n, max_n, sem, converged))

        simulate(factory, progress_callback=callback)

        assert len(calls) == 12
        assert all(c[1] == 12 for c in calls)

    def test_parallel_stops_at_max_realizations(
        self, complete_graph_k4: csr_array
    ) -> None:
        factory = _make_factory(
            complete_graph_k4,
            convergence_threshold=0.0001,
            min_realizations=10,
            max_realizations=20,
            num_workers=2,
        )

        result = simulate(factory, np.random.default_rng(999))

        assert len(result.scalar_outputs) == 20
        assert not result.convergence.converged


class TestStochasticStatistics:
    """Statistical properties of SIR simulation."""

    def test_final_size_increases_with_r0(self, complete_graph_k4: csr_array) -> None:
        """Higher R0 should produce larger epidemics on average."""
        runs = 30

        factory_low = _make_factory(
            complete_graph_k4,
            tau=0.3,
            gamma=1.0,
            t_end=20.0,
            min_realizations=runs,
            max_realizations=runs,
        )
        result_low = simulate(factory_low, np.random.default_rng(100))

        factory_high = _make_factory(
            complete_graph_k4,
            tau=2.0,
            gamma=1.0,
            t_end=20.0,
            min_realizations=runs,
            max_realizations=runs,
        )
        result_high = simulate(factory_high, np.random.default_rng(100))

        assert result_high.scalar_output_mean > result_low.scalar_output_mean

    def test_susceptible_monotonically_decreases(self, chain_graph: csr_array) -> None:
        """S(t) should never increase over time."""
        factory = _make_factory(chain_graph, tau=1.0, gamma=0.5, t_end=10.0)

        for seed in range(10):
            traj, _, _ = run_once(factory, 10.0, np.random.default_rng(seed))
            diffs = np.diff(traj["s"])
            assert np.all(diffs <= 0)

    def test_recovered_monotonically_increases(self, chain_graph: csr_array) -> None:
        """R(t) should never decrease over time."""
        factory = _make_factory(chain_graph, tau=1.0, gamma=0.5, t_end=10.0)

        for seed in range(10):
            traj, _, _ = run_once(factory, 10.0, np.random.default_rng(seed))
            diffs = np.diff(traj["r"])
            assert np.all(diffs >= 0)

    def test_epidemic_curve_has_interior_peak(
        self, complete_graph_k4: csr_array
    ) -> None:
        """Mean I(t) should peak somewhere between start and end."""
        factory = _make_factory(
            complete_graph_k4,
            tau=2.0,
            gamma=1.0,
            t_end=15.0,
            min_realizations=50,
            max_realizations=50,
        )

        result = simulate(factory, np.random.default_rng(42), n_points=100)

        peak_idx = np.argmax(result.means["i"])
        assert peak_idx > 0
        assert peak_idx < len(result.means["i"]) - 1

    def test_denser_network_larger_epidemic(
        self, chain_graph: csr_array, complete_graph_k4: csr_array
    ) -> None:
        """Complete graph should have larger epidemics than chain."""
        factory_chain = _make_factory(
            chain_graph,
            tau=1.0,
            gamma=0.5,
            t_end=20.0,
            min_realizations=30,
            max_realizations=30,
        )
        factory_complete = _make_factory(
            complete_graph_k4,
            tau=1.0,
            gamma=0.5,
            t_end=20.0,
            min_realizations=30,
            max_realizations=30,
        )

        result_chain = simulate(factory_chain, np.random.default_rng(999))
        result_complete = simulate(factory_complete, np.random.default_rng(999))

        assert result_complete.scalar_output_mean >= result_chain.scalar_output_mean

    def test_final_size_bounded_by_network_size(
        self, complete_graph_k4: csr_array
    ) -> None:
        """Final size should never exceed network size."""
        factory = _make_factory(complete_graph_k4, tau=2.0, gamma=0.5, t_end=20.0)
        n = complete_graph_k4.shape[0]

        for seed in range(20):
            _, scalar, _ = run_once(factory, 20.0, np.random.default_rng(seed))
            assert scalar <= n


class TestSubCriticalFiltering:
    """Tests for sub-critical epidemic run filtering."""

    def test_filter_disabled_by_default(self) -> None:
        config = SIRConfig()
        assert config.filter_subcritical is False

    def test_filter_discards_subcritical_runs(
        self, complete_graph_k4: csr_array
    ) -> None:
        """Filtering discards runs with final size <= initial_infected."""
        factory = _make_factory(
            complete_graph_k4,
            tau=0.5,
            gamma=1.0,
            t_end=20.0,
            filter_subcritical=True,
            min_realizations=10,
            max_realizations=500,
        )

        result = simulate(factory, np.random.default_rng(42))

        if len(result.scalar_outputs) > 0:
            assert all(fs > 1 for fs in result.scalar_outputs)

    def test_filter_tracks_discarded_count(self, complete_graph_k4: csr_array) -> None:
        """Accepted + discarded should not exceed max_realizations."""
        factory = _make_factory(
            complete_graph_k4,
            tau=0.5,
            gamma=1.0,
            t_end=20.0,
            filter_subcritical=True,
            min_realizations=10,
            max_realizations=200,
        )

        result = simulate(factory, np.random.default_rng(99))

        n_accepted = result.convergence.n_realizations
        n_discarded = result.convergence.n_discarded
        assert n_accepted + n_discarded <= 200

    def test_filter_disabled_includes_subcritical(
        self, complete_graph_k4: csr_array
    ) -> None:
        """With filter off and low tau/high gamma, some runs should be sub-critical."""
        factory = _make_factory(
            complete_graph_k4,
            tau=0.3,
            gamma=2.0,
            t_end=20.0,
            min_realizations=100,
            max_realizations=100,
        )

        result = simulate(factory, np.random.default_rng(7))

        subcritical = [fs for fs in result.scalar_outputs if fs <= 1]
        assert len(subcritical) > 0

    def test_all_subcritical_returns_empty_result(self) -> None:
        """When all runs are sub-critical, return valid empty result."""
        adjacency = csr_array(np.zeros((4, 4), dtype=np.float64))
        factory = _make_factory(
            adjacency,
            filter_subcritical=True,
            min_realizations=10,
            max_realizations=20,
            t_end=10.0,
        )

        result = simulate(factory, np.random.default_rng(0))

        assert len(result.scalar_outputs) == 0
        assert result.convergence.n_realizations == 0
        assert result.convergence.n_discarded == 20

    def test_convergence_info_n_discarded_default(self) -> None:
        """Backward compat: n_discarded defaults to 0."""
        info = ConvergenceInfo(
            converged=True, n_realizations=10, relative_sem=0.01, threshold=0.02
        )
        assert info.n_discarded == 0

    def test_filter_with_parallel_execution(self, complete_graph_k4: csr_array) -> None:
        """Filtering should work correctly in parallel mode."""
        factory = _make_factory(
            complete_graph_k4,
            tau=0.5,
            gamma=1.0,
            t_end=20.0,
            filter_subcritical=True,
            min_realizations=10,
            max_realizations=200,
            num_workers=2,
        )

        result = simulate(factory, np.random.default_rng(42))

        if len(result.scalar_outputs) > 0:
            assert all(fs > 1 for fs in result.scalar_outputs)

        n_accepted = result.convergence.n_realizations
        n_discarded = result.convergence.n_discarded
        assert n_accepted + n_discarded <= 200
