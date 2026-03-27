"""Tests for experiment orchestration."""

import numpy as np

from craeft.experiment import Experiment, ExperimentResult
from craeft.networks.generation.generator import ErdosRenyiGenerator
from craeft.point_processes.epidemics.simulator import SIRSimulator
from craeft.point_processes.epidemics.sir import SIRConfig
from craeft.point_processes.gillespie import EnsembleResult
from craeft.point_processes.process import ConvergenceConfig


def _quick_config() -> tuple[SIRConfig, ConvergenceConfig]:
    """Minimal SIR config for fast tests."""
    return (
        SIRConfig(tau=0.5, gamma=1.0),
        ConvergenceConfig(min_realizations=10, max_realizations=10, t_end=5.0),
    )


class TestExperimentResult:
    """Tests for ExperimentResult container."""

    def test_len(self) -> None:
        results = (EnsembleResult.__new__(EnsembleResult),) * 3
        er = ExperimentResult(results=results)
        assert len(er) == 3

    def test_iter(self) -> None:
        results = (EnsembleResult.__new__(EnsembleResult),) * 2
        er = ExperimentResult(results=results)
        assert list(er) == list(results)

    def test_getitem(self) -> None:
        results = (EnsembleResult.__new__(EnsembleResult),) * 2
        er = ExperimentResult(results=results)
        assert er[0] is results[0]
        assert er[1] is results[1]


class TestExperiment:
    """Tests for Experiment orchestration."""

    def test_returns_experiment_result(self) -> None:
        sir_config, convergence = _quick_config()
        experiment = Experiment(
            generator=ErdosRenyiGenerator(n=50, p=0.1),
            simulator=SIRSimulator(config=sir_config, convergence=convergence),
            n_networks=2,
        )
        result = experiment.run(np.random.default_rng(42))
        assert isinstance(result, ExperimentResult)

    def test_correct_number_of_results(self) -> None:
        n_networks = 3
        sir_config, convergence = _quick_config()
        experiment = Experiment(
            generator=ErdosRenyiGenerator(n=50, p=0.1),
            simulator=SIRSimulator(config=sir_config, convergence=convergence),
            n_networks=n_networks,
        )
        result = experiment.run(np.random.default_rng(42))
        assert len(result) == n_networks

    def test_each_result_is_ensemble_result(self) -> None:
        sir_config, convergence = _quick_config()
        experiment = Experiment(
            generator=ErdosRenyiGenerator(n=50, p=0.1),
            simulator=SIRSimulator(config=sir_config, convergence=convergence),
            n_networks=2,
        )
        result = experiment.run(np.random.default_rng(42))
        for sr in result:
            assert isinstance(sr, EnsembleResult)

    def test_reproducible_with_same_seed(self) -> None:
        sir_config, convergence = _quick_config()
        experiment = Experiment(
            generator=ErdosRenyiGenerator(n=50, p=0.1),
            simulator=SIRSimulator(config=sir_config, convergence=convergence),
            n_networks=2,
        )
        a = experiment.run(np.random.default_rng(42))
        b = experiment.run(np.random.default_rng(42))
        for ra, rb in zip(a, b):
            assert np.allclose(ra.means["i"], rb.means["i"])

    def test_different_seeds_differ(self) -> None:
        sir_config, convergence = _quick_config()
        experiment = Experiment(
            generator=ErdosRenyiGenerator(n=50, p=0.1),
            simulator=SIRSimulator(config=sir_config, convergence=convergence),
            n_networks=2,
        )
        a = experiment.run(np.random.default_rng(1))
        b = experiment.run(np.random.default_rng(2))
        # At least one result pair should differ
        any_different = any(
            not np.allclose(ra.means["i"], rb.means["i"]) for ra, rb in zip(a, b)
        )
        assert any_different
