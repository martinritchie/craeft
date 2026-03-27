"""Tests for SIR configuration."""

from craeft.point_processes.epidemics.sir import SIRConfig


class TestSIRConfigR0:
    """Tests for the R0 computed property."""

    def test_r0_ratio_of_tau_to_gamma(self) -> None:
        config = SIRConfig(tau=2.0, gamma=0.5)
        assert config.R0 == 4.0

    def test_r0_less_than_one(self) -> None:
        config = SIRConfig(tau=0.5, gamma=1.0)
        assert config.R0 == 0.5

    def test_r0_equals_one(self) -> None:
        config = SIRConfig(tau=1.0, gamma=1.0)
        assert config.R0 == 1.0
