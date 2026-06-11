"""Network definitions for the future IPPO implementation.

The actual actor-critic modules will be implemented here after the environment
and runner contracts are finalized.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IPPONetworkConfig:
    """Configuration placeholder for IPPO actor-critic networks."""

    hidden_dim: int = 128
    activation: str = "tanh"
    shared_parameters: bool = True


def build_networks(_config: IPPONetworkConfig):
    """Build IPPO networks.

    Raises:
        NotImplementedError: The learning algorithm is not implemented yet.
    """

    raise NotImplementedError("IPPO networks are not implemented yet")
