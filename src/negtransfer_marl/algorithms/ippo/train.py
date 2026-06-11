"""Training entrypoints for the future IPPO implementation."""

from __future__ import annotations

from dataclasses import dataclass

from negtransfer_marl.algorithms.ippo.networks import IPPONetworkConfig
from negtransfer_marl.algorithms.ippo.rollout import RolloutConfig


@dataclass(frozen=True)
class IPPOTrainConfig:
    """Minimal IPPO training configuration placeholder."""

    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    clip_eps: float = 0.2
    update_epochs: int = 4
    minibatches: int = 4
    network: IPPONetworkConfig = IPPONetworkConfig()
    rollout: RolloutConfig = RolloutConfig()


def train_ippo(*_args, **_kwargs):
    """Train an IPPO agent.

    Raises:
        NotImplementedError: The learning algorithm is not implemented yet.
    """

    raise NotImplementedError("IPPO training is not implemented yet")
