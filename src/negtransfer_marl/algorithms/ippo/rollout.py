"""Rollout collection utilities for the future IPPO implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutConfig:
    """Configuration placeholder for rollout collection."""

    num_envs: int = 64
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95


def collect_rollout(*_args, **_kwargs):
    """Collect one rollout batch.

    Raises:
        NotImplementedError: The learning algorithm is not implemented yet.
    """

    raise NotImplementedError("IPPO rollout collection is not implemented yet")
