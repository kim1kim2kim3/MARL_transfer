"""Baseline method specifications for Overcooked transfer experiments."""

from negtransfer_marl.baselines.encoder_only import ENCODER_ONLY_BASELINE
from negtransfer_marl.baselines.full_transfer import FULL_TRANSFER_BASELINE
from negtransfer_marl.baselines.scratch import SCRATCH_BASELINE
from negtransfer_marl.baselines.skill_only import SKILL_ONLY_BASELINE

__all__ = [
    "ENCODER_ONLY_BASELINE",
    "FULL_TRANSFER_BASELINE",
    "SCRATCH_BASELINE",
    "SKILL_ONLY_BASELINE",
]
