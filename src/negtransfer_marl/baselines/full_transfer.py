"""Full policy transfer baseline placeholder."""

from __future__ import annotations

from negtransfer_marl.baselines.scratch import BaselineSpec

FULL_TRANSFER_BASELINE = BaselineSpec(
    name="full_transfer",
    description="Initialize or regularize the target policy with the full source policy.",
    uses_source_policy=True,
    uses_source_encoder=True,
    uses_source_skills=True,
    uses_source_coordination=True,
)
