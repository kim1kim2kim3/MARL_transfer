"""Encoder-only transfer baseline placeholder."""

from __future__ import annotations

from negtransfer_marl.baselines.scratch import BaselineSpec

ENCODER_ONLY_BASELINE = BaselineSpec(
    name="encoder_only",
    description="Reuse only the source observation encoder, then learn target policy online.",
    uses_source_policy=False,
    uses_source_encoder=True,
    uses_source_skills=False,
    uses_source_coordination=False,
)
