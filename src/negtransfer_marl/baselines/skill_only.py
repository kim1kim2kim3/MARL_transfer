"""Skill-only transfer baseline placeholder."""

from __future__ import annotations

from negtransfer_marl.baselines.scratch import BaselineSpec

SKILL_ONLY_BASELINE = BaselineSpec(
    name="skill_only",
    description="Reuse source individual skill modules without source coordination priors.",
    uses_source_policy=False,
    uses_source_encoder=True,
    uses_source_skills=True,
    uses_source_coordination=False,
)
