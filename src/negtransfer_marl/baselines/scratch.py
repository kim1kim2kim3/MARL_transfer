"""Scratch baseline: train on the target task without source knowledge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    """Minimal baseline metadata shared by runner and experiment configs."""

    name: str
    description: str
    uses_source_policy: bool
    uses_source_encoder: bool
    uses_source_skills: bool
    uses_source_coordination: bool


SCRATCH_BASELINE = BaselineSpec(
    name="scratch",
    description="Target-task online learning with random initialization and no source transfer.",
    uses_source_policy=False,
    uses_source_encoder=False,
    uses_source_skills=False,
    uses_source_coordination=False,
)
