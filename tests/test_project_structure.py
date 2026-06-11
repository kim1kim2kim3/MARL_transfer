from __future__ import annotations

from negtransfer_marl.algorithms.ippo import IPPONetworkConfig, IPPOTrainConfig, RolloutConfig
from negtransfer_marl.baselines import (
    ENCODER_ONLY_BASELINE,
    FULL_TRANSFER_BASELINE,
    SCRATCH_BASELINE,
    SKILL_ONLY_BASELINE,
)
from negtransfer_marl.runners.train_overcooked import BASELINES, build_run_spec


def test_baseline_registry_contains_expected_methods() -> None:
    assert set(BASELINES) == {"scratch", "full_transfer", "encoder_only", "skill_only"}
    assert SCRATCH_BASELINE.name == "scratch"
    assert FULL_TRANSFER_BASELINE.uses_source_policy
    assert ENCODER_ONLY_BASELINE.uses_source_encoder
    assert SKILL_ONLY_BASELINE.uses_source_skills
    assert not SKILL_ONLY_BASELINE.uses_source_coordination


def test_ippo_placeholder_configs_are_importable() -> None:
    assert IPPONetworkConfig().hidden_dim == 128
    assert RolloutConfig().num_steps == 128
    assert IPPOTrainConfig().network.shared_parameters


def test_train_overcooked_run_spec_validation() -> None:
    spec = build_run_spec("pair_a", "scratch", 7)
    assert spec.pair == "pair_a"
    assert spec.method == "scratch"
    assert spec.seed == 7
    assert spec.source_layout == "forced_coordination"
    assert spec.target_layout == "forced_coordination_reversed"
