"""Command-line surface for future Overcooked training runs.

This runner currently validates pair/method selection and environment creation.
The actual IPPO/MAPPO training loop is the next implementation step.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax

from negtransfer_marl.baselines import (
    ENCODER_ONLY_BASELINE,
    FULL_TRANSFER_BASELINE,
    SCRATCH_BASELINE,
    SKILL_ONLY_BASELINE,
)
from negtransfer_marl.baselines.scratch import BaselineSpec
from negtransfer_marl.envs import get_pair_spec, make_pair_envs

BASELINES: dict[str, BaselineSpec] = {
    baseline.name: baseline
    for baseline in (
        SCRATCH_BASELINE,
        FULL_TRANSFER_BASELINE,
        ENCODER_ONLY_BASELINE,
        SKILL_ONLY_BASELINE,
    )
}


@dataclass(frozen=True)
class OvercookedRunSpec:
    """Validated Overcooked run selection."""

    pair: str
    method: str
    seed: int
    source_layout: str
    target_layout: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="pair_a", help="Registered pair id, e.g. pair_a")
    parser.add_argument("--method", default="scratch", choices=sorted(BASELINES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate pair/method and run one source/target env reset-step smoke check.",
    )
    return parser.parse_args()


def build_run_spec(pair_id: str, method: str, seed: int) -> OvercookedRunSpec:
    pair = get_pair_spec(pair_id)
    if method not in BASELINES:
        known = ", ".join(sorted(BASELINES))
        raise KeyError(f"unknown baseline method {method!r}; known methods: {known}")

    return OvercookedRunSpec(
        pair=pair.pair_id,
        method=method,
        seed=seed,
        source_layout=pair.source_layout,
        target_layout=pair.target_layout,
    )


def smoke_check_run(spec: OvercookedRunSpec, *, max_steps: int) -> None:
    source_env, target_env = make_pair_envs(spec.pair, max_steps=max_steps)
    key = jax.random.PRNGKey(spec.seed)

    for name, env in (("source", source_env), ("target", target_env)):
        obs, state = env.reset(key)
        actions = {agent: 0 for agent in env.agents}
        next_obs, _next_state, rewards, dones, infos = env.step(key, state, actions)
        obs_shapes = {agent: value.shape for agent, value in obs.items()}
        next_obs_shapes = {agent: value.shape for agent, value in next_obs.items()}
        print(
            f"{name}: obs={obs_shapes}, next_obs={next_obs_shapes}, "
            f"rewards={rewards}, dones={dones}, info_keys={sorted(infos)}"
        )


def main() -> None:
    args = parse_args()
    spec = build_run_spec(args.pair, args.method, args.seed)
    baseline = BASELINES[spec.method]

    print(f"jax backend: {jax.default_backend()}")
    print(f"jax devices: {jax.devices()}")
    print(f"run spec: {spec}")
    print(f"baseline: {baseline}")
    smoke_check_run(spec, max_steps=args.max_steps)

    if not args.check_only:
        raise SystemExit(
            "Training loop is not implemented yet. Use --check-only for structure/env validation."
        )


if __name__ == "__main__":
    main()
