"""Smoke-check every registered Overcooked source-target pair."""

from __future__ import annotations

import jax

from negtransfer_marl.envs import PAIR_SPECS, make_pair_envs


def _smoke_step(env, key):
    obs, state = env.reset(key)
    actions = {agent: 0 for agent in env.agents}
    next_obs, _next_state, rewards, dones, infos = env.step(key, state, actions)
    return obs, next_obs, rewards, dones, infos


def main() -> None:
    print(f"jax backend: {jax.default_backend()}")
    print(f"jax devices: {jax.devices()}")

    for idx, (pair_id, pair) in enumerate(sorted(PAIR_SPECS.items())):
        key = jax.random.PRNGKey(idx)
        source_env, target_env = make_pair_envs(pair_id)
        source = _smoke_step(source_env, key)
        target = _smoke_step(target_env, key)
        print(
            f"{pair_id}: {pair.source_layout} -> {pair.target_layout} | "
            f"source_obs={ {agent: value.shape for agent, value in source[0].items()} } | "
            f"target_obs={ {agent: value.shape for agent, value in target[0].items()} }"
        )

    print("overcooked pair smoke check: ok")


if __name__ == "__main__":
    main()
