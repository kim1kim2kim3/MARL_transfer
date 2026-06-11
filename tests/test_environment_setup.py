from __future__ import annotations

import jax
import jaxmarl
from jaxmarl.environments.overcooked import layouts


def test_jaxmarl_overcooked_reset_and_step() -> None:
    env = jaxmarl.make("overcooked", layout=layouts.overcooked_layouts["forced_coord"])
    key = jax.random.PRNGKey(0)

    obs, state = env.reset(key)
    assert set(obs) == set(env.agents)
    assert all(value.shape[-1] == 26 for value in obs.values())

    actions = {agent: 0 for agent in env.agents}
    next_obs, _next_state, rewards, dones, infos = env.step(key, state, actions)

    assert set(next_obs) == set(env.agents)
    assert set(rewards) == set(env.agents)
    assert "__all__" in dones
    assert "shaped_reward" in infos
