from __future__ import annotations

import jax
import pytest
import yaml

from negtransfer_marl.envs import (
    OVERCOOKED_AI_LAYOUT_GRIDS,
    OVERCOOKED_LAYOUTS,
    PAIR_SPECS,
    get_pair_spec,
    make_overcooked_env,
    make_pair_envs,
    normalize_overcooked_ai_grid,
)


def _as_list(array) -> list[int]:
    return [int(value) for value in array.tolist()]


def test_agent_identity_order_is_preserved_for_role_swap_pair() -> None:
    source = OVERCOOKED_LAYOUTS["forced_coordination"]
    target = OVERCOOKED_LAYOUTS["forced_coordination_reversed"]

    assert _as_list(source["agent_idx"]) == [8, 11]
    assert _as_list(target["agent_idx"]) == [11, 8]


def test_registered_layout_grids_are_rectangular() -> None:
    for grid in OVERCOOKED_AI_LAYOUT_GRIDS.values():
        rows = normalize_overcooked_ai_grid(grid)
        assert len({len(row) for row in rows}) == 1


@pytest.mark.parametrize("layout_name", sorted(OVERCOOKED_LAYOUTS))
def test_registered_overcooked_layouts_reset_and_step(layout_name: str) -> None:
    env = make_overcooked_env(layout_name)
    key = jax.random.PRNGKey(0)

    obs, state = env.reset(key)
    actions = {agent: 0 for agent in env.agents}
    next_obs, _next_state, rewards, dones, infos = env.step(key, state, actions)

    assert set(obs) == set(env.agents)
    assert set(next_obs) == set(env.agents)
    assert set(rewards) == set(env.agents)
    assert "__all__" in dones
    assert "shaped_reward" in infos


@pytest.mark.parametrize("pair_id", sorted(PAIR_SPECS))
def test_registered_pairs_reset_and_step(pair_id: str) -> None:
    pair = get_pair_spec(pair_id)
    source_env, target_env = make_pair_envs(pair_id)
    key = jax.random.PRNGKey(1)

    assert source_env.num_agents == 2
    assert target_env.num_agents == 2
    assert pair.source_layout in OVERCOOKED_LAYOUTS
    assert pair.target_layout in OVERCOOKED_LAYOUTS

    for env in (source_env, target_env):
        obs, state = env.reset(key)
        actions = {agent: 0 for agent in env.agents}
        next_obs, _next_state, rewards, dones, infos = env.step(key, state, actions)
        assert set(obs) == set(env.agents)
        assert set(next_obs) == set(env.agents)
        assert set(rewards) == set(env.agents)
        assert "__all__" in dones
        assert "shaped_reward" in infos


def test_pair_config_matches_code_registry() -> None:
    with open("configs/env/overcooked/pairs.yaml") as handle:
        config = yaml.safe_load(handle)

    assert set(config["pairs"]) == set(PAIR_SPECS)
    for pair_id, pair_config in config["pairs"].items():
        pair = PAIR_SPECS[pair_id]
        assert pair_config["source_layout"] == pair.source_layout
        assert pair_config["target_layout"] == pair.target_layout
