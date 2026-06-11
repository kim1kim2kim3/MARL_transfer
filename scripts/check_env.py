"""Smoke-check the local uv environment for this project."""

from __future__ import annotations

import importlib

import jax
import jaxmarl
from jaxmarl.environments.overcooked import layouts


def _version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    modules = [
        "jax",
        "jaxmarl",
        "flax",
        "optax",
        "gymnax",
        "overcooked_ai_py",
        "hydra",
        "omegaconf",
        "numpy",
        "pandas",
        "matplotlib",
    ]
    for module_name in modules:
        print(f"{module_name}: {_version(module_name)}")

    print(f"jax devices: {jax.devices()}")

    env = jaxmarl.make("overcooked", layout=layouts.overcooked_layouts["forced_coord"])
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    actions = {agent: 0 for agent in env.agents}
    next_obs, _next_state, rewards, dones, infos = env.step(key, state, actions)

    print(f"overcooked agents: {env.agents}")
    print(f"overcooked obs shapes: { {agent: value.shape for agent, value in obs.items()} }")
    next_obs_shapes = {agent: value.shape for agent, value in next_obs.items()}
    print(f"overcooked next obs shapes: {next_obs_shapes}")
    print(f"overcooked rewards: {rewards}")
    print(f"overcooked dones: {dones}")
    print(f"overcooked info keys: {sorted(infos.keys())}")
    print("environment smoke check: ok")


if __name__ == "__main__":
    main()
