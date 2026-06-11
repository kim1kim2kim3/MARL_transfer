"""Overcooked layout and source-target pair registry for transfer experiments.

The documentation in ``docs/overcooked_environment_pairs_ko.md`` uses the
Overcooked-AI layout notation:

- ``X``: non-walkable counter/wall
- ``S``: serving location
- ``D``: dish dispenser
- ``O``: onion dispenser
- ``P``: pot
- ``1``/``2``: agent identity-specific start positions
- space: walkable floor

JaxMARL's classic Overcooked environment uses a different symbolic parser, so
this module keeps a small project-owned converter. In particular, it preserves
agent identity order from ``1`` and ``2``; this matters for role-lock-in transfer
experiments where swapping agent starts is the target manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Final

import jax.numpy as jnp
import jaxmarl
from flax.core.frozen_dict import FrozenDict
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

_LAYOUT_KEYS: Final = (
    "wall_idx",
    "agent_idx",
    "goal_idx",
    "plate_pile_idx",
    "onion_pile_idx",
    "pot_idx",
)

_OBJECT_KEYS: Final = {
    "S": "goal_idx",
    "D": "plate_pile_idx",
    "O": "onion_pile_idx",
    "P": "pot_idx",
}


@dataclass(frozen=True)
class OvercookedPairSpec:
    """Source-target pair metadata used by the CAST experiment suite."""

    pair_id: str
    title: str
    source_layout: str
    target_layout: str
    main_mismatch: str
    failure_mode: str
    diagnostics: tuple[str, ...]


OVERCOOKED_AI_LAYOUT_GRIDS: Final[dict[str, str]] = {
    # Source layouts from Overcooked-AI.
    "forced_coordination": """
        XXXPX
        O X1P
        O2X X
        D X X
        XXXSX
    """,
    "coordination_ring": """
        XXXPX
        X 1 P
        D2X X
        O   X
        XOSXX
    """,
    "bottleneck": """
        XXOXDXX
        X 1X2 X
        X  X  X
        X     X
        XSXXPPX
    """,
    "counter_circuit_o_1order": """
        XXXPPXXX
        X  2   X
        D XXXX S
        X  1   X
        XXXOOXXX
    """,
    # Custom target layouts from docs/overcooked_environment_pairs_ko.md.
    "forced_coordination_reversed": """
        XXXPX
        O X2P
        O1X X
        D X X
        XXXSX
    """,
    "open_parallel_room": """
        XXXXXXXXX
        XO P P OX
        X1     2X
        XD  S  DX
        XXXXXXXXX
    """,
    "wide_bottleneck": """
        XXOXDXX
        X 1 2 X
        X     X
        X     X
        XSXXPPX
    """,
}


PAIR_SPECS: Final[dict[str, OvercookedPairSpec]] = {
    "pair_a": OvercookedPairSpec(
        pair_id="pair_a",
        title="Role Identity Lock-in",
        source_layout="forced_coordination",
        target_layout="forced_coordination_reversed",
        main_mismatch="same task structure with reversed agent-role assignment",
        failure_mode="role identity lock-in",
        diagnostics=(
            "agent_specific_subtask_frequency",
            "role_entropy",
            "time_to_first_soup",
            "target_auc",
            "role_swapped_oracle",
        ),
    ),
    "pair_b": OvercookedPairSpec(
        pair_id="pair_b",
        title="Fixed-role Handoff to Ring Movement Coordination",
        source_layout="forced_coordination",
        target_layout="coordination_ring",
        main_mismatch="fixed spatial role versus ring movement coordination",
        failure_mode="handoff/fixed-role prior delays movement convention discovery",
        diagnostics=(
            "blocking_count",
            "no_op_rate",
            "movement_direction_consistency",
            "inter_agent_distance",
            "target_auc",
        ),
    ),
    "pair_c": OvercookedPairSpec(
        pair_id="pair_c",
        title="Handoff-based Cooperation to Parallel Independent Cooking",
        source_layout="forced_coordination",
        target_layout="open_parallel_room",
        main_mismatch="handoff-based cooperation versus parallel independent cooking",
        failure_mode="source handoff habit suppresses target parallelism",
        diagnostics=(
            "simultaneous_progress_rate",
            "handoff_count",
            "role_entropy",
            "distance_traveled_per_soup",
            "time_to_first_soup",
            "target_auc",
        ),
    ),
    "pair_d": OvercookedPairSpec(
        pair_id="pair_d",
        title="Bottleneck Waiting to Open-passage Movement",
        source_layout="bottleneck",
        target_layout="wide_bottleneck",
        main_mismatch="narrow-passage waiting versus open-passage simultaneous movement",
        failure_mode="waiting/yielding convention becomes unnecessary delay",
        diagnostics=(
            "no_op_rate_near_former_bottleneck",
            "waiting_duration",
            "simultaneous_movement_ratio",
            "blocking_count",
            "soup_completion_time",
        ),
    ),
    "pair_e": OvercookedPairSpec(
        pair_id="pair_e",
        title="Ring Movement Convention to Counter/Station Coordination",
        source_layout="coordination_ring",
        target_layout="counter_circuit_o_1order",
        main_mismatch="circular movement convention versus counter/station coordination",
        failure_mode="movement route prior delays efficient station/counter sequencing",
        diagnostics=(
            "counter_usage_efficiency",
            "station_visit_sequence",
            "route_efficiency",
            "idle_time_around_counters",
            "target_auc",
        ),
    ),
}


def normalize_overcooked_ai_grid(grid: str) -> tuple[str, ...]:
    """Return rectangular grid rows while preserving meaningful interior spaces."""

    rows = [row.rstrip() for row in dedent(grid).strip("\n").splitlines()]
    if not rows:
        raise ValueError("layout grid is empty")

    width = len(rows[0])
    non_rectangular = [(idx, len(row)) for idx, row in enumerate(rows) if len(row) != width]
    if non_rectangular:
        raise ValueError(f"layout grid must be rectangular; row lengths={non_rectangular}")

    return tuple(rows)


def overcooked_ai_grid_to_jaxmarl_layout(grid: str) -> FrozenDict:
    """Convert an Overcooked-AI style grid string to a JaxMARL layout.

    The classic JaxMARL Overcooked environment supports onion soup layouts. Tomato
    cells are intentionally rejected so experiments do not silently run with a
    changed recipe/station semantics.
    """

    rows = normalize_overcooked_ai_grid(grid)
    height = len(rows)
    width = len(rows[0])
    layout_lists: dict[str, list[int]] = {key: [] for key in _LAYOUT_KEYS}
    labeled_agents: dict[int, int] = {}
    anonymous_agents: list[int] = []

    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            flat_idx = width * row_idx + col_idx
            if cell == " ":
                continue
            if cell == "X":
                layout_lists["wall_idx"].append(flat_idx)
            elif cell in _OBJECT_KEYS:
                layout_lists[_OBJECT_KEYS[cell]].append(flat_idx)
                layout_lists["wall_idx"].append(flat_idx)
            elif cell in {"1", "2"}:
                labeled_agents[int(cell)] = flat_idx
            elif cell == "A":
                anonymous_agents.append(flat_idx)
            elif cell == "T":
                raise ValueError("tomato layouts are not supported by the onion-only JaxMARL setup")
            else:
                raise ValueError(
                    f"unsupported layout cell {cell!r} at row={row_idx}, col={col_idx}"
                )

    if labeled_agents:
        expected_labels = tuple(range(1, len(labeled_agents) + 1))
        actual_labels = tuple(sorted(labeled_agents))
        if actual_labels != expected_labels:
            raise ValueError(
                "agent labels must be consecutive starting at 1; "
                f"expected={expected_labels}, actual={actual_labels}"
            )
        layout_lists["agent_idx"] = [labeled_agents[label] for label in expected_labels]
    else:
        layout_lists["agent_idx"] = anonymous_agents

    if len(layout_lists["agent_idx"]) != 2:
        raise ValueError(f"classic JaxMARL Overcooked requires exactly 2 agents: {rows}")

    layout = {key: jnp.array(values, dtype=jnp.int32) for key, values in layout_lists.items()}
    layout["height"] = height
    layout["width"] = width
    return FrozenDict(layout)


def get_layout(layout_name: str) -> FrozenDict:
    """Return a JaxMARL-compatible layout by project layout name."""

    try:
        return OVERCOOKED_LAYOUTS[layout_name]
    except KeyError as exc:
        known = ", ".join(sorted(OVERCOOKED_LAYOUTS))
        raise KeyError(
            f"unknown Overcooked layout {layout_name!r}; known layouts: {known}"
        ) from exc


def get_pair_spec(pair_id: str) -> OvercookedPairSpec:
    """Return metadata for one source-target pair."""

    normalized = pair_id.lower()
    try:
        return PAIR_SPECS[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(PAIR_SPECS))
        raise KeyError(f"unknown Overcooked pair {pair_id!r}; known pairs: {known}") from exc


def make_overcooked_env(
    layout_name: str,
    *,
    max_steps: int = 400,
    random_reset: bool = False,
) -> MultiAgentEnv:
    """Create a JaxMARL classic Overcooked environment for a registered layout."""

    return jaxmarl.make(
        "overcooked",
        layout=get_layout(layout_name),
        max_steps=max_steps,
        random_reset=random_reset,
    )


def make_pair_envs(
    pair_id: str,
    *,
    max_steps: int = 400,
    random_reset: bool = False,
) -> tuple[MultiAgentEnv, MultiAgentEnv]:
    """Create source and target environments for a registered source-target pair."""

    pair = get_pair_spec(pair_id)
    source_env = make_overcooked_env(
        pair.source_layout,
        max_steps=max_steps,
        random_reset=random_reset,
    )
    target_env = make_overcooked_env(
        pair.target_layout,
        max_steps=max_steps,
        random_reset=random_reset,
    )
    return source_env, target_env


OVERCOOKED_LAYOUTS: Final[dict[str, FrozenDict]] = {
    name: overcooked_ai_grid_to_jaxmarl_layout(grid)
    for name, grid in OVERCOOKED_AI_LAYOUT_GRIDS.items()
}
