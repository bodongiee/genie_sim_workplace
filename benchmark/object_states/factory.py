# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from . import BaseObjState, AABB, OnTop, Pose, Inside, Open, Pass2People, OnShelf, PourWater


_ALL_STATES = frozenset([AABB, Pose, Inside, Pass2People, OnShelf, PourWater])

_ABILITY_TO_STATE_MAPPING = {
    "openable": [Open],
}

_DEFAULT_STATE_SET = frozenset([AABB, Pose, Inside, OnTop, Pass2People, OnShelf, PourWater])


def get_default_states():
    return _DEFAULT_STATE_SET


def get_all_states():
    return _ALL_STATES


def get_state_name(state):
    return state.__name__


def get_state_from_name(name):
    return next(state for state in _ALL_STATES if get_state_name(state) == name)


def get_states_for_ability(ability):
    if ability not in _ABILITY_TO_STATE_MAPPING:
        return []
    return _ABILITY_TO_STATE_MAPPING[ability]
