# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from bddl.backend_abc import BDDLBackend
from bddl.logic_base import BinaryAtomicFormula, UnaryAtomicFormula


import object_states


class ObjectStateUnaryPredicate(UnaryAtomicFormula):
    STATE_CLASS = None
    STATE_NAME = None

    def _evaluate(self, obj, **kwargs):
        return obj.states[self.STATE_CLASS].get_value(**kwargs)

    def _sample(self, obj, binary_state, **kwargs):
        return obj.states[self.STATE_CLASS].set_value(binary_state, **kwargs)


class ObjectStateBinaryPredicate(BinaryAtomicFormula):
    STATE_CLASS = None
    STATE_NAME = None

    def _evaluate(self, obj1, obj2, **kwargs):
        return obj1.states[self.STATE_CLASS].get_value(obj2, **kwargs)

    def _sample(self, obj1, obj2, binary_state, **kwargs):
        return obj1.states[self.STATE_CLASS].set_value(obj2, binary_state, **kwargs)


def get_unary_predicate_for_state(state_class, state_name):
    return type(
        state_class.__name__ + "StateUnaryPredicate",
        (ObjectStateUnaryPredicate,),
        {"STATE_CLASS": state_class, "STATE_NAME": state_name},
    )


def get_binary_predicate_for_state(state_class, state_name):
    return type(
        state_class.__name__ + "StateBinaryPredicate",
        (ObjectStateBinaryPredicate,),
        {"STATE_CLASS": state_class, "STATE_NAME": state_name},
    )


# Better add remaining predicates
SUPPORTED_PREDICATES = {
    "inside": get_binary_predicate_for_state(object_states.Inside, "inside"),
    "ontop": get_binary_predicate_for_state(object_states.OnTop, "ontop"),
    "open": get_unary_predicate_for_state(object_states.Open, "open"),
    "pass2people": get_unary_predicate_for_state(
        object_states.Pass2People, "pass2people"
    ),
    "onshelf": get_unary_predicate_for_state(object_states.OnShelf, "onshelf"),
    "pourwater": get_binary_predicate_for_state(object_states.PourWater, "pourwater"),
}


class IsaacSimBDDLBackend(BDDLBackend):
    def get_predicate_class(self, predicate_name):
        return SUPPORTED_PREDICATES[predicate_name]
