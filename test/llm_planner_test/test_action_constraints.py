from __future__ import annotations

import inspect

from coraplex.querying.predicates import GripperIsFree
from coraplex.locations.pose_validator import IsObjectReachableBy
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.container import CloseAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from llm_planner.env_reasoner.action_constraints import (
    ConditionKind,
    action_condition,
    action_conditions,
)

# %% action_condition reuses the action's EQL condition method


class TestActionConditionReusesConditionMethod:
    def test_description_is_the_condition_method_docstring(self) -> None:
        condition = action_condition(PickUpAction, ConditionKind.PRE)

        assert condition.description == inspect.getdoc(PickUpAction.pre_condition)

    def test_predicates_are_the_classes_the_method_constructs(self) -> None:
        condition = action_condition(PickUpAction, ConditionKind.PRE)

        assert set(condition.predicates) == {
            GripperIsFree.__name__,
            IsObjectReachableBy.__name__,
        }

    def test_kind_is_the_requested_kind(self) -> None:
        assert action_condition(PickUpAction, ConditionKind.POST).kind is (
            ConditionKind.POST
        )

    def test_overriding_action_is_marked_declared(self) -> None:
        assert action_condition(PickUpAction, ConditionKind.PRE).is_declared is True


# %% inherited default condition


class TestInheritedDefaultCondition:
    def test_action_without_its_own_pre_condition_is_not_declared(self) -> None:
        assert action_condition(CloseAction, ConditionKind.PRE).is_declared is False

    def test_no_predicates_are_read_from_the_default(self) -> None:
        assert action_condition(CloseAction, ConditionKind.PRE).predicates == []

    def test_description_falls_back_to_the_base_docstring(self) -> None:
        condition = action_condition(CloseAction, ConditionKind.PRE)

        assert condition.description == (
            inspect.getdoc(ActionDescription.pre_condition) or ""
        )


# %% static extraction limits


class TestStaticExtractionLimits:
    def test_predicate_reached_only_through_a_helper_is_not_listed(self) -> None:
        condition = action_condition(ReachAction, ConditionKind.POST)

        assert condition.is_declared is True
        assert condition.predicates == []


# %% action_conditions


class TestActionConditions:
    def test_maps_action_name_to_pre_then_post(self) -> None:
        conditions = action_conditions([PickUpAction])

        assert [condition.kind for condition in conditions["PickUpAction"]] == [
            ConditionKind.PRE,
            ConditionKind.POST,
        ]
