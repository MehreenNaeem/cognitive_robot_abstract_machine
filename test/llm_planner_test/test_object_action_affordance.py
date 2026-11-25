from __future__ import annotations

from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.composite.tool_based import (
    MixingAction,
    PouringAction,
)
from llm_planner.env_reasoner.action_constraints import (
    ConditionKind,
    action_condition,
    get_action_constraints_names,
)
from llm_planner.env_reasoner.object_action_affordance import (
    doable_actions_for,
    get_object_action_list,
    object_action_affordances,
)
from llm_planner.env_reasoner.object_constraints import (
    ObjectAffordance,
    ObjectAffordanceReasoner,
    actions_available_for,
)
from llm_planner.env_reasoner.reasoner_graphs.object_graphs import semantic_annotation_graph
from semantic_digital_twin.semantic_annotations.mixins import (
    HasHandle,
    HasMechanicalJoint,
    HasRootBody,
    IsPerceivable,
    IsStorageSpace,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CookingContainer,
    Door,
    Drawer,
    FoodContainer,
    Furniture,
    Milk,
    Sink,
    Pan,
    Table,
    ToolWithHandle,
    TunaCan,
    Vase,
)

# %% doable_actions_for


class TestDoableActionsFor:
    def test_reports_the_reasoner_affordances(self) -> None:
        result = doable_actions_for(Milk)

        assert result.affordances == ObjectAffordanceReasoner().affordances_for(Milk)

    def test_lists_the_actions_the_affordances_unlock(self) -> None:
        result = doable_actions_for(Milk)

        assert [
            entry.action for entry in result.doable_actions
        ] == actions_available_for(ObjectAffordanceReasoner(), Milk)

    def test_pourable_container_can_be_poured(self) -> None:
        actions = {entry.action for entry in doable_actions_for(Milk).doable_actions}

        assert PouringAction in actions

    def test_object_without_affordances_has_no_doable_actions(self) -> None:
        assert doable_actions_for(Vase).doable_actions == []

    def test_mixins_hold_every_base_class(self) -> None:
        assert doable_actions_for(Milk).mixins == [FoodContainer, IsPerceivable]

    def test_mixins_leave_out_the_root_body_base(self) -> None:
        assert Sink.__bases__ == (HasRootBody,)
        assert doable_actions_for(Sink).mixins == []

    def test_mixins_leave_out_the_mechanical_joint_base_but_keep_the_rest(self) -> None:
        assert Door.__bases__ == (HasHandle, HasMechanicalJoint)
        assert doable_actions_for(Door).mixins == [HasHandle]

    def test_mixins_leave_out_the_furniture_base_but_keep_the_rest(self) -> None:
        assert Furniture in Drawer.__bases__
        assert Furniture not in doable_actions_for(Drawer).mixins
        assert IsStorageSpace in doable_actions_for(Drawer).mixins

    def test_entry_carries_the_action_required_parameters(self) -> None:
        pick_up = next(
            entry
            for entry in doable_actions_for(Milk).doable_actions
            if entry.action is PickUpAction
        )

        assert pick_up.required_parameters == get_action_constraints_names(PickUpAction)

    def test_entry_carries_the_action_pre_and_post_condition(self) -> None:
        place = next(
            entry
            for entry in doable_actions_for(Milk).doable_actions
            if entry.action is PlaceAction
        )

        assert place.pre_condition == action_condition(PlaceAction, ConditionKind.PRE)
        assert place.post_condition == action_condition(PlaceAction, ConditionKind.POST)

    def test_entry_required_affordances_are_the_ones_the_object_satisfies(self) -> None:
        pour = next(
            entry
            for entry in doable_actions_for(Milk).doable_actions
            if entry.action is PouringAction
        )

        assert pour.required_affordances == frozenset({ObjectAffordance.POURABLE})

    def test_cooking_container_can_be_mixed_in(self) -> None:
        actions = {entry.action for entry in doable_actions_for(Pan).doable_actions}

        assert MixingAction in actions

    def test_mixing_entry_lists_its_parameter_type_requirements(self) -> None:
        mixing = next(
            entry
            for entry in doable_actions_for(Pan).doable_actions
            if entry.action is MixingAction
        )

        assert {
            (requirement.parameter, requirement.accepted_types)
            for requirement in mixing.parameter_type_requirements
        } == {
            ("tool", frozenset({ToolWithHandle})),
            ("container", frozenset({CookingContainer})),
        }

    def test_unconstrained_action_has_no_parameter_type_requirements(self) -> None:
        pick_up = next(
            entry
            for entry in doable_actions_for(Milk).doable_actions
            if entry.action is PickUpAction
        )

        assert pick_up.parameter_type_requirements == []

    def test_place_target_entry_lists_its_accepted_location_types(self) -> None:
        place = next(
            entry
            for entry in doable_actions_for(Table).doable_actions
            if entry.action is PlaceAction
        )

        assert [
            (requirement.parameter, requirement.accepted_types)
            for requirement in place.parameter_type_requirements
        ] == [("target_location", frozenset({IsStorageSpace, Table}))]


# %% object_action_affordances


class TestObjectActionAffordances:
    def test_covers_every_concrete_object_under_the_root(self) -> None:
        result = object_action_affordances(FoodContainer)

        expected = {
            leaf.__name__
            for leaf in semantic_annotation_graph(FoodContainer).concrete_annotations()
        }
        assert set(result) == expected

    def test_object_entry_matches_the_single_object_query(self) -> None:
        from_graph = object_action_affordances(FoodContainer)["Milk"]

        assert [entry.action for entry in from_graph.doable_actions] == [
            entry.action for entry in doable_actions_for(Milk).doable_actions
        ]

    def test_each_entry_carries_the_object_mixins(self) -> None:
        result = object_action_affordances(FoodContainer)

        assert result["Milk"].mixins == [FoodContainer, IsPerceivable]
        assert result["TunaCan"].mixins == [FoodContainer]

    def test_non_pourable_container_cannot_be_poured(self) -> None:
        result = object_action_affordances(FoodContainer)

        actions = {entry.action for entry in result["TunaCan"].doable_actions}
        assert PouringAction not in actions


# %% get_object_action_list


class TestGetObjectActionList:
    def test_returns_the_doable_action_class_names(self) -> None:
        assert get_object_action_list(Milk) == {
            entry.action.__name__ for entry in doable_actions_for(Milk).doable_actions
        }

    def test_pourable_container_names_pouring(self) -> None:
        assert get_object_action_list(Milk) == {
            PickUpAction.__name__,
            PlaceAction.__name__,
            PouringAction.__name__,
        }

    def test_object_without_doable_actions_gives_an_empty_set(self) -> None:
        assert get_object_action_list(Vase) == set()
