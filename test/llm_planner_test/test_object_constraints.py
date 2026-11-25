from __future__ import annotations

from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
    PouringAction,
)
from llm_planner.env_reasoner.object_constraints import (
    ACTION_PARAMETER_TYPE_REQUIREMENTS,
    ObjectAffordance,
    ObjectAffordanceReasoner,
    actions_available_for,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasDoors,
    HasHandle,
    IsStorageSpace,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Bowl,
    Cabinet,
    CookingContainer,
    Drawer,
    Fork,
    Milk,
    Pan,
    Plate,
    Table,
    ToolWithHandle,
    Vase,
    Whisk,
)

# %% actions_available_for


class TestActionsAvailableFor:
    def setup_method(self) -> None:
        self.reasoner = ObjectAffordanceReasoner()

    def test_cutable_graspable_object_gets_pick_up_place_and_cutting(self) -> None:
        actions = actions_available_for(self.reasoner, Apple)
        assert actions == [PickUpAction, PlaceAction, CuttingAction]

    def test_pourable_graspable_object_gets_pick_up_place_and_pouring(self) -> None:
        actions = actions_available_for(self.reasoner, Milk)
        assert actions == [PickUpAction, PlaceAction, PouringAction]

    def test_openable_closable_storage_space_also_gets_place(self) -> None:
        actions = actions_available_for(self.reasoner, Drawer)
        assert actions == [OpenAction, CloseAction, PlaceAction]

    def test_object_without_affordances_gets_no_actions(self) -> None:
        assert actions_available_for(self.reasoner, Vase) == []


# %% mixing parameter type requirements


class TestMixingParameterTypeRequirements:
    def setup_method(self) -> None:
        self.reasoner = ObjectAffordanceReasoner()

    def test_cooking_container_gets_mixing(self) -> None:
        assert MixingAction in actions_available_for(self.reasoner, Pan)

    def test_handled_tool_gets_mixing(self) -> None:
        assert MixingAction in actions_available_for(self.reasoner, Whisk)

    def test_receptacle_that_is_not_a_cooking_container_does_not_get_mixing(
        self,
    ) -> None:
        assert MixingAction not in actions_available_for(self.reasoner, Drawer)

    def test_mixing_constrains_its_tool_and_container_parameters(self) -> None:
        mixing_requirements = {
            (requirement.parameter, requirement.accepted_types)
            for requirement in ACTION_PARAMETER_TYPE_REQUIREMENTS
            if requirement.action is MixingAction
        }
        assert mixing_requirements == {
            ("tool", frozenset({ToolWithHandle})),
            ("container", frozenset({CookingContainer})),
        }


# %% placing parameter type requirements


class TestPlacingParameterTypeRequirements:
    def setup_method(self) -> None:
        self.reasoner = ObjectAffordanceReasoner()

    def test_table_gets_place_as_a_target_location(self) -> None:
        assert actions_available_for(self.reasoner, Table) == [PlaceAction]

    def test_storage_space_gets_place_as_a_target_location(self) -> None:
        assert PlaceAction in actions_available_for(self.reasoner, Drawer)

    def test_object_that_is_neither_graspable_nor_storage_does_not_get_place(
        self,
    ) -> None:
        assert PlaceAction not in actions_available_for(self.reasoner, Vase)

    def test_placing_constrains_its_target_location_parameter(self) -> None:
        place_requirements = {
            (requirement.parameter, requirement.accepted_types)
            for requirement in ACTION_PARAMETER_TYPE_REQUIREMENTS
            if requirement.action is PlaceAction
        }
        assert place_requirements == {
            ("target_location", frozenset({IsStorageSpace, Table})),
        }


# %% open and close affordances


class TestOpenAndCloseAffordances:
    def setup_method(self) -> None:
        self.reasoner = ObjectAffordanceReasoner()

    def test_object_with_doors_is_openable_and_closable(self) -> None:
        assert issubclass(Cabinet, HasDoors)
        assert {
            ObjectAffordance.OPENABLE,
            ObjectAffordance.CLOSABLE,
        } <= self.reasoner.affordances_for(Cabinet)

    def test_object_with_a_handle_stays_openable_and_closable(self) -> None:
        assert issubclass(Drawer, HasHandle)
        assert {
            ObjectAffordance.OPENABLE,
            ObjectAffordance.CLOSABLE,
        } <= self.reasoner.affordances_for(Drawer)

    def test_object_with_doors_gets_open_and_close_actions(self) -> None:
        actions = actions_available_for(self.reasoner, Cabinet)
        assert OpenAction in actions
        assert CloseAction in actions

    def test_object_with_neither_doors_nor_a_handle_is_not_openable(self) -> None:
        assert not issubclass(Vase, (HasDoors, HasHandle))
        assert ObjectAffordance.OPENABLE not in self.reasoner.affordances_for(Vase)


# %% cutlery with a supporting surface


class TestCutleryWithSupportingSurface:
    def setup_method(self) -> None:
        self.reasoner = ObjectAffordanceReasoner()

    def test_plate_is_a_graspable_receptacle(self) -> None:
        assert self.reasoner.affordances_for(Plate) == {
            ObjectAffordance.GRASPABLE,
            ObjectAffordance.RECEPTACLE,
        }

    def test_bowl_is_a_graspable_receptacle(self) -> None:
        assert self.reasoner.affordances_for(Bowl) == {
            ObjectAffordance.GRASPABLE,
            ObjectAffordance.RECEPTACLE,
        }

    def test_cutlery_without_a_supporting_surface_is_only_graspable(self) -> None:
        assert self.reasoner.affordances_for(Fork) == {ObjectAffordance.GRASPABLE}
