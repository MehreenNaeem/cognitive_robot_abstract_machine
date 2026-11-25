"""
Tests for :mod:`llm_planner.env_reasoner.reasoner_graphs.AFG`.
"""

from __future__ import annotations

from dataclasses import replace
from http import HTTPStatus
from itertools import product
from unittest.mock import patch

import pytest
from typing_extensions import Any

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.pose_validator import IsObjectReachableBy
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.view_manager import ViewManager
from llm_planner.env_reasoner.object_action_affordance import get_object_action_list
from llm_planner.env_reasoner.reasoner_graphs.AFG import (
    ActionFeasibilityGraph,
    APPROACH_DIRECTIONS,
    ARMS,
    ActionWithoutConditions,
    ConditionParameter,
    FEASIBILITY_SHAPES,
    FEASIBILITY_TREE_TEMPLATE,
    FeasibilityNodeKind,
    MissingConditionParameter,
    are_action_conditions_met,
    feasibility_tree,
    feasibility_visualizer,
    object_action_validation,
)
from llm_planner.env_reasoner.reasoner_graphs.object_graphs import (
    BodyWithoutSemanticAnnotation,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door, Milk
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

MILK_BODY_NAME = "milk.stl"
"""
An annotated apartment object.
"""

DRAWER_BODY_NAME = "cabinet10_drawer_top"
"""
An apartment drawer, closed when the world is loaded.
"""

HELD_BODY_NAME = "held_body"
"""
Name of the body a gripper holds in these tests.
"""

OPENED_DRAWER_POSITION = 0.3
"""
A joint position at which the drawer counts as open.
"""


# %% construction


class TestActionFeasibilityGraphConstruction:
    def test_the_graph_keeps_the_objects_actions_and_world_it_is_given(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        graph = ActionFeasibilityGraph(
            object_names=[MILK_BODY_NAME],
            actions=[PickUpAction],
            world=world,
            context=context,
        )

        assert graph.object_names == [MILK_BODY_NAME]
        assert graph.actions == [PickUpAction]
        assert graph.world is world
        assert graph.context is context


# %% object_action_validation


class TestObjectActionValidation:
    def test_the_actions_of_a_body_are_those_of_its_annotation_type(
        self, apartment_world_copy
    ) -> None:
        assert object_action_validation(MILK_BODY_NAME, apartment_world_copy) == sorted(
            get_object_action_list(Milk)
        )

    def test_a_body_without_an_annotation_raises(self, apartment_world_copy) -> None:
        with pytest.raises(BodyWithoutSemanticAnnotation):
            object_action_validation(
                apartment_world_copy.root.name.name, apartment_world_copy
            )


# %% action conditions


@pytest.fixture
def robot_scene(apartment_world_pr2_copy_with_context):
    """
    :return: The world, the context, and the parameters of a gripper and a closed drawer.
    """
    world, _, context = apartment_world_pr2_copy_with_context
    parameters = {
        ConditionParameter.END_EFFECTOR: ViewManager.get_end_effector_view(
            Arms.LEFT, context.robot
        ),
        ConditionParameter.OBJECT_DESIGNATOR: world.get_body_by_name(DRAWER_BODY_NAME),
        ConditionParameter.WORLD: world,
        ConditionParameter.DOOR: Door,
    }
    return world, context, parameters


def hold_body(world, parameters) -> None:
    """
    Attach a new body to the tool frame of the gripper, so the gripper is holding it.
    """
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=parameters[ConditionParameter.END_EFFECTOR].tool_frame,
                child=Body(name=PrefixedName(HELD_BODY_NAME)),
            )
        )


def open_drawer(world) -> None:
    """
    Move the drawer to a position where it counts as open.
    """
    drawer = world.get_body_by_name(DRAWER_BODY_NAME)
    world.compute_parent_connection(drawer).position = OPENED_DRAWER_POSITION


class TestConditionParameterTypes:
    @pytest.mark.parametrize(
        "parameter, value_type",
        [
            (ConditionParameter.END_EFFECTOR, Any),
            (ConditionParameter.OBJECT_DESIGNATOR, Body),
            (ConditionParameter.WORLD, World),
            (ConditionParameter.DOOR, type[Door]),
            (ConditionParameter.CONTEXT, Context),
            (ConditionParameter.ARM, Arms),
            (ConditionParameter.GRASP_DESCRIPTION, GraspDescription),
        ],
    )
    def test_a_parameter_declares_the_type_of_its_value(
        self, parameter, value_type
    ) -> None:
        assert parameter.value_type == value_type

    def test_every_parameter_has_its_own_keyword(self) -> None:
        keywords = [parameter.keyword for parameter in ConditionParameter]

        assert len(set(keywords)) == len(ConditionParameter)


class TestGripperConditions:
    def test_a_free_gripper_can_pick_up(self, robot_scene) -> None:
        _, _, parameters = robot_scene

        assert are_action_conditions_met(PickUpAction, parameters) is True

    def test_a_holding_gripper_cannot_pick_up(self, robot_scene) -> None:
        world, _, parameters = robot_scene
        hold_body(world, parameters)

        assert are_action_conditions_met(PickUpAction, parameters) is False

    def test_a_free_gripper_cannot_place(self, robot_scene) -> None:
        _, _, parameters = robot_scene

        assert are_action_conditions_met(PlaceAction, parameters) is False

    def test_a_holding_gripper_can_place(self, robot_scene) -> None:
        world, _, parameters = robot_scene
        hold_body(world, parameters)

        assert are_action_conditions_met(PlaceAction, parameters) is True


class TestContainerConditions:
    def test_a_closed_drawer_can_be_opened(self, robot_scene) -> None:
        _, _, parameters = robot_scene

        assert are_action_conditions_met(OpenAction, parameters) is True

    def test_an_open_drawer_cannot_be_opened(self, robot_scene) -> None:
        world, _, parameters = robot_scene
        open_drawer(world)

        assert are_action_conditions_met(OpenAction, parameters) is False

    def test_a_closed_drawer_cannot_be_closed(self, robot_scene) -> None:
        _, _, parameters = robot_scene

        assert are_action_conditions_met(CloseAction, parameters) is False

    def test_an_open_drawer_can_be_closed(self, robot_scene) -> None:
        world, _, parameters = robot_scene
        open_drawer(world)

        assert are_action_conditions_met(CloseAction, parameters) is True

    def test_a_holding_gripper_cannot_open_a_closed_drawer(self, robot_scene) -> None:
        world, _, parameters = robot_scene
        hold_body(world, parameters)

        assert are_action_conditions_met(OpenAction, parameters) is False


class TestNavigationCondition:
    @pytest.mark.parametrize("object_is_reachable", [True, False])
    def test_navigating_is_needed_when_the_object_is_out_of_reach(
        self, robot_scene, object_is_reachable
    ) -> None:
        world, context, parameters = robot_scene
        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            parameters[ConditionParameter.END_EFFECTOR],
        )
        values = {
            ConditionParameter.CONTEXT: context,
            ConditionParameter.ARM: Arms.LEFT,
            ConditionParameter.OBJECT_DESIGNATOR: world.get_body_by_name(
                MILK_BODY_NAME
            ),
            ConditionParameter.GRASP_DESCRIPTION: grasp_description,
        }

        with patch.object(
            IsObjectReachableBy, "__call__", return_value=object_is_reachable
        ):
            assert are_action_conditions_met(NavigateAction, values) is (
                not object_is_reachable
            )

    def test_the_condition_is_the_opposite_of_the_real_reachability_check(
        self, robot_scene
    ) -> None:
        world, context, parameters = robot_scene
        milk = world.get_body_by_name(MILK_BODY_NAME)
        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            parameters[ConditionParameter.END_EFFECTOR],
        )
        values = {
            ConditionParameter.CONTEXT: context,
            ConditionParameter.ARM: Arms.LEFT,
            ConditionParameter.OBJECT_DESIGNATOR: milk,
            ConditionParameter.GRASP_DESCRIPTION: grasp_description,
        }

        assert are_action_conditions_met(NavigateAction, values) is not (
            IsObjectReachableBy(
                context=context,
                arm=Arms.LEFT,
                object_designator=milk,
                grasp_description=grasp_description,
            )()
        )


class TestMisuse:
    def test_a_missing_parameter_raises(self) -> None:
        with pytest.raises(MissingConditionParameter) as caught:
            are_action_conditions_met(PickUpAction, {})

        assert caught.value.parameter is ConditionParameter.END_EFFECTOR

    def test_an_action_without_conditions_raises(self) -> None:
        with pytest.raises(ActionWithoutConditions):
            are_action_conditions_met(ParkArmsAction, {})


# %% selected_objects


@pytest.fixture
def feasibility_graph(apartment_world_pr2_copy_with_context) -> ActionFeasibilityGraph:
    """
    :return: A graph of a milk and a drawer, with one action that suits each of them.
    """
    world, _, context = apartment_world_pr2_copy_with_context
    return ActionFeasibilityGraph(
        object_names=[MILK_BODY_NAME, DRAWER_BODY_NAME],
        actions=[PickUpAction, OpenAction],
        world=world,
        context=context,
    )


class TestSelectedObjects:
    def test_every_action_lists_the_objects_it_is_valid_for(
        self, feasibility_graph
    ) -> None:
        assert feasibility_graph.selected_objects() == {
            action: [
                object_name
                for object_name in feasibility_graph.object_names
                if action.__name__
                in object_action_validation(object_name, feasibility_graph.world)
            ]
            for action in feasibility_graph.actions
        }

    def test_an_object_is_left_out_of_an_action_not_valid_for_it(
        self, feasibility_graph
    ) -> None:
        assert MILK_BODY_NAME not in feasibility_graph.selected_objects()[OpenAction]

    def test_an_object_is_listed_under_an_action_valid_for_it(
        self, feasibility_graph
    ) -> None:
        assert MILK_BODY_NAME in feasibility_graph.selected_objects()[PickUpAction]

    def test_an_action_valid_for_no_object_has_an_empty_list(
        self, feasibility_graph
    ) -> None:
        graph = replace(
            feasibility_graph, object_names=[MILK_BODY_NAME], actions=[OpenAction]
        )

        assert graph.selected_objects() == {OpenAction: []}

    def test_an_included_action_lists_every_object(self, feasibility_graph) -> None:
        selected = feasibility_graph.selected_objects(action_included=[OpenAction])

        assert selected[OpenAction] == feasibility_graph.object_names

    def test_an_included_action_leaves_the_other_actions_filtered(
        self, feasibility_graph
    ) -> None:
        selected = feasibility_graph.selected_objects(action_included=[OpenAction])

        assert (
            selected[PickUpAction] == feasibility_graph.selected_objects()[PickUpAction]
        )

    def test_an_included_action_outside_the_actions_is_added(
        self, feasibility_graph
    ) -> None:
        selected = feasibility_graph.selected_objects(action_included=[NavigateAction])

        assert selected[NavigateAction] == feasibility_graph.object_names

    def test_changing_an_included_list_leaves_the_object_names_alone(
        self, feasibility_graph
    ) -> None:
        selected = feasibility_graph.selected_objects(action_included=[OpenAction])

        selected[OpenAction].clear()

        assert feasibility_graph.object_names == [MILK_BODY_NAME, DRAWER_BODY_NAME]


# %% _is_action_executable


def arm_and_direction_of(executable_action) -> tuple:
    """
    :return: The arm and the approach direction an executable action was checked with.
    """
    values = executable_action.values
    return (
        values[ConditionParameter.ARM],
        values[ConditionParameter.GRASP_DESCRIPTION].approach_direction,
    )


ALL_COMBINATIONS = set(product(ARMS, APPROACH_DIRECTIONS))
"""
Every arm with every approach direction.
"""


class TestIsActionExecutable:
    def test_a_free_robot_can_pick_up_with_every_combination(
        self, feasibility_graph
    ) -> None:
        results = feasibility_graph._is_action_executable(PickUpAction, MILK_BODY_NAME)

        assert len(results) == len(ALL_COMBINATIONS)
        assert set(map(arm_and_direction_of, results)) == ALL_COMBINATIONS

    def test_a_result_names_its_action(self, feasibility_graph) -> None:
        results = feasibility_graph._is_action_executable(PickUpAction, MILK_BODY_NAME)

        assert {result.action for result in results} == {PickUpAction}

    def test_a_result_holds_the_parameters_it_was_checked_with(
        self, feasibility_graph
    ) -> None:
        graph = feasibility_graph

        for result in graph._is_action_executable(PickUpAction, MILK_BODY_NAME):
            arm = result.values[ConditionParameter.ARM]
            end_effector = ViewManager.get_end_effector_view(arm, graph.context.robot)

            assert result.values[
                ConditionParameter.OBJECT_DESIGNATOR
            ] is graph.world.get_body_by_name(MILK_BODY_NAME)
            assert result.values[ConditionParameter.WORLD] is graph.world
            assert result.values[ConditionParameter.CONTEXT] is graph.context
            assert result.values[ConditionParameter.DOOR] is Door
            assert result.values[ConditionParameter.END_EFFECTOR] is end_effector
            assert (
                result.values[ConditionParameter.GRASP_DESCRIPTION].end_effector
                is end_effector
            )

    def test_only_the_arm_with_a_free_gripper_can_pick_up(
        self, feasibility_graph
    ) -> None:
        graph = feasibility_graph
        hold_body(
            graph.world,
            {
                ConditionParameter.END_EFFECTOR: ViewManager.get_end_effector_view(
                    Arms.LEFT, graph.context.robot
                )
            },
        )

        results = graph._is_action_executable(PickUpAction, MILK_BODY_NAME)

        assert set(map(arm_and_direction_of, results)) == set(
            product([Arms.RIGHT], APPROACH_DIRECTIONS)
        )

    def test_only_the_arm_holding_a_body_can_place(self, feasibility_graph) -> None:
        graph = feasibility_graph
        hold_body(
            graph.world,
            {
                ConditionParameter.END_EFFECTOR: ViewManager.get_end_effector_view(
                    Arms.LEFT, graph.context.robot
                )
            },
        )

        results = graph._is_action_executable(PlaceAction, MILK_BODY_NAME)

        assert set(map(arm_and_direction_of, results)) == set(
            product([Arms.LEFT], APPROACH_DIRECTIONS)
        )

    def test_a_free_robot_has_nothing_to_place(self, feasibility_graph) -> None:
        assert (
            feasibility_graph._is_action_executable(PlaceAction, MILK_BODY_NAME) == []
        )

    def test_a_closed_drawer_can_be_opened(self, feasibility_graph) -> None:
        results = feasibility_graph._is_action_executable(OpenAction, DRAWER_BODY_NAME)

        assert set(map(arm_and_direction_of, results)) == ALL_COMBINATIONS

    def test_an_open_drawer_cannot_be_opened(self, feasibility_graph) -> None:
        open_drawer(feasibility_graph.world)

        assert (
            feasibility_graph._is_action_executable(OpenAction, DRAWER_BODY_NAME) == []
        )

    @pytest.mark.parametrize(
        "object_is_reachable, expected_combinations",
        [(True, set()), (False, ALL_COMBINATIONS)],
    )
    def test_navigating_is_kept_only_for_an_object_out_of_reach(
        self, feasibility_graph, object_is_reachable, expected_combinations
    ) -> None:
        with patch.object(
            IsObjectReachableBy, "__call__", return_value=object_is_reachable
        ):
            results = feasibility_graph._is_action_executable(
                NavigateAction, MILK_BODY_NAME
            )

        assert set(map(arm_and_direction_of, results)) == expected_combinations

    def test_an_action_without_conditions_raises(self, feasibility_graph) -> None:
        with pytest.raises(ActionWithoutConditions):
            feasibility_graph._is_action_executable(ParkArmsAction, MILK_BODY_NAME)


# %% build_action_graph


class TestBuildActionGraph:
    def test_results_are_sorted_by_action_object_and_arm(
        self, feasibility_graph
    ) -> None:
        graph = feasibility_graph

        assert graph.build_action_graph() == {
            PickUpAction: {
                MILK_BODY_NAME: {
                    arm: [
                        result.values
                        for result in graph._is_action_executable(
                            PickUpAction, MILK_BODY_NAME
                        )
                        if result.values[ConditionParameter.ARM] is arm
                    ]
                    for arm in ARMS
                }
            },
            OpenAction: {
                DRAWER_BODY_NAME: {
                    arm: [
                        result.values
                        for result in graph._is_action_executable(
                            OpenAction, DRAWER_BODY_NAME
                        )
                        if result.values[ConditionParameter.ARM] is arm
                    ]
                    for arm in ARMS
                }
            },
        }

    def test_an_action_only_lists_the_objects_it_is_valid_for(
        self, feasibility_graph
    ) -> None:
        feasibility = feasibility_graph.build_action_graph()

        assert list(feasibility[OpenAction]) == [DRAWER_BODY_NAME]
        assert list(feasibility[PickUpAction]) == [MILK_BODY_NAME]

    def test_every_object_lists_every_arm(self, feasibility_graph) -> None:
        feasibility = feasibility_graph.build_action_graph()

        assert list(feasibility[PickUpAction][MILK_BODY_NAME]) == list(ARMS)

    def test_an_arm_lists_the_parameters_of_every_direction_it_works_from(
        self, feasibility_graph
    ) -> None:
        feasibility = feasibility_graph.build_action_graph()

        left_arm_parameters = feasibility[PickUpAction][MILK_BODY_NAME][Arms.LEFT]

        assert [
            parameters[ConditionParameter.GRASP_DESCRIPTION].approach_direction
            for parameters in left_arm_parameters
        ] == list(APPROACH_DIRECTIONS)
        assert {
            parameters[ConditionParameter.ARM] for parameters in left_arm_parameters
        } == {Arms.LEFT}

    def test_an_arm_whose_conditions_fail_has_an_empty_list(
        self, feasibility_graph
    ) -> None:
        graph = feasibility_graph
        hold_body(
            graph.world,
            {
                ConditionParameter.END_EFFECTOR: ViewManager.get_end_effector_view(
                    Arms.LEFT, graph.context.robot
                )
            },
        )

        feasibility = graph.build_action_graph()

        assert feasibility[PickUpAction][MILK_BODY_NAME][Arms.LEFT] == []
        assert len(feasibility[PickUpAction][MILK_BODY_NAME][Arms.RIGHT]) == len(
            APPROACH_DIRECTIONS
        )

    def test_an_included_action_lists_every_object(self, feasibility_graph) -> None:
        with patch.object(IsObjectReachableBy, "__call__", return_value=False):
            feasibility = feasibility_graph.build_action_graph(
                action_included=[NavigateAction]
            )

        assert list(feasibility[NavigateAction]) == feasibility_graph.object_names

    def test_an_action_without_objects_is_left_out(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        graph = ActionFeasibilityGraph(
            object_names=[MILK_BODY_NAME],
            actions=[PickUpAction, OpenAction],
            world=world,
            context=context,
        )

        assert list(graph.build_action_graph()) == [PickUpAction]

    def test_an_object_both_arms_fail_for_is_left_out(self, feasibility_graph) -> None:
        graph = feasibility_graph
        executable_actions = graph._is_action_executable

        def nothing_executable_on_drawer(action, object_name):
            if object_name == DRAWER_BODY_NAME:
                return []
            return executable_actions(action, object_name)

        with patch.object(
            graph, "_is_action_executable", side_effect=nothing_executable_on_drawer
        ):
            feasibility = graph.build_action_graph(action_included=[PickUpAction])

        assert list(feasibility[PickUpAction]) == [MILK_BODY_NAME]

    def test_an_action_whose_objects_all_fail_is_left_out(
        self, feasibility_graph
    ) -> None:
        graph = feasibility_graph

        with patch.object(graph, "_is_action_executable", return_value=[]):
            feasibility = graph.build_action_graph()

        assert feasibility == {}


# %% visualize


def condition_values(arm, approach_direction) -> dict:
    """
    :return: The condition values of one arm and approach direction.
    """
    return {
        ConditionParameter.ARM: arm,
        ConditionParameter.GRASP_DESCRIPTION: GraspDescription(
            approach_direction, VerticalAlignment.NoAlignment, None
        ),
    }


@pytest.fixture
def partly_empty_feasibility() -> dict:
    """
    :return: A feasibility with an empty arm, an object with only empty arms, and an
        action with no objects.
    """
    return {
        PickUpAction: {
            MILK_BODY_NAME: {
                Arms.RIGHT: [
                    condition_values(Arms.RIGHT, ApproachDirection.FRONT),
                    condition_values(Arms.RIGHT, ApproachDirection.BACK),
                ],
                Arms.LEFT: [],
            }
        },
        OpenAction: {DRAWER_BODY_NAME: {Arms.RIGHT: [], Arms.LEFT: []}},
        PlaceAction: {},
    }


def captions_of_kind(tree, kind) -> list[str]:
    """
    :return: The captions of the nodes of ``kind``, in the order they were added.
    """
    return [node.caption for node in tree.nodes() if node.kind is kind]


def children_captions(tree, caption) -> list[str]:
    """
    :return: The captions of the children of the node captioned ``caption``.
    """
    index = next(i for i in tree.node_indices() if tree[i].caption == caption)
    return [tree[child].caption for child in sorted(tree.successor_indices(index))]


class TestFeasibilityTree:
    def test_every_action_with_an_executable_arm_is_a_node(
        self, partly_empty_feasibility
    ) -> None:
        tree = feasibility_tree(partly_empty_feasibility)

        assert captions_of_kind(tree, FeasibilityNodeKind.ACTION) == [
            PickUpAction.__name__
        ]

    def test_an_object_whose_arms_are_all_empty_is_left_out(
        self, partly_empty_feasibility
    ) -> None:
        tree = feasibility_tree(partly_empty_feasibility)

        assert captions_of_kind(tree, FeasibilityNodeKind.OBJECT) == [MILK_BODY_NAME]

    def test_an_empty_arm_is_left_out(self, partly_empty_feasibility) -> None:
        tree = feasibility_tree(partly_empty_feasibility)

        assert captions_of_kind(tree, FeasibilityNodeKind.ARM) == [Arms.RIGHT.name]

    def test_an_action_points_to_its_objects(self, partly_empty_feasibility) -> None:
        tree = feasibility_tree(partly_empty_feasibility)

        assert children_captions(tree, PickUpAction.__name__) == [MILK_BODY_NAME]

    def test_an_object_points_to_its_arms(self, partly_empty_feasibility) -> None:
        tree = feasibility_tree(partly_empty_feasibility)

        assert children_captions(tree, MILK_BODY_NAME) == [Arms.RIGHT.name]

    def test_an_arm_points_to_the_directions_it_works_from(
        self, partly_empty_feasibility
    ) -> None:
        tree = feasibility_tree(partly_empty_feasibility)

        assert children_captions(tree, Arms.RIGHT.name) == [
            ApproachDirection.FRONT.name,
            ApproachDirection.BACK.name,
        ]

    def test_an_action_is_drawn_as_a_box(self, partly_empty_feasibility) -> None:
        tree = feasibility_tree(partly_empty_feasibility)

        assert {
            node.shape
            for node in tree.nodes()
            if node.kind is FeasibilityNodeKind.ACTION
        } == {FEASIBILITY_SHAPES[FeasibilityNodeKind.ACTION]}
        assert FEASIBILITY_SHAPES[FeasibilityNodeKind.ACTION] == "box"

    def test_a_feasibility_with_nothing_executable_is_an_empty_tree(self) -> None:
        tree = feasibility_tree(
            {OpenAction: {DRAWER_BODY_NAME: {Arms.RIGHT: [], Arms.LEFT: []}}}
        )

        assert tree.num_nodes() == 0

    def test_a_built_feasibility_draws_every_action_it_can_execute(
        self, feasibility_graph
    ) -> None:
        tree = feasibility_tree(feasibility_graph.build_action_graph())

        assert captions_of_kind(tree, FeasibilityNodeKind.ACTION) == [
            PickUpAction.__name__,
            OpenAction.__name__,
        ]


class TestFeasibilityPage:
    def test_the_page_serves_every_node_with_its_shape(
        self, partly_empty_feasibility
    ) -> None:
        visualizer = feasibility_visualizer(
            partly_empty_feasibility, open_browser=False
        )
        client = visualizer.build_application().test_client()

        nodes = client.get("/graph").get_json()["nodes"]

        assert [(node["label"], node["shape"]) for node in nodes] == [
            (node.label, node.shape)
            for node in feasibility_tree(partly_empty_feasibility).nodes()
        ]

    def test_the_page_is_the_expandable_tree(self, partly_empty_feasibility) -> None:
        visualizer = feasibility_visualizer(
            partly_empty_feasibility, open_browser=False
        )

        assert visualizer.template == FEASIBILITY_TREE_TEMPLATE
        assert (
            visualizer.build_application().test_client().get("/").status_code
            == HTTPStatus.OK
        )


class TestEndNodeParameters:
    def test_an_end_node_lists_its_condition_parameters(
        self, partly_empty_feasibility
    ) -> None:
        tree = feasibility_tree(partly_empty_feasibility)
        front = next(
            node
            for node in tree.nodes()
            if node.caption == ApproachDirection.FRONT.name
        )

        assert front.details == (
            f"{ConditionParameter.ARM.keyword}: {Arms.RIGHT.name}",
            f"{ConditionParameter.GRASP_DESCRIPTION.keyword}: "
            f"{ApproachDirection.FRONT.name}, {VerticalAlignment.NoAlignment.name}",
        )

    def test_an_end_node_label_shows_its_direction_then_its_parameters(
        self, partly_empty_feasibility
    ) -> None:
        tree = feasibility_tree(partly_empty_feasibility)
        front = next(
            node
            for node in tree.nodes()
            if node.caption == ApproachDirection.FRONT.name
        )

        assert front.label.splitlines() == [front.caption, *front.details]

    def test_a_node_above_the_end_shows_only_its_caption(
        self, partly_empty_feasibility
    ) -> None:
        tree = feasibility_tree(partly_empty_feasibility)

        assert {
            node.label
            for node in tree.nodes()
            if node.kind is not FeasibilityNodeKind.APPROACH_DIRECTION
        } == {PickUpAction.__name__, MILK_BODY_NAME, Arms.RIGHT.name}

    def test_every_parameter_of_a_built_feasibility_is_named_readably(
        self, feasibility_graph
    ) -> None:
        graph = feasibility_graph
        tree = feasibility_tree(graph.build_action_graph())
        end_node = next(
            node
            for node in tree.nodes()
            if node.kind is FeasibilityNodeKind.APPROACH_DIRECTION
        )
        end_effector = ViewManager.get_end_effector_view(
            Arms.RIGHT, graph.context.robot
        )

        assert end_node.details == (
            f"{ConditionParameter.END_EFFECTOR.keyword}: {end_effector.name.name}",
            f"{ConditionParameter.OBJECT_DESIGNATOR.keyword}: {MILK_BODY_NAME}",
            f"{ConditionParameter.WORLD.keyword}: {graph.world.name}",
            f"{ConditionParameter.DOOR.keyword}: {Door.__name__}",
            f"{ConditionParameter.CONTEXT.keyword}: {graph.context.robot.name.name}",
            f"{ConditionParameter.ARM.keyword}: {Arms.RIGHT.name}",
            f"{ConditionParameter.GRASP_DESCRIPTION.keyword}: "
            f"{ApproachDirection.FRONT.name}, {VerticalAlignment.NoAlignment.name}",
        )
