"""
Tests of checking an action's pre-condition without performing it, for every action
:data:`~llm_planner.env_planner.action_validation.CHECKABLE_ACTIONS` names.
"""

from __future__ import annotations

import inspect
import operator
from copy import deepcopy

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.pose_validator import IsObjectReachableBy
from coraplex.querying.predicates import GripperIsFree, GripperIsNotFree
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
    PouringAction,
)
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.testing import setup_world
from coraplex.view_manager import ViewManager
from llm_planner.env_planner.action_validation import (
    ActionNotCheckable,
    CHECKABLE_ACTIONS,
    ConditionStatement,
    check_post_condition,
    check_pre_condition,
)
from krrood.entity_query_language.factories import and_, evaluate_condition, or_
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import OR, Not
from llm_planner.env_reasoner.action_constraints import ConditionKind
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_body_gripped
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CuttingKnife,
    PouringCup,
    Whisk,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

MILK_BODY_NAME = "milk.stl"
"""
An apartment object far enough from the robot's spawn that it is not reachable.
"""

OCCUPIED_BODY_NAME = "cabinet10"
"""
A solid apartment cabinet, so a navigation target placed on it is occupied.
"""

DRAWER_BODY_NAME = "cabinet10_drawer_top"
"""
An apartment drawer, closed by default (its joint starts at position 0).
"""


def body_named(name: str) -> Body:
    """
    A free-standing body, unattached to any world.

    Used for actions whose pre-condition is the unconditional default and so never
    reads the body's geometry or place in a world.

    :param name: The body's name.
    """
    return Body(name=PrefixedName(name))


@pytest.fixture()
def bare_context() -> Context:
    """
    A context with an empty world and no robot, for a pre-condition that reads
    neither.
    """
    return Context(world=World(), robot=None)


@pytest.fixture(scope="session")
def _pr2_apartment_setup() -> World:
    """
    The apartment world with a PR2 standing at its spawn point, built once per session.
    """
    return setup_world()


@pytest.fixture()
def action_world(_pr2_apartment_setup):
    """
    A private copy of the apartment world, with its PR2 and a context for it.
    """
    world = deepcopy(_pr2_apartment_setup)
    robot = PR2.from_world(world)
    return world, robot, Context(world=world, robot=robot)


# %% actions whose pre-condition is the unconditional default


class TestTriviallySatisfiedActions:
    """
    :class:`CloseAction`, :class:`MixingAction`, :class:`CuttingAction` and
    :class:`PouringAction` declare no ``pre_condition`` of their own, so they inherit
    :meth:`~coraplex.robot_plans.actions.base.ActionDescription.pre_condition`'s
    unconditional ``True`` - always satisfied, with no false statement to report.
    """

    def test_close_action_is_always_satisfied(self, bare_context) -> None:
        action = CloseAction(object_designator=body_named("handle"), arm=Arms.LEFT)

        check = check_pre_condition(action, bare_context)

        assert check.action is CloseAction
        assert check.kind is ConditionKind.PRE
        assert check.satisfied is True
        assert check.false_statement_names == []

    def test_mixing_action_is_always_satisfied(self, bare_context) -> None:
        action = MixingAction(
            container=body_named("bowl"),
            arm=Arms.LEFT,
            tool=Whisk(root=body_named("whisk")),
        )

        check = check_pre_condition(action, bare_context)

        assert check.satisfied is True
        assert check.false_statement_names == []

    def test_cutting_action_is_always_satisfied(self, bare_context) -> None:
        action = CuttingAction(
            object_to_cut=body_named("bread"),
            arm=Arms.LEFT,
            tool=CuttingKnife(root=body_named("knife")),
        )

        check = check_pre_condition(action, bare_context)

        assert check.satisfied is True
        assert check.false_statement_names == []

    def test_pouring_action_is_always_satisfied(self, bare_context) -> None:
        action = PouringAction(
            target_container=body_named("cup"),
            source_container=PouringCup(root=body_named("pot")),
            arm=Arms.LEFT,
        )

        check = check_pre_condition(action, bare_context)

        assert check.satisfied is True
        assert check.false_statement_names == []


# %% actions whose pre-condition reads the world and the robot


class TestNavigateActionPreCondition:
    def test_the_robots_own_pose_is_free(self, action_world) -> None:
        world, robot, context = action_world
        action = NavigateAction(target_location=robot.root.global_pose)

        check = check_pre_condition(action, context)

        assert check.satisfied is True

    def test_a_pose_inside_a_cabinet_is_not_free(self, action_world) -> None:
        world, robot, context = action_world
        cabinet = world.get_body_by_name(OCCUPIED_BODY_NAME)
        target = Pose.from_xyz_rpy(
            float(cabinet.global_pose.x),
            float(cabinet.global_pose.y),
            0.1,
            reference_frame=world.root,
        )

        check = check_pre_condition(NavigateAction(target_location=target), context)

        assert check.satisfied is False


class TestOpenActionPreCondition:
    def test_an_object_far_from_the_robot_is_not_reachable(self, action_world) -> None:
        world, robot, context = action_world
        milk = world.get_body_by_name(MILK_BODY_NAME)
        action = OpenAction(object_designator=milk, arm=Arms.LEFT)

        check = check_pre_condition(action, context)

        assert check.satisfied is False


class TestPickUpActionPreCondition:
    def test_an_object_far_from_the_robot_is_not_reachable(self, action_world) -> None:
        world, robot, context = action_world
        milk = world.get_body_by_name(MILK_BODY_NAME)
        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            ViewManager.get_end_effector_view(Arms.LEFT, robot),
        )
        action = PickUpAction(
            object_designator=milk, arm=Arms.LEFT, grasp_description=grasp_description
        )

        check = check_pre_condition(action, context)

        assert check.satisfied is False
        assert check.false_statement_names == ["IsObjectReachableBy"]


class TestPlaceActionPreCondition:
    def test_placing_with_an_empty_gripper_is_not_satisfied(self, action_world) -> None:
        """
        A place is only satisfied once the object is actually held (see
        :meth:`~coraplex.robot_plans.actions.core.placing.PlaceAction.pre_condition`);
        a fresh robot holding nothing cannot place anything yet.
        """
        world, robot, context = action_world
        milk = world.get_body_by_name(MILK_BODY_NAME)
        action = PlaceAction(
            object_designator=milk,
            target_location=robot.root.global_pose,
            arm=Arms.LEFT,
        )

        check = check_pre_condition(action, context)

        assert check.satisfied is False


# %% post-conditions


class TestTriviallySatisfiedPostConditions:
    """
    :class:`MixingAction`, :class:`CuttingAction` and :class:`PouringAction` declare no
    ``post_condition`` either, so it is the same unconditional ``True`` as their
    pre-condition.
    """

    def test_mixing_action_is_always_satisfied(self, bare_context) -> None:
        action = MixingAction(
            container=body_named("bowl"),
            arm=Arms.LEFT,
            tool=Whisk(root=body_named("whisk")),
        )

        check = check_post_condition(action, bare_context)

        assert check.kind is ConditionKind.POST
        assert check.satisfied is True
        assert check.false_statement_names == []

    def test_cutting_action_is_always_satisfied(self, bare_context) -> None:
        action = CuttingAction(
            object_to_cut=body_named("bread"),
            arm=Arms.LEFT,
            tool=CuttingKnife(root=body_named("knife")),
        )

        check = check_post_condition(action, bare_context)

        assert check.satisfied is True
        assert check.false_statement_names == []

    def test_pouring_action_is_always_satisfied(self, bare_context) -> None:
        action = PouringAction(
            target_container=body_named("cup"),
            source_container=PouringCup(root=body_named("pot")),
            arm=Arms.LEFT,
        )

        check = check_post_condition(action, bare_context)

        assert check.satisfied is True
        assert check.false_statement_names == []


class TestNavigateActionPostCondition:
    def test_the_robot_is_already_at_its_own_pose(self, action_world) -> None:
        world, robot, context = action_world
        action = NavigateAction(target_location=robot.root.global_pose)

        check = check_post_condition(action, context)

        assert check.satisfied is True

    def test_the_robot_is_not_at_a_pose_far_away(self, action_world) -> None:
        world, robot, context = action_world
        far_away = Pose.from_xyz_rpy(0, 0, 0, reference_frame=world.root)

        check = check_post_condition(NavigateAction(target_location=far_away), context)

        assert check.satisfied is False


class TestCloseActionPostCondition:
    def test_a_drawer_that_was_never_opened_is_already_closed(
        self, action_world
    ) -> None:
        world, robot, context = action_world
        drawer = world.get_body_by_name(DRAWER_BODY_NAME)
        action = CloseAction(object_designator=drawer, arm=Arms.LEFT)

        check = check_post_condition(action, context)

        assert check.satisfied is True


class TestOpenActionPostCondition:
    def test_a_drawer_that_was_never_opened_is_not_open(self, action_world) -> None:
        world, robot, context = action_world
        drawer = world.get_body_by_name(DRAWER_BODY_NAME)
        action = OpenAction(object_designator=drawer, arm=Arms.LEFT)

        check = check_post_condition(action, context)

        assert check.satisfied is False


class TestPickUpActionPostCondition:
    def test_an_object_nothing_is_holding_is_not_in_the_gripper(
        self, action_world
    ) -> None:
        world, robot, context = action_world
        milk = world.get_body_by_name(MILK_BODY_NAME)
        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            ViewManager.get_end_effector_view(Arms.LEFT, robot),
        )
        action = PickUpAction(
            object_designator=milk, arm=Arms.LEFT, grasp_description=grasp_description
        )

        check = check_post_condition(action, context)

        assert check.satisfied is False


class TestPlaceActionPostCondition:
    def test_an_object_already_at_the_target_is_placed(self, action_world) -> None:
        """
        The condition only reads the world's current state, so an object already
        sitting at its target and not held reads as placed without anything having
        moved it there.
        """
        world, robot, context = action_world
        milk = world.get_body_by_name(MILK_BODY_NAME)
        action = PlaceAction(
            object_designator=milk, target_location=milk.global_pose, arm=Arms.LEFT
        )

        check = check_post_condition(action, context)

        assert check.satisfied is True


# %% actions outside CHECKABLE_ACTIONS


class TestUnsupportedAction:
    def test_an_action_outside_checkable_actions_is_rejected_for_pre_condition(
        self, bare_context
    ) -> None:
        action = ParkArmsAction(Arms.BOTH)

        with pytest.raises(ActionNotCheckable) as excinfo:
            check_pre_condition(action, bare_context)

        assert excinfo.value.action is action
        assert ParkArmsAction.__name__ in str(excinfo.value)
        for action_type in CHECKABLE_ACTIONS:
            assert action_type.__name__ in str(excinfo.value)

    def test_an_action_outside_checkable_actions_is_rejected_for_post_condition(
        self, bare_context
    ) -> None:
        action = ParkArmsAction(Arms.BOTH)

        with pytest.raises(ActionNotCheckable) as excinfo:
            check_post_condition(action, bare_context)

        assert excinfo.value.action is action


# %% which statement of a condition failed


class TestConditionStatement:
    def test_every_operand_of_a_failed_condition_is_reported(
        self, action_world
    ) -> None:
        """
        A place is allowed either because the gripper is not free or because the object
        is held, so both statements of that ``or_`` are reported with their own truth.
        """
        world, robot, context = action_world
        action = PlaceAction(
            object_designator=world.get_body_by_name(MILK_BODY_NAME),
            target_location=robot.root.global_pose,
            arm=Arms.LEFT,
        )

        statement = check_pre_condition(action, context).statement

        assert statement.name == OR.__name__
        assert statement.satisfied is False
        gripper_is_not_free, object_is_held = statement.operands
        assert gripper_is_not_free.name == Not.__name__
        assert gripper_is_not_free.satisfied is False
        assert [operand.name for operand in gripper_is_not_free.operands] == [
            GripperIsFree.__name__
        ]
        assert gripper_is_not_free.operands[0].satisfied is True
        assert object_is_held.name == is_body_gripped.__name__
        assert object_is_held.satisfied is False

    def test_a_statement_is_described_by_the_documentation_of_what_it_checks(
        self, action_world
    ) -> None:
        """
        The milk stands out of reach, so the open is refused by the reachability
        predicate, which says what it checks.
        """
        world, robot, context = action_world
        action = OpenAction(
            object_designator=world.get_body_by_name(MILK_BODY_NAME), arm=Arms.LEFT
        )

        statement = check_pre_condition(action, context).statement

        reachability = statement.operands[1]
        assert reachability.name == IsObjectReachableBy.__name__
        assert reachability.satisfied is False
        assert reachability.description != ""
        assert " ".join(inspect.getdoc(IsObjectReachableBy).split()).startswith(
            reachability.description
        )

    def test_the_false_statements_are_the_innermost_ones_that_failed(
        self, action_world
    ) -> None:
        """
        A statement combining others is not itself the reason: the place fails because
        the gripper is free and because the milk is not held, not because an ``or_``
        of the two is false.
        """
        world, robot, context = action_world
        action = PlaceAction(
            object_designator=world.get_body_by_name(MILK_BODY_NAME),
            target_location=robot.root.global_pose,
            arm=Arms.LEFT,
        )

        check = check_pre_condition(action, context)

        assert check.false_statement_names == [
            Not.__name__,
            is_body_gripped.__name__,
        ]

    def test_a_statement_reached_through_two_branches_is_reported_once(
        self, action_world
    ) -> None:
        """
        One predicate standing in two branches of a condition is one reason it failed,
        not one reason per branch, while a second predicate that reads alike is its
        own reason.
        """
        _, robot, _ = action_world
        end_effector = ViewManager.get_end_effector_view(Arms.LEFT, robot)
        shared = GripperIsNotFree(end_effector)
        condition = and_(shared, or_(shared, GripperIsNotFree(end_effector)))

        statement = ConditionStatement.from_expression(
            condition, evaluate_condition(condition)
        )

        assert [failure.name for failure in statement.innermost_failures] == [
            GripperIsNotFree.__name__,
            GripperIsNotFree.__name__,
        ]

    def test_two_statements_that_read_alike_are_both_reported(
        self, action_world
    ) -> None:
        """
        An open is refused both because the handle is not in the gripper and because
        the drawer did not move, two comparisons that read as the same operator.
        """
        world, _, context = action_world
        action = OpenAction(
            object_designator=world.get_body_by_name(DRAWER_BODY_NAME), arm=Arms.LEFT
        )

        check = check_post_condition(action, context)

        greater_than = Comparator.operation_name_map[operator.gt]
        assert check.false_statement_names == [
            greater_than,
            allclose.__name__,
            greater_than,
        ]

    def test_a_satisfied_condition_has_nothing_to_report(self, bare_context) -> None:
        check = check_pre_condition(
            CloseAction(object_designator=body_named("handle"), arm=Arms.LEFT),
            bare_context,
        )

        assert check.satisfied is True
        assert check.statement is None
        assert check.false_statements == []

    def test_the_report_shows_every_statement_with_whether_it_holds(
        self, action_world
    ) -> None:
        world, robot, context = action_world
        action = PlaceAction(
            object_designator=world.get_body_by_name(MILK_BODY_NAME),
            target_location=robot.root.global_pose,
            arm=Arms.LEFT,
        )

        report = check_pre_condition(action, context).statement.report()

        assert report == "\n".join(
            [
                f"{OR.__name__}: fails",
                f"    {Not.__name__}: fails",
                f"        {GripperIsFree.__name__}: holds",
                f"    {is_body_gripped.__name__}: fails",
            ]
        )

    def test_a_statement_reads_as_its_name_and_what_it_checks(
        self, action_world
    ) -> None:
        """
        An operator combines other statements rather than checking something itself,
        so it is named without a description, while a predicate carries one.
        """
        world, robot, context = action_world
        action = PlaceAction(
            object_designator=world.get_body_by_name(MILK_BODY_NAME),
            target_location=robot.root.global_pose,
            arm=Arms.LEFT,
        )

        gripper_is_not_free, object_is_held = check_pre_condition(
            action, context
        ).false_statements

        assert gripper_is_not_free.summary == Not.__name__
        assert object_is_held.summary == (
            f"{is_body_gripped.__name__}: {object_is_held.description}"
        )
