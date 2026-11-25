"""
Tests of saying in plain words what the robot does for one grounded action.
"""

from __future__ import annotations

import pytest

from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.querying.predicates import GripperIsFree, GripperIsNotFree
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.view_manager import ViewManager
from llm_planner.env_planner.action_validation import (
    ConditionStatement,
    check_pre_condition,
)
from krrood.entity_query_language.factories import (
    and_,
    evaluate_condition,
    not_,
    or_,
)
from llm_planner.env_planner.execution_outcome import ExecutionOutcome
from llm_planner.env_planner.triplet_generator import Triplet
from llm_planner.env_planner.triplet_grounding import GroundedTriplet, TripletGrounding
from llm_planner.env_reasoner.scene_summary import (
    REPAIR_MARKER,
    ActionSummary,
    interpretation_CS,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Milk,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose

MILK_BODY_NAME = "milk.stl"
"""
An apartment object far enough from the robot's spawn that it is not reachable.
"""

DRAWER_BODY_NAME = "cabinet10_drawer_top"
"""
The apartment drawer the fixture annotates, handle and all.
"""


def grounded_triplet(world, source, target, action) -> GroundedTriplet:
    """
    :param world: The world to ground in.
    :param source: The object the action is performed on.
    :param target: The object it is performed against.
    :param action: The action to perform.
    :return: The grounded triplet.
    """
    return TripletGrounding(world=world).ground(
        Triplet(source_object=source, target_object=target, action=action)
    )


# %% what the robot does


class TestActionSummary:
    def test_a_pick_up_says_the_action_the_object_the_arm_and_the_grasp(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, _ = apartment_world_pr2_copy_with_context
        grounded = grounded_triplet(world, Milk, Drawer, PickUpAction)

        summary = ActionSummary(grounded, ExecutionOutcome.PERFORMED)

        assert summary.description == (
            "Pick up milk.stl with the left arm, approaching from the front: performed"
        )

    def test_an_action_that_takes_hold_of_nothing_says_nothing_about_a_grasp(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, _ = apartment_world_pr2_copy_with_context
        grounded = grounded_triplet(world, Drawer, Drawer, OpenAction)

        summary = ActionSummary(grounded, ExecutionOutcome.NOT_PLANNED)

        assert summary.description == (
            "Open cabinet10_drawer_top with the left arm: not planned"
        )

    def test_a_put_down_says_where_the_object_goes(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, robot, _ = apartment_world_pr2_copy_with_context
        grounded = GroundedTriplet(
            action=PlaceAction,
            source_object_body=world.get_body_by_name(MILK_BODY_NAME),
            arm=Arms.RIGHT,
            target_object_body=world.get_body_by_name(DRAWER_BODY_NAME),
            target_location=Pose.from_xyz_rpy(
                2.4, 2.2, 1.0, reference_frame=world.root
            ),
            grasp_description=GraspDescription(
                ApproachDirection.LEFT,
                VerticalAlignment.TOP,
                ViewManager.get_end_effector_view(Arms.RIGHT, robot),
            ),
        )

        summary = ActionSummary(grounded, ExecutionOutcome.PERFORMED)

        assert summary.description == (
            "Place milk.stl on cabinet10_drawer_top at (2.40, 2.20, 1.00) "
            "with the right arm, approaching from the left, aligned with the top: "
            "performed"
        )

    def test_the_sentence_is_what_the_summary_reads_as(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, _ = apartment_world_pr2_copy_with_context
        summary = ActionSummary(
            grounded_triplet(world, Drawer, Drawer, OpenAction),
            ExecutionOutcome.PERFORMED,
        )

        assert str(summary) == summary.description


# %% repair actions


class TestRepairSummary:
    def test_a_repair_action_is_marked_before_its_outcome(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, _ = apartment_world_pr2_copy_with_context
        grounded = grounded_triplet(world, Drawer, Drawer, OpenAction)
        plain = ActionSummary(grounded, ExecutionOutcome.PERFORMED)

        repair = ActionSummary(grounded, ExecutionOutcome.PERFORMED, is_repair=True)

        assert repair.description == plain.description.replace(
            f": {ExecutionOutcome.PERFORMED}",
            f"{REPAIR_MARKER}: {ExecutionOutcome.PERFORMED}",
        )

    def test_an_action_of_the_plan_is_not_marked(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, _ = apartment_world_pr2_copy_with_context

        summary = ActionSummary(
            grounded_triplet(world, Drawer, Drawer, OpenAction),
            ExecutionOutcome.PERFORMED,
        )

        assert REPAIR_MARKER not in summary.description


# %% why the robot could not do it


class TestFailedActionSummary:
    def test_a_refused_action_says_which_statement_of_its_condition_failed(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        """
        The milk stands out of reach, so the open is refused before it is performed and
        the summary says by which statement.
        """
        world, _, context = apartment_world_pr2_copy_with_context
        milk = world.get_body_by_name(MILK_BODY_NAME)
        check = check_pre_condition(
            OpenAction(object_designator=milk, arm=Arms.LEFT), context
        )

        summary = ActionSummary(
            grounded_triplet(world, Milk, Milk, OpenAction),
            ExecutionOutcome.CONDITION_FAILED,
            check,
        )

        assert summary.description == "\n".join(
            [
                "Open milk.stl with the left arm: condition failed",
                f"    Condition_type: {check.kind.name}",
                "    Statements:",
                *(f"        {reason}" for reason in interpretation_CS(check.statement)),
            ]
        )

    def test_an_action_nothing_refused_says_nothing_about_a_condition(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, _ = apartment_world_pr2_copy_with_context
        summary = ActionSummary(
            grounded_triplet(world, Drawer, Drawer, OpenAction),
            ExecutionOutcome.PERFORMED,
        )

        assert summary.failure_phrase == ""


# %% condition statements in words


def statement_of(condition) -> ConditionStatement:
    """
    :param condition: The condition to read.
    :return: The condition as a statement, evaluated.
    """
    return ConditionStatement.from_expression(condition, evaluate_condition(condition))


def reading_of(predicate, satisfied: bool) -> str:
    """
    :param predicate: The predicate class a statement stands for.
    :param satisfied: Whether it holds.
    :return: The statement as the interpretation reads it.
    """
    return f"{predicate.__name__}:{satisfied}"


@pytest.fixture()
def end_effector(apartment_world_pr2_copy_with_context):
    """
    The left end effector of the robot, whose gripper starts out free.
    """
    _, robot, _ = apartment_world_pr2_copy_with_context
    return ViewManager.get_end_effector_view(Arms.LEFT, robot)


class TestStatementInterpretation:
    def test_every_predicate_of_the_statement_reads_as_its_name_and_whether_it_holds(
        self, end_effector
    ) -> None:
        statement = statement_of(
            and_(GripperIsFree(end_effector), GripperIsNotFree(end_effector))
        )

        assert interpretation_CS(statement) == [
            reading_of(GripperIsFree, True),
            reading_of(GripperIsNotFree, False),
        ]

    def test_the_predicates_inside_an_or_are_read_one_by_one(
        self, end_effector
    ) -> None:
        statement = statement_of(
            and_(
                GripperIsFree(end_effector),
                or_(GripperIsNotFree(end_effector), GripperIsFree(end_effector)),
            )
        )

        assert interpretation_CS(statement) == [
            reading_of(GripperIsFree, True),
            reading_of(GripperIsNotFree, False),
            reading_of(GripperIsFree, True),
        ]

    def test_the_predicate_inside_a_not_is_read(self, end_effector) -> None:
        statement = statement_of(
            and_(GripperIsNotFree(end_effector), not_(GripperIsFree(end_effector)))
        )

        assert interpretation_CS(statement) == [
            reading_of(GripperIsNotFree, False),
            reading_of(GripperIsFree, True),
        ]

    def test_a_statement_without_operands_has_no_reading(self) -> None:
        predicate = ConditionStatement(name="is_open", description="", satisfied=True)

        assert interpretation_CS(predicate) == []
