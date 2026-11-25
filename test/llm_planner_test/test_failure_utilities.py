"""
Tests for :mod:`llm_planner.env_reasoner.failure_handler.failure_utilities`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from coraplex.datastructures.enums import Arms
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from llm_planner.env_reasoner.failure_handler import (
    failure_utilities as failure_utilities_module,
)
from llm_planner.env_planner.sequence_execution import TripletExecution
from coraplex.view_manager import ViewManager
from llm_planner.env_reasoner.failure_handler.failure_utilities import (
    MAX_QUERY_ATTEMPTS,
    NEAR_DISTANCE,
    MalformedRecoveryResponse,
    RecoveryQueryFailed,
    UnknownActionObjectPair,
    check_recovery_response,
    ActionProcessor,
    ContextPerformer,
    get_action_descriptions,
    ground_action,
    query_recovery_actions,
    NO_HELD_BODY,
    NO_NEAR_OBJECT,
    RobotState,
    WorldWithoutRobot,
    get_action_object_pair,
    get_robot_states,
    remove_reversed_duplicates,
)
from coraplex.plans.plan_node import ActionLike
from coraplex.robot_plans.actions.base import ActionDescription
from llm_planner.env_planner.action_descriptors import plan_grounding
from llm_planner.env_planner.execution_outcome import ExecutionOutcome
from llm_planner.env_reasoner.action_constraints import ConditionKind
from llm_planner.env_planner.triplet_grounding import GroundedTriplet
from llm_planner.env_reasoner.reasoner_graphs.AFG import ConditionParameter
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

MILK_BODY_NAME = "milk.stl"
"""
An annotated apartment object.
"""

DRAWER_BODY_NAME = "cabinet10_drawer_top"
"""
An apartment drawer.
"""

FEASIBILITY = {
    PickUpAction: {MILK_BODY_NAME: {Arms.RIGHT: [{}], Arms.LEFT: []}},
    OpenAction: {
        DRAWER_BODY_NAME: {Arms.RIGHT: [{}], Arms.LEFT: [{}]},
        MILK_BODY_NAME: {Arms.RIGHT: [], Arms.LEFT: [{}]},
    },
}
"""
A feasibility result with one object for one action and two for another.
"""

HELD_BODY_NAME = "held_body"
"""
Name of the body a gripper holds in these tests.
"""


# %% action object pairs


class TestActionObjectPairs:
    def test_every_object_of_every_action_is_one_pair(self) -> None:
        assert get_action_object_pair(FEASIBILITY) == [
            f"{PickUpAction.__name__},{MILK_BODY_NAME}",
            f"{OpenAction.__name__},{DRAWER_BODY_NAME}",
            f"{OpenAction.__name__},{MILK_BODY_NAME}",
        ]

    def test_no_feasible_actions_give_no_pairs(self) -> None:
        assert get_action_object_pair({}) == []


# %% robot states


def place_robot_next_to(world, robot, body_name: str, offset: float) -> None:
    """
    Put the robot's root at the floor below a body, moved ``offset`` along x.
    """
    position = world.get_body_by_name(body_name).global_pose.to_position()
    robot.set_root_pose(
        Pose.from_xyz_rpy(
            position.x + offset, position.y, 0.0, reference_frame=world.root
        )
    )


def hold_body(world, robot, arm: Arms) -> None:
    """
    Attach a new body to the tool frame of the arm's gripper, so the gripper holds it.
    """
    end_effector = ViewManager.get_end_effector_view(arm, robot)
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=end_effector.tool_frame,
                child=Body(name=PrefixedName(HELD_BODY_NAME)),
            )
        )


class TestHeldBodies:
    def test_free_grippers_hold_nothing(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, _ = apartment_world_pr2_copy_with_context

        assert get_robot_states(world, []).held_bodies == {}

    def test_a_holding_gripper_reports_the_body_under_its_arm(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, robot, _ = apartment_world_pr2_copy_with_context
        hold_body(world, robot, Arms.LEFT)

        assert get_robot_states(world, []).held_bodies == {Arms.LEFT: HELD_BODY_NAME}


class TestNearObjects:
    def test_an_object_beside_the_robot_is_near(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, robot, _ = apartment_world_pr2_copy_with_context
        place_robot_next_to(world, robot, MILK_BODY_NAME, NEAR_DISTANCE / 2)

        assert get_robot_states(world, [MILK_BODY_NAME]).near_objects == [
            MILK_BODY_NAME
        ]

    def test_an_object_beyond_the_threshold_is_not_near(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, robot, _ = apartment_world_pr2_copy_with_context
        place_robot_next_to(world, robot, MILK_BODY_NAME, NEAR_DISTANCE * 2)

        assert get_robot_states(world, [MILK_BODY_NAME]).near_objects == []

    def test_the_threshold_can_be_given(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, robot, _ = apartment_world_pr2_copy_with_context
        place_robot_next_to(world, robot, MILK_BODY_NAME, NEAR_DISTANCE * 2)

        assert get_robot_states(
            world, [MILK_BODY_NAME], threshold=NEAR_DISTANCE * 3
        ).near_objects == [MILK_BODY_NAME]


def test_the_default_threshold_is_one_metre() -> None:
    assert NEAR_DISTANCE == 1.0


def test_a_world_without_a_robot_raises(apartment_world_copy) -> None:
    with pytest.raises(WorldWithoutRobot):
        get_robot_states(apartment_world_copy, [])


# %% reversed duplicates

SUPPORT = "Support"
"""
A relation name.
"""

NEAR = "Near"
"""
Another relation name.
"""


class TestReversedDuplicates:
    def test_the_reversed_relation_is_dropped(self) -> None:
        assert remove_reversed_duplicates(
            [
                [MILK_BODY_NAME, NEAR, DRAWER_BODY_NAME],
                [DRAWER_BODY_NAME, NEAR, MILK_BODY_NAME],
            ]
        ) == [[MILK_BODY_NAME, NEAR, DRAWER_BODY_NAME]]

    def test_a_different_relation_between_the_same_objects_is_kept(self) -> None:
        relations = [
            [MILK_BODY_NAME, NEAR, DRAWER_BODY_NAME],
            [DRAWER_BODY_NAME, SUPPORT, MILK_BODY_NAME],
        ]

        assert remove_reversed_duplicates(relations) == relations

    def test_an_exact_repeat_is_dropped(self) -> None:
        assert remove_reversed_duplicates(
            [
                [MILK_BODY_NAME, SUPPORT, DRAWER_BODY_NAME],
                [MILK_BODY_NAME, SUPPORT, DRAWER_BODY_NAME],
            ]
        ) == [[MILK_BODY_NAME, SUPPORT, DRAWER_BODY_NAME]]


# %% robot state in words


class TestRobotStateDescription:
    def test_each_holding_arm_is_one_line_naming_the_arm_and_the_body(self) -> None:
        held_lines = RobotState(held_bodies={Arms.LEFT: MILK_BODY_NAME}).description()[
            :-1
        ]

        assert len(held_lines) == 1
        assert Arms.LEFT.name in held_lines[0]
        assert MILK_BODY_NAME in held_lines[0]

    def test_free_grippers_are_said_to_hold_nothing(self) -> None:
        assert RobotState().description()[0] == NO_HELD_BODY

    def test_the_near_objects_are_the_last_line(self) -> None:
        last_line = RobotState(
            near_objects=[MILK_BODY_NAME, DRAWER_BODY_NAME]
        ).description()[-1]

        assert MILK_BODY_NAME in last_line
        assert DRAWER_BODY_NAME in last_line

    def test_no_near_object_is_said_so(self) -> None:
        assert RobotState().description()[-1] == NO_NEAR_OBJECT


# %% recovery response


PICK_UP_MILK = f"{PickUpAction.__name__},{MILK_BODY_NAME}"
"""
An offered action and object pair.
"""

OPEN_DRAWER = f"{OpenAction.__name__},{DRAWER_BODY_NAME}"
"""
Another offered action and object pair.
"""

OFFERED_PAIRS = [PICK_UP_MILK, OPEN_DRAWER]
"""
The pairs the language model may choose from.
"""

VALID_RESPONSE = f'["{OPEN_DRAWER}", "{PICK_UP_MILK}"]'
"""
An answer choosing two offered pairs.
"""

MALFORMED_RESPONSE = "Open the drawer first."
"""
An answer holding no JSON list.
"""


@dataclass
class ScriptedLanguageModel:
    """
    A language model answering with prepared texts, recording what it was asked.
    """

    answers: list[str]
    """
    The answers, in the order they are given.
    """

    prompts: list[str] = field(default_factory=list)
    """
    Every prompt asked, in order.
    """

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers[len(self.prompts) - 1]


class TestCheckRecoveryResponse:
    def test_a_list_of_offered_pairs_is_returned_in_order(self) -> None:
        assert check_recovery_response(VALID_RESPONSE, OFFERED_PAIRS) == [
            OPEN_DRAWER,
            PICK_UP_MILK,
        ]

    def test_text_around_the_list_is_ignored(self) -> None:
        response = f"Here is the plan:\n```json\n{VALID_RESPONSE}\n```"

        assert check_recovery_response(response, OFFERED_PAIRS) == [
            OPEN_DRAWER,
            PICK_UP_MILK,
        ]

    def test_an_answer_without_a_list_is_malformed(self) -> None:
        with pytest.raises(MalformedRecoveryResponse):
            check_recovery_response(MALFORMED_RESPONSE, OFFERED_PAIRS)

    def test_a_list_of_non_text_is_malformed(self) -> None:
        with pytest.raises(MalformedRecoveryResponse):
            check_recovery_response('[["OpenAction", "drawer"]]', OFFERED_PAIRS)

    def test_an_empty_list_is_malformed(self) -> None:
        with pytest.raises(MalformedRecoveryResponse):
            check_recovery_response("[]", OFFERED_PAIRS)

    def test_a_pair_that_was_not_offered_is_unknown(self) -> None:
        with pytest.raises(UnknownActionObjectPair):
            check_recovery_response(
                f'["{CloseAction.__name__},{DRAWER_BODY_NAME}"]', OFFERED_PAIRS
            )


class TestQueryRecoveryActions:
    def test_a_valid_first_answer_is_returned(self) -> None:
        model = ScriptedLanguageModel(answers=[VALID_RESPONSE])

        assert query_recovery_actions(model, "prompt", OFFERED_PAIRS) == [
            OPEN_DRAWER,
            PICK_UP_MILK,
        ]
        assert model.prompts == ["prompt"]

    def test_a_bad_answer_is_asked_again(self) -> None:
        model = ScriptedLanguageModel(answers=[MALFORMED_RESPONSE, VALID_RESPONSE])

        assert query_recovery_actions(model, "prompt", OFFERED_PAIRS) == [
            OPEN_DRAWER,
            PICK_UP_MILK,
        ]
        assert model.prompts == ["prompt", "prompt"]

    def test_bad_answers_on_every_attempt_raise(self) -> None:
        model = ScriptedLanguageModel(answers=[MALFORMED_RESPONSE] * MAX_QUERY_ATTEMPTS)

        with pytest.raises(RecoveryQueryFailed) as failure:
            query_recovery_actions(model, "prompt", OFFERED_PAIRS)

        assert len(model.prompts) == MAX_QUERY_ATTEMPTS
        assert [type(error) for error in failure.value.errors] == [
            MalformedRecoveryResponse
        ] * MAX_QUERY_ATTEMPTS


def test_the_model_is_asked_at_most_three_times() -> None:
    assert MAX_QUERY_ATTEMPTS == 3


# %% action descriptions


class TestActionDescriptions:
    def test_the_pair_s_action_object_and_arms_are_returned(self) -> None:
        assert get_action_descriptions(OPEN_DRAWER, FEASIBILITY) == {
            OpenAction: {DRAWER_BODY_NAME: FEASIBILITY[OpenAction][DRAWER_BODY_NAME]}
        }

    def test_other_objects_of_the_same_action_are_left_out(self) -> None:
        assert list(get_action_descriptions(OPEN_DRAWER, FEASIBILITY)[OpenAction]) == [
            DRAWER_BODY_NAME
        ]

    def test_a_pair_not_in_the_graph_raises(self) -> None:
        with pytest.raises(UnknownActionObjectPair):
            get_action_descriptions(
                f"{CloseAction.__name__},{DRAWER_BODY_NAME}", FEASIBILITY
            )


# %% grounding an action


GROUNDED_BODY = Body(name=PrefixedName(MILK_BODY_NAME))
"""
The body the grounded action is performed on.
"""

GROUNDED_GRASP = object()
"""
Stands in for the grasp the conditions were met with.
"""

CONDITION_VALUES = {
    ConditionParameter.OBJECT_DESIGNATOR: GROUNDED_BODY,
    ConditionParameter.ARM: Arms.RIGHT,
    ConditionParameter.GRASP_DESCRIPTION: GROUNDED_GRASP,
}
"""
Condition values holding every argument a pick up needs.
"""


class TestGroundAction:
    def test_values_with_every_required_argument_are_grounded(self) -> None:
        assert ground_action(PickUpAction, CONDITION_VALUES) == GroundedTriplet(
            action=PickUpAction,
            source_object_body=GROUNDED_BODY,
            arm=Arms.RIGHT,
            grasp_description=GROUNDED_GRASP,
        )

    def test_an_argument_the_action_does_not_need_is_left_out(self) -> None:
        grounded = ground_action(OpenAction, CONDITION_VALUES)
        assert grounded.grasp_description is None

    def test_a_missing_required_argument_gives_none(self) -> None:
        values = {
            parameter: value
            for parameter, value in CONDITION_VALUES.items()
            if parameter is not ConditionParameter.GRASP_DESCRIPTION
        }
        assert ground_action(PickUpAction, values) is None

    def test_a_required_argument_conditions_never_hold_gives_none(self) -> None:
        assert ground_action(PlaceAction, CONDITION_VALUES) is None


# %% processing an action


@dataclass
class RecordsPerformedSteps(ActionProcessor):
    """
    An action processor that records the steps handed to its performing instead of
    running them, so what its post-conditions report can be read without moving a robot.
    """

    performed_steps: list[ActionLike] = field(default_factory=list)
    """
    Every step handed to :meth:`ActionProcessor.perform_step`, in the order it was
    handed over.
    """

    def perform_step(self, step: ActionLike) -> None:
        self.performed_steps.append(step)


def drawer_values(world) -> dict[ConditionParameter, object]:
    """
    :param world: The world holding the annotated drawer.
    :return: Condition values naming the drawer and the left arm.
    """
    return {
        ConditionParameter.OBJECT_DESIGNATOR: world.get_body_by_name(DRAWER_BODY_NAME),
        ConditionParameter.ARM: Arms.LEFT,
    }


def move_drawer(world, position: float) -> None:
    """
    :param world: The world holding the annotated drawer.
    :param position: The joint position to move the drawer to, in metres.
    """
    drawer = world.get_body_by_name(DRAWER_BODY_NAME)
    connection = next(
        connection
        for connection in world.connections
        if connection.child is drawer and isinstance(connection, ActiveConnection1DOF)
    )
    connection.position = position
    world.notify_state_change()


class TestActionProcessor:
    def test_the_planned_steps_are_performed_in_order(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        processor = RecordsPerformedSteps(context=context)
        values = drawer_values(world)

        processor.process(CloseAction, values)

        expected = plan_grounding([ground_action(CloseAction, values)], context)
        assert [type(step) for step in processor.performed_steps] == [
            type(step) for step in expected
        ]

    def test_a_held_post_condition_is_reported_as_performed(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context

        execution = RecordsPerformedSteps(context=context).process(
            CloseAction, drawer_values(world)
        )

        assert execution.outcome is ExecutionOutcome.PERFORMED

    def test_the_execution_reports_the_grounded_action(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        values = drawer_values(world)

        execution = RecordsPerformedSteps(context=context).process(CloseAction, values)

        assert execution.grounded == ground_action(CloseAction, values)
        assert execution.triplet is None

    def test_a_failed_post_condition_is_reported_with_its_check(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        move_drawer(world, 0.4)

        execution = RecordsPerformedSteps(context=context).process(
            CloseAction, drawer_values(world)
        )

        assert execution.outcome is ExecutionOutcome.CONDITION_FAILED
        assert execution.failed_check.kind is ConditionKind.POST
        assert execution.failed_check.action is CloseAction

    def test_an_action_without_steps_is_reported_as_not_planned(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        """
        The milk has no handle to open it by, so opening it plans into no steps.
        """
        world, _, context = apartment_world_pr2_copy_with_context
        values = {
            ConditionParameter.OBJECT_DESIGNATOR: world.get_body_by_name(
                MILK_BODY_NAME
            ),
            ConditionParameter.ARM: Arms.LEFT,
        }
        processor = RecordsPerformedSteps(context=context)

        execution = processor.process(OpenAction, values)

        assert execution.outcome is ExecutionOutcome.NOT_PLANNED
        assert processor.performed_steps == []

    def test_values_missing_a_required_argument_give_none(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        processor = RecordsPerformedSteps(context=context)

        assert processor.process(PlaceAction, drawer_values(world)) is None
        assert processor.performed_steps == []


@dataclass
class RecordsStartsAndSteps:
    """
    An action performer that records the actions started and the steps handed to it
    instead of showing or performing them.
    """

    started: list[TripletExecution] = field(default_factory=list)
    """
    Every execution handed to :meth:`start_action`, in the order it was handed over.
    """

    performed_steps: list[ActionLike] = field(default_factory=list)
    """
    Every step handed to :meth:`perform_action_step`, in the order it was handed over.
    """

    def start_action(self, execution: TripletExecution) -> None:
        self.started.append(execution)

    def perform_action_step(self, step: ActionLike) -> None:
        self.performed_steps.append(step)


PLANNED_STEP = ParkArmsAction(Arms.BOTH)
"""
The one step every action plans into in the performer tests.
"""


class TestActionPerformer:
    def test_a_processor_performs_in_its_own_context_by_default(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context

        assert ActionProcessor(context=context).performer == ContextPerformer(
            context=context
        )

    def test_the_grounded_action_is_started_as_running(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        performer = RecordsStartsAndSteps()
        values = drawer_values(world)

        with patch.object(
            failure_utilities_module, "plan_grounding", return_value=[PLANNED_STEP]
        ):
            ActionProcessor(context=context, performer=performer).process(
                CloseAction, values
            )

        assert performer.started == [
            TripletExecution(
                None, ground_action(CloseAction, values), ExecutionOutcome.RUNNING
            )
        ]

    def test_the_steps_are_performed_by_the_performer(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        performer = RecordsStartsAndSteps()

        with patch.object(
            failure_utilities_module, "plan_grounding", return_value=[PLANNED_STEP]
        ):
            ActionProcessor(context=context, performer=performer).process(
                CloseAction, drawer_values(world)
            )

        assert performer.performed_steps == [PLANNED_STEP]

    def test_values_that_cannot_be_grounded_start_nothing(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        performer = RecordsStartsAndSteps()

        ActionProcessor(context=context, performer=performer).process(
            PlaceAction, drawer_values(world)
        )

        assert performer.started == []
