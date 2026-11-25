"""
utilities functions used for failure handling
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

from typing_extensions import TYPE_CHECKING, Optional, Protocol

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import execute_single
from coraplex.querying.predicates import GripperIsNotFree
from coraplex.view_manager import ViewManager
from llm_planner.env_planner.action_descriptors import plan_grounding
from llm_planner.env_planner.action_validation import (
    CHECKABLE_ACTIONS,
    ConditionCheck,
    check_post_condition,
)
from llm_planner.env_planner.execution_outcome import ExecutionOutcome
from llm_planner.env_planner.sequence_execution import TripletExecution
from llm_planner.env_planner.triplet_grounding import ActionArgument, GroundedTriplet
from llm_planner.env_reasoner.action_constraints import get_action_constraints_names
from llm_planner.env_reasoner.reasoner_graphs.AFG import ARMS
from semantic_digital_twin.reasoning.predicates import (
    compute_euclidean_planar_distance,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import Vector3

if TYPE_CHECKING:
    from llm_planner.env_reasoner.LLM_utilities import LanguageModel
    from coraplex.plans.plan_node import ActionLike
    from coraplex.robot_plans.actions.base import ActionDescription
    from llm_planner.env_reasoner.reasoner_graphs.AFG import (
        ActionFeasibility,
        ConditionValues,
    )
    from semantic_digital_twin.world import World

# %% action object pairs


ACTION_OBJECT_SEPARATOR = ","
"""
What separates the action name from the object name in an action and object pair.
"""


def get_action_object_pair(feasibility: ActionFeasibility) -> list[str]:
    """
    :param feasibility: The feasible actions, by action, object and arm.
    :return: Every action with every object it is feasible for, each as
        ``action_name,object_name``, in the order of ``feasibility``.
    """
    return [
        f"{action.__name__}{ACTION_OBJECT_SEPARATOR}{object_name}"
        for action, objects in feasibility.items()
        for object_name in objects
    ]


NEAR_DISTANCE = 1.0
"""
Distance in metres on the floor plane below which an object counts as near the robot.
"""


NO_HELD_BODY = "No gripper holds anything."
"""
The robot state line when every gripper is free.
"""

NO_NEAR_OBJECT = "No object is near the robot."
"""
The robot state line when none of the given objects is near the robot.
"""


class WorldWithoutRobot(Exception):
    """
    Raised when a world holds no robot to report the state of.
    """

    def __init__(self, world: World) -> None:
        self.world = world
        """
        The world without a robot.
        """
        super().__init__(f"The world {world.name} holds no robot.")


@dataclass
class RobotState:
    """
    What the robot holds and which objects it stands near.
    """

    held_bodies: dict[Arms, str] = field(default_factory=dict)
    """
    Name of the body each holding arm holds; free arms are left out.
    """

    near_objects: list[str] = field(default_factory=list)
    """
    Names of the given objects near the robot, in the order they were given.
    """

    def description(self) -> list[str]:
        """
        :return: One line per holding arm, or :data:`NO_HELD_BODY`, followed by one
            line naming the near objects, or :data:`NO_NEAR_OBJECT`.
        """
        held_lines = [
            f"The {arm.name} arm holds {body_name}."
            for arm, body_name in self.held_bodies.items()
        ] or [NO_HELD_BODY]
        if not self.near_objects:
            return held_lines + [NO_NEAR_OBJECT]
        return held_lines + [f"The robot is near: {', '.join(self.near_objects)}."]


def get_robot_states(
    world: World, object_names: list[str], threshold: float = NEAR_DISTANCE
) -> RobotState:
    """
    :param world: The world holding the robot.
    :param object_names: Names of the bodies to check the robot's distance to.
    :param threshold: Distance on the floor plane below which an object is near.
    :return: What each arm of the robot holds and which objects are near it.
    """
    robots = world.get_semantic_annotations_by_type(AbstractRobot)
    if not robots:
        raise WorldWithoutRobot(world)
    robot = robots[0]

    held_bodies = {}
    for arm in ARMS:
        end_effector = ViewManager.get_end_effector_view(arm, robot)
        if not GripperIsNotFree(end_effector)():
            continue
        held_body = world.compute_child_kinematic_structure_entities(
            end_effector.tool_frame
        )[0]
        held_bodies[arm] = held_body.name.name

    near_objects = [
        object_name
        for object_name in object_names
        if compute_euclidean_planar_distance(
            robot.root, world.get_body_by_name(object_name), Vector3.Z()
        )
        < threshold
    ]
    return RobotState(held_bodies=held_bodies, near_objects=near_objects)


# %% reversed duplicates


def remove_reversed_duplicates(relations: list[list[str]]) -> list[list[str]]:
    """
    Keep one of each relation between the same two objects, whichever way round it is
    written.

    :param relations: Relations as ``[source_name, relation_name, target_name]``.
    :return: ``relations`` without the ones whose relation and pair of objects appear
        earlier, in either direction.
    """
    seen = set()
    unique_relations = []
    for source_name, relation_name, target_name in relations:
        key = (relation_name, frozenset((source_name, target_name)))
        if key in seen:
            continue
        seen.add(key)
        unique_relations.append([source_name, relation_name, target_name])
    return unique_relations


# %% recovery response


MAX_QUERY_ATTEMPTS = 3
"""
How often the language model is asked before its answers are given up on.
"""

JSON_LIST_START = "["
"""
The character a JSON list starts with.
"""


NO_ENTRY = "none"
"""
What a list in the prompt reads as when it has no entries.
"""


def bulleted_lines(lines: Iterable[str]) -> str:
    """
    :param lines: The entries of a list in the prompt.
    :return: One ``- `` bulleted line per entry, or a single :data:`NO_ENTRY` bullet
        when there are none.
    """
    return "\n".join(f"- {line}" for line in lines) or f"- {NO_ENTRY}"


class RecoveryResponseError(Exception):
    """
    Raised when an answer to the recovery prompt cannot be used.
    """


class MalformedRecoveryResponse(RecoveryResponseError):
    """
    Raised when an answer holds no non-empty JSON list of action and object pairs.
    """

    def __init__(self, response: str) -> None:
        self.response = response
        """
        The unusable answer.
        """
        super().__init__(
            f"No non-empty JSON list of action and object pairs in: {response!r}"
        )


class UnknownActionObjectPair(RecoveryResponseError):
    """
    Raised when an answer chooses a pair that was not offered.
    """

    def __init__(self, pair: str, offered_pairs: list[str]) -> None:
        self.pair = pair
        """
        The pair that was not offered.
        """
        self.offered_pairs = offered_pairs
        """
        The pairs that were offered.
        """
        super().__init__(f"{pair!r} is not one of {offered_pairs}")


class RecoveryQueryFailed(Exception):
    """
    Raised when every attempt to ask the language model gave an unusable answer.
    """

    def __init__(self, errors: list[RecoveryResponseError]) -> None:
        self.errors = errors
        """
        Why each answer was unusable, in the order they were given.
        """
        super().__init__(
            f"No usable answer in {len(errors)} attempts: {[str(error) for error in errors]}"
        )


def first_list_of_texts(response: str) -> list[str]:
    """
    Pick the JSON list out of an answer, ignoring any text or code fence around it.

    :param response: The language model's raw answer.
    :return: The first JSON list in the answer.
    :raises MalformedRecoveryResponse: If the answer holds no JSON list, or the first
        one is empty or holds something other than texts.
    """
    decoder = json.JSONDecoder()
    start = response.find(JSON_LIST_START)
    while start >= 0:
        try:
            candidate, _ = decoder.raw_decode(response, start)
        except json.JSONDecodeError:
            start = response.find(JSON_LIST_START, start + 1)
            continue
        if not candidate or not all(isinstance(element, str) for element in candidate):
            raise MalformedRecoveryResponse(response)
        return candidate
    raise MalformedRecoveryResponse(response)


def check_recovery_response(response: str, offered_pairs: list[str]) -> list[str]:
    """
    :param response: The language model's raw answer to the recovery prompt.
    :param offered_pairs: The ``action_name,object_name`` pairs it could choose from.
    :return: The chosen pairs, in the order they are to be performed.
    :raises MalformedRecoveryResponse: If the answer holds no list of pairs.
    :raises UnknownActionObjectPair: If it chooses a pair that was not offered.
    """
    chosen_pairs = [pair.strip() for pair in first_list_of_texts(response)]
    for pair in chosen_pairs:
        if pair not in offered_pairs:
            raise UnknownActionObjectPair(pair, offered_pairs)
    return chosen_pairs


def query_recovery_actions(
    language_model: LanguageModel,
    prompt: str,
    offered_pairs: list[str],
    attempts: int = MAX_QUERY_ATTEMPTS,
) -> list[str]:
    """
    Ask the language model the recovery prompt until it gives a usable answer.

    :param language_model: The model to ask.
    :param prompt: The recovery prompt.
    :param offered_pairs: The ``action_name,object_name`` pairs it could choose from.
    :param attempts: How often to ask at most.
    :return: The pairs of the first usable answer, in the order they are to be
        performed.
    :raises RecoveryQueryFailed: If no answer was usable.
    """
    errors = []
    for _ in range(attempts):
        try:
            return check_recovery_response(
                language_model.complete(prompt), offered_pairs
            )
        except RecoveryResponseError as error:
            errors.append(error)
    raise RecoveryQueryFailed(errors)


# %% action descriptions


def get_action_descriptions(
    action_pair: str, actions_graph: ActionFeasibility
) -> ActionFeasibility:
    """
    :param action_pair: An ``action_name,object_name`` pair.
    :param actions_graph: The feasible actions, by action, object and arm.
    :return: The part of ``actions_graph`` for the pair alone: its action, its object
        and the parameters of every arm.
    :raises UnknownActionObjectPair: If the pair is not in ``actions_graph``.
    """
    offered_pairs = get_action_object_pair(actions_graph)
    if action_pair not in offered_pairs:
        raise UnknownActionObjectPair(action_pair, offered_pairs)
    action_name, object_name = action_pair.split(ACTION_OBJECT_SEPARATOR, 1)
    action = next(action for action in actions_graph if action.__name__ == action_name)
    return {action: {object_name: actions_graph[action][object_name]}}


# %% grounding an action


GROUNDED_TRIPLET_ARGUMENTS = (ActionArgument.OBJECT, ActionArgument.ARM)
"""
The arguments a :class:`GroundedTriplet` cannot be built without, whatever its action.
"""


def ground_action(
    action: type[ActionDescription], values: ConditionValues
) -> Optional[GroundedTriplet]:
    """
    Ground an action in the values its conditions were met with.

    :param action: The action to ground, e.g. a key of an
        :data:`~llm_planner.env_reasoner.reasoner_graphs.AFG.ActionFeasibility`.
    :param values: One set of condition values the action is feasible with.
    :return: The action with the object, arm and grasp it is performed with, or None
        if ``values`` lack an argument the action requires.
    """
    supplied = {parameter.keyword: value for parameter, value in values.items()}
    action_arguments = get_action_constraints_names(action)
    if not {*action_arguments, *GROUNDED_TRIPLET_ARGUMENTS}.issubset(supplied):
        return None
    return GroundedTriplet(
        action=action,
        source_object_body=supplied[ActionArgument.OBJECT],
        arm=supplied[ActionArgument.ARM],
        grasp_description=(
            supplied[ActionArgument.GRASP]
            if ActionArgument.GRASP in action_arguments
            else None
        ),
    )


# %% processing an action


class ActionPerformer(Protocol):
    """
    What starts and performs the actions an :class:`ActionProcessor` processes.
    """

    def start_action(self, execution: TripletExecution) -> None:
        """
        Note that an action was grounded and starts running.

        :param execution: The action, reported as :attr:`ExecutionOutcome.RUNNING`.
        """

    def perform_action_step(self, step: ActionLike) -> None:
        """
        Perform one step of the running action.

        :param step: The step to perform.
        """


@dataclass
class ContextPerformer:
    """
    Performs the steps of an action in a context, with nothing shown when the action
    starts.
    """

    context: Context
    """
    The context the steps are performed in, holding the world and robot.
    """

    def start_action(self, execution: TripletExecution) -> None:
        """
        Nothing is shown when an action starts.

        :param execution: The action, reported as :attr:`ExecutionOutcome.RUNNING`.
        """

    def perform_action_step(self, step: ActionLike) -> None:
        """
        Perform one step in :attr:`context`.

        :param step: The step to perform.
        """
        execute_single(step, context=self.context).perform()


@dataclass
class ActionProcessor:
    """
    Grounds a feasible action, plans it and performs its steps, checking the
    post-condition of every step that declares one.

    ..note:: Performing needs an execution environment, e.g. ``with simulated_robot:``
        around :meth:`process`.
    """

    context: Context
    """
    The context the action is planned and performed in, holding the world and robot.
    """

    performer: Optional[ActionPerformer] = field(default=None, kw_only=True)
    """
    What each grounded action is started with and each of its steps performed by; a
    :class:`ContextPerformer` in :attr:`context` when not given.
    """

    def __post_init__(self) -> None:
        if self.performer is None:
            self.performer = ContextPerformer(context=self.context)

    def process(
        self, action: type[ActionDescription], values: ConditionValues
    ) -> Optional[TripletExecution]:
        """
        :param action: The action to perform.
        :param values: One set of condition values the action is feasible with.
        :return: What became of the action, performing stopping at the first step whose
            post-condition did not hold; None if ``values`` lack an argument the action
            requires, so nothing was grounded.
        """
        grounded = ground_action(action, values)
        if grounded is None:
            return None
        self.performer.start_action(
            TripletExecution(None, grounded, ExecutionOutcome.RUNNING)
        )
        steps = plan_grounding([grounded], self.context)
        if not steps:
            return TripletExecution(None, grounded, ExecutionOutcome.NOT_PLANNED)
        for step in steps:
            failed_check = self.run_step(step)
            if failed_check is not None:
                return TripletExecution(
                    None,
                    grounded,
                    ExecutionOutcome.CONDITION_FAILED,
                    failed_check,
                    last_step=step,
                )
        return TripletExecution(
            None, grounded, ExecutionOutcome.PERFORMED, last_step=steps[-1]
        )

    def run_step(self, step: ActionLike) -> Optional[ConditionCheck]:
        """
        :param step: The step to perform.
        :return: The step's post-condition if it does not hold after performing, else
            None; a step outside
            :data:`~llm_planner.env_planner.action_validation.CHECKABLE_ACTIONS` is not
            checked.
        """
        self.perform_step(step)
        if not isinstance(step, CHECKABLE_ACTIONS):
            return None
        post_condition = check_post_condition(step, self.context)
        return None if post_condition.satisfied else post_condition

    def perform_step(self, step: ActionLike) -> None:
        """
        Perform one step through :attr:`performer`.

        :param step: The step to perform.
        """
        self.performer.perform_action_step(step)
