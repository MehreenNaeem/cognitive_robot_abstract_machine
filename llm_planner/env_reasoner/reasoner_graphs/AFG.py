"""TODO
Donot remove this
all_relevent_objects_bodies -> from (self.selected_objects)
Do_able actions list -> PickUpAction,PlaceAction,OpenAction,CloseAction
write Action condition function:
    1. if action is valid for object -> object actionable properties (get_semantic_annotation_of_body from object_graphs.py) (DONE)
    2. TODO check pre condition of action with given object
        write conditions w.r.t to object and actions
    if all above two checks passed then append the action pair in the list
    3. grounding the actions with multiple parameters and make final list

"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, StrEnum
from itertools import product

import rustworkx
from krrood.rustworkx_utils.graph_visualizer_base import GraphLayout
from typing_extensions import Any, ClassVar

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.pose_validator import IsObjectReachableBy
from coraplex.querying.predicates import GripperIsFree, GripperIsNotFree
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.predicate import Predicate
from llm_planner.env_planner.triplet_grounding import ARMS
from llm_planner.env_reasoner.object_action_affordance import get_object_action_list
from llm_planner.env_reasoner.reasoner_graphs.goal_state import GoalTreeVisualizer
from llm_planner.env_reasoner.reasoner_graphs.object_graphs import (
    get_semantic_annotation_of_body,
)
from semantic_digital_twin.reasoning.predicates import is_container_open
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% utilities


def object_action_validation(body_name: str, world: World) -> list[str]:
    """
    List the actions that can be performed on the object a body belongs to.

    :param body_name: The name of the body, e.g. ``"milk.stl"``.
    :param world: The world holding the body.
    :return: The class name of every action applicable to the type of the body's
        semantic annotation, sorted, e.g. ``["PickUpAction", "PlaceAction"]``.
    :raises BodyWithoutSemanticAnnotation: If no annotation is rooted at the body.
    """
    object_annotation = get_semantic_annotation_of_body(body_name, world)
    return sorted(get_object_action_list(type(object_annotation)))


# %% action conditions


class ConditionParameter(Enum):
    """
    A parameter an action condition needs.
    """

    END_EFFECTOR = "end_effector", Any
    OBJECT_DESIGNATOR = "object_designator", Body
    WORLD = "world", World
    DOOR = "door", type[Door]
    CONTEXT = "context", Context
    ARM = "arm", Arms
    GRASP_DESCRIPTION = "grasp_description", GraspDescription

    def __init__(self, keyword: str, value_type: Any) -> None:
        self.keyword = keyword
        """
        The keyword the predicate takes the parameter by.
        """

        self.value_type = value_type
        """
        The type of the parameter's value.
        """


class ActionWithoutConditions(Exception):
    """
    Raised when the conditions of an action are not defined.
    """

    def __init__(self, action: type[ActionDescription]) -> None:
        super().__init__(f"No conditions are defined for {action.__name__}.")
        self.action = action
        """
        The action that was looked up.
        """


class MissingConditionParameter(Exception):
    """
    Raised when a condition is checked without a parameter it needs.
    """

    def __init__(self, parameter: ConditionParameter) -> None:
        super().__init__(f"The condition needs the parameter {parameter!r}.")
        self.parameter = parameter
        """
        The parameter that was not given.
        """


@dataclass(frozen=True)
class ActionCondition:
    """
    One condition that has to hold before an action can be performed.
    """

    predicate: Callable[..., Any]
    """
    The predicate class or function to check.
    """

    parameters: tuple[ConditionParameter, ...]
    """
    The parameters the predicate needs.
    """

    negated: bool = False
    """
    Whether the condition requires the predicate to be false.
    """

    def holds(self, values: Mapping[ConditionParameter, Any]) -> bool:
        """
        :param values: The value of each parameter; extra ones are ignored.
        :return: Whether the condition holds for those values.
        :raises MissingConditionParameter: If a needed parameter has no value.
        """
        for parameter in self.parameters:
            if parameter not in values:
                raise MissingConditionParameter(parameter)
        result = self.predicate(
            **{parameter.keyword: values[parameter] for parameter in self.parameters}
        )
        is_true = bool(result() if isinstance(result, Predicate) else result)
        return is_true != self.negated


GRIPPER_IS_FREE = ActionCondition(GripperIsFree, (ConditionParameter.END_EFFECTOR,))
"""
The gripper holds nothing.
"""

CONTAINER_IS_OPEN_PARAMETERS = (
    ConditionParameter.OBJECT_DESIGNATOR,
    ConditionParameter.WORLD,
    ConditionParameter.DOOR,
)


ACTION_CONDITIONS: dict[type[ActionDescription], tuple[ActionCondition, ...]] = {
    PickUpAction: (GRIPPER_IS_FREE,),
    PlaceAction: (
        ActionCondition(GripperIsNotFree, (ConditionParameter.END_EFFECTOR,)),
    ),
    OpenAction: (
        GRIPPER_IS_FREE,
        ActionCondition(is_container_open, CONTAINER_IS_OPEN_PARAMETERS, negated=True),
    ),
    CloseAction: (
        GRIPPER_IS_FREE,
        ActionCondition(is_container_open, CONTAINER_IS_OPEN_PARAMETERS),
    ),
    NavigateAction: (
        ActionCondition(
            IsObjectReachableBy,
            (
                ConditionParameter.CONTEXT,
                ConditionParameter.ARM,
                ConditionParameter.OBJECT_DESIGNATOR,
                ConditionParameter.GRASP_DESCRIPTION,
            ),
            negated=True,
        ),
    ),
}
"""
The conditions that have to hold before each action can be performed.
"""


def are_action_conditions_met(
    action: type[ActionDescription], values: Mapping[ConditionParameter, Any]
) -> bool:
    """
    :param action: The action to check.
    :param values: The value of each parameter its conditions need.
    :return: Whether every condition of the action holds.
    :raises ActionWithoutConditions: If :data:`ACTION_CONDITIONS` does not list the action.
    :raises MissingConditionParameter: If a checked condition lacks a parameter.
    """
    if action not in ACTION_CONDITIONS:
        raise ActionWithoutConditions(action)
    return all(condition.holds(values) for condition in ACTION_CONDITIONS[action])


# %% action feasibility graph


APPROACH_DIRECTIONS = (
    ApproachDirection.FRONT,
    ApproachDirection.BACK,
    ApproachDirection.RIGHT,
    ApproachDirection.LEFT,
)
"""
The directions a gripper approaches an object from when an action is checked.
"""


ConditionValues = Mapping[ConditionParameter, Any]
"""
The value of each parameter the conditions of an action are checked with.
"""

ActionFeasibility = dict[
    type[ActionDescription], dict[str, dict[Arms, list[ConditionValues]]]
]
"""
The condition values an action is executable with, by action, object name and arm.
"""


@dataclass(frozen=True)
class ExecutableAction:
    """
    An action together with the parameters its conditions are met with.
    """

    action: type[ActionDescription]
    """
    The action that can be performed.
    """

    values: ConditionValues
    """
    The value of each parameter the conditions were checked with.
    """


@dataclass
class ActionFeasibilityGraph:
    """
    Graph of which actions can be performed on which objects of a world.
    """

    object_names: list[str]
    """
    Names of the bodies of the objects the actions may be performed with
    """

    actions: list[type[ActionDescription]]
    """
    The actions to check, e.g. :class:`PickUpAction`.
    """

    world: World
    """
    The world holding the objects.
    """

    context: Context
    """
    The context of the robot that performs the actions.
    """

    def selected_objects(
        self, action_included: Sequence[type[ActionDescription]] = ()
    ) -> dict[type[ActionDescription], list[str]]:
        """
        Pair every action with the objects it can be performed on.

        :param action_included: Actions paired with every object, whether or not they
            are valid for it and whether or not they are in :attr:`actions`.
        :return: The names of the objects of each action, in the order of
            :attr:`object_names`.
        :raises BodyWithoutSemanticAnnotation: If no annotation is rooted at an object.
        """
        valid_actions = {
            object_name: object_action_validation(object_name, self.world)
            for object_name in self.object_names
        }
        objects_of_actions = {
            action: [
                object_name
                for object_name in self.object_names
                if action.__name__ in valid_actions[object_name]
            ]
            for action in self.actions
        }
        for action in action_included:
            objects_of_actions[action] = list(self.object_names)
        return objects_of_actions

    def _is_action_executable(
        self, action: type[ActionDescription], object_name: str
    ) -> list[ExecutableAction]:
        """
        Check an action on an object with every arm and every approach direction.

        :param action: The action to check.
        :param object_name: The name of the body of the object the action acts on.
        :return: One entry per combination whose conditions are met; empty if none is.
        :raises ActionWithoutConditions: If :data:`ACTION_CONDITIONS` does not list the
            action.
        """
        object_designator = self.world.get_body_by_name(object_name)
        results = []
        for arm, approach_direction in product(ARMS, APPROACH_DIRECTIONS):
            end_effector = ViewManager.get_end_effector_view(arm, self.context.robot)
            values = {
                ConditionParameter.END_EFFECTOR: end_effector,
                ConditionParameter.OBJECT_DESIGNATOR: object_designator,
                ConditionParameter.WORLD: self.world,
                ConditionParameter.DOOR: Door,
                ConditionParameter.CONTEXT: self.context,
                ConditionParameter.ARM: arm,
                ConditionParameter.GRASP_DESCRIPTION: GraspDescription(
                    approach_direction, VerticalAlignment.NoAlignment, end_effector
                ),
            }
            if are_action_conditions_met(action, values):
                results.append(ExecutableAction(action, values))
        return results

    def build_action_graph(
        self, action_included: Sequence[type[ActionDescription]] = ()
    ) -> ActionFeasibility:
        """
        Check every action on every object selected for it, sorted by action, object
        and arm.

        :param action_included: Actions checked on every object, as in
            :meth:`selected_objects`.
        :return: For each action, each of its objects, and each of :data:`ARMS`, the
            parameters of every approach direction the action's conditions are met
            with; an arm with no such direction has an empty list. An object no arm
            works for is left out, and so is an action left without objects.
        """
        feasibility = {}
        for action, object_names in self.selected_objects(action_included).items():
            objects_of_action = {}
            for object_name in object_names:
                results = self._is_action_executable(action, object_name)
                if not results:
                    continue
                parameters_of_arms = {arm: [] for arm in ARMS}
                for result in results:
                    parameters_of_arms[result.values[ConditionParameter.ARM]].append(
                        result.values
                    )
                objects_of_action[object_name] = parameters_of_arms
            if objects_of_action:
                feasibility[action] = objects_of_action
        return feasibility


# %% visualization


FEASIBILITY_TREE_PORT = 8052
"""
Port the feasibility tree page is served on, apart from the goal tree's page.
"""

FEASIBILITY_TREE_TITLE = "Action feasibility"
"""
Title of the feasibility tree page.
"""

FEASIBILITY_TREE_TEMPLATE = "feasibility_tree.jinja"
"""
The feasibility tree page, showing only the actions until a node is clicked to expand
it.
"""

PARAMETER_DESCRIPTIONS: dict[ConditionParameter, Callable[[Any], str]] = {
    ConditionParameter.END_EFFECTOR: lambda end_effector: end_effector.name.name,
    ConditionParameter.OBJECT_DESIGNATOR: lambda body: body.name.name,
    ConditionParameter.WORLD: lambda world: world.name,
    ConditionParameter.DOOR: lambda door: door.__name__,
    ConditionParameter.CONTEXT: lambda context: context.robot.name.name,
    ConditionParameter.ARM: lambda arm: arm.name,
    ConditionParameter.GRASP_DESCRIPTION: lambda grasp: (
        f"{grasp.approach_direction.name}, {grasp.vertical_alignment.name}"
    ),
}
"""
How each parameter's value is written on an end node of the tree.
"""


class FeasibilityNodeKind(StrEnum):
    """
    What a node of the feasibility tree stands for.
    """

    ACTION = "action"
    OBJECT = "object"
    ARM = "arm"
    APPROACH_DIRECTION = "approach_direction"


FEASIBILITY_SHAPES: dict[FeasibilityNodeKind, str] = {
    FeasibilityNodeKind.ACTION: "box",
    FeasibilityNodeKind.OBJECT: "ellipse",
    FeasibilityNodeKind.ARM: "circle",
    FeasibilityNodeKind.APPROACH_DIRECTION: "box",
}
"""
The shape each kind of node is drawn in.
"""

FEASIBILITY_COLORS: dict[FeasibilityNodeKind, str] = {
    FeasibilityNodeKind.ACTION: "#e07b39",
    FeasibilityNodeKind.OBJECT: "#4a90d9",
    FeasibilityNodeKind.ARM: "#7bc67b",
    FeasibilityNodeKind.APPROACH_DIRECTION: "#d9c84a",
}
"""
The colour each kind of node is filled with.
"""


@dataclass(frozen=True)
class FeasibilityNode:
    """
    One node of the feasibility tree.
    """

    kind: FeasibilityNodeKind
    """
    What the node stands for.
    """

    caption: str
    """
    What the node stands for, e.g. ``PickUpAction`` or ``FRONT``.
    """

    details: tuple[str, ...] = ()
    """
    Further lines drawn below the caption, one per condition parameter of an end node.
    """

    @property
    def label(self) -> str:
        """
        The text drawn on the node: the caption, then each detail on its own line.
        """
        return "\n".join((self.caption, *self.details))

    @property
    def shape(self) -> str:
        """
        The shape the node is drawn in.
        """
        return FEASIBILITY_SHAPES[self.kind]

    @property
    def color(self) -> str:
        """
        The colour the node is filled with.
        """
        return FEASIBILITY_COLORS[self.kind]


def describe_condition_values(values: ConditionValues) -> tuple[str, ...]:
    """
    :param values: The condition values of one end node.
    :return: One ``keyword: value`` line per parameter, in the order of
        :class:`ConditionParameter`.
    """
    return tuple(
        f"{parameter.keyword}: {PARAMETER_DESCRIPTIONS[parameter](values[parameter])}"
        for parameter in ConditionParameter
        if parameter in values
    )


class FeasibilityTreeVisualizer(GoalTreeVisualizer):
    """
    A page drawing the feasibility tree, where clicking a node shows or hides the
    nodes below it.
    """

    template: ClassVar[str] = FEASIBILITY_TREE_TEMPLATE
    """
    The page template the tree is drawn with.
    """


def feasibility_tree(feasibility: ActionFeasibility) -> rustworkx.PyDiGraph:
    """
    :param feasibility: The output of :meth:`ActionFeasibilityGraph.build_action_graph`.
    :return: A tree from each action to its objects, their arms and the approach
        directions each arm works from; empty arms, and objects and actions left
        without any arm, are left out.
    """
    tree = rustworkx.PyDiGraph()
    for action, objects in feasibility.items():
        executable_objects = {
            object_name: {arm: values for arm, values in arms.items() if values}
            for object_name, arms in objects.items()
        }
        executable_objects = {
            object_name: arms
            for object_name, arms in executable_objects.items()
            if arms
        }
        if not executable_objects:
            continue
        action_index = tree.add_node(
            FeasibilityNode(FeasibilityNodeKind.ACTION, action.__name__)
        )
        for object_name, arms in executable_objects.items():
            object_index = tree.add_child(
                action_index,
                FeasibilityNode(FeasibilityNodeKind.OBJECT, object_name),
                None,
            )
            for arm, condition_values in arms.items():
                arm_index = tree.add_child(
                    object_index,
                    FeasibilityNode(FeasibilityNodeKind.ARM, arm.name),
                    None,
                )
                for values in condition_values:
                    tree.add_child(
                        arm_index,
                        FeasibilityNode(
                            FeasibilityNodeKind.APPROACH_DIRECTION,
                            values[
                                ConditionParameter.GRASP_DESCRIPTION
                            ].approach_direction.name,
                            describe_condition_values(values),
                        ),
                        None,
                    )
    return tree


def feasibility_visualizer(
    feasibility: ActionFeasibility, open_browser: bool = True
) -> FeasibilityTreeVisualizer:
    """
    :param feasibility: The output of :meth:`ActionFeasibilityGraph.build_action_graph`.
    :param open_browser: Whether the running page opens in a web browser.
    :return: A visualizer of :func:`feasibility_tree`, not yet running.
    """
    return FeasibilityTreeVisualizer(
        graph=feasibility_tree(feasibility),
        label_getter=lambda node: node.label,
        color_getter=lambda node: node.color,
        layout=GraphLayout.LAYERED,
        title=FEASIBILITY_TREE_TITLE,
        port=FEASIBILITY_TREE_PORT,
        open_browser=open_browser,
    )


def visualize(
    feasibility: ActionFeasibility, open_browser: bool = True
) -> FeasibilityTreeVisualizer:
    """
    Serve a page of the feasibility tree.

    The page is served from a background thread, so it stays reachable only while the
    calling process keeps running.

    :param feasibility: The output of :meth:`ActionFeasibilityGraph.build_action_graph`.
    :param open_browser: Whether to open the page in a web browser.
    :return: The running visualizer.
    """
    visualizer = feasibility_visualizer(feasibility, open_browser)
    visualizer.run()
    return visualizer
