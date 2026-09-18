import logging
from typing import List, Optional, Union

from coraplex.datastructures.dataclasses import Context
from coraplex.locations.base import DeferredLocation
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    PouringAction,
)
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from krrood.entity_query_language.factories import a, variable
from semantic_digital_twin.spatial_types import Pose
from coraplex.datastructures.enums import Arms
from coraplex.locations.factories import reachability_location
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.base import ActionDescription
from llm_planner.env_planner.triplet_grounding import ActionArgument, GroundedTriplet
from semantic_digital_twin.world import World
from semantic_digital_twin.semantic_annotations.mixins import HasDoors, HasHandle
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from llm_planner.env_reasoner.action_constraints import (
    INTERACTIVE_ACTION,
    TWO_OBJECT_ACTION,
)

logger = logging.getLogger(__name__)

HANDLE_ACTIONS = (OpenAction, CloseAction)
"""
The actions performed on an object's handle rather than on the object itself.
"""


def goto_location(location: Union[Pose, Body], context: Context, arm: Arms):
    return [
        ParkArmsAction(Arms.BOTH),
        MoveTorsoAction(TorsoState.HIGH),
        # Tries to find a pick-up position for the robot that uses the given arm
        a(NavigateAction)(
            target_location=variable(
                Pose,
                domain=DeferredLocation(
                    lambda: reachability_location(
                        location,
                        context,
                        arm,
                    )
                ),
            ),
            keep_joint_states=True,
        ),
    ]


def annotations_of(body: Body, world: World) -> List[SemanticAnnotation]:
    """
    The semantic annotations a body is the root of.

    ..note:: A body's own ``get_semantic_annotations_by_type`` cannot answer this: the
        back-reference it reads is never filled in, so the world's annotations are
        searched instead.

    :param body: The body to look up.
    :param world: The world the body belongs to.
    :return: Every annotation rooted at that body, in the order the world holds them.
    """
    return [
        annotation
        for annotation in world.semantic_annotations
        if getattr(annotation, "root", None) is body
    ]


def get_access_handle(body: Body, world: World) -> Optional[Body]:
    """
    The handle to take hold of to open or close an object.

    Which handle that is follows from what the body is: a drawer states its own handle,
    while a cabinet states its doors and each door states the handle. An object whose
    annotation names no handle is searched for a handle among the bodies below it, which
    is how a drawer whose annotation was inferred rather than written out still opens.

    :param body: The body to be opened or closed.
    :param world: The world the body belongs to.
    :return: The handle's body, or None if the object has none, which is logged as a
        warning.
    """
    for semantic_body in annotations_of(body, world):
        handle = _handle_of(semantic_body)
        if handle is not None:
            return handle.root
    handle_below = _handle_below(body, world)
    if handle_below is not None:
        return handle_below.root
    logger.warning("%s has no handle to open or close it by.", body.name)
    return None


def _handle_of(semantic_body: SemanticAnnotation) -> Optional[Handle]:
    """
    The handle an annotation names, directly or through one of its doors.

    :param semantic_body: The annotation of the object to be opened or closed.
    :return: The handle, or None if the annotation names none.
    """
    if isinstance(semantic_body, HasHandle) and semantic_body.handle is not None:
        return semantic_body.handle
    if isinstance(semantic_body, HasDoors):
        for door in semantic_body.doors:
            if door.handle is not None:
                return door.handle
    return None


def _handle_below(body: Body, world: World) -> Optional[Handle]:
    """
    The handle annotated somewhere under a body, e.g. the grip on a drawer's front.

    :param body: The body to search below.
    :param world: The world the body belongs to.
    :return: The first handle found, or None if nothing under the body is one.
    """
    below = set(world.compute_descendent_child_kinematic_structure_entities(body))
    for handle in world.get_semantic_annotations_by_type(Handle):
        if handle.root in below:
            return handle
    return None


def plan_grounding(grounded_triplets: List[GroundedTriplet], context: Context) -> Plan:
    """
    The plan that carries out a grounded sequence.

    Every triplet contributes the navigation that brings the robot within reach of what
    it acts on, followed by the action itself. A triplet whose action cannot be built -
    one still missing an argument the grounding could not supply, such as a place to put
    an object that nothing fits on - is left out with a warning, so the rest of the
    sequence is still performed.

    :param grounded_triplets: The sequence to perform, as
        :meth:`~llm_planner.env_planner.triplet_grounding.TripletGrounding.ground`
        resolves each triplet.
    :param context: The context the plan is performed in, holding the world and robot.
    :return: The plan to perform.
    """
    children = []
    for grounded_triplet in grounded_triplets:
        action = action_designator(grounded_triplet, context)
        if action is None:
            continue
        children.extend(
            action_designator_plan(grounded_triplet, context, grounded_triplet.arm)
        )
        children.append(action)
    return sequential(children, context=context).plan


def action_designator(
    grounded_triplet: GroundedTriplet, context: Context
) -> Optional[ActionDescription]:
    """
    The action a grounded triplet is performed as.

    An opening or closing action is performed on the handle rather than on the object
    itself, so the object's handle takes the place of the object it gives access to.

    :param grounded_triplet: The triplet to perform.
    :param context: The context the action is performed in.
    :return: The action, or None if it cannot be built, which is logged as a warning.
    """
    missing = grounded_triplet.missing_parameters()
    if missing:
        logger.warning(
            "%s is left out of the plan: no %s was found for it.",
            grounded_triplet.action.__name__,
            ", ".join(parameter.name for parameter in missing),
        )
        return None
    arguments = dict(grounded_triplet.action_arguments())
    if grounded_triplet.action in HANDLE_ACTIONS:
        handle = get_access_handle(grounded_triplet.source_object_body, context.world)
        if handle is None:
            logger.warning(
                "%s is left out of the plan: %s has no handle to perform it on.",
                grounded_triplet.action.__name__,
                grounded_triplet.source_object_body.name,
            )
            return None
        arguments[ActionArgument.OBJECT] = handle
    return grounded_triplet.action(**arguments)


def action_designator_plan(grounded_action, context: Context, arm: Arms):
    child_plan = []
    # before interactive action plans add goto_location
    if grounded_action.action in INTERACTIVE_ACTION:
        if grounded_action.action in TWO_OBJECT_ACTION:
            loc_togo = grounded_action.target_location
        else:
            loc_togo = grounded_action.source_object_body
        if grounded_action.action in HANDLE_ACTIONS:
            handle = get_access_handle(
                grounded_action.source_object_body, context.world
            )
            loc_togo = handle.global_pose
        child_plan.extend(goto_location(loc_togo, context, grounded_action.arm))
    return child_plan
