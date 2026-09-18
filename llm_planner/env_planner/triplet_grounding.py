"""
Grounding a checked triplet in a world.

A triplet names an action and the object types it is performed on and against (see
:mod:`llm_planner.env_planner.triplet_generator`); an action plan is built from concrete
arguments instead - the body to pick up, the arm to use, the pose to place at. This
module states which arguments a triplet's action needs, and resolves the triplet's
objects into them.
"""

from __future__ import annotations

import logging
from dataclasses import MISSING, dataclass, field, fields
from enum import StrEnum
from pathlib import Path

from typing_extensions import Any, Dict, List, Optional

from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.view_manager import ViewManager
from llm_planner.env_planner.triplet_generator import Triplet
from llm_planner.env_planner.utilities import DEFAULT_PLACE_SIZE, unoccupied_region
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

logger = logging.getLogger(__name__)


class ActionArgument(StrEnum):
    """
    The arguments an action plan is built from that a triplet can be resolved into.

    The names are the ones the actions declare, so a resolved triplet can be handed to
    an action's constructor as keyword arguments.
    """

    OBJECT = "object_designator"
    """
    The body the action is performed on.
    """

    TARGET_LOCATION = "target_location"
    """
    The pose the object is brought to.
    """

    ARM = "arm"
    """
    The arm that performs the action.
    """

    GRASP = "grasp_description"
    """
    How the end effector takes hold of the object.
    """


class ObjectNotInWorld(LookupError):
    """
    Raised when a triplet names an object type the world holds no instance of.
    """

    def __init__(self, object_type: type[SemanticAnnotation]) -> None:
        super().__init__(f"the world holds no {object_type.__name__}.")
        self.object_type = object_type
        """
        The object type that could not be resolved.
        """


class NoRobot(RuntimeError):
    """
    The world holds no robot, so nothing can be said about how it takes hold of an
    object.

    Reported as a warning rather than raised, like :class:`NoFreePlace`: the rest of the
    triplet is still worth grounding, and the grasp then shows up as missing.
    """

    def __init__(self, action: type[ActionDescription]) -> None:
        super().__init__(
            f"the world holds no robot, so no grasp can be described for "
            f"{action.__name__}."
        )
        self.action = action
        """
        The action that needs a grasp.
        """


class NoFreePlace(RuntimeError):
    """
    Nothing can be put down on the object a triplet places against.

    Reported as a warning rather than raised: a triplet whose target is full is still
    worth grounding, and the action then names no place to put the object down (see
    :meth:`GroundedTriplet.missing_parameters`).
    """

    def __init__(self, target_body: Body) -> None:
        super().__init__(f"nothing fits on {target_body.name} any more.")
        self.target_body = target_body
        """
        The body that has no free place left.
        """


@dataclass(frozen=True)
class ActionParameter:
    """
    One argument an action plan is built from.
    """

    name: str
    """
    Name of the argument, as the action's constructor takes it.
    """

    type_name: str
    """
    Name of the type the action declares for it, e.g. ``Body`` or ``Pose``.
    """


@dataclass
class GroundedTriplet:
    """
    One triplet resolved into the things its action plan is performed with.
    """

    action: type[ActionDescription]
    """
    The action to perform.
    """

    source_object_body: Body
    """
    The body the action is performed on.
    """

    arm: Arms
    """
    The arm that performs it.
    """

    target_object_body: Optional[Body] = None
    """
    The body the action is performed against, or None when the triplet names the same
    object twice, as an action on a single object does.
    """

    target_location: Optional[Pose] = None
    """
    Where the object is put down, or None when the action puts nothing down and when
    nothing fits on the object it would be put down on.
    """

    grasp_description: Optional[GraspDescription] = None
    """
    How the end effector takes hold of the object, or None when the action takes hold
    of nothing and when the world holds no robot to describe it for.
    """

    def action_arguments(self) -> Dict[str, Any]:
        """
        :return: The resolved arguments, keyed by the name the action takes them under,
            leaving out the ones this triplet says nothing about.
        """
        arguments: Dict[str, Any] = {
            ActionArgument.OBJECT: self.source_object_body,
            ActionArgument.ARM: self.arm,
        }
        if self.target_location is not None:
            arguments[ActionArgument.TARGET_LOCATION] = self.target_location
        if self.grasp_description is not None:
            arguments[ActionArgument.GRASP] = self.grasp_description
        return arguments

    def missing_parameters(self) -> List[ActionParameter]:
        """
        :return: The arguments the action still needs that this grounding could not
            supply, e.g. a place to put the object down that nothing fits in.
        """
        supplied = self.action_arguments()
        return [
            parameter
            for parameter in TripletGrounding.required_parameters(self.action)
            if parameter.name not in supplied
        ]


@dataclass
class TripletGrounding:
    """
    Turns a checked triplet into the things its action plan is performed with: the
    bodies its object types name, where to put the object down, and the arm to use.

    A world holds many objects of the same type - ten drawers, several countertops - so
    which one a triplet means has to be decided. The one nearest the object being acted
    on is taken, and the choice is then kept for the rest of this grounding: the drawer
    a sequence opens is the drawer it places into and closes again.
    """

    world: World
    """
    The world the triplet's object types are resolved in.
    """

    arm: Arms = Arms.LEFT
    """
    The arm every action is performed with.
    """

    placing_yaw: float = 0.0
    """
    The yaw an object is put down with, in radians, keeping the place's position.
    """

    approach_direction: ApproachDirection = ApproachDirection.FRONT
    """
    The side the end effector comes at an object from.
    """

    vertical_alignment: VerticalAlignment = VerticalAlignment.NoAlignment
    """
    How the end effector lines up with the object along z.
    """

    _chosen: Dict[type[SemanticAnnotation], Body] = field(
        default_factory=dict, init=False
    )
    """
    The body already chosen for an object type, so every triplet of a sequence means
    the same one.
    """

    @staticmethod
    def required_parameters(
        action: type[ActionDescription],
    ) -> List[ActionParameter]:
        """
        The arguments an action's plan cannot be built without.

        Every other field the action declares carries a default and is left to it.

        :param action: The action to describe.
        :return: One :class:`ActionParameter` per argument without a default, in the
            order the action declares them.
        """
        return [
            ActionParameter(name=field_.name, type_name=_type_name(field_.type))
            for field_ in fields(action)
            if field_.default is MISSING and field_.default_factory is MISSING
        ]

    def needs(self, action: type[ActionDescription], argument: ActionArgument) -> bool:
        """
        :param action: The action to check.
        :param argument: The argument to look for.
        :return: Whether the action's plan cannot be built without that argument.
        """
        return any(
            parameter.name == argument for parameter in self.required_parameters(action)
        )

    def bodies_of(self, object_type: type[SemanticAnnotation]) -> List[Body]:
        """
        :param object_type: The object type a triplet named.
        :return: The root body of every annotation of that type in the world, in the
            order the world holds them.
        """
        return [
            annotation.root
            for annotation in self.world.get_semantic_annotations_by_type(object_type)
        ]

    def body_of(
        self, object_type: type[SemanticAnnotation], near: Optional[Body] = None
    ) -> Body:
        """
        The body a triplet's object type means.

        The world can hold several objects of one type; the one nearest ``near`` is
        taken, or nearest the robot when the triplet gives nothing to be near. Once a
        type has been resolved, that body stands for it for the rest of this grounding.

        :param object_type: The object type a triplet named.
        :param near: The body the object is used with, e.g. what is being put into it.
        :return: The body of that object.
        :raises ObjectNotInWorld: If the world holds no such object.
        """
        if object_type in self._chosen:
            return self._chosen[object_type]
        bodies = self.bodies_of(object_type)
        if not bodies:
            raise ObjectNotInWorld(object_type)
        reference = near or self.robot_body()
        chosen = (
            bodies[0]
            if reference is None
            else min(bodies, key=lambda body: _distance_between(body, reference))
        )
        self._chosen[object_type] = chosen
        return chosen

    def robot(self) -> Optional[AbstractRobot]:
        """
        :return: The world's robot, or None if it holds none.
        """
        robots = self.world.get_semantic_annotations_by_type(AbstractRobot)
        return robots[0] if robots else None

    def robot_body(self) -> Optional[Body]:
        """
        :return: The root body of the world's robot, or None if it holds no robot.
        """
        robot = self.robot()
        return None if robot is None else robot.root

    def grasp_for(self, action: type[ActionDescription]) -> Optional[GraspDescription]:
        """
        How the robot takes hold of an object it performs an action on.

        :param action: The action the grasp is described for.
        :return: The grasp the action's end effector performs, or None if the world
            holds no robot, which is logged as a warning.
        """
        robot = self.robot()
        if robot is None:
            logger.warning("%s", NoRobot(action))
            return None
        return GraspDescription(
            self.approach_direction,
            self.vertical_alignment,
            ViewManager.get_end_effector_view(self.arm, robot),
        )

    def target_location_on(
        self, target_body: Body, source_body: Body
    ) -> Optional[Pose]:
        """
        Where to put an object down on another one: the free place nearest to where the
        object stands now, turned to :attr:`placing_yaw`.

        :param target_body: The body to put the object on.
        :param source_body: The body being put down, whose size decides how much room a
            place has to have.
        :return: The pose to place at, or None if nothing fits on the target body any
            more, which is logged as a warning.
        """
        occupancy = unoccupied_region(
            target_body, self.world, place_size=self.size_of(source_body)
        )
        place = occupancy.closest_free_place(source_body.global_pose)
        if place is None:
            logger.warning("%s", NoFreePlace(target_body))
            return None
        return Pose.from_xyz_rpy(
            place.x,
            place.y,
            place.z,
            roll=place.roll,
            pitch=place.pitch,
            yaw=self.placing_yaw,
            reference_frame=place.reference_frame,
        )

    @staticmethod
    def size_of(body: Body) -> Scale:
        """
        :param body: The body to measure.
        :return: The size of the body's bounding box, or the default place size for a
            body that carries no collision shape to measure.
        """
        bounds = body.collision.as_bounding_box_collection_in_frame(body)
        if not bounds.bounding_boxes:
            return DEFAULT_PLACE_SIZE
        return bounds.bounding_box().scale

    def ground(self, triplet: Triplet) -> GroundedTriplet:
        """
        Resolve a triplet into the things its action plan is performed with.

        :param triplet: The checked triplet to resolve.
        :return: The bodies its object types name, the arm, and the pose to place at
            when the action puts something down and something still fits there.
        :raises ObjectNotInWorld: If the world holds neither of its objects.
        """
        source_object_body = self.body_of(triplet.source_object)
        target_object_body = (
            None
            if triplet.target_object is triplet.source_object
            else self.body_of(triplet.target_object, near=source_object_body)
        )
        target_location = None
        if self.needs(triplet.action, ActionArgument.TARGET_LOCATION):
            target_location = self.target_location_on(
                target_object_body or source_object_body, source_object_body
            )
        grasp_description = None
        if self.needs(triplet.action, ActionArgument.GRASP):
            grasp_description = self.grasp_for(triplet.action)
        return GroundedTriplet(
            action=triplet.action,
            source_object_body=source_object_body,
            target_object_body=target_object_body,
            target_location=target_location,
            grasp_description=grasp_description,
            arm=self.arm,
        )


def _type_name(annotation: Any) -> str:
    """
    The plain name of a dataclass field's type annotation.

    :meth:`~coraplex.plans.designator.Designator.fields` resolves a field's annotation
    from its declared string (``"Body"``) to the real type it names the first time it
    runs, mutating the dataclass's shared :class:`~dataclasses.Field` in place - so the
    same field can be seen as either depending on what else has already asked for it in
    the process. This reads a name either way.

    :param annotation: A field's ``type``, either the declared string or the resolved
        type.
    :return: The name alone, e.g. ``"Body"``.
    """
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", str(annotation))


def _distance_between(body: Body, other: Body) -> float:
    """
    :param body: One body.
    :param other: The body to measure to.
    :return: The straight-line distance between the two, in metres.
    """
    here, there = body.global_pose, other.global_pose
    return float(
        ((here.x - there.x) ** 2 + (here.y - there.y) ** 2 + (here.z - there.z) ** 2)
        ** 0.5
    )


# %% usage example
"""
if __name__ == "__main__":
    import os

    from coraplex.testing import setup_world
    from llm_planner.env_planner.triplet_generator import TripletResponseChecker
    from semantic_digital_twin.adapters.mesh import STLParser
    from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Bowl,
        Drawer,
        Handle,
    )
    from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
    from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

    RECORDED_RESPONSE = (
        Path(__file__).parents[1]
        / "resources"
        / "llm_responses"
        / "put_the_bowl_in_the_drawer.json"
    )
    BOWL_MESH = (
        Path(__file__).parents[2] / "coraplex" / "resources" / "objects" / "bowl.stl"
    )

    world = setup_world()
    bowl = STLParser(str(BOWL_MESH)).parse()
    with world.modify_world():
        world.merge_world_at_pose(
            bowl,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                2.4, 2.2, 1.0, reference_frame=world.root
            ),
        )
    with world.modify_world():
        WorldReasoner(world).reason()
        world.add_semantic_annotation(Bowl(root=world.get_body_by_name("bowl.stl")))
        world.add_semantic_annotation_recursively(
            Drawer(
                root=world.get_body_by_name("cabinet10_drawer_top"),
                handle=Handle(root=world.get_body_by_name("handle_cab10_t")),
            )
        )

    print("--- the response the language model gave ---")
    print(RECORDED_RESPONSE.read_text())

    checker = TripletResponseChecker(object_types=[Bowl, Drawer])
    triplets = checker.parse(RECORDED_RESPONSE.read_text())

    grounding = TripletGrounding(world=world)
    print("--- grounded in the world ---")
    for triplet in triplets:
        # A placing triplet needs somewhere to put the bowl, and a closed drawer has no
        # room in it: the plan opens the drawer first, which here is done by hand.
        if grounding.needs(triplet.action, ActionArgument.TARGET_LOCATION):
            drawer_body = grounding.body_of(triplet.target_object)
            connection = next(
                connection
                for connection in world.connections
                if connection.child is drawer_body
                and isinstance(connection, ActiveConnection1DOF)
            )
            connection.position = 0.4
            world.notify_state_change()

        grounded = grounding.ground(triplet)
        print(
            f"{grounded.action.__name__}("
            f"object={grounded.source_object_body.name.name}, "
            f"arm={grounded.arm.name}"
            + (
                ""
                if grounded.target_object_body is None
                else f", against={grounded.target_object_body.name.name}"
            )
            + ")"
        )
        if grounded.target_location is not None:
            place = grounded.target_location
            print(
                f"    put down at ({float(place.x):.2f}, {float(place.y):.2f}, "
                f"{float(place.z):.2f}), yaw {float(place.yaw):.2f}"
            )
        missing = [parameter.name for parameter in grounded.missing_parameters()]
        if missing:
            print(f"    the plan still needs: {', '.join(missing)}")
        arguments = ", ".join(sorted(grounded.action_arguments()))
        print(f"    the plan is built from: {arguments}")
"""
