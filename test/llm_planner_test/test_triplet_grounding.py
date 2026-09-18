"""
Tests of grounding a checked triplet in a world: which arguments its action plan needs,
and what the triplet's object types resolve to in the apartment.
"""

from __future__ import annotations

import logging
from dataclasses import MISSING, fields

import pytest

from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.view_manager import ViewManager
from llm_planner.env_planner import triplet_grounding
from llm_planner.env_planner.triplet_generator import Triplet
from llm_planner.env_planner.triplet_grounding import (
    ActionArgument,
    ActionParameter,
    NoFreePlace,
    NoRobot,
    ObjectNotInWorld,
    TripletGrounding,
)
from llm_planner.env_planner.utilities import unoccupied_region
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Drawer,
    Milk,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

DRAWER_BODY_NAME = "cabinet10_drawer_top"
"""
The body of the apartment drawer the fixture annotates.
"""

MILK_BODY_NAME = "milk.stl"
"""
The body of the milk the fixture annotates.
"""


def triplet_of(source, target, action) -> Triplet:
    return Triplet(source_object=source, target_object=target, action=action)


def opened(world, body, position: float) -> None:
    """
    Move the connection that carries a body to the given joint position.

    :param world: The world the body belongs to.
    :param body: The body to move, e.g. a drawer.
    :param position: The joint position to move it to, in metres.
    """
    connection = next(
        connection
        for connection in world.connections
        if connection.child is body and isinstance(connection, ActiveConnection1DOF)
    )
    connection.position = position
    world.notify_state_change()


class CollectedWarnings(logging.Handler):
    """
    The warnings a logger emitted, so a test can assert on what was reported.

    ``caplog`` records nothing in this workspace - the installed pytest plugins leave
    its handler off the root logger - so the handler is attached by hand.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture()
def reported() -> CollectedWarnings:
    """
    Collect what :mod:`llm_planner.env_planner.triplet_grounding` warns about.
    """
    handler = CollectedWarnings()
    triplet_grounding.logger.addHandler(handler)
    yield handler
    triplet_grounding.logger.removeHandler(handler)


@pytest.fixture()
def grounding(apartment_world_copy) -> TripletGrounding:
    return TripletGrounding(world=apartment_world_copy)


# %% the arguments an action plan is built from


class TestRequiredParameters:
    def test_names_every_argument_the_action_declares_without_a_default(self) -> None:
        expected = [
            field.name
            for field in fields(PickUpAction)
            if field.default is MISSING and field.default_factory is MISSING
        ]

        parameters = TripletGrounding.required_parameters(PickUpAction)

        assert [parameter.name for parameter in parameters] == expected

    def test_carries_the_type_the_action_declares_for_each_argument(self) -> None:
        [object_designator, *_] = TripletGrounding.required_parameters(PickUpAction)

        assert object_designator == ActionParameter(
            name=ActionArgument.OBJECT, type_name="Body"
        )

    def test_leaves_out_the_arguments_that_carry_a_default(self) -> None:
        """
        An argument with a default is the action's own business, not the caller's.
        """
        defaulted = [
            field.name
            for field in fields(PlaceAction)
            if field.default is not MISSING or field.default_factory is not MISSING
        ]
        names = [
            parameter.name
            for parameter in TripletGrounding.required_parameters(PlaceAction)
        ]

        assert not set(names) & set(defaulted)

    def test_an_opening_plan_needs_only_the_container_and_the_arm(self) -> None:
        names = [
            parameter.name
            for parameter in TripletGrounding.required_parameters(OpenAction)
        ]

        assert names == [ActionArgument.OBJECT, ActionArgument.ARM]

    def test_only_an_action_that_puts_something_down_needs_a_place_for_it(
        self, grounding
    ) -> None:
        assert grounding.needs(PlaceAction, ActionArgument.TARGET_LOCATION)
        assert not grounding.needs(PickUpAction, ActionArgument.TARGET_LOCATION)
        assert not grounding.needs(OpenAction, ActionArgument.TARGET_LOCATION)


# %% resolving an object type to a body


class TestObjectBodies:
    def test_resolves_an_object_type_to_the_worlds_body(self, grounding) -> None:
        assert grounding.body_of(Milk).name.name == MILK_BODY_NAME

    def test_lists_every_body_of_that_type(self, grounding) -> None:
        assert [body.name.name for body in grounding.bodies_of(Drawer)] == [
            DRAWER_BODY_NAME
        ]

    def test_rejects_an_object_the_world_does_not_hold(self, grounding) -> None:
        with pytest.raises(ObjectNotInWorld) as excinfo:
            grounding.body_of(Apple)

        assert excinfo.value.object_type is Apple

    def test_a_bodys_size_is_the_size_of_its_bounding_box(
        self, grounding, apartment_world_copy
    ) -> None:
        milk = apartment_world_copy.get_body_by_name(MILK_BODY_NAME)
        bounds = milk.collision.as_bounding_box_collection_in_frame(milk).bounding_box()

        assert grounding.size_of(milk) == bounds.scale


# %% grounding a triplet


class TestGround:
    def test_takes_the_bodies_of_both_objects(self, grounding) -> None:
        grounded = grounding.ground(triplet_of(Milk, Drawer, PickUpAction))

        assert grounded.source_object_body.name.name == MILK_BODY_NAME
        assert grounded.target_object_body.name.name == DRAWER_BODY_NAME

    def test_keeps_the_action_of_the_triplet(self, grounding) -> None:
        grounded = grounding.ground(triplet_of(Drawer, Drawer, CloseAction))

        assert grounded.action is CloseAction

    def test_an_action_on_a_single_object_has_no_target_body(self, grounding) -> None:
        """
        A triplet naming the same object twice - opening a drawer - performs the action
        against nothing.
        """
        grounded = grounding.ground(triplet_of(Drawer, Drawer, OpenAction))

        assert grounded.target_object_body is None

    def test_the_left_arm_is_used_unless_another_is_asked_for(
        self, grounding, apartment_world_copy
    ) -> None:
        assert grounding.ground(triplet_of(Drawer, Drawer, OpenAction)).arm is Arms.LEFT

        right_handed = TripletGrounding(world=apartment_world_copy, arm=Arms.RIGHT)

        assert (
            right_handed.ground(triplet_of(Drawer, Drawer, OpenAction)).arm
            is Arms.RIGHT
        )

    def test_an_action_that_puts_nothing_down_has_no_place_to_put_it(
        self, grounding
    ) -> None:
        grounded = grounding.ground(triplet_of(Milk, Drawer, PickUpAction))

        assert grounded.target_location is None

    def test_a_placing_triplet_is_grounded_on_a_free_place_of_its_target(
        self, grounding, apartment_world_copy
    ) -> None:
        drawer = apartment_world_copy.get_body_by_name(DRAWER_BODY_NAME)
        opened(apartment_world_copy, drawer, 0.4)
        milk = apartment_world_copy.get_body_by_name(MILK_BODY_NAME)

        grounded = grounding.ground(triplet_of(Milk, Drawer, PlaceAction))

        free_places = unoccupied_region(
            drawer, apartment_world_copy, place_size=grounding.size_of(milk)
        ).free
        assert [
            (float(place.x), float(place.y), float(place.z)) for place in free_places
        ].count(
            (
                float(grounded.target_location.x),
                float(grounded.target_location.y),
                float(grounded.target_location.z),
            )
        ) == 1

    def test_the_place_is_the_one_nearest_to_the_object(
        self, grounding, apartment_world_copy
    ) -> None:
        drawer = apartment_world_copy.get_body_by_name(DRAWER_BODY_NAME)
        opened(apartment_world_copy, drawer, 0.4)
        milk = apartment_world_copy.get_body_by_name(MILK_BODY_NAME)
        occupancy = unoccupied_region(
            drawer, apartment_world_copy, place_size=grounding.size_of(milk)
        )

        grounded = grounding.ground(triplet_of(Milk, Drawer, PlaceAction))

        nearest = occupancy.closest_free_place(milk.global_pose)
        assert float(grounded.target_location.x) == pytest.approx(float(nearest.x))
        assert float(grounded.target_location.y) == pytest.approx(float(nearest.y))

    def test_the_object_is_put_down_turned_the_way_the_grounding_says(
        self, apartment_world_copy
    ) -> None:
        drawer = apartment_world_copy.get_body_by_name(DRAWER_BODY_NAME)
        opened(apartment_world_copy, drawer, 0.4)
        grounding = TripletGrounding(world=apartment_world_copy, placing_yaw=1.57)

        grounded = grounding.ground(triplet_of(Milk, Drawer, PlaceAction))

        assert float(grounded.target_location.yaw) == pytest.approx(1.57)

    def test_placing_on_a_full_target_is_grounded_without_a_place(
        self, grounding, apartment_world_copy, reported
    ) -> None:
        """
        A closed drawer has the cabinet directly above its floor, so nothing fits in it.
        The rest of the triplet is still worth grounding, so the full target is a
        warning rather than the end of the grounding.
        """
        drawer = apartment_world_copy.get_body_by_name(DRAWER_BODY_NAME)

        grounded = grounding.ground(triplet_of(Milk, Drawer, PlaceAction))

        assert grounded.target_location is None
        assert grounded.source_object_body.name.name == MILK_BODY_NAME
        assert reported.messages == [str(NoFreePlace(drawer))]

    def test_a_target_with_room_left_is_not_warned_about(
        self, grounding, apartment_world_copy, reported
    ) -> None:
        drawer = apartment_world_copy.get_body_by_name(DRAWER_BODY_NAME)
        opened(apartment_world_copy, drawer, 0.4)

        grounding.ground(triplet_of(Milk, Drawer, PlaceAction))

        assert reported.messages == []

    def test_the_place_a_full_target_cannot_give_is_reported_as_missing(
        self, grounding
    ) -> None:
        """
        A placing plan cannot be built without a place, so what the grounding could not
        supply has to show up as missing rather than pass silently.
        """
        grounded = grounding.ground(triplet_of(Milk, Drawer, PlaceAction))

        assert [parameter.name for parameter in grounded.missing_parameters()] == [
            ActionArgument.TARGET_LOCATION
        ]


# %% handing the arguments to the action


class TestActionArguments:
    def test_the_arguments_are_keyed_the_way_the_action_takes_them(
        self, grounding
    ) -> None:
        grounded = grounding.ground(triplet_of(Drawer, Drawer, OpenAction))

        assert grounded.action_arguments() == {
            ActionArgument.OBJECT: grounded.source_object_body,
            ActionArgument.ARM: grounded.arm,
        }

    def test_an_opening_plan_can_be_built_from_them_alone(self, grounding) -> None:
        grounded = grounding.ground(triplet_of(Drawer, Drawer, OpenAction))

        assert grounded.missing_parameters() == []

    def test_what_a_triplet_cannot_say_is_reported_as_missing(self, grounding) -> None:
        """
        A pick-up needs a grasp, which names how the gripper approaches rather than what
        the task is about, so no triplet supplies it.
        """
        grounded = grounding.ground(triplet_of(Milk, Drawer, PickUpAction))

        assert [parameter.name for parameter in grounded.missing_parameters()] == [
            "grasp_description"
        ]


class TestGrasp:
    """
    An action that takes hold of an object needs to be told how, which follows from the
    robot's end effector rather than from the triplet.
    """

    def test_only_an_action_that_takes_hold_gets_a_grasp(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, _ = apartment_world_pr2_copy_with_context
        grounding = TripletGrounding(world=world)

        assert (
            grounding.ground(triplet_of(Milk, Drawer, PickUpAction)).grasp_description
            is not None
        )
        assert (
            grounding.ground(triplet_of(Drawer, Drawer, OpenAction)).grasp_description
            is None
        )

    def test_the_grasp_uses_the_end_effector_of_the_arm_that_performs_it(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, robot, _ = apartment_world_pr2_copy_with_context
        grounding = TripletGrounding(world=world, arm=Arms.RIGHT)

        grasp = grounding.ground(
            triplet_of(Milk, Drawer, PickUpAction)
        ).grasp_description

        assert grasp.end_effector is ViewManager.get_end_effector_view(
            Arms.RIGHT, robot
        )

    def test_the_grasp_comes_at_the_object_the_way_the_grounding_says(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, _ = apartment_world_pr2_copy_with_context
        grounding = TripletGrounding(
            world=world,
            approach_direction=ApproachDirection.LEFT,
            vertical_alignment=VerticalAlignment.TOP,
        )

        grasp = grounding.ground(
            triplet_of(Milk, Drawer, PickUpAction)
        ).grasp_description

        assert grasp.approach_direction is ApproachDirection.LEFT
        assert grasp.vertical_alignment is VerticalAlignment.TOP

    def test_the_front_of_the_object_is_approached_unless_another_side_is_asked_for(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, _ = apartment_world_pr2_copy_with_context

        grasp = (
            TripletGrounding(world=world)
            .ground(triplet_of(Milk, Drawer, PickUpAction))
            .grasp_description
        )

        assert grasp.approach_direction is ApproachDirection.FRONT
        assert grasp.vertical_alignment is VerticalAlignment.NoAlignment

    def test_a_pick_up_plan_can_be_built_from_the_grounded_arguments_alone(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, _ = apartment_world_pr2_copy_with_context
        grounding = TripletGrounding(world=world)

        grounded = grounding.ground(triplet_of(Milk, Drawer, PickUpAction))

        assert grounded.missing_parameters() == []
        assert set(grounded.action_arguments()) == {
            ActionArgument.OBJECT,
            ActionArgument.ARM,
            ActionArgument.GRASP,
        }


# %% which of several objects of one type a triplet means


SECOND_DRAWER_BODY_NAME = "cabinet1_drawer_bottom"
"""
Another apartment drawer, far from the one the fixture annotates.
"""


def annotate_second_drawer(world) -> Drawer:
    """
    Annotate a second drawer, so a triplet naming ``Drawer`` has a choice to make.

    :param world: The world to annotate in.
    :return: The added annotation.
    """
    drawer = Drawer(root=world.get_body_by_name(SECOND_DRAWER_BODY_NAME))
    with world.modify_world():
        world.add_semantic_annotation(drawer)
    return drawer


class TestChoosingBetweenObjects:
    def test_the_object_nearest_the_one_being_acted_on_is_taken(
        self, grounding, apartment_world_copy
    ) -> None:
        annotate_second_drawer(apartment_world_copy)
        milk = apartment_world_copy.get_body_by_name(MILK_BODY_NAME)
        nearest = min(
            grounding.bodies_of(Drawer),
            key=lambda body: float(
                (body.global_pose.x - milk.global_pose.x) ** 2
                + (body.global_pose.y - milk.global_pose.y) ** 2
                + (body.global_pose.z - milk.global_pose.z) ** 2
            ),
        )

        grounded = grounding.ground(triplet_of(Milk, Drawer, PickUpAction))

        assert grounded.target_object_body is nearest

    def test_the_object_chosen_once_is_kept_for_the_rest_of_the_sequence(
        self, grounding, apartment_world_copy
    ) -> None:
        """
        The drawer a sequence opens has to be the drawer it then places into and closes
        again, or the plan opens one drawer and puts the milk in another.
        """
        annotate_second_drawer(apartment_world_copy)
        opened_drawer = grounding.ground(
            triplet_of(Drawer, Drawer, OpenAction)
        ).source_object_body

        placed_into = grounding.ground(
            triplet_of(Milk, Drawer, PickUpAction)
        ).target_object_body
        closed = grounding.ground(
            triplet_of(Drawer, Drawer, CloseAction)
        ).source_object_body

        assert placed_into is opened_drawer
        assert closed is opened_drawer

    def test_without_anything_to_be_near_the_robots_object_is_taken(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        """
        Opening a drawer names no second object, so the robot standing in the world is
        what decides which drawer is meant.
        """
        world, robot, _ = apartment_world_pr2_copy_with_context
        grounding = TripletGrounding(world=world)
        annotate_second_drawer(world)
        nearest = min(
            grounding.bodies_of(Drawer),
            key=lambda body: float(
                (body.global_pose.x - robot.root.global_pose.x) ** 2
                + (body.global_pose.y - robot.root.global_pose.y) ** 2
            ),
        )

        grounded = grounding.ground(triplet_of(Drawer, Drawer, OpenAction))

        assert grounded.source_object_body is nearest

    def test_a_world_without_a_robot_falls_back_to_the_first_object(
        self, grounding, apartment_world_copy
    ) -> None:
        annotate_second_drawer(apartment_world_copy)

        assert grounding.robot_body() is None
        assert grounding.body_of(Drawer) is grounding.bodies_of(Drawer)[0]
