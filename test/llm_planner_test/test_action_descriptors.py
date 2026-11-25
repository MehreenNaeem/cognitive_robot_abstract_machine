"""
Tests of the handle an object is opened and closed by, in the apartment.
"""

from __future__ import annotations

import logging

import pytest

from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import ActionNode
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from llm_planner.env_planner import action_descriptors
from llm_planner.env_planner.action_descriptors import (
    action_designator,
    annotations_of,
    get_access_handle,
    plan_grounding,
)
from llm_planner.env_planner.triplet_generator import Triplet
from llm_planner.env_planner.triplet_grounding import TripletGrounding
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Door,
    Drawer,
    Handle,
    Milk,
)

DRAWER_BODY_NAME = "cabinet10_drawer_top"
"""
The apartment drawer the fixture annotates, handle and all.
"""

DRAWER_HANDLE_BODY_NAME = "handle_cab10_t"
"""
The grip on that drawer's front.
"""

CABINET_BODY_NAME = "cabinet1"
"""
An apartment cabinet, which is opened by a door rather than by itself.
"""

CABINET_DOOR_BODY_NAME = "cabinet1_door_top_left"
CABINET_DOOR_HANDLE_BODY_NAME = "handle_cab1_top_door"

MILK_BODY_NAME = "milk.stl"
"""
An object that is not opened at all.
"""


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
    Collect what :mod:`llm_planner.env_planner.action_descriptors` warns about.
    """
    handler = CollectedWarnings()
    action_descriptors.logger.addHandler(handler)
    yield handler
    action_descriptors.logger.removeHandler(handler)


def annotate_cabinet(world) -> Cabinet:
    """
    Annotate an apartment cabinet with a door that has a handle.

    :param world: The world to annotate in.
    :return: The cabinet annotation.
    """
    cabinet = Cabinet(
        root=world.get_body_by_name(CABINET_BODY_NAME),
        doors=[
            Door(
                root=world.get_body_by_name(CABINET_DOOR_BODY_NAME),
                handle=Handle(
                    root=world.get_body_by_name(CABINET_DOOR_HANDLE_BODY_NAME)
                ),
            )
        ],
    )
    with world.modify_world():
        world.add_semantic_annotation_recursively(cabinet)
    return cabinet


# %% the annotations of a body


class TestAnnotationsOfBody:
    def test_finds_the_annotation_a_body_is_the_root_of(
        self, apartment_world_copy
    ) -> None:
        drawer_body = apartment_world_copy.get_body_by_name(DRAWER_BODY_NAME)

        annotations = annotations_of(drawer_body, apartment_world_copy)

        assert [type(annotation) for annotation in annotations] == [Drawer]

    def test_a_body_no_annotation_is_rooted_at_has_none(
        self, apartment_world_copy
    ) -> None:
        door_body = apartment_world_copy.get_body_by_name(CABINET_DOOR_BODY_NAME)

        assert annotations_of(door_body, apartment_world_copy) == []


# %% the handle an object is opened by


class TestAccessHandle:
    def test_a_drawer_is_opened_by_its_own_handle(self, apartment_world_copy) -> None:
        drawer_body = apartment_world_copy.get_body_by_name(DRAWER_BODY_NAME)
        [drawer] = annotations_of(drawer_body, apartment_world_copy)

        handle = get_access_handle(drawer_body, apartment_world_copy)

        assert handle is drawer.handle.root
        assert handle.name.name == DRAWER_HANDLE_BODY_NAME

    def test_a_cabinet_is_opened_by_the_handle_of_its_door(
        self, apartment_world_copy
    ) -> None:
        cabinet = annotate_cabinet(apartment_world_copy)
        cabinet_body = apartment_world_copy.get_body_by_name(CABINET_BODY_NAME)

        handle = get_access_handle(cabinet_body, apartment_world_copy)

        [door] = cabinet.doors
        assert handle is door.handle.root
        assert handle.name.name == CABINET_DOOR_HANDLE_BODY_NAME

    def test_a_door_is_opened_by_its_own_handle(self, apartment_world_copy) -> None:
        """
        The rule follows the annotation, so a door asked about directly answers with its
        own handle rather than being searched for one.
        """
        cabinet = annotate_cabinet(apartment_world_copy)
        [door] = cabinet.doors

        assert get_access_handle(door.root, apartment_world_copy) is door.handle.root

    def test_an_object_whose_annotation_names_no_handle_is_searched_for_one(
        self, apartment_world_copy
    ) -> None:
        """
        A drawer annotation the reasoner inferred carries no handle, while the grip on
        its front is annotated as one - that grip is what opens it.
        """
        drawer_body = apartment_world_copy.get_body_by_name(DRAWER_BODY_NAME)
        [drawer] = annotations_of(drawer_body, apartment_world_copy)
        drawer.handle = None

        handle = get_access_handle(drawer_body, apartment_world_copy)

        assert handle.name.name == DRAWER_HANDLE_BODY_NAME

    def test_an_object_with_no_handle_anywhere_is_reported(
        self, apartment_world_copy, reported
    ) -> None:
        milk_body = apartment_world_copy.get_body_by_name(MILK_BODY_NAME)

        assert get_access_handle(milk_body, apartment_world_copy) is None
        assert reported.messages == [
            f"{milk_body.name} has no handle to open or close it by."
        ]

    def test_finding_a_handle_is_not_warned_about(
        self, apartment_world_copy, reported
    ) -> None:
        drawer_body = apartment_world_copy.get_body_by_name(DRAWER_BODY_NAME)

        get_access_handle(drawer_body, apartment_world_copy)

        assert reported.messages == []


# %% the plan a grounded sequence becomes


def grounded_sequence(world, *triplets) -> list:
    """
    Ground triplets one after another, as a sequence is grounded.

    :param world: The world to ground them in.
    :param triplets: The triplets to ground, in execution order.
    :return: The grounded triplets.
    """
    grounding = TripletGrounding(world=world)
    return [grounding.ground(triplet) for triplet in triplets]


def triplet_of(source, target, action) -> Triplet:
    return Triplet(source_object=source, target_object=target, action=action)


def performed_actions(steps, context) -> list[str]:
    """
    :param steps: The steps :func:`plan_grounding` returned.
    :param context: The context to compose them into a plan in.
    :return: The name of every action designator in those steps, in execution order.
    """
    plan = sequential(steps, context=context).plan
    return [
        type(node.designator).__name__
        for node in plan.nodes
        if isinstance(node, ActionNode)
    ]


class TestActionDesignator:
    def test_an_opening_action_is_performed_on_the_handle(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        """
        A drawer is opened by pulling its handle, so the handle is what the action takes
        hold of - not the drawer the triplet named.
        """
        world, _, context = apartment_world_pr2_copy_with_context
        [grounded] = grounded_sequence(world, triplet_of(Drawer, Drawer, OpenAction))

        action = action_designator(grounded, context)

        assert isinstance(action, OpenAction)
        assert action.object_designator is get_access_handle(
            grounded.source_object_body, world
        )

    def test_a_picking_action_is_performed_on_the_object_itself(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        [grounded] = grounded_sequence(world, triplet_of(Milk, Drawer, PickUpAction))

        action = action_designator(grounded, context)

        assert isinstance(action, PickUpAction)
        assert action.object_designator is grounded.source_object_body
        assert action.arm is grounded.arm
        assert action.grasp_description is grounded.grasp_description

    def test_an_action_still_missing_an_argument_is_left_out(
        self, apartment_world_pr2_copy_with_context, reported
    ) -> None:
        """
        A closed drawer gives a placing triplet nowhere to put the object, and an action
        that cannot be built has to be reported rather than guessed at.
        """
        world, _, context = apartment_world_pr2_copy_with_context
        [grounded] = grounded_sequence(world, triplet_of(Milk, Drawer, PlaceAction))

        assert action_designator(grounded, context) is None
        assert reported.messages == [
            "PlaceAction is left out of the plan: no target_location was found for it."
        ]

    def test_an_action_on_something_with_no_handle_is_left_out(
        self, apartment_world_pr2_copy_with_context, reported
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        [grounded] = grounded_sequence(world, triplet_of(Milk, Milk, OpenAction))

        assert action_designator(grounded, context) is None
        assert reported.messages[-1] == (
            f"OpenAction is left out of the plan: "
            f"{grounded.source_object_body.name} has no handle to perform it on."
        )


class TestPlanGrounding:
    def test_every_action_is_reached_before_it_is_performed(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        """
        The robot has to stand within reach of what it acts on, so each action is
        preceded by the navigation that brings it there (see :func:`goto_location`).
        """
        world, _, context = apartment_world_pr2_copy_with_context
        grounded_triplets = grounded_sequence(
            world,
            triplet_of(Drawer, Drawer, OpenAction),
            triplet_of(Milk, Drawer, PickUpAction),
        )

        steps = plan_grounding(grounded_triplets, context)

        assert performed_actions(steps, context) == [
            ParkArmsAction.__name__,
            MoveTorsoAction.__name__,
            OpenAction.__name__,
            ParkArmsAction.__name__,
            MoveTorsoAction.__name__,
            PickUpAction.__name__,
        ]

    def test_the_steps_are_composed_into_a_plan_in_the_given_context(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        grounded_triplets = grounded_sequence(
            world, triplet_of(Drawer, Drawer, OpenAction)
        )

        plan = sequential(
            plan_grounding(grounded_triplets, context), context=context
        ).plan

        assert plan.context is context
        assert plan.world is world

    def test_a_triplet_that_cannot_be_performed_leaves_the_rest_of_the_plan(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        grounded_triplets = grounded_sequence(
            world,
            triplet_of(Drawer, Drawer, OpenAction),
            triplet_of(Milk, Drawer, PlaceAction),
        )

        steps = plan_grounding(grounded_triplets, context)

        assert PlaceAction.__name__ not in performed_actions(steps, context)
        assert OpenAction.__name__ in performed_actions(steps, context)

    def test_an_empty_sequence_yields_no_steps(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context

        assert performed_actions(plan_grounding([], context), context) == []
