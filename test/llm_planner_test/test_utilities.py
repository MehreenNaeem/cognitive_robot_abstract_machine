"""
Tests of finding the free space on a body, in the apartment: a countertop, and a drawer
both closed and open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pytest

from llm_planner.env_planner.utilities import (
    BOUNDING_BOX_COLOR,
    MAXIMUM_PLACES_PER_AXIS,
    DEFAULT_PLACE_SIZE,
    FREE_PLACE_COLOR,
    PLACE_ALPHA,
    PlaceGrid,
    PlaceState,
    add_bounding_box_to_world,
    add_region_to_world,
    place_grid_for,
    supporting_layer,
    unoccupied_region,
)
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
    StateChangeCallback,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


class Notification(StrEnum):
    """
    What a world notified its callbacks of.
    """

    MODEL = "model"
    """
    The world's model changed: something was added to or removed from it.
    """

    STATE = "state"
    """
    The world's state changed: its frames moved.
    """


@dataclass(eq=False)
class NotificationOrder(ModelChangeCallback, StateChangeCallback):
    """
    Records the order a world notified of model and state changes in.
    """

    notifications: list[Notification] = field(default_factory=list)
    """
    What was notified, oldest first.
    """

    def on_model_change(self, **kwargs) -> None:
        self.notifications.append(Notification.MODEL)

    def on_state_change(self, **kwargs) -> None:
        self.notifications.append(Notification.STATE)


def displacement(before: Pose, after: Pose) -> tuple[float, float, float]:
    """
    How far something moved between two poses.

    :param before: Where it was.
    :param after: Where it ended up.
    :return: The difference along each axis, in metres.
    """
    return (
        float(after.x) - float(before.x),
        float(after.y) - float(before.y),
        float(after.z) - float(before.z),
    )


DRAWER_NAME = "cabinet10_drawer_top"
"""
The apartment drawer the tests open and close.
"""

COUNTERTOP_NAME = "table_area_main"
"""
An apartment countertop with nothing standing on it.
"""

COARSE_GRID = PlaceGrid(along_x=3, along_y=3)
"""
The grid the searches that are not about coverage use: covering a whole countertop costs
one collision query per place, and these tests do not need the places to tile it.
"""


def opened(world: World, body: Body, position: float) -> None:
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


# %% the layer an object is put down on


class TestSupportingLayer:
    def test_stands_on_the_bodys_top_face(self, apartment_world_copy) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        bounds = countertop.collision.as_bounding_box_collection_in_frame(
            apartment_world_copy.root
        ).bounding_box()

        layer = supporting_layer(countertop, apartment_world_copy, 0.1)

        assert layer.min_z == pytest.approx(float(bounds.z_interval.upper))

    def test_is_as_tall_as_asked_and_as_wide_as_the_body(
        self, apartment_world_copy
    ) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        bounds = countertop.collision.as_bounding_box_collection_in_frame(
            apartment_world_copy.root
        ).bounding_box()

        layer = supporting_layer(countertop, apartment_world_copy, 0.1)

        assert layer.height == pytest.approx(0.1)
        assert layer.depth == pytest.approx(bounds.depth)
        assert layer.width == pytest.approx(bounds.width)


# %% which places on a body are free


class TestUnoccupiedRegion:
    def test_a_countertop_with_nothing_on_it_is_free(
        self, apartment_world_copy
    ) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)

        occupancy = unoccupied_region(
            countertop, apartment_world_copy, grid=PlaceGrid(3, 3)
        )

        assert occupancy.is_free
        assert occupancy.occupied == []

    def test_every_place_of_the_grid_is_answered_for(
        self, apartment_world_copy
    ) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        grid = PlaceGrid(3, 2)

        occupancy = unoccupied_region(countertop, apartment_world_copy, grid=grid)

        assert len(occupancy.free) + len(occupancy.occupied) == grid.place_count

    def test_the_grid_covering_the_body_is_searched_by_default(
        self, apartment_world_copy
    ) -> None:
        drawer = apartment_world_copy.get_body_by_name(DRAWER_NAME)
        grid = place_grid_for(drawer, apartment_world_copy)

        occupancy = unoccupied_region(drawer, apartment_world_copy)

        assert len(occupancy.free) + len(occupancy.occupied) == grid.place_count

    def test_a_free_place_lies_in_the_bodys_supporting_layer(
        self, apartment_world_copy
    ) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        layer = supporting_layer(countertop, apartment_world_copy, 0.1)

        [place, *_] = unoccupied_region(
            countertop, apartment_world_copy, grid=COARSE_GRID
        ).free

        assert layer.x_interval.lower <= float(place.x) <= layer.x_interval.upper
        assert layer.y_interval.lower <= float(place.y) <= layer.y_interval.upper
        assert float(place.z) == pytest.approx(float(layer.z_interval.center()))

    def test_a_closed_drawer_has_nowhere_to_put_anything(
        self, apartment_world_copy
    ) -> None:
        """
        The cabinet stands directly above a closed drawer's floor, so the layer it
        supports is taken all the way across.
        """
        drawer = apartment_world_copy.get_body_by_name(DRAWER_NAME)

        occupancy = unoccupied_region(drawer, apartment_world_copy, grid=COARSE_GRID)

        assert not occupancy.is_free
        assert occupancy.free == []

    def test_opening_a_drawer_frees_the_part_that_stuck_out(
        self, apartment_world_copy
    ) -> None:
        drawer = apartment_world_copy.get_body_by_name(DRAWER_NAME)
        opened(apartment_world_copy, drawer, 0.4)

        occupancy = unoccupied_region(drawer, apartment_world_copy, grid=COARSE_GRID)

        assert occupancy.is_free
        assert occupancy.occupied, "the part still inside the cabinet stays taken"

    def test_rejects_a_grid_with_no_places_in_it(self) -> None:
        with pytest.raises(ValueError):
            PlaceGrid(along_x=0, along_y=3)


class TestGridAllowance:
    """
    A search can be told to stay within an allowance, so a large body does not cost
    hundreds of collision queries.
    """

    def test_a_grid_larger_on_both_axes_is_cut_down(self) -> None:
        grid = PlaceGrid(along_x=8, along_y=16)

        assert grid.at_most(MAXIMUM_PLACES_PER_AXIS) == PlaceGrid(
            along_x=MAXIMUM_PLACES_PER_AXIS, along_y=MAXIMUM_PLACES_PER_AXIS
        )

    def test_a_grid_long_on_one_axis_only_is_left_alone(self) -> None:
        grid = PlaceGrid(along_x=16, along_y=2)

        assert grid.at_most(MAXIMUM_PLACES_PER_AXIS) is grid

    def test_a_grid_already_within_the_allowance_is_left_alone(self) -> None:
        grid = PlaceGrid(along_x=MAXIMUM_PLACES_PER_AXIS, along_y=2)

        assert grid.at_most(MAXIMUM_PLACES_PER_AXIS) is grid

    def test_a_search_within_the_allowance_checks_fewer_places(
        self, apartment_world_copy
    ) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        covering = place_grid_for(countertop, apartment_world_copy)
        assert covering.along_x > MAXIMUM_PLACES_PER_AXIS
        assert covering.along_y > MAXIMUM_PLACES_PER_AXIS

        occupancy = unoccupied_region(
            countertop, apartment_world_copy, maximum_allowance=True
        )

        assert len(occupancy.free) + len(occupancy.occupied) == (
            MAXIMUM_PLACES_PER_AXIS**2
        )

    def test_a_search_is_only_cut_down_when_it_is_asked_for(
        self, apartment_world_copy
    ) -> None:
        """
        The allowance has to be asked for: a search that says nothing covers the body.
        """
        drawer = apartment_world_copy.get_body_by_name(DRAWER_NAME)
        covering = place_grid_for(drawer, apartment_world_copy)

        occupancy = unoccupied_region(drawer, apartment_world_copy)

        assert len(occupancy.free) + len(occupancy.occupied) == covering.place_count


class TestPlaceGridFor:
    """
    The grid that covers as much of a body as places of a given size can.
    """

    def test_takes_as_many_places_as_fit_along_each_axis(
        self, apartment_world_copy
    ) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        layer = supporting_layer(countertop, apartment_world_copy, DEFAULT_PLACE_SIZE.z)

        grid = place_grid_for(countertop, apartment_world_copy)

        assert grid.along_x == int(layer.depth / DEFAULT_PLACE_SIZE.x)
        assert grid.along_y == int(layer.width / DEFAULT_PLACE_SIZE.y)

    def test_no_place_hangs_over_the_body(self, apartment_world_copy) -> None:
        """
        A place that sticks out is checked against whatever stands beside the body
        rather than on it, so the grid stops at the last one that fits whole.
        """
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        layer = supporting_layer(countertop, apartment_world_copy, DEFAULT_PLACE_SIZE.z)

        grid = place_grid_for(countertop, apartment_world_copy)

        assert grid.along_x * DEFAULT_PLACE_SIZE.x <= layer.depth
        assert grid.along_y * DEFAULT_PLACE_SIZE.y <= layer.width

    def test_what_is_left_over_is_shorter_than_one_place(
        self, apartment_world_copy
    ) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        layer = supporting_layer(countertop, apartment_world_copy, DEFAULT_PLACE_SIZE.z)

        grid = place_grid_for(countertop, apartment_world_copy)

        assert layer.depth - grid.along_x * DEFAULT_PLACE_SIZE.x < DEFAULT_PLACE_SIZE.x
        assert layer.width - grid.along_y * DEFAULT_PLACE_SIZE.y < DEFAULT_PLACE_SIZE.y

    def test_a_bigger_place_is_searched_for_in_fewer_spots(
        self, apartment_world_copy
    ) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)

        small = place_grid_for(countertop, apartment_world_copy, Scale(0.1, 0.1, 0.1))
        large = place_grid_for(countertop, apartment_world_copy, Scale(0.2, 0.2, 0.1))

        assert large.place_count < small.place_count

    def test_a_body_smaller_than_one_place_still_gets_one(
        self, apartment_world_copy
    ) -> None:
        drawer = apartment_world_copy.get_body_by_name(DRAWER_NAME)

        grid = place_grid_for(drawer, apartment_world_copy, Scale(10.0, 10.0, 0.1))

        assert grid == PlaceGrid(along_x=1, along_y=1)

    def test_the_places_of_the_fitted_grid_lie_on_the_body(
        self, apartment_world_copy
    ) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        layer = supporting_layer(countertop, apartment_world_copy, DEFAULT_PLACE_SIZE.z)
        grid = place_grid_for(countertop, apartment_world_copy)

        occupancy = unoccupied_region(
            countertop, apartment_world_copy, grid=PlaceGrid(grid.along_x, 1)
        )

        for place in occupancy.free + occupancy.occupied:
            assert layer.x_interval.lower <= float(place.x) <= layer.x_interval.upper
            assert layer.y_interval.lower <= float(place.y) <= layer.y_interval.upper


# %% choosing one of the free places


class TestClosestFreePlace:
    def test_takes_the_place_nearest_to_the_target(self, apartment_world_copy) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        occupancy = unoccupied_region(
            countertop, apartment_world_copy, grid=COARSE_GRID
        )
        target = occupancy.free[-1]

        assert occupancy.closest_free_place(target) is target

    def test_a_body_with_no_free_place_has_none(self, apartment_world_copy) -> None:
        drawer = apartment_world_copy.get_body_by_name(DRAWER_NAME)
        occupancy = unoccupied_region(drawer, apartment_world_copy, grid=COARSE_GRID)
        target = Pose.from_xyz_rpy(0, 0, 0, reference_frame=apartment_world_copy.root)

        assert occupancy.closest_free_place(target) is None


# %% showing the search in a viewer


class TestAddRegionToWorld:
    def test_adds_one_region_per_place_state(self, apartment_world_copy) -> None:
        drawer = apartment_world_copy.get_body_by_name(DRAWER_NAME)
        opened(apartment_world_copy, drawer, 0.4)
        occupancy = unoccupied_region(drawer, apartment_world_copy, grid=COARSE_GRID)

        regions = add_region_to_world(occupancy, drawer, apartment_world_copy)

        assert [region.name.name for region in regions] == [
            f"{DRAWER_NAME}_{PlaceState.FREE}_places",
            f"{DRAWER_NAME}_{PlaceState.OCCUPIED}_places",
        ]
        assert [len(region.area.shapes) for region in regions] == [
            len(occupancy.free),
            len(occupancy.occupied),
        ]

    def test_the_regions_are_part_of_the_world(self, apartment_world_copy) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        occupancy = unoccupied_region(
            countertop, apartment_world_copy, grid=COARSE_GRID
        )

        regions = add_region_to_world(occupancy, countertop, apartment_world_copy)

        assert set(regions) <= set(apartment_world_copy.regions)

    def test_leaves_out_a_state_no_place_is_in(self, apartment_world_copy) -> None:
        """
        A closed drawer has no free place, and an empty region would draw nothing.
        """
        drawer = apartment_world_copy.get_body_by_name(DRAWER_NAME)
        occupancy = unoccupied_region(drawer, apartment_world_copy, grid=COARSE_GRID)

        regions = add_region_to_world(occupancy, drawer, apartment_world_copy)

        assert [region.name.name for region in regions] == [
            f"{DRAWER_NAME}_{PlaceState.OCCUPIED}_places"
        ]

    def test_a_drawn_place_sits_where_the_place_was_searched(
        self, apartment_world_copy
    ) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        occupancy = unoccupied_region(
            countertop, apartment_world_copy, grid=COARSE_GRID
        )

        [free_places] = add_region_to_world(occupancy, countertop, apartment_world_copy)

        drawn = apartment_world_copy.transform(
            free_places.area.shapes[0].origin, apartment_world_copy.root
        )
        assert float(drawn.x) == pytest.approx(float(occupancy.free[0].x))
        assert float(drawn.y) == pytest.approx(float(occupancy.free[0].y))
        assert float(drawn.z) == pytest.approx(float(occupancy.free[0].z))

    def test_a_drawn_place_follows_the_body_it_was_searched_on(
        self, apartment_world_copy
    ) -> None:
        """
        The places belong to the drawer, so closing it takes them along instead of
        leaving them hanging where the drawer used to be.
        """
        drawer = apartment_world_copy.get_body_by_name(DRAWER_NAME)
        opened(apartment_world_copy, drawer, 0.4)
        occupancy = unoccupied_region(drawer, apartment_world_copy, grid=COARSE_GRID)
        [free_places, *_] = add_region_to_world(occupancy, drawer, apartment_world_copy)
        drawn_before = apartment_world_copy.transform(
            free_places.area.shapes[0].origin, apartment_world_copy.root
        )
        drawer_before = drawer.global_pose

        opened(apartment_world_copy, drawer, 0.1)

        drawn_after = apartment_world_copy.transform(
            free_places.area.shapes[0].origin, apartment_world_copy.root
        )
        assert displacement(drawn_before, drawn_after) == pytest.approx(
            displacement(drawer_before, drawer.global_pose), abs=1e-6
        )

    def test_a_place_is_drawn_see_through_in_its_states_colour(
        self, apartment_world_copy
    ) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        occupancy = unoccupied_region(
            countertop, apartment_world_copy, grid=COARSE_GRID
        )

        [free_places] = add_region_to_world(occupancy, countertop, apartment_world_copy)

        assert free_places.area.shapes[0].color == FREE_PLACE_COLOR
        assert FREE_PLACE_COLOR.A == PLACE_ALPHA

    def test_the_worlds_frames_are_republished_after_the_regions_are_added(
        self, apartment_world_copy
    ) -> None:
        """
        A region is a frame of its own, and a viewer draws its places only in a frame it
        can transform. Whoever publishes frames does so on a state change, so the state
        has to be republished after the model gained the region - otherwise the new
        frame's first transform is younger than the rest of the tree, and a viewer
        looking the places up at the tree's common time finds nothing there.
        """
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        occupancy = unoccupied_region(
            countertop, apartment_world_copy, grid=COARSE_GRID
        )
        spy = NotificationOrder(_world=apartment_world_copy)

        add_region_to_world(occupancy, countertop, apartment_world_copy)

        assert spy.notifications[-1] == Notification.STATE
        assert Notification.MODEL in spy.notifications


# %% a body's bounding box


class TestAddBoundingBoxToWorld:
    def test_the_box_encloses_the_bodys_collision_shapes(
        self, apartment_world_copy
    ) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        bounds = countertop.collision.as_bounding_box_collection_in_frame(
            countertop
        ).bounding_box()

        region = add_bounding_box_to_world(countertop, apartment_world_copy)

        [box] = region.area.shapes
        assert box.scale.x == pytest.approx(bounds.depth)
        assert box.scale.y == pytest.approx(bounds.width)
        assert box.scale.z == pytest.approx(bounds.height)

    def test_the_region_is_named_after_the_body(self, apartment_world_copy) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)

        region = add_bounding_box_to_world(countertop, apartment_world_copy)

        assert region.name.name == f"{COUNTERTOP_NAME}_bounding_box"
        assert region.name.prefix == countertop.name.prefix

    def test_the_region_is_part_of_the_world(self, apartment_world_copy) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)

        region = add_bounding_box_to_world(countertop, apartment_world_copy)

        assert region in apartment_world_copy.regions

    def test_the_box_is_drawn_see_through(self, apartment_world_copy) -> None:
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)

        region = add_bounding_box_to_world(countertop, apartment_world_copy)

        assert region.area.shapes[0].color == BOUNDING_BOX_COLOR
        assert BOUNDING_BOX_COLOR.A == PLACE_ALPHA

    def test_the_box_follows_the_body(self, apartment_world_copy) -> None:
        drawer = apartment_world_copy.get_body_by_name(DRAWER_NAME)
        region = add_bounding_box_to_world(drawer, apartment_world_copy)
        drawn_before = apartment_world_copy.transform(
            region.area.shapes[0].origin, apartment_world_copy.root
        )
        drawer_before = drawer.global_pose

        opened(apartment_world_copy, drawer, 0.4)

        drawn_after = apartment_world_copy.transform(
            region.area.shapes[0].origin, apartment_world_copy.root
        )
        moved = displacement(drawer_before, drawer.global_pose)
        assert moved[0] != pytest.approx(0.0), "the drawer did not move at all"
        assert displacement(drawn_before, drawn_after) == pytest.approx(moved, abs=1e-6)

    def test_the_worlds_frames_are_republished_after_the_box_is_added(
        self, apartment_world_copy
    ) -> None:
        """
        The box is drawn in a frame of its own, which a viewer can only use once
        something has published it (see
        :attr:`~semantic_digital_twin.adapters.ros.tf_publisher.TFPublisher.republish_interval_seconds`).
        """
        countertop = apartment_world_copy.get_body_by_name(COUNTERTOP_NAME)
        spy = NotificationOrder(_world=apartment_world_copy)

        add_bounding_box_to_world(countertop, apartment_world_copy)

        assert spy.notifications[-1] == Notification.STATE
