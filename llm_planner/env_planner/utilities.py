"""
Finding the free space on a body.

An object is put down on the layer of air directly above a body's top face: the surface
of a countertop or a table, and the inside of an opened drawer, whose top face is its
floor. Which parts of that layer are still free is answered by
:func:`~semantic_digital_twin.reasoning.predicates.is_place_occupied`, asked once per
candidate place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import List, Optional

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.predicates import is_place_occupied
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import (
    BoundingBox,
    Box,
    Color,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Region

DEFAULT_PLACE_SIZE = Scale(0.1, 0.1, 0.1)
"""
Size of the volume checked at each candidate place: a ten centimetre cube, about what a
mug or a bowl takes up. Pass the object's own bounding box scale for an exact answer.
"""


MAXIMUM_PLACES_PER_AXIS = 4
"""
The most places an axis is divided into when a search is asked to stay within its
allowance (see :func:`unoccupied_region`).
"""


@dataclass(frozen=True)
class PlaceGrid:
    """
    How many candidate places a body's supporting layer is divided into, per axis.
    """

    along_x: int
    """
    Places along the layer's x axis.
    """

    along_y: int
    """
    Places along the layer's y axis.
    """

    def __post_init__(self) -> None:
        if self.along_x < 1 or self.along_y < 1:
            raise ValueError(f"a grid holds at least one place per axis, got {self}.")

    @property
    def place_count(self) -> int:
        """
        :return: How many places the grid holds, and so how many collision queries
            searching it costs.
        """
        return self.along_x * self.along_y

    def at_most(self, places: int) -> PlaceGrid:
        """
        The grid cut down to at most ``places`` per axis.

        Only a grid that is larger than that on *both* axes is cut down; one that is
        long on a single axis is left as it is.

        :param places: The most places an axis may be divided into.
        :return: The cut-down grid, or this one if it is small enough already.
        """
        if self.along_x > places and self.along_y > places:
            return PlaceGrid(along_x=places, along_y=places)
        return self


@dataclass
class RegionOccupancy:
    """
    The candidate places checked on one body, split into the free and the taken ones.
    """

    free: List[Pose] = field(default_factory=list)
    """
    Where an object of the checked size fits, in the order the places were searched.
    """

    occupied: List[Pose] = field(default_factory=list)
    """
    Where something already stands.
    """

    @property
    def is_free(self) -> bool:
        """
        :return: Whether any place on the body is still free.
        """
        return bool(self.free)

    def closest_free_place(self, target: Pose) -> Optional[Pose]:
        """
        :param target: The pose to stay near, e.g. where the object is now.
        :return: The free place nearest to it, or None if the body has none.
        """
        if not self.free:
            return None
        return min(self.free, key=lambda place: _distance(place, target))


def supporting_layer(body: Body, world: World, height: float) -> BoundingBox:
    """
    The layer of air a body supports: the volume of the given height standing on its top
    face, as wide and as deep as the body itself.

    :param body: The body to stand something on.
    :param world: The world the body belongs to.
    :param height: How tall the layer is, in metres.
    :return: The layer, in the world's root frame.
    """
    bounds = body.collision.as_bounding_box_collection_in_frame(
        world.root
    ).bounding_box()
    return BoundingBox(
        min_x=float(bounds.x_interval.lower),
        min_y=float(bounds.y_interval.lower),
        min_z=float(bounds.z_interval.upper),
        max_x=float(bounds.x_interval.upper),
        max_y=float(bounds.y_interval.upper),
        max_z=float(bounds.z_interval.upper) + height,
        origin=HomogeneousTransformationMatrix(reference_frame=world.root),
    )


def place_grid_for(
    body: Body,
    world: World,
    place_size: Scale = DEFAULT_PLACE_SIZE,
) -> PlaceGrid:
    """
    The grid that covers as much of a body as places of the given size can.

    Each axis takes as many places as fit along it whole, so the places sit edge to edge
    with no gap between them and none of them hangs over the body. What is left over is
    the remainder of the axis, always shorter than one place.

    A body narrower than a single place still gets one place, which then hangs over its
    edges - the smallest grid there is.

    :param body: The body to put something on.
    :param world: The world the body belongs to.
    :param place_size: Size of the volume that has to be free at a place.
    :return: The grid to search that body with.
    """
    layer = supporting_layer(body, world, place_size.z)
    return PlaceGrid(
        along_x=max(1, int(layer.depth / place_size.x)),
        along_y=max(1, int(layer.width / place_size.y)),
    )


def unoccupied_region(
    body: Body,
    world: World,
    place_size: Scale = DEFAULT_PLACE_SIZE,
    grid: Optional[PlaceGrid] = None,
    maximum_allowance: bool = False,
) -> RegionOccupancy:
    """
    Search a body's supporting layer for the places an object still fits in.

    Every place of the grid is checked with
    :func:`~semantic_digital_twin.reasoning.predicates.is_place_occupied`; the body
    itself is ignored, since standing on it is the point.

    A closed drawer has its cabinet directly above its floor and comes back with nothing
    free; the same drawer opened comes back with the part of it that has stuck out.

    ..note:: Each place costs one collision query against every body in the world, so a
        grid that covers a large body takes a while; pass a coarser
        :class:`PlaceGrid` to trade coverage for time.

    :param body: The body to put something on.
    :param world: The world the body belongs to.
    :param place_size: Size of the volume that has to be free at a place.
    :param grid: How the layer is divided into places, or None to cover as much of the
        body as places of that size can (see :func:`place_grid_for`).
    :param maximum_allowance: Whether the search stays within
        :data:`MAXIMUM_PLACES_PER_AXIS` places per axis, which keeps a large body from
        costing hundreds of collision queries at the price of covering it more coarsely.
    :return: The free and the taken places.
    """
    grid = grid or place_grid_for(body, world, place_size)
    if maximum_allowance:
        grid = grid.at_most(MAXIMUM_PLACES_PER_AXIS)
    layer = supporting_layer(body, world, place_size.z)
    checked_volume = _volume_at_origin(place_size, world)

    occupancy = RegionOccupancy()
    for pose in _places_in(layer, grid, world):
        taken = is_place_occupied(checked_volume, pose, world, [body])
        (occupancy.occupied if taken else occupancy.free).append(pose)
    return occupancy


def _places_in(layer: BoundingBox, grid: PlaceGrid, world: World) -> List[Pose]:
    """
    :param layer: The layer to divide into candidate places.
    :param grid: How the layer is divided.
    :param world: The world the poses are expressed in.
    :return: The centre of every place of the grid, row by row.
    """
    return [
        Pose.from_xyz_rpy(
            _place_center(
                layer.x_interval.lower, layer.x_interval.upper, grid.along_x, x
            ),
            _place_center(
                layer.y_interval.lower, layer.y_interval.upper, grid.along_y, y
            ),
            float(layer.z_interval.center()),
            reference_frame=world.root,
        )
        for x in range(grid.along_x)
        for y in range(grid.along_y)
    ]


def _place_center(lower: float, upper: float, places: int, index: int) -> float:
    """
    :param lower: Where the axis starts.
    :param upper: Where the axis ends.
    :param places: How many places the axis is divided into.
    :param index: Which place to take, counted from ``lower``.
    :return: The middle of that place.
    """
    return float(lower) + (index + 0.5) * (float(upper) - float(lower)) / places


def _volume_at_origin(place_size: Scale, world: World) -> BoundingBox:
    """
    :param place_size: The size the volume has.
    :param world: The world the volume is expressed in.
    :return: A box of that size centred on its own origin, as
        :func:`~semantic_digital_twin.reasoning.predicates.is_place_occupied` expects
        the region it moves to a pose.
    """
    return BoundingBox(
        min_x=-place_size.x / 2,
        min_y=-place_size.y / 2,
        min_z=-place_size.z / 2,
        max_x=place_size.x / 2,
        max_y=place_size.y / 2,
        max_z=place_size.z / 2,
        origin=HomogeneousTransformationMatrix(reference_frame=world.root),
    )


def _distance(place: Pose, target: Pose) -> float:
    """
    :param place: One of the searched places.
    :param target: The pose to measure to.
    :return: The straight-line distance between the two, in metres.
    """
    return float(
        (
            (place.x - target.x) ** 2
            + (place.y - target.y) ** 2
            + (place.z - target.z) ** 2
        )
        ** 0.5
    )


"""
# usage Example:
region = unoccupied_region(Counter_top[1].root,world)  
[place, *_] = region.free
Free_pose = Pose.from_xyz_rpy(
    place.x, place.y, place.z, yaw=1.57,
    reference_frame=place.reference_frame,                                                                                                                              
) 

"""

# %% showing the search in a viewer

PLACE_ALPHA = 0.35
"""
How opaque a drawn place is.

A region marks out space rather than filling it, so it is drawn see-through and
whatever stands there stays visible.
"""

FREE_PLACE_COLOR = Color(0.2, 0.8, 0.4, PLACE_ALPHA)
"""
Colour of a place an object still fits in.
"""

OCCUPIED_PLACE_COLOR = Color(0.9, 0.3, 0.3, PLACE_ALPHA)
"""
Colour of a place something already stands in.
"""

BOUNDING_BOX_COLOR = Color(0.3, 0.6, 1.0, PLACE_ALPHA)
"""
Colour of a body's bounding box.
"""


class PlaceState(StrEnum):
    """
    What a searched place turned out to be, and how its region is named.
    """

    FREE = "free"
    """
    Nothing stands there.
    """

    OCCUPIED = "occupied"
    """
    Something already does.
    """


def add_region_to_world(
    occupancy: RegionOccupancy,
    body: Body,
    world: World,
    place_size: Scale = DEFAULT_PLACE_SIZE,
) -> List[Region]:
    """
    Put the searched places into the world as regions, so a viewer draws them.

    One region per :class:`PlaceState`, each a box per place in its own colour, attached
    to the body itself: opening the drawer the places were searched on carries them
    along.

    :param occupancy: The searched places, as :func:`unoccupied_region` returns them.
    :param body: The body the places were searched on.
    :param world: The world to add the regions to.
    :param place_size: The size each place was checked with.
    :return: The regions added, in :class:`PlaceState` order, leaving out the states
        with no place in them.
    """
    regions = []
    with world.modify_world():
        for state, places, color in (
            (PlaceState.FREE, occupancy.free, FREE_PLACE_COLOR),
            (PlaceState.OCCUPIED, occupancy.occupied, OCCUPIED_PLACE_COLOR),
        ):
            if not places:
                continue
            region = _region_of_places(state, places, body, world, place_size, color)
            world.add_region(region)
            world.add_connection(FixedConnection(parent=body, child=region))
            regions.append(region)
    # A new region is a new frame, and a viewer can only draw a marker in a frame it can
    # transform: republishing the state puts every frame of the world, the new ones
    # included, on one fresh timestamp for whoever publishes them.
    world.notify_state_change()
    return regions


def _region_of_places(
    state: PlaceState,
    places: List[Pose],
    body: Body,
    world: World,
    place_size: Scale,
    color: Color,
) -> Region:
    """
    :param state: What the places turned out to be, naming the region.
    :param places: The places to draw, in the world's root frame.
    :param body: The body the region is attached to, and whose frame it is drawn in.
    :param world: The world the places are transformed in.
    :param place_size: The size of each drawn box.
    :param color: The colour of each drawn box.
    :return: One region holding a box per place.
    """
    boxes = [
        Box(
            origin=world.transform(place, body).to_homogeneous_matrix(),
            scale=place_size,
            color=color,
        )
        for place in places
    ]
    for box in boxes:
        box.origin.reference_frame = None
    return Region.from_shape_collection(
        name=PrefixedName(f"{body.name.name}_{state}_places", body.name.prefix),
        shape_collection=ShapeCollection(boxes),
    )


def add_bounding_box_to_world(body: Body, world: World) -> Region:
    """
    Put a body's bounding box into the world as a region, so a viewer draws it.

    The box encloses every collision shape the body has and is drawn in the body's own
    frame, so it travels with the body.

    :param body: The body whose bounding box is drawn.
    :param world: The world to add the region to.
    :return: The region added.
    """
    bounds = body.collision.as_bounding_box_collection_in_frame(body).bounding_box()
    box = bounds.as_shape()
    box.color = BOUNDING_BOX_COLOR
    box.origin.reference_frame = None
    region = Region.from_shape_collection(
        name=PrefixedName(f"{body.name.name}_bounding_box", body.name.prefix),
        shape_collection=ShapeCollection(shapes=[box]),
    )
    with world.modify_world():
        world.add_region(region)
        world.add_connection(FixedConnection(parent=body, child=region))
    world.notify_state_change()
    return region
