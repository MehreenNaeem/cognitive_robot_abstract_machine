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

DEFAULT_RESOLUTION = 3
"""
How many candidate places each horizontal axis is divided into.

Every place costs one collision query against every body in the world, so the grid stays
coarse unless a caller asks for a finer one.
"""


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


def unoccupied_region(
    body: Body,
    world: World,
    place_size: Scale = DEFAULT_PLACE_SIZE,
    resolution: int = DEFAULT_RESOLUTION,
) -> RegionOccupancy:
    """
    Search a body's supporting layer for the places an object still fits in.

    The layer is divided into a ``resolution`` by ``resolution`` grid and every cell is
    checked with
    :func:`~semantic_digital_twin.reasoning.predicates.is_place_occupied`; the body
    itself is ignored, since standing on it is the point.

    A closed drawer has its cabinet directly above its floor and comes back with nothing
    free; the same drawer opened comes back with the part of it that has stuck out.

    :param body: The body to put something on.
    :param world: The world the body belongs to.
    :param place_size: Size of the volume that has to be free at a place.
    :param resolution: How many places each horizontal axis is divided into.
    :return: The free and the taken places.
    :raises ValueError: If the resolution is not positive.
    """
    if resolution < 1:
        raise ValueError(f"resolution must be at least 1, got {resolution}.")
    layer = supporting_layer(body, world, place_size.z)
    checked_volume = _volume_at_origin(place_size, world)

    occupancy = RegionOccupancy()
    for pose in _places_in(layer, resolution, world):
        taken = is_place_occupied(checked_volume, pose, world, [body])
        (occupancy.occupied if taken else occupancy.free).append(pose)
    return occupancy


def _places_in(layer: BoundingBox, resolution: int, world: World) -> List[Pose]:
    """
    :param layer: The layer to divide into candidate places.
    :param resolution: How many places each horizontal axis is divided into.
    :param world: The world the poses are expressed in.
    :return: The centre of every cell of the grid, row by row.
    """
    return [
        Pose.from_xyz_rpy(
            _cell_center(layer.x_interval.lower, layer.x_interval.upper, resolution, x),
            _cell_center(layer.y_interval.lower, layer.y_interval.upper, resolution, y),
            float(layer.z_interval.center()),
            reference_frame=world.root,
        )
        for x in range(resolution)
        for y in range(resolution)
    ]


def _cell_center(lower: float, upper: float, resolution: int, index: int) -> float:
    """
    :param lower: Where the axis starts.
    :param upper: Where the axis ends.
    :param resolution: How many cells the axis is divided into.
    :param index: Which cell to take, counted from ``lower``.
    :return: The middle of that cell.
    """
    return float(lower) + (index + 0.5) * (float(upper) - float(lower)) / resolution


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

def visualize_bbox(body: Body, world: World):
    bounds = body.collision.as_bounding_box_collection_in_frame(body).bounding_box()

    box = bounds.as_shape()
    box.color = Color(0.3, 0.6, 1.0, 0.3)
    box.origin.reference_frame = None
    region = Region.from_shape_collection(
        name=PrefixedName(f"{body.name.name}_bounding_box", body.name.prefix),
        shape_collection=ShapeCollection(shapes=[box]),
    )
    with world.modify_world():
        world.add_region(region)
        world.add_connection(FixedConnection(parent=body, child=region))
    world.notify_state_change()

    #TODO, fix the frame, fix the region currently it shown boxes which are apart from each other