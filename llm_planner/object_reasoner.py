'''

location
type
visible
reachable
parent
child
near_to -> use contact
receptacle
is_container_open
is_place_occupied

'''
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

import numpy as np

from krrood.entity_query_language.factories import variable, an, entity
from krrood.entity_query_language.predicate import symbolic_function
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import Pose, HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.world_entity import Body, KinematicStructureEntity
from semantic_digital_twin.reasoning.predicates import visible, reachable, is_container_open, is_place_occupied
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


def semantic_annotation_of(body: Body, world: World) -> Optional[SemanticAnnotation]:
    """
    Looks up the semantic annotation whose root body name matches `body`'s name.
    :param body: The body whose semantic annotation should be resolved.
    :param world: The world providing the semantic annotations to search.
    :return: The matching semantic annotation, or None if none was found.
    """
    semantic_annotation = variable(SemanticAnnotation, world.semantic_annotations)
    query = an(
        entity(semantic_annotation).where(
            semantic_annotation.root.name.name.lower() == body.name.name.lower()
        )
    )
    results = list(query.evaluate())
    # If more than one match is possible, decide how to handle ambiguity here.
    return results[0] if results else None


def type_object(body: Body, world: World) -> Optional[str]:
    """
    Resolves the semantic *type* of `body`.
    :param body: The body whose semantic type should be resolved.
    :param world: The world providing the semantic annotations to search.
    :return: The name of the matching semantic type, or None if none was found.
    """
    semantic_annotation = semantic_annotation_of(body, world)
    return semantic_annotation.name.name if semantic_annotation else None


from dataclasses import dataclass
from typing import List, Optional


@dataclass(eq=False)
class ObjectDescriptor:
    """
    A semantic descriptor for a Body.
    All properties are computed dynamically from the current world state.
    """
    body: Body

    @property
    def location(self) -> Pose:
        """Current world pose of the object."""
        return self.body.global_pose

    def of_type(self, world: World) -> str | None:
        """Semantic type of the object."""
        return type_object(self.body, world)


    '''def reachable(self, root: Body, tip: Body) -> bool:
        """Whether the object is reachable by the robot."""
        return reachable(self.body.global_pose, root, tip)'''

    def parent(self):
        """Returns the parent name and index"""
        return self.body.parent_kinematic_structure_entity.name.name,self.body.parent_kinematic_structure_entity.index

    def children(self) -> List[List[Any]]:
        """Returns the name and index of each child body."""
        return [[child.name.name, child.index] for child in self.body.child_kinematic_structure_entities]

    def is_receptacle(self, world: World) -> bool:
        """
        Whether another object can be placed on or inside this object, i.e. whether it
        offers a supporting surface (flat, like a table) or a cavity (hollow, like a drawer).
        :param world: The world providing the semantic annotations to search.
        """
        semantic_annotation = semantic_annotation_of(self.body, world)
        return isinstance(semantic_annotation, HasSupportingSurface)

    def is_container_open(self, world: World, door) -> Optional[bool]:
        """Whether this body (if a container) is open."""
        return is_container_open(self.body, world, door)

    def is_place_occupied(
        self,
        world: World,
        allowed_bodies=None,
    ) -> Optional[bool]:
        """Checks whether a placement region is occupied, if this body is a receptacle."""
        box = self.body.collision.as_bounding_box_collection_in_frame(world.root).bounding_boxes[0]
        if not self.is_receptacle(world):
            return None
        return is_place_occupied(
            box,
            self.body.global_pose,
            world,
            allowed_bodies,
        )


#######
'''from semantic_digital_twin.robots.pr2 import PR2KinectV1,PR2
from semantic_digital_twin.world import World
world = World()
robot = PR2.from_world(world)
camera_body = robot.root._world.get_body_in_branch_by_name(
                robot.root, "wide_stereo_optical_frame"
            )
pr2_camera = PR2KinectV1.setup_default_configuration_in_world_below_robot_root(camera_body)

@symbolic_function
def get_visible_bodies(camera: Body) -> List[KinematicStructureEntity]:
    """
    Get all bodies and regions that are visible from the given camera using a segmentation mask.

    :param camera: The camera for which the visible objects should be returned
    :return: A list of bodies/regions that are visible from the camera
    """
    rt = RayTracer(world)
    rt.update_scene()

    # This ignores the camera orientation and sets it to identity
    cam_pose = np.eye(4, dtype=float)
    cam_pose[:3, 3] = camera.global_transform.to_np()[:3, 3]

    seg = rt.create_segmentation_mask(
        HomogeneousTransformationMatrix(cam_pose, reference_frame=world.root),
        resolution=256,
        min_distance=0.2,
    )
    indices = np.unique(seg)
    indices = indices[indices > -1]
    bodies = [world.kinematic_structure[i] for i in indices]

    return bodies'''