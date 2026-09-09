"""
Grounding a checked triplet in a world.

A triplet names an action and the object types it is performed on and against (see
:mod:`llm_planner.env_planner.triplet_generator`); an action plan is built from concrete
arguments instead - the body to pick up, the arm to use, the pose to place at. This
module states which arguments a triplet's action needs, and resolves the triplet's
objects into them.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields

from typing_extensions import Any, Dict, List

from coraplex.robot_plans.actions.base import ActionDescription
from llm_planner.env_planner.triplet_generator import Triplet
from coraplex.datastructures.enums import Arms
from semantic_digital_twin.world import World

from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
)


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
class TripletGrounding:
    """
    Turns a checked triplet into the arguments its action plan is built from.
    """

    world: World
    """
    The world the triplet's object types are resolved in.
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
            ActionParameter(name=field.name, type_name=str(field.type))
            for field in fields(action)
            if field.default is MISSING and field.default_factory is MISSING
        ]
    
    def get_object_bodies_by_type(self,object_type: type[SemanticAnnotation],world: World)-> list[Body]:
        return world.get_semantic_annotations_by_type(object_type)

    def get_object_designator(self, object_sa: SemanticAnnotation) -> Dict[str, Any]:
        return object_sa['root']

    def get_arm(self):
        'Select suitable reachable arm which is near to the given object. use predicates.py from semantic_digital_twin '
        return

    def get_target_location(self,source_object:Body,target_object:Body):

        return
