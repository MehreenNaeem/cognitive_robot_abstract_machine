"""
Helpers of the demos: loading object models named after their folder, and showing the
interaction graph of selected bodies in the CRAM viewer, both in the graph view's
Interaction tab and in the Semantic Digital Twin scene's Interaction layer.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field

from typing_extensions import List, TYPE_CHECKING

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from llm_planner.env_reasoner.reasoner_graphs.Interactive_graph import (
    EdgeRemovalRule,
    InteractionGraph,
    SpatialRelation,
    default_edge_removal_rules,
    default_relations,
)

if TYPE_CHECKING:
    from cramera.live.bridge import Bridge
    from semantic_digital_twin.world_description.world_entity import Body


# %% object models named after their folder

UNNAMED_BODY_NAME = "frame"
"""
What a body without a name in its model file is called, after the object's name.
"""


class NoObjectBody(Exception):
    """
    Raised when no body of an object model carries geometry.
    """

    def __init__(self, file_path: str) -> None:
        super().__init__(f"No body of the object model {file_path} has geometry.")
        self.file_path = file_path
        """
        The model file.
        """


class MultipleObjectBodies(Exception):
    """
    Raised when several bodies of an object model carry geometry, so none of them is
    the object.
    """

    def __init__(self, file_path: str, bodies: List[Body]) -> None:
        names = ", ".join(body.name.name for body in bodies)
        super().__init__(
            f"Several bodies of the object model {file_path} have geometry: {names}."
        )
        self.file_path = file_path
        """
        The model file.
        """
        self.bodies = bodies
        """
        Every body with geometry.
        """


@dataclass
class ObjectModelParser:
    """
    Parses an object model, a MuJoCo file in a folder of its own such as
    ``bottled_drink_1/model.xml``, into a world whose bodies are named after that
    folder: the body with the geometry is the object, e.g. ``bottled_drink_1``, and
    every other body is prefixed with it, e.g. ``bottled_drink_1_world``.

    Model files of one dataset name their bodies alike, so this keeps several objects
    apart in one world.
    """

    file_path: str
    """
    The path of the model file.
    """

    @property
    def object_name(self) -> str:
        """
        :return: The name of the object, the folder its model file lies in.
        """
        return os.path.basename(os.path.dirname(os.path.abspath(self.file_path)))

    def parse(self) -> World:
        """
        :return: The world of the object, its bodies named after the object.
        :raises NoObjectBody: If no body has geometry.
        :raises MultipleObjectBodies: If several bodies have geometry.
        """
        world = MJCFParser(file_path=self.file_path).parse()
        object_body = self.object_body(world)
        with world.modify_world():
            for index, body in enumerate(world.bodies):
                body.update_name(self.body_name(body, index, object_body))
        return world

    def object_body(self, world: World) -> Body:
        """
        :param world: The parsed world of the object.
        :return: The one body with geometry.
        :raises NoObjectBody: If no body has geometry.
        :raises MultipleObjectBodies: If several bodies have geometry.
        """
        bodies = [
            body for body in world.bodies if body.collision.shapes or body.visual.shapes
        ]
        if not bodies:
            raise NoObjectBody(self.file_path)
        if len(bodies) > 1:
            raise MultipleObjectBodies(self.file_path, bodies)
        return bodies[0]

    def body_name(self, body: Body, index: int, object_body: Body) -> PrefixedName:
        """
        :param body: A body of the object's world.
        :param index: The body's position among the world's bodies.
        :param object_body: The body with the geometry.
        :return: The object's name for ``object_body``; for any other body, the
            object's name followed by the body's own name, or by
            :data:`UNNAMED_BODY_NAME` and ``index`` when it has none.
        """
        if body is object_body:
            return PrefixedName(self.object_name)
        if not body.name.name:
            return PrefixedName(f"{self.object_name}_{UNNAMED_BODY_NAME}_{index}")
        return PrefixedName(f"{self.object_name}_{body.name.name}")


# %% the interaction graph in the viewer


@dataclass
class SceneInteractionDisplay:
    """
    Shows the selected objects of a world in the CRAM viewer: the spatial relations
    between them, and on click, each object's description.
    """

    world: World
    """
    The world the shown bodies belong to.
    """

    bridge: Bridge
    """
    The live bridge the viewer reads the graph from.
    """

    relations: List[SpatialRelation] = field(default_factory=default_relations)
    """
    The relations checked between every ordered pair of shown objects.
    """

    removal_rules: List[EdgeRemovalRule] = field(
        default_factory=default_edge_removal_rules
    )
    """
    The rules removing relation edges that contradict others.
    """

    def graph_of(self, bodies: Iterable[Body]) -> InteractionGraph:
        """
        :param bodies: The bodies of the objects to show; one without annotation is
            left out.
        :return: The graph of the objects and the relations between them.
        """
        return InteractionGraph.from_selected_bodies(
            self.world, bodies, self.relations, self.removal_rules
        )

    def show(self, bodies: Iterable[Body]) -> InteractionGraph:
        """
        Show the graph of ``bodies`` in the viewer.

        :param bodies: The bodies of the objects to show; one without annotation is
            left out.
        :return: The shown graph.
        """
        graph = self.graph_of(bodies)
        self.show_graph(graph)
        return graph

    def show_graph(self, graph: InteractionGraph) -> None:
        """
        Show ``graph`` in the viewer, replacing the one shown before.

        :param graph: The graph to show.
        """
        self.bridge.publish_interaction_graph(graph)
