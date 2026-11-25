"""
A graph of the objects in a scene: each node says what an object affords, which
properties it has and which actions it can take part in, and each edge is a spatial
relation between two objects.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field

import rustworkx
from krrood.rustworkx_utils.visualization.visnetwork_graph_visualizer import (
    VisEdge,
    VisNetworkGraphVisualizer,
    VisNode,
)

from llm_planner.env_reasoner.object_action_affordance import (
    doable_actions_for,
    get_object_action_list,
)
from llm_planner.env_reasoner.object_constraints import (
    ObjectAffordance,
    ObjectAffordanceReasoner,
)
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.reasoning.predicates import (
    InsideOf,
    compute_euclidean_planar_distance,
    contact,
    is_supported_by,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

logger = logging.getLogger(__name__)

# %% exceptions


class DuplicateObjectName(Exception):
    """
    Raised when an object is added under a name the graph already holds.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"The interaction graph already has an object named {name!r}.")
        self.name = name
        """
        The name that is already taken.
        """


class UnknownObjectName(Exception):
    """
    Raised when a name does not match any object in the graph.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"The interaction graph has no object named {name!r}.")
        self.name = name
        """
        The name that was requested.
        """


class UnannotatedBody(Exception):
    """
    Raised when a body is the root of no semantic annotation in its world.
    """

    def __init__(self, body: Body) -> None:
        super().__init__(f"No semantic annotation has body {body.name} as its root.")
        self.body = body
        """
        The body without annotation.
        """


class AmbiguousBodyAnnotation(Exception):
    """
    Raised when a body is the root of more than one semantic annotation in its world.
    """

    def __init__(self, body: Body, annotations: list[HasRootBody]) -> None:
        names = ", ".join(type(annotation).__name__ for annotation in annotations)
        super().__init__(
            f"Body {body.name} is the root of several semantic annotations: {names}."
        )
        self.body = body
        """
        The body with several annotations.
        """
        self.annotations = annotations
        """
        Every semantic annotation that has :attr:`body` as its root.
        """


# %% nodes


@dataclass
class InteractionNode:
    """
    One object in the scene, with what can be done with it.
    """

    name: str
    """
    Unique name of the object within its graph, e.g. the name of its root body.
    """

    annotation_type: type[SemanticAnnotation]
    """
    The semantic annotation class the object is an instance of.
    """

    affordances: frozenset[ObjectAffordance]
    """
    The manipulation affordances the object provides.
    """

    properties: list[str]
    """
    Names of the object's structural properties, e.g. ``HasDoors``.
    """

    actions: set[str]
    """
    Class names of the actions the object can take part in.
    """

    annotation: HasRootBody
    """
    The world annotation this node describes.
    """

    @classmethod
    def from_annotation(cls, annotation: HasRootBody) -> InteractionNode:
        """
        Describe an object of the world, named after its root body.

        :param annotation: The world annotation of the object, e.g. a
            :class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.CounterTop`.
        :return: The node describing the object.
        """
        annotation_type = type(annotation)
        return cls(
            name=annotation.root.name.name,
            annotation_type=annotation_type,
            affordances=ObjectAffordanceReasoner().affordances_for(annotation_type),
            properties=sorted(
                mixin.__name__ for mixin in doable_actions_for(annotation_type).mixins
            ),
            actions=get_object_action_list(annotation_type),
            annotation=annotation,
        )

    def position(self) -> Point3:
        """
        :return: Where the object stands in the world: the centre of its root body's
            collision geometry, or the body's origin when it has none.
        """
        body = self.annotation.root
        if not body.has_collision():
            return body.global_pose.to_position()
        centre = (
            body.collision.as_bounding_box_collection_in_frame(body)
            .bounding_box()
            .center
        )
        return body.global_pose.to_homogeneous_matrix() @ centre

    def description(self) -> list[str]:
        """
        :return: The object's type, affordances, properties and actions, one line each.
        """
        return [
            f"Type: {self.annotation_type.__name__}",
            "Affordances: "
            + ", ".join(sorted(affordance.value for affordance in self.affordances)),
            "Properties: " + ", ".join(self.properties),
            "Actions: " + ", ".join(sorted(self.actions)),
        ]


# %% edges


class SpatialRelation(ABC):
    """
    A directed spatial relation that can hold from one object to another.
    """

    @abstractmethod
    def holds(self, source: InteractionNode, target: InteractionNode) -> bool:
        """
        Decide whether the relation holds from ``source`` to ``target``.

        :param source: The object the edge starts at.
        :param target: The object the edge ends at.
        :return: Whether the relation holds.
        """


@dataclass
class Support(SpatialRelation):
    """
    ``source`` supports ``target`` from below, e.g. a counter top holding a milk,
    judged by the collision geometry of their root bodies with
    :func:`~semantic_digital_twin.reasoning.predicates.is_supported_by`.
    """

    def holds(self, source: InteractionNode, target: InteractionNode) -> bool:
        return is_supported_by(target.annotation.root, source.annotation.root)


MINIMUM_CONTAINMENT_RATIO = 0.9
"""
Default fraction of an object that has to lie inside another for :class:`Containment`
to hold, the same cut-off

"""


@dataclass
class Containment(SpatialRelation):
    """
    ``source`` contains ``target``, e.g. a drawer holding a spoon, judged by the share
    of ``target``'s root body inside ``source``'s with
    :class:`~semantic_digital_twin.reasoning.predicates.InsideOf`.
    """

    minimum_ratio: float = MINIMUM_CONTAINMENT_RATIO
    """
    Fraction of ``target``, from 0 to 1, that has to lie inside ``source``'s bounding
    box.
    """

    def holds(self, source: InteractionNode, target: InteractionNode) -> bool:
        ratio = InsideOf(target.annotation.root, source.annotation.root)()
        return ratio >= self.minimum_ratio


@dataclass
class Contact(SpatialRelation):
    """
    ``source`` touches ``target``, judged by the collision geometry of their root
    bodies with :func:`~semantic_digital_twin.reasoning.predicates.contact`.
    """

    def holds(self, source: InteractionNode, target: InteractionNode) -> bool:
        return contact(source.annotation.root, target.annotation.root)


DEFAULT_NEAR_THRESHOLD = 1.0
"""
Default largest horizontal distance, in metres, at which :class:`Near` holds.
"""


@dataclass
class Near(SpatialRelation):
    """
    ``source`` is close to ``target`` in the horizontal plane, judged by the distance
    between their root bodies with
    :func:`~semantic_digital_twin.reasoning.predicates.compute_euclidean_planar_distance`,
    ignoring height.
    """

    threshold: float = DEFAULT_NEAR_THRESHOLD
    """
    Largest horizontal distance, in metres, at which the two objects count as near.
    """

    def holds(self, source: InteractionNode, target: InteractionNode) -> bool:
        distance = compute_euclidean_planar_distance(
            source.annotation.root, target.annotation.root, Vector3.Z()
        )
        return float(distance) <= self.threshold


@dataclass
class InteractionEdge:
    """
    A spatial relation holding from one object to another.
    """

    source: InteractionNode
    """
    The object the relation starts at.
    """

    target: InteractionNode
    """
    The object the relation ends at.
    """

    relation: SpatialRelation
    """
    The relation that holds from :attr:`source` to :attr:`target`.
    """


def default_relations(
    near_threshold: float = DEFAULT_NEAR_THRESHOLD,
) -> list[SpatialRelation]:
    """
    :param near_threshold: The :attr:`Near.threshold` to use.
    :return: One instance of every spatial relation.
    """
    return [Support(), Containment(), Contact(), Near(threshold=near_threshold)]


# %% edge removal rules


class EdgeRemovalRule(ABC):
    """
    A rule deciding which relation edges contradict others and have to be removed.
    """

    @abstractmethod
    def removes(self, edge: InteractionEdge, graph: InteractionGraph) -> bool:
        """
        Decide whether ``edge`` has to be removed from ``graph``.

        :param edge: An edge of ``graph``.
        :param graph: The graph ``edge`` belongs to, as it is before any removal.
        :return: Whether ``edge`` is removed.
        """


@dataclass
class ContainmentExcludesSupport(EdgeRemovalRule):
    """
    No :class:`Support` edge, in either direction, between two objects when one
    contains the other.
    """

    def removes(self, edge: InteractionEdge, graph: InteractionGraph) -> bool:
        if not isinstance(edge.relation, Support):
            return False
        relations = graph.relations_between(
            edge.source.name, edge.target.name
        ) + graph.relations_between(edge.target.name, edge.source.name)
        return any(isinstance(relation, Containment) for relation in relations)


def default_edge_removal_rules() -> list[EdgeRemovalRule]:
    """
    :return: One instance of every edge removal rule.
    """
    return [ContainmentExcludesSupport()]


# %% visualization

NODE_COLOR = "#fbbf3a"
"""
Fill colour of the circle drawn for each object.
"""

NODE_BORDER_COLOR = "#e0a01b"
"""
Border colour of the circle drawn for each object.
"""

NODE_FONT_COLOR = "#222222"
"""
Colour of the object name written inside its circle.
"""

EDGE_LENGTH = 300
"""
Resting length, in pixels, of each arrow, keeping objects far enough apart for the
captions to be read.
"""

EDGE_FONT_COLOR = "#dddddd"
"""
Colour of the relation caption written along each arrow.
"""

EDGE_ROUNDNESS_STEP = 0.3
"""
How much more each further edge between the same two objects bends, so that they are
drawn apart instead of on top of each other.
"""


@dataclass
class InteractionGraphVisualizer(VisNetworkGraphVisualizer):
    """
    An interactive page drawing objects as named circles and relations as arrows
    captioned with the relation; clicking an object shows its
    :meth:`InteractionNode.description`.
    """

    def graph_nodes(self) -> list[VisNode]:
        """
        :return: One named circle per object.
        """
        return [
            {**node, "shape": "circle", "font": {"color": NODE_FONT_COLOR}}
            for node in super().graph_nodes()
        ]

    def graph_edges(self) -> list[VisEdge]:
        """
        :return: One arrow per relation edge, captioned along the arrow with the
            relation's name.
        """
        edges_between_pair: dict[frozenset[int], int] = {}
        edges = []
        for index, (source, target, relation) in self.graph.edge_index_map().items():
            pair = frozenset((source, target))
            position = edges_between_pair.get(pair, 0)
            edges_between_pair[pair] = position + 1
            edges.append(
                {
                    "id": str(index),
                    "from": str(source),
                    "to": str(target),
                    "label": type(relation).__name__,
                    "length": EDGE_LENGTH,
                    "font": {
                        "align": "middle",
                        "color": EDGE_FONT_COLOR,
                        "strokeWidth": 0,
                    },
                    "smooth": {
                        "type": "curvedCW",
                        "roundness": position * EDGE_ROUNDNESS_STEP,
                    },
                }
            )
        return edges


# %% graph


@dataclass
class InteractionGraph:
    """
    Objects of a scene as nodes, connected by the spatial relations between them.
    """

    relations: list[SpatialRelation]
    """
    Relations :meth:`add_relation_edges` checks between every ordered pair of objects,
    e.g. :func:`default_relations`.
    """

    _graph: rustworkx.PyDiGraph = field(
        default_factory=rustworkx.PyDiGraph, init=False, repr=False
    )
    """
    The directed multigraph holding :class:`InteractionNode` payloads on its nodes and
    :class:`SpatialRelation` payloads on its edges.
    """

    _index_by_name: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    """
    Index of each node in :attr:`_graph`, keyed by :attr:`InteractionNode.name`.
    """

    @classmethod
    def from_world(
        cls,
        world: World,
        bodies: Iterable[Body],
        relations: list[SpatialRelation],
    ) -> InteractionGraph:
        """
        Build a graph with one node per body, described by the semantic annotation the
        body is the root of.

        :param world: The world the bodies and their annotations belong to.
        :param bodies: The bodies of the objects, e.g.
            ``[world.get_body_by_name("milk.stl")]``; a repeated body is added once.
        :param relations: Relations to check, e.g. :func:`default_relations`.
        :return: The graph, without edges.
        :raises UnannotatedBody: If a body is the root of no annotation.
        :raises AmbiguousBodyAnnotation: If a body is the root of several annotations.
        """
        graph = cls(relations=relations)
        annotated = world.get_semantic_annotations_by_type(HasRootBody)
        for body in dict.fromkeys(bodies):
            graph.add_annotation(cls.annotation_of(body, annotated))
        return graph

    @classmethod
    def from_annotated_bodies(
        cls,
        world: World,
        bodies: Iterable[Body],
        relations: list[SpatialRelation],
    ) -> InteractionGraph:
        """
        Build a graph like :meth:`from_world`, leaving out every body that is the root
        of no annotation.

        :param world: The world the bodies and their annotations belong to.
        :param bodies: The bodies of the objects; a repeated body is added once.
        :param relations: Relations to check, e.g. :func:`default_relations`.
        :return: The graph, without edges.
        :raises AmbiguousBodyAnnotation: If a body is the root of several annotations.
        """
        annotated_roots = {
            annotation.root
            for annotation in world.get_semantic_annotations_by_type(HasRootBody)
        }
        annotated_bodies = []
        for body in bodies:
            if body not in annotated_roots:
                logger.info(
                    "Leaving %s out of the graph: it has no annotation.", body.name
                )
                continue
            annotated_bodies.append(body)
        return cls.from_world(world, annotated_bodies, relations)

    @classmethod
    def from_selected_bodies(
        cls,
        world: World,
        bodies: Iterable[Body],
        relations: list[SpatialRelation],
        removal_rules: Iterable[EdgeRemovalRule],
    ) -> InteractionGraph:
        """
        Build the graph of the annotated ``bodies``, connected by every relation that
        holds between them and that none of ``removal_rules`` removes.

        :param world: The world the bodies and their annotations belong to.
        :param bodies: The bodies of the objects; one without annotation is left out.
        :param relations: Relations to check, e.g. :func:`default_relations`.
        :param removal_rules: Rules removing contradicting edges, e.g.
            :func:`default_edge_removal_rules`.
        :return: The graph, with its edges.
        :raises AmbiguousBodyAnnotation: If a body is the root of several annotations.
        """
        graph = cls.from_annotated_bodies(world, bodies, relations)
        graph.add_relation_edges()
        graph.remove_edges(removal_rules)
        return graph

    @staticmethod
    def annotation_of(body: Body, annotations: list[HasRootBody]) -> HasRootBody:
        """
        :param body: The body whose annotation is looked up.
        :param annotations: The annotations to search.
        :return: The one annotation in ``annotations`` that has ``body`` as its root.
        :raises UnannotatedBody: If no annotation has ``body`` as its root.
        :raises AmbiguousBodyAnnotation: If several annotations have ``body`` as
            their root.
        """
        matches = [annotation for annotation in annotations if annotation.root is body]
        if not matches:
            raise UnannotatedBody(body)
        if len(matches) > 1:
            raise AmbiguousBodyAnnotation(body, matches)
        return matches[0]

    @property
    def nodes(self) -> list[InteractionNode]:
        """
        :return: Every object in the graph, in the order it was added.
        """
        return list(self._graph.nodes())

    @property
    def edges(self) -> list[InteractionEdge]:
        """
        :return: Every relation edge in the graph, in the order it was added.
        """
        return [edge for _, edge in self._indexed_edges()]

    def node(self, name: str) -> InteractionNode:
        """
        :param name: The name of an object in the graph.
        :return: The node of that object.
        :raises UnknownObjectName: If no object has that name.
        """
        return self._graph[self._index_of(name)]

    def add_node(self, node: InteractionNode) -> InteractionNode:
        """
        Add an object to the graph.

        :param node: The object to add.
        :return: ``node``.
        :raises DuplicateObjectName: If an object with the same name is in the graph.
        """
        if node.name in self._index_by_name:
            raise DuplicateObjectName(node.name)
        self._index_by_name[node.name] = self._graph.add_node(node)
        return node

    def add_annotation(self, annotation: HasRootBody) -> InteractionNode:
        """
        Add a world object, named after its root body.

        :param annotation: The world annotation of the object.
        :return: The node added for it.
        """
        return self.add_node(InteractionNode.from_annotation(annotation))

    def add_edge(
        self, source_name: str, target_name: str, relation: SpatialRelation
    ) -> None:
        """
        Record that ``relation`` holds from one object to another.

        :param source_name: Name of the object the relation starts at.
        :param target_name: Name of the object the relation ends at.
        :param relation: The relation that holds.
        :raises UnknownObjectName: If either name matches no object.
        """
        self._graph.add_edge(
            self._index_of(source_name), self._index_of(target_name), relation
        )

    def add_relation_edges(self) -> None:
        """
        Add an edge for every relation in :attr:`relations` that holds between an
        ordered pair of distinct objects.
        """
        for source in self.nodes:
            for target in self.nodes:
                if source is target:
                    continue
                for relation in self.relations:
                    if relation.holds(source, target):
                        self.add_edge(source.name, target.name, relation)

    def remove_edges(self, rules: Iterable[EdgeRemovalRule]) -> list[InteractionEdge]:
        """
        Remove every edge that one of ``rules`` removes, each rule judging the graph as
        it is before any removal.

        :param rules: The rules to apply, e.g. :func:`default_edge_removal_rules`.
        :return: The removed edges, in the order they were added.
        """
        rules = list(rules)
        removed = [
            (index, edge)
            for index, edge in self._indexed_edges()
            if any(rule.removes(edge, self) for rule in rules)
        ]
        for index, _ in removed:
            self._graph.remove_edge_from_index(index)
        return [edge for _, edge in removed]

    def relations_between(
        self, source_name: str, target_name: str
    ) -> list[SpatialRelation]:
        """
        :param source_name: Name of the object the relations start at.
        :param target_name: Name of the object the relations end at.
        :return: Every relation recorded from the one object to the other.
        :raises UnknownObjectName: If either name matches no object.
        """
        source_index = self._index_of(source_name)
        target_index = self._index_of(target_name)
        return [
            relation
            for _, target, relation in self._graph.out_edges(source_index)
            if target == target_index
        ]

    def visualizer(self, open_browser: bool = True) -> InteractionGraphVisualizer:
        """
        :param open_browser: Whether :meth:`InteractionGraphVisualizer.run` opens the
            page in a web browser.
        :return: A visualizer of this graph, not yet running.
        """
        return InteractionGraphVisualizer(
            graph=self._graph,
            information_getter=InteractionNode.description,
            label_getter=lambda node: node.name,
            default_node_color=NODE_COLOR,
            default_border_color=NODE_BORDER_COLOR,
            title="Interaction graph",
            open_browser=open_browser,
        )

    def visualize(self, open_browser: bool = True) -> InteractionGraphVisualizer:
        """
        Serve an interactive page of the graph: click an object to see its
        affordances, properties and actions.

        The page is served from a background thread, so it stays reachable only while
        the calling process keeps running.

        :param open_browser: Whether to open the page in a web browser.
        :return: The running visualizer.
        """
        visualizer = self.visualizer(open_browser)
        visualizer.run()
        return visualizer

    def show_in(self, visualizer: InteractionGraphVisualizer) -> None:
        """
        Make ``visualizer`` draw this graph instead of the one it draws now; a running
        page shows it on its next refresh.

        :param visualizer: The visualizer to draw this graph.
        """
        visualizer.graph = self._graph

    def _indexed_edges(self) -> list[tuple[int, InteractionEdge]]:
        """
        :return: Every edge with its index in :attr:`_graph`, in the order it was added.
        """
        indexed_edges = []
        for index in self._graph.edge_indices():
            source, target = self._graph.get_edge_endpoints_by_index(index)
            edge = InteractionEdge(
                self._graph[source],
                self._graph[target],
                self._graph.get_edge_data_by_index(index),
            )
            indexed_edges.append((index, edge))
        return indexed_edges

    def _index_of(self, name: str) -> int:
        """
        :param name: The name of an object in the graph.
        :return: The index of its node in :attr:`_graph`.
        :raises UnknownObjectName: If no object has that name.
        """
        if name not in self._index_by_name:
            raise UnknownObjectName(name)
        return self._index_by_name[name]


"""
# %% usage example
if __name__ == "__main__":
    from coraplex.testing import setup_world
    from semantic_digital_twin.semantic_annotations.semantic_annotations import Spoon

    world = setup_world()
    spoon = STLParser(
        os.path.join(
            "/home/mnaeem/cognitive_robot_abstract_machine/coraplex/resources/objects/",
            "spoon.stl",
        )
    ).parse()
    with world.modify_world():
        connection = FixedConnection(
            parent=world.get_body_by_name("cabinet10_drawer_top"),
            child=spoon.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.05, -0.05, 0.01
            ),
        )
        world.merge_world(spoon, connection)

    spoon_body = world.get_body_by_name("spoon.stl")
    drawer_top = world.get_body_by_name("cabinet10_drawer_top")
    counter_top = world.get_body_by_name("island_countertop")
    milk = world.get_body_by_name("milk.stl")
    with world.modify_world():
        world_reasoner = WorldReasoner(world)
        world_reasoner.reason()
        world.add_semantic_annotation(Spoon(root=spoon_body))

    graph = InteractionGraph.from_world(
        world, [spoon_body, drawer_top, counter_top, milk], default_relations()
    )
    graph.add_relation_edges()
    graph.remove_edges(default_edge_removal_rules())

    for edge in graph.edges:
        print(
            f"{edge.source.name} -{type(edge.relation).__name__}-> {edge.target.name}"
        )

    graph.visualize()
    input("Showing the graph in the browser; press Enter to stop.\n")
"""
