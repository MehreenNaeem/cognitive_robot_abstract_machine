"""
The interaction graph of a task: the objects selected for it, and the spatial relations
between them (see :mod:`llm_planner.env_reasoner.reasoner_graphs.Interactive_graph`), in the shape the
viewer's Interaction tab draws it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List, TYPE_CHECKING

from cramera.body_geometry import POSE_PRECISION

if TYPE_CHECKING:
    from llm_planner.env_reasoner.reasoner_graphs.Interactive_graph import (
        InteractionGraph,
    )

INTERACTION_OBJECT_GROUP = "object"
"""
Colour group every object of the graph is drawn in.
"""


@dataclass(frozen=True)
class InteractionObjectEntry:
    """
    One object of the graph as the viewer draws it.
    """

    name: str
    """
    Name of the object's root body, unique within the graph.
    """

    lines: List[str]
    """
    The object's type, affordances, properties and actions, one line each.
    """

    position: List[float]
    """
    Where the object stands in the world, as ``[x, y, z]``.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The object in the JSON shape the panel reads.
        """
        return {
            "id": self.name,
            "label": self.name,
            "group": INTERACTION_OBJECT_GROUP,
            "lines": list(self.lines),
            "position": list(self.position),
        }


@dataclass(frozen=True)
class InteractionRelationEntry:
    """
    One spatial relation of the graph as the viewer draws it.
    """

    source: str
    """
    Name of the object the relation starts at.
    """

    target: str
    """
    Name of the object the relation ends at.
    """

    relation: str
    """
    Name of the relation, e.g. ``Support``.
    """

    def to_payload(self) -> Dict[str, str]:
        """
        The relation in the JSON shape the panel reads.
        """
        return {"from": self.source, "to": self.target, "label": self.relation}


@dataclass
class InteractionGraphSnapshot:
    """
    The whole graph at one moment.
    """

    objects: List[InteractionObjectEntry] = field(default_factory=list)
    """
    Every object of the graph, in the order it was added.
    """

    relations: List[InteractionRelationEntry] = field(default_factory=list)
    """
    Every relation of the graph, in the order it was added.
    """

    signature: str = ""
    """
    Signature of the graph's objects and relations, so the viewer only rebuilds the
    drawing when the graph itself changed. Object positions are left out, so moving
    an object does not rebuild it.
    """

    @classmethod
    def from_graph(cls, graph: InteractionGraph) -> InteractionGraphSnapshot:
        """
        :param graph: The graph to serialize.
        :return: The snapshot of ``graph``.
        """
        objects = [
            InteractionObjectEntry(
                name=node.name,
                lines=node.description(),
                position=[
                    round(float(value), POSE_PRECISION)
                    for value in node.position().to_np()[:3]
                ],
            )
            for node in graph.nodes
        ]
        relations = [
            InteractionRelationEntry(
                source=edge.source.name,
                target=edge.target.name,
                relation=type(edge.relation).__name__,
            )
            for edge in graph.edges
        ]
        signature = "|".join(
            [entry.name for entry in objects]
            + [
                "%s>%s>%s" % (entry.source, entry.target, entry.relation)
                for entry in relations
            ]
        )
        return cls(objects=objects, relations=relations, signature=signature)

    def to_payload(self) -> Dict[str, Any]:
        """
        The snapshot in the JSON shape the panel reads.
        """
        return {
            "signature": self.signature,
            "nodes": [entry.to_payload() for entry in self.objects],
            "edges": [entry.to_payload() for entry in self.relations],
        }
