"""
The action feasibility tree of a failure repair: which actions can be performed on
which objects, with which arm and from which approach direction (see
:func:`llm_planner.env_reasoner.reasoner_graphs.AFG.feasibility_tree`), in the shape
the viewer's AFG tab draws it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import rustworkx

    from llm_planner.env_reasoner.reasoner_graphs.AFG import FeasibilityNode


@dataclass(frozen=True)
class FeasibilityNodeEntry:
    """
    One node of the tree as the viewer draws it.
    """

    identifier: str
    """
    Identifier of the node, unique within the tree.
    """

    caption: str
    """
    What the node stands for, e.g. ``PickUpAction`` or ``FRONT``.
    """

    group: str
    """
    Colour group of the node, the kind of thing it stands for.
    """

    lines: List[str]
    """
    The condition values of an approach direction, one line each, shown on click.
    """

    @classmethod
    def from_node(cls, index: int, node: FeasibilityNode) -> FeasibilityNodeEntry:
        """
        :param index: The node's index in its tree.
        :param node: The node to serialize.
        :return: The entry of ``node``.
        """
        return cls(
            identifier=str(index),
            caption=node.caption,
            group=node.kind.value,
            lines=list(node.details),
        )

    def to_payload(self) -> Dict[str, Any]:
        """
        The node in the JSON shape the panel reads.
        """
        return {
            "id": self.identifier,
            "label": self.caption,
            "group": self.group,
            "lines": list(self.lines),
        }


@dataclass(frozen=True)
class FeasibilityEdgeEntry:
    """
    The arrow from a node to one of its children.
    """

    parent: str
    """
    Identifier of the parent's node.
    """

    child: str
    """
    Identifier of the child's node.
    """

    def to_payload(self) -> Dict[str, str]:
        """
        The arrow in the JSON shape the panel reads.
        """
        return {"from": self.parent, "to": self.child}


@dataclass
class FeasibilityTreeSnapshot:
    """
    The whole tree at one moment.
    """

    nodes: List[FeasibilityNodeEntry] = field(default_factory=list)
    """
    Every node of the tree, in the order it was added.
    """

    edges: List[FeasibilityEdgeEntry] = field(default_factory=list)
    """
    Every arrow from a node to a child, in the order it was added.
    """

    signature: str = ""
    """
    Signature of the tree's nodes and arrows, so the viewer only rebuilds the drawing
    when the tree itself changed.
    """

    @classmethod
    def from_tree(cls, tree: rustworkx.PyDiGraph) -> FeasibilityTreeSnapshot:
        """
        :param tree: The tree to serialize, its nodes
            :class:`~llm_planner.env_reasoner.reasoner_graphs.AFG.FeasibilityNode`.
        :return: The snapshot of ``tree``.
        """
        nodes = [
            FeasibilityNodeEntry.from_node(index, tree[index])
            for index in tree.node_indices()
        ]
        edges = [
            FeasibilityEdgeEntry(parent=str(parent), child=str(child))
            for parent, child in tree.edge_list()
        ]
        signature = "|".join(
            [
                "%s:%s:%s" % (entry.identifier, entry.caption, ",".join(entry.lines))
                for entry in nodes
            ]
            + ["%s>%s" % (entry.parent, entry.child) for entry in edges]
        )
        return cls(nodes=nodes, edges=edges, signature=signature)

    def to_payload(self) -> Dict[str, Any]:
        """
        The snapshot in the JSON shape the panel reads.
        """
        return {
            "signature": self.signature,
            "nodes": [entry.to_payload() for entry in self.nodes],
            "edges": [entry.to_payload() for entry in self.edges],
        }
