"""
The goal state of a task (see :mod:`llm_planner.env_reasoner.reasoner_graphs.goal_state`) as the
``AND``/``OR`` tree the viewer's Goal tab draws, with all its text written on the
nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_planner.env_reasoner.reasoner_graphs.goal_state import GoalDefinition, GoalExpression


@dataclass(frozen=True)
class GoalNodeEntry:
    """
    One expression of the goal as the viewer draws it.
    """

    identifier: str
    """
    Identifier of the node, unique within the tree.
    """

    caption: str
    """
    Everything the expression says, one line each, written on the node.
    """

    group: str
    """
    Colour group of the node, the kind of its expression.
    """

    def to_payload(self) -> Dict[str, str]:
        """
        The node in the JSON shape the panel reads.
        """
        return {"id": self.identifier, "label": self.caption, "group": self.group}


@dataclass(frozen=True)
class GoalOperandEntry:
    """
    The arrow from a connective to one of its operands.
    """

    connective: str
    """
    Identifier of the connective's node.
    """

    operand: str
    """
    Identifier of the operand's node.
    """

    def to_payload(self) -> Dict[str, str]:
        """
        The arrow in the JSON shape the panel reads.
        """
        return {"from": self.connective, "to": self.operand}


@dataclass
class GoalTreeSnapshot:
    """
    The whole goal at one moment.
    """

    goal: str = ""
    """
    The goal in PDDL; it determines the whole tree, so it also serves as the signature
    the viewer rebuilds the drawing on.
    """

    nodes: List[GoalNodeEntry] = field(default_factory=list)
    """
    Every expression of the goal, each before its operands.
    """

    operands: List[GoalOperandEntry] = field(default_factory=list)
    """
    Every arrow from a connective to an operand, connectives in the order of
    :attr:`nodes`.
    """

    @classmethod
    def from_goal(cls, goal: GoalDefinition) -> GoalTreeSnapshot:
        """
        :param goal: The goal to serialize.
        :return: The snapshot of ``goal``.
        """
        snapshot = cls(goal=goal.to_pddl())
        snapshot._add(goal.condition)
        return snapshot

    def _add(self, expression: GoalExpression) -> None:
        """
        Add ``expression`` and everything below it.

        :param expression: The expression to add.
        """
        identifier = str(len(self.nodes))
        self.nodes.append(
            GoalNodeEntry(
                identifier=identifier,
                caption=expression.caption(),
                group=expression.kind,
            )
        )
        for child in expression.children():
            self.operands.append(
                GoalOperandEntry(connective=identifier, operand=str(len(self.nodes)))
            )
            self._add(child)

    def to_payload(self) -> Dict[str, Any]:
        """
        The snapshot in the JSON shape the panel reads.
        """
        return {
            "signature": self.goal,
            "goal": self.goal,
            "nodes": [node.to_payload() for node in self.nodes],
            "edges": [operand.to_payload() for operand in self.operands],
        }
