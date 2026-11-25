"""
The LLM planner's tree: a task, the actions a language model grounded it into, and the
plan steps each of those became.

A task statement reaches the robot in three steps - the task, the triplets a language
model proposed for it (see :mod:`llm_planner.env_planner.triplet_generator`), and the
plan each triplet is performed as - and this is that chain in the shape the viewer draws
it. The statuses come from the plans themselves, so a step colours as it runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from coraplex.plans.plan import Plan
    from cramera.live.bridge import TaskStatusName

TASK_NODE_ID = "llm_task"
"""
Id of the tree's root, the task the language model was given.
"""

TASK_GROUP = "task"
"""
Colour group of the root node, which is a task rather than a plan node.
"""


@dataclass
class GroundedStep:
    """
    One triplet the task was grounded into, and the plan it is performed as.
    """

    label: str
    """
    Name of the action the triplet named, e.g. ``PickUpAction``.
    """

    plan: Plan
    """
    The plan that carries the triplet out.
    """


@dataclass(frozen=True)
class ExecutionSummaryRow:
    """
    What became of one performed triplet or repair action, as one line under the task.
    """

    label: str
    """
    The summary the execution reads as, e.g. "Open cabinet10_drawer_top with the left
    arm: performed".
    """

    status: TaskStatusName
    """
    The status the row is coloured with.
    """


@dataclass(frozen=True)
class LlmPlanNode:
    """
    One node of the tree as the viewer draws it.
    """

    id: str
    """
    Id of this node, unique within the tree.
    """

    parent: Optional[str]
    """
    Id of this node's parent, or None for the task.
    """

    label: str
    """
    What the node is called: the task statement, or an action's name.
    """

    status: str
    """
    Execution status, as the plan reports it (see
    :meth:`~cramera.live.bridge.Bridge.snapshot_plan`).
    """

    group: str
    """
    Colour group the viewer draws this node in.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The node in the JSON shape the panel reads.
        """
        return {
            "id": self.id,
            "parent": self.parent,
            "label": self.label,
            "status": self.status,
            "group": self.group,
        }


@dataclass
class LlmPlanSnapshot:
    """
    The whole tree at one moment: the task, its grounded steps and their plan steps.
    """

    task: str = ""
    """
    The task statement the language model was given.
    """

    nodes: List[LlmPlanNode] = field(default_factory=list)
    """
    Every node of the tree, flattened with parent references, task first.
    """

    signature: str = ""
    """
    Id signature of the tree's shape, so the viewer only rebuilds it when the tree
    itself changed rather than on every status update.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The snapshot in the JSON shape the panel reads.
        """
        return {
            "task": self.task,
            "signature": self.signature,
            "nodes": [node.to_payload() for node in self.nodes],
        }
