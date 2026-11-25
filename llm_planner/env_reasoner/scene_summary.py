"""
Saying in plain words what the robot does for one grounded action, and why a condition
refused it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from typing_extensions import List, Optional

from coraplex.datastructures.enums import VerticalAlignment
from krrood.entity_query_language.operators.core_logical_operators import AND, OR, Not
from llm_planner.env_planner.action_validation import (
    ConditionCheck,
    ConditionStatement,
)
from llm_planner.env_planner.execution_outcome import ExecutionOutcome
from llm_planner.env_planner.triplet_grounding import GroundedTriplet

ACTION_SUFFIX = "Action"
"""
The ending every action class name carries, dropped from the verb it reads as.
"""

REPAIR_MARKER = " (repair)"
"""
What marks an action tried to repair a failure, apart from the actions of the plan.
"""

INDENT = "    "
"""
One level of indentation under the sentence an action reads as.
"""

LOGICAL_OPERATORS = (AND, OR, Not)
"""
The operators combining statements, whose operands are read instead of themselves.
"""


def words_of(name: str) -> str:
    """
    An identifier read as the words it is written from, e.g. ``PickUpAction`` as
    "pick up action".

    :param name: The identifier to read.
    :return: Its words, lower case, separated by spaces.
    """
    return " ".join(re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", name)).lower()


# %% condition statements in words


def interpretation_CS(statement: ConditionStatement) -> List[str]:
    """
    The predicates a statement combines, each as its name and whether it holds, e.g.
    ``"is_body_gripped:True"``. The predicates inside an ``AND``, ``OR`` or ``Not``
    operand are read one by one; the operators themselves are left out.

    :param statement: The statement to read.
    :return: One reading per predicate, in the order the condition names them. Empty
        for a statement without operands.
    """
    reason = []
    for operand in statement.operands:
        if isinstance(operand.expression, LOGICAL_OPERATORS):
            reason.extend(interpretation_CS(operand))
            continue
        reason.append(f"{operand.name}:{operand.satisfied}")
    return reason


# %% one action in words


@dataclass
class ActionSummary:
    """
    What the robot does for one grounded action, in one sentence, and which statement
    of a condition refused it where one did.

    Reads the action, the bodies, the arm and the grasp the grounding resolved, so a
    plan can be followed in plain words rather than in designators.
    """

    grounded: GroundedTriplet
    """
    The action to describe, with the things it is performed with.
    """

    outcome: ExecutionOutcome
    """
    How far the action got.
    """

    failed_check: Optional[ConditionCheck] = None
    """
    The condition that refused the action, or None where none did.
    """

    is_repair: bool = False
    """
    Whether the action was tried to repair a failure rather than taken from the plan.
    """

    @property
    def description(self) -> str:
        """
        :return: The sentence the action reads as, followed by the condition that
            refused it where one did.
        """
        return (
            f"{self.action_phrase}{self.target_phrase}{self.arm_phrase}"
            f"{self.grasp_phrase}{self.repair_phrase}: {self.outcome}"
            f"{self.failure_phrase}"
        )

    @property
    def action_phrase(self) -> str:
        """
        :return: What is done to what, e.g. "Pick up milk.stl".
        """
        verb = words_of(self.grounded.action.__name__.removesuffix(ACTION_SUFFIX))
        return f"{verb.capitalize()} {self.grounded.source_object_body.name.name}"

    @property
    def target_phrase(self) -> str:
        """
        :return: Where the object is put down, e.g. " on cabinet10_drawer_top at
            (2.40, 2.20, 1.00)". Empty for an action that puts nothing down.
        """
        if self.grounded.target_location is None:
            return ""
        place = self.grounded.target_location
        on_body = (
            ""
            if self.grounded.target_object_body is None
            else f" on {self.grounded.target_object_body.name.name}"
        )
        return (
            f"{on_body} at ({float(place.x):.2f}, {float(place.y):.2f}, "
            f"{float(place.z):.2f})"
        )

    @property
    def arm_phrase(self) -> str:
        """
        :return: The arm the action is performed with, e.g. " with the left arm".
        """
        return f" with the {words_of(self.grounded.arm.name)} arm"

    @property
    def grasp_phrase(self) -> str:
        """
        :return: How the end effector comes at the object, e.g. ", approaching from the
            front, aligned with the top". Empty for an action that takes hold of
            nothing.
        """
        grasp = self.grounded.grasp_description
        if grasp is None:
            return ""
        approach = f", approaching from the {words_of(grasp.approach_direction.name)}"
        if grasp.vertical_alignment is VerticalAlignment.NoAlignment:
            return approach
        return f"{approach}, aligned with the {words_of(grasp.vertical_alignment.name)}"

    @property
    def repair_phrase(self) -> str:
        """
        :return: :data:`REPAIR_MARKER` for an action tried to repair a failure, else
            empty.
        """
        return REPAIR_MARKER if self.is_repair else ""

    @property
    def failure_phrase(self) -> str:
        """
        :return: The kind of condition that refused the action and, one line each, the
            statements it failed on, indented under the sentence. Empty where no
            condition refused the action.
        """
        if self.failed_check is None:
            return ""
        lines: List[str] = [
            f"{INDENT}Condition_type: {self.failed_check.kind.name}",
            f"{INDENT}Statements:",
        ]
        if self.failed_check.statement is not None:
            lines.extend(
                f"{INDENT * 2}{reason}"
                for reason in interpretation_CS(self.failed_check.statement)
            )
        return "\n" + "\n".join(lines)

    def __str__(self) -> str:
        return self.description
