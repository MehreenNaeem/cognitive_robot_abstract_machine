"""
How far performing one predicted triplet got.
"""

from __future__ import annotations

from enum import StrEnum


class ExecutionOutcome(StrEnum):
    """
    How far one triplet got.
    """

    RUNNING = "running"
    """
    The triplet was grounded and its steps are being performed.
    """

    PERFORMED = "performed"
    """
    Every step ran and every condition around it held.
    """

    NOT_PLANNED = "not planned"
    """
    The grounding produced no action to perform, e.g. nothing the object still fits on.
    """

    CONDITION_FAILED = "condition failed"
    """
    A step's pre- or post-condition did not hold, so the triplet was cut short.
    """

    REPAIRED = "repaired"
    """
    A step's condition failed, and the repair actions brought about what that step was
    meant to, so the triplet was not run again.
    """
