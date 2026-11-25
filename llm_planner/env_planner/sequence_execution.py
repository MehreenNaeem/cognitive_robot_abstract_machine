"""
What became of performed actions: one record per triplet or repair action, and the
record of a whole sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Optional, TYPE_CHECKING

from llm_planner.env_planner.execution_outcome import ExecutionOutcome
from llm_planner.env_reasoner.scene_summary import ActionSummary

if TYPE_CHECKING:
    from coraplex.plans.plan_node import ActionLike
    from llm_planner.env_planner.action_validation import ConditionCheck
    from llm_planner.env_planner.triplet_generator import Triplet
    from llm_planner.env_planner.triplet_grounding import GroundedTriplet


@dataclass(frozen=True)
class TripletExecution:
    """
    What became of one triplet of the sequence, or of one action tried to repair a
    failure.
    """

    triplet: Optional[Triplet]
    """
    The triplet this reports on, or None for an action tried to repair a failure.
    """

    grounded: GroundedTriplet
    """
    The bodies it was resolved to act on.
    """

    outcome: ExecutionOutcome
    """
    How far it got.
    """

    failed_check: Optional[ConditionCheck] = None
    """
    The condition that did not hold, and the predicates that made it false. None unless
    the outcome is :attr:`ExecutionOutcome.CONDITION_FAILED`.
    """

    last_step: Optional[ActionLike] = None
    """
    The last step run: the one whose condition failed, or the final step when every
    condition held. None when nothing was planned.
    """

    @property
    def summary(self) -> ActionSummary:
        """
        :return: What became of the triplet, in plain words.
        """
        return ActionSummary(
            self.grounded,
            self.outcome,
            self.failed_check,
            is_repair=self.triplet is None,
        )


@dataclass(frozen=True)
class SequenceExecution:
    """
    What became of a whole predicted sequence, and what it reads as in plain words.
    """

    executions: List[TripletExecution] = field(default_factory=list)
    """
    What became of every triplet and repair action that was run, in execution order.
    """

    summaries: List[ActionSummary] = field(default_factory=list)
    """
    What each of those executions reads as, one per execution and in the same order.
    """

    def record(self, execution: TripletExecution) -> None:
        """
        Add an execution and its summary to the end of the record.

        :param execution: What became of the triplet or repair action just run.
        """
        self.executions.append(execution)
        self.summaries.append(execution.summary)


@dataclass(frozen=True)
class StartedAction:
    """
    A triplet or repair action that started running, and where in its run it started.
    """

    execution: TripletExecution
    """
    The action, reported as :attr:`ExecutionOutcome.RUNNING`.
    """

    recorded_before: int
    """
    How many executions the run held when the action started.
    """

    def is_in_progress(self, run: SequenceExecution) -> bool:
        """
        :param run: The run the action started in.
        :return: Whether the action is still running: nothing was recorded in ``run``
            since it started.
        """
        return len(run.executions) == self.recorded_before
