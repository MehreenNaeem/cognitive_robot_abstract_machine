"""
Tests of performing a predicted triplet sequence in the apartment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from unittest.mock import patch

import pytest

from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.plans.plan_node import ActionLike
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from llm_planner.env_planner.action_validation import (
    ConditionCheck,
    check_pre_condition,
)
from llm_planner.env_planner import executor as executor_module
from llm_planner.env_planner.execution_outcome import ExecutionOutcome
from llm_planner.env_planner.executor import PlanExecutor, TripletExecution
from llm_planner.env_planner.sequence_execution import SequenceExecution
from llm_planner.env_planner.triplet_generator import Triplet
from llm_planner.env_planner.triplet_grounding import TripletGrounding
from llm_planner.env_reasoner.scene_summary import ActionSummary
from llm_planner.env_reasoner.action_constraints import ConditionKind
from llm_planner.env_reasoner.reasoner_graphs.AFG import ActionFeasibility
from llm_planner.env_reasoner.reasoner_graphs.goal_state import (
    GoalAtom,
    GoalConjunction,
    GoalDefinition,
)
from llm_planner.env_reasoner.reasoner_graphs.Interactive_graph import (
    InteractionGraph,
    default_edge_removal_rules,
    default_relations,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Milk,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import List, Optional, Type

MILK_BODY_NAME = "milk.stl"
"""
An apartment object far enough from the robot's spawn that it is not reachable.
"""

DRAWER_BODY_NAME = "cabinet10_drawer_top"
"""
The apartment drawer the fixture annotates, handle and all.
"""

GRIPPER_BODY_NAME = "l_gripper_palm_link"
"""
A body of the PR2's left gripper, where a held object is attached.
"""


def triplet_of(source, target, action) -> Triplet:
    return Triplet(source_object=source, target_object=target, action=action)


@dataclass
class StepRecorder(PlanExecutor):
    """
    A plan executor that records its steps instead of performing them, so the order a
    sequence is run in can be read without moving a robot.
    """

    performed_steps: List[ActionLike] = field(default_factory=list)
    """
    Every step handed to :meth:`run_step`, in the order it was handed over.
    """

    failing_action: Optional[Type[ActionDescription]] = None
    """
    The action to report an unsatisfied pre-condition for, or None to let every step
    succeed.
    """

    def run_step(self, step: ActionLike) -> Optional[ConditionCheck]:
        self.performed_steps.append(step)
        if self.failing_action is None or not isinstance(step, self.failing_action):
            return None
        return ConditionCheck(
            action=type(step), kind=ConditionKind.PRE, satisfied=False
        )


@dataclass
class RecordsPerformedSteps(PlanExecutor):
    """
    A plan executor that records the steps handed to its performing instead of running
    them, so what a step's conditions let through can be read without moving a robot.
    """

    performed_steps: List[ActionLike] = field(default_factory=list)
    """
    Every step handed to :meth:`PlanExecutor.perform_step`, in the order it was handed
    over.
    """

    def perform_step(self, step: ActionLike) -> None:
        self.performed_steps.append(step)


@dataclass
class ReportsPreparedExecutions(PlanExecutor):
    """
    A plan executor that reports prepared executions instead of grounding and
    performing its triplets, so what the sequence makes of them can be read on its own.
    """

    prepared: List[TripletExecution] = field(default_factory=list)
    """
    What to report for each triplet, in the order the triplets are handed over.
    """

    reported: int = 0
    """
    How many of the prepared executions have been reported already.
    """

    def run_triplet(self, triplet: Triplet) -> TripletExecution:
        execution = self.prepared[self.reported]
        self.reported += 1
        return execution


@dataclass
class RecordsDisplayedGraphs(PlanExecutor):
    """
    A plan executor that records the graphs it would display instead of serving them,
    so what is plotted can be read without a web page.
    """

    displayed_graphs: List[InteractionGraph] = field(default_factory=list)
    """
    Every graph handed to :meth:`display_graph`, in the order it was handed over.
    """

    def display_graph(self, graph: InteractionGraph) -> None:
        self.displayed_graphs.append(graph)


@dataclass
class RecordsDisplayedFeasibility(PlanExecutor):
    """
    A plan executor that records the feasible actions it would display instead of
    showing them.
    """

    displayed_feasibilities: List[ActionFeasibility] = field(default_factory=list)
    """
    Every feasibility handed to :meth:`display_action_feasibility`, in the order it was
    handed over.
    """

    def display_action_feasibility(self, feasibility: ActionFeasibility) -> None:
        self.displayed_feasibilities.append(feasibility)


@dataclass
class RecordsRecordedExecutions(PlanExecutor):
    """
    A plan executor that also lists every execution handed to :meth:`record`.
    """

    recorded: List[TripletExecution] = field(default_factory=list)
    """
    Every execution handed to :meth:`record`, in the order it was handed over.
    """

    def record(self, execution: TripletExecution) -> None:
        self.recorded.append(execution)
        super().record(execution)


@dataclass
class RecordsStepsAndGraphs(StepRecorder, RecordsDisplayedGraphs):
    """
    A plan executor that records both its steps and the graphs it would display.
    """


def execution_of(
    world,
    source,
    target,
    action,
    outcome: ExecutionOutcome = ExecutionOutcome.PERFORMED,
    failed_check: Optional[ConditionCheck] = None,
) -> TripletExecution:
    """
    :param world: The world to ground the triplet in.
    :param source: The object the action is performed on.
    :param target: The object it is performed against.
    :param action: The action to perform.
    :param outcome: How far the triplet is to have got.
    :param failed_check: The condition that refused it, where one did.
    :return: The execution to report for that triplet.
    """
    triplet = triplet_of(source, target, action)
    return TripletExecution(
        triplet=triplet,
        grounded=TripletGrounding(world=world).ground(triplet),
        outcome=outcome,
        failed_check=failed_check,
    )


def performed_actions(recorder: StepRecorder) -> List[str]:
    """
    :param recorder: The executor to read.
    :return: The name of every action among its recorded steps, in execution order.
    """
    return [
        type(step).__name__
        for step in recorder.performed_steps
        if isinstance(step, ActionDescription)
    ]


# %% extending the executor


def test_a_subclass_can_add_fields_without_defaults(
    apartment_world_pr2_copy_with_context,
) -> None:
    _, _, context = apartment_world_pr2_copy_with_context

    @dataclass
    class NamesItsTask(PlanExecutor):
        task: str

    assert NamesItsTask(context=context, task="tidy up").task == "tidy up"


# %% running a sequence


class TestRunTriplets:
    def test_the_triplets_are_performed_in_the_given_order(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        recorder = StepRecorder(context=context)

        run = recorder.run_triplets(
            [
                triplet_of(Drawer, Drawer, OpenAction),
                triplet_of(Milk, Drawer, PickUpAction),
            ]
        )

        assert [execution.outcome for execution in run.executions] == [
            ExecutionOutcome.PERFORMED,
            ExecutionOutcome.PERFORMED,
        ]
        actions = performed_actions(recorder)
        assert actions.index(OpenAction.__name__) < actions.index(PickUpAction.__name__)

    def test_a_triplet_that_cannot_be_planned_is_skipped(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        """
        A triplet the grounding could not supply every argument for - here a place to
        put the milk down - performs nothing, and the sequence carries on without it.
        """
        _, _, context = apartment_world_pr2_copy_with_context
        recorder = StepRecorder(context=context)

        run = recorder.run_triplets(
            [
                triplet_of(Milk, Drawer, PlaceAction),
                triplet_of(Drawer, Drawer, OpenAction),
            ]
        )

        assert [execution.outcome for execution in run.executions] == [
            ExecutionOutcome.NOT_PLANNED,
            ExecutionOutcome.PERFORMED,
        ]
        assert PlaceAction.__name__ not in performed_actions(recorder)

    def test_a_failed_condition_ends_the_sequence(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        recorder = StepRecorder(context=context, failing_action=OpenAction)

        run = recorder.run_triplets(
            [
                triplet_of(Drawer, Drawer, OpenAction),
                triplet_of(Milk, Drawer, PickUpAction),
            ]
        )

        assert [execution.outcome for execution in run.executions] == [
            ExecutionOutcome.CONDITION_FAILED
        ]
        assert run.executions[0].failed_check.action is OpenAction
        assert PickUpAction.__name__ not in performed_actions(recorder)

    def test_one_grounding_serves_the_whole_sequence(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        """
        The drawer a sequence opens is the drawer it closes again.
        """
        _, _, context = apartment_world_pr2_copy_with_context
        recorder = StepRecorder(context=context)

        opening, closing = recorder.run_triplets(
            [
                triplet_of(Drawer, Drawer, OpenAction),
                triplet_of(Drawer, Drawer, CloseAction),
            ]
        ).executions

        assert (
            opening.grounded.source_object_body is closing.grounded.source_object_body
        )

    def test_an_empty_sequence_performs_nothing(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        recorder = StepRecorder(context=context)

        run = recorder.run_triplets([])

        assert run.executions == []
        assert run.summaries == []
        assert recorder.performed_steps == []


# %% summarising what was performed


class TestSummaries:
    def test_an_execution_is_summarised_as_its_own_action(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, _ = apartment_world_pr2_copy_with_context
        triplet = triplet_of(Milk, Drawer, PickUpAction)
        execution = TripletExecution(
            triplet=triplet,
            grounded=TripletGrounding(world=world).ground(triplet),
            outcome=ExecutionOutcome.PERFORMED,
        )

        assert execution.summary == ActionSummary(execution.grounded, execution.outcome)

    def test_a_sequence_is_summarised_one_summary_per_execution(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        reporter = ReportsPreparedExecutions(
            context=context,
            prepared=[
                execution_of(world, Drawer, Drawer, OpenAction),
                execution_of(world, Milk, Drawer, PickUpAction),
            ],
        )

        run = reporter.run_triplets(
            [execution.triplet for execution in reporter.prepared]
        )

        assert run.summaries == [execution.summary for execution in run.executions]

    def test_a_cut_short_sequence_is_summarised_up_to_the_condition_that_failed(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        refused = execution_of(
            world,
            Drawer,
            Drawer,
            OpenAction,
            outcome=ExecutionOutcome.CONDITION_FAILED,
            failed_check=ConditionCheck(
                action=OpenAction, kind=ConditionKind.PRE, satisfied=False
            ),
        )
        reporter = ReportsPreparedExecutions(
            context=context,
            prepared=[refused, execution_of(world, Milk, Drawer, PickUpAction)],
        )

        run = reporter.run_triplets(
            [execution.triplet for execution in reporter.prepared]
        )

        assert run.summaries == [refused.summary]
        assert run.summaries[0].failed_check is refused.failed_check


# %% running one step


class TestRunStep:
    def test_a_step_outside_the_checkable_actions_is_performed_unchecked(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        step = ParkArmsAction(Arms.BOTH)
        recorder = RecordsPerformedSteps(context=context)

        assert recorder.run_step(step) is None
        assert recorder.performed_steps == [step]

    def test_a_step_whose_pre_condition_fails_is_not_performed(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        action = OpenAction(
            object_designator=world.get_body_by_name(MILK_BODY_NAME), arm=Arms.LEFT
        )
        recorder = RecordsPerformedSteps(context=context)

        recorder.run_step(action)

        assert recorder.performed_steps == []

    def test_a_step_whose_pre_condition_fails_reports_it(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        """
        The milk stands out of the robot's reach, so the action is not performed and
        the check that refused it is handed back.
        """
        world, _, context = apartment_world_pr2_copy_with_context
        action = OpenAction(
            object_designator=world.get_body_by_name(MILK_BODY_NAME), arm=Arms.LEFT
        )

        failed_check = PlanExecutor(context=context).run_step(action)

        assert failed_check == check_pre_condition(action, context)
        assert failed_check.satisfied is False


# %% selecting the objects relevant to the task


def names_with_parent(body, world) -> List[str]:
    """
    :param body: The body to name.
    :param world: The world the body is in.
    :return: The name of the body followed by the name of the body it is attached to,
        unless that is the world's root.
    """
    parent = body.parent_connection.parent
    if parent is world.root:
        return [body.name.name]
    return [body.name.name, parent.name.name]


class TestGoalState:
    def test_a_new_executor_has_no_goal_state(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context

        assert PlanExecutor(context=context).goal_state is None

    def test_the_executor_keeps_the_goal_state_it_is_given(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        goal = GoalDefinition(
            GoalConjunction(
                [GoalAtom(InsideOf.__name__, MILK_BODY_NAME, DRAWER_BODY_NAME)]
            )
        )

        assert PlanExecutor(context=context, goal_state=goal).goal_state is goal


@dataclass
class SilentLanguageModel:
    """
    A language model that is never expected to be asked.
    """

    def complete(self, prompt: str) -> str:
        raise AssertionError(f"the language model was asked: {prompt}")


class TestLanguageModel:
    def test_a_new_executor_has_no_language_model(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context

        assert PlanExecutor(context=context).language_model is None

    def test_the_executor_keeps_the_language_model_it_is_given(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        model = SilentLanguageModel()

        assert (
            PlanExecutor(context=context, language_model=model).language_model is model
        )

    def test_a_failed_condition_hands_the_language_model_to_the_failure_resolver(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        model = SilentLanguageModel()
        executor = PlanExecutor(context=context, language_model=model)
        failed = TripletExecution(
            triplet=None, grounded=None, outcome=ExecutionOutcome.CONDITION_FAILED
        )

        with (
            patch.object(executor, "run_triplet", return_value=failed),
            patch.object(executor_module, "FailureResolver") as resolver_class,
        ):
            executor.run_triplets([None])

        assert resolver_class.call_args.kwargs["language_model"] is model


# %% repairing a failed triplet


FAILED_TRIPLET = triplet_of(Milk, Drawer, PickUpAction)
"""
The triplet whose condition fails in these tests.
"""

NEXT_TRIPLET = triplet_of(Drawer, Drawer, CloseAction)
"""
The triplet after the one that fails.
"""


def execution_with(triplet: Triplet, outcome: ExecutionOutcome) -> TripletExecution:
    """
    :param triplet: The triplet the execution reports on.
    :param outcome: How far it got.
    :return: An execution of the triplet that grounded nothing.
    """
    return TripletExecution(triplet=triplet, grounded=None, outcome=outcome)


FAILED = execution_with(FAILED_TRIPLET, ExecutionOutcome.CONDITION_FAILED)
"""
The failed execution of :data:`FAILED_TRIPLET`.
"""

RETRIED = execution_with(FAILED_TRIPLET, ExecutionOutcome.PERFORMED)
"""
The execution of :data:`FAILED_TRIPLET` once it was repaired.
"""

NEXT = execution_with(NEXT_TRIPLET, ExecutionOutcome.PERFORMED)
"""
The execution of :data:`NEXT_TRIPLET`.
"""


class TestFailureRepair:
    def test_a_new_executor_repairs_failures(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context

        assert PlanExecutor(context=context).failure_check is True

    def test_the_repair_actions_of_the_answer_are_tried(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context, language_model=SilentLanguageModel())

        with (
            patch.object(executor, "run_triplet", side_effect=[FAILED, RETRIED]),
            patch.object(executor_module, "FailureResolver") as resolver_class,
        ):
            executor.run_triplets([FAILED_TRIPLET])

        resolver = resolver_class.return_value
        resolver.available_actions.assert_called_once_with(
            resolver.ask_query.return_value
        )
        resolver.try_actions.assert_called_once_with(
            resolver.available_actions.return_value
        )

    def test_the_resolver_is_given_what_the_executor_knows(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        goal = GoalDefinition(GoalConjunction([]))
        executor = PlanExecutor(
            context=context,
            goal_state=goal,
            language_model=SilentLanguageModel(),
            selected_objects=[MILK_BODY_NAME],
        )

        with (
            patch.object(executor, "run_triplet", side_effect=[FAILED, RETRIED]),
            patch.object(executor_module, "FailureResolver") as resolver_class,
        ):
            executor.run_triplets([FAILED_TRIPLET])

        assert resolver_class.call_args.kwargs == {
            "summary": FAILED.summary,
            "goal_state": goal,
            "objects": executor.selected_objects,
            "context": context,
            "objects_interaction_graph": executor.interaction_graph,
            "language_model": executor.language_model,
            "run": executor.run,
            "performer": executor,
        }

    def test_the_failed_triplet_is_run_again_after_the_repair(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context, language_model=SilentLanguageModel())

        with (
            patch.object(
                executor, "run_triplet", side_effect=[FAILED, RETRIED]
            ) as run_triplet,
            patch.object(executor_module, "FailureResolver"),
        ):
            executor.run_triplets([FAILED_TRIPLET])

        assert [call.args for call in run_triplet.call_args_list] == [
            (FAILED_TRIPLET,),
            (FAILED_TRIPLET,),
        ]

    def test_a_repaired_triplet_lets_the_sequence_go_on(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context, language_model=SilentLanguageModel())

        with (
            patch.object(executor, "run_triplet", side_effect=[FAILED, RETRIED, NEXT]),
            patch.object(executor_module, "FailureResolver"),
        ):
            run = executor.run_triplets([FAILED_TRIPLET, NEXT_TRIPLET])

        assert run.executions == [FAILED, RETRIED, NEXT]

    def test_a_triplet_still_failing_after_the_repair_ends_the_sequence(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context, language_model=SilentLanguageModel())

        with (
            patch.object(executor, "run_triplet", side_effect=[FAILED, FAILED]),
            patch.object(executor_module, "FailureResolver"),
        ):
            run = executor.run_triplets([FAILED_TRIPLET, NEXT_TRIPLET])

        assert run.executions == [FAILED, FAILED]

    def test_no_repair_is_made_when_failure_check_is_off(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(
            context=context, language_model=SilentLanguageModel(), failure_check=False
        )

        with (
            patch.object(executor, "run_triplet", side_effect=[FAILED]),
            patch.object(executor_module, "FailureResolver") as resolver_class,
        ):
            run = executor.run_triplets([FAILED_TRIPLET, NEXT_TRIPLET])

        resolver_class.assert_not_called()
        assert run.executions == [FAILED]

    def test_no_repair_is_made_without_a_language_model(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context)

        with (
            patch.object(executor, "run_triplet", side_effect=[FAILED]),
            patch.object(executor_module, "FailureResolver") as resolver_class,
        ):
            executor.run_triplets([FAILED_TRIPLET])

        resolver_class.assert_not_called()

    def test_the_feasible_actions_of_the_repair_are_displayed(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = RecordsDisplayedFeasibility(
            context=context, language_model=SilentLanguageModel(), visualize_graph=True
        )

        with (
            patch.object(executor, "run_triplet", side_effect=[FAILED, RETRIED]),
            patch.object(executor_module, "FailureResolver") as resolver_class,
        ):
            executor.run_triplets([FAILED_TRIPLET])

        assert executor.displayed_feasibilities == [
            resolver_class.return_value.actions_graph
        ]

    def test_the_feasible_actions_are_not_displayed_otherwise(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = RecordsDisplayedFeasibility(
            context=context, language_model=SilentLanguageModel()
        )

        with (
            patch.object(executor, "run_triplet", side_effect=[FAILED, RETRIED]),
            patch.object(executor_module, "FailureResolver"),
        ):
            executor.run_triplets([FAILED_TRIPLET])

        assert executor.displayed_feasibilities == []


REPAIR = execution_with(None, ExecutionOutcome.PERFORMED)
"""
An action performed to repair :data:`FAILED_TRIPLET`.
"""


@dataclass
class RecordsOneRepair:
    """
    A failure resolver that records one repair action in the run it is given, instead
    of asking a language model and performing anything.
    """

    run: SequenceExecution
    """
    The record the repair action is added to.
    """

    def __init__(self, run: SequenceExecution, **_) -> None:
        self.run = run

    def ask_query(self) -> list[str]:
        return []

    def available_actions(self, chosen_pairs: list[str]) -> list:
        return []

    def try_actions(self, available_actions: list) -> list[bool]:
        self.run.record(REPAIR)
        return [True]


class TestSequenceRecord:
    def test_a_new_executor_has_recorded_nothing(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context

        assert PlanExecutor(context=context).run == SequenceExecution()

    def test_running_triplets_returns_the_executor_s_record(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context)

        with patch.object(executor, "run_triplet", side_effect=[NEXT]):
            run = executor.run_triplets([NEXT_TRIPLET])

        assert run is executor.run

    def test_each_sequence_starts_a_new_record(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context)

        with patch.object(executor, "run_triplet", side_effect=[NEXT, NEXT]):
            executor.run_triplets([NEXT_TRIPLET])
            run = executor.run_triplets([NEXT_TRIPLET])

        assert run.executions == [NEXT]

    def test_repair_actions_are_recorded_between_the_failure_and_the_retry(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context, language_model=SilentLanguageModel())

        with (
            patch.object(executor, "run_triplet", side_effect=[FAILED, RETRIED]),
            patch.object(executor_module, "FailureResolver", RecordsOneRepair),
        ):
            run = executor.run_triplets([FAILED_TRIPLET])

        assert run.executions == [FAILED, REPAIR, RETRIED]
        assert run.summaries == [execution.summary for execution in run.executions]


# %% the step a triplet got to


FIRST_STEP = ParkArmsAction(Arms.BOTH)
"""
The first step of a planned triplet in these tests.
"""

OPEN_STEP = OpenAction(object_designator=None, arm=Arms.LEFT)
"""
The second step of a planned triplet in these tests.
"""

CLOSE_STEP = CloseAction(object_designator=None, arm=Arms.LEFT)
"""
The last step of a planned triplet in these tests.
"""

PLANNED_STEPS = [FIRST_STEP, OPEN_STEP, CLOSE_STEP]
"""
The steps every triplet plans into in these tests.
"""


class TestLastStep:
    def test_a_performed_triplet_reports_its_last_step(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        recorder = StepRecorder(context=context)

        with patch.object(
            executor_module, "plan_grounding", return_value=PLANNED_STEPS
        ):
            execution = recorder.run_triplet(triplet_of(Drawer, Drawer, OpenAction))

        assert execution.last_step is CLOSE_STEP

    def test_a_failed_triplet_reports_the_step_whose_condition_failed(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        recorder = StepRecorder(context=context, failing_action=OpenAction)

        with patch.object(
            executor_module, "plan_grounding", return_value=PLANNED_STEPS
        ):
            execution = recorder.run_triplet(triplet_of(Drawer, Drawer, OpenAction))

        assert execution.last_step is OPEN_STEP

    def test_a_triplet_that_was_not_planned_reports_no_step(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        recorder = StepRecorder(context=context)

        with patch.object(executor_module, "plan_grounding", return_value=[]):
            execution = recorder.run_triplet(triplet_of(Drawer, Drawer, OpenAction))

        assert execution.last_step is None

    def test_the_repair_is_given_the_step_whose_condition_failed(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context, language_model=SilentLanguageModel())
        failed = TripletExecution(
            triplet=FAILED_TRIPLET,
            grounded=None,
            outcome=ExecutionOutcome.CONDITION_FAILED,
            last_step=OPEN_STEP,
        )

        with (
            patch.object(executor, "run_triplet", return_value=failed),
            patch.object(
                executor, "failure_repair", return_value=RETRIED
            ) as failure_repair,
        ):
            executor.run_triplets([FAILED_TRIPLET])

        failure_repair.assert_called_once_with(failed, OPEN_STEP)


# %% a repair that brings about the failed step


def post_condition(satisfied: bool) -> ConditionCheck:
    """
    :param satisfied: Whether the post-condition holds.
    :return: A post-condition check of opening that reports it.
    """
    return ConditionCheck(
        action=OpenAction, kind=ConditionKind.POST, satisfied=satisfied
    )


FAILED_AT_OPEN = TripletExecution(
    triplet=FAILED_TRIPLET,
    grounded=None,
    outcome=ExecutionOutcome.CONDITION_FAILED,
    last_step=OPEN_STEP,
)
"""
A failed execution whose failed step is checkable.
"""


class TestTargetOccupancy:
    def test_a_new_executor_does_not_show_the_target_occupancy(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context

        executor = PlanExecutor(context=context)

        assert executor.target_occupancy is False
        assert executor.grounding.show_occupancy is False

    def test_the_grounding_shows_the_target_occupancy_when_asked_to(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context

        executor = PlanExecutor(context=context, target_occupancy=True)

        assert executor.grounding.show_occupancy is True


class TestActionInProgress:
    def test_a_new_executor_has_no_action_in_progress(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context

        assert PlanExecutor(context=context).action_in_progress() is None

    def test_a_triplet_is_in_progress_as_running_once_grounded(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        recorder = StepRecorder(context=context)
        triplet = triplet_of(Drawer, Drawer, OpenAction)

        with patch.object(
            executor_module, "plan_grounding", return_value=PLANNED_STEPS
        ):
            execution = recorder.run_triplet(triplet)

        assert recorder.action_in_progress() == TripletExecution(
            triplet, execution.grounded, ExecutionOutcome.RUNNING
        )

    def test_an_action_is_no_longer_in_progress_once_something_is_recorded(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context)
        executor.start_action(execution_with(None, ExecutionOutcome.RUNNING))

        executor.run.record(NEXT)

        assert executor.action_in_progress() is None

    def test_a_new_sequence_has_no_action_in_progress(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context)
        executor.start_action(execution_with(None, ExecutionOutcome.RUNNING))

        executor.run_triplets([])

        assert executor.action_in_progress() is None

    def test_every_triplet_is_recorded_through_record(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = RecordsRecordedExecutions(context=context)

        with patch.object(executor, "run_triplet", side_effect=[NEXT]):
            run = executor.run_triplets([NEXT_TRIPLET])

        assert executor.recorded == run.executions

    def test_a_repair_step_is_performed_and_refreshes_the_interaction_graph(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context, selected_objects=[MILK_BODY_NAME])

        with patch.object(executor, "perform_step") as perform_step:
            executor.perform_action_step(FIRST_STEP)

        perform_step.assert_called_once_with(FIRST_STEP)
        assert relation_triples(executor.interaction_graph) == relation_triples(
            executor.graph_relation([MILK_BODY_NAME], False)
        )


class TestRepairedTriplet:
    def test_the_triplet_is_not_run_again_when_the_failed_step_s_effect_holds(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context, language_model=SilentLanguageModel())

        with (
            patch.object(executor, "run_triplet") as run_triplet,
            patch.object(executor_module, "FailureResolver"),
            patch.object(
                executor_module,
                "check_post_condition",
                return_value=post_condition(True),
            ),
        ):
            executor.failure_repair(FAILED_AT_OPEN, OPEN_STEP)

        run_triplet.assert_not_called()

    def test_a_triplet_not_run_again_is_reported_as_repaired(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context, language_model=SilentLanguageModel())

        with (
            patch.object(executor_module, "FailureResolver"),
            patch.object(
                executor_module,
                "check_post_condition",
                return_value=post_condition(True),
            ),
        ):
            repaired = executor.failure_repair(FAILED_AT_OPEN, OPEN_STEP)

        assert repaired == TripletExecution(
            triplet=FAILED_AT_OPEN.triplet,
            grounded=FAILED_AT_OPEN.grounded,
            outcome=ExecutionOutcome.REPAIRED,
            last_step=OPEN_STEP,
        )

    def test_the_triplet_is_run_again_when_the_failed_step_s_effect_does_not_hold(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context, language_model=SilentLanguageModel())

        with (
            patch.object(executor, "run_triplet", return_value=RETRIED) as run_triplet,
            patch.object(executor_module, "FailureResolver"),
            patch.object(
                executor_module,
                "check_post_condition",
                return_value=post_condition(False),
            ),
        ):
            result = executor.failure_repair(FAILED_AT_OPEN, OPEN_STEP)

        run_triplet.assert_called_once_with(FAILED_TRIPLET)
        assert result is RETRIED

    def test_the_effect_of_a_step_without_conditions_is_never_taken_as_achieved(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context

        assert PlanExecutor(context=context).repair_achieved(FIRST_STEP) is False

    def test_a_repaired_triplet_lets_the_sequence_go_on(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context, language_model=SilentLanguageModel())

        with (
            patch.object(executor, "run_triplet", side_effect=[FAILED_AT_OPEN, NEXT]),
            patch.object(executor_module, "FailureResolver", RecordsOneRepair),
            patch.object(
                executor_module,
                "check_post_condition",
                return_value=post_condition(True),
            ),
        ):
            run = executor.run_triplets([FAILED_TRIPLET, NEXT_TRIPLET])

        assert [execution.outcome for execution in run.executions] == [
            ExecutionOutcome.CONDITION_FAILED,
            REPAIR.outcome,
            ExecutionOutcome.REPAIRED,
            ExecutionOutcome.PERFORMED,
        ]


class TestRepairMarker:
    def test_an_execution_without_a_triplet_is_summarised_as_a_repair(self) -> None:
        assert REPAIR.summary.is_repair is True

    def test_an_execution_of_a_triplet_is_not_summarised_as_a_repair(self) -> None:
        assert NEXT.summary.is_repair is False


class TestSelectedObjects:
    def test_a_new_executor_has_no_selected_objects(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context

        assert PlanExecutor(context=context).selected_objects == []

    def test_the_robot_bodies_are_recorded_on_creation(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, robot, context = apartment_world_pr2_copy_with_context

        assert PlanExecutor(context=context).robot_body_names == {
            robot_body.name.name for robot_body in robot.bodies
        }

    def test_both_bodies_and_their_parents_are_added(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        robot_body_names = PlanExecutor(context=context).robot_body_names
        milk = world.get_body_by_name(MILK_BODY_NAME)
        drawer = world.get_body_by_name(DRAWER_BODY_NAME)

        selected_objects = PlanExecutor.add_object(
            [], milk, drawer, robot_body_names, world.root.name.name
        )

        assert selected_objects == names_with_parent(milk, world) + names_with_parent(
            drawer, world
        )

    def test_a_missing_target_is_ignored(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        robot_body_names = PlanExecutor(context=context).robot_body_names
        milk = world.get_body_by_name(MILK_BODY_NAME)

        selected_objects = PlanExecutor.add_object(
            [], milk, None, robot_body_names, world.root.name.name
        )

        assert selected_objects == names_with_parent(milk, world)

    def test_a_missing_source_is_ignored(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        robot_body_names = PlanExecutor(context=context).robot_body_names
        drawer = world.get_body_by_name(DRAWER_BODY_NAME)

        selected_objects = PlanExecutor.add_object(
            [], None, drawer, robot_body_names, world.root.name.name
        )

        assert selected_objects == names_with_parent(drawer, world)

    def test_an_already_selected_object_is_not_added_again(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        robot_body_names = PlanExecutor(context=context).robot_body_names
        drawer = world.get_body_by_name(DRAWER_BODY_NAME)

        selected_objects = PlanExecutor.add_object(
            [], drawer, drawer, robot_body_names, world.root.name.name
        )
        selected_objects = PlanExecutor.add_object(
            selected_objects, drawer, None, robot_body_names, world.root.name.name
        )

        assert selected_objects == names_with_parent(drawer, world)

    def test_a_body_without_parent_adds_only_itself(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        robot_body_names = PlanExecutor(context=context).robot_body_names

        selected_objects = PlanExecutor.add_object(
            [], world.root, None, robot_body_names, world.root.name.name
        )

        assert selected_objects == [world.root.name.name]

    def test_the_world_root_as_parent_is_skipped(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        robot_body_names = PlanExecutor(context=context).robot_body_names
        free_object = Body(name=PrefixedName("free_object"))
        with world.modify_world():
            world.add_connection(FixedConnection(parent=world.root, child=free_object))

        selected_objects = PlanExecutor.add_object(
            [], free_object, None, robot_body_names, world.root.name.name
        )

        assert selected_objects == [free_object.name.name]

    def test_a_robot_body_as_source_is_skipped(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, robot, context = apartment_world_pr2_copy_with_context
        robot_body_names = PlanExecutor(context=context).robot_body_names
        drawer = world.get_body_by_name(DRAWER_BODY_NAME)

        selected_objects = PlanExecutor.add_object(
            [], robot.root, drawer, robot_body_names, world.root.name.name
        )

        assert selected_objects == names_with_parent(drawer, world)

    def test_a_robot_body_as_target_is_skipped(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        robot_body_names = PlanExecutor(context=context).robot_body_names
        milk = world.get_body_by_name(MILK_BODY_NAME)
        gripper = world.get_body_by_name(GRIPPER_BODY_NAME)

        selected_objects = PlanExecutor.add_object(
            [], milk, gripper, robot_body_names, world.root.name.name
        )

        assert selected_objects == names_with_parent(milk, world)

    def test_a_robot_body_as_parent_is_skipped(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        """
        An object held in the gripper is selected, the gripper it hangs from is not.
        """
        world, _, context = apartment_world_pr2_copy_with_context
        robot_body_names = PlanExecutor(context=context).robot_body_names
        held_object = Body(name=PrefixedName("held_object"))
        with world.modify_world():
            world.add_connection(
                FixedConnection(
                    parent=world.get_body_by_name(GRIPPER_BODY_NAME), child=held_object
                )
            )

        selected_objects = PlanExecutor.add_object(
            [], held_object, None, robot_body_names, world.root.name.name
        )

        assert selected_objects == [held_object.name.name]

    def test_running_a_triplet_selects_its_grounded_objects(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        """
        The objects are selected once grounded, even when the triplet cannot be planned.
        """
        world, _, context = apartment_world_pr2_copy_with_context
        recorder = StepRecorder(context=context)

        execution = recorder.run_triplet(triplet_of(Milk, Drawer, PlaceAction))

        assert recorder.selected_objects == names_with_parent(
            execution.grounded.source_object_body, world
        ) + names_with_parent(execution.grounded.target_object_body, world)


# %% graph of the selected objects


class TestGraphRelation:
    def test_the_annotated_selected_objects_are_the_nodes(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context

        graph = PlanExecutor(context=context).graph_relation(
            [MILK_BODY_NAME, world.root.name.name, DRAWER_BODY_NAME], False
        )

        assert [node.name for node in graph.nodes] == [MILK_BODY_NAME, DRAWER_BODY_NAME]

    def test_the_edges_are_the_default_relations_after_the_removal_rules(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        expected = InteractionGraph.from_world(
            world,
            [
                world.get_body_by_name(MILK_BODY_NAME),
                world.get_body_by_name(DRAWER_BODY_NAME),
            ],
            default_relations(),
        )
        expected.add_relation_edges()
        expected.remove_edges(default_edge_removal_rules())

        graph = PlanExecutor(context=context).graph_relation(
            [MILK_BODY_NAME, DRAWER_BODY_NAME], False
        )

        assert relation_triples(graph) == relation_triples(expected)

    def test_the_graph_is_displayed_when_asked_to(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = RecordsDisplayedGraphs(context=context)

        graph = executor.graph_relation([MILK_BODY_NAME], True)

        assert executor.displayed_graphs == [graph]

    def test_the_graph_is_not_displayed_otherwise(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        executor = RecordsDisplayedGraphs(context=context)

        executor.graph_relation([MILK_BODY_NAME], False)

        assert executor.displayed_graphs == []

    def test_a_running_page_is_given_the_new_graph(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context)
        executor.graph_visualizer = InteractionGraph(relations=[]).visualizer(
            open_browser=False
        )
        running_page = executor.graph_visualizer

        executor.display_graph(executor.graph_relation([MILK_BODY_NAME], False))

        assert executor.graph_visualizer is running_page
        assert [node["label"] for node in running_page.graph_nodes()] == [
            MILK_BODY_NAME
        ]

    def test_a_performed_triplet_builds_the_graph_of_the_selected_objects(
        self, apartment_world_pr2_copy_with_context, monkeypatch
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        monkeypatch.setattr(
            executor_module,
            "plan_grounding",
            lambda grounded, context: [ParkArmsAction(Arms.BOTH)],
        )
        recorder = RecordsStepsAndGraphs(context=context, visualize_graph=True)

        execution = recorder.run_triplet(triplet_of(Milk, Drawer, PickUpAction))

        assert execution.outcome is ExecutionOutcome.PERFORMED
        assert recorder.displayed_graphs == [recorder.interaction_graph]
        assert [node.name for node in recorder.interaction_graph.nodes] == [
            node.name
            for node in recorder.graph_relation(recorder.selected_objects, False).nodes
        ]

    def test_a_triplet_that_is_not_performed_builds_no_graph(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        recorder = RecordsStepsAndGraphs(context=context, visualize_graph=True)

        recorder.run_triplet(triplet_of(Milk, Drawer, PlaceAction))

        assert recorder.interaction_graph is None
        assert recorder.displayed_graphs == []


def relation_triples(graph: InteractionGraph) -> List[tuple[str, str, str]]:
    """
    :param graph: The graph to read.
    :return: Every edge as its source name, target name and relation name.
    """
    return [
        (edge.source.name, edge.target.name, type(edge.relation).__name__)
        for edge in graph.edges
    ]


# %% performing for real


class TestPerformedSequence:
    def test_opening_a_drawer_opens_it(
        self, apartment_world_pr2_copy_with_context
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        executor = PlanExecutor(context=context)
        drawer = world.get_body_by_name(DRAWER_BODY_NAME)
        drawer_joint = world.compute_parent_connection(drawer)

        with simulated_robot:
            run = executor.run_triplets([triplet_of(Drawer, Drawer, OpenAction)])

        assert run.executions[0].outcome is ExecutionOutcome.PERFORMED
        assert drawer_joint.position > 0.0
