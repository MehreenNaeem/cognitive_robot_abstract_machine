"""
Tests for :mod:`llm_planner.env_reasoner.failure_handler.repair_generator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from typing_extensions import Optional
from unittest.mock import patch

import pytest

from coraplex.locations.pose_validator import IsObjectReachableBy
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.datastructures.enums import Arms
from llm_planner.env_reasoner.failure_handler.failure_utilities import (
    MAX_QUERY_ATTEMPTS,
    ActionProcessor,
    ContextPerformer,
    RecoveryQueryFailed,
    get_action_descriptions,
    get_action_object_pair,
    get_robot_states,
)
from llm_planner.env_planner.execution_outcome import ExecutionOutcome
from llm_planner.env_planner.sequence_execution import (
    SequenceExecution,
    TripletExecution,
)
from llm_planner.env_reasoner.failure_handler.repair_generator import (
    FailureResolver,
    REPAIR_ACTIONS,
)
from llm_planner.env_reasoner.reasoner_graphs.AFG import ActionFeasibilityGraph
from llm_planner.env_reasoner.reasoner_graphs.Interactive_graph import (
    InteractionGraph,
    InteractionNode,
    SpatialRelation,
)
from llm_planner.env_reasoner.reasoner_graphs.goal_state import (
    GoalAtom,
    GoalConjunction,
    GoalDefinition,
)
from semantic_digital_twin.reasoning.predicates import InsideOf

MILK_BODY_NAME = "milk.stl"
"""
An annotated apartment object.
"""

DRAWER_BODY_NAME = "cabinet10_drawer_top"
"""
An apartment drawer.
"""


@dataclass
class ScriptedLanguageModel:
    """
    A language model answering with prepared texts, recording what it was asked.
    """

    answers: list[str]
    """
    The answers, in the order they are given.
    """

    prompts: list[str] = field(default_factory=list)
    """
    Every prompt asked, in order.
    """

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers[len(self.prompts) - 1]


@dataclass
class AlwaysHolds(SpatialRelation):
    """
    A relation that holds between any two objects, so a graph gets edges without
    depending on where the objects stand.
    """

    def holds(self, source: InteractionNode, target: InteractionNode) -> bool:
        return True


@pytest.fixture
def interaction_graph(apartment_world_pr2_copy_with_context) -> InteractionGraph:
    """
    :return: The graph of the milk and the drawer, related both ways.
    """
    world, _, _ = apartment_world_pr2_copy_with_context
    graph = InteractionGraph.from_world(
        world,
        [
            world.get_body_by_name(MILK_BODY_NAME),
            world.get_body_by_name(DRAWER_BODY_NAME),
        ],
        [AlwaysHolds()],
    )
    graph.add_relation_edges()
    return graph


# %% construction


class TestFailureResolverConstruction:
    def test_the_resolver_keeps_what_it_is_given(
        self, apartment_world_pr2_copy_with_context, interaction_graph
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        goal = GoalDefinition(
            GoalConjunction(
                [GoalAtom(InsideOf.__name__, MILK_BODY_NAME, DRAWER_BODY_NAME)]
            )
        )

        resolver = FailureResolver(
            summary=[],
            goal_state=goal,
            objects=[MILK_BODY_NAME],
            actions=[PickUpAction],
            context=context,
            objects_interaction_graph=interaction_graph,
            language_model=ScriptedLanguageModel(answers=[]),
        )

        assert resolver.summary == []
        assert resolver.goal_state is goal
        assert resolver.objects == [MILK_BODY_NAME]
        assert resolver.actions == [PickUpAction]
        assert resolver.context is context
        assert resolver.objects_interaction_graph is interaction_graph

    def test_the_default_actions_are_the_repair_actions(
        self, apartment_world_pr2_copy_with_context, interaction_graph
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context

        goal = GoalDefinition(
            GoalConjunction(
                [GoalAtom(InsideOf.__name__, MILK_BODY_NAME, DRAWER_BODY_NAME)]
            )
        )

        resolver = FailureResolver(
            summary=[],
            goal_state=goal,
            objects=[],
            context=context,
            objects_interaction_graph=interaction_graph,
            language_model=ScriptedLanguageModel(answers=[]),
        )

        assert resolver.actions == list(REPAIR_ACTIONS)

    def test_the_repair_actions_are_the_five_core_actions(self) -> None:
        assert REPAIR_ACTIONS == (
            PickUpAction,
            PlaceAction,
            OpenAction,
            CloseAction,
            NavigateAction,
        )


# %% feasible actions


class TestFeasibleActions:
    def test_the_feasible_actions_are_those_of_the_feasibility_graph_with_navigation(
        self, apartment_world_pr2_copy_with_context, interaction_graph
    ) -> None:
        world, _, context = apartment_world_pr2_copy_with_context
        resolver = FailureResolver(
            summary=[],
            goal_state=GoalDefinition(
                GoalConjunction(
                    [GoalAtom(InsideOf.__name__, MILK_BODY_NAME, DRAWER_BODY_NAME)]
                )
            ),
            objects=[MILK_BODY_NAME, DRAWER_BODY_NAME],
            actions=[PickUpAction, OpenAction],
            context=context,
            objects_interaction_graph=interaction_graph,
            language_model=ScriptedLanguageModel(answers=[]),
        )

        with patch.object(IsObjectReachableBy, "__call__", return_value=False):
            feasibility = resolver.feasible_actions()
            expected = ActionFeasibilityGraph(
                object_names=resolver.objects,
                actions=resolver.actions,
                world=world,
                context=context,
            ).build_action_graph([NavigateAction])

        assert feasibility == expected
        assert NavigateAction in feasibility


# %% current state


class TestCurrentState:
    def test_every_relation_is_listed_once_in_one_direction(
        self, apartment_world_pr2_copy_with_context, interaction_graph
    ) -> None:
        _, _, context = apartment_world_pr2_copy_with_context
        resolver = FailureResolver(
            summary=[],
            goal_state=GoalDefinition(
                GoalConjunction(
                    [GoalAtom(InsideOf.__name__, MILK_BODY_NAME, DRAWER_BODY_NAME)]
                )
            ),
            objects=[MILK_BODY_NAME, DRAWER_BODY_NAME],
            context=context,
            objects_interaction_graph=interaction_graph,
            language_model=ScriptedLanguageModel(answers=[]),
        )

        assert resolver.current_state() == [
            [MILK_BODY_NAME, AlwaysHolds.__name__, DRAWER_BODY_NAME],
        ]


# %% prompt


@dataclass
class DescribedFailure:
    """
    A failed action that only carries its description.
    """

    description: str
    """
    The failed action and why it failed, in words.
    """


FAILURE_DESCRIPTION = "Pick up milk.stl with the left arm: failed, the drawer is closed"
"""
The description of the failure the prompt is built for.
"""


@pytest.fixture
def prompt_resolver(
    apartment_world_pr2_copy_with_context, interaction_graph
) -> FailureResolver:
    """
    :return: A resolver of a failed pick up, for the milk and the drawer.
    """
    _, _, context = apartment_world_pr2_copy_with_context
    return FailureResolver(
        summary=DescribedFailure(FAILURE_DESCRIPTION),
        goal_state=GoalDefinition(
            GoalConjunction(
                [GoalAtom(InsideOf.__name__, MILK_BODY_NAME, DRAWER_BODY_NAME)]
            )
        ),
        objects=[MILK_BODY_NAME, DRAWER_BODY_NAME],
        actions=[PickUpAction, OpenAction],
        context=context,
        objects_interaction_graph=interaction_graph,
        language_model=ScriptedLanguageModel(answers=[]),
    )


@pytest.fixture
def prompt(prompt_resolver) -> str:
    with patch.object(IsObjectReachableBy, "__call__", return_value=False):
        return prompt_resolver.build_prompt()


class TestPrompt:
    def test_the_prompt_holds_the_goal_state(self, prompt_resolver, prompt) -> None:
        assert prompt_resolver.goal_state_summary() in prompt

    def test_the_prompt_holds_the_failure(self, prompt) -> None:
        assert FAILURE_DESCRIPTION in prompt

    def test_the_prompt_holds_every_object_relation(
        self, prompt_resolver, prompt
    ) -> None:
        for source_name, relation_name, target_name in prompt_resolver.current_state():
            assert f"{source_name} {relation_name} {target_name}" in prompt

    def test_the_prompt_holds_the_robot_state(self, prompt_resolver, prompt) -> None:
        robot_state = get_robot_states(
            prompt_resolver.context.world, prompt_resolver.objects
        )

        for line in robot_state.description():
            assert line in prompt

    def test_the_prompt_lists_every_feasible_action_object_pair(
        self, prompt_resolver, prompt
    ) -> None:
        with patch.object(IsObjectReachableBy, "__call__", return_value=False):
            pairs = get_action_object_pair(prompt_resolver.feasible_actions())

        for pair in pairs:
            assert f"- {pair}" in prompt


# %% asking the language model


class TestAskQuery:
    def test_the_checked_pairs_of_the_answer_are_returned(
        self, prompt_resolver
    ) -> None:
        with patch.object(IsObjectReachableBy, "__call__", return_value=False):
            pairs = get_action_object_pair(prompt_resolver.feasible_actions())
            prompt_resolver.language_model = ScriptedLanguageModel(
                answers=[f'["{pairs[1]}", "{pairs[0]}"]']
            )

            assert prompt_resolver.ask_query("prompt") == [pairs[1], pairs[0]]

    def test_the_given_prompt_is_asked(self, prompt_resolver) -> None:
        with patch.object(IsObjectReachableBy, "__call__", return_value=False):
            pairs = get_action_object_pair(prompt_resolver.feasible_actions())
            model = ScriptedLanguageModel(answers=[f'["{pairs[0]}"]'])
            prompt_resolver.language_model = model

            prompt_resolver.ask_query("prompt")

        assert model.prompts == ["prompt"]

    def test_a_pair_that_is_not_feasible_is_never_accepted(
        self, prompt_resolver
    ) -> None:
        prompt_resolver.language_model = ScriptedLanguageModel(
            answers=[f'["{CloseAction.__name__},{MILK_BODY_NAME}"]']
            * MAX_QUERY_ATTEMPTS
        )

        with patch.object(IsObjectReachableBy, "__call__", return_value=False):
            with pytest.raises(RecoveryQueryFailed):
                prompt_resolver.ask_query("prompt")


# %% available actions


class TestAvailableActions:
    def test_each_chosen_pair_gives_its_description_in_order(
        self, prompt_resolver
    ) -> None:
        with patch.object(IsObjectReachableBy, "__call__", return_value=False):
            feasibility = prompt_resolver.feasible_actions()
        chosen_pairs = [
            f"{NavigateAction.__name__},{DRAWER_BODY_NAME}",
            f"{OpenAction.__name__},{DRAWER_BODY_NAME}",
            f"{NavigateAction.__name__},{MILK_BODY_NAME}",
        ]

        assert prompt_resolver.available_actions(chosen_pairs) == [
            get_action_descriptions(pair, feasibility) for pair in chosen_pairs
        ]

    def test_there_is_one_description_per_chosen_pair(self, prompt_resolver) -> None:
        with patch.object(IsObjectReachableBy, "__call__", return_value=False):
            prompt_resolver.feasible_actions()
        chosen_pairs = [
            f"{NavigateAction.__name__},{DRAWER_BODY_NAME}",
            f"{NavigateAction.__name__},{MILK_BODY_NAME}",
        ]

        assert len(prompt_resolver.available_actions(chosen_pairs)) == len(chosen_pairs)

    def test_no_chosen_pairs_give_no_descriptions(self, prompt_resolver) -> None:
        with patch.object(IsObjectReachableBy, "__call__", return_value=False):
            prompt_resolver.feasible_actions()

        assert prompt_resolver.available_actions([]) == []

    def test_the_feasible_actions_are_worked_out_when_not_yet_known(
        self, prompt_resolver
    ) -> None:
        pair = f"{PickUpAction.__name__},{MILK_BODY_NAME}"

        with patch.object(IsObjectReachableBy, "__call__", return_value=False):
            available = prompt_resolver.available_actions([pair])

        assert available == [
            get_action_descriptions(pair, prompt_resolver.actions_graph)
        ]


# %% trying the available actions


@dataclass
class ScriptedActionProcessor(ActionProcessor):
    """
    An action processor that reports scripted outcomes instead of performing, and
    records what it was asked to process.
    """

    outcomes: list[Optional[ExecutionOutcome]] = field(default_factory=list)
    """
    The outcome each call to :meth:`process` reports, in call order; None stands for
    values that could not be grounded.
    """

    processed: list[tuple] = field(default_factory=list)
    """
    The action and values of every call to :meth:`process`, in call order.
    """

    def process(self, action, values) -> Optional[TripletExecution]:
        self.processed.append((action, values))
        outcome = self.outcomes[len(self.processed) - 1]
        if outcome is None:
            return None
        return TripletExecution(triplet=None, grounded=None, outcome=outcome)


FIRST_VALUES = {"approach": "front"}
"""
Stands in for the first set of condition values of an arm.
"""

SECOND_VALUES = {"approach": "back"}
"""
Stands in for the second set of condition values of an arm.
"""

THIRD_VALUES = {"approach": "left"}
"""
Stands in for the condition values of the other arm.
"""

OPEN_DRAWER = {
    OpenAction: {
        DRAWER_BODY_NAME: {
            Arms.RIGHT: [FIRST_VALUES, SECOND_VALUES],
            Arms.LEFT: [THIRD_VALUES],
        }
    }
}
"""
An action pair with three sets of values over two arms.
"""

PICK_UP_MILK = {PickUpAction: {MILK_BODY_NAME: {Arms.RIGHT: [FIRST_VALUES]}}}
"""
An action pair with a single set of values.
"""

PERFORMED = ExecutionOutcome.PERFORMED
FAILED = ExecutionOutcome.CONDITION_FAILED


def scripted(
    resolver: FailureResolver, outcomes: list[Optional[ExecutionOutcome]]
) -> ScriptedActionProcessor:
    """
    :param resolver: The resolver to hand the processor to.
    :param outcomes: What each processing reports, in call order.
    :return: The processor the resolver now uses.
    """
    resolver.action_processor = ScriptedActionProcessor(
        context=resolver.context, outcomes=outcomes
    )
    return resolver.action_processor


class TestTryActions:
    def test_the_resolver_processes_actions_in_its_own_context(
        self, prompt_resolver
    ) -> None:
        assert prompt_resolver.action_processor == ActionProcessor(
            context=prompt_resolver.context
        )

    def test_the_remaining_values_of_a_pair_are_skipped_once_one_succeeds(
        self, prompt_resolver
    ) -> None:
        processor = scripted(prompt_resolver, [FAILED, PERFORMED])

        prompt_resolver.try_actions([OPEN_DRAWER])

        assert processor.processed == [
            (OpenAction, FIRST_VALUES),
            (OpenAction, SECOND_VALUES),
        ]

    def test_every_value_of_every_arm_is_tried_before_the_next_pair(
        self, prompt_resolver
    ) -> None:
        processor = scripted(prompt_resolver, [FAILED, FAILED, FAILED, PERFORMED])

        prompt_resolver.try_actions([OPEN_DRAWER, PICK_UP_MILK])

        assert processor.processed == [
            (OpenAction, FIRST_VALUES),
            (OpenAction, SECOND_VALUES),
            (OpenAction, THIRD_VALUES),
            (PickUpAction, FIRST_VALUES),
        ]

    def test_each_pair_reports_whether_any_of_its_values_succeeded(
        self, prompt_resolver
    ) -> None:
        scripted(prompt_resolver, [FAILED, FAILED, FAILED, PERFORMED])

        assert prompt_resolver.try_actions([OPEN_DRAWER, PICK_UP_MILK]) == [
            False,
            True,
        ]

    def test_values_that_cannot_be_grounded_are_passed_over(
        self, prompt_resolver
    ) -> None:
        scripted(prompt_resolver, [None, PERFORMED])

        assert prompt_resolver.try_actions([OPEN_DRAWER]) == [True]

    def test_no_pairs_give_no_results(self, prompt_resolver) -> None:
        assert prompt_resolver.try_actions([]) == []


class TestPerformer:
    def test_the_resolver_hands_its_performer_to_its_action_processor(
        self, prompt_resolver
    ) -> None:
        performer = ContextPerformer(context=prompt_resolver.context)

        resolver = replace(prompt_resolver, performer=performer)

        assert resolver.action_processor.performer is performer


class TestRecordedAttempts:
    def test_a_new_resolver_has_recorded_nothing(self, prompt_resolver) -> None:
        assert prompt_resolver.run == SequenceExecution()

    def test_every_attempt_of_every_pair_is_recorded_in_order(
        self, prompt_resolver
    ) -> None:
        scripted(prompt_resolver, [FAILED, PERFORMED, FAILED])

        prompt_resolver.try_actions([OPEN_DRAWER, PICK_UP_MILK])

        assert [execution.outcome for execution in prompt_resolver.run.executions] == [
            FAILED,
            PERFORMED,
            FAILED,
        ]

    def test_every_attempt_is_summarised(self, prompt_resolver) -> None:
        scripted(prompt_resolver, [FAILED, PERFORMED])

        prompt_resolver.try_actions([OPEN_DRAWER])

        assert prompt_resolver.run.summaries == [
            execution.summary for execution in prompt_resolver.run.executions
        ]

    def test_values_that_cannot_be_grounded_are_not_recorded(
        self, prompt_resolver
    ) -> None:
        scripted(prompt_resolver, [None, PERFORMED])

        prompt_resolver.try_actions([OPEN_DRAWER])

        assert [execution.outcome for execution in prompt_resolver.run.executions] == [
            PERFORMED
        ]

    def test_attempts_are_recorded_in_the_run_the_resolver_is_given(
        self, prompt_resolver
    ) -> None:
        given = SequenceExecution()
        prompt_resolver.run = given
        scripted(prompt_resolver, [PERFORMED])

        prompt_resolver.try_actions([PICK_UP_MILK])

        assert prompt_resolver.run is given
        assert len(given.executions) == 1
