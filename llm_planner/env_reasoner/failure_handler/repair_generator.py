from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import TYPE_CHECKING, Iterable, Iterator, Optional

from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from llm_planner.env_planner.execution_outcome import ExecutionOutcome
from llm_planner.env_planner.sequence_execution import SequenceExecution
from llm_planner.env_reasoner.failure_handler.failure_utilities import (
    ActionProcessor,
    get_action_descriptions,
    get_action_object_pair,
    get_robot_states,
    query_recovery_actions,
    remove_reversed_duplicates,
    bulleted_lines,
)
from llm_planner.env_reasoner.reasoner_graphs.AFG import (
    ActionFeasibilityGraph,
    ExecutableAction,
)

if TYPE_CHECKING:
    from llm_planner.env_reasoner.LLM_utilities import LanguageModel
    from llm_planner.env_reasoner.failure_handler.failure_utilities import (
        ActionPerformer,
    )
    from coraplex.datastructures.dataclasses import Context
    from coraplex.robot_plans.actions.base import ActionDescription
    from llm_planner.env_reasoner.reasoner_graphs.AFG import ActionFeasibility
    from llm_planner.env_reasoner.reasoner_graphs.Interactive_graph import (
        InteractionGraph,
    )
    from llm_planner.env_reasoner.reasoner_graphs.goal_state import GoalDefinition
    from llm_planner.env_reasoner.scene_summary import ActionSummary

# %% failure resolver


REPAIR_ACTIONS = (PickUpAction, PlaceAction, OpenAction, CloseAction, NavigateAction)
"""
Actions a repair is built from unless others are given.
"""


PROMPT_TEMPLATE = (
    Path(__file__).parents[2] / "resources" / "prompts" / "failure_recovery.txt"
)
"""
The prompt asking for the action and object pairs that recover from a failure, with
``goal_state``, ``failure``, ``object_relations``, ``robot_state`` and
``action_object_pairs`` placeholders.
"""


@dataclass
class FailureResolver:
    """
    Gathers what is known about a failed plan into a prompt asking for its repair.
    """

    summary: list[ActionSummary]
    """
    What happened to each action of the failed plan, in order.
    """

    goal_state: GoalDefinition
    """
    The goal the plan was meant to bring about.
    """

    objects: list[str]
    """
    Names of the bodies the repair may act on.
    """

    context: Context
    """
    The context the plan ran in.
    """

    objects_interaction_graph: InteractionGraph
    """
    The spatial relations between the objects of the task.
    """

    language_model: LanguageModel
    """
    The model asked how to recover, e.g. an
    :class:`~llm_planner.env_reasoner.LLM_utilities.OllamaLanguageModel`.
    """

    actions: list[type[ActionDescription]] = field(
        default_factory=lambda: list(REPAIR_ACTIONS)
    )
    """
    Actions the repair may use.
    """

    actions_graph: Optional[ActionFeasibility] = field(default=None, init=False)
    """
    The result of the last :meth:`feasible_actions`, or None before it ran.
    """

    run: SequenceExecution = field(default_factory=SequenceExecution, kw_only=True)
    """
    The record every attempt of :meth:`try_actions` is added to.
    """

    performer: Optional[ActionPerformer] = field(default=None, kw_only=True)
    """
    What each action tried in :meth:`try_actions` is started with and each of its steps
    performed by; the steps are performed in :attr:`context` when not given.
    """

    action_processor: ActionProcessor = field(init=False)
    """
    Grounds, plans and performs each action tried in :meth:`try_actions`, in
    :attr:`context`.
    """

    def __post_init__(self) -> None:
        self.action_processor = ActionProcessor(
            context=self.context, performer=self.performer
        )

    def failure_summary(self) -> str:
        """
        :return: The failed action and why it failed, in words.
        """
        return self.summary.description

    def goal_state_summary(self) -> str:
        """
        :return: The goal of the plan, in words.
        """
        return self.goal_state.to_pddl()

    def feasible_actions(self) -> ActionFeasibility:
        """
        :return: The actions that can be performed on :attr:`objects`, by object and
            arm, with navigation checked on every object.
        """
        graph = ActionFeasibilityGraph(
            object_names=self.objects,
            actions=self.actions,
            world=self.context.world,
            context=self.context,
        )
        self.actions_graph = graph.build_action_graph([NavigateAction])
        return self.actions_graph

    def current_state(self) -> list[list[str]]:
        """
        :return: Every relation of :attr:`objects_interaction_graph` as
            ``[source_name, relation_name, target_name]``, in the order it was added,
            with a relation that also holds the other way round listed once.
        """
        return remove_reversed_duplicates(
            [
                [edge.source.name, type(edge.relation).__name__, edge.target.name]
                for edge in self.objects_interaction_graph.edges
            ]
        )

    def build_prompt(self) -> str:
        """
        :return: The prompt asking for one or a sequence of the feasible action and
            object pairs that recovers from the failure, given the goal state, the
            failure, the relations between the objects and the state of the robot.
        """
        robot_state = get_robot_states(self.context.world, self.objects)
        return PROMPT_TEMPLATE.read_text().format(
            goal_state=self.goal_state_summary(),
            failure=self.failure_summary(),
            object_relations=bulleted_lines(
                f"{source_name} {relation_name} {target_name}"
                for source_name, relation_name, target_name in self.current_state()
            ),
            robot_state=bulleted_lines(robot_state.description()),
            action_object_pairs=bulleted_lines(
                get_action_object_pair(self.feasible_actions())
            ),
        )

    def ask_query(self) -> list[str]:
        """
        :param prompt: The recovery prompt, e.g. :meth:`build_prompt`.
        :return: The feasible ``action_name,object_name`` pairs the language model
            chose, in the order they are to be performed.
        :raises RecoveryQueryFailed: If none of its answers was usable.
        """
        return query_recovery_actions(
            self.language_model,
            self.build_prompt(),
            get_action_object_pair(self.feasible_actions()),
        )

    def available_actions(self, chosen_pairs: list[str]) -> list[ActionFeasibility]:
        """
        :param chosen_pairs: ``action_name,object_name`` pairs, e.g. from
            :meth:`ask_query`.
        :return: One part of :attr:`actions_graph` per chosen pair, in the same order,
            working the feasible actions out first if not yet known.
        :raises UnknownActionObjectPair: If a pair is not in :attr:`actions_graph`.
        """
        if self.actions_graph is None:
            self.feasible_actions()
        available = []
        for action_pair in chosen_pairs:
            available.append(get_action_descriptions(action_pair, self.actions_graph))
        return available

    def try_actions(self, available_actions: list[ActionFeasibility]) -> list[bool]:
        """
        Perform each action pair, trying its condition values one after another until
        one succeeds.

        A pair whose values all fail does not stop the pairs after it.

        :param available_actions: One part of :attr:`actions_graph` per action pair,
            e.g. from :meth:`available_actions`.
        :return: Whether any set of values succeeded, one per action pair, in the same
            order.
        """
        return [self.try_action_pair(action_pair) for action_pair in available_actions]

    def try_action_pair(self, action_pair: ActionFeasibility) -> bool:
        """
        Try the condition values of one action pair by arm and in order, recording
        every attempt in :attr:`run`, until one is performed.

        Values that lack an argument the action requires are passed over unrecorded.

        :param action_pair: The action, object and per-arm condition values of one
            action pair.
        :return: Whether any set of values was performed.
        """
        for attempt in self.attempts_of(action_pair):
            execution = self.action_processor.process(attempt.action, attempt.values)
            if execution is None:
                continue
            self.run.record(execution)
            if execution.outcome is ExecutionOutcome.PERFORMED:
                return True
        return False

    @staticmethod
    def attempts_of(action_pair: ActionFeasibility) -> Iterator[ExecutableAction]:
        """
        :param action_pair: The action, object and per-arm condition values of one
            action pair.
        :return: Every set of condition values of the pair with its action, by arm
            and in order.
        """
        return (
            ExecutableAction(action, values)
            for action, objects in action_pair.items()
            for arms in objects.values()
            for values_of_arm in arms.values()
            for values in values_of_arm
        )
