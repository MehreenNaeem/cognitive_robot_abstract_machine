"""
Performing a predicted triplet sequence: grounding each triplet, planning it, and
running its steps with the action's own pre- and post-condition checked around them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Optional, TYPE_CHECKING

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import execute_single
from coraplex.plans.plan_node import ActionLike
from llm_planner.env_planner.action_descriptors import plan_grounding
from llm_planner.env_planner.action_validation import (
    CHECKABLE_ACTIONS,
    ConditionCheck,
    check_post_condition,
    check_pre_condition,
)
from llm_planner.env_planner.execution_outcome import ExecutionOutcome
from llm_planner.env_planner.sequence_execution import (
    SequenceExecution,
    StartedAction,
    TripletExecution,
)
from llm_planner.env_planner.triplet_generator import Triplet
from llm_planner.env_planner.triplet_grounding import GroundedTriplet, TripletGrounding
from llm_planner.env_reasoner.failure_handler.repair_generator import FailureResolver
from llm_planner.env_reasoner.reasoner_graphs.Interactive_graph import (
    InteractionGraph,
    InteractionGraphVisualizer,
    default_edge_removal_rules,
    default_relations,
)
from llm_planner.env_reasoner.scene_summary import ActionSummary
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from llm_planner.env_reasoner.LLM_utilities import LanguageModel
    from llm_planner.env_reasoner.reasoner_graphs.AFG import ActionFeasibility
    from llm_planner.env_reasoner.reasoner_graphs.goal_state import GoalDefinition


# %% performing a predicted sequence


@dataclass
class PlanExecutor:
    """
    Performs a predicted triplet sequence in a world.

    Each triplet is grounded into the bodies it acts on, planned into its navigation and
    action steps, and performed step by step. One grounding serves the whole sequence,
    so the drawer a sequence opens is the drawer it places into and closes again.

    ..note:: Performing needs an execution environment, e.g. ``with simulated_robot:``
        around :meth:`run_triplets`.
    """

    context: Context
    """
    The context every triplet is planned and performed in, holding the world and robot.
    """

    grounding: TripletGrounding = field(init=False)
    """
    Resolves each triplet's object types to the bodies acted on.
    """

    selected_objects: List[str] = field(default_factory=list, kw_only=True)
    """
    The names of the bodies relevant to the task, and of the bodies they are attached
    to, in the order they were first grounded.
    """

    robot_body_names: set[str] = field(init=False)
    """
    The names of the robot's own bodies, recorded on creation so that an object the
    robot later holds is not counted among them.
    """

    goal_state: Optional[GoalDefinition] = field(default=None, kw_only=True)
    """
    The goal the task is meant to bring about, or None when it was not calculated.
    """

    language_model: Optional[LanguageModel] = field(default=None, kw_only=True)
    """
    The model asked how to recover from a failed condition, or None when none is
    available.
    """

    failure_check: bool = field(default=True, kw_only=True)
    """
    Whether a triplet whose condition failed is repaired and run again, which needs
    :attr:`language_model`.
    """

    visualize_graph: bool = field(default=False, kw_only=True)
    """
    Whether the graph of the selected objects is plotted after each performed triplet,
    and the feasible actions of each failure repair are shown.
    """

    target_occupancy: bool = field(default=False, kw_only=True)
    """
    Whether the free and taken places searched on the target of a placing triplet are
    shown in the world, as regions a viewer draws.
    """

    run: SequenceExecution = field(default_factory=SequenceExecution, init=False)
    """
    What became of every triplet and repair action of the last sequence, in execution
    order.
    """

    started: Optional[StartedAction] = field(default=None, init=False)
    """
    The triplet or repair action that started last, or None before any did in the
    current sequence.
    """

    interaction_graph: Optional[InteractionGraph] = field(default=None, init=False)
    """
    The graph of the selected objects and their relations, as of the last performed
    triplet, or None before any triplet was performed.
    """

    graph_visualizer: Optional[InteractionGraphVisualizer] = field(
        default=None, init=False
    )
    """
    The page plotting :attr:`interaction_graph`, or None before it is first plotted.
    """

    def __post_init__(self) -> None:
        self.grounding = TripletGrounding(
            world=self.context.world, show_occupancy=self.target_occupancy
        )
        self.robot_body_names = {
            robot_body.name.name for robot_body in self.context.robot.bodies
        }

    def run_triplets(self, triplets: List[Triplet]) -> SequenceExecution:
        """
        Perform a predicted sequence, one triplet after another.

        A triplet that could not be planned is reported and skipped, since the rest of
        the sequence does not depend on it. A triplet whose condition failed is repaired
        and run once more when :meth:`repairs_failures`. A failed condition that
        remains ends the sequence: its later triplets act on a world the earlier ones
        were meant to bring about.

        :param triplets: The predicted sequence, in execution order.
        :return: :attr:`run`, holding what became of every triplet up to and including
            the one that failed, each with what it reads as; a repaired triplet is
            reported before its repair actions and again after them.
        """
        self.run = SequenceExecution()
        self.started = None
        for triplet in triplets:
            execution = self.run_triplet(triplet)
            self.record(execution)
            if execution.outcome is not ExecutionOutcome.CONDITION_FAILED:
                continue
            if not self.repairs_failures():
                break
            execution = self.failure_repair(execution, execution.last_step)
            self.record(execution)
            if execution.outcome is ExecutionOutcome.CONDITION_FAILED:
                break
        return self.run

    def record(self, execution: TripletExecution) -> None:
        """
        Add what became of a triplet to :attr:`run`.

        :param execution: What became of the triplet just run.
        """
        self.run.record(execution)

    def start_action(self, execution: TripletExecution) -> None:
        """
        Note that a triplet or repair action was grounded and starts running.

        :param execution: The action, reported as :attr:`ExecutionOutcome.RUNNING`.
        """
        self.started = StartedAction(execution, len(self.run.executions))

    def action_in_progress(self) -> Optional[TripletExecution]:
        """
        :return: The triplet or repair action running now, reported as
            :attr:`ExecutionOutcome.RUNNING`; None when the last one started has been
            recorded since.
        """
        if self.started is None or not self.started.is_in_progress(self.run):
            return None
        return self.started.execution

    def repairs_failures(self) -> bool:
        """
        :return: Whether a failed triplet is repaired: :attr:`failure_check` is on and
            there is a :attr:`language_model` to ask for the repair.
        """
        return self.failure_check and self.language_model is not None

    def failure_repair(
        self, execution: TripletExecution, failure_step: ActionLike
    ) -> TripletExecution:
        """
        Ask the language model how to recover from a failed triplet, try the recovery
        actions it chose, recording every attempt in :attr:`run`, and run the triplet
        again unless the repair already brought about what the failed step was meant
        to.

        :param execution: The execution whose condition failed.
        :param failure_step: The step whose condition failed.
        :return: The triplet reported as :attr:`ExecutionOutcome.REPAIRED` when the
            failed step's post-condition holds after the repair, else what became of it
            when run again.
        :raises RecoveryQueryFailed: If none of the language model's answers was usable.
        """
        resolver = FailureResolver(
            summary=execution.summary,
            goal_state=self.goal_state,
            objects=self.selected_objects,
            context=self.context,
            objects_interaction_graph=self.interaction_graph,
            language_model=self.language_model,
            run=self.run,
            performer=self,
        )
        chosen_pairs = resolver.ask_query()
        if self.visualize_graph:
            self.display_action_feasibility(resolver.actions_graph)
        resolver.try_actions(resolver.available_actions(chosen_pairs))
        if self.repair_achieved(failure_step):
            return TripletExecution(
                execution.triplet,
                execution.grounded,
                ExecutionOutcome.REPAIRED,
                last_step=failure_step,
            )
        return self.run_triplet(execution.triplet)

    def repair_achieved(self, failure_step: ActionLike) -> bool:
        """
        :param failure_step: The step whose condition failed.
        :return: Whether the step's post-condition now holds; a step outside
            :data:`~llm_planner.env_planner.action_validation.CHECKABLE_ACTIONS` has
            none to check, so it never counts as achieved.
        """
        if not isinstance(failure_step, CHECKABLE_ACTIONS):
            return False
        return check_post_condition(failure_step, self.context).satisfied

    def run_triplet(self, triplet: Triplet) -> TripletExecution:
        """
        Ground one triplet and perform the steps it plans into.

        :param triplet: The triplet to perform.
        :return: How far it got, the condition that stopped it if one did, and the
            last step run.
        """
        grounded = self.grounding.ground(triplet)
        self.start_action(TripletExecution(triplet, grounded, ExecutionOutcome.RUNNING))
        self.selected_objects = self.add_object(
            self.selected_objects,
            grounded.source_object_body,
            grounded.target_object_body,
            self.robot_body_names,
            self.context.world.root.name.name,
        )
        steps = plan_grounding(
            [grounded], self.context
        )  # add additional low level actions if needed
        if not steps:
            return TripletExecution(triplet, grounded, ExecutionOutcome.NOT_PLANNED)
        for step in steps:
            failed_check = self.run_step(step)

            # calculate interaction edges of objects
            self.interaction_graph = self.graph_relation(
                self.selected_objects, self.visualize_graph
            )
            if failed_check is not None:
                return TripletExecution(
                    triplet,
                    grounded,
                    ExecutionOutcome.CONDITION_FAILED,
                    failed_check,
                    last_step=step,
                )
        return TripletExecution(
            triplet, grounded, ExecutionOutcome.PERFORMED, last_step=steps[-1]
        )

    def graph_relation(
        self, selected_objects: List[str], visualize: bool
    ) -> InteractionGraph:
        """
        Build the graph of the selected objects, connected by the spatial relations
        between them. Objects without a semantic annotation are left out.

        :param selected_objects: The names of the bodies to put in the graph.
        :param visualize: Whether to plot the graph.
        :return: The graph.
        """
        world = self.context.world
        graph = InteractionGraph.from_selected_bodies(
            world,
            [world.get_body_by_name(name) for name in selected_objects],
            default_relations(),
            default_edge_removal_rules(),
        )
        if visualize:
            self.display_graph(graph)
        return graph

    def display_graph(self, graph: InteractionGraph) -> None:
        """
        Plot ``graph`` on one web page shared by all triplets: the first call starts
        the page, later calls show their graph on it at its next refresh.

        :param graph: The graph to plot.
        """
        if self.graph_visualizer is None:
            self.graph_visualizer = graph.visualize()
            return
        graph.show_in(self.graph_visualizer)

    def display_action_feasibility(self, feasibility: ActionFeasibility) -> None:
        """
        Show the actions a failure repair could choose from. Nothing is shown here; an
        executor with somewhere to show them overrides this.

        :param feasibility: The feasible actions of the repair, by action, object and
            arm.
        """

    @staticmethod
    def add_object(
        selected_objects: List[str],
        source_object_body: Optional[Body],
        target_object_body: Optional[Body],
        robot_body_names: set[str],
        world_root_name: str,
    ) -> List[str]:
        """
        Add the names of both bodies, and of the bodies they are attached to, to the
        selected objects. A missing body, a body of the robot, the world's root as a
        parent and a name already selected are skipped.

        :param selected_objects: The names of the objects selected so far.
        :param source_object_body: The body the action is performed on, or None.
        :param target_object_body: The body the action is performed against, or None.
        :param robot_body_names: The names of the robot's bodies, which are never selected.
        :param world_root_name: The name of the world's root, which is never selected as
            a parent.
        :return: The selected object names, with both bodies and their parents added.
        """
        body_name = selected_objects
        for object_body in (source_object_body, target_object_body):
            if object_body is None:
                continue
            if object_body.name.name in robot_body_names:
                continue
            if object_body.name.name not in body_name:
                body_name.append(object_body.name.name)
            parent_connection = object_body.parent_connection
            if parent_connection is None:
                continue
            parent_name = parent_connection.parent.name.name
            if parent_name == world_root_name:
                continue
            if parent_name not in body_name and parent_name not in robot_body_names:
                body_name.append(parent_name)
        return body_name

    def run_step(self, step: ActionLike) -> Optional[ConditionCheck]:
        """
        Perform one step, checking the conditions the action declares around it.

        A step outside :data:`~llm_planner.env_planner.action_validation.CHECKABLE_ACTIONS`
        is performed unchecked. The pre-condition is checked before performing, so a
        step that cannot run does not run at all.

        :param step: The step to perform.
        :return: The condition that did not hold, or None if the step ran and both held.
        """
        if not isinstance(step, CHECKABLE_ACTIONS):
            self.perform_step(step)
            return None
        pre_condition = check_pre_condition(step, self.context)
        if not pre_condition.satisfied:
            return pre_condition
        self.perform_step(step)
        post_condition = check_post_condition(step, self.context)
        return None if post_condition.satisfied else post_condition

    def perform_step(self, step: ActionLike) -> None:
        """
        Perform one step in this executor's context.

        :param step: The step to perform.
        """
        execute_single(step, context=self.context).perform()

    def perform_action_step(self, step: ActionLike) -> None:
        """
        Perform one step of a repair action, then work out the graph of the selected
        objects again, as after every step of a triplet.

        :param step: The step to perform.
        """
        self.perform_step(step)
        self.interaction_graph = self.graph_relation(
            self.selected_objects, self.visualize_graph
        )


# TODO add goal check condition
