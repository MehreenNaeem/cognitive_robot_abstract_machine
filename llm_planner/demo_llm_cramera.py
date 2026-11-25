import os
from dataclasses import dataclass

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
)
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.plan_node import ActionLike

from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction

from coraplex.testing import setup_world
from llm_planner.demo_utilities import SceneInteractionDisplay
from llm_planner.env_planner.execution_outcome import ExecutionOutcome
from llm_planner.env_planner.executor import PlanExecutor
from llm_planner.env_planner.sequence_execution import TripletExecution
from llm_planner.env_reasoner.reasoner_graphs.AFG import (
    ActionFeasibility,
    feasibility_tree,
)
from llm_planner.env_reasoner.reasoner_graphs.Interactive_graph import InteractionGraph
from llm_planner.env_planner.triplet_generator import (
    ActionSequencePredictor,
    preprocessing_objects,
    semantic_annotation_in_world,
)
from llm_planner.env_reasoner.reasoner_graphs.goal_state import GoalState
from llm_planner.env_reasoner.LLM_utilities import OllamaLanguageModel
from llm_planner.ollama_config import URL_OLLAMA, USER_ID_OLLAMA
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.predicates import (
    Above,
    InsideOf,
    contact,
    is_supporting,
)
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Spoon,
    Drawer,
    Handle,
    Wardrobe,
    Cabinet,
    Door,
    CounterTop,
    Sink,
    Bread,
    Bottle,
    Apple,
    Plate,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color, Scale
from coraplex.datastructures.enums import VisualizationBackend
from coraplex.visualization import WorldVisualization
from cramera.live.bridge import BRIDGE, TaskStatusName
from cramera.live.llm_plan import ExecutionSummaryRow
from cramera.live.runner import start
from llm_planner.demo_utilities import ObjectModelParser

# %% showing what became of every execution in the panel


OUTCOME_STATUSES = {
    ExecutionOutcome.RUNNING: TaskStatusName.RUNNING,
    ExecutionOutcome.PERFORMED: TaskStatusName.SUCCEEDED,
    ExecutionOutcome.REPAIRED: TaskStatusName.SUCCEEDED,
    ExecutionOutcome.CONDITION_FAILED: TaskStatusName.FAILED,
    ExecutionOutcome.NOT_PLANNED: TaskStatusName.FAILED,
}
"""
The status each outcome is coloured with in the LLM-planner panel.
"""


@dataclass
class PanelPlanExecutor(PlanExecutor):
    """
    A plan executor that shows the summary of every execution of its run in the
    LLM-planner panel, under the task.
    """

    task: str
    """
    The task statement the summaries are shown under.
    """

    visualization: WorldVisualization
    """
    The running visualization each step's plan is attached to.
    """

    scene_display: SceneInteractionDisplay
    """
    Shows the interaction graph of the selected objects in the viewer.
    """

    def record(self, execution: TripletExecution) -> None:
        """
        Record what became of a triplet, then show it.
        """
        super().record(execution)
        self.publish_summaries()

    def start_action(self, execution: TripletExecution) -> None:
        """
        Note that a triplet or repair action starts running, then show it as running.
        """
        super().start_action(execution)
        self.publish_summaries()

    def perform_step(self, step: ActionLike) -> None:
        """
        Follow the step in the visualization while it runs, then show the summaries
        recorded so far.
        """
        node = execute_single(step, context=self.context)
        self.visualization.attach_plan(node.plan)
        node.perform()
        self.publish_summaries()

    def publish_summaries(self) -> None:
        """
        Show the summary of every execution recorded in :attr:`run` so far, followed by
        the action running now, if any.
        """
        executions = list(self.run.executions)
        in_progress = self.action_in_progress()
        if in_progress is not None:
            executions.append(in_progress)
        BRIDGE.publish_execution_summaries(
            self.task,
            [
                ExecutionSummaryRow(
                    label=str(execution.summary),
                    status=OUTCOME_STATUSES[execution.outcome],
                )
                for execution in executions
            ],
        )

    def display_graph(self, graph: InteractionGraph) -> None:
        """
        Show the graph in the Interaction tab of the panel's graph view and in the
        Interaction layer of the Semantic Digital Twin scene.
        """
        self.scene_display.show_graph(graph)

    def display_action_feasibility(self, feasibility: ActionFeasibility) -> None:
        """
        Show the actions a failure repair could choose from in the AFG tab of the
        panel's graph view.
        """
        BRIDGE.publish_feasibility_tree(feasibility_tree(feasibility))


world = setup_world()
spoon = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "coraplex", "resources", "objects", "spoon.stl"
    )
).parse()
bowl = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "coraplex", "resources", "objects", "bowl.stl"
    )
).parse()
bread = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "coraplex", "resources", "objects", "bread.stl"
    )
).parse()

mj_objects_dir = "/home/mnaeem/cognitive_robot_abstract_machine/llm_planner/objects_data"
plate = ObjectModelParser(f"{mj_objects_dir}/plate_1/model.xml").parse()
bottle =  ObjectModelParser(f"{mj_objects_dir}/bottled_drink_1/model.xml").parse()
apple = ObjectModelParser(f"{mj_objects_dir}/apple_14/model.xml").parse()



# Distinct colors so the objects read clearly against the apartment.
world.get_body_by_name("milk.stl").visual.shapes[0].color = Color(0.9, 0.93, 1.0)
bowl.root.visual.shapes[0].color = Color(0.9, 0.75, 0.5)
spoon.root.visual.shapes[0].color = Color(0.6, 0.6, 0.65)
bread.root.visual.shapes[0].color = Color(0.396, 0.263, 0.129)



with world.modify_world():
    connection_b = FixedConnection(
        parent=world.get_body_by_name("island_countertop"),
        child=bowl.root,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            -0.35, -0.04, 0.05
        ),
    )
    connection = FixedConnection(
        parent=world.get_body_by_name("cabinet10_drawer_top"),
        child=spoon.root,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            -0.05, -0.05, 0
        ),
    )
    connection_t = FixedConnection(
        parent=world.get_body_by_name("table_area_main"),
        child=bread.root,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            -0.35, -0.04, 0.75
        ),
    )
    connection_bt = FixedConnection(
        parent=world.get_body_by_name("table_area_main"),
        child=bottle.root,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            -0.1, -0.04, 0.8
        ),
    )

    connection_p = FixedConnection(
        parent=world.get_body_by_name("table_area_main"),
        child=plate.root,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            0.3, -0.1, 0.73
        ),
    )

    connection_a = FixedConnection(
        parent=world.get_body_by_name("table_area_main"),
        child=apple.root,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            0.5, -0.2, 0.76
        ),
    )
    world.merge_world(spoon, connection)
    world.merge_world(bowl, connection_b)
    world.merge_world(bread, connection_t)
    world.merge_world(bottle, connection_bt)
    world.merge_world(plate, connection_p)
    world.merge_world(apple, connection_a)

visualization = WorldVisualization.from_environment(
    world, default_backend=VisualizationBackend.CRAMERA
).start()

pr2 = PR2.from_world(world)
context = Context(world=world, robot=pr2, _debug=False, ros_node=visualization.ros_node)

with world.modify_world():
    world_reasoner = WorldReasoner(world)
    world_reasoner.reason()
    world.add_semantic_annotations(
        [
            Bowl(root=world.get_body_by_name("bowl.stl")),
            Spoon(root=world.get_body_by_name("spoon.stl")),
            Plate(root=world.get_body_by_name("plate_1")),
            Bottle(root=world.get_body_by_name("bottled_drink_1")),
            Apple(root=world.get_body_by_name("apple_14")),
        ]
    )
    world.add_semantic_annotation_recursively(
        Drawer(
            root=world.get_body_by_name("cabinet10_drawer_top"),
            handle=Handle(root=world.get_body_by_name("handle_cab10_t")),
        )
    )
plan = sequential(
    [
        ParkArmsAction(Arms.BOTH),
        MoveTorsoAction(TorsoState.HIGH),
    ],
    context=context,
).plan
visualization.attach_plan(plan)

with simulated_robot:
    plan.perform()

# %% the goal state of the task

TASK = "Place the Apple in the Plate"
language_model = OllamaLanguageModel(url=URL_OLLAMA, user_id=USER_ID_OLLAMA)
excluded_object_names = [PR2.__name__, Handle.__name__]

goal_state = GoalState(
    task=TASK,
    object_types=preprocessing_objects(
        semantic_annotation_in_world(world), excluded_object_names
    ),
    predicates=[InsideOf, Above, is_supporting, contact],
    world=world,
    language_model=language_model,
)
print("Asking LLM for the goal state...")
goal = goal_state.goal()
print(f"Goal state: {goal.to_pddl()}")
BRIDGE.publish_goal(goal)

# %% performing what the language model predicted

predictor = ActionSequencePredictor.from_world(
    world=world,
    language_model=language_model,
    excluded_object_names=excluded_object_names,
)
print("Asking LLM...")
act_trip = predictor.predict(TASK, vis=True, TS=False)
print("LLM Response: ")
for triplet in act_trip:
    print(
        f"{triplet.action.__name__}: "
        f"{triplet.source_object.__name__} -> {triplet.target_object.__name__}"
    )

executor = PanelPlanExecutor(
    context=context,
    task=TASK,
    visualization=visualization,
    scene_display=SceneInteractionDisplay(world=world, bridge=BRIDGE),
    goal_state=goal,
    language_model=language_model,
    visualize_graph=True,
    target_occupancy=True,
)
with simulated_robot:
    run = executor.run_triplets(act_trip)

for execution, summary in zip(run.executions, run.summaries):
    print(summary)
    if execution.outcome is ExecutionOutcome.NOT_PLANNED:
        missing = [
            parameter.name for parameter in execution.grounded.missing_parameters()
        ]
        print(f"    the plan still needs: {', '.join(missing)}")
print()
