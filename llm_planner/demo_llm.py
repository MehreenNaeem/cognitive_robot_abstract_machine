import os

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    VisualizationBackend,
)
from coraplex.datastructures.grasp import GraspDescription

from coraplex.execution_environment import simulated_robot
from coraplex.locations.factories import reachability_location
from coraplex.plans.factories import sequential, execute_single
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.view_manager import ViewManager
from coraplex.visualization import WorldVisualization
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction

from coraplex.testing import setup_world
from llm_planner.env_planner.triplet_generator import (
    ActionSequencePredictor,
    preprocessing_objects,
    semantic_annotation_in_world,
)
from llm_planner.env_reasoner.LLM_utilities import OllamaLanguageModel
from llm_planner.env_reasoner.reasoner_graphs.goal_state import GoalState
from llm_planner.env_planner.execution_outcome import ExecutionOutcome
from llm_planner.env_planner.executor import PlanExecutor
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
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color
from coraplex.datastructures.enums import VisualizationBackend
from coraplex.visualization import WorldVisualization
from cramera.live.runner import start

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


# Distinct colors so the objects read clearly against the apartment.
world.get_body_by_name("milk.stl").visual.shapes[0].color = Color(0.9, 0.93, 1.0)
bowl.root.visual.shapes[0].color = Color(0.9, 0.75, 0.5)
spoon.root.visual.shapes[0].color = Color(0.6, 0.6, 0.65)
bread.root.visual.shapes[0].color = Color(0.396, 0.263, 0.129)

with world.modify_world():
    '''world.merge_world_at_pose(
        bowl,
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            2.4, 2.2, 0.98, reference_frame=world.root
        ),
    )'''
    connection_b = FixedConnection(
        parent=world.get_body_by_name("island_countertop"),
        child=bowl.root,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            -0.35, -0.04, 0.05
        ),
    )
    connection_t = FixedConnection(
        parent=world.get_body_by_name("table_area_main"),
        child=bread.root,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            -0.35, -0.04, 0.75
        ),
    )
    connection = FixedConnection(
        parent=world.get_body_by_name("cabinet10_drawer_top"),
        child=spoon.root,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            -0.05, -0.05, 0
        ),
    )
    world.merge_world(spoon, connection)
    world.merge_world(bowl, connection_b)
    world.merge_world(bread, connection_t)

try:
    import rclpy

    rclpy.init()
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    node = rclpy.create_node("viz_marker")
    v = VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
except ImportError:
    node = None
pr2 = PR2.from_world(world)
context = Context(world=world, robot=pr2, _debug=False, ros_node=node)

"""visualization = WorldVisualization.from_environment(
    world, default_backend=VisualizationBackend.CRAMERA
).start()

pr2 = PR2.from_world(world)
context = Context(world=world, robot=pr2, _debug=False, ros_node=visualization.ros_node)"""

with world.modify_world():
    world_reasoner = WorldReasoner(world)
    world_reasoner.reason()
    world.add_semantic_annotations(
        [
            Bowl(root=world.get_body_by_name("bowl.stl")),
            Spoon(root=world.get_body_by_name("spoon.stl")),
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
# visualization.attach_plan(plan)

with simulated_robot:
    plan.perform()


drawers = world.get_semantic_annotations_by_type(Drawer)
doors = world.get_semantic_annotations_by_type(Door)
cabinets = world.get_semantic_annotations_by_type(Cabinet)
wardrobe = world.get_semantic_annotations_by_type(Wardrobe)
Counter_top = world.get_semantic_annotations_by_type(CounterTop)
context.evaluate_conditions = False



task = "Place the bowl on the table"
language_model = OllamaLanguageModel(url=URL_OLLAMA, user_id=USER_ID_OLLAMA)
excluded_object_names = [PR2.__name__, Handle.__name__]

goal_state = GoalState(
    task=task,
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
#goal.visualize()

predictor = ActionSequencePredictor.from_world(
    world=world,
    language_model=language_model,
    excluded_object_names=excluded_object_names,
)
print("Asking LLM...")
act_trip = predictor.predict(task, vis=True,TS=True)
print("LLM Response: ")
for triplet in act_trip:
    print(
        f"{triplet.action.__name__}: "
        f"{triplet.source_object.__name__} -> {triplet.target_object.__name__}"
    )

executor = PlanExecutor(
    context=context,
    goal_state=goal,
    language_model=language_model,
    visualize_graph=False,
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


# TODO try to put object in drawer, put objects in cabinet,
# reduce the computational power
