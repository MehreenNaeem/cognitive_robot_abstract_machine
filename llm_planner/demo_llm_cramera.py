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
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.view_manager import ViewManager
from coraplex.visualization import WorldVisualization
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction

from coraplex.testing import setup_world
from llm_planner.env_planner.triplet_generator import (
    ActionSequencePredictor,
    OllamaLanguageModel,
)
from llm_planner.ollama_config import URL_OLLAMA, USER_ID_OLLAMA
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import TorsoState
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
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color
from coraplex.datastructures.enums import VisualizationBackend
from coraplex.visualization import WorldVisualization
from cramera.live.bridge import BRIDGE
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

# Distinct colors so the objects read clearly against the apartment.
world.get_body_by_name("milk.stl").visual.shapes[0].color = Color(0.9, 0.93, 1.0)
bowl.root.visual.shapes[0].color = Color(0.9, 0.75, 0.5)
spoon.root.visual.shapes[0].color = Color(0.6, 0.6, 0.65)

with world.modify_world():
    world.merge_world_at_pose(
        bowl,
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            2.4, 2.2, 1, reference_frame=world.root
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

'''try:
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
context = Context(world=world, robot=pr2, _debug=False, ros_node=node)'''

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
from llm_planner.env_planner.triplet_grounding import TripletGrounding,ActionArgument
from llm_planner.env_planner.action_descriptors import plan_grounding
predictor = ActionSequencePredictor.from_world(
    world=world,
    language_model=OllamaLanguageModel(url=URL_OLLAMA, user_id=USER_ID_OLLAMA),
    excluded_object_names=[PR2.__name__, Handle.__name__],
)
print("Asking LLM...")
TASK = "Put the milk on the Table"
act_trip = predictor.predict(TASK,vis=True)
grounding = TripletGrounding(world=world)
print('LLM Response: ')
for triplet in act_trip:
    print(
        f"{triplet.action.__name__}: "
        f"{triplet.source_object.__name__} -> {triplet.target_object.__name__}"
    )
    grounded = grounding.ground(triplet)
    print(
        f"{grounded.action.__name__}("
        f"object={grounded.source_object_body.name.name}, "
        f"arm={grounded.arm.name}"
        + (
            ""
            if grounded.target_object_body is None
            else f", against={grounded.target_object_body.name.name}"
        )
        + ")"
    )
    if grounded.target_location is not None:
        place = grounded.target_location
        print(
            f"    put down at ({float(place.x):.2f}, {float(place.y):.2f}, "
            f"{float(place.z):.2f}), yaw {float(place.yaw):.2f}"
        )
    missing = [parameter.name for parameter in grounded.missing_parameters()]
    if missing:
        print(f"    the plan still needs: {', '.join(missing)}")
    arguments = ", ".join(sorted(grounded.action_arguments()))
    print(f"    the plan is built from: {arguments}")
    plan_1 = plan_grounding([grounded], context)
    # the LLM-Planner panel draws the task, this step under it and the plan steps it
    # performs, and follows their status while the plan runs
    BRIDGE.publish_llm_plan_step(TASK, grounded.action.__name__, plan_1)
    visualization.attach_plan(plan_1)
    with simulated_robot:
        plan_1.perform()

print()
