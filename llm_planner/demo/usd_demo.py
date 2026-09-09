from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Type

import numpy as np
import pytest

from coraplex.datastructures.enums import VisualizationBackend
from coraplex.visualization import WorldVisualization
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.usd.parser import USDParser, UsdMeshShapeBuilder
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from cramera.live.runner import start


logger = logging.getLogger(__name__)

ROBOT_SPAWN_POSITION_XY = (1.5, 2.5)

APARTMENT_PREFIX = "apartment"
"""
The prefix of every body name in the parsed apartment, and so the first half of every
frame id the TF publisher advertises.
"""

TF_PUBLISH_PERIOD_SECONDS = 0.1
"""
How often the tf tree is republished, short enough to stay inside RViz's default ten
second transform buffer.
"""


def setup_world(robot_type: Type[AbstractRobot] = PR2) -> World:
    """
    Build the apartment world with a robot standing in front of the kitchen counter.

    :param robot_type: Robot to place in the apartment.
    """
    logger.setLevel(logging.DEBUG)


    print("setup_world: parsing apartment USD...")
    apartment_world = USDParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "resources",
            "usd",
            "apartment",
            "apartmentICRA.usda",
        ),
        prefix=APARTMENT_PREFIX,
    ).parse()
    print("setup_world: parsing object meshes...")

    return apartment_world

# BROWSER DISPLAY
world = setup_world()
visualization = WorldVisualization.from_environment(
    world, default_backend=VisualizationBackend.CRAMERA
).start()


'''
#RVIZ DISPLAY
try:
    import rclpy

    rclpy.init()
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher

    node = rclpy.create_node("viz_marker")
    # Both publishers must stay referenced for as long as the node runs: dropping one
    # collects it, and its ROS publisher with it.
    visualization = VizMarkerPublisher(_world=world, node=node)
    tf_publisher = TFPublisher(_world=world, node=node)
    # TFPublisher publishes on a world state change, and this world never changes after
    # it is built, so tf would be published once and then go stale in RViz's buffer.
    node.create_timer(TF_PUBLISH_PERIOD_SECONDS, tf_publisher.on_state_change)
except ImportError:
    node = None

if node is not None:
    print(
        f"Visualize in RViz with Fixed Frame '{world.root.name}' and a MarkerArray "
        f"display on '{visualization.topic_name}', its Durability Policy set to "
        f"Transient Local."
    )
    # Both publishers only publish on a world change, so the process has to outlive
    # this script's last line for RViz to ever receive them.
    rclpy.spin(node)
'''