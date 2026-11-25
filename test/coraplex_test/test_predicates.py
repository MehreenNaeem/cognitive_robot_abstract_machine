"""
Tests of the predicates that read the gripper's occupancy from the kinematic
structure.
"""

from __future__ import annotations

import pytest

from coraplex.datastructures.enums import Arms
from coraplex.querying.predicates import GripperIsFree, GripperIsNotFree
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Tuple

HELD_BODY_NAME = "held_body"
"""
The name of the body the tests hang below the tool frame to occupy the gripper.
"""


@pytest.fixture
def world_with_left_end_effector(
    apartment_world_pr2_copy_with_context,
) -> Tuple[World, EndEffector]:
    """
    The apartment robot and the view of its left gripper, which holds nothing.
    """
    world, robot, _ = apartment_world_pr2_copy_with_context
    return world, ViewManager.get_end_effector_view(Arms.LEFT, robot)


def hang_body_below(world: World, end_effector: EndEffector) -> Body:
    """
    Attach a body to the gripper's tool frame, so the gripper counts as occupied.

    :param world: The world the gripper belongs to.
    :param end_effector: The gripper to occupy.
    :return: The attached body.
    """
    held_body = Body(name=PrefixedName(HELD_BODY_NAME))
    with world.modify_world():
        world.add_connection(
            FixedConnection(parent=end_effector.tool_frame, child=held_body)
        )
    return held_body


# %% reading the occupancy


class TestGripperOccupancy:
    def test_an_empty_gripper_is_free(self, world_with_left_end_effector) -> None:
        """
        The tool frame is itself part of the branch below it and does not occupy the
        gripper.
        """
        _, end_effector = world_with_left_end_effector

        assert GripperIsFree(end_effector)() is True
        assert GripperIsNotFree(end_effector)() is False

    def test_a_gripper_holding_a_body_is_not_free(
        self, world_with_left_end_effector
    ) -> None:
        world, end_effector = world_with_left_end_effector
        hang_body_below(world, end_effector)

        assert GripperIsFree(end_effector)() is False
        assert GripperIsNotFree(end_effector)() is True


# %% leaving the world as it was found


class TestBranchQueryIsUnchanged:
    def test_checking_occupancy_leaves_the_branch_query_unchanged(
        self, world_with_left_end_effector
    ) -> None:
        """
        The branch below the tool frame is cached and handed to its readers by
        reference, so a reader that drops the tool frame from it drops it for every
        later reader too.
        """
        world, end_effector = world_with_left_end_effector
        branch_before = list(
            world.get_kinematic_structure_entities_of_branch(end_effector.tool_frame)
        )

        GripperIsFree(end_effector)()

        assert (
            list(
                world.get_kinematic_structure_entities_of_branch(
                    end_effector.tool_frame
                )
            )
            == branch_before
        )

    def test_checking_occupancy_repeatedly_leaves_the_branch_query_unchanged(
        self, world_with_left_end_effector
    ) -> None:
        world, end_effector = world_with_left_end_effector
        held_body = hang_body_below(world, end_effector)
        branch_before = list(
            world.get_kinematic_structure_entities_of_branch(end_effector.tool_frame)
        )

        GripperIsFree(end_effector)()
        GripperIsNotFree(end_effector)()

        branch_after = list(
            world.get_kinematic_structure_entities_of_branch(end_effector.tool_frame)
        )
        assert branch_after == branch_before
        assert held_body in branch_after
