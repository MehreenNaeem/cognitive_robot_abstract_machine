"""
Tests of the LLM planner's tree: a task, the actions it was grounded into and the plan
steps each of those performs.
"""

from __future__ import annotations

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.plans.plan_node import TaskStatus
from semantic_digital_twin.datastructures.definitions import TorsoState

from cramera.knowledge.enums import PlanNodeGroup
from cramera.live.bridge import Bridge, TaskStatusName
from cramera.live.llm_plan import TASK_GROUP, TASK_NODE_ID

from .test_live_bridge import shaped_body, world_with

TASK = "Put the bowl in the drawer"
"""
The task statement the tree is rooted at.
"""


@pytest.fixture()
def context() -> Context:
    """
    A context holding a world with nothing but a body in it, enough to build a plan.
    """
    return Context(world=world_with(shaped_body("world", "bowl.stl")), robot=None)


def two_step_plan(context: Context):
    """
    A plan performing two actions, as a grounded triplet's plan does.

    :param context: The context the plan belongs to.
    """
    return sequential(
        [ParkArmsAction(Arms.BOTH), MoveTorsoAction(TorsoState.HIGH)], context=context
    ).plan


def nodes_by_id(bridge: Bridge) -> dict:
    return {node["id"]: node for node in bridge.get_llm_plan()["nodes"]}


def labels_under(bridge: Bridge, parent: str) -> list:
    return [
        node["label"]
        for node in bridge.get_llm_plan()["nodes"]
        if node["parent"] == parent
    ]


# %% the shape of the tree


class TestPublishedTree:
    def test_nothing_is_published_before_a_task_is(self):
        assert Bridge().get_llm_plan() == {"task": "", "signature": "", "nodes": []}

    def test_the_task_is_the_root_of_the_tree(self, context):
        bridge = Bridge()

        bridge.publish_llm_plan_step(TASK, "PickUpAction", two_step_plan(context))

        [root] = [
            node for node in bridge.get_llm_plan()["nodes"] if node["parent"] is None
        ]
        assert root["id"] == TASK_NODE_ID
        assert root["label"] == TASK
        assert root["group"] == TASK_GROUP
        assert bridge.get_llm_plan()["task"] == TASK

    def test_a_grounded_action_hangs_off_the_task(self, context):
        bridge = Bridge()

        bridge.publish_llm_plan_step(TASK, "PickUpAction", two_step_plan(context))

        assert labels_under(bridge, TASK_NODE_ID) == ["PickUpAction"]

    def test_the_plans_actions_hang_off_the_action_they_perform(self, context):
        bridge = Bridge()

        bridge.publish_llm_plan_step(TASK, "PickUpAction", two_step_plan(context))

        [step] = [
            node
            for node in bridge.get_llm_plan()["nodes"]
            if node["parent"] == TASK_NODE_ID
        ]
        assert labels_under(bridge, step["id"]) == [
            ParkArmsAction.__name__,
            MoveTorsoAction.__name__,
        ]

    def test_every_step_of_a_task_is_kept(self, context):
        """
        A task is grounded and performed step by step, so the tree grows rather than
        being replaced.
        """
        bridge = Bridge()

        bridge.publish_llm_plan_step(TASK, "OpenAction", two_step_plan(context))
        bridge.publish_llm_plan_step(TASK, "PickUpAction", two_step_plan(context))

        assert labels_under(bridge, TASK_NODE_ID) == ["OpenAction", "PickUpAction"]

    def test_a_new_task_starts_a_new_tree(self, context):
        bridge = Bridge()
        bridge.publish_llm_plan_step(TASK, "OpenAction", two_step_plan(context))

        bridge.publish_llm_plan_step("Open the cabinet", "OpenAction", two_step_plan(context))

        assert bridge.get_llm_plan()["task"] == "Open the cabinet"
        assert labels_under(bridge, TASK_NODE_ID) == ["OpenAction"]

    def test_the_signature_changes_only_with_the_tree(self, context):
        """
        The viewer rebuilds the tree when the signature changes, so a status update must
        leave it alone.
        """
        bridge = Bridge()
        bridge.publish_llm_plan_step(TASK, "OpenAction", two_step_plan(context))
        after_first_step = bridge.get_llm_plan()["signature"]

        bridge.snapshot_llm_plan()
        assert bridge.get_llm_plan()["signature"] == after_first_step

        bridge.publish_llm_plan_step(TASK, "PickUpAction", two_step_plan(context))
        assert bridge.get_llm_plan()["signature"] != after_first_step

    def test_an_action_is_coloured_as_the_plan_tab_colours_it(self, context):
        bridge = Bridge()

        bridge.publish_llm_plan_step(TASK, "PickUpAction", two_step_plan(context))

        groups = {
            node["group"]
            for node in bridge.get_llm_plan()["nodes"]
            if node["parent"] not in (None, TASK_NODE_ID)
        }
        assert groups == {PlanNodeGroup.ACTION.value}


# %% what the tree says about progress


class TestPublishedStatus:
    def test_a_task_nothing_has_run_of_is_created(self, context):
        bridge = Bridge()

        bridge.publish_llm_plan_step(TASK, "PickUpAction", two_step_plan(context))

        assert nodes_by_id(bridge)[TASK_NODE_ID]["status"] == TaskStatusName.CREATED

    def test_an_action_that_is_running_is_reported_as_running(self, context):
        bridge = Bridge()
        plan = two_step_plan(context)
        bridge.publish_llm_plan_step(TASK, "PickUpAction", plan)
        park_arms = plan.get_nodes_by_designator_type(ParkArmsAction)[0]

        park_arms.status = TaskStatus.RUNNING
        bridge.snapshot_llm_plan()

        published = nodes_by_id(bridge)
        step_id = [
            node["id"]
            for node in published.values()
            if node["parent"] == TASK_NODE_ID
        ][0]
        assert published[f"{step_id}_plan_node_{id(park_arms)}"]["status"] == (
            TaskStatusName.RUNNING
        )

    def test_a_running_step_makes_the_task_run(self, context):
        bridge = Bridge()
        plan = two_step_plan(context)
        bridge.publish_llm_plan_step(TASK, "PickUpAction", plan)

        plan.get_nodes_by_designator_type(ParkArmsAction)[0].status = TaskStatus.RUNNING
        bridge.snapshot_llm_plan()

        assert nodes_by_id(bridge)[TASK_NODE_ID]["status"] == TaskStatusName.RUNNING

    def test_a_task_whose_steps_are_only_partly_done_is_still_running(self, context):
        """
        One finished step out of two is progress, not a finished task.
        """
        bridge = Bridge()
        done = two_step_plan(context)
        bridge.publish_llm_plan_step(TASK, "OpenAction", done)
        bridge.publish_llm_plan_step(TASK, "PickUpAction", two_step_plan(context))

        done.root.status = TaskStatus.SUCCEEDED
        bridge.snapshot_llm_plan()

        assert nodes_by_id(bridge)[TASK_NODE_ID]["status"] == TaskStatusName.RUNNING

    def test_a_task_every_step_of_which_succeeded_is_done(self, context):
        bridge = Bridge()
        first, second = two_step_plan(context), two_step_plan(context)
        bridge.publish_llm_plan_step(TASK, "OpenAction", first)
        bridge.publish_llm_plan_step(TASK, "PickUpAction", second)

        first.root.status = TaskStatus.SUCCEEDED
        second.root.status = TaskStatus.SUCCEEDED
        bridge.snapshot_llm_plan()

        assert nodes_by_id(bridge)[TASK_NODE_ID]["status"] == TaskStatusName.SUCCEEDED
