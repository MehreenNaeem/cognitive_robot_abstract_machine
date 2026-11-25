"""
Tests of the action feasibility tree the live bridge serves: which actions can be
performed on which objects, with which arm and from which approach direction.
"""

from __future__ import annotations

import pytest
import rustworkx

from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from cramera.live.bridge import Bridge
from cramera.live.http import serve
from llm_planner.env_reasoner.reasoner_graphs.AFG import (
    ConditionParameter,
    ConditionValues,
    feasibility_tree,
)

from .test_live_http import get_json

MILK_BODY_NAME = "milk.stl"
"""
The object that can be picked up.
"""

DRAWER_BODY_NAME = "cabinet10_drawer_top"
"""
The object that can be opened.
"""


def condition_values(
    arm: Arms, approach_direction: ApproachDirection
) -> ConditionValues:
    """
    :return: The condition values of one arm and approach direction.
    """
    return {
        ConditionParameter.ARM: arm,
        ConditionParameter.GRASP_DESCRIPTION: GraspDescription(
            approach_direction, VerticalAlignment.NoAlignment, None
        ),
    }


@pytest.fixture()
def tree() -> rustworkx.PyDiGraph:
    """
    The milk picked up with the right arm from two directions, and the drawer opened
    with the left arm from one.
    """
    return feasibility_tree(
        {
            PickUpAction: {
                MILK_BODY_NAME: {
                    Arms.RIGHT: [
                        condition_values(Arms.RIGHT, ApproachDirection.FRONT),
                        condition_values(Arms.RIGHT, ApproachDirection.BACK),
                    ],
                    Arms.LEFT: [],
                }
            },
            OpenAction: {
                DRAWER_BODY_NAME: {
                    Arms.RIGHT: [],
                    Arms.LEFT: [condition_values(Arms.LEFT, ApproachDirection.FRONT)],
                }
            },
        }
    )


@pytest.fixture()
def smaller_tree() -> rustworkx.PyDiGraph:
    """
    Only the drawer opened with the left arm.
    """
    return feasibility_tree(
        {
            OpenAction: {
                DRAWER_BODY_NAME: {
                    Arms.LEFT: [condition_values(Arms.LEFT, ApproachDirection.FRONT)]
                }
            }
        }
    )


def published(tree: rustworkx.PyDiGraph) -> dict:
    """
    :return: What a bridge serves once ``tree`` is published on it.
    """
    bridge = Bridge()
    bridge.publish_feasibility_tree(tree)
    return bridge.get_feasibility_tree()


# %% what the bridge serves


class TestPublishedFeasibilityTree:
    def test_nothing_is_published_before_a_tree_is(self):
        assert Bridge().get_feasibility_tree() == {
            "signature": "",
            "nodes": [],
            "edges": [],
        }

    def test_every_node_is_labelled_with_its_caption(self, tree):
        nodes = published(tree)["nodes"]

        assert [node["label"] for node in nodes] == [
            node.caption for node in tree.nodes()
        ]

    def test_a_node_is_grouped_by_what_it_stands_for(self, tree):
        nodes = published(tree)["nodes"]

        assert [node["group"] for node in nodes] == [
            node.kind.value for node in tree.nodes()
        ]

    def test_a_node_shows_its_details_on_click(self, tree):
        nodes = published(tree)["nodes"]

        assert [node["lines"] for node in nodes] == [
            list(node.details) for node in tree.nodes()
        ]

    def test_every_edge_of_the_tree_is_an_arrow(self, tree):
        payload = published(tree)
        caption_by_id = {node["id"]: node["label"] for node in payload["nodes"]}

        drawn = [
            (caption_by_id[edge["from"]], caption_by_id[edge["to"]])
            for edge in payload["edges"]
        ]

        assert drawn == [
            (tree[parent].caption, tree[child].caption)
            for parent, child in tree.edge_list()
        ]

    def test_publishing_again_replaces_the_tree(self, tree, smaller_tree):
        bridge = Bridge()
        bridge.publish_feasibility_tree(tree)

        bridge.publish_feasibility_tree(smaller_tree)

        assert bridge.get_feasibility_tree() == published(smaller_tree)


# %% when the viewer rebuilds the drawing


class TestSignature:
    def test_the_same_tree_has_the_same_signature(self, tree):
        assert published(tree)["signature"] == published(tree)["signature"]

    def test_another_tree_has_another_signature(self, tree, smaller_tree):
        assert published(tree)["signature"] != published(smaller_tree)["signature"]


# %% the endpoint


def test_the_endpoint_serves_the_published_tree(tree):
    bridge = Bridge()
    bridge.publish_feasibility_tree(tree)
    httpd = serve(bridge, 0)
    try:
        served = get_json("http://localhost:%d/feasibility" % httpd.server_address[1])
    finally:
        httpd.shutdown()

    assert served == bridge.get_feasibility_tree()
