"""
Tests of the goal tree the live bridge serves: the ``AND``/``OR`` tree of a task's goal
state.
"""

from __future__ import annotations

import pytest

from cramera.live.bridge import Bridge
from cramera.live.http import serve
from llm_planner.env_reasoner.reasoner_graphs.goal_state import (
    GoalAtom,
    GoalConjunction,
    GoalDefinition,
    GoalDisjunction,
    GoalExpression,
)
from semantic_digital_twin.reasoning.predicates import InsideOf, contact

from .test_live_http import get_json

SPOON_BODY_NAME = "spoon.stl"
"""
The body the goal places.
"""

DRAWER_BODY_NAMES = ["drawer_top1", "drawer_bottom1"]
"""
The bodies the spoon may end up in.
"""


@pytest.fixture()
def goal() -> GoalDefinition:
    """
    The spoon inside one of two drawers, touching the first.
    """
    return GoalDefinition(
        GoalConjunction(
            [
                GoalDisjunction(
                    [
                        GoalAtom(InsideOf.__name__, SPOON_BODY_NAME, name)
                        for name in DRAWER_BODY_NAMES
                    ]
                ),
                GoalAtom(contact.__name__, SPOON_BODY_NAME, DRAWER_BODY_NAMES[0]),
            ]
        )
    )


def expressions_of(goal: GoalDefinition) -> list[GoalExpression]:
    """
    :return: Every expression of ``goal``, in the order its tree holds them.
    """
    return goal.tree().nodes()


def in_pddl(arrow: tuple[GoalExpression, GoalExpression]) -> tuple[str, str]:
    """
    :return: ``arrow`` with both expressions written in PDDL, to order arrows by.
    """
    connective, operand = arrow
    return connective.to_pddl(), operand.to_pddl()


def published(goal: GoalDefinition) -> dict:
    """
    :return: What a bridge serves once ``goal`` is published on it.
    """
    bridge = Bridge()
    bridge.publish_goal(goal)
    return bridge.get_goal_tree()


# %% what the bridge serves


class TestPublishedGoalTree:
    def test_nothing_is_published_before_a_goal_is(self):
        assert Bridge().get_goal_tree() == {
            "signature": "",
            "goal": "",
            "nodes": [],
            "edges": [],
        }

    def test_the_goal_is_served_in_pddl(self, goal):
        assert published(goal)["goal"] == goal.to_pddl()

    def test_every_expression_is_a_node_showing_its_whole_caption(self, goal):
        nodes = published(goal)["nodes"]

        assert [node["label"] for node in nodes] == [
            expression.caption() for expression in expressions_of(goal)
        ]

    def test_a_node_is_grouped_by_the_kind_of_its_expression(self, goal):
        nodes = published(goal)["nodes"]

        assert [node["group"] for node in nodes] == [
            expression.kind for expression in expressions_of(goal)
        ]

    def test_a_node_carries_nothing_to_show_on_click(self, goal):
        nodes = published(goal)["nodes"]

        assert {tuple(sorted(node)) for node in nodes} == {("group", "id", "label")}

    def test_every_operand_is_an_unlabelled_edge_from_its_connective(self, goal):
        payload = published(goal)
        expression_by_id = dict(
            zip([node["id"] for node in payload["nodes"]], expressions_of(goal))
        )

        drawn = [
            (expression_by_id[edge["from"]], expression_by_id[edge["to"]])
            for edge in payload["edges"]
        ]

        assert {tuple(sorted(edge)) for edge in payload["edges"]} == {("from", "to")}
        assert sorted(drawn, key=in_pddl) == sorted(
            [
                (expression, child)
                for expression in expressions_of(goal)
                for child in expression.children()
            ],
            key=in_pddl,
        )

    def test_publishing_again_replaces_the_goal(self, goal):
        bridge = Bridge()
        bridge.publish_goal(goal)
        smaller = GoalDefinition(GoalConjunction([goal.condition.operands[1]]))

        bridge.publish_goal(smaller)

        assert bridge.get_goal_tree()["goal"] == smaller.to_pddl()


# %% when the viewer rebuilds the drawing


class TestSignature:
    def test_the_same_goal_has_the_same_signature(self, goal):
        assert published(goal)["signature"] == published(goal)["signature"]

    def test_another_goal_has_another_signature(self, goal):
        smaller = GoalDefinition(GoalConjunction([goal.condition.operands[1]]))

        assert published(goal)["signature"] != published(smaller)["signature"]


# %% the endpoint


def test_the_endpoint_serves_the_published_goal(goal):
    bridge = Bridge()
    bridge.publish_goal(goal)
    httpd = serve(bridge, 0)
    try:
        served = get_json("http://localhost:%d/goal" % httpd.server_address[1])
    finally:
        httpd.shutdown()

    assert served == bridge.get_goal_tree()
