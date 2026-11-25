"""
Tests of the interaction graph the live bridge serves: the selected objects of a task and
the spatial relations between them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cramera.body_geometry import POSE_PRECISION
from cramera.live.bridge import Bridge
from cramera.live.http import serve
from cramera.live.interaction_graph import INTERACTION_OBJECT_GROUP
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from llm_planner.env_reasoner.reasoner_graphs.Interactive_graph import (
    InteractionGraph,
    InteractionNode,
    SpatialRelation,
)

from .test_live_http import get_json

MILK_BODY_NAME = "milk.stl"
"""
An annotated apartment object.
"""

DRAWER_BODY_NAME = "cabinet10_drawer_top"
"""
Another annotated apartment object.
"""


@dataclass
class AlwaysHolds(SpatialRelation):
    """
    A relation that holds between any two objects, so a graph gets edges without
    depending on where the objects stand.
    """

    def holds(self, source: InteractionNode, target: InteractionNode) -> bool:
        return True


def apartment_graph(world, relations) -> InteractionGraph:
    """
    :param world: The apartment world.
    :param relations: The relations to check.
    :return: The graph of the milk and the drawer, with the edges ``relations`` give.
    """
    graph = InteractionGraph.from_world(
        world,
        [
            world.get_body_by_name(MILK_BODY_NAME),
            world.get_body_by_name(DRAWER_BODY_NAME),
        ],
        relations,
    )
    graph.add_relation_edges()
    return graph


@pytest.fixture()
def graph_without_edges(apartment_world_copy) -> InteractionGraph:
    return apartment_graph(apartment_world_copy, [])


@pytest.fixture()
def graph_with_edges(apartment_world_copy) -> InteractionGraph:
    return apartment_graph(apartment_world_copy, [AlwaysHolds()])


def published(graph: InteractionGraph) -> dict:
    """
    :param graph: The graph to publish.
    :return: What a bridge serves once ``graph`` is published on it.
    """
    bridge = Bridge()
    bridge.publish_interaction_graph(graph)
    return bridge.get_interaction_graph()


# %% what the bridge serves


class TestPublishedInteractionGraph:
    def test_nothing_is_published_before_a_graph_is(self):
        assert Bridge().get_interaction_graph() == {
            "signature": "",
            "nodes": [],
            "edges": [],
        }

    def test_every_object_is_a_node_named_after_it(self, graph_without_edges):
        nodes = published(graph_without_edges)["nodes"]

        assert [node["id"] for node in nodes] == [
            node.name for node in graph_without_edges.nodes
        ]
        assert [node["label"] for node in nodes] == [
            node.name for node in graph_without_edges.nodes
        ]

    def test_a_node_carries_the_description_of_its_object(self, graph_without_edges):
        nodes = published(graph_without_edges)["nodes"]

        assert [node["lines"] for node in nodes] == [
            node.description() for node in graph_without_edges.nodes
        ]

    def test_a_node_is_placed_where_its_object_stands(self, graph_without_edges):
        nodes = published(graph_without_edges)["nodes"]

        assert [node["position"] for node in nodes] == [
            [round(value, POSE_PRECISION) for value in node.position().to_np()[:3]]
            for node in graph_without_edges.nodes
        ]

    def test_every_node_is_drawn_as_an_object(self, graph_without_edges):
        nodes = published(graph_without_edges)["nodes"]

        assert {node["group"] for node in nodes} == {INTERACTION_OBJECT_GROUP}

    def test_every_relation_is_an_edge_labelled_with_its_name(self, graph_with_edges):
        edges = published(graph_with_edges)["edges"]

        assert edges == [
            {
                "from": edge.source.name,
                "to": edge.target.name,
                "label": type(edge.relation).__name__,
            }
            for edge in graph_with_edges.edges
        ]

    def test_publishing_again_replaces_the_graph(
        self, graph_with_edges, graph_without_edges
    ):
        bridge = Bridge()
        bridge.publish_interaction_graph(graph_with_edges)

        bridge.publish_interaction_graph(graph_without_edges)

        assert bridge.get_interaction_graph()["edges"] == []


# %% when the viewer rebuilds the drawing


class TestSignature:
    def test_the_same_graph_has_the_same_signature(self, graph_with_edges):
        assert (
            published(graph_with_edges)["signature"]
            == published(graph_with_edges)["signature"]
        )

    def test_a_new_edge_changes_the_signature(
        self, graph_with_edges, graph_without_edges
    ):
        assert (
            published(graph_with_edges)["signature"]
            != published(graph_without_edges)["signature"]
        )

    def test_moving_an_object_keeps_the_signature(
        self, apartment_world_copy, graph_with_edges
    ):
        before = published(graph_with_edges)
        milk = apartment_world_copy.get_body_by_name(MILK_BODY_NAME)
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            1.0, 1.0, 1.0, reference_frame=apartment_world_copy.root
        )

        after = published(graph_with_edges)

        assert after["nodes"] != before["nodes"]
        assert after["signature"] == before["signature"]


# %% the endpoint


@pytest.fixture()
def bridge_with_graph(graph_with_edges) -> Bridge:
    bridge = Bridge()
    bridge.publish_interaction_graph(graph_with_edges)
    return bridge


def test_the_endpoint_serves_the_published_graph(bridge_with_graph):
    httpd = serve(bridge_with_graph, 0)
    try:
        served = get_json("http://localhost:%d/interaction" % httpd.server_address[1])
    finally:
        httpd.shutdown()

    assert served == bridge_with_graph.get_interaction_graph()
