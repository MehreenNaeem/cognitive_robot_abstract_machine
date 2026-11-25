"""
Tests for :mod:`llm_planner.demo_utilities`: showing the interaction graph of selected
bodies in the CRAM viewer.
"""

from __future__ import annotations

import os
import shutil

import pytest

from cramera.live.bridge import Bridge
from cramera.live.interaction_graph import InteractionGraphSnapshot
from llm_planner.demo_utilities import (
    MultipleObjectBodies,
    NoObjectBody,
    ObjectModelParser,
    SceneInteractionDisplay,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from llm_planner.env_reasoner.reasoner_graphs.Interactive_graph import (
    InteractionGraph,
    default_edge_removal_rules,
    default_relations,
)

MILK_BODY_NAME = "milk.stl"
"""
An annotated apartment object.
"""

DRAWER_BODY_NAME = "cabinet10_drawer_top"
"""
Another annotated apartment object.
"""


@pytest.fixture()
def display(apartment_world_copy) -> SceneInteractionDisplay:
    return SceneInteractionDisplay(world=apartment_world_copy, bridge=Bridge())


def selected_bodies(world) -> list:
    """
    :return: The milk, the drawer and the world root, which has no annotation.
    """
    return [
        world.get_body_by_name(MILK_BODY_NAME),
        world.get_body_by_name(DRAWER_BODY_NAME),
        world.root,
    ]


# %% building the graph


class TestGraphOfBodies:
    def test_the_annotated_bodies_are_the_nodes(
        self, display, apartment_world_copy
    ) -> None:
        graph = display.graph_of(selected_bodies(apartment_world_copy))

        assert [node.name for node in graph.nodes] == [MILK_BODY_NAME, DRAWER_BODY_NAME]

    def test_the_edges_are_the_default_relations_after_the_removal_rules(
        self, display, apartment_world_copy
    ) -> None:
        bodies = selected_bodies(apartment_world_copy)
        expected = InteractionGraph.from_selected_bodies(
            apartment_world_copy,
            bodies,
            default_relations(),
            default_edge_removal_rules(),
        )

        graph = display.graph_of(bodies)

        assert [
            (edge.source.name, edge.target.name, type(edge.relation))
            for edge in graph.edges
        ] == [
            (edge.source.name, edge.target.name, type(edge.relation))
            for edge in expected.edges
        ]


# %% showing the graph


class TestShownGraph:
    def test_showing_bodies_publishes_their_graph(
        self, display, apartment_world_copy
    ) -> None:
        graph = display.show(selected_bodies(apartment_world_copy))

        assert (
            display.bridge.get_interaction_graph()
            == InteractionGraphSnapshot.from_graph(graph).to_payload()
        )

    def test_showing_a_graph_publishes_it(self, display, apartment_world_copy) -> None:
        graph = InteractionGraph.from_world(
            apartment_world_copy,
            [apartment_world_copy.get_body_by_name(MILK_BODY_NAME)],
            relations=[],
        )

        display.show_graph(graph)

        assert (
            display.bridge.get_interaction_graph()
            == InteractionGraphSnapshot.from_graph(graph).to_payload()
        )


# %% object models named after their folder

OBJECT_MODEL = os.path.join(os.path.dirname(__file__), "data", "cup_1", "model.xml")
"""
A MuJoCo object model: an unnamed body holding the body ``object`` with the geometry.
"""

MODEL_WITH_TWO_OBJECT_BODIES = os.path.join(
    os.path.dirname(__file__), "data", "cup_with_lid", "model.xml"
)
"""
A MuJoCo model in which two bodies, a cup and its lid, carry geometry.
"""

MODEL_WITHOUT_GEOMETRY = os.path.join(
    os.path.dirname(__file__), "data", "empty_model", "model.xml"
)
"""
A MuJoCo model in which no body carries geometry.
"""

OBJECT_FOLDER_NAME = "cup_1"
"""
The folder :data:`OBJECT_MODEL` lies in.
"""


def object_model_in(folder) -> str:
    """
    :param folder: The folder to copy :data:`OBJECT_MODEL`'s folder to.
    :return: The path of the copied model.
    """
    shutil.copytree(os.path.dirname(OBJECT_MODEL), folder)
    return os.path.join(folder, os.path.basename(OBJECT_MODEL))


def merged(*objects: World) -> World:
    """
    :return: One world with each of ``objects`` fixed to its root, side by side.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
    for index, object_world in enumerate(objects):
        with world.modify_world():
            world.merge_world(
                object_world,
                FixedConnection(
                    parent=root,
                    child=object_world.root,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=float(index)
                    ),
                ),
            )
    return world


class TestObjectModelParser:
    def test_the_object_is_named_after_its_folder(self) -> None:
        parser = ObjectModelParser(OBJECT_MODEL)

        assert parser.object_name == OBJECT_FOLDER_NAME

    def test_the_body_with_the_geometry_carries_the_object_name(self) -> None:
        world = ObjectModelParser(OBJECT_MODEL).parse()

        body = world.get_body_by_name(OBJECT_FOLDER_NAME)

        assert body.collision.shapes
        assert body.visual.shapes

    def test_no_body_keeps_a_name_of_the_model_file(self) -> None:
        world = ObjectModelParser(OBJECT_MODEL).parse()

        assert all(
            body.name.name.startswith(OBJECT_FOLDER_NAME) for body in world.bodies
        )

    def test_two_objects_share_no_body_name(self, tmp_path) -> None:
        other = object_model_in(tmp_path / "cup_2")

        world = merged(
            ObjectModelParser(OBJECT_MODEL).parse(), ObjectModelParser(other).parse()
        )

        names = [body.name.name for body in world.bodies]
        assert len(names) == len(set(names))
        assert world.get_body_by_name("cup_2").collision.shapes

    def test_a_model_with_several_bodies_with_geometry_is_refused(self) -> None:
        with pytest.raises(MultipleObjectBodies):
            ObjectModelParser(MODEL_WITH_TWO_OBJECT_BODIES).parse()

    def test_a_model_without_geometry_is_refused(self) -> None:
        with pytest.raises(NoObjectBody):
            ObjectModelParser(MODEL_WITHOUT_GEOMETRY).parse()
