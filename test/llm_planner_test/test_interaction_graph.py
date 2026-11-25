"""
Tests for :mod:`llm_planner.env_reasoner.reasoner_graphs.Interactive_graph`.
"""

from __future__ import annotations

import pytest

from llm_planner.env_reasoner.reasoner_graphs.Interactive_graph import (
    DEFAULT_NEAR_THRESHOLD,
    MINIMUM_CONTAINMENT_RATIO,
    AmbiguousBodyAnnotation,
    Containment,
    ContainmentExcludesSupport,
    Contact,
    DuplicateObjectName,
    InteractionGraph,
    InteractionGraphVisualizer,
    InteractionNode,
    Near,
    SpatialRelation,
    Support,
    UnannotatedBody,
    UnknownObjectName,
    default_edge_removal_rules,
    default_relations,
)
from llm_planner.env_reasoner.object_action_affordance import (
    doable_actions_for,
    get_object_action_list,
)
from llm_planner.env_reasoner.object_constraints import ObjectAffordanceReasoner
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Drawer,
    Milk,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CounterTop,
    Spoon,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

MILK_BODY_NAME = "milk.stl"
"""
The body of the milk the apartment fixture annotates.
"""

DRAWER_BODY_NAME = "cabinet10_drawer_top"
"""
The body of the drawer the apartment fixture annotates.
"""


class AlwaysHolds(SpatialRelation):
    """
    A relation that holds between every ordered pair of objects.
    """

    def holds(self, source: InteractionNode, target: InteractionNode) -> bool:
        return True


BOX_SIZE = 1.0
"""
Edge length of a cube made by :func:`box_body` unless another size is given.
"""

OVERLAPPING_OFFSET = 0.9
"""
Distance between neighbouring cube centres, less than :data:`BOX_SIZE` so neighbours
overlap and the outer cubes stay apart.
"""


def box_body(name: str, offset: float = 0.0, size: float = BOX_SIZE) -> Body:
    body = Body(name=PrefixedName(name))
    body.collision = ShapeCollection(
        [
            Box(
                scale=Scale(size, size, size),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    offset, 0, 0, 0, 0, 0, reference_frame=body
                ),
            )
        ]
    )
    return body


@pytest.fixture()
def boxes_in_a_row() -> InteractionGraph:
    """
    A milk, an apple and a spoon as cubes along the x axis: the milk touches the apple,
    the apple touches the spoon, and the milk and the spoon are apart.
    """
    world = World()
    milk = box_body("milk")
    apple = box_body("apple", OVERLAPPING_OFFSET)
    spoon = box_body("spoon", 2 * OVERLAPPING_OFFSET)
    with world.modify_world():
        for body in (milk, apple, spoon):
            world.add_kinematic_structure_entity(body)
        world.add_connection(
            Connection6DoF.create_with_dofs(parent=milk, child=apple, world=world)
        )
        world.add_connection(
            Connection6DoF.create_with_dofs(parent=apple, child=spoon, world=world)
        )
        world.add_semantic_annotations(
            [Milk(root=milk), Apple(root=apple), Spoon(root=spoon)]
        )
    return InteractionGraph.from_world(
        world, [milk, apple, spoon], relations=[Contact()]
    )


def fix_at(
    world: World, parent: Body, child: Body, x: float = 0.0, z: float = 0.0
) -> None:
    """
    Fix ``child`` to ``parent`` at the given offset; call inside ``world.modify_world()``.
    """
    world.add_connection(
        FixedConnection(
            parent=parent,
            child=child,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=x, z=z, reference_frame=parent
            ),
        )
    )


FLOATING_HEIGHT = 3.0
"""
Height of the apple's cube centre above the counter top's in :func:`stacked_boxes`,
far enough that the two do not touch.
"""


@pytest.fixture()
def stacked_boxes() -> InteractionGraph:
    """
    A milk cube resting on a counter top cube, and an apple cube floating above the
    counter top.
    """
    world = World()
    counter_top = box_body("counter_top")
    milk = box_body("milk")
    apple = box_body("apple")
    with world.modify_world():
        fix_at(world, counter_top, milk, z=BOX_SIZE)
        fix_at(world, counter_top, apple, z=FLOATING_HEIGHT)
        world.add_semantic_annotations(
            [CounterTop(root=counter_top), Milk(root=milk), Apple(root=apple)]
        )
    return InteractionGraph.from_world(
        world, [counter_top, milk, apple], relations=[Support()]
    )


NEAR_THRESHOLD = 1.5
"""
Planar distance below which :func:`spread_boxes` checks objects are near.
"""

NEAR_OFFSET = 1.0
"""
Planar distance of the apple from the milk in :func:`spread_boxes`, below
:data:`NEAR_THRESHOLD`.
"""

FAR_OFFSET = 3.0
"""
Planar distance of the spoon from the milk in :func:`spread_boxes`, which puts the
spoon beyond :data:`NEAR_THRESHOLD` from both the milk and the apple.
"""


@pytest.fixture()
def spread_boxes() -> InteractionGraph:
    """
    A milk cube, an apple cube :data:`NEAR_OFFSET` away from it and high above it, and
    a spoon cube :data:`FAR_OFFSET` away from it.
    """
    world = World()
    milk = box_body("milk")
    apple = box_body("apple")
    spoon = box_body("spoon")
    with world.modify_world():
        fix_at(world, milk, apple, x=NEAR_OFFSET, z=FLOATING_HEIGHT)
        fix_at(world, milk, spoon, x=FAR_OFFSET)
        world.add_semantic_annotations(
            [Milk(root=milk), Apple(root=apple), Spoon(root=spoon)]
        )
    return InteractionGraph.from_world(
        world, [milk, apple, spoon], relations=[Near(threshold=NEAR_THRESHOLD)]
    )


SMALL_BOX_SIZE = 0.2
"""
Edge length of the cubes :func:`boxes_in_a_drawer` puts in and around the drawer.
"""

HALF_INSIDE_OFFSET = BOX_SIZE / 2
"""
Offset of the apple's centre from the drawer's in :func:`boxes_in_a_drawer`, which
puts the apple's centre on the drawer's side face, half of it inside.
"""

BELOW_HALF_RATIO = 0.4
"""
A minimum containment ratio the half-inside apple of :func:`boxes_in_a_drawer` meets.
"""


@pytest.fixture()
def boxes_in_a_drawer() -> InteractionGraph:
    """
    A drawer cube with a spoon cube fully inside it, an apple cube half inside its
    side, and a milk cube outside it.
    """
    world = World()
    drawer = box_body("drawer")
    spoon = box_body("spoon", size=SMALL_BOX_SIZE)
    apple = box_body("apple", size=SMALL_BOX_SIZE)
    milk = box_body("milk", size=SMALL_BOX_SIZE)
    with world.modify_world():
        fix_at(world, drawer, spoon)
        fix_at(world, drawer, apple, x=HALF_INSIDE_OFFSET)
        fix_at(world, drawer, milk, x=FAR_OFFSET)
        world.add_semantic_annotations(
            [
                Drawer(root=drawer),
                Spoon(root=spoon),
                Apple(root=apple),
                Milk(root=milk),
            ]
        )
    return InteractionGraph.from_world(
        world, [drawer, spoon, apple, milk], relations=[Containment()]
    )


def bodies_of(world: World, *names: str) -> list[Body]:
    return [world.get_body_by_name(name) for name in names]


@pytest.fixture()
def apartment_graph(apartment_world_copy) -> InteractionGraph:
    return InteractionGraph.from_world(
        apartment_world_copy,
        bodies_of(apartment_world_copy, MILK_BODY_NAME, DRAWER_BODY_NAME),
        relations=[],
    )


# %% nodes from world objects


class TestObjectsFromWorld:
    def test_adds_one_node_per_requested_object(self, apartment_graph) -> None:
        assert sorted(node.name for node in apartment_graph.nodes) == sorted(
            [MILK_BODY_NAME, DRAWER_BODY_NAME]
        )

    def test_node_keeps_the_annotation_it_describes(
        self, apartment_world_copy, apartment_graph
    ) -> None:
        milk = apartment_world_copy.get_semantic_annotations_by_type(Milk)[0]
        assert apartment_graph.node(MILK_BODY_NAME).annotation is milk

    def test_node_affordances_come_from_the_reasoner(self, apartment_graph) -> None:
        expected = ObjectAffordanceReasoner().affordances_for(Drawer)
        assert apartment_graph.node(DRAWER_BODY_NAME).affordances == expected

    def test_node_properties_are_the_object_mixins(self, apartment_graph) -> None:
        expected = sorted(mixin.__name__ for mixin in doable_actions_for(Drawer).mixins)
        assert apartment_graph.node(DRAWER_BODY_NAME).properties == expected

    def test_node_actions_are_the_object_action_list(self, apartment_graph) -> None:
        assert apartment_graph.node(MILK_BODY_NAME).actions == get_object_action_list(
            Milk
        )

    def test_repeated_body_adds_its_object_once(self, apartment_world_copy) -> None:
        graph = InteractionGraph.from_world(
            apartment_world_copy,
            bodies_of(apartment_world_copy, MILK_BODY_NAME, MILK_BODY_NAME),
            relations=[],
        )
        assert [node.name for node in graph.nodes] == [MILK_BODY_NAME]

    def test_rejects_a_body_without_annotation(self, apartment_world_copy) -> None:
        with pytest.raises(UnannotatedBody):
            InteractionGraph.from_world(
                apartment_world_copy, [apartment_world_copy.root], relations=[]
            )

    def test_rejects_a_body_with_several_annotations(
        self, apartment_world_copy
    ) -> None:
        (milk_body,) = bodies_of(apartment_world_copy, MILK_BODY_NAME)
        with apartment_world_copy.modify_world():
            apartment_world_copy.add_semantic_annotation(Apple(root=milk_body))
        with pytest.raises(AmbiguousBodyAnnotation):
            InteractionGraph.from_world(apartment_world_copy, [milk_body], relations=[])

    def test_annotated_bodies_leave_out_a_body_without_annotation(
        self, apartment_world_copy
    ) -> None:
        graph = InteractionGraph.from_annotated_bodies(
            apartment_world_copy,
            bodies_of(apartment_world_copy, MILK_BODY_NAME)
            + [apartment_world_copy.root],
            relations=[],
        )
        assert [node.name for node in graph.nodes] == [MILK_BODY_NAME]


# %% where an object stands


class TestPosition:
    def test_an_object_stands_at_the_centre_of_its_geometry(
        self, boxes_in_a_row
    ) -> None:
        position = boxes_in_a_row.node("apple").position()
        assert position.to_np()[:3].tolist() == pytest.approx(
            [OVERLAPPING_OFFSET, 0, 0]
        )

    def test_an_object_without_geometry_stands_at_its_origin(self) -> None:
        world = World()
        milk = Body(name=PrefixedName("milk"))
        with world.modify_world():
            world.add_kinematic_structure_entity(milk)
            world.add_semantic_annotation(Milk(root=milk))
        graph = InteractionGraph.from_world(world, [milk], relations=[])

        assert graph.node("milk").position().to_np()[:3].tolist() == pytest.approx(
            milk.global_pose.to_position().to_np()[:3].tolist()
        )


# %% object names


class TestObjectNames:
    def test_rejects_a_name_already_in_the_graph(
        self, apartment_world_copy, apartment_graph
    ) -> None:
        milk = apartment_world_copy.get_semantic_annotations_by_type(Milk)[0]
        with pytest.raises(DuplicateObjectName):
            apartment_graph.add_annotation(milk)

    def test_unknown_name_is_rejected(self, apartment_graph) -> None:
        with pytest.raises(UnknownObjectName):
            apartment_graph.node("green_apple")


# %% relation edges


class TestRelationEdges:
    def test_added_edge_is_listed_between_its_objects(self, apartment_graph) -> None:
        support = Support()
        apartment_graph.add_edge(DRAWER_BODY_NAME, MILK_BODY_NAME, support)
        assert apartment_graph.relations_between(DRAWER_BODY_NAME, MILK_BODY_NAME) == [
            support
        ]

    def test_edges_are_directed(self, apartment_graph) -> None:
        apartment_graph.add_edge(DRAWER_BODY_NAME, MILK_BODY_NAME, Support())
        assert apartment_graph.relations_between(MILK_BODY_NAME, DRAWER_BODY_NAME) == []

    def test_add_relation_edges_connects_every_ordered_pair_that_holds(
        self, apartment_world_copy
    ) -> None:
        relation = AlwaysHolds()
        graph = InteractionGraph.from_world(
            apartment_world_copy,
            bodies_of(apartment_world_copy, MILK_BODY_NAME, DRAWER_BODY_NAME),
            relations=[relation],
        )
        graph.add_relation_edges()
        assert [
            (edge.source.name, edge.target.name, edge.relation) for edge in graph.edges
        ] == [
            (MILK_BODY_NAME, DRAWER_BODY_NAME, relation),
            (DRAWER_BODY_NAME, MILK_BODY_NAME, relation),
        ]

    def test_default_relations_are_the_four_spatial_relations(self) -> None:
        assert [type(relation) for relation in default_relations(NEAR_THRESHOLD)] == [
            Support,
            Containment,
            Contact,
            Near,
        ]

    def test_default_near_relation_uses_the_given_threshold(self) -> None:
        (near,) = [
            relation
            for relation in default_relations(NEAR_THRESHOLD)
            if isinstance(relation, Near)
        ]
        assert near.threshold == NEAR_THRESHOLD

    def test_default_near_relation_uses_the_default_threshold(self) -> None:
        (near,) = [
            relation for relation in default_relations() if isinstance(relation, Near)
        ]
        assert near.threshold == DEFAULT_NEAR_THRESHOLD


# %% contact


class TestContact:
    def test_holds_for_touching_objects(self, boxes_in_a_row) -> None:
        assert Contact().holds(
            boxes_in_a_row.node("milk"), boxes_in_a_row.node("apple")
        )

    def test_does_not_hold_for_objects_apart(self, boxes_in_a_row) -> None:
        assert not Contact().holds(
            boxes_in_a_row.node("milk"), boxes_in_a_row.node("spoon")
        )

    def test_edges_connect_only_touching_objects(self, boxes_in_a_row) -> None:
        boxes_in_a_row.add_relation_edges()
        assert sorted(
            (edge.source.name, edge.target.name) for edge in boxes_in_a_row.edges
        ) == [
            ("apple", "milk"),
            ("apple", "spoon"),
            ("milk", "apple"),
            ("spoon", "apple"),
        ]


# %% support


class TestSupport:
    def test_holds_from_the_supporting_to_the_supported_object(
        self, stacked_boxes
    ) -> None:
        assert Support().holds(
            stacked_boxes.node("counter_top"), stacked_boxes.node("milk")
        )

    def test_does_not_hold_from_the_supported_to_the_supporting_object(
        self, stacked_boxes
    ) -> None:
        assert not Support().holds(
            stacked_boxes.node("milk"), stacked_boxes.node("counter_top")
        )

    def test_does_not_hold_for_an_object_floating_above(self, stacked_boxes) -> None:
        assert not Support().holds(
            stacked_boxes.node("counter_top"), stacked_boxes.node("apple")
        )

    def test_edges_connect_only_supporting_to_supported(self, stacked_boxes) -> None:
        stacked_boxes.add_relation_edges()
        assert [
            (edge.source.name, edge.target.name) for edge in stacked_boxes.edges
        ] == [("counter_top", "milk")]


# %% near


class TestNear:
    def test_threshold_defaults_to_the_default_near_threshold(self) -> None:
        assert Near().threshold == DEFAULT_NEAR_THRESHOLD

    def test_holds_within_the_threshold_whatever_the_height(self, spread_boxes) -> None:
        assert Near(threshold=NEAR_THRESHOLD).holds(
            spread_boxes.node("milk"), spread_boxes.node("apple")
        )

    def test_does_not_hold_beyond_the_threshold(self, spread_boxes) -> None:
        assert not Near(threshold=NEAR_THRESHOLD).holds(
            spread_boxes.node("milk"), spread_boxes.node("spoon")
        )

    def test_edges_connect_only_objects_within_the_threshold(
        self, spread_boxes
    ) -> None:
        spread_boxes.add_relation_edges()
        assert sorted(
            (edge.source.name, edge.target.name) for edge in spread_boxes.edges
        ) == [("apple", "milk"), ("milk", "apple")]


# %% containment


class TestContainment:
    def test_holds_for_an_object_inside(self, boxes_in_a_drawer) -> None:
        assert Containment().holds(
            boxes_in_a_drawer.node("drawer"), boxes_in_a_drawer.node("spoon")
        )

    def test_does_not_hold_from_the_inner_to_the_outer_object(
        self, boxes_in_a_drawer
    ) -> None:
        assert not Containment().holds(
            boxes_in_a_drawer.node("spoon"), boxes_in_a_drawer.node("drawer")
        )

    def test_does_not_hold_for_an_object_outside(self, boxes_in_a_drawer) -> None:
        assert not Containment().holds(
            boxes_in_a_drawer.node("drawer"), boxes_in_a_drawer.node("milk")
        )

    def test_default_ratio_rejects_an_object_half_inside(
        self, boxes_in_a_drawer
    ) -> None:
        assert Containment().minimum_ratio == MINIMUM_CONTAINMENT_RATIO
        assert not Containment().holds(
            boxes_in_a_drawer.node("drawer"), boxes_in_a_drawer.node("apple")
        )

    def test_lower_ratio_accepts_an_object_half_inside(self, boxes_in_a_drawer) -> None:
        assert Containment(minimum_ratio=BELOW_HALF_RATIO).holds(
            boxes_in_a_drawer.node("drawer"), boxes_in_a_drawer.node("apple")
        )

    def test_edges_connect_only_container_to_contained(self, boxes_in_a_drawer) -> None:
        boxes_in_a_drawer.add_relation_edges()
        assert [
            (edge.source.name, edge.target.name) for edge in boxes_in_a_drawer.edges
        ] == [("drawer", "spoon")]


# %% edge removal


def relation_triples(graph: InteractionGraph) -> list[tuple[str, str, type]]:
    return [
        (edge.source.name, edge.target.name, type(edge.relation))
        for edge in graph.edges
    ]


class TestContainmentExcludesSupport:
    def test_removes_support_in_both_directions_between_contained_objects(
        self, apartment_graph
    ) -> None:
        apartment_graph.add_edge(DRAWER_BODY_NAME, MILK_BODY_NAME, Containment())
        apartment_graph.add_edge(DRAWER_BODY_NAME, MILK_BODY_NAME, Support())
        apartment_graph.add_edge(MILK_BODY_NAME, DRAWER_BODY_NAME, Support())
        apartment_graph.remove_edges([ContainmentExcludesSupport()])
        assert relation_triples(apartment_graph) == [
            (DRAWER_BODY_NAME, MILK_BODY_NAME, Containment)
        ]

    def test_keeps_other_relations_between_contained_objects(
        self, apartment_graph
    ) -> None:
        apartment_graph.add_edge(DRAWER_BODY_NAME, MILK_BODY_NAME, Containment())
        apartment_graph.add_edge(DRAWER_BODY_NAME, MILK_BODY_NAME, Near())
        apartment_graph.remove_edges([ContainmentExcludesSupport()])
        assert relation_triples(apartment_graph) == [
            (DRAWER_BODY_NAME, MILK_BODY_NAME, Containment),
            (DRAWER_BODY_NAME, MILK_BODY_NAME, Near),
        ]

    def test_keeps_support_without_containment(self, apartment_graph) -> None:
        apartment_graph.add_edge(DRAWER_BODY_NAME, MILK_BODY_NAME, Support())
        apartment_graph.remove_edges([ContainmentExcludesSupport()])
        assert relation_triples(apartment_graph) == [
            (DRAWER_BODY_NAME, MILK_BODY_NAME, Support)
        ]

    def test_returns_the_removed_edges(self, apartment_graph) -> None:
        support = Support()
        apartment_graph.add_edge(DRAWER_BODY_NAME, MILK_BODY_NAME, Containment())
        apartment_graph.add_edge(MILK_BODY_NAME, DRAWER_BODY_NAME, support)
        (removed,) = apartment_graph.remove_edges([ContainmentExcludesSupport()])
        assert (removed.source.name, removed.target.name, removed.relation) == (
            MILK_BODY_NAME,
            DRAWER_BODY_NAME,
            support,
        )


# %% graph of selected bodies


class AlwaysContains(Containment):
    """
    A containment that holds between every ordered pair of objects.
    """

    def holds(self, source: InteractionNode, target: InteractionNode) -> bool:
        return True


class AlwaysSupports(Support):
    """
    A support that holds between every ordered pair of objects.
    """

    def holds(self, source: InteractionNode, target: InteractionNode) -> bool:
        return True


class TestSelectedBodies:
    def test_the_annotated_bodies_are_the_nodes(self, apartment_world_copy) -> None:
        graph = InteractionGraph.from_selected_bodies(
            apartment_world_copy,
            bodies_of(apartment_world_copy, MILK_BODY_NAME, DRAWER_BODY_NAME)
            + [apartment_world_copy.root],
            relations=[],
            removal_rules=[],
        )
        assert [node.name for node in graph.nodes] == [MILK_BODY_NAME, DRAWER_BODY_NAME]

    def test_every_relation_that_holds_is_an_edge(self, apartment_world_copy) -> None:
        graph = InteractionGraph.from_selected_bodies(
            apartment_world_copy,
            bodies_of(apartment_world_copy, MILK_BODY_NAME, DRAWER_BODY_NAME),
            relations=[AlwaysHolds()],
            removal_rules=[],
        )
        assert [(edge.source.name, edge.target.name) for edge in graph.edges] == [
            (MILK_BODY_NAME, DRAWER_BODY_NAME),
            (DRAWER_BODY_NAME, MILK_BODY_NAME),
        ]

    def test_the_removal_rules_are_applied(self, apartment_world_copy) -> None:
        graph = InteractionGraph.from_selected_bodies(
            apartment_world_copy,
            bodies_of(apartment_world_copy, MILK_BODY_NAME, DRAWER_BODY_NAME),
            relations=[AlwaysContains(), AlwaysSupports()],
            removal_rules=[ContainmentExcludesSupport()],
        )
        assert [type(edge.relation) for edge in graph.edges] == [
            AlwaysContains,
            AlwaysContains,
        ]


def test_default_edge_removal_rules_contain_containment_excludes_support() -> None:
    assert [type(rule) for rule in default_edge_removal_rules()] == [
        ContainmentExcludesSupport
    ]


# %% visualization


@pytest.fixture()
def apartment_visualizer(apartment_graph) -> InteractionGraphVisualizer:
    return apartment_graph.visualizer(open_browser=False)


class TestVisualization:
    def test_a_graph_shown_in_a_visualizer_replaces_the_drawn_graph(
        self, apartment_world_copy, apartment_visualizer
    ) -> None:
        milk_graph = InteractionGraph.from_world(
            apartment_world_copy,
            bodies_of(apartment_world_copy, MILK_BODY_NAME),
            relations=[],
        )

        milk_graph.show_in(apartment_visualizer)

        assert [node["label"] for node in apartment_visualizer.graph_nodes()] == [
            MILK_BODY_NAME
        ]

    def test_draws_one_circle_per_object_labelled_with_its_name(
        self, apartment_graph, apartment_visualizer
    ) -> None:
        nodes = apartment_visualizer.graph_nodes()
        assert [node["label"] for node in nodes] == [
            node.name for node in apartment_graph.nodes
        ]
        assert {node["shape"] for node in nodes} == {"circle"}

    def test_clicked_object_shows_its_affordances_properties_and_actions(
        self, apartment_graph, apartment_visualizer
    ) -> None:
        milk = apartment_graph.node(MILK_BODY_NAME)
        milk_index = apartment_graph.nodes.index(milk)
        assert apartment_visualizer.node_details(milk_index) == milk.description()

    def test_description_lists_affordances_properties_and_actions(
        self, apartment_graph
    ) -> None:
        milk = apartment_graph.node(MILK_BODY_NAME)
        description = "\n".join(milk.description())
        for word in (
            *(affordance.value for affordance in milk.affordances),
            *milk.properties,
            *milk.actions,
        ):
            assert word in description

    def test_captions_each_edge_with_its_relation(
        self, apartment_graph, apartment_visualizer
    ) -> None:
        apartment_graph.add_edge(DRAWER_BODY_NAME, MILK_BODY_NAME, Containment())
        apartment_graph.add_edge(MILK_BODY_NAME, DRAWER_BODY_NAME, Near())
        assert [edge["label"] for edge in apartment_visualizer.graph_edges()] == [
            Containment.__name__,
            Near.__name__,
        ]

    def test_draws_every_relation_between_the_same_two_objects(
        self, apartment_graph, apartment_visualizer
    ) -> None:
        apartment_graph.add_edge(DRAWER_BODY_NAME, MILK_BODY_NAME, Containment())
        apartment_graph.add_edge(DRAWER_BODY_NAME, MILK_BODY_NAME, Near())
        edges = apartment_visualizer.graph_edges()
        assert len({edge["id"] for edge in edges}) == len(apartment_graph.edges)

    def test_relations_between_the_same_two_objects_bend_apart(
        self, apartment_graph, apartment_visualizer
    ) -> None:
        apartment_graph.add_edge(DRAWER_BODY_NAME, MILK_BODY_NAME, Containment())
        apartment_graph.add_edge(MILK_BODY_NAME, DRAWER_BODY_NAME, Near())
        roundness = [
            edge["smooth"]["roundness"] for edge in apartment_visualizer.graph_edges()
        ]
        assert len(set(roundness)) == len(roundness)
