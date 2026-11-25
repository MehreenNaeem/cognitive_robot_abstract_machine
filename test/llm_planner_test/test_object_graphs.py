from __future__ import annotations

import pytest

from llm_planner.env_reasoner.reasoner_graphs.object_graphs import (
    AnnotationOrigin,
    BodyWithoutSemanticAnnotation,
    UnknownSemanticAnnotation,
    get_semantic_annotation_of_body,
    semantic_annotation_graph,
    semantic_annotation_type_by_name,
    visualize_semantic_annotation_graph,
    visualize_semantic_annotation_graph_for_name,
)
from semantic_digital_twin.semantic_annotations.mixins import HasDoors, IsPerceivable
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Cereal,
    Drawer,
    Fridge,
    Food,
    FoodContainer,
    Milk,
    Oven,
    TunaCan,
)

# %% semantic_annotation_graph


class TestSemanticAnnotationGraph:
    def test_root_node_is_the_queried_annotation_type(self) -> None:
        assert semantic_annotation_graph(HasDoors).annotation_type is HasDoors

    def test_direct_subclass_becomes_a_direct_child_node(self) -> None:
        children = [
            node.annotation_type
            for node in semantic_annotation_graph(HasDoors).children
        ]
        assert Cabinet in children

    def test_indirect_subclass_is_expanded_under_its_direct_parent(self) -> None:
        cabinet_node = next(
            node
            for node in semantic_annotation_graph(HasDoors).children
            if node.annotation_type is Cabinet
        )
        assert Fridge in [child.annotation_type for child in cabinet_node.children]

    def test_indirect_subclass_is_not_also_a_direct_child_of_the_root(self) -> None:
        children = [
            node.annotation_type
            for node in semantic_annotation_graph(HasDoors).children
        ]
        assert Fridge not in children

    def test_class_with_no_subclasses_has_no_children(self) -> None:
        assert semantic_annotation_graph(Oven).children == []


# %% visualize_semantic_annotation_graph


class TestVisualizeSemanticAnnotationGraph:
    def test_written_page_contains_every_node_name(self, tmp_path) -> None:
        graph = semantic_annotation_graph(HasDoors)
        output_path = tmp_path / "graph.html"

        visualize_semantic_annotation_graph(graph, output_path, open_in_browser=False)

        page = output_path.read_text()
        assert "HasDoors" in page
        assert "Cabinet" in page
        assert "Fridge" in page

    def test_leaf_node_is_not_rendered_as_expandable(self, tmp_path) -> None:
        graph = semantic_annotation_graph(Oven)
        output_path = tmp_path / "graph.html"

        visualize_semantic_annotation_graph(graph, output_path, open_in_browser=False)

        page = output_path.read_text()
        assert "<details" not in page

    def test_returns_the_output_path(self, tmp_path) -> None:
        graph = semantic_annotation_graph(Oven)
        output_path = tmp_path / "graph.html"

        result = visualize_semantic_annotation_graph(
            graph, output_path, open_in_browser=False
        )

        assert result == output_path


# %% walk


class TestWalk:
    def test_yields_the_root_first(self) -> None:
        graph = semantic_annotation_graph(FoodContainer)

        assert next(iter(graph.walk())) is graph

    def test_visits_every_node_in_the_subtree(self) -> None:
        graph = semantic_annotation_graph(Food)

        visited = {node.annotation_type for node in graph.walk()}

        assert {Food, FoodContainer, Milk, Cereal, TunaCan} <= visited


# %% concrete_annotations


class TestConcreteAnnotations:
    def test_lists_only_leaf_types(self) -> None:
        graph = semantic_annotation_graph(FoodContainer)

        leaves = graph.concrete_annotations()

        assert Milk in leaves
        assert Cereal in leaves
        assert TunaCan in leaves

    def test_excludes_branch_types(self) -> None:
        graph = semantic_annotation_graph(Food)

        assert FoodContainer not in graph.concrete_annotations()

    def test_a_leaf_root_returns_itself(self) -> None:
        assert semantic_annotation_graph(Oven).concrete_annotations() == [Oven]


# %% find and find_by_name


class TestFind:
    def test_returns_the_node_for_a_descendant_type(self) -> None:
        graph = semantic_annotation_graph(FoodContainer)

        assert graph.find(Milk).annotation_type is Milk

    def test_returns_none_for_a_type_outside_the_subtree(self) -> None:
        assert semantic_annotation_graph(FoodContainer).find(Cabinet) is None

    def test_find_by_name_is_case_insensitive(self) -> None:
        graph = semantic_annotation_graph(FoodContainer)

        assert graph.find_by_name("milk").annotation_type is Milk

    def test_find_by_name_returns_none_when_no_node_matches(self) -> None:
        assert semantic_annotation_graph(FoodContainer).find_by_name("banana") is None


# %% lineage_to


class TestLineageTo:
    def test_traces_root_to_target_inclusive(self) -> None:
        graph = semantic_annotation_graph(Food)

        assert graph.lineage_to(Milk) == [Food, FoodContainer, Milk]

    def test_single_element_when_target_is_the_root(self) -> None:
        graph = semantic_annotation_graph(FoodContainer)

        assert graph.lineage_to(FoodContainer) == [FoodContainer]

    def test_empty_when_target_is_not_in_the_subtree(self) -> None:
        assert semantic_annotation_graph(FoodContainer).lineage_to(Cabinet) == []


# %% origin_of


class TestOriginOf:
    def test_reports_the_full_lineage(self) -> None:
        graph = semantic_annotation_graph(Food)

        assert graph.origin_of(Milk).lineage == [Food, FoodContainer, Milk]

    def test_generated_from_is_the_direct_graph_parent(self) -> None:
        graph = semantic_annotation_graph(Food)

        assert graph.origin_of(Milk).generated_from is FoodContainer

    def test_generated_from_is_none_for_the_root(self) -> None:
        graph = semantic_annotation_graph(FoodContainer)

        assert graph.origin_of(FoodContainer).generated_from is None

    def test_extra_mixins_hold_bases_off_the_lineage(self) -> None:
        graph = semantic_annotation_graph(FoodContainer)

        assert graph.origin_of(Milk).extra_mixins == [IsPerceivable]

    def test_no_extra_mixins_when_only_base_is_the_graph_parent(self) -> None:
        graph = semantic_annotation_graph(FoodContainer)

        assert graph.origin_of(TunaCan).extra_mixins == []

    def test_returns_none_for_a_type_outside_the_subtree(self) -> None:
        assert semantic_annotation_graph(FoodContainer).origin_of(Cabinet) is None

    def test_result_is_an_annotation_origin(self) -> None:
        graph = semantic_annotation_graph(FoodContainer)

        assert isinstance(graph.origin_of(Milk), AnnotationOrigin)


# %% semantic_annotation_type_by_name


class TestSemanticAnnotationTypeByName:
    def test_resolves_a_class_defined_in_semantic_annotations(self) -> None:
        assert semantic_annotation_type_by_name("FoodContainer") is FoodContainer

    def test_resolves_a_mixin_defined_in_mixins(self) -> None:
        assert semantic_annotation_type_by_name("HasDoors") is HasDoors

    def test_unknown_name_raises_and_records_the_name(self) -> None:
        with pytest.raises(UnknownSemanticAnnotation) as raised:
            semantic_annotation_type_by_name("NotAnAnnotation")

        assert raised.value.name == "NotAnAnnotation"


# %% visualize_semantic_annotation_graph mixin labels


class TestVisualizeMixinLabels:
    def _render(self, tmp_path) -> str:
        output_path = tmp_path / "graph.html"
        visualize_semantic_annotation_graph(
            semantic_annotation_graph(FoodContainer),
            output_path,
            open_in_browser=False,
        )
        return output_path.read_text()

    def test_subclass_shows_its_mixin_base_classes(self, tmp_path) -> None:
        assert (
            f'Milk <span class="mixins">also: {IsPerceivable.__name__}</span>'
            in self._render(tmp_path)
        )

    def test_subclass_without_extra_mixins_has_no_also_note(self, tmp_path) -> None:
        page = self._render(tmp_path)

        tuna_can_fragment = page[page.index("TunaCan") : page.index("TunaCan") + 40]
        assert "also:" not in tuna_can_fragment

    def test_root_is_not_annotated_with_its_own_superclass(self, tmp_path) -> None:
        page = self._render(tmp_path)

        assert f"<summary>{FoodContainer.__name__}</summary>" in page
        assert f"{FoodContainer.__name__} <span" not in page


# %% visualize_semantic_annotation_graph_for_name


class TestVisualizeSemanticAnnotationGraphForName:
    def test_builds_the_graph_for_the_named_class(self, tmp_path) -> None:
        output_path = tmp_path / "graph.html"

        visualize_semantic_annotation_graph_for_name(
            "FoodContainer", output_path, open_in_browser=False
        )

        page = output_path.read_text()
        assert "<h1>FoodContainer subclass graph</h1>" in page
        assert Milk.__name__ in page

    def test_returns_the_output_path(self, tmp_path) -> None:
        output_path = tmp_path / "graph.html"

        result = visualize_semantic_annotation_graph_for_name(
            "FoodContainer", output_path, open_in_browser=False
        )

        assert result == output_path

    def test_unknown_name_raises(self, tmp_path) -> None:
        with pytest.raises(UnknownSemanticAnnotation):
            visualize_semantic_annotation_graph_for_name(
                "NotAnAnnotation", tmp_path / "graph.html", open_in_browser=False
            )


# %% get_semantic_annotation_of_body

MILK_BODY_NAME = "milk.stl"
"""
The apartment body the milk annotation is rooted at.
"""

DRAWER_BODY_NAME = "cabinet10_drawer_top"
"""
The apartment body a drawer annotation is rooted at.
"""


class TestSemanticAnnotationOfBody:
    def test_a_body_belongs_to_the_annotation_rooted_at_it(
        self, apartment_world_copy
    ) -> None:
        annotation = get_semantic_annotation_of_body(
            MILK_BODY_NAME, apartment_world_copy
        )

        assert annotation is apartment_world_copy.get_semantic_annotations_by_type(
            Milk
        )[0]

    def test_the_annotation_found_is_rooted_at_the_named_body(
        self, apartment_world_copy
    ) -> None:
        annotation = get_semantic_annotation_of_body(
            DRAWER_BODY_NAME, apartment_world_copy
        )

        assert annotation.root is apartment_world_copy.get_body_by_name(
            DRAWER_BODY_NAME
        )

    def test_a_body_no_annotation_is_rooted_at_raises(
        self, apartment_world_copy
    ) -> None:
        with pytest.raises(BodyWithoutSemanticAnnotation):
            get_semantic_annotation_of_body(
                apartment_world_copy.root.name.name, apartment_world_copy
            )
