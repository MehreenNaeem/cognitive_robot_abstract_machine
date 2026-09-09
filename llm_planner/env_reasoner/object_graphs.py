from __future__ import annotations

import html
import inspect
import webbrowser
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from semantic_digital_twin.semantic_annotations import mixins, semantic_annotations


DEFAULT_GRAPH_HTML_PATH = Path("semantic_annotation_graph.html")

#: Modules scanned to resolve a semantic annotation class from its name.
SEMANTIC_ANNOTATION_MODULES = (semantic_annotations, mixins)


class UnknownSemanticAnnotation(Exception):
    """
    Raised when a name does not match any semantic annotation class.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"No semantic annotation class named {name!r}.")
        self.name = name
        """
        The unresolved name that was requested.
        """


def semantic_annotation_type_by_name(name: str) -> type:
    """
    Resolve a semantic annotation class from its name.

    :param name: The class name, e.g. ``"FoodContainer"``.
    :return: The matching class from :data:`SEMANTIC_ANNOTATION_MODULES`.
    :raises UnknownSemanticAnnotation: If no class in those modules has that name.
    """
    types_by_name = {
        candidate.__name__: candidate
        for module in SEMANTIC_ANNOTATION_MODULES
        for _, candidate in inspect.getmembers(module, inspect.isclass)
        if candidate.__module__ == module.__name__
    }
    if name not in types_by_name:
        raise UnknownSemanticAnnotation(name)
    return types_by_name[name]


def mixin_bases(annotation_type: type, lineage: list[type]) -> list[type]:
    """
    List the base classes of ``annotation_type`` that are not on ``lineage``.

    These are the extra behaviours a type mixes in on top of its place in a subclass
    graph, e.g. :class:`~semantic_digital_twin.semantic_annotations.mixins.IsPerceivable`
    for :class:`Milk`.

    :param annotation_type: The class whose bases are inspected.
    :param lineage: The chain of annotation types from a graph root to
        ``annotation_type``; bases on this chain are its graph ancestors, not mixins.
    :return: The remaining base classes, in declaration order.
    """
    return [
        base
        for base in annotation_type.__bases__
        if base not in lineage and base is not object
    ]


@dataclass
class SemanticAnnotationGraphNode:
    """
    A node in a semantic annotation class's subclass graph.
    """

    annotation_type: type
    """
    The semantic annotation class or mixin this node represents.
    """

    children: list[SemanticAnnotationGraphNode]
    """
    One node per class that uses :attr:`annotation_type` as an immediate base class,
    each expanded with their own subclasses in turn.
    """

    def print_tree(self, indent: int = 0) -> None:
        """
        Print this node and every descendant, indented by depth.

        :param indent: Nesting depth of this node, used to compute its indentation.
        """
        print("  " * indent + self.annotation_type.__name__)
        for child in self.children:
            child.print_tree(indent + 1)

    def walk(self) -> Iterator[SemanticAnnotationGraphNode]:
        """
        Yield this node and every descendant, depth first.
        """
        yield self
        for child in self.children:
            yield from child.walk()

    def concrete_annotations(self) -> list[type]:
        """
        List the leaf annotation types in this subtree.

        These are the concrete objects the root stands for, e.g. the children a
        ``FoodContainer`` graph resolves to such as :class:`Milk` and ``Cereal``.

        :return: The :attr:`annotation_type` of every node without children.
        """
        return [node.annotation_type for node in self.walk() if not node.children]

    def find(self, annotation_type: type) -> SemanticAnnotationGraphNode | None:
        """
        Locate the node for ``annotation_type`` in this subtree.

        :param annotation_type: The semantic annotation class to search for.
        :return: The matching node, or ``None`` if it is not in this subtree.
        """
        return next(
            (node for node in self.walk() if node.annotation_type is annotation_type),
            None,
        )

    def find_by_name(self, name: str) -> SemanticAnnotationGraphNode | None:
        """
        Locate the node whose annotation class has the given name, ignoring case.

        :param name: The class name to search for, e.g. ``"Milk"``.
        :return: The matching node, or ``None`` if no node in this subtree matches.
        """
        lowered = name.lower()
        return next(
            (
                node
                for node in self.walk()
                if node.annotation_type.__name__.lower() == lowered
            ),
            None,
        )

    def lineage_to(self, annotation_type: type) -> list[type]:
        """
        Trace the chain of annotation types from this root down to ``annotation_type``.

        :param annotation_type: The semantic annotation class to trace to.
        :return: The annotation types from this root to ``annotation_type`` inclusive,
            or an empty list if ``annotation_type`` is not in this subtree.
        """
        if self.annotation_type is annotation_type:
            return [self.annotation_type]
        for child in self.children:
            tail = child.lineage_to(annotation_type)
            if tail:
                return [self.annotation_type, *tail]
        return []

    def origin_of(self, annotation_type: type) -> AnnotationOrigin | None:
        """
        Describe where ``annotation_type`` sits in this subtree and what else it is.

        :param annotation_type: The semantic annotation class to describe, e.g.
            :class:`Milk`.
        :return: An :class:`AnnotationOrigin` giving the node it is generated from and
            the extra mixins it carries, or ``None`` if it is not in this subtree.
        """
        lineage = self.lineage_to(annotation_type)
        if not lineage:
            return None
        return AnnotationOrigin(
            annotation_type=annotation_type,
            lineage=lineage,
            extra_mixins=mixin_bases(annotation_type, lineage),
        )


@dataclass
class AnnotationOrigin:
    """
    Where a semantic annotation type is placed in a subclass graph and what other
    behaviour it mixes in.
    """

    annotation_type: type
    """
    The semantic annotation class this describes.
    """

    lineage: list[type]
    """
    The chain of annotation types from the graph root to :attr:`annotation_type`
    inclusive; consecutive entries are base and subclass.
    """

    extra_mixins: list[type]
    """
    Base classes of :attr:`annotation_type` that are not on :attr:`lineage`, e.g.
    :class:`~semantic_digital_twin.semantic_annotations.mixins.IsPerceivable`.
    """

    @property
    def generated_from(self) -> type | None:
        """
        The annotation type :attr:`annotation_type` is directly derived from within the
        graph, i.e. the node it was generated from.
        """
        return self.lineage[-2] if len(self.lineage) >= 2 else None


def semantic_annotation_graph(annotation_type: type) -> SemanticAnnotationGraphNode:
    """
    Build the subclass graph of a semantic annotation class or mixin.

    Recursively expands every class defined in
    :mod:`semantic_digital_twin.semantic_annotations.semantic_annotations` that uses
    `annotation_type` as an immediate base class, then does the same for each of those
    classes, and so on down the hierarchy.

    :param annotation_type: A semantic annotation class or mixin, e.g. ``HasDoors``.
    :return: The root node for `annotation_type`, whose children are the classes that
        directly derive from it, each expanded the same way.
    """
    direct_children = [
        candidate
        for _, candidate in inspect.getmembers(semantic_annotations, inspect.isclass)
        if candidate.__module__ == semantic_annotations.__name__
        and annotation_type in candidate.__bases__
    ]
    return SemanticAnnotationGraphNode(
        annotation_type=annotation_type,
        children=[semantic_annotation_graph(child) for child in direct_children],
    )


def _node_label(annotation_type: type, lineage: list[type]) -> str:
    """
    Build the HTML shown for one node: its class name, followed by an *also:* note
    listing its mixin base classes when it has any.

    :param annotation_type: The class this node represents.
    :param lineage: The chain of annotation types from the graph root to this node.
    :return: The escaped label markup.
    """
    name = html.escape(annotation_type.__name__)
    mixins = mixin_bases(annotation_type, lineage)
    if not mixins:
        return name
    mixin_names = ", ".join(html.escape(mixin.__name__) for mixin in mixins)
    return f'{name} <span class="mixins">also: {mixin_names}</span>'


def _node_to_html(
    node: SemanticAnnotationGraphNode, ancestors: tuple[type, ...] = ()
) -> str:
    """
    Render `node` as a collapsible ``<details>`` element, or a plain ``<li>`` if it has
    no children, so a click on its summary is what a browser natively expands.

    Each node also shows the mixin base classes it adds on top of its place in the
    graph, e.g. ``Milk`` is annotated with *also: IsPerceivable*.

    :param node: The graph node to render.
    :param ancestors: The annotation types from the graph root down to `node`'s parent.
    :return: The HTML fragment for `node` and every descendant.
    """
    lineage = [*ancestors, node.annotation_type]
    is_root = not ancestors
    label = (
        html.escape(node.annotation_type.__name__)
        if is_root
        else _node_label(node.annotation_type, lineage)
    )
    if not node.children:
        return f"<li>{label}</li>"

    child_items = "".join(
        _node_to_html(child, tuple(lineage)) for child in node.children
    )
    return f"<li><details open><summary>{label}</summary><ul>{child_items}</ul></details></li>"


def visualize_semantic_annotation_graph(
    root: SemanticAnnotationGraphNode,
    output_path: Path | str = DEFAULT_GRAPH_HTML_PATH,
    *,
    open_in_browser: bool = True,
) -> Path:
    """
    Render `root` as a self-contained, interactive HTML page and write it to disk.

    Every node with children is a ``<details>``/``<summary>`` element, which browsers
    natively expand or collapse on click, so no script is needed for the interaction.

    :param root: The graph node to render, typically from :func:`semantic_annotation_graph`.
    :param output_path: File path the HTML page is written to.
    :param open_in_browser: Whether to open `output_path` in the default web browser
        after writing it.
    :return: `output_path`, as a :class:`~pathlib.Path`.
    """
    title = html.escape(root.annotation_type.__name__)
    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title} subclass graph</title>
<style>
  body {{ font-family: sans-serif; }}
  ul {{ list-style-type: none; }}
  summary {{ cursor: pointer; }}
  .mixins {{ color: #666; font-style: italic; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>{title} subclass graph</h1>
<ul>{_node_to_html(root)}</ul>
</body>
</html>
"""
    output_path = Path(output_path)
    output_path.write_text(page)
    if open_in_browser:
        webbrowser.open(output_path.resolve().as_uri())
    return output_path


def visualize_semantic_annotation_graph_for_name(
    annotation_name: str,
    output_path: Path | str = DEFAULT_GRAPH_HTML_PATH,
    *,
    open_in_browser: bool = True,
) -> Path:
    """
    Build and render the subclass graph for the annotation class named `annotation_name`.

    :param annotation_name: The name of a semantic annotation class, e.g.
        ``"FoodContainer"``.
    :param output_path: File path the HTML page is written to.
    :param open_in_browser: Whether to open `output_path` in the default web browser
        after writing it.
    :return: `output_path`, as a :class:`~pathlib.Path`.
    :raises UnknownSemanticAnnotation: If `annotation_name` names no known class.
    """
    graph = semantic_annotation_graph(semantic_annotation_type_by_name(annotation_name))
    return visualize_semantic_annotation_graph(
        graph, output_path, open_in_browser=open_in_browser
    )



#Usage Example:
if __name__ == "__main__":
    annotation_name = input("Semantic annotation class: ").strip()
    graph = semantic_annotation_graph(semantic_annotation_type_by_name(annotation_name))
    graph.print_tree()

    print("\nConcrete objects:", [t.__name__ for t in graph.concrete_annotations()])
    for leaf in graph.concrete_annotations():
        origin = graph.origin_of(leaf)
        print(f"  {leaf.__name__} also is:", [t.__name__ for t in origin.extra_mixins])

    visualize_semantic_annotation_graph(graph)

