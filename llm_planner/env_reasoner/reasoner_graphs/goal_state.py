"""
The goal state of a task, made neurosymbolically: a language model names the relations
between object types that hold once the task is done, and each relation is resolved
into the bodies of the world, giving a PDDL goal of ``AND`` and ``OR`` connectives.
"""

from __future__ import annotations

import inspect
import itertools
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import rustworkx
from krrood.rustworkx_utils.graph_visualizer_base import (
    GraphLayout,
    GraphLayoutOptions,
)
from krrood.rustworkx_utils.visualization.visnetwork_graph_visualizer import (
    VisNetworkGraphVisualizer,
    VisNode,
)
from typing_extensions import Any, Callable, ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Flask

from llm_planner.env_reasoner.LLM_utilities import LanguageModel
from llm_planner.env_planner.triplet_grounding import ObjectNotInWorld
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

GoalPredicate = Callable[..., Any]
"""
A predicate a goal relation may name, e.g.
:class:`~semantic_digital_twin.reasoning.predicates.InsideOf` or
:func:`~semantic_digital_twin.reasoning.predicates.contact`; it is named by its
``__name__``.
"""

PROMPT_TEMPLATE = Path(__file__).parents[2] / "resources" / "prompts" / "goal_state.txt"
"""
The prompt asking for the goal relations, with ``task``, ``objects`` and ``predicates``
placeholders.
"""

RELATION_LENGTH = 3
"""
How many names one relation of the response holds: object, predicate, object.
"""

JSON_LIST_START = "["
"""
The character a JSON list starts with.
"""

GOAL_TREE_PORT = 8051
"""
Port the goal tree page is served on, apart from the interaction graph's page.
"""

GOAL_TREE_TEMPLATE_DIRECTORY = Path(__file__).parents[2] / "resources" / "templates"
"""
Directory holding the goal tree page.
"""

GOAL_TREE_TEMPLATE = "goal_tree.jinja"
"""
The goal tree page, drawing the tree with every node's and edge's text on it.
"""

GOAL_TREE_NODE_SPACING = 260
"""
Horizontal distance between neighbouring nodes of the goal tree, wide enough for a
predicate box.
"""

GOAL_TREE_LEVEL_SEPARATION = 180
"""
Vertical distance between the levels of the goal tree.
"""

GOAL_TREE_TITLE = "Goal tree"
"""
Title of the goal tree page.
"""

CONJUNCTION_COLOR = "#e07b39"
"""
Colour an ``AND`` node is drawn in.
"""

DISJUNCTION_COLOR = "#4a90d9"
"""
Colour an ``OR`` node is drawn in.
"""

CONNECTIVE_SHAPE = "circle"
"""
Shape an ``AND`` or ``OR`` node is drawn in, with the connective inside.
"""

PREDICATE_SHAPE = "box"
"""
Shape a predicate node is drawn in, with the predicate and its bodies inside.
"""

ATOM_COLOR = "#7bc67b"
"""
Colour a predicate node is drawn in.
"""


# %% exceptions


class GoalResponseError(ValueError):
    """
    Base class for every way a goal response of the language model can be unusable.
    """


class MalformedGoalResponseError(GoalResponseError):
    """
    Raised when a response is not a JSON list of ``[object, predicate, object]`` lists.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"The goal response is malformed: {reason}.")
        self.reason = reason
        """
        What is wrong with the response's shape.
        """


class UnknownGoalObjectError(GoalResponseError):
    """
    Raised when a relation names an object outside the ones the prompt offered.
    """

    def __init__(self, object_name: str, available_names: list[str]) -> None:
        super().__init__(
            f"Unknown object {object_name!r}; available objects: {available_names}."
        )
        self.object_name = object_name
        """
        The object name the response used.
        """

        self.available_names = available_names
        """
        The object names the prompt offered.
        """


class UnknownGoalPredicateError(GoalResponseError):
    """
    Raised when a relation names a predicate outside the ones the prompt offered.
    """

    def __init__(self, predicate_name: str, available_names: list[str]) -> None:
        super().__init__(
            f"Unknown predicate {predicate_name!r}; "
            f"available predicates: {available_names}."
        )
        self.predicate_name = predicate_name
        """
        The predicate name the response used.
        """

        self.available_names = available_names
        """
        The predicate names the prompt offered.
        """


# %% relations between object types


@dataclass(frozen=True)
class GoalRelation:
    """
    One relation between two object types that holds once the task is done.
    """

    source_object: type[SemanticAnnotation]
    """
    The object the predicate is said of first, e.g. the spoon in "spoon inside drawer".
    """

    predicate: GoalPredicate
    """
    The predicate relating the two objects.
    """

    target_object: type[SemanticAnnotation]
    """
    The object the predicate is said of second, e.g. the drawer.
    """


# %% goal expressions


class LogicalConnective(StrEnum):
    """
    The PDDL connectives a goal combines its predicates with.
    """

    AND = "AND"
    """
    Every operand holds.
    """

    OR = "OR"
    """
    At least one operand holds.
    """


class GoalExpressionKind(StrEnum):
    """
    What a goal expression is.
    """

    PREDICATE = "predicate"
    """
    A predicate applied to two bodies.
    """

    CONJUNCTION = "conjunction"
    """
    An ``AND`` over expressions.
    """

    DISJUNCTION = "disjunction"
    """
    An ``OR`` over expressions.
    """


class GoalExpression(ABC):
    """
    A node of a goal: a predicate applied to bodies, or a connective over expressions.
    """

    kind: ClassVar[GoalExpressionKind]
    """
    What the expression is.
    """

    color: ClassVar[str]
    """
    Colour the expression's node is drawn in.
    """

    shape: ClassVar[str]
    """
    Shape the expression's node is drawn in.
    """

    @abstractmethod
    def to_pddl(self) -> str:
        """
        :return: The expression in PDDL.
        """

    @abstractmethod
    def label(self) -> str:
        """
        :return: The text the expression's node is captioned with.
        """

    @abstractmethod
    def description(self) -> list[str]:
        """
        :return: Everything the expression says, one line each.
        """

    @abstractmethod
    def children(self) -> list[GoalExpression]:
        """
        :return: The expressions directly below this one.
        """

    def caption(self) -> str:
        """
        :return: The :meth:`description`, one line each, as the expression's node
            shows it.
        """
        return "\n".join(self.description())


@dataclass
class GoalAtom(GoalExpression):
    """
    A predicate applied to two bodies, e.g. ``(InsideOf spoon.stl drawer_top1)``.
    """

    predicate_name: str
    """
    Name of the predicate.
    """

    source_body_name: str
    """
    Name of the body the predicate is said of first.
    """

    target_body_name: str
    """
    Name of the body the predicate is said of second.
    """

    kind: ClassVar[GoalExpressionKind] = GoalExpressionKind.PREDICATE
    color: ClassVar[str] = ATOM_COLOR
    shape: ClassVar[str] = PREDICATE_SHAPE

    def to_pddl(self) -> str:
        return (
            f"({self.predicate_name} {self.source_body_name} {self.target_body_name})"
        )

    def label(self) -> str:
        return self.to_pddl()

    def description(self) -> list[str]:
        return [
            f"{self.source_body_name}",
            self.predicate_name,
            f"{self.target_body_name}",
        ]

    def children(self) -> list[GoalExpression]:
        return []


@dataclass
class ConnectiveExpression(GoalExpression, ABC):
    """
    A connective over goal expressions.
    """

    operands: list[GoalExpression]
    """
    The expressions the connective combines, in order.
    """

    connective: ClassVar[LogicalConnective]
    """
    The connective this expression applies.
    """

    shape: ClassVar[str] = CONNECTIVE_SHAPE

    def to_pddl(self) -> str:
        operands = " ".join(operand.to_pddl() for operand in self.operands)
        return f"({self.connective} {operands})"

    def label(self) -> str:
        return self.connective

    def description(self) -> list[str]:
        return [self.connective]

    def children(self) -> list[GoalExpression]:
        return list(self.operands)


@dataclass
class GoalConjunction(ConnectiveExpression):
    """
    Holds when every operand holds.
    """

    connective: ClassVar[LogicalConnective] = LogicalConnective.AND
    kind: ClassVar[GoalExpressionKind] = GoalExpressionKind.CONJUNCTION
    color: ClassVar[str] = CONJUNCTION_COLOR


@dataclass
class GoalDisjunction(ConnectiveExpression):
    """
    Holds when at least one operand holds.
    """

    connective: ClassVar[LogicalConnective] = LogicalConnective.OR
    kind: ClassVar[GoalExpressionKind] = GoalExpressionKind.DISJUNCTION
    color: ClassVar[str] = DISJUNCTION_COLOR


@dataclass
class GoalTreeVisualizer(VisNetworkGraphVisualizer):
    """
    A page drawing a goal tree with all its text on it: each connective as a circle
    around ``AND`` or ``OR``, each predicate as a box showing its
    :meth:`GoalExpression.description`, and an unlabelled arrow from each connective
    to each of its operands. Nodes cannot be clicked for details.
    """

    layout_options: ClassVar[GraphLayoutOptions] = GraphLayoutOptions(
        layout_options={
            GraphLayout.LAYERED: {
                "layout": {
                    "hierarchical": {
                        "direction": "UD",
                        "sortMethod": "directed",
                        "nodeSpacing": GOAL_TREE_NODE_SPACING,
                        "levelSeparation": GOAL_TREE_LEVEL_SEPARATION,
                    }
                },
                "physics": {"enabled": False},
            },
        }
    )
    """
    The top-down layout the goal tree is drawn in.
    """

    template: ClassVar[str] = GOAL_TREE_TEMPLATE
    """
    The page template the tree is drawn with.
    """

    def graph_nodes(self) -> list[VisNode]:
        """
        :return: One node per expression, in the expression's shape.
        """
        return [
            {**node, "shape": self.graph[int(node["id"])].shape}
            for node in super().graph_nodes()
        ]

    def build_application(self) -> Flask:
        """
        :return: The Flask application serving the page and the graph endpoint.
        """
        from flask import Flask, jsonify, render_template

        application = Flask(__name__, template_folder=GOAL_TREE_TEMPLATE_DIRECTORY)

        @application.route("/")
        def index() -> str:
            return render_template(
                self.template,
                title=self.title,
                layout_options_json=json.dumps(
                    self.layout_options.get_options(self.layout)
                ),
                interval_ms=int(self.refresh_interval_seconds * 1000),
            )

        @application.route("/graph")
        def graph():
            return jsonify(nodes=self.graph_nodes(), edges=self.graph_edges())

        return application


@dataclass
class GoalDefinition:
    """
    The ``(:goal ...)`` section of a PDDL problem.
    """

    condition: GoalConjunction
    """
    The condition the world must meet, one operand per goal relation.
    """

    def to_pddl(self) -> str:
        """
        :return: The goal section in PDDL, e.g. ``(:goal (AND (InsideOf a b)))``.
        """
        return f"(:goal {self.condition.to_pddl()})"

    def tree(self) -> rustworkx.PyDiGraph:
        """
        :return: The goal as a tree, each expression a node pointing to its operands,
            in depth-first order from :attr:`condition`.
        """
        tree = rustworkx.PyDiGraph()
        self._add_subtree(tree, self.condition)
        return tree

    def visualizer(self, open_browser: bool = True) -> GoalTreeVisualizer:
        """
        :param open_browser: Whether the running page opens in a web browser.
        :return: A visualizer of :meth:`tree`, not yet running.
        """
        return GoalTreeVisualizer(
            graph=self.tree(),
            label_getter=GoalExpression.caption,
            color_getter=lambda expression: expression.color,
            layout=GraphLayout.LAYERED,
            title=GOAL_TREE_TITLE,
            port=GOAL_TREE_PORT,
            open_browser=open_browser,
        )

    def visualize(self, open_browser: bool = True) -> GoalTreeVisualizer:
        """
        Serve a page of the goal's ``AND``/``OR`` tree.

        The page is served from a background thread, so it stays reachable only while
        the calling process keeps running.

        :param open_browser: Whether to open the page in a web browser.
        :return: The running visualizer.
        """
        visualizer = self.visualizer(open_browser)
        visualizer.run()
        return visualizer

    def _add_subtree(
        self, tree: rustworkx.PyDiGraph, expression: GoalExpression
    ) -> int:
        """
        :param tree: The tree to add to.
        :param expression: The expression to add with everything below it.
        :return: The index of ``expression``'s node.
        """
        index = tree.add_node(expression)
        for child in expression.children():
            tree.add_edge(index, self._add_subtree(tree, child), None)
        return index


# %% goal state


@dataclass
class GoalState:
    """
    The goal state of a task: asks a language model which relations between the
    offered objects hold once the task is done, and resolves them into the bodies of
    the world.
    """

    task: str
    """
    The natural language task, e.g. "put spoon in the drawer".
    """

    object_types: list[type[SemanticAnnotation]]
    """
    The semantic annotations of the world the goal may name.
    """

    predicates: list[GoalPredicate]
    """
    The predicates the goal may name.
    """

    world: World
    """
    The world whose bodies the goal is resolved into.
    """

    language_model: LanguageModel
    """
    The model asked for the goal relations, e.g. an
    :class:`~llm_planner.env_reasoner.LLM_utilities.OllamaLanguageModel`.
    """

    def goal(self) -> GoalDefinition:
        """
        :return: The goal of :attr:`task`, from asking the language model to resolving
            its answer.
        :raises GoalResponseError: If the answer is unusable.
        :raises ObjectNotInWorld: If the answer names an object the world lacks.
        """
        return self.resolve(self.query_language_model(self.build_prompt()))

    # %% prompt

    def build_prompt(self) -> str:
        """
        :return: The prompt asking for the goal relations of :attr:`task`.
        """
        return PROMPT_TEMPLATE.read_text().format(
            task=self.task,
            objects="\n".join(
                f"- {object_type.__name__}" for object_type in self.object_types
            ),
            predicates="\n".join(
                self._predicate_line(predicate) for predicate in self.predicates
            ),
        )

    @staticmethod
    def _predicate_line(predicate: GoalPredicate) -> str:
        """
        :param predicate: An offered predicate.
        :return: The predicate's name and, if documented, the first paragraph of its
            documentation.
        """
        documentation = inspect.getdoc(predicate)
        if not documentation:
            return f"- {predicate.__name__}"
        summary = " ".join(documentation.split("\n\n")[0].split())
        return f"- {predicate.__name__}: {summary}"

    # %% language model

    def query_language_model(self, prompt: str) -> list[GoalRelation]:
        """
        :param prompt: The prompt to ask, e.g. :meth:`build_prompt`.
        :return: The checked relations of the language model's answer.
        :raises GoalResponseError: If the answer is unusable.
        """
        return self.check_response(self.language_model.complete(prompt))

    # %% response check

    def check_response(self, response: str) -> list[GoalRelation]:
        """
        :param response: The language model's raw answer.
        :return: One relation per ``[object, predicate, object]`` list of the answer.
        :raises MalformedGoalResponseError: If the answer has another shape.
        :raises UnknownGoalObjectError: If it names an object that was not offered.
        :raises UnknownGoalPredicateError: If it names a predicate that was not offered.
        """
        return [
            GoalRelation(
                source_object=self._object_named(source_name),
                predicate=self._predicate_named(predicate_name),
                target_object=self._object_named(target_name),
            )
            for source_name, predicate_name, target_name in self._relation_names(
                response
            )
        ]

    @staticmethod
    def _relation_names(response: str) -> list[list[str]]:
        """
        :param response: The language model's raw answer.
        :return: The answer's relations, still as names.
        :raises MalformedGoalResponseError: If the answer is not a JSON list of lists
            of three names.
        """
        relations = GoalState._list_of_lists_in(response)
        for relation in relations:
            if not isinstance(relation, list) or len(relation) != RELATION_LENGTH:
                raise MalformedGoalResponseError(
                    f"{relation!r} is not a list of {RELATION_LENGTH} names"
                )
            if not all(isinstance(name, str) for name in relation):
                raise MalformedGoalResponseError(f"{relation!r} holds a non-name")
        return relations

    @staticmethod
    def _list_of_lists_in(response: str) -> list[list[Any]]:
        """
        Pick the JSON list of lists out of an answer, ignoring any text, code fence or
        other JSON around it.

        :param response: The language model's raw answer.
        :return: The first JSON list in the answer whose elements are all lists.
        :raises MalformedGoalResponseError: If the answer holds no such list.
        """
        decoder = json.JSONDecoder()
        start = response.find(JSON_LIST_START)
        while start >= 0:
            try:
                candidate, end = decoder.raw_decode(response, start)
            except json.JSONDecodeError:
                start = response.find(JSON_LIST_START, start + 1)
                continue
            if isinstance(candidate, list) and all(
                isinstance(element, list) for element in candidate
            ):
                return candidate
            start = response.find(JSON_LIST_START, start + 1)
        raise MalformedGoalResponseError("no JSON list of relations in the answer")

    def _object_named(self, name: str) -> type[SemanticAnnotation]:
        """
        :raises UnknownGoalObjectError: If no offered object has that name.
        """
        for object_type in self.object_types:
            if object_type.__name__ == name:
                return object_type
        raise UnknownGoalObjectError(
            name, [object_type.__name__ for object_type in self.object_types]
        )

    def _predicate_named(self, name: str) -> GoalPredicate:
        """
        :raises UnknownGoalPredicateError: If no offered predicate has that name.
        """
        for predicate in self.predicates:
            if predicate.__name__ == name:
                return predicate
        raise UnknownGoalPredicateError(
            name, [predicate.__name__ for predicate in self.predicates]
        )

    # %% resolution into bodies

    def resolve(self, relations: list[GoalRelation]) -> GoalDefinition:
        """
        Resolve relations between object types into a goal over the world's bodies.

        All relations must hold together. A relation between objects that have several
        bodies holds for any one pair of those bodies.

        :param relations: The relations that hold once the task is done.
        :return: The goal, an ``AND`` with one operand per relation.
        :raises ObjectNotInWorld: If a relation names an object the world lacks.
        """
        return GoalDefinition(
            GoalConjunction(
                [self._resolve_relation(relation) for relation in relations]
            )
        )

    def _resolve_relation(self, relation: GoalRelation) -> GoalExpression:
        """
        :param relation: One relation between object types.
        :return: The relation applied to the only pair of bodies, or an ``OR`` of it
            applied to every pair of bodies.
        :raises ObjectNotInWorld: If either object is missing from the world.
        """
        atoms = [
            GoalAtom(relation.predicate.__name__, source_name, target_name)
            for source_name, target_name in itertools.product(
                self._body_names_of(relation.source_object),
                self._body_names_of(relation.target_object),
            )
        ]
        if len(atoms) == 1:
            return atoms[0]
        return GoalDisjunction(atoms)

    def _body_names_of(self, object_type: type[SemanticAnnotation]) -> list[str]:
        """
        :param object_type: An object type a relation names.
        :return: The name of the root body of every annotation of that type, in the
            order the world holds them.
        :raises ObjectNotInWorld: If the world holds no such annotation.
        """
        names = list(
            dict.fromkeys(
                annotation.root.name.name
                for annotation in self.world.get_semantic_annotations_by_type(
                    object_type
                )
            )
        )
        if not names:
            raise ObjectNotInWorld(object_type)
        return names
