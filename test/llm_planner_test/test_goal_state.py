"""
Tests for :mod:`llm_planner.env_reasoner.reasoner_graphs.goal_state`.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from dataclasses import dataclass, field

import pytest

from llm_planner.env_planner.triplet_grounding import ObjectNotInWorld
from llm_planner.env_reasoner.reasoner_graphs.goal_state import (
    CONNECTIVE_SHAPE,
    GOAL_TREE_PORT,
    PREDICATE_SHAPE,
    GoalAtom,
    GoalConjunction,
    GoalDefinition,
    GoalDisjunction,
    GoalExpression,
    GoalExpressionKind,
    GoalRelation,
    GoalState,
    LogicalConnective,
    MalformedGoalResponseError,
    UnknownGoalObjectError,
    UnknownGoalPredicateError,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.predicates import InsideOf, contact
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Milk,
    Spoon,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

TASK = "put spoon in the drawer"
"""
The task every goal state in these tests is made for.
"""

SPOON_BODY_NAME = "spoon.stl"
"""
Name of the only spoon's body.
"""

DRAWER_BODY_NAMES = ["drawer_top1", "drawer_bottom1"]
"""
Names of the two drawers' bodies, in the order the world holds them.
"""


@dataclass
class ScriptedLanguageModel:
    """
    A language model answering with prepared texts, recording what it was asked.
    """

    answers: list[str]
    """
    The answers, in the order they are given.
    """

    prompts: list[str] = field(default_factory=list)
    """
    Every prompt asked, in order.
    """

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers[len(self.prompts) - 1]


def world_with(annotations: list[SemanticAnnotation]) -> World:
    """
    :param annotations: The annotations the world holds, each on its own body.
    :return: A world holding ``annotations`` in the given order.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    with world.modify_world():
        world.add_body(root)
        for annotation in annotations:
            world.add_connection(FixedConnection(parent=root, child=annotation.root))
            world.add_semantic_annotation(annotation)
    return world


def body_named(name: str) -> Body:
    return Body(name=PrefixedName(name))


@pytest.fixture()
def kitchen() -> World:
    """
    A world with one spoon and two drawers.
    """
    return world_with(
        [Spoon(root=body_named(SPOON_BODY_NAME))]
        + [Drawer(root=body_named(name)) for name in DRAWER_BODY_NAMES]
    )


def goal_state_of(world: World, answers: list[str] = ()) -> GoalState:
    """
    :param world: The world the goal is resolved in.
    :param answers: What the language model answers, in order.
    :return: A goal state for :data:`TASK` offering the spoon, the drawer, and the
        :class:`InsideOf` and :func:`contact` predicates.
    """
    return GoalState(
        task=TASK,
        object_types=[Spoon, Drawer],
        predicates=[InsideOf, contact],
        world=world,
        language_model=ScriptedLanguageModel(answers=list(answers)),
    )


def response_of(*relations: tuple[str, str, str]) -> str:
    """
    :param relations: The relations, each as object, predicate and object name.
    :return: The relations as the language model is asked to write them.
    """
    return json.dumps([list(relation) for relation in relations])


SPOON_IN_DRAWER = (Spoon.__name__, InsideOf.__name__, Drawer.__name__)
"""
The relation that the spoon is inside the drawer, as the language model writes it.
"""

SPOON_TOUCHES_DRAWER = (Spoon.__name__, contact.__name__, Drawer.__name__)
"""
The relation that the spoon touches the drawer, as the language model writes it.
"""


# %% prompt


class TestPrompt:
    def test_states_the_task(self, kitchen):
        assert TASK in goal_state_of(kitchen).build_prompt()

    def test_lists_every_offered_object(self, kitchen):
        prompt = goal_state_of(kitchen).build_prompt()

        assert all(object_type.__name__ in prompt for object_type in [Spoon, Drawer])

    def test_lists_every_offered_predicate(self, kitchen):
        prompt = goal_state_of(kitchen).build_prompt()

        assert all(predicate.__name__ in prompt for predicate in [InsideOf, contact])


# %% asking the language model


class TestQueryLanguageModel:
    def test_asks_with_the_built_prompt(self, kitchen):
        goal_state = goal_state_of(kitchen, [response_of(SPOON_IN_DRAWER)])

        goal_state.query_language_model(goal_state.build_prompt())

        assert goal_state.language_model.prompts == [goal_state.build_prompt()]

    def test_returns_the_relations_of_the_answer(self, kitchen):
        goal_state = goal_state_of(
            kitchen, [response_of(SPOON_IN_DRAWER, SPOON_TOUCHES_DRAWER)]
        )

        relations = goal_state.query_language_model(goal_state.build_prompt())

        assert relations == [
            GoalRelation(Spoon, InsideOf, Drawer),
            GoalRelation(Spoon, contact, Drawer),
        ]


# %% checking the response


class TestCheckResponse:
    @pytest.mark.parametrize(
        "response",
        [
            "not json",
            json.dumps({"relation": list(SPOON_IN_DRAWER)}),
            json.dumps([list(SPOON_IN_DRAWER[:2])]),
            json.dumps([[Spoon.__name__, InsideOf.__name__, 1]]),
        ],
        ids=["not json", "not a list", "two elements", "not a string"],
    )
    def test_rejects_a_response_of_another_shape(self, kitchen, response):
        with pytest.raises(MalformedGoalResponseError):
            goal_state_of(kitchen).check_response(response)

    @pytest.mark.parametrize(
        "template",
        [
            "Here is the goal state:\n%s\nThat is all.",
            "```json\n%s\n```",
            'The objects are ["Spoon", "Drawer"].\n%s',
        ],
        ids=["prose around it", "code fence", "a list of names before it"],
    )
    def test_reads_only_the_list_of_relations_out_of_the_answer(
        self, kitchen, template
    ):
        relations = response_of(SPOON_IN_DRAWER, SPOON_TOUCHES_DRAWER)

        assert goal_state_of(kitchen).check_response(
            template % relations
        ) == goal_state_of(kitchen).check_response(relations)

    def test_rejects_an_answer_without_a_list_of_lists(self, kitchen):
        with pytest.raises(MalformedGoalResponseError):
            goal_state_of(kitchen).check_response('The goal is ["Spoon", "Drawer"].')

    def test_rejects_an_object_that_was_not_offered(self, kitchen):
        response = response_of((Milk.__name__, InsideOf.__name__, Drawer.__name__))

        with pytest.raises(UnknownGoalObjectError) as excinfo:
            goal_state_of(kitchen).check_response(response)

        assert excinfo.value.object_name == Milk.__name__

    def test_rejects_a_predicate_that_was_not_offered(self, kitchen):
        response = response_of((Spoon.__name__, "OnTopOf", Drawer.__name__))

        with pytest.raises(UnknownGoalPredicateError) as excinfo:
            goal_state_of(kitchen).check_response(response)

        assert excinfo.value.predicate_name == "OnTopOf"


# %% resolving into a goal


class TestResolve:
    def test_every_relation_is_one_operand_of_the_conjunction(self, kitchen):
        goal = goal_state_of(kitchen).resolve(
            [
                GoalRelation(Spoon, InsideOf, Drawer),
                GoalRelation(Spoon, contact, Drawer),
            ]
        )

        assert isinstance(goal.condition, GoalConjunction)
        assert len(goal.condition.operands) == 2

    def test_an_object_of_several_bodies_is_a_disjunction_over_them(self, kitchen):
        goal = goal_state_of(kitchen).resolve([GoalRelation(Spoon, InsideOf, Drawer)])

        assert goal.condition.operands == [
            GoalDisjunction(
                [
                    GoalAtom(InsideOf.__name__, SPOON_BODY_NAME, drawer_name)
                    for drawer_name in DRAWER_BODY_NAMES
                ]
            )
        ]

    def test_objects_of_one_body_each_are_a_single_atom(self):
        world = world_with(
            [Spoon(root=body_named(SPOON_BODY_NAME))]
            + [Drawer(root=body_named(DRAWER_BODY_NAMES[0]))]
        )

        goal = goal_state_of(world).resolve([GoalRelation(Spoon, InsideOf, Drawer)])

        assert goal.condition.operands == [
            GoalAtom(InsideOf.__name__, SPOON_BODY_NAME, DRAWER_BODY_NAMES[0])
        ]

    def test_both_objects_of_several_bodies_give_every_pair(self):
        spoon_names = ["spoon_a", "spoon_b"]
        world = world_with(
            [Spoon(root=body_named(name)) for name in spoon_names]
            + [Drawer(root=body_named(name)) for name in DRAWER_BODY_NAMES]
        )

        goal = goal_state_of(world).resolve([GoalRelation(Spoon, InsideOf, Drawer)])

        assert goal.condition.operands == [
            GoalDisjunction(
                [
                    GoalAtom(InsideOf.__name__, spoon_name, drawer_name)
                    for spoon_name in spoon_names
                    for drawer_name in DRAWER_BODY_NAMES
                ]
            )
        ]

    def test_an_object_missing_from_the_world_is_rejected(self, kitchen):
        with pytest.raises(ObjectNotInWorld) as excinfo:
            goal_state_of(kitchen).resolve([GoalRelation(Milk, InsideOf, Drawer)])

        assert excinfo.value.object_type is Milk


# %% the goal in PDDL


class TestPddl:
    def test_an_atom_is_the_predicate_applied_to_both_bodies(self):
        atom = GoalAtom(InsideOf.__name__, SPOON_BODY_NAME, DRAWER_BODY_NAMES[0])

        assert atom.to_pddl() == "(%s %s %s)" % (
            InsideOf.__name__,
            SPOON_BODY_NAME,
            DRAWER_BODY_NAMES[0],
        )

    def test_a_connective_wraps_its_operands(self):
        atoms = [
            GoalAtom(InsideOf.__name__, SPOON_BODY_NAME, name)
            for name in DRAWER_BODY_NAMES
        ]

        assert GoalDisjunction(atoms).to_pddl() == "(%s %s)" % (
            LogicalConnective.OR,
            " ".join(atom.to_pddl() for atom in atoms),
        )

    def test_the_task_resolves_into_the_goal_of_the_example(self, kitchen):
        goal_state = goal_state_of(
            kitchen, [response_of(SPOON_IN_DRAWER, SPOON_TOUCHES_DRAWER)]
        )

        goal = goal_state.goal()

        assert goal.to_pddl() == (
            "(:goal (AND "
            "(OR (InsideOf spoon.stl drawer_top1) (InsideOf spoon.stl drawer_bottom1)) "
            "(OR (contact spoon.stl drawer_top1) (contact spoon.stl drawer_bottom1))))"
        )


# %% the goal as a tree


@pytest.fixture()
def example_goal(kitchen) -> GoalDefinition:
    return goal_state_of(kitchen).resolve(
        [GoalRelation(Spoon, InsideOf, Drawer), GoalRelation(Spoon, contact, Drawer)]
    )


def by_pddl(expression: GoalExpression) -> str:
    return expression.to_pddl()


class TestTree:
    def test_every_expression_is_a_node(self, example_goal):
        disjunctions = example_goal.condition.operands

        assert example_goal.tree().nodes() == [example_goal.condition] + [
            expression
            for disjunction in disjunctions
            for expression in [disjunction] + disjunction.operands
        ]

    def test_every_expression_points_to_its_operands(self, example_goal):
        tree = example_goal.tree()

        for index in tree.node_indices():
            successors = [tree[child] for child in tree.successor_indices(index)]
            assert sorted(successors, key=by_pddl) == sorted(
                tree[index].children(), key=by_pddl
            )

    def test_a_node_is_labelled_with_its_connective_or_atom(self, example_goal):
        disjunction = example_goal.condition.operands[0]

        assert example_goal.condition.label() == LogicalConnective.AND
        assert disjunction.label() == LogicalConnective.OR
        assert disjunction.operands[0].label() == disjunction.operands[0].to_pddl()

    def test_the_visualizer_draws_the_tree_without_running(self, example_goal):
        visualizer = example_goal.visualizer(open_browser=False)

        assert visualizer.graph.nodes() == example_goal.tree().nodes()
        assert visualizer.port == GOAL_TREE_PORT
        assert visualizer.open_browser is False


# %% kinds of expression


class TestKind:
    @pytest.mark.parametrize(
        "expression, kind",
        [
            (
                GoalAtom(InsideOf.__name__, SPOON_BODY_NAME, "drawer"),
                GoalExpressionKind.PREDICATE,
            ),
            (GoalConjunction([]), GoalExpressionKind.CONJUNCTION),
            (GoalDisjunction([]), GoalExpressionKind.DISJUNCTION),
        ],
        ids=["predicate", "conjunction", "disjunction"],
    )
    def test_every_expression_names_its_kind(self, expression, kind):
        assert expression.kind is kind


# %% the goal tree page


class TestDescription:
    def test_a_predicate_reads_from_its_first_body_to_its_second(self):
        atom = GoalAtom(InsideOf.__name__, SPOON_BODY_NAME, DRAWER_BODY_NAMES[0])

        assert atom.description() == [
            SPOON_BODY_NAME,
            InsideOf.__name__,
            DRAWER_BODY_NAMES[0],
        ]

    @pytest.mark.parametrize(
        "connective_type",
        [GoalConjunction, GoalDisjunction],
        ids=["conjunction", "disjunction"],
    )
    def test_a_connective_names_only_itself(self, connective_type):
        atoms = [
            GoalAtom(InsideOf.__name__, SPOON_BODY_NAME, name)
            for name in DRAWER_BODY_NAMES
        ]

        assert connective_type(atoms).description() == [connective_type.connective]


class TestGoalTreePage:
    def test_every_node_shows_its_whole_description(self, example_goal):
        visualizer = example_goal.visualizer(open_browser=False)

        assert [node["label"] for node in visualizer.graph_nodes()] == [
            "\n".join(expression.description())
            for expression in example_goal.tree().nodes()
        ]

    def test_a_connective_is_a_circle_and_a_predicate_a_box(self, example_goal):
        visualizer = example_goal.visualizer(open_browser=False)

        assert [node["shape"] for node in visualizer.graph_nodes()] == [
            (
                PREDICATE_SHAPE
                if expression.kind is GoalExpressionKind.PREDICATE
                else CONNECTIVE_SHAPE
            )
            for expression in example_goal.tree().nodes()
        ]

    def test_every_operand_is_an_unlabelled_arrow_from_its_connective(
        self, example_goal
    ):
        visualizer = example_goal.visualizer(open_browser=False)

        assert [
            (edge["from"], edge["to"], "label" in edge)
            for edge in visualizer.graph_edges()
        ] == [
            (str(parent), str(child), False)
            for parent, child in visualizer.graph.edge_list()
        ]

    def test_the_page_offers_no_details_on_click(self, example_goal):
        client = (
            example_goal.visualizer(open_browser=False)
            .build_application()
            .test_client()
        )

        page = client.get("/").get_data(as_text=True)

        assert "Click a node" not in page
        assert "node/" not in page
        assert client.get("/node/0").status_code == HTTPStatus.NOT_FOUND

    def test_the_page_serves_the_drawn_tree(self, example_goal):
        visualizer = example_goal.visualizer(open_browser=False)
        client = visualizer.build_application().test_client()

        assert client.get("/graph").get_json() == {
            "nodes": visualizer.graph_nodes(),
            "edges": visualizer.graph_edges(),
        }


# %% captions drawn on the tree


class TestCaption:
    def test_a_caption_is_the_description_line_by_line(self, example_goal):
        atom = example_goal.condition.operands[0].operands[0]

        assert atom.caption() == "\n".join(atom.description())
