from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
import pytest

from llm_planner.env_planner.triplet_generator import (
    MANUAL_ACTION_RULES,
    ActionSequencePredictionError,
    ActionSequencePredictor,
    ActionNotDoableOnObjectError,
    ActionRuleSource,
    MalformedTripletResponseError,
    TripletField,
    TripletPromptGenerator,
    TripletResponseChecker,
    UnknownTripletActionError,
    UnknownTripletObjectError,
    UnknownSemanticAnnotationNameError,
    preprocessing_objects,
    semantic_annotation_types_by_name,
)
from llm_planner.env_reasoner.action_constraints import (
    INTERACTIVE_ACTION,
    LOCOMOTION_ACTIONS,
    ConditionKind,
    action_condition,
)
from llm_planner.env_reasoner.object_action_affordance import (
    doable_actions_for,
    get_object_action_list,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Bowl,
    Drawer,
    Handle,
    Milk,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

RECORDED_RESPONSE_PATH = (
    Path(__file__).parents[2]
    / "llm_planner"
    / "resources"
    / "llm_responses"
    / "put_the_bowl_in_the_drawer.json"
)


# %% TripletPromptGenerator


class TestTripletPromptGenerator:
    def setup_method(self) -> None:
        self.generator = TripletPromptGenerator(
            task="Put the apple in the drawer", object_types=[Apple, Drawer]
        )

    def test_task_section_contains_the_task_command(self) -> None:
        assert self.generator.task in self.generator.task_section()

    def test_object_list_section_names_every_candidate_object(self) -> None:
        section = self.generator.object_list_section()
        assert "Apple" in section
        assert "Drawer" in section

    def test_action_list_section_names_every_interactive_action(self) -> None:
        section = self.generator.action_list_section()
        expected = ", ".join(action.__name__ for action in INTERACTIVE_ACTION)
        assert section == f"Available actions: {expected}"

    def test_build_prompt_puts_the_actions_right_after_the_objects(self) -> None:
        prompt = self.generator.build_prompt()
        objects_index = prompt.index(self.generator.object_list_section())
        actions_index = prompt.index(self.generator.action_list_section())
        assert actions_index > objects_index

    def test_object_affordance_section_reflects_the_reasoner(self) -> None:
        section = self.generator.object_affordance_section()
        apple_affordances = self.generator.reasoner.affordances_for(Apple)
        assert all(affordance.value in section for affordance in apple_affordances)

    def test_object_properties_section_lists_each_object_mixins(self) -> None:
        section = self.generator.object_properties_section()
        drawer_mixins = {mixin.__name__ for mixin in doable_actions_for(Drawer).mixins}
        assert drawer_mixins
        assert all(mixin_name in section for mixin_name in drawer_mixins)

    def test_action_rules_section_states_action_preconditions(self) -> None:
        generator = TripletPromptGenerator(
            task=self.generator.task,
            object_types=self.generator.object_types,
            rule_source=ActionRuleSource.DECLARED,
        )
        section = generator.action_rules_section()
        pick_up_pre_condition = action_condition(PickUpAction, ConditionKind.PRE)
        description = " ".join(pick_up_pre_condition.description.split())
        assert f"- {PickUpAction.__name__}: {description}" in section

    def test_action_rules_section_skips_locomotion_actions(self) -> None:
        for source in ActionRuleSource:
            generator = TripletPromptGenerator(
                task=self.generator.task,
                object_types=self.generator.object_types,
                rule_source=source,
            )
            section = generator.action_rules_section()
            assert NavigateAction in LOCOMOTION_ACTIONS
            for action in LOCOMOTION_ACTIONS:
                assert f"- {action.__name__}:" not in section

    def test_action_rules_section_defaults_to_the_manual_rules(self) -> None:
        assert self.generator.rule_source is ActionRuleSource.MANUAL
        section = self.generator.action_rules_section()
        assert (
            f"- {PickUpAction.__name__}: {MANUAL_ACTION_RULES[PickUpAction]}" in section
        )

    def test_manual_action_rules_section_covers_every_interactive_action(self) -> None:
        section = self.generator.action_rules_section()
        for action in INTERACTIVE_ACTION:
            assert f"- {action.__name__}: {MANUAL_ACTION_RULES[action]}" in section

    def test_object_actionable_list_section_uses_get_object_action_list(self) -> None:
        section = self.generator.object_actionable_list_section()
        for object_type in (Apple, Drawer):
            for action_name in get_object_action_list(object_type):
                assert action_name in section

    def test_build_prompt_assembles_every_section_in_order(self) -> None:
        prompt = self.generator.build_prompt()
        sections = [
            self.generator.task_section(),
            self.generator.object_instructions_section(),
            self.generator.object_list_section(),
            self.generator.action_list_section(),
            self.generator.object_affordance_section(),
            self.generator.object_properties_section(),
            self.generator.action_rules_section(),
            self.generator.object_actionable_list_section(),
            self.generator.format_example_section(),
        ]
        assert prompt == "\n\n".join(sections)


# %% preprocessing_objects


class TestPreprocessingObjects:
    def test_resolves_a_name_back_to_its_semantic_annotation_type(self) -> None:
        assert preprocessing_objects(["Drawer"], []) == [Drawer]

    def test_drops_a_name_containing_an_excluded_name(self) -> None:
        assert preprocessing_objects(["Drawer", "Handle"], ["Handle"]) == [Drawer]

    def test_drops_a_name_an_excluded_name_only_appears_inside(self) -> None:
        assert preprocessing_objects(["Drawer", "SoapBottle"], ["Bottle"]) == [Drawer]

    def test_resolves_a_name_carrying_its_worlds_prefix(self) -> None:
        prefixed_name = str(PrefixedName(Drawer.__name__, "apartment"))
        assert preprocessing_objects([prefixed_name], []) == [Drawer]

    def test_keeps_each_type_once_when_a_name_repeats(self) -> None:
        assert preprocessing_objects(["Drawer", "Apple", "Drawer"], []) == [
            Drawer,
            Apple,
        ]

    def test_raises_on_a_name_belonging_to_no_semantic_annotation_type(self) -> None:
        with pytest.raises(UnknownSemanticAnnotationNameError) as excinfo:
            preprocessing_objects(["Drawer", "NotAnAnnotation"], [])

        assert excinfo.value.annotation_name == "NotAnAnnotation"

    def test_result_is_accepted_as_the_generators_object_types(self) -> None:
        object_types = preprocessing_objects(["Apple", "Drawer"], [])
        generator = TripletPromptGenerator(
            task="Put it away", object_types=object_types
        )
        assert generator.object_list_section() == "Available objects: Apple, Drawer"


# %% semantic_annotation_types_by_name


def test_semantic_annotation_types_by_name_keys_each_type_by_its_class_name() -> None:
    types_by_name = semantic_annotation_types_by_name()
    assert types_by_name[Drawer.__name__] is Drawer
    assert types_by_name[Apple.__name__] is Apple


# %% TripletResponseChecker


def response_of(*triplets: dict[str, str]) -> str:
    """Serialize triplets the way the LLM is asked to return them."""
    return json.dumps(list(triplets))


def triplet_of(source: str, target: str, action: str) -> dict[str, str]:
    """Build one response triplet keyed by the fields the prompt asks for."""
    return {
        TripletField.SOURCE_OBJECT: source,
        TripletField.TARGET_OBJECT: target,
        TripletField.ACTION: action,
    }


class TestTripletResponseChecker:
    def setup_method(self) -> None:
        self.checker = TripletResponseChecker(object_types=[Apple, Drawer])
        self.valid = triplet_of(Apple.__name__, Drawer.__name__, PlaceAction.__name__)

    def test_parses_a_well_formed_response_into_resolved_types(self) -> None:
        [triplet] = self.checker.parse(response_of(self.valid))

        assert triplet.source_object is Apple
        assert triplet.target_object is Drawer
        assert triplet.action is PlaceAction

    def test_parses_every_triplet_in_the_response(self) -> None:
        second = triplet_of(Drawer.__name__, Drawer.__name__, OpenAction.__name__)

        assert len(self.checker.parse(response_of(self.valid, second))) == 2

    def test_rejects_a_response_that_is_not_json(self) -> None:
        with pytest.raises(MalformedTripletResponseError):
            self.checker.parse("Sure! Here are the triplets:")

    def test_rejects_a_response_that_is_not_a_list(self) -> None:
        with pytest.raises(MalformedTripletResponseError):
            self.checker.parse(json.dumps(self.valid))

    def test_rejects_a_triplet_missing_a_field(self) -> None:
        incomplete = dict(self.valid)
        del incomplete[TripletField.ACTION]

        with pytest.raises(MalformedTripletResponseError):
            self.checker.parse(response_of(incomplete))

    def test_rejects_a_triplet_carrying_an_unexpected_field(self) -> None:
        extra = dict(self.valid) | {"reasoning": "because"}

        with pytest.raises(MalformedTripletResponseError):
            self.checker.parse(response_of(extra))

    def test_rejects_a_field_that_is_not_a_string(self) -> None:
        wrong_type = dict(self.valid) | {TripletField.ACTION: 7}

        with pytest.raises(MalformedTripletResponseError):
            self.checker.parse(response_of(wrong_type))

    def test_rejects_an_object_outside_the_available_objects(self) -> None:
        unavailable = triplet_of(Milk.__name__, Drawer.__name__, PlaceAction.__name__)

        with pytest.raises(UnknownTripletObjectError) as excinfo:
            self.checker.parse(response_of(unavailable))

        assert excinfo.value.object_name == Milk.__name__

    def test_rejects_an_action_outside_the_available_actions(self) -> None:
        unavailable = triplet_of(
            Apple.__name__, Drawer.__name__, NavigateAction.__name__
        )

        with pytest.raises(UnknownTripletActionError) as excinfo:
            self.checker.parse(response_of(unavailable))

        assert excinfo.value.action_name == NavigateAction.__name__

    def test_rejects_an_action_not_doable_on_the_source_object(self) -> None:
        assert OpenAction.__name__ not in get_object_action_list(Apple)
        not_doable = triplet_of(Apple.__name__, Drawer.__name__, OpenAction.__name__)

        with pytest.raises(ActionNotDoableOnObjectError) as excinfo:
            self.checker.parse(response_of(not_doable))

        assert excinfo.value.object_type is Apple
        assert excinfo.value.action is OpenAction

    def test_checker_accepts_the_generators_object_types(self) -> None:
        generator = TripletPromptGenerator(task="t", object_types=[Apple, Drawer])
        checker = TripletResponseChecker(object_types=generator.object_types)

        assert checker.parse(response_of(self.valid))


# %% a real LLM response


class TestRecordedLlmResponse:
    def setup_method(self) -> None:
        self.checker = TripletResponseChecker(object_types=[Bowl, Drawer])

    def test_recorded_response_passes_every_check(self) -> None:
        triplets = self.checker.parse(RECORDED_RESPONSE_PATH.read_text())

        assert [triplet.action for triplet in triplets] == [PickUpAction, PlaceAction]
        assert all(triplet.source_object is Bowl for triplet in triplets)
        assert all(triplet.target_object is Drawer for triplet in triplets)

    def test_recorded_response_names_only_offered_objects(self) -> None:
        offered = {Bowl.__name__, Drawer.__name__}
        for entry in self.checker.check_format(RECORDED_RESPONSE_PATH.read_text()):
            assert entry[TripletField.SOURCE_OBJECT] in offered
            assert entry[TripletField.TARGET_OBJECT] in offered


# %% action sequence prediction


@dataclass
class ScriptedLanguageModel:
    """A language model answering with prepared texts, recording what it was asked."""

    answers: list[str]
    prompts: list[str] = dataclasses.field(default_factory=list)

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers[len(self.prompts) - 1]


def world_of(*annotations: SemanticAnnotation) -> World:
    """Build a world holding one body per given annotation."""
    world = World()
    root = Body(name=PrefixedName("root"))
    with world.modify_world():
        world.add_body(root)
        for annotation in annotations:
            world.add_connection(FixedConnection(parent=root, child=annotation.root))
            world.add_semantic_annotation(annotation)
    return world


def annotation_of(annotation_type: type[SemanticAnnotation]) -> SemanticAnnotation:
    """Build an annotation of the given type on a body named after it."""
    return annotation_type(root=Body(name=PrefixedName(annotation_type.__name__)))


class TestActionSequencePredictor:
    def setup_method(self) -> None:
        self.world = world_of(annotation_of(Bowl), annotation_of(Drawer))
        self.sequence = response_of(
            triplet_of(Bowl.__name__, Drawer.__name__, PickUpAction.__name__),
            triplet_of(Bowl.__name__, Drawer.__name__, PlaceAction.__name__),
        )

    def predictor(self, *answers: str, attempts: int = 3) -> ActionSequencePredictor:
        return ActionSequencePredictor.from_world(
            world=self.world,
            language_model=ScriptedLanguageModel(answers=list(answers)),
            attempts=attempts,
        )

    def test_a_worlds_predictor_offers_the_worlds_annotation_types(self) -> None:
        assert self.predictor().object_types == [Bowl, Drawer]

    def test_a_worlds_predictor_leaves_out_the_excluded_names(self) -> None:
        predictor = ActionSequencePredictor.from_world(
            world=world_of(annotation_of(Bowl), annotation_of(Handle)),
            language_model=ScriptedLanguageModel(answers=[]),
            excluded_object_names=[Handle.__name__],
        )

        assert predictor.object_types == [Bowl]

    def test_a_worlds_predictor_keeps_the_remaining_fields(self) -> None:
        predictor = ActionSequencePredictor.from_world(
            world=self.world,
            language_model=ScriptedLanguageModel(answers=[]),
            attempts=7,
        )

        assert predictor.attempts == 7

    def test_the_objects_can_be_given_without_a_world(self) -> None:
        predictor = ActionSequencePredictor(
            object_types=[Bowl, Drawer],
            language_model=ScriptedLanguageModel(answers=[self.sequence]),
        )

        assert len(predictor.predict("Put the bowl in the drawer")) == 2

    def test_predicts_the_checked_triplets_of_the_answer(self) -> None:
        triplets = self.predictor(self.sequence).predict("Put the bowl in the drawer")

        assert [triplet.action for triplet in triplets] == [PickUpAction, PlaceAction]
        assert [triplet.source_object for triplet in triplets] == [Bowl, Bowl]

    def test_asks_with_the_prompt_built_for_the_task(self) -> None:
        predictor = self.predictor(self.sequence)
        task = "Put the bowl in the drawer"

        predictor.predict(task)

        assert predictor.language_model.prompts == [
            predictor.prompt_generator(task).build_prompt()
        ]

    def test_asks_again_after_a_rejected_answer(self) -> None:
        predictor = self.predictor("not json at all", self.sequence)

        assert len(predictor.predict("Put the bowl in the drawer")) == 2

    def test_the_second_prompt_states_why_the_first_answer_was_rejected(self) -> None:
        predictor = self.predictor("not json at all", self.sequence)
        task = "Put the bowl in the drawer"

        predictor.predict(task)

        first, second = predictor.language_model.prompts
        with pytest.raises(MalformedTripletResponseError) as excinfo:
            TripletResponseChecker(object_types=[Bowl, Drawer]).parse("not json at all")
        assert second == ActionSequencePredictor.retry_prompt(first, excinfo.value)

    def test_gives_up_after_the_last_attempt(self) -> None:
        predictor = self.predictor("no", "json", "here", attempts=3)
        task = "Put the bowl in the drawer"

        with pytest.raises(ActionSequencePredictionError) as excinfo:
            predictor.predict(task)

        assert excinfo.value.task == task
        assert len(predictor.language_model.prompts) == 3
        assert all(
            isinstance(failure, MalformedTripletResponseError)
            for failure in excinfo.value.failures
        )

    def test_rejects_a_predictor_that_would_never_ask(self) -> None:
        with pytest.raises(ValueError):
            self.predictor(attempts=0)
