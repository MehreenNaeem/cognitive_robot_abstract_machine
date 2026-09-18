"""
Tests of the action-sequence prediction the task-grounding panel asks for.

The language model is scripted, so nothing here talks to a real endpoint: what is
under test is which objects a task is grounded in, and the payload the panel receives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from llm_planner.env_planner.triplet_generator import (
    ActionSequencePredictionError,
    TripletField,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Drawer,
    Milk,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

from cramera import action_sequence
from cramera.action_sequence import (
    LanguageModelVariable,
    NoTask,
    PredictedAction,
    TaskGrounding,
    configured_language_model,
)


@dataclass
class ScriptedLanguageModel:
    """A language model answering with prepared texts, recording what it was asked."""

    answers: list[str]
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers[len(self.prompts) - 1]


def annotated_world(*annotation_types: type[SemanticAnnotation]) -> World:
    """A world holding one annotation of each given type, on a body of its own."""
    world = World()
    root = Body(name=PrefixedName("root"))
    with world.modify_world():
        world.add_body(root)
        for annotation_type in annotation_types:
            body = Body(name=PrefixedName(annotation_type.__name__))
            world.add_connection(FixedConnection(parent=root, child=body))
            world.add_semantic_annotation(annotation_type(root=body))
    return world


def triplet(source: str, target: str, action: str) -> dict[str, str]:
    return {
        TripletField.SOURCE_OBJECT: source,
        TripletField.TARGET_OBJECT: target,
        TripletField.ACTION: action,
    }


PUT_THE_BOWL_IN_THE_DRAWER = json.dumps(
    [
        triplet(Bowl.__name__, Drawer.__name__, PickUpAction.__name__),
        triplet(Bowl.__name__, Drawer.__name__, PlaceAction.__name__),
    ]
)
"""
An answer that passes every check wherever a bowl and a drawer are among the objects.
"""


def grounding(*answers: str, object_types=None) -> TaskGrounding:
    return TaskGrounding(
        object_types=object_types if object_types is not None else [Bowl, Drawer],
        language_model=ScriptedLanguageModel(answers=list(answers)),
    )


# %% grounding a task without a running demo


class TestTaskGrounding:
    def test_predicts_the_actions_in_the_order_they_are_performed(self) -> None:
        prediction = grounding(PUT_THE_BOWL_IN_THE_DRAWER).predict(
            "Put the bowl in the drawer"
        )

        assert [action.action for action in prediction.actions] == [
            PickUpAction.__name__,
            PlaceAction.__name__,
        ]
        assert prediction.actions[0].source_object == Bowl.__name__
        assert prediction.actions[0].target_object == Drawer.__name__

    def test_grounds_only_in_the_objects_it_was_given(self) -> None:
        live = grounding(PUT_THE_BOWL_IN_THE_DRAWER, object_types=[Bowl, Drawer, Milk])

        prediction = live.predict("Put the bowl in the drawer")

        assert prediction.objects == [Bowl.__name__, Drawer.__name__, Milk.__name__]

    def test_reports_the_task_it_grounded(self) -> None:
        prediction = grounding(PUT_THE_BOWL_IN_THE_DRAWER).predict(
            "  Put the bowl in the drawer  "
        )

        assert prediction.task == "Put the bowl in the drawer"

    def test_the_prompt_names_the_objects_it_grounds_in(self) -> None:
        live = grounding(PUT_THE_BOWL_IN_THE_DRAWER)

        live.predict("Put the bowl in the drawer")

        [prompt] = live.language_model.prompts
        assert (
            prompt
            == live.predictor()
            .prompt_generator("Put the bowl in the drawer")
            .build_prompt()
        )

    def test_rejects_an_empty_task(self) -> None:
        with pytest.raises(NoTask):
            grounding().predict("   ")

    def test_reports_an_answer_that_never_passed_the_checks(self) -> None:
        with pytest.raises(ActionSequencePredictionError):
            grounding("not json", "still not", "nor this").predict(
                "Put the bowl in the drawer"
            )


# %% the payload the panel reads


class TestPredictionPayload:
    def test_an_action_is_sent_as_the_panel_names_its_parts(self) -> None:
        action = PredictedAction(
            action=PickUpAction.__name__,
            source_object=Bowl.__name__,
            target_object=Drawer.__name__,
        )

        assert action.to_payload() == {
            "action": PickUpAction.__name__,
            "sourceObject": Bowl.__name__,
            "targetObject": Drawer.__name__,
        }

    def test_a_prediction_is_sent_with_its_task_actions_and_objects(self) -> None:
        prediction = grounding(PUT_THE_BOWL_IN_THE_DRAWER).predict(
            "Put the bowl in the drawer"
        )

        payload = prediction.to_payload()

        assert payload["ok"] is True
        assert payload["task"] == "Put the bowl in the drawer"
        assert payload["objects"] == [Bowl.__name__, Drawer.__name__]
        assert payload["actions"] == [
            action.to_payload() for action in prediction.actions
        ]


# %% which model is asked


class TestConfiguredLanguageModel:
    def test_takes_the_endpoint_the_environment_names(self, monkeypatch) -> None:
        monkeypatch.setenv(LanguageModelVariable.URL, "http://elsewhere/api")
        monkeypatch.setenv(LanguageModelVariable.TOKEN, "Bearer other")
        monkeypatch.setenv(LanguageModelVariable.MODEL, "other-model")

        model = configured_language_model()

        assert model.url == "http://elsewhere/api"
        assert model.user_id == "Bearer other"
        assert model.model == "other-model"

    def test_falls_back_to_the_configured_ollama_endpoint(self, monkeypatch) -> None:
        for variable in LanguageModelVariable:
            monkeypatch.delenv(variable, raising=False)

        assert configured_language_model().url == action_sequence.URL_OLLAMA
