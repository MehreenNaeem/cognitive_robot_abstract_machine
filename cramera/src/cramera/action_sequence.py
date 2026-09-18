"""
Grounding a task statement into the action sequence that carries it out.

The panel sends a natural language task ("Put the bowl in the drawer"); this module
answers with the ordered triplets a language model proposes for it, checked against the
objects the task may act on (see :mod:`llm_planner.env_planner.triplet_generator`).

The objects a task may be grounded in are the ones a running demo's world holds; see
:mod:`cramera.live.action_sequence`, which reads them off the attached world.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Any, Dict, List

from llm_planner.env_planner.triplet_generator import (
    ActionSequencePredictor,
    LanguageModel,
    OllamaLanguageModel,
    Triplet,
)
from llm_planner.ollama_config import DEFAULT_MODEL, URL_OLLAMA, USER_ID_OLLAMA
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from cramera.payload import CrameraPayload


class LanguageModelVariable(StrEnum):
    """
    Environment variables naming the language model the panel asks.
    """

    URL = "CRAMERA_LLM_URL"
    """
    Address of the chat completions endpoint.
    """

    TOKEN = "CRAMERA_LLM_TOKEN"
    """
    Authorization header value the endpoint expects.
    """

    MODEL = "CRAMERA_LLM_MODEL"
    """
    Name of the model to prompt.
    """


def configured_language_model() -> OllamaLanguageModel:
    """
    :return: The model named by :class:`LanguageModelVariable`, falling back to the
        endpoint :mod:`llm_planner.ollama_config` is configured with.
    """
    return OllamaLanguageModel(
        url=os.environ.get(LanguageModelVariable.URL, URL_OLLAMA),
        user_id=os.environ.get(LanguageModelVariable.TOKEN, USER_ID_OLLAMA),
        model=os.environ.get(LanguageModelVariable.MODEL, DEFAULT_MODEL),
    )


class NoTask(ValueError):
    """
    Raised when a prediction is asked for without a task statement.
    """

    def __init__(self) -> None:
        super().__init__("no task given")


@dataclass
class PredictedAction(CrameraPayload):
    """
    One action of a predicted sequence, as the panel lists it.
    """

    action: str
    """
    Name of the action to perform.
    """

    source_object: str
    """
    Name of the object the action is performed on.
    """

    target_object: str
    """
    Name of the object the action is performed against.
    """

    @classmethod
    def of_triplet(cls, triplet: Triplet) -> PredictedAction:
        """
        :param triplet: The checked triplet to name.
        """
        return cls(
            action=triplet.action.__name__,
            source_object=triplet.source_object.__name__,
            target_object=triplet.target_object.__name__,
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "sourceObject": self.source_object,
            "targetObject": self.target_object,
        }


@dataclass
class ActionSequencePrediction(CrameraPayload):
    """
    The answer to one task statement: the actions to perform, and the objects they were
    chosen from.
    """

    task: str
    """
    The task statement that was grounded.
    """

    actions: List[PredictedAction] = field(default_factory=list)
    """
    The actions to perform, in execution order.
    """

    objects: List[str] = field(default_factory=list)
    """
    Names of the object types the language model was offered.
    """

    def to_payload(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "task": self.task,
            "actions": [action.to_payload() for action in self.actions],
            "objects": self.objects,
        }


@dataclass
class TaskGrounding:
    """
    Grounds a task statement in the objects it may act on.
    """

    object_types: List[type[SemanticAnnotation]]
    """
    The objects a task may be grounded in.
    """

    language_model: LanguageModel = field(
        default_factory=lambda: configured_language_model()
    )
    """
    The model asked to ground the task, by default the one the environment names.
    """

    def predictor(self) -> ActionSequencePredictor:
        """
        :return: The predictor grounding a task in :attr:`object_types`.
        """
        return ActionSequencePredictor(
            object_types=self.object_types, language_model=self.language_model
        )

    def predict(self, task: str) -> ActionSequencePrediction:
        """
        Ground a task statement into the sequence of actions that carries it out.

        :param task: The natural language task the panel asked about.
        :return: The predicted sequence.
        :raises NoTask: If the task statement is empty.
        :raises ActionSequencePredictionError: If no answer passed every check.
        :raises LanguageModelUnreachable: If the model's endpoint did not answer.
        """
        task = task.strip()
        if not task:
            raise NoTask()
        predictor = self.predictor()
        return ActionSequencePrediction(
            task=task,
            actions=[
                PredictedAction.of_triplet(triplet)
                for triplet in predictor.predict(task)
            ],
            objects=[object_type.__name__ for object_type in predictor.object_types],
        )
