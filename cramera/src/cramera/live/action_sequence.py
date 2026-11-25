"""
Grounding a task statement in the world a running demo executes in.

Only the demo process holds that world, so this lives behind the live bridge: the
objects a task may name are exactly the semantic annotations of the world the demo
attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, TYPE_CHECKING

from llm_planner.env_planner.triplet_generator import (
    preprocessing_objects,
    semantic_annotation_in_world,
)
from llm_planner.env_reasoner.LLM_utilities import LanguageModel
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle

from cramera.action_sequence import (
    ActionSequencePrediction,
    TaskGrounding,
    configured_language_model,
)

if TYPE_CHECKING:
    from cramera.live.bridge import Bridge


class NoAttachedWorld(RuntimeError):
    """
    Raised when a prediction is asked for before a demo has attached its world.
    """

    def __init__(self) -> None:
        super().__init__("no demo is attached — nothing to ground the task in")


@dataclass
class LiveActionSequence:
    """
    Predicts the action sequence for a task in the world the attached demo is running.
    """

    bridge: Bridge
    """
    The bridge holding the executing world.
    """

    language_model: LanguageModel = field(
        default_factory=lambda: configured_language_model()
    )
    """
    The model asked to ground the task, by default the one the environment names.
    """

    def excluded_object_names(self) -> List[str]:
        """
        :return: Names no task acts on: the robot performing it, and the handles that
            are part of another object rather than objects of their own.
        """
        excluded = [Handle.__name__]
        if self.bridge.robot is not None:
            excluded.append(type(self.bridge.robot).__name__)
        return excluded

    def grounding(self) -> TaskGrounding:
        """
        :return: The grounding of a task in the attached world's objects.
        :raises NoAttachedWorld: If no demo has attached a world yet.
        """
        if self.bridge.world is None:
            raise NoAttachedWorld()
        return TaskGrounding(
            object_types=preprocessing_objects(
                semantic_annotation_in_world(self.bridge.world),
                self.excluded_object_names(),
            ),
            language_model=self.language_model,
        )

    def predict(self, task: str) -> ActionSequencePrediction:
        """
        Ground a task statement in the attached world.

        :param task: The natural language task the panel asked about.
        :return: The predicted sequence.
        :raises NoTask: If the task statement is empty.
        :raises NoAttachedWorld: If no demo has attached a world yet.
        :raises ActionSequencePredictionError: If no answer passed every check.
        :raises LanguageModelUnreachable: If the model's endpoint did not answer.
        """
        return self.grounding().predict(task)
