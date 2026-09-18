"""
Tests of grounding a task in the world a running demo attached.

The language model is scripted, so nothing here talks to a real endpoint: what is under
test is which objects the attached world contributes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Drawer,
    Handle,
)
from semantic_digital_twin.world_description.world_entity import Body

from cramera.live.action_sequence import LiveActionSequence, NoAttachedWorld
from cramera.live.bridge import Bridge

from .test_action_sequence import (
    PUT_THE_BOWL_IN_THE_DRAWER,
    ScriptedLanguageModel,
    annotated_world,
)


@dataclass
class AttachedRobot:
    """A stand-in for the robot a demo attached: only its type name is read."""


@pytest.fixture()
def attached_bridge() -> Bridge:
    """A bridge bound to a world holding a bowl, a drawer and the drawer's handle."""
    bridge = Bridge()
    bridge.attach(annotated_world(Bowl, Drawer, Handle))
    return bridge


def grounding(bridge: Bridge, *answers: str) -> LiveActionSequence:
    return LiveActionSequence(
        bridge=bridge, language_model=ScriptedLanguageModel(answers=list(answers))
    )


class TestLiveActionSequence:
    def test_predicts_the_actions_in_the_order_they_are_performed(
        self, attached_bridge
    ) -> None:
        prediction = grounding(attached_bridge, PUT_THE_BOWL_IN_THE_DRAWER).predict(
            "Put the bowl in the drawer"
        )

        assert [action.action for action in prediction.actions] == [
            PickUpAction.__name__,
            PlaceAction.__name__,
        ]

    def test_offers_the_worlds_objects_but_not_its_handles(
        self, attached_bridge
    ) -> None:
        prediction = grounding(attached_bridge, PUT_THE_BOWL_IN_THE_DRAWER).predict(
            "Put the bowl in the drawer"
        )

        assert prediction.objects == [Bowl.__name__, Drawer.__name__]

    def test_excludes_the_attached_robot_from_the_objects(
        self, attached_bridge
    ) -> None:
        attached_bridge.robot = AttachedRobot()

        excluded = grounding(attached_bridge).excluded_object_names()

        assert excluded == [Handle.__name__, AttachedRobot.__name__]

    def test_the_prompt_names_the_worlds_objects(self, attached_bridge) -> None:
        live = grounding(attached_bridge, PUT_THE_BOWL_IN_THE_DRAWER)

        live.predict("Put the bowl in the drawer")

        [prompt] = live.language_model.prompts
        assert (
            prompt
            == live.grounding()
            .predictor()
            .prompt_generator("Put the bowl in the drawer")
            .build_prompt()
        )

    def test_rejects_a_task_while_no_demo_is_attached(self) -> None:
        with pytest.raises(NoAttachedWorld):
            grounding(Bridge()).predict("Put the bowl in the drawer")
