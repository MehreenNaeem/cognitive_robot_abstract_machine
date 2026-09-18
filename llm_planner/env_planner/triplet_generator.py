from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from requests import RequestException
from typing_extensions import Any, Optional, Protocol

from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    PouringAction,
)
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from llm_planner.env_reasoner.action_constraints import (
    AVAILABLE_ACTIONS,
    INTERACTIVE_ACTION,
    LOCOMOTION_ACTIONS,
    ConditionKind,
    action_condition,
)
from llm_planner.env_reasoner.object_action_affordance import (
    doable_actions_for,
    get_object_action_list,
)
from llm_planner.env_reasoner.object_constraints import ObjectAffordanceReasoner
from llm_planner.ollama_config import DEFAULT_MODEL, chat_with_model, get_response
from krrood.utils import recursive_subclasses

import semantic_digital_twin.semantic_annotations.semantic_annotations
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

PREFIX_SEPARATOR = "/"
"""
What :meth:`~semantic_digital_twin.datastructures.prefixed_name.PrefixedName.__str__`
puts between a name's prefix and the name itself.
"""


class TripletField(StrEnum):
    """
    The keys one triplet carries in the LLM's JSON response.
    """

    SOURCE_OBJECT = "source_object"
    """
    The object the action is performed on.
    """

    TARGET_OBJECT = "target_object"
    """
    The object the action is performed against, e.g. the receptacle an object is placed
    into.
    """

    ACTION = "action"
    """
    The action to perform.
    """


class ActionRuleSource(StrEnum):
    """
    Where :meth:`TripletPromptGenerator.action_rules_section` takes each action's rule
    text from.
    """

    MANUAL = "manual"
    """
    The hand-written rules in :data:`MANUAL_ACTION_RULES`.
    """

    DECLARED = "declared"
    """
    The precondition each action declares in its Entity Query Language condition method.
    """


#: Hand-written rule for each object-directed action, phrased for the LLM in terms of
#: the object affordances the action requires of its target.
MANUAL_ACTION_RULES: dict[type[ActionDescription], str] = {
    PickUpAction: (
        "allows the robot to pick up a perceivable object. Compatible objects have "
        '"graspable" as object affordance.'
    ),
    PlaceAction: (
        "attempts to place an object the robot is holding onto or into the target "
        'object. Valid target receptacle objects have "receptacle" as object '
        "affordance. Object must be pickedup before PlaceAction."
    ),
    OpenAction: 'attempts to open an object having "openable" as object affordance.',
    CloseAction: 'attempts to close an object having "closable" as object affordance.',
    CuttingAction: (
        'attempts to slice/cut an object having "cutable" as object affordance.'
    ),
    PouringAction: (
        'attempts to empty the liquid from an object which has "pourable" as object '
        "affordance, to target receptacle objects."
    ),
}


@dataclass
class TripletPromptGenerator:
    """
    Builds the natural-language prompt that asks an LLM to ground a task into
    (object, action, reasoning) triplets.

    Assembles the prompt from a natural language task command and a fixed list of
    candidate object types, together with the affordances, object properties, action
    preconditions and per-object actionable list computed in
    :mod:`~llm_planner.env_reasoner.object_action_affordance`.
    """

    task: str
    """
    Natural language instruction the LLM should ground into triplets.
    """

    object_types: list[type[SemanticAnnotation]]
    """
    Candidate semantic annotation types the LLM may choose objects from.
    """

    reasoner: ObjectAffordanceReasoner = field(default_factory=ObjectAffordanceReasoner)
    """
    Reasoner used to compute each candidate object type's affordances.
    """

    rule_source: ActionRuleSource = ActionRuleSource.MANUAL
    """
    Which wording :meth:`action_rules_section` states each action's rule in.
    ActionRuleSource.DECLARED gives the EQL-derived wording.
    """

    def task_section(self) -> str:
        """
        :return: Point 1, the task command to ground.
        """
        return f'Generate a sequence of action triplets for executing the given Task: "{self.task}"'

    def object_instructions_section(self) -> str:
        """
        :return: Point 2, general instructions on how to reason about objects.
        """
        return (
            "Only propose a triplet whose action appears in that object's "
            '"Object actionable list" below, and respect the "Object properties" '
            'and "Action rules".'
        )

    def object_list_section(self) -> str:
        """
        :return: Point 3, the candidate objects the LLM may choose from.
        """
        names = ", ".join(object_type.__name__ for object_type in self.object_types)
        return f"Available objects: {names}"

    def action_list_section(self) -> str:
        """
        :return: Point 4, the object-directed actions the LLM may choose from, from
            :data:`~llm_planner.env_reasoner.action_constraints.INTERACTIVE_ACTION`.
        """
        names = ", ".join(action.__name__ for action in INTERACTIVE_ACTION)
        return f"Available actions: {names}"

    def object_affordance_section(self) -> str:
        """
        :return: Point 5, each candidate object's affordances, computed via
            :attr:`reasoner`.
        """
        lines = [
            f"- {object_type.__name__}: "
            + ", ".join(
                sorted(
                    affordance.value
                    for affordance in self.reasoner.affordances_for(object_type)
                )
            )
            for object_type in self.object_types
        ]
        return "Object affordances:\n" + "\n".join(lines)

    def object_properties_section(self) -> str:
        """
        :return: Point 6, each candidate object's structural properties: the mixin base
            classes it carries, e.g. ``HasDoors``, from
            :attr:`~llm_planner.env_reasoner.object_action_affordance.ObjectActionAffordances.mixins`.
        """
        lines = [
            f"- {object_type.__name__}: "
            + ", ".join(
                sorted(
                    mixin.__name__ for mixin in doable_actions_for(object_type).mixins
                )
            )
            for object_type in self.object_types
        ]
        return "Object properties:\n" + "\n".join(lines)

    def action_rules_section(self) -> str:
        """
        :return: Point 7, the rule each object-directed available action follows, stated
            in the wording :attr:`rule_source` selects.

        Actions in
        :data:`~llm_planner.env_reasoner.action_constraints.LOCOMOTION_ACTIONS` move
        the robot rather than a target object and are left out.
        """
        if self.rule_source is ActionRuleSource.MANUAL:
            return "Action rules:\n" + "\n".join(self._manual_rule_lines())
        return "Action rules:\n" + "\n".join(self._declared_rule_lines())

    def _manual_rule_lines(self) -> list[str]:
        """
        :return: One ``- <action>: <rule>`` line per available action that
            :data:`MANUAL_ACTION_RULES` has a rule for.
        """
        return [
            f"- {action.__name__}: {rule}"
            for action, rule in MANUAL_ACTION_RULES.items()
            if action in AVAILABLE_ACTIONS and action not in LOCOMOTION_ACTIONS
        ]

    def _declared_rule_lines(self) -> list[str]:
        """
        :return: One ``- <action>: <precondition>`` line per available action that
            declares a precondition in its Entity Query Language condition method.
        """
        lines = []
        for action in AVAILABLE_ACTIONS:
            if action in LOCOMOTION_ACTIONS:
                continue
            pre_condition = action_condition(action, ConditionKind.PRE)
            if not pre_condition.is_declared:
                continue
            description = " ".join(pre_condition.description.split())
            lines.append(f"- {action.__name__}: {description}")
        return lines

    def object_actionable_list_section(self) -> str:
        """
        :return: Point 8, the actions each candidate object can take part in, from
            :func:`~llm_planner.env_reasoner.object_action_affordance.get_object_action_list`.
        """
        lines = [
            f"- {object_type.__name__}: "
            + ", ".join(sorted(get_object_action_list(object_type)))
            for object_type in self.object_types
        ]
        return "Object actionable list:\n" + "\n".join(lines)

    def format_example_section(self) -> str:
        """
        :return: Point 9, the expected response format, shown as an example.
        """
        fields = ", ".join(
            f'"{field_name}": "<one of the available {described}>"'
            for field_name, described in (
                (TripletField.SOURCE_OBJECT, "objects"),
                (TripletField.TARGET_OBJECT, "objects"),
                (TripletField.ACTION, "actions"),
            )
        )
        return (
            "Respond with a JSON list of triplets and nothing else, in this exact "
            f"shape:\n[{{{fields}}}]"
        )

    def build_prompt(self) -> str:
        """
        :return: The full prompt, assembling every section from point 1 to 9 in
            order.
        """
        sections = [
            self.task_section(),
            self.object_instructions_section(),
            self.object_list_section(),
            self.action_list_section(),
            self.object_affordance_section(),
            self.object_properties_section(),
            self.action_rules_section(),
            self.object_actionable_list_section(),
            self.format_example_section(),
        ]
        return "\n\n".join(sections)


def semantic_annotation_in_world(world: World) -> list[str]:
    """
    List the distinct names of every semantic annotation in a world.

    :param world: The world whose semantic annotations are read.
    :return: Each annotation's name as a string, in first-seen order and without
        duplicates, e.g. ``["Milk", "Handle", "Drawer"]``.
    """
    return list(
        dict.fromkeys(str(annotation.name) for annotation in world.semantic_annotations)
    )


class UnknownSemanticAnnotationNameError(ValueError):
    """
    Raised when a name does not belong to any
    :class:`~semantic_digital_twin.world_description.world_entity.SemanticAnnotation`
    subclass.
    """

    def __init__(self, annotation_name: str, known_names: list[str]) -> None:
        super().__init__(
            f"'{annotation_name}' names no SemanticAnnotation subclass. "
            f"Known names: {', '.join(sorted(known_names))}."
        )
        self.annotation_name = annotation_name
        """
        The name that could not be resolved.
        """

        self.known_names = known_names
        """
        Every name that would have resolved.
        """


def semantic_annotation_types_by_name() -> dict[str, type[SemanticAnnotation]]:
    """
    :return: Every semantic annotation type, keyed by its class name, which is the name
        an annotation carries unless it was given one explicitly.
    """
    return {
        annotation_type.__name__: annotation_type
        for annotation_type in recursive_subclasses(SemanticAnnotation)
    }


def preprocessing_objects(
    annotation_names: list[str], excluded_names: list[str]
) -> list[type[SemanticAnnotation]]:
    """
    Drop the unwanted names from a list of annotation names and resolve the rest back to
    their semantic annotation types.

    :param annotation_names: The names to filter, as
        :func:`semantic_annotation_in_world` returns them.
    :param excluded_names: The names to leave out. A name is dropped when any of these
        appears anywhere in it.
    :return: The type of every kept name, ready to be passed as
        :attr:`TripletPromptGenerator.object_types`.
    :raises UnknownSemanticAnnotationNameError: If a kept name belongs to no semantic
        annotation type.
    """
    types_by_name = semantic_annotation_types_by_name()
    kept_names = [
        annotation_name
        for annotation_name in annotation_names
        if not any(excluded in annotation_name for excluded in excluded_names)
    ]

    annotation_types = []
    for annotation_name in kept_names:
        unprefixed_name = annotation_name.rsplit(PREFIX_SEPARATOR, maxsplit=1)[-1]
        if unprefixed_name not in types_by_name:
            raise UnknownSemanticAnnotationNameError(
                annotation_name, list(types_by_name)
            )
        annotation_types.append(types_by_name[unprefixed_name])
    return list(dict.fromkeys(annotation_types))


"""
# Usage Example:
if __name__ == "__main__":
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Apple,
        Drawer,
        Milk,
    )

    generator = TripletPromptGenerator(
        task="Put the apple in the drawer",
        object_types=[Apple, Drawer, Milk],
    )
    print(generator.build_prompt())
"""


# %% response checking


class TripletResponseError(ValueError):
    """
    Base class for every way an LLM triplet response can fail to be usable.
    """


class MalformedTripletResponseError(TripletResponseError):
    """
    Raised when a response is not a JSON list of triplets carrying exactly the fields
    :class:`TripletField` names, each with a string value.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"Response is not a list of triplets: {reason}")
        self.reason = reason
        """
        What about the response's shape was wrong.
        """


class UnknownTripletObjectError(TripletResponseError):
    """
    Raised when a triplet names an object outside the ones the prompt offered.
    """

    def __init__(self, object_name: str, available_names: list[str]) -> None:
        super().__init__(
            f"'{object_name}' is not an available object. "
            f"Available: {', '.join(sorted(available_names))}."
        )
        self.object_name = object_name
        """
        The name the response used.
        """

        self.available_names = available_names
        """
        The object names the prompt offered.
        """


class UnknownTripletActionError(TripletResponseError):
    """
    Raised when a triplet names an action outside the ones the prompt offered.
    """

    def __init__(self, action_name: str, available_names: list[str]) -> None:
        super().__init__(
            f"'{action_name}' is not an available action. "
            f"Available: {', '.join(sorted(available_names))}."
        )
        self.action_name = action_name
        """
        The name the response used.
        """

        self.available_names = available_names
        """
        The action names the prompt offered.
        """


class ActionNotDoableOnObjectError(TripletResponseError):
    """
    Raised when a triplet pairs an available object with an available action that the
    object's actionable list does not allow.
    """

    def __init__(
        self, object_type: type[SemanticAnnotation], action: type[ActionDescription]
    ) -> None:
        super().__init__(
            f"{action.__name__} is not doable on {object_type.__name__}. "
            f"Doable: {', '.join(sorted(get_object_action_list(object_type)))}."
        )
        self.object_type = object_type
        """
        The object the action was proposed for.
        """

        self.action = action
        """
        The action proposed for it.
        """


@dataclass
class Triplet:
    """
    One checked (source object, target object, action) triplet, with each name the LLM
    returned resolved to the class it denotes.
    """

    source_object: type[SemanticAnnotation]
    """
    The object the action is performed on.
    """

    target_object: type[SemanticAnnotation]
    """
    The object the action is performed against.
    """

    action: type[ActionDescription]
    """
    The action to perform.
    """


@dataclass
class TripletResponseChecker:
    """
    Checks that an LLM's response to a
    :meth:`TripletPromptGenerator.build_prompt` prompt has the shape the prompt asked
    for and only names objects and actions the prompt offered.
    """

    object_types: list[type[SemanticAnnotation]]
    """
    The object types the prompt offered, i.e. the generator's
    :attr:`TripletPromptGenerator.object_types`.
    """

    actions: list[type[ActionDescription]] = field(
        default_factory=lambda: list(INTERACTIVE_ACTION)
    )
    """
    The actions the prompt offered.
    """

    def parse(self, response: str) -> list[Triplet]:
        """
        Check a response and resolve every triplet in it.

        :param response: The LLM's raw response text.
        :return: One :class:`Triplet` per entry, in the order the response listed them.
        :raises MalformedTripletResponseError: If the response is not a JSON list of
            triplets with exactly the expected fields.
        :raises UnknownTripletObjectError: If a triplet names an unavailable object.
        :raises UnknownTripletActionError: If a triplet names an unavailable action.
        :raises ActionNotDoableOnObjectError: If a triplet's action is not in its source
            object's actionable list.
        """
        return [self.check_elements(entry) for entry in self.check_format(response)]

    def check_format(self, response: str) -> list[dict[str, str]]:
        """
        Check that a response is a JSON list of triplets shaped as the prompt asked.

        :param response: The LLM's raw response text.
        :return: The triplet entries, still as raw names.
        :raises MalformedTripletResponseError: If the response is not such a list.
        """
        entries = self._loaded_json(response)
        if not isinstance(entries, list):
            raise MalformedTripletResponseError(
                f"expected a list, got {type(entries).__name__}"
            )
        for entry in entries:
            self._check_entry_format(entry)
        return entries

    def check_elements(self, entry: dict[str, str]) -> Triplet:
        """
        Resolve one format-checked triplet's names and check the pairing they describe.

        :param entry: One triplet entry, as :meth:`check_format` returns it.
        :return: The resolved triplet.
        :raises UnknownTripletObjectError: If it names an unavailable object.
        :raises UnknownTripletActionError: If it names an unavailable action.
        :raises ActionNotDoableOnObjectError: If the action is not in the source
            object's actionable list.
        """
        source_object = self._resolve_object(entry[TripletField.SOURCE_OBJECT])
        target_object = self._resolve_object(entry[TripletField.TARGET_OBJECT])
        action = self._resolve_action(entry[TripletField.ACTION])
        if action.__name__ not in get_object_action_list(source_object):
            raise ActionNotDoableOnObjectError(source_object, action)
        return Triplet(
            source_object=source_object, target_object=target_object, action=action
        )

    @staticmethod
    def _loaded_json(response: str) -> object:
        """
        :param response: The LLM's raw response text.
        :return: The response parsed as JSON.
        :raises MalformedTripletResponseError: If it is not valid JSON.
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError as error:
            raise MalformedTripletResponseError(f"not valid JSON ({error})") from error

    @staticmethod
    def _check_entry_format(entry: object) -> None:
        """
        :param entry: One element of the response list.
        :raises MalformedTripletResponseError: If it is not an object carrying exactly
            the :class:`TripletField` keys with string values.
        """
        if not isinstance(entry, dict):
            raise MalformedTripletResponseError(
                f"a triplet is a {type(entry).__name__}, not an object"
            )
        expected_fields = {field_name.value for field_name in TripletField}
        if set(entry) != expected_fields:
            raise MalformedTripletResponseError(
                f"a triplet has fields {sorted(entry)}, expected "
                f"{sorted(expected_fields)}"
            )
        for field_name, value in entry.items():
            if not isinstance(value, str):
                raise MalformedTripletResponseError(
                    f"'{field_name}' is a {type(value).__name__}, not a string"
                )

    def _resolve_object(self, object_name: str) -> type[SemanticAnnotation]:
        """
        :param object_name: The object name a triplet used.
        :return: The offered object type of that name.
        :raises UnknownTripletObjectError: If no offered object has that name.
        """
        available_names = [object_type.__name__ for object_type in self.object_types]
        for object_type in self.object_types:
            if object_type.__name__ == object_name:
                return object_type
        raise UnknownTripletObjectError(object_name, available_names)

    def _resolve_action(self, action_name: str) -> type[ActionDescription]:
        """
        :param action_name: The action name a triplet used.
        :return: The offered action of that name.
        :raises UnknownTripletActionError: If no offered action has that name.
        """
        for action in self.actions:
            if action.__name__ == action_name:
                return action
        raise UnknownTripletActionError(
            action_name, [action.__name__ for action in self.actions]
        )


"""
# %% usage example

if __name__ == "__main__":
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Bowl,
        Drawer,
    )

    RECORDED_RESPONSE = (
        Path(__file__).parents[1]
        / "resources"
        / "llm_responses"
        / "put_the_bowl_in_the_drawer.json"
    )

    generator = TripletPromptGenerator(
        task="Put the bowl in the drawer", object_types=[Bowl, Drawer]
    )
    print(generator.build_prompt())

    print("\n--- response ---")
    llm_response = RECORDED_RESPONSE.read_text()
    print(llm_response)

    print("--- checked triplets ---")
    checker = TripletResponseChecker(object_types=generator.object_types)
    for triplet in checker.parse(llm_response):
        print(
            f"{triplet.action.__name__}: "
            f"{triplet.source_object.__name__} -> {triplet.target_object.__name__}"
        )

    print("\n--- a response the checker rejects ---")
    wrong_action = llm_response.replace(PickUpAction.__name__, OpenAction.__name__)
    try:
        checker.parse(wrong_action)
    except TripletResponseError as error:
        print(f"{type(error).__name__}: {error}")
"""

# %% action sequence prediction


class LanguageModel(Protocol):
    """
    A model that answers a prompt with text.
    """

    def complete(self, prompt: str) -> str:
        """
        :param prompt: The prompt to answer.
        :return: The model's answer.
        """


class ChatCompletionField(StrEnum):
    """
    The keys of the chat completion payload an OpenAI-compatible endpoint answers with.
    """

    CHOICES = "choices"
    """
    The answers the model produced.
    """

    MESSAGE = "message"
    """
    One choice's message.
    """

    CONTENT = "content"
    """
    A message's text.
    """


class EmptyChatCompletionError(RuntimeError):
    """
    Raised when a chat completion payload carries no answer to read.
    """

    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__(f"Chat completion carries no choice: {payload}")
        self.payload = payload
        """
        The payload that was answered with.
        """


@dataclass
class ChatCompletion:
    """
    The part of an OpenAI-compatible chat completion payload that is read back.
    """

    content: str
    """
    The text of the model's first answer.
    """

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> ChatCompletion:
        """
        Read a chat completion payload.

        :param payload: The endpoint's decoded JSON answer.
        :return: The completion it carries.
        :raises EmptyChatCompletionError: If the payload carries no choice.
        """
        choices = payload.get(ChatCompletionField.CHOICES, [])
        if not choices:
            raise EmptyChatCompletionError(payload)
        return cls(
            content=choices[0][ChatCompletionField.MESSAGE][ChatCompletionField.CONTENT]
        )


class LanguageModelUnreachable(RuntimeError):
    """
    Raised when the language model's endpoint could not be reached at all.
    """

    def __init__(self, url: str, reason: Exception) -> None:
        super().__init__(f"the language model at {url} could not be reached: {reason}")
        self.url = url
        """
        The endpoint that did not answer.
        """

        self.reason = reason
        """
        What the request failed with.
        """


@dataclass
class OllamaLanguageModel:
    """
    A :class:`LanguageModel` served by an Ollama chat completions endpoint.
    """

    url: str
    """
    Address of the chat completions endpoint.
    """

    user_id: str
    """
    Authorization header value the endpoint expects.
    """

    model: str = DEFAULT_MODEL
    """
    Name of the model to prompt.
    """

    def complete(self, prompt: str) -> str:
        """
        :param prompt: The prompt to answer.
        :return: The model's answer, with any reasoning block removed.
        :raises LanguageModelUnreachable: If the endpoint could not be reached.
        """
        try:
            payload = chat_with_model(prompt, self.url, self.user_id, self.model)
        except RequestException as error:
            raise LanguageModelUnreachable(self.url, error) from error
        return get_response(ChatCompletion.from_json(payload).content)


class ActionSequencePredictionError(RuntimeError):
    """
    Raised when no attempt produced a response that passes every check.
    """

    def __init__(self, task: str, failures: list[TripletResponseError]) -> None:
        reasons = "; ".join(
            f"attempt {number}: {type(failure).__name__}: {failure}"
            for number, failure in enumerate(failures, start=1)
        )
        super().__init__(f'No usable action sequence for "{task}" ({reasons}).')
        self.task = task
        """
        The task that was to be grounded.
        """

        self.failures = failures
        """
        Why each attempt was rejected, in attempt order.
        """


@dataclass
class ActionSequencePredictor:
    """
    Predicts the action sequence that carries out a task, given the objects it may act
    on.

    Builds the prompt from those objects, asks a language model, checks the response and
    retries with the rejection reason appended until an answer passes every check.
    """

    object_types: list[type[SemanticAnnotation]]
    """
    The objects a task may be grounded in.
    """

    language_model: LanguageModel
    """
    The model asked to ground the task.
    """

    attempts: int = 3
    """
    How many times the model is asked before giving up.
    """

    rule_source: ActionRuleSource = ActionRuleSource.MANUAL
    """
    Which wording the prompt states each action's rule in.
    """

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {self.attempts}.")

    @classmethod
    def from_world(
        cls,
        world: World,
        language_model: LanguageModel,
        excluded_object_names: Optional[list[str]] = None,
        **predictor_arguments: Any,
    ) -> ActionSequencePredictor:
        """
        A predictor grounding tasks in the objects a world holds.

        :param world: The world whose semantic annotations the objects are taken from.
        :param language_model: The model asked to ground the task.
        :param excluded_object_names: Annotation names to keep out of the prompt, e.g.
            the robot's and its handles'.
        :param predictor_arguments: The remaining fields, e.g. ``attempts``.
        """
        return cls(
            object_types=preprocessing_objects(
                semantic_annotation_in_world(world), excluded_object_names or []
            ),
            language_model=language_model,
            **predictor_arguments,
        )

    def prompt_generator(self, task: str) -> TripletPromptGenerator:
        """
        :param task: The natural language instruction to ground.
        :return: The generator that builds the prompt for that task.
        """
        return TripletPromptGenerator(
            task=task, object_types=self.object_types, rule_source=self.rule_source
        )

    def predict(self, task: str,vis=False) -> list[Triplet]:
        """
        Ground a task into the sequence of triplets that carries it out.

        :param task: The natural language instruction to ground.
        :return: One checked :class:`Triplet` per action, in execution order.
        :raises ActionSequencePredictionError: If no attempt passed every check.
        """
        generator = self.prompt_generator(task)
        checker = TripletResponseChecker(object_types=generator.object_types)
        prompt = generator.build_prompt()
        if vis:
            print(f"Prompt:\n{prompt}")

        failures: list[TripletResponseError] = []
        while len(failures) < self.attempts:
            response = self.language_model.complete(prompt)
            try:
                return checker.parse(response)
            except TripletResponseError as failure:
                failures.append(failure)
                prompt = self.retry_prompt(generator.build_prompt(), failure)
        raise ActionSequencePredictionError(task, failures)

    @staticmethod
    def retry_prompt(prompt: str, failure: TripletResponseError) -> str:
        """
        :param prompt: The prompt the rejected response answered.
        :param failure: Why that response was rejected.
        :return: The prompt to ask again with, stating what was wrong with the previous
            answer.
        """
        return (
            f"{prompt}\n\nYour previous answer was rejected: {failure} "
            "Answer again, correcting this."
        )
